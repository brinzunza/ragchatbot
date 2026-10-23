"""
The agent loop.

Each user message runs a loop: the model either answers or asks for tools;
requested tools run and their results go back to the model; this repeats
until it answers or hits the step limit. run_agent is a plain generator of
event dicts that the API layer turns into server-sent events, so a tool
that takes a while (or waits for the user's approval) simply blocks this
generator without touching the web server's event loop.
"""

import json
import re
import time
import uuid
from datetime import datetime
from typing import Iterator, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import chat_manager
import context_manager
from agent_tools import ToolBox, create_permission_request, wait_for_permission
from llm_prompts import _chat_llm, agent_system_prompt, agent_tool_hints
from qa_workflow import get_sources_metadata

RESULT_PREVIEW_CHARS = 700
MAX_PERMISSION_CODE_CHARS = 6000


def today_string() -> str:
    """Today's date as the model should read it, e.g. 'Friday, October 23, 2026'."""
    return datetime.now().strftime("%A, %B %d, %Y").replace(" 0", " ")


def is_tools_unsupported(error: Exception) -> bool:
    """True when the model/provider can't do tool calling at all (as opposed to failing on one call)."""
    if isinstance(error, NotImplementedError):
        return True
    text = str(error).lower()
    return "does not support tools" in text or "tools are not supported" in text


def _text_of(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


# Messages that need no tools at all. Small local models tend to reach for a
# tool on any message, even "hello", so these skip tool calling outright.
_SMALL_TALK = re.compile(
    r"^\s*(hi|hey|hello|yo|howdy|good (morning|afternoon|evening)|thanks|thank you|thx|ok|okay|cool|great|nice|"
    r"bye|goodbye|how are you|what'?s up|who are you|what can you do)\b[\s\w,!.?']{0,30}$",
    re.IGNORECASE,
)


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK.match(text)) and len(text.split()) <= 6


# The user telling Athena something about themselves ("my name is actually
# John"). That is for the conversation context to hold, not for a document
# search; left to itself a small model searches the documents for it anyway.
_SELF_STATEMENT = re.compile(
    r"^\s*(actually|no|sorry|btw|also)?[\s,.!-]*(my (name|email|age|job|birthday|pronouns?|favou?rite \w+) is|"
    r"call me|please call me|i go by|i prefer|from now on)\b",
    re.IGNORECASE,
)


# Asking about something already said in the conversation ("repeat the bio you
# wrote"). The answer is sitting in the context; a small model will still
# reach for a tool - once it ran Python just to print the sentence back.
_ABOUT_EARLIER_REPLY = re.compile(
    r"\b(repeat|say (that|it) again|what did (you|i) (say|write|ask)|(you|i) (just |earlier |previously )?(wrote|said|told you|asked)|"
    r"(the )?(last|previous|above|earlier) (answer|message|response|reply)|rephrase|reword|shorten (that|it)|"
    r"make (it|that) (shorter|longer|simpler)|translate (that|it))\b",
    re.IGNORECASE,
)


def stated_facts(history: List[Tuple[str, str]]) -> List[str]:
    """What the user has said about themselves earlier in this conversation, word for word."""
    return [question.strip() for question, _ in history
            if _SELF_STATEMENT.match(question) and "?" not in question]


def skips_tools(text: str, has_history: bool = False) -> bool:
    """
    Messages that need no tools: small talk, the user stating a fact about
    themselves, or (with earlier conversation to draw on) a request about
    something already said.
    """
    if is_small_talk(text) or (bool(_SELF_STATEMENT.match(text)) and "?" not in text):
        return True
    return has_history and bool(_ABOUT_EARLIER_REPLY.search(text))


def find_text_tool_call(text: str, tool_names) -> Optional[dict]:
    """
    Some small models write a tool call out as JSON text instead of using the
    tool-calling channel, e.g. {"name": "search_documents", "parameters": {...}}.
    Find such a call (a known tool name only) so it can run like a real one.
    Returns {"call": tool call dict, "before": the prose ahead of it}.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("name") in tool_names:
            args = value.get("parameters", value.get("arguments", value.get("args", {})))
            if isinstance(args, dict):
                call = {"name": value["name"], "args": args, "id": uuid.uuid4().hex[:12], "type": "tool_call"}
                return {"call": call, "before": text[:index].replace("```json", "").replace("```", "").strip()}
    return None


_CONTEXT_ACTION = re.compile(r"\b(compact|summari[sz]e|forget|remove|delete|drop|clear|erase|keep only|narrow)\b", re.IGNORECASE)
_CONTEXT_SUBJECT = re.compile(r"\b(context|conversation|memory|chat history|history)\b", re.IGNORECASE)
_REMEMBER = re.compile(r"\b(remember|keep in mind|don'?t forget|note that)\b", re.IGNORECASE)


def context_tools_for(text: str) -> set:
    """
    Which memory tools this message actually asks for. The model only gets
    them when the user's own words call for it: left on offer all the time,
    a small model used keep_only_context on a "/compact ..." message and
    wiped the whole conversation.
    """
    allowed = set()
    if _CONTEXT_ACTION.search(text) and _CONTEXT_SUBJECT.search(text):
        allowed |= {"compact_context", "keep_only_context", "forget_context"}
    if _REMEMBER.search(text):
        allowed.add("remember")
    return allowed


# An answer with no words in it ({} or []) - a model glitch, not a reply.
_EMPTY_ANSWER = re.compile(r"[\s{}\[\]\"',:.]*")


def is_empty_answer(text: str) -> bool:
    return bool(_EMPTY_ANSWER.fullmatch(text))


_TOOL_CALL_START = re.compile(r'^\s*(?:```(?:json)?\s*)?\{\s*"name"\s*:\s*"(\w+)"')


def looks_like_tool_call(text: str, tool_names) -> bool:
    """Text that starts out as a call to one of our tools - whether or not it is valid JSON."""
    match = _TOOL_CALL_START.match(text)
    return bool(match) and match.group(1) in tool_names


def history_to_messages(history: List[Tuple[str, str]]) -> list:
    """Turn context_manager.build_history pairs into chat messages."""
    messages = []
    for question, answer in history:
        if question == "summary of the earlier conversation":
            messages.append(SystemMessage(content=f"Summary of the earlier conversation: {answer}"))
        else:
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=answer))
    return messages


def agent_loop(messages: list, toolbox: ToolBox, max_steps: int, include_context_tools: bool = True,
               use_tools: bool = True) -> Iterator[dict]:
    """
    Run the model/tool loop over `messages`, yielding events.

    Returns (via StopIteration.value) the final answer text.
    Raises whatever the model call raises; the caller decides what that means.
    """
    tools = toolbox.tools(include_context_tools)
    tools_by_name = {tool.name: tool for tool in tools}
    llm = _chat_llm()
    llm_with_tools = llm.bind_tools(tools)

    steps_taken = 0
    previous_calls = set()
    # With documents uploaded, the web is only for what they don't cover.
    # Small models reach for the web first anyway, so enforce the order.
    docs_exist = toolbox.has_documents
    docs_searched = False
    malformed_retries = 0

    while True:
        out_of_steps = steps_taken >= max_steps or not use_tools
        if steps_taken >= max_steps:
            messages.append(HumanMessage(content="You have used all your tool calls. Answer now with what you have."))
        runner = llm if out_of_steps else llm_with_tools

        aggregate = None
        turn_text = ""
        saw_tool_call = False
        for chunk in runner.stream(messages):
            aggregate = chunk if aggregate is None else aggregate + chunk
            if getattr(chunk, "tool_call_chunks", None):
                saw_tool_call = True
            text = _text_of(chunk)
            if text and not saw_tool_call:
                turn_text += text
                yield {"type": "chunk", "content": text}

        tool_calls = [] if out_of_steps or aggregate is None else (aggregate.tool_calls or [])
        streamed_text = turn_text
        if not tool_calls and not out_of_steps:
            found = find_text_tool_call(turn_text, tools_by_name)
            if found:
                tool_calls = [found["call"]]
                turn_text = found["before"]
                aggregate = AIMessage(content="", tool_calls=tool_calls)
        if not tool_calls and not out_of_steps and malformed_retries < 2 and looks_like_tool_call(turn_text, tools_by_name):
            # A tool call written out as text but not valid JSON - typically
            # code with unescaped quotes inside a string. Showing it as the
            # answer would be junk; ask the model to try again.
            malformed_retries += 1
            yield {"type": "reset"}
            messages.append(AIMessage(content=turn_text))
            messages.append(HumanMessage(content=(
                "That tool call was not valid JSON (quotes inside strings must be escaped). Either call the "
                "tool again with valid arguments, or just answer in plain sentences.")))
            continue
        if not tool_calls:
            return turn_text

        # Any text before the tool calls was the model thinking aloud, not the answer.
        if streamed_text:
            yield {"type": "reset"}
        messages.append(aggregate)

        for call in tool_calls:
            steps_taken += 1
            step_id = steps_taken
            name, args = call["name"], call.get("args") or {}
            yield {"type": "step", "id": step_id, "tool": name, "args": args, "thought": turn_text.strip()}
            turn_text = ""

            signature = (name, json.dumps(args, sort_keys=True, default=str))
            status = "ok"
            if name in ("web_search", "fetch_url") and docs_exist and not docs_searched:
                result, status = ("skipped: the user has uploaded documents. search them first with search_documents; "
                                  "use the web only if they don't have the answer."), "skipped"
            elif name not in tools_by_name:
                result, status = f"error: there is no tool called {name!r}", "error"
            elif signature in previous_calls:
                result, status = ("error: you already made this exact call. use the earlier result, "
                                  "try different arguments, or answer now."), "error"
            else:
                previous_calls.add(signature)
                docs_searched = docs_searched or name in ("search_documents", "read_document")
                approved = True
                if toolbox.needs_approval(name):
                    request_id = create_permission_request(toolbox.chat_id)
                    code = str(args.get("code", ""))[:MAX_PERMISSION_CODE_CHARS]
                    yield {"type": "permission_request", "id": request_id, "step": step_id, "tool": name, "code": code}
                    approved = wait_for_permission(request_id)
                if not approved:
                    result, status = "the user did not allow this code to run. do not retry it; continue without it or explain.", "denied"
                else:
                    try:
                        result = str(tools_by_name[name].invoke(args))
                        if result.startswith("error:"):
                            status = "error"
                    except Exception as e:
                        result, status = f"error: {e}", "error"

            yield {"type": "step_result", "id": step_id, "status": status,
                   "result": result[:RESULT_PREVIEW_CHARS] + ("…" if len(result) > RESULT_PREVIEW_CHARS else "")}
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))


def run_agent(chat_id: str, question: str, history: List[Tuple[str, str]], config: dict) -> Iterator[dict]:
    """
    Answer one user message with the agent, yielding SSE-ready event dicts:
    chunk, reset, step, step_result, permission_request, sources, complete,
    or a single `fallback` event when the model can't call tools at all.
    Provider errors propagate to the caller.
    """
    start = time.time()
    toolbox = ToolBox(chat_id, config)
    toolbox.context_tools_allowed = context_tools_for(question)
    names = toolbox.document_names()
    knowledge = ("Documents the user has uploaded: " + ", ".join(names) + ". Search them when a question is about "
                 "what the documents contain (not for things the user said in this conversation), and prefer them "
                 "to your memory or the web for that."
                 if names else "The user has not uploaded any documents to this chat.")
    offered = {tool.name for tool in toolbox.tools()}
    tool_lines = "\n".join(f"- {hint}" for name, hint in agent_tool_hints.items() if name in offered)
    system_prompt = (agent_system_prompt.replace("{tools}", "Tools:\n" + tool_lines).replace("{today}", today_string())
                     + f"\n\n{knowledge}")
    facts = stated_facts(history)
    if facts:
        # Pinned here, not left in the chat history: a document search result
        # sits right next to the question and a model will trust it over an
        # earlier message that says the opposite.
        system_prompt += ("\n\nThe user has told you the following. It is true and overrides anything in the "
                          "documents or the web:\n" + "\n".join(f"- {fact}" for fact in facts))
    messages = [SystemMessage(content=system_prompt)] + history_to_messages(history) + [HumanMessage(content=question)]

    steps = 0
    used_web = False

    def drive(loop):
        """Pass the loop's events through, tallying steps; returns its final answer."""
        nonlocal steps, used_web
        while True:
            try:
                event = next(loop)
            except StopIteration as stop:
                return stop.value or ""
            if event["type"] == "step":
                steps += 1
                used_web = used_web or event["tool"] in ("web_search", "fetch_url")
            yield event

    try:
        answer = yield from drive(agent_loop(messages, toolbox, config["max_agent_steps"],
                                             use_tools=not skips_tools(question, bool(history))))
        if is_empty_answer(answer):
            # A glitch, not a reply. Ask once more for plain sentences, without tools.
            yield {"type": "reset"}
            messages.append(HumanMessage(content="Reply to my last message in plain sentences."))
            answer = yield from drive(agent_loop(messages, toolbox, config["max_agent_steps"], use_tools=False))
            if is_empty_answer(answer):
                answer = "I couldn't come up with an answer to that. Try rephrasing it."
                yield {"type": "reset"}
                yield {"type": "chunk", "content": answer}
                answer = ""  # not worth keeping in the context
    except Exception as e:
        if steps == 0 and is_tools_unsupported(e):
            yield {"type": "fallback"}
            return
        raise

    if answer.strip():
        try:
            context_manager.record_turn(chat_id, question, answer)
        except Exception as e:
            print(f"WARNING: failed to record context: {e}")

    # Cite documents when the answer leans on them: bolded spans, or no web
    # research that could equally have supplied it.
    if toolbox.retrieved_docs and ("**" in answer or not used_web):
        sources = get_sources_metadata(toolbox.retrieved_docs)
        if sources:
            yield {"type": "sources", "content": sources}
    if answer.strip() and context_manager.maybe_auto_compact(chat_id):
        yield {"type": "notice", "message": "older messages were folded into the context summary."}
    yield {"type": "complete", "response_time": time.time() - start, "steps": steps}
