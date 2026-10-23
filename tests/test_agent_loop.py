import chat_manager
import pytest

import agent
import agent_tools
from fakes import ScriptedLLM, call, text


@pytest.fixture(autouse=True)
def no_documents(monkeypatch):
    monkeypatch.setattr(agent_tools, "has_documents", lambda path: False)


def run(chat_id, monkeypatch, llm, question="what is 2+2?", history=(), **config):
    monkeypatch.setattr(agent, "_chat_llm", lambda: llm)
    full = {**chat_manager.get_config(chat_id), **config}
    return list(agent.run_agent(chat_id, question, list(history), full))


def kinds(events):
    return [event["type"] for event in events]


def answer_of(events):
    answer = ""
    for event in events:
        if event["type"] == "chunk":
            answer += event["content"]
        elif event["type"] == "reset":
            answer = ""
    return answer


def test_a_direct_answer_streams_and_is_recorded(chat_id, monkeypatch):
    events = run(chat_id, monkeypatch, ScriptedLLM(text("Hello", " there")), question="tell me a joke")
    assert kinds(events) == ["chunk", "chunk", "complete"]
    assert answer_of(events) == "Hello there"
    entry = chat_manager.get_context(chat_id)[0]
    assert (entry["question"], entry["answer"]) == ("tell me a joke", "Hello there")


def test_a_tool_call_runs_and_its_result_reaches_the_model(chat_id, monkeypatch):
    llm = ScriptedLLM(call("calculator", {"expression": "2+2"}), text("It is 4"))
    events = run(chat_id, monkeypatch, llm)
    assert kinds(events) == ["step", "step_result", "chunk", "complete"]
    assert events[0]["tool"] == "calculator" and events[0]["args"] == {"expression": "2+2"}
    assert events[1]["status"] == "ok" and events[1]["result"] == "4"
    assert events[-1]["steps"] == 1
    tool_message = llm.calls[1][-1]
    assert type(tool_message).__name__ == "ToolMessage" and tool_message.content == "4"


def test_an_unknown_tool_is_reported_to_the_model_not_raised(chat_id, monkeypatch):
    events = run(chat_id, monkeypatch, ScriptedLLM(call("delete_everything", {}), text("sorry")))
    assert events[1]["status"] == "error" and "no tool called" in events[1]["result"]
    assert answer_of(events) == "sorry"


def test_an_identical_repeated_call_is_refused(chat_id, monkeypatch):
    same = {"expression": "2+2"}
    llm = ScriptedLLM(call("calculator", same), call("calculator", same, "call-2"), text("4"))
    events = run(chat_id, monkeypatch, llm)
    results = [e for e in events if e["type"] == "step_result"]
    assert [r["status"] for r in results] == ["ok", "error"]
    assert "already made this exact call" in results[1]["result"]


def test_tool_errors_become_results_the_model_can_react_to(chat_id, monkeypatch):
    llm = ScriptedLLM(call("calculator", {"expression": "__import__('os')"}), text("can't"))
    result = [e for e in run(chat_id, monkeypatch, llm) if e["type"] == "step_result"][0]
    assert result["status"] == "error" and result["result"].startswith("error:")


def test_the_step_limit_forces_a_final_answer(chat_id, monkeypatch):
    llm = ScriptedLLM(call("calculator", {"expression": "1+1"}), text("done"))
    events = run(chat_id, monkeypatch, llm, max_agent_steps=1)
    assert kinds(events).count("step") == 1 and answer_of(events) == "done"
    nudge = llm.calls[-1][-1]
    assert type(nudge).__name__ == "HumanMessage" and "used all your tool calls" in nudge.content
    assert llm.with_tools == [True, False]   # the last call was made without tools


def test_text_before_a_tool_call_is_reset_and_kept_as_the_thought(chat_id, monkeypatch):
    llm = ScriptedLLM(text("Let me check."), text("unused"))
    llm.turns = [
        [*text("Let me check. "), *call("calculator", {"expression": "3*3"})],
        text("9"),
    ]
    events = run(chat_id, monkeypatch, llm)
    assert kinds(events)[:4] == ["chunk", "reset", "step", "step_result"]
    assert events[2]["thought"] == "Let me check."
    assert answer_of(events) == "9"


def test_a_tool_call_written_out_as_json_text_still_runs(chat_id, monkeypatch):
    llm = ScriptedLLM(
        text('Checking. {"name": "calculator", "parameters": {"expression": "3*3"}}'),
        text("9"),
    )
    events = run(chat_id, monkeypatch, llm)
    assert "reset" in kinds(events)
    step = next(e for e in events if e["type"] == "step")
    assert step["tool"] == "calculator" and step["thought"] == "Checking."
    assert answer_of(events) == "9"


def test_small_talk_never_binds_tools(chat_id, monkeypatch):
    llm = ScriptedLLM(text("Hi!"))
    run(chat_id, monkeypatch, llm, question="hello!")
    assert llm.with_tools == [False]


def test_messages_about_earlier_replies_skip_tools_when_there_is_history(chat_id, monkeypatch):
    llm = ScriptedLLM(text("As I said: x"))
    run(chat_id, monkeypatch, llm, question="repeat what you wrote", history=[("q", "a")])
    assert llm.with_tools == [False]


def test_stated_facts_are_pinned_in_the_system_prompt(chat_id, monkeypatch):
    llm = ScriptedLLM(text("John"))
    run(chat_id, monkeypatch, llm, question="what is my name?",
        history=[("my name is actually John", "Got it."), ("what is 2+2?", "4")])
    system = llm.calls[0][0].content
    assert "overrides anything in the documents" in system and "- my name is actually John" in system
    assert "what is 2+2?" not in system


def test_only_offered_tools_are_described_to_the_model(chat_id, monkeypatch):
    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="run a quick python check of recursion", permission_mode="safe")
    system = llm.calls[0][0].content
    assert "run_python" not in system and "calculator" in system
    assert "run_python" not in llm.bound_names

    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="run a quick python check of recursion", permission_mode="bypass")
    assert "run_python" in llm.calls[0][0].content and "run_python" in llm.bound_names


def test_memory_tools_only_reach_the_model_when_the_message_asks(chat_id, monkeypatch):
    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="explain recursion")
    assert not llm.bound_names & {"compact_context", "keep_only_context", "forget_context", "remember"}

    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="remember my dog is Rex")
    assert "remember" in llm.bound_names and "forget_context" not in llm.bound_names

    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="forget everything about john in the context")
    assert "forget_context" in llm.bound_names


# --- code execution permissions ---

CODE = {"code": "print(1)"}


def spy_on_python(monkeypatch):
    ran = []
    monkeypatch.setattr(agent_tools, "run_python_code", lambda code, workspace: ran.append(code) or "1")
    return ran


def drive_with_decision(chat_id, monkeypatch, approve, mode="ask"):
    monkeypatch.setattr(agent, "_chat_llm", lambda: ScriptedLLM(call("run_python", CODE), text("done")))
    config = {**chat_manager.get_config(chat_id), "permission_mode": mode}
    events = []
    for event in agent.run_agent(chat_id, "run some python", [], config):
        events.append(event)
        if event["type"] == "permission_request":
            assert agent_tools.resolve_permission_request(chat_id, event["id"], approve)
    return events


def test_ask_mode_waits_for_approval_then_runs(chat_id, monkeypatch):
    ran = spy_on_python(monkeypatch)
    events = drive_with_decision(chat_id, monkeypatch, approve=True)
    request = next(e for e in events if e["type"] == "permission_request")
    assert request["code"] == "print(1)" and request["tool"] == "run_python" and request["step"] == 1
    assert kinds(events).index("permission_request") < kinds(events).index("step_result")
    assert ran == ["print(1)"]
    assert next(e for e in events if e["type"] == "step_result")["status"] == "ok"


def test_denied_code_does_not_run_and_the_model_is_told(chat_id, monkeypatch):
    ran = spy_on_python(monkeypatch)
    events = drive_with_decision(chat_id, monkeypatch, approve=False)
    result = next(e for e in events if e["type"] == "step_result")
    assert ran == [] and result["status"] == "denied" and "did not allow" in result["result"]


def test_bypass_runs_without_asking(chat_id, monkeypatch):
    ran = spy_on_python(monkeypatch)
    events = drive_with_decision(chat_id, monkeypatch, approve=True, mode="bypass")
    assert "permission_request" not in kinds(events) and ran == ["print(1)"]


def test_an_unanswered_request_counts_as_denied(chat_id, monkeypatch):
    ran = spy_on_python(monkeypatch)
    monkeypatch.setattr(agent, "wait_for_permission", lambda request_id: False)
    monkeypatch.setattr(agent, "_chat_llm", lambda: ScriptedLLM(call("run_python", CODE), text("done")))
    config = {**chat_manager.get_config(chat_id), "permission_mode": "ask"}
    events = list(agent.run_agent(chat_id, "run some python", [], config))
    assert ran == [] and next(e for e in events if e["type"] == "step_result")["status"] == "denied"


def test_safe_mode_cannot_run_code_even_if_the_model_asks(chat_id, monkeypatch):
    ran = spy_on_python(monkeypatch)
    events = drive_with_decision(chat_id, monkeypatch, approve=True, mode="safe")
    result = next(e for e in events if e["type"] == "step_result")
    assert ran == [] and result["status"] == "error" and "no tool called" in result["result"]


# --- documents first ---

def test_the_web_is_off_limits_until_the_documents_have_been_searched(chat_id, monkeypatch):
    monkeypatch.setattr(agent_tools, "has_documents", lambda path: True)
    monkeypatch.setattr(agent_tools, "list_documents", lambda *a, **k: [{"name": "resume", "chunk_count": 1}])
    web_calls = []

    def web_search(self, query):
        """Search the web."""
        web_calls.append(query)
        return "web result"

    def search_documents(self, query):
        """Search the documents."""
        return "doc result"

    monkeypatch.setattr(agent_tools.ToolBox, "web_search", web_search)
    monkeypatch.setattr(agent_tools.ToolBox, "search_documents", search_documents)

    llm = ScriptedLLM(call("web_search", {"query": "refund days"}), call("search_documents", {"query": "refund days"}, "c2"),
                      call("web_search", {"query": "refund days"}, "c3"), text("answer"))
    events = run(chat_id, monkeypatch, llm, question="how many days do I have to return an item?")
    statuses = [e["status"] for e in events if e["type"] == "step_result"]
    assert statuses == ["skipped", "ok", "ok"]
    assert web_calls == ["refund days"]   # only the call made after the documents were searched
    assert "resume" in llm.calls[0][0].content


# --- fallback and glitches ---

def test_a_model_without_tool_support_triggers_the_fallback(chat_id, monkeypatch):
    assert run(chat_id, monkeypatch, ScriptedLLM(bind_error=NotImplementedError("no tools"))) == [{"type": "fallback"}]
    unsupported = ScriptedLLM(stream_error=Exception("model does not support tools"))
    assert run(chat_id, monkeypatch, unsupported) == [{"type": "fallback"}]


def test_other_provider_errors_propagate(chat_id, monkeypatch):
    with pytest.raises(Exception, match="insufficient_quota"):
        run(chat_id, monkeypatch, ScriptedLLM(stream_error=Exception("insufficient_quota")))


def test_an_empty_answer_is_retried_once_and_the_junk_is_not_remembered(chat_id, monkeypatch):
    llm = ScriptedLLM(text("{}"), text("A real answer"))
    events = run(chat_id, monkeypatch, llm, question="tell me a joke")
    assert answer_of(events) == "A real answer"
    assert llm.with_tools == [True, False]
    assert [e["answer"] for e in chat_manager.get_context(chat_id)] == ["A real answer"]


def test_two_empty_answers_end_with_an_apology_that_is_not_remembered(chat_id, monkeypatch):
    events = run(chat_id, monkeypatch, ScriptedLLM(text("{}"), text("[]")), question="tell me a joke")
    assert "couldn't come up with an answer" in answer_of(events)
    assert chat_manager.get_context(chat_id) == []
    assert kinds(events)[-1] == "complete"


def test_sources_are_cited_when_documents_were_searched_and_the_answer_leans_on_them(chat_id, monkeypatch):
    from langchain_core.documents import Document
    doc = Document(page_content="45 days", metadata={"file_name": "x/policy.md", "start_index": 0, "chunk_length": 7})
    monkeypatch.setattr(agent_tools, "has_documents", lambda path: True)
    monkeypatch.setattr(agent_tools, "list_documents", lambda *a, **k: [{"name": "policy", "chunk_count": 1}])
    monkeypatch.setattr(agent_tools, "get_retriever", lambda path, top_k: type("R", (), {"invoke": lambda self, q: [doc]})())

    llm = ScriptedLLM(call("search_documents", {"query": "returns"}), text("You have **45 days**."))
    events = run(chat_id, monkeypatch, llm, question="how long do I have to return something?")
    sources = next(e for e in events if e["type"] == "sources")
    assert sources["content"] == [{"name": "policy", "start_index": 0, "length": 7}]


def test_a_malformed_text_tool_call_is_retried_instead_of_shown(chat_id, monkeypatch):
    broken = '{"name": "run_python", "parameters": {"code": "print("hi")"}}'    # unescaped quotes: not valid JSON
    llm = ScriptedLLM(text(broken), text("Here is a plain answer."))
    events = run(chat_id, monkeypatch, llm, question="compare heat pumps with a table and code")
    assert answer_of(events) == "Here is a plain answer."
    assert kinds(events).count("reset") == 1 and "step" not in kinds(events)
    nudge = llm.calls[1][-1]
    assert type(nudge).__name__ == "HumanMessage" and "not valid JSON" in nudge.content
    assert llm.calls[1][-2].content == broken                         # the model sees what it wrote
    assert [e["answer"] for e in chat_manager.get_context(chat_id)] == ["Here is a plain answer."]


def test_a_malformed_tool_call_is_only_retried_twice(chat_id, monkeypatch):
    broken = '{"name": "run_python", "parameters": {"code": "print("hi")"}}'
    llm = ScriptedLLM(text(broken), text(broken), text(broken))
    events = run(chat_id, monkeypatch, llm, question="compare heat pumps with a table and code")
    assert kinds(events).count("reset") == 2 and len(llm.calls) == 3      # gives up; the last text is the answer


def test_auto_compact_runs_after_the_answer_and_says_so(chat_id, monkeypatch):
    import context_manager
    monkeypatch.setattr(context_manager, "maybe_auto_compact", lambda chat: True)
    events = run(chat_id, monkeypatch, ScriptedLLM(text("hi")), question="tell me a joke")
    assert kinds(events) == ["chunk", "notice", "complete"]
    assert "folded" in events[1]["message"]

    monkeypatch.setattr(context_manager, "maybe_auto_compact", lambda chat: False)
    assert "notice" not in kinds(run(chat_id, monkeypatch, ScriptedLLM(text("hi")), question="tell me a joke"))


def test_the_model_is_told_todays_date_and_to_search_for_recent_things(chat_id, monkeypatch):
    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="who won the 2026 world cup?")
    system = llm.calls[0][0].content
    assert f"Today's date is {agent.today_string()}" in system
    assert "{today}" not in system
    assert "hasn't happened yet" in system and "search the web" in system


def test_today_string_reads_naturally():
    import re
    assert re.fullmatch(r"[A-Z][a-z]+day, [A-Z][a-z]+ \d{1,2}, \d{4}", agent.today_string())
    assert " 0" not in agent.today_string()    # 'October 5', not 'October 05'


# --- code only when the message is about code ---

def test_run_python_is_not_offered_unless_the_message_is_about_computing_or_code(chat_id, monkeypatch):
    for question in ("compare heat pumps and gas furnaces in a table", "summarize this and show a table", "what is the capital of France?"):
        llm = ScriptedLLM(text("ok"))
        run(chat_id, monkeypatch, llm, question=question, permission_mode="bypass")
        assert "run_python" not in llm.bound_names and "run_python" not in llm.calls[0][0].content, question

    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="use python to sum the numbers 1 to 100", permission_mode="bypass")
    assert "run_python" in llm.bound_names


def test_a_follow_up_to_a_code_exchange_keeps_code_available(chat_id, monkeypatch):
    llm = ScriptedLLM(text("ok"))
    run(chat_id, monkeypatch, llm, question="yes, go ahead", permission_mode="bypass",
        history=[("can you run that script?", "I can, shall I?")])
    assert "run_python" in llm.bound_names


# --- small documents are read up front ---

HOMEWORK = ("# Project 2\n\nDue: Friday, October 30, 2026\n\n1. Parser - 3 hours\n2. Thread pool - 6 hours\n")


@pytest.fixture
def homework(chat_id, monkeypatch):
    import os
    monkeypatch.setattr(agent_tools, "has_documents", lambda path: True)
    monkeypatch.setattr(agent_tools, "list_documents", lambda *a, **k: [{"name": "homework2", "chunk_count": 1}])
    _, files = chat_manager.get_chat_paths(chat_id)
    os.makedirs(files, exist_ok=True)
    with open(os.path.join(files, "homework2.md"), "w") as f:
        f.write(HOMEWORK)
    return chat_id


SUMMARY_REQUEST = "summarize this project, show the due date, and show a table of how long each component should take"


def test_pointing_at_this_project_reads_the_small_document_for_the_model(homework, monkeypatch):
    llm = ScriptedLLM(text("Summary, due Oct 30, table."))
    events = run(homework, monkeypatch, llm, question=SUMMARY_REQUEST)

    assert kinds(events) == ["step", "step_result", "chunk", "complete"]
    step, result = events[0], events[1]
    assert (step["tool"], step["args"], step["id"]) == ("read_document", {"name": "homework2"}, 1)
    assert result["status"] == "ok" and "Due: Friday, October 30, 2026" in result["result"]
    assert events[-1]["steps"] == 1

    user_message = llm.calls[0][-1]
    assert user_message.content.startswith(SUMMARY_REQUEST) and "Parser - 3 hours" in user_message.content
    assert "already read for you" in user_message.content
    system = llm.calls[0][0].content
    assert "included in their message" in system and "You have no tools" in system and "search_documents" not in system


def test_with_the_document_in_hand_and_nothing_asking_for_tools_the_model_answers_directly(homework, monkeypatch):
    llm = ScriptedLLM(text("answer"))
    run(homework, monkeypatch, llm, question=SUMMARY_REQUEST, permission_mode="bypass")
    assert llm.with_tools == [False]


def test_the_remembered_question_is_the_users_words_not_the_document_text(homework, monkeypatch):
    run(homework, monkeypatch, ScriptedLLM(text("answer")), question=SUMMARY_REQUEST)
    (entry,) = chat_manager.get_context(homework)
    assert entry["question"] == SUMMARY_REQUEST and "Parser - 3 hours" not in entry["question"]


def test_asking_for_the_web_after_a_preread_keeps_tools_but_not_the_document_ones(homework, monkeypatch):
    llm = ScriptedLLM(call("web_search", {"query": "threads tutorial"}), text("answer"))
    monkeypatch.setattr(agent_tools, "search_web", lambda query, **kw: ([{"title": "T", "href": "", "body": "b"}], ""))
    events = run(homework, monkeypatch, llm, question="summarize this project and find an online tutorial for it")
    assert llm.with_tools[0] is True
    assert not llm.bound_names & {"search_documents", "read_document", "list_documents", "run_python"}
    assert "web_search" in llm.bound_names
    # the pre-read counts as having looked at the documents, so the web isn't redirected
    assert [e["status"] for e in events if e["type"] == "step_result"] == ["ok", "ok"]


def test_asking_for_code_after_a_preread_offers_python(homework, monkeypatch):
    llm = ScriptedLLM(text("answer"))
    run(homework, monkeypatch, llm, question="solve the problems in this homework and verify them with python",
        permission_mode="bypass")
    assert "run_python" in llm.bound_names and not llm.bound_names & {"search_documents", "read_document"}
    assert llm.with_tools == [True]


def test_step_numbers_continue_after_the_preread(homework, monkeypatch):
    llm = ScriptedLLM(call("calculator", {"expression": "3+6"}), text("9 hours"))
    events = run(homework, monkeypatch, llm, question="calculate the total hours for this project")
    steps = [e for e in events if e["type"] == "step"]
    assert [(e["id"], e["tool"]) for e in steps] == [(1, "read_document"), (2, "calculator")]
    assert [e["id"] for e in events if e["type"] == "step_result"] == [1, 2]
    assert events[-1]["steps"] == 2


def test_documents_too_big_to_hand_over_whole_are_searched_instead(homework, monkeypatch):
    import os
    _, files = chat_manager.get_chat_paths(homework)
    with open(os.path.join(files, "homework2.md"), "w") as f:
        f.write("x" * (agent_tools.MAX_AUTO_READ_CHARS + 1))
    llm = ScriptedLLM(text("answer"))
    events = run(homework, monkeypatch, llm, question=SUMMARY_REQUEST)
    assert "step" not in kinds(events) and "search_documents" in llm.bound_names and llm.with_tools == [True]
    assert "Search them" in llm.calls[0][0].content


def test_a_question_that_does_not_point_at_the_document_does_not_trigger_a_preread(homework, monkeypatch):
    llm = ScriptedLLM(text("4"))
    events = run(homework, monkeypatch, llm, question="what is 2+2?")
    assert "step" not in kinds(events) and "Parser - 3 hours" not in llm.calls[0][-1].content


def test_every_uploaded_document_is_read_when_they_fit_together(homework, monkeypatch):
    import os
    monkeypatch.setattr(agent_tools, "list_documents", lambda *a, **k: [
        {"name": "homework2", "chunk_count": 1}, {"name": "rubric", "chunk_count": 1}])
    _, files = chat_manager.get_chat_paths(homework)
    with open(os.path.join(files, "rubric.md"), "w") as f:
        f.write("Rubric: correctness 50%")
    llm = ScriptedLLM(text("answer"))
    events = run(homework, monkeypatch, llm, question="summarize my documents and the assignment")
    assert [e["args"]["name"] for e in events if e["type"] == "step"] == ["homework2", "rubric"]
    content = llm.calls[0][-1].content
    assert "[homework2]" in content and "[rubric]" in content and "documents" in content
