import pytest

import agent


@pytest.mark.parametrize("message", [
    "hello!", "hi", "Hey there", "thanks", "thank you so much", "good morning", "ok", "who are you", "what can you do",
])
def test_small_talk_skips_tools(message):
    assert agent.skips_tools(message)


@pytest.mark.parametrize("message", [
    "hello, what does the contract say about refunds?",
    "what is my name?",
    "summarize the document",
    "thanks, now compute 17% of 4320",
])
def test_real_questions_keep_tools(message):
    assert not agent.skips_tools(message)


@pytest.mark.parametrize("message", [
    "my name is actually John", "Actually, my name is John.", "call me J", "please call me Jo",
    "i prefer short answers", "from now on answer in spanish", "my email is a@b.co",
])
def test_the_user_stating_a_fact_about_themselves_skips_tools(message):
    assert agent.skips_tools(message)


def test_a_question_mark_turns_tools_back_on():
    assert not agent.skips_tools("my name is? tell me what the document says")
    assert not agent.skips_tools("my manager is asking, summarize the doc")


@pytest.mark.parametrize("message", [
    "repeat the one sentence bio you wrote about me", "what did you say about refunds?", "translate that to spanish",
    "now make it shorter", "rephrase that", "what did I ask you earlier?", "say that again",
])
def test_asking_about_earlier_replies_skips_tools_only_with_history(message):
    assert agent.skips_tools(message, has_history=True)
    assert not agent.skips_tools(message, has_history=False)


def test_new_requests_keep_tools_even_with_history():
    assert not agent.skips_tools("write a one sentence bio about me", has_history=True)
    assert not agent.skips_tools("what is 2+2", has_history=True)


@pytest.mark.parametrize("message, expected", [
    ("/compact remove all context about john", {"compact_context", "keep_only_context", "forget_context"}),
    ("please summarize our conversation", {"compact_context", "keep_only_context", "forget_context"}),
    ("forget everything about john in the context", {"compact_context", "keep_only_context", "forget_context"}),
    ("remember my dog is Rex", {"remember"}),
    ("don't forget that i prefer metric units", {"remember"}),
    ("what is my name?", set()),
    ("remove the second paragraph from the draft", set()),   # an action, but not on the conversation
    ("summarize the document", set()),
])
def test_memory_tools_are_gated_on_the_users_words(message, expected):
    assert agent.context_tools_for(message) == expected


def test_a_text_tool_call_is_recognised_only_for_known_tools():
    found = agent.find_text_tool_call(
        'I will check.\n\n{"name": "search_documents", "parameters": {"query": "days"}}', {"search_documents"})
    assert found["call"]["name"] == "search_documents"
    assert found["call"]["args"] == {"query": "days"}
    assert found["before"] == "I will check."

    fenced = agent.find_text_tool_call('```json\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n```', {"calculator"})
    assert fenced["call"]["args"] == {"expression": "1+1"} and fenced["before"] == ""

    assert agent.find_text_tool_call('{"name": "rm_rf", "parameters": {}}', {"calculator"}) is None
    assert agent.find_text_tool_call('Use {"a": 1} like so', {"calculator"}) is None
    assert agent.find_text_tool_call("no json here", {"calculator"}) is None
    assert agent.find_text_tool_call('{"name": "calculator", "parameters": "not a dict"}', {"calculator"}) is None


@pytest.mark.parametrize("answer, empty", [
    ("{}", True), ("[]", True), (" ", True), ("", True), ('{"a": ""}', False), ("{} hi", False), ("Done.", False),
])
def test_empty_answers_are_recognised(answer, empty):
    assert agent.is_empty_answer(answer) is empty


def test_stated_facts_come_from_earlier_self_statements_only():
    history = [
        ("summary of the earlier conversation", "..."),
        ("my name is actually John", "Got it."),
        ("what is my name?", "John"),
        ("call me J.", "ok"),
    ]
    assert agent.stated_facts(history) == ["my name is actually John", "call me J."]


def test_history_becomes_messages_with_the_summary_first():
    messages = agent.history_to_messages([
        ("summary of the earlier conversation", "they like tea"),
        ("q1", "a1"),
    ])
    assert [type(m).__name__ for m in messages] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert "they like tea" in messages[0].content
    assert messages[1].content == "q1" and messages[2].content == "a1"


@pytest.mark.parametrize("error, unsupported", [
    (NotImplementedError("x"), True),
    (Exception("registry.ollama.ai/library/gemma does not support tools"), True),
    (Exception("Tools are not supported"), True),
    (Exception("connection refused"), False),
    (Exception("insufficient_quota"), False),
])
def test_tool_unsupported_detection(error, unsupported):
    assert agent.is_tools_unsupported(error) is unsupported


@pytest.mark.parametrize("text, expected", [
    ('{"name": "run_python", "parameters": {"code": "print("hi")"}}', True),     # invalid JSON: unescaped quotes
    ('```json\n{"name": "run_python", "parameters": {', True),
    ('  {"name":"calculator"', True),
    ('{"name": "delete_everything", "parameters": {}}', False),                   # not one of our tools
    ('Here is the call: {"name": "run_python"}', False),                          # prose first: not a bare call
    ('The name field is {"name": "x"}', False),
    ("plain answer", False),
])
def test_text_that_starts_as_a_tool_call_is_recognised_even_when_malformed(text, expected):
    assert agent.looks_like_tool_call(text, {"run_python", "calculator"}) is expected
