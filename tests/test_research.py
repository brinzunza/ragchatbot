import json
import threading

import chat_manager
import pytest

import research


class Chain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def classify(error):
    text = str(error)
    return ("no_credits", "no credits") if "quota" in text else ("error", text)


def scripted_loop(*outcomes):
    """agent_loop stand-in: each call plays one outcome - a note, or an exception to raise."""
    outcomes = list(outcomes)
    attempts = []

    def loop(messages, toolbox, max_steps, include_context_tools=True):
        attempts.append(messages[-1].content)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        yield {"type": "step", "id": 1, "tool": "web_search", "args": {}, "thought": ""}
        return outcome

    loop.attempts = attempts
    return loop


@pytest.fixture
def job(chat_id, monkeypatch):
    monkeypatch.setattr(research, "ToolBox", lambda *a, **k: type("Box", (), {
        "document_names": lambda self: [],
        "tools": lambda self, include_context_tools=True: [],
    })())
    return {"id": "abc123def456", "topic": "heat pumps", "status": "running", "phase": "planning", "current": "",
            "plan": [], "notes": [], "report": "", "error": "", "error_code": "", "created_at": "", "finished_at": ""}


def run_job(chat_id, job, monkeypatch, planner="1. First sub-question\n- Second sub-question", loop=None,
            writer="REPORT", cancel=None):
    monkeypatch.setattr(research, "get_research_planner", lambda: Chain(planner))
    monkeypatch.setattr(research, "get_research_writer", lambda: Chain(writer))
    monkeypatch.setattr(research, "agent_loop", loop or scripted_loop("note one", "note two"))
    research._save(chat_id, job)
    research._active[job["id"]] = cancel or threading.Event()
    research._run(chat_id, job, research._active[job["id"]], classify)
    return research.get_job(chat_id, job["id"])


def test_plans_are_parsed_from_lists_of_any_style():
    assert research._parse_plan("1. What is X?\n2) Why does Y matter?\n- How does Z work?\n* And W here?") == [
        "What is X?", "Why does Y matter?", "How does Z work?", "And W here?"]
    assert research._parse_plan("short\nA proper sub-question") == ["A proper sub-question"]   # too-short lines are noise
    assert len(research._parse_plan("\n".join(f"Question number {i} here" for i in range(9)))) == research.MAX_SUBQUESTIONS


def test_a_job_plans_researches_each_question_then_writes_the_report(chat_id, job, monkeypatch):
    result = run_job(chat_id, job, monkeypatch)
    assert result["status"] == "done" and result["report"] == "REPORT"
    assert result["plan"] == ["First sub-question", "Second sub-question"]
    assert [(n["question"], n["text"], n["tools"]) for n in result["notes"]] == [
        ("First sub-question", "note one", ["web_search"]), ("Second sub-question", "note two", ["web_search"])]
    assert result["finished_at"] and result["current"] == ""
    entry = chat_manager.get_context(chat_id)[0]
    assert entry["question"] == "/research heat pumps" and entry["answer"] == "REPORT"


def test_an_unparseable_plan_falls_back_to_the_topic_itself(chat_id, job, monkeypatch):
    result = run_job(chat_id, job, monkeypatch, planner="ok", loop=scripted_loop("only note"))
    assert result["plan"] == ["heat pumps"] and result["status"] == "done"


def test_cancelling_stops_the_job_before_it_writes_a_report(chat_id, job, monkeypatch):
    flag = threading.Event()
    flag.set()
    result = run_job(chat_id, job, monkeypatch, cancel=flag)
    assert result["status"] == "cancelled" and result["report"] == "" and result["notes"] == []
    assert chat_manager.get_context(chat_id) == []


def test_a_failed_sub_question_is_retried_once(chat_id, job, monkeypatch):
    loop = scripted_loop(Exception("connection dropped"), "recovered note", "note two")
    result = run_job(chat_id, job, monkeypatch, loop=loop)
    assert result["status"] == "done" and result["notes"][0]["text"] == "recovered note"
    assert len(loop.attempts) == 3


def test_a_sub_question_that_keeps_failing_is_noted_and_the_job_goes_on(chat_id, job, monkeypatch):
    loop = scripted_loop(Exception("dropped"), Exception("dropped again"), "note two")
    result = run_job(chat_id, job, monkeypatch, loop=loop)
    assert result["status"] == "done"
    assert "could not be researched" in result["notes"][0]["text"] and result["notes"][1]["text"] == "note two"


def test_an_error_every_step_would_hit_fails_the_whole_job_with_its_message(chat_id, job, monkeypatch):
    result = run_job(chat_id, job, monkeypatch, loop=scripted_loop(Exception("insufficient_quota")))
    assert result["status"] == "error" and result["error_code"] == "no_credits" and result["error"] == "no credits"
    assert result["report"] == ""


def test_a_failing_planner_fails_the_job(chat_id, job, monkeypatch):
    result = run_job(chat_id, job, monkeypatch, planner=Exception("boom"))
    assert result["status"] == "error" and result["error"] == "boom"


def test_the_report_writer_gets_one_retry(chat_id, job, monkeypatch):
    monkeypatch.setattr(research, "get_research_writer", lambda: None)   # replaced inside run_job; keep import happy
    calls = []

    class Flaky:
        def invoke(self, inputs):
            calls.append(inputs)
            if len(calls) == 1:
                raise Exception("dropped")
            return "SECOND TRY"

    monkeypatch.setattr(research, "get_research_planner", lambda: Chain("1. First sub-question"))
    monkeypatch.setattr(research, "get_research_writer", lambda: Flaky())
    monkeypatch.setattr(research, "agent_loop", scripted_loop("note"))
    research._save(chat_id, job)
    research._active[job["id"]] = threading.Event()
    research._run(chat_id, job, research._active[job["id"]], classify)
    assert research.get_job(chat_id, job["id"])["report"] == "SECOND TRY" and len(calls) == 2


def test_a_running_job_with_no_live_thread_is_reported_interrupted(chat_id, job):
    research._save(chat_id, job)    # status "running", but nothing is running it
    result = research.get_job(chat_id, job["id"])
    assert result["status"] == "interrupted" and "restarted" in result["error"]
    assert research.get_job(chat_id, job["id"])["status"] == "interrupted"    # and it's saved that way


def test_job_ids_are_validated_before_touching_the_filesystem(chat_id):
    assert research.get_job(chat_id, "../../etc/passwd") is None
    assert research.get_job(chat_id, "nothere00000") is None
    assert research.get_job("no-such-chat", "abc123def456") is None


def test_only_one_job_runs_per_chat_and_topics_are_required(chat_id, monkeypatch):
    monkeypatch.setattr(research, "_run", lambda *args: None)    # start the thread, but do no work
    first = research.start_research(chat_id, "  solar panels  ", classify)
    assert first["topic"] == "solar panels" and first["status"] == "running"
    with pytest.raises(ValueError, match="already running"):
        research.start_research(chat_id, "another topic", classify)
    with pytest.raises(ValueError, match="topic"):
        research.start_research(chat_id, "   ", classify)
    with pytest.raises(ValueError, match="not found"):
        research.start_research("no-such-chat", "x", classify)
    assert research.cancel_job(chat_id, first["id"]) is True
    assert research.cancel_job(chat_id, "unknown00000") is False
    assert json.loads(open(research._job_path(chat_id, first["id"])).read())["topic"] == "solar panels"


def test_researchers_are_told_todays_date(chat_id, job, monkeypatch):
    seen = []

    def loop(messages, toolbox, max_steps, include_context_tools=True):
        seen.append(messages[0].content)
        return "note"
        yield   # pragma: no cover - makes this a generator

    run_job(chat_id, job, monkeypatch, planner="1. First sub-question", loop=loop)
    assert research.today_string() in seen[0] and "{today}" not in seen[0]
