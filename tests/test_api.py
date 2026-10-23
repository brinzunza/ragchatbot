import json

import chat_manager
import pytest
from fastapi.testclient import TestClient

import agent_tools
import context_manager
import main
import research
import llm_config


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def api_chat(client):
    return client.post("/chats", json={"name": "api test"}).json()["id"]


def sse(response):
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


# --- config ---

def test_config_defaults_roundtrip_and_validation(client, api_chat):
    defaults = client.get(f"/chats/{api_chat}/config").json()
    assert defaults == {"context_enabled": True, "max_context_entries": 4, "max_context_tokens": 3000,
                        "auto_compact": False, "agent_enabled": True, "permission_mode": "ask",
                        "max_agent_steps": 8, "web_search_enabled": True}

    changed = {**defaults, "permission_mode": "bypass", "max_agent_steps": 3, "web_search_enabled": False}
    assert client.put(f"/chats/{api_chat}/config", json=changed).json() == changed
    assert client.get(f"/chats/{api_chat}/config").json() == changed

    for bad in ({"permission_mode": "yolo"}, {"max_agent_steps": 0}, {"max_agent_steps": 99}, {"max_context_entries": 21},
                {"max_context_tokens": 100}, {"max_context_tokens": 99999}):
        assert client.put(f"/chats/{api_chat}/config", json={**defaults, **bad}).status_code == 422
    assert client.get("/chats/nope/config").status_code == 404


# --- context ---

def test_context_endpoints(client, api_chat, monkeypatch):
    context_manager.record_turn(api_chat, "my name is John", "ok")
    context_manager.record_turn(api_chat, "capital of France?", "Paris")
    body = client.get(f"/chats/{api_chat}/context").json()
    assert [e["question"] for e in body["entries"]] == ["my name is John", "capital of France?"]

    removed = client.post(f"/chats/{api_chat}/context/forget", json={"topic": "john"}).json()
    assert removed["removed"] == 1 and [e["question"] for e in removed["entries"]] == ["capital of France?"]
    assert client.post(f"/chats/{api_chat}/context/forget", json={"topic": "  "}).status_code == 400
    assert client.post(f"/chats/{api_chat}/context/keep", json={"topic": ""}).status_code == 400

    entry_id = removed["entries"][0]["id"]
    assert client.delete(f"/chats/{api_chat}/context/{entry_id}").status_code == 200
    assert client.delete(f"/chats/{api_chat}/context/{entry_id}").status_code == 404

    context_manager.record_turn(api_chat, "q", "a")
    assert client.delete(f"/chats/{api_chat}/context").status_code == 200
    assert client.get(f"/chats/{api_chat}/context").json()["entries"] == []


def test_compact_accepts_an_optional_focus_and_reports_model_errors(client, api_chat, monkeypatch):
    context_manager.record_turn(api_chat, "q", "a")
    seen = []
    monkeypatch.setattr(context_manager, "compact", lambda chat_id, focus="": seen.append(focus) or {"before": 1, "after": 1})
    assert client.post(f"/chats/{api_chat}/context/compact").status_code == 200
    assert client.post(f"/chats/{api_chat}/context/compact", json={"focus": "the budget"}).status_code == 200
    assert seen == ["", "the budget"]

    def fail(chat_id, focus=""):
        raise Exception("Your credit balance is too low to access the Anthropic API")
    monkeypatch.setattr(context_manager, "compact", fail)
    response = client.post(f"/chats/{api_chat}/context/compact")
    assert response.status_code == 500 and response.json()["detail"]["code"] == "no_credits"


def test_context_usage_endpoints(client, api_chat):
    context_manager.record_turn(api_chat, "my name is John", "ok, John")
    usage = client.get(f"/chats/{api_chat}/context/usage").json()
    assert usage["total_exchanges"] == 1 and usage["sent_exchanges"] == 1 and usage["left_out"] == 0
    assert usage["budget_tokens"] == 3000 and 0 < usage["used_tokens"] < 50
    assert client.get(f"/chats/{api_chat}/context").json()["usage"] == usage
    # changes to the context come back with fresh usage
    forgotten = client.post(f"/chats/{api_chat}/context/forget", json={"topic": "john"}).json()
    assert forgotten["usage"]["total_exchanges"] == 0 and forgotten["usage"]["used_tokens"] == 0
    assert client.get("/chats/nope/context/usage").status_code == 404


def test_fold_endpoint(client, api_chat, monkeypatch):
    config = chat_manager.get_config(api_chat)
    client.put(f"/chats/{api_chat}/config", json={**config, "max_context_entries": 1})
    for i in range(3):
        context_manager.record_turn(api_chat, f"q{i}", "a")
    monkeypatch.setattr(context_manager, "_fold_into_summary", lambda pieces, focus="": "FOLDED SUMMARY")
    body = client.post(f"/chats/{api_chat}/context/fold").json()
    assert body["folded"] == 2 and body["usage"]["left_out"] == 0
    assert [e["kind"] for e in body["entries"]] == ["summary", "turn"]
    assert client.post(f"/chats/{api_chat}/context/fold").json()["folded"] == 0     # nothing left to fold

    def fail(pieces, focus=""):
        raise Exception("insufficient_quota")
    context_manager.record_turn(api_chat, "q9", "a")
    monkeypatch.setattr(context_manager, "_fold_into_summary", fail)
    response = client.post(f"/chats/{api_chat}/context/fold")
    assert response.status_code == 500 and response.json()["detail"]["code"] == "no_credits"


def test_undo_endpoint(client, api_chat):
    context_manager.record_turn(api_chat, "q1", "a1")
    assert client.post(f"/chats/{api_chat}/context/undo", json={"question": "other"}).json() == {"removed": False}
    assert client.post(f"/chats/{api_chat}/context/undo", json={"question": "q1"}).json() == {"removed": True}
    assert client.get(f"/chats/{api_chat}/context").json()["entries"] == []
    assert client.post("/chats/nope/context/undo", json={"question": "q1"}).status_code == 404


# --- provider settings ---

def test_provider_settings_flow(client, env_file):
    status = client.get("/settings/api-key").json()
    assert (status["provider_id"], status["has_key"]) == ("ollama", False)

    # keys pasted into the wrong box are caught
    assert client.post("/settings/api-key", json={"api_key": "sk-nope", "provider": "anthropic"}).status_code == 400
    assert client.post("/settings/api-key", json={"api_key": "sk-ant-abc", "provider": "openai"}).status_code == 400
    assert client.post("/settings/api-key", json={"api_key": "  ", "provider": "openai"}).status_code == 400

    status = client.post("/settings/api-key", json={"api_key": "sk-ant-abcdefghijkl", "provider": "anthropic"}).json()
    assert (status["provider_id"], status["provider"], status["has_key"]) == ("anthropic", "Claude", True)
    assert status["providers"]["anthropic"]["has_key"] and not status["providers"]["openai"]["has_key"]
    assert "abcdefg" not in json.dumps(status)    # the key itself never comes back

    assert client.post("/settings/provider", json={"provider": "openai"}).status_code == 400    # no key yet
    assert client.post("/settings/provider", json={"provider": "gemini"}).status_code == 422
    assert client.post("/settings/provider", json={"provider": "ollama"}).json()["provider_id"] == "ollama"
    assert client.post("/settings/provider", json={"provider": "anthropic"}).json()["provider_id"] == "anthropic"

    status = client.delete("/settings/api-key", params={"provider": "anthropic"}).json()
    assert status["provider_id"] == "ollama" and not status["providers"]["anthropic"]["has_key"]
    assert client.delete("/settings/api-key", params={"provider": "gemini"}).status_code == 422


def test_ollama_model_endpoints(client, monkeypatch):
    monkeypatch.setattr(llm_config, "list_ollama_models", lambda: [{"name": "llama3.1:8b", "size_gb": 4.9}])
    saved = []
    monkeypatch.setattr(llm_config, "_write_env_value", lambda name, value: saved.append((name, value)))
    listing = client.get("/settings/ollama-models").json()
    assert listing["reachable"] and listing["models"][0]["name"] == "llama3.1:8b" and listing["selected"] == "llama3.2:latest"
    assert client.post("/settings/ollama-model", json={"model": "llama3.1:8b"}).status_code == 200
    assert saved == [("OLLAMA_MODEL", "llama3.1:8b")]
    assert client.post("/settings/ollama-model", json={"model": "nope:1b"}).status_code == 400

    monkeypatch.setattr(llm_config, "list_ollama_models", lambda: None)
    assert client.get("/settings/ollama-models").json()["reachable"] is False


# --- permissions ---

def test_permission_endpoint(client, api_chat):
    assert client.post(f"/chats/{api_chat}/agent/permissions/unknown", json={"approve": True}).status_code == 404
    request_id = agent_tools.create_permission_request(api_chat)
    assert client.post(f"/chats/{api_chat}/agent/permissions/{request_id}", json={"approve": False}).status_code == 200
    assert agent_tools.wait_for_permission(request_id, timeout=1) is False
    assert client.post("/chats/nope/agent/permissions/x", json={"approve": True}).status_code == 404


# --- chat stream ---

def fake_agent(*events, error=None):
    def run(chat_id, question, history, config):
        yield from events
        if error:
            raise error
    return run


def test_the_agent_stream_is_passed_through_as_server_sent_events(client, api_chat, monkeypatch):
    monkeypatch.setattr(main, "run_agent", fake_agent(
        {"type": "step", "id": 1, "tool": "calculator", "args": {}, "thought": ""},
        {"type": "chunk", "content": "4"}, {"type": "complete", "response_time": 0.1, "steps": 1}))
    response = client.post(f"/chats/{api_chat}/qa/chat/stream", json={"content": "2+2"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in sse(response)] == ["step", "chunk", "complete"]


def test_a_provider_failure_arrives_as_an_error_event_with_a_code(client, api_chat, monkeypatch):
    monkeypatch.setattr(main, "run_agent", fake_agent(error=Exception("insufficient_quota")))
    (event,) = sse(client.post(f"/chats/{api_chat}/qa/chat/stream", json={"content": "hi"}))
    assert event["type"] == "error" and event["code"] == "no_credits"
    assert "no credits" in event["message"]


def test_a_model_without_tools_falls_back_to_classic_with_a_notice(client, api_chat, monkeypatch):
    monkeypatch.setattr(main, "run_agent", fake_agent({"type": "fallback"}))
    monkeypatch.setattr(main, "classic_events", lambda *a: iter([
        {"type": "chunk", "content": "hi"}, {"type": "complete", "response_time": 0.1}]))
    events = sse(client.post(f"/chats/{api_chat}/qa/chat/stream", json={"content": "hi"}))
    assert [e["type"] for e in events] == ["notice", "chunk", "complete"]
    assert "classic mode" in events[0]["message"]


def test_with_the_agent_off_the_classic_flow_needs_documents(client, api_chat, monkeypatch):
    config = chat_manager.get_config(api_chat)
    client.put(f"/chats/{api_chat}/config", json={**config, "agent_enabled": False})
    monkeypatch.setattr(main, "has_documents", lambda path: False)
    response = client.post(f"/chats/{api_chat}/qa/chat/stream", json={"content": "hi"})
    assert response.status_code == 400 and "upload documents" in response.json()["detail"]

    monkeypatch.setattr(main, "has_documents", lambda path: True)
    monkeypatch.setattr(main, "classic_events", lambda *a: iter([{"type": "complete", "response_time": 0}]))
    assert [e["type"] for e in sse(client.post(f"/chats/{api_chat}/qa/chat/stream", json={"content": "hi"}))] == ["complete"]


def test_an_unknown_chat_is_a_404(client):
    assert client.post("/chats/nope/qa/chat/stream", json={"content": "hi"}).status_code == 404


# --- research ---

def test_research_endpoints(client, api_chat, monkeypatch):
    monkeypatch.setattr(research, "_run", lambda *args: None)
    assert client.post(f"/chats/{api_chat}/research", json={"topic": "  "}).status_code == 400
    started = client.post(f"/chats/{api_chat}/research", json={"topic": "heat pumps"})
    assert started.status_code == 200
    job_id = started.json()["id"]
    assert client.post(f"/chats/{api_chat}/research", json={"topic": "another"}).status_code == 409
    assert client.get(f"/chats/{api_chat}/research/{job_id}").json()["topic"] == "heat pumps"
    assert client.post(f"/chats/{api_chat}/research/{job_id}/cancel").status_code == 200
    assert client.get(f"/chats/{api_chat}/research/nothere00000").status_code == 404
    assert client.post(f"/chats/{api_chat}/research/nothere00000/cancel").status_code == 404
    assert client.post("/chats/nope/research", json={"topic": "x"}).status_code == 404
