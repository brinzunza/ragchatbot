import httpx
import pytest

import llm_config as config


def env_lines(env_file):
    return [line for line in env_file.read_text().splitlines() if line]


def test_with_no_keys_ollama_answers():
    assert config.get_active_provider() == "ollama"
    info = config.get_model_info()
    assert (info["provider"], info["location"], info["provider_id"]) == ("Ollama", "local", "ollama")
    assert info["model"] == "llama3.2:latest"


def test_a_key_selects_its_provider_and_openai_wins_a_tie(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert config.get_active_provider() == "anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert config.get_active_provider() == "openai"


def test_an_explicit_choice_beats_the_default_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert config.get_active_provider() == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert config.get_active_provider() == "ollama"   # keys are kept, just not used


def test_choosing_a_provider_without_a_key_falls_through(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert config.get_active_provider() == "ollama"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert config.get_active_provider() == "openai"


def test_an_empty_key_counts_as_no_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert config.get_active_provider() == "ollama"


def test_saving_a_key_writes_env_and_makes_that_provider_active(env_file):
    env_file.write_text("SOMETHING_ELSE=keep-me\n")
    config.set_api_key("sk-ant-abc123456789", "anthropic")
    assert config.get_active_provider() == "anthropic"
    assert env_lines(env_file) == ["SOMETHING_ELSE=keep-me", "ANTHROPIC_API_KEY=sk-ant-abc123456789", "LLM_PROVIDER=anthropic"]

    config.set_api_key("sk-openai-key-123456", "openai")
    assert config.get_active_provider() == "openai"
    assert "OPENAI_API_KEY=sk-openai-key-123456" in env_lines(env_file) and "LLM_PROVIDER=openai" in env_lines(env_file)


def test_replacing_a_key_does_not_duplicate_it(env_file):
    config.set_api_key("sk-first", "openai")
    config.set_api_key("sk-second", "openai")
    assert [l for l in env_lines(env_file) if l.startswith("OPENAI_API_KEY")] == ["OPENAI_API_KEY=sk-second"]


def test_removing_the_active_key_hands_over_to_the_next_provider(env_file):
    config.set_api_key("sk-ant-keep-me-1234", "anthropic")
    config.set_api_key("sk-openai-key-1234", "openai")
    config.clear_api_key()                        # the active one: openai
    assert config.get_active_provider() == "anthropic"
    assert env_lines(env_file) == ["ANTHROPIC_API_KEY=sk-ant-keep-me-1234"]
    config.clear_api_key("anthropic")
    assert config.get_active_provider() == "ollama" and env_lines(env_file) == []


def test_removing_an_inactive_key_leaves_the_choice_alone(env_file):
    config.set_api_key("sk-openai-key-1234", "openai")
    config.set_api_key("sk-ant-keep-me-1234", "anthropic")   # now active
    config.clear_api_key("openai")
    assert config.get_active_provider() == "anthropic"
    assert "LLM_PROVIDER=anthropic" in env_lines(env_file)


def test_switching_providers_keeps_keys(env_file):
    config.set_api_key("sk-openai-key-1234", "openai")
    config.set_api_key("sk-ant-keep-me-1234", "anthropic")
    config.set_active_provider("ollama")
    assert config.get_active_provider() == "ollama"
    config.set_active_provider("openai")
    assert config.get_active_provider() == "openai"
    assert config.get_providers_status()["anthropic"]["has_key"]


def test_switching_to_a_provider_with_no_key_is_refused():
    with pytest.raises(ValueError, match="OpenAI API key"):
        config.set_active_provider("openai")
    with pytest.raises(ValueError):
        config.set_active_provider("gemini")
    with pytest.raises(ValueError):
        config.set_api_key("sk-x", "gemini")


def test_keys_are_masked_for_display(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
    masked = config.get_masked_api_key("openai")
    assert masked.startswith("sk-1") and masked.endswith("cdef") and "567890ab" not in masked
    assert config.get_masked_api_key("anthropic") is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "short")
    assert config.get_masked_api_key("anthropic") == "*****"


def test_provider_status_covers_every_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    status = config.get_providers_status()
    assert set(status) == {"openai", "anthropic", "ollama"}
    assert status["anthropic"]["has_key"] and not status["openai"]["has_key"] and not status["ollama"]["has_key"]
    assert status["ollama"]["model"] == "llama3.2:latest" and status["anthropic"]["model"] == "claude-haiku-5-5"


def test_get_llm_builds_the_active_providers_model(monkeypatch):
    assert type(config.get_llm()).__name__ == "ChatOllama"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert type(config.get_llm(num_predict=10, num_ctx=10)).__name__ == "ChatOpenAI"   # Ollama-only options are dropped

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = config.get_llm(num_predict=123)
    assert type(llm).__name__ == "ChatAnthropic" and llm.max_tokens == 123 and llm.model == "claude-haiku-5-5"
    assert config.get_llm().max_tokens == 2000   # Claude needs a cap, so there's a default


def test_model_names_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert config.get_model_info()["model"] == "llama3.1:8b"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-custom")
    assert config.get_model_info()["model"] == "claude-custom"


# --- Ollama models ---

def fake_tags(monkeypatch, models=None, error=None):
    def get(url, timeout):
        if error:
            raise error
        return httpx.Response(200, json={"models": models}, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)


def test_ollama_models_exclude_embedding_models_and_sort_by_size(monkeypatch):
    fake_tags(monkeypatch, [
        {"name": "llama3.1:8b", "size": 4_900_000_000},
        {"name": "nomic-embed-text:latest", "size": 274_000_000},
        {"name": "llama3.2:latest", "size": 2_000_000_000},
    ])
    assert config.list_ollama_models() == [
        {"name": "llama3.2:latest", "size_gb": 2.0}, {"name": "llama3.1:8b", "size_gb": 4.9}]


def test_ollama_unreachable_is_none_not_an_error(monkeypatch):
    fake_tags(monkeypatch, error=httpx.ConnectError("refused"))
    assert config.list_ollama_models() is None


def test_choosing_an_ollama_model(monkeypatch, env_file):
    fake_tags(monkeypatch, [{"name": "llama3.1:8b", "size": 4_900_000_000},
                            {"name": "nomic-embed-text:latest", "size": 274_000_000}])
    config.set_ollama_model("llama3.1:8b")
    assert config.get_model_info()["model"] == "llama3.1:8b"
    assert env_lines(env_file) == ["OLLAMA_MODEL=llama3.1:8b"]
    with pytest.raises(ValueError, match="isn't installed"):
        config.set_ollama_model("nope:1b")
    with pytest.raises(ValueError, match="isn't installed"):
        config.set_ollama_model("nomic-embed-text:latest")   # an embedding model can't chat


def test_choosing_an_ollama_model_while_ollama_is_down(monkeypatch):
    fake_tags(monkeypatch, error=httpx.ConnectError("refused"))
    with pytest.raises(ValueError, match="can't reach Ollama"):
        config.set_ollama_model("llama3.1:8b")
