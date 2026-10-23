"""
Shared fixtures. Every test runs against a temporary chats folder and a
temporary working directory with an empty .env, with every provider key
removed from the environment - so no test can touch real chats or the real
.env, or call a real model.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))   # `import main` is backend/main.py ...
sys.path.append(os.path.join(ROOT, "app"))          # ... not app/main.py


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "OLLAMA_MODEL",
                 "OPENAI_MODEL", "ANTHROPIC_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    work = tmp_path / "cwd"
    work.mkdir()
    (work / ".env").touch()
    monkeypatch.chdir(work)

    import chat_manager
    monkeypatch.setattr(chat_manager, "CHATS_ROOT", str(tmp_path / "chats"))
    os.makedirs(chat_manager.CHATS_ROOT, exist_ok=True)
    # Listing documents opens a Chroma index; none of these tests need one
    monkeypatch.setattr(chat_manager, "list_documents", lambda *args, **kwargs: [])
    yield


@pytest.fixture
def chat_id():
    import chat_manager
    return chat_manager.create_chat("test chat")["id"]


@pytest.fixture
def env_file(tmp_path):
    return tmp_path / "cwd" / ".env"
