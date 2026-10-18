import json
import os
import shutil
import threading
import uuid
from datetime import datetime
from typing import List, Optional

from document_processor import list_documents

CHATS_ROOT = "chats"


def _chat_dir(chat_id: str) -> str:
    return os.path.join(CHATS_ROOT, chat_id)


def _meta_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "meta.json")


def _messages_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "messages.json")


def _context_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "context.json")


def _config_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "config.json")


def get_workspace_path(chat_id: str) -> str:
    """Absolute path of the chat's scratch folder for agent-run code (created on demand)."""
    path = os.path.abspath(os.path.join(_chat_dir(chat_id), "workspace"))
    os.makedirs(path, exist_ok=True)
    return path


def chat_exists(chat_id: str) -> bool:
    return os.path.exists(_meta_path(chat_id))


def get_messages(chat_id: str) -> List[dict]:
    """
    Load a chat's persisted conversation history.

    Args:
        chat_id: The chat's id

    Returns:
        List of message dicts, oldest first. Empty if none saved yet.

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    path = _messages_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_messages(chat_id: str, messages: List[dict]) -> None:
    """
    Persist a chat's full conversation history, overwriting whatever was
    saved before. The frontend calls this with its current in-memory
    message list at each settled point (a message finishing, not mid-stream)
    rather than one message at a time, so this is a full replace, not an
    append.

    Args:
        chat_id: The chat's id
        messages: The complete message list to save

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    with open(_messages_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(messages, f)


def get_chat_paths(chat_id: str):
    """
    Resolve a chat's chroma_db/files paths.

    Args:
        chat_id: The chat's id

    Returns:
        (chroma_db_path, files_path) tuple

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    chat_dir = _chat_dir(chat_id)
    return os.path.join(chat_dir, "chroma_db"), os.path.join(chat_dir, "files")


def list_chats() -> List[dict]:
    """
    List all chats with their document counts, oldest first.

    Returns:
        List of {id, name, created_at, document_count} dicts
    """
    if not os.path.exists(CHATS_ROOT):
        return []

    chats = []
    for entry in os.listdir(CHATS_ROOT):
        meta_path = _meta_path(entry)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        chroma_db_path, files_path = get_chat_paths(entry)
        meta["document_count"] = len(list_documents(chroma_db_path, files_path))
        chats.append(meta)

    chats.sort(key=lambda c: c["created_at"])
    return chats


def create_chat(name: str, copy_from: Optional[str] = None, in_progress_upload_chat_ids: Optional[set] = None) -> dict:
    """
    Create a new chat, either empty or as a copy of another chat's documents.

    Copying is a plain filesystem copy of the source chat's chroma_db/ and
    files/ directories, not a re-embed through the Chroma API - so cloning
    even a large collection is near-instant and re-uses the already-computed
    summaries and embeddings at no cost.

    Args:
        name: Display name for the new chat
        copy_from: Optional id of an existing chat to clone documents from
        in_progress_upload_chat_ids: chat ids that currently have an upload in
            progress (from backend/main.py's upload_progress) - copying a
            chat mid-write risks cloning a half-written collection

    Returns:
        The new chat's metadata dict (matches list_chats()'s shape)

    Raises:
        ValueError: if copy_from is set but invalid, or has an upload in progress
    """
    if copy_from is not None:
        if not chat_exists(copy_from):
            raise ValueError(f"Chat not found: {copy_from}")
        if in_progress_upload_chat_ids and copy_from in in_progress_upload_chat_ids:
            raise ValueError("Source chat has an upload in progress - try again once it finishes")

    chat_id = str(uuid.uuid4())
    chat_dir = _chat_dir(chat_id)
    os.makedirs(chat_dir, exist_ok=True)

    if copy_from is not None:
        src_chroma_db, src_files = get_chat_paths(copy_from)
        if os.path.exists(src_chroma_db):
            shutil.copytree(src_chroma_db, os.path.join(chat_dir, "chroma_db"))
        if os.path.exists(src_files):
            shutil.copytree(src_files, os.path.join(chat_dir, "files"))

    os.makedirs(os.path.join(chat_dir, "files"), exist_ok=True)

    meta = {
        "id": chat_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
    }
    with open(_meta_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    chroma_db_path, files_path = get_chat_paths(chat_id)
    meta["document_count"] = len(list_documents(chroma_db_path, files_path))
    return meta


def delete_chat(chat_id: str) -> bool:
    """
    Delete a chat entirely - its documents, files, and metadata.

    Args:
        chat_id: The chat's id

    Returns:
        True if the chat was found and deleted, False if it didn't exist
    """
    if not chat_exists(chat_id):
        return False
    shutil.rmtree(_chat_dir(chat_id))
    return True


# --- Conversation context & per-chat configuration ---
#
# The context is what the model is shown of the conversation so far, kept
# separate from messages.json (what the UI displays) so the user can prune,
# compact, or narrow it without touching the visible chat history.
#
# Each entry is one of:
#   {"id", "kind": "turn",    "question", "answer", "created_at"}
#   {"id", "kind": "summary", "text",               "created_at"}
# Compacting or narrowing the context folds entries into a single summary
# entry, so there is at most one, and it's always first.

DEFAULT_CONFIG = {
    "context_enabled": True,
    "max_context_entries": 4,
    # Agent: lets the model decide which tools to use per message. When off
    # (or when the model can't call tools) the classic retrieve -> generate
    # pipeline answers instead.
    "agent_enabled": True,
    # safe: the agent can't run code. ask: each run needs the user's OK.
    # bypass: code runs without asking.
    "permission_mode": "ask",
    "max_agent_steps": 8,
    "web_search_enabled": True,
}

PERMISSION_MODES = ("safe", "ask", "bypass")

# Serializes read-modify-write cycles on context.json: a streaming response
# appending its turn can land at the same moment as a delete or compact.
_context_lock = threading.Lock()


def _write_json_atomic(path: str, data) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def get_context(chat_id: str) -> List[dict]:
    """
    Load a chat's context entries, oldest first.

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    path = _context_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_context(chat_id: str, updater) -> List[dict]:
    """
    Atomically apply updater(entries) -> new_entries to a chat's context.

    Args:
        chat_id: The chat's id
        updater: Callable taking the current entry list and returning the
            list to save. Runs under a lock, so keep it quick - do slow
            work (like LLM calls) before calling this, and have the updater
            only splice the result into whatever is current.

    Returns:
        The saved entry list

    Raises:
        ValueError: if the chat doesn't exist
    """
    with _context_lock:
        entries = get_context(chat_id)
        new_entries = updater(entries)
        _write_json_atomic(_context_path(chat_id), new_entries)
        return new_entries


def get_config(chat_id: str) -> dict:
    """
    Load a chat's configuration, with defaults filled in for anything unset.

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    config = dict(DEFAULT_CONFIG)
    path = _config_path(chat_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def save_config(chat_id: str, config: dict) -> dict:
    """
    Persist a chat's configuration. Only known keys are kept.

    Raises:
        ValueError: if the chat doesn't exist
    """
    if not chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    merged = {key: config.get(key, default) for key, default in DEFAULT_CONFIG.items()}
    _write_json_atomic(_config_path(chat_id), merged)
    return merged
