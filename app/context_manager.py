"""
Conversation context for a chat: what the model is shown of the exchange so
far, kept separate from the UI's message list so it can be pruned, compacted,
or narrowed to a topic without touching the visible history.

Storage lives in chat_manager (context.json / config.json per chat); this
module is the logic on top - building the history passed to the generator,
recording finished turns, and the compact / keep-only operations.
"""

import uuid
from datetime import datetime
from typing import List, Tuple

import chat_manager
from llm_prompts import get_context_compactor, get_topic_matcher, parse_topic_match

# A long past answer would crowd the retrieved documents out of a small
# local model's context window, so each past answer is clipped to this many
# characters when it's sent along with a new question. The full answer stays
# in the stored context; only what's sent to the model is clipped.
MAX_HISTORY_ANSWER_CHARS = 1500

# Compaction sends the transcript to the model in batches no larger than
# this, carrying a running summary from one batch to the next, so a long
# context can't overflow the model's window in a single call.
MAX_COMPACT_BATCH_CHARS = 12000
MAX_COMPACT_FIELD_CHARS = 3000


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _now() -> str:
    return datetime.now().isoformat()


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _summary_entry(text: str) -> dict:
    return {"id": _new_id(), "kind": "summary", "text": text, "created_at": _now()}


def _render_entry(entry: dict) -> str:
    if entry["kind"] == "summary":
        return f"Earlier summary: {entry['text']}"
    question = _truncate(entry["question"], MAX_COMPACT_FIELD_CHARS)
    answer = _truncate(entry["answer"], MAX_COMPACT_FIELD_CHARS)
    return f"User: {question}\nAssistant: {answer}"


def build_history(chat_id: str) -> List[Tuple[str, str]]:
    """
    The (question, answer) pairs to show the model alongside a new question.

    Empty when the chat's context is switched off. Otherwise the summary
    entry (if any) always comes first - it stands in for everything older -
    followed by the most recent `max_context_entries` exchanges.
    """
    config = chat_manager.get_config(chat_id)
    if not config["context_enabled"]:
        return []

    entries = chat_manager.get_context(chat_id)
    history = [
        ("summary of the earlier conversation", entry["text"])
        for entry in entries if entry["kind"] == "summary"
    ]
    turns = [entry for entry in entries if entry["kind"] == "turn"]
    for entry in turns[-config["max_context_entries"]:]:
        history.append((entry["question"], _truncate(entry["answer"], MAX_HISTORY_ANSWER_CHARS)))
    return history


def record_turn(chat_id: str, question: str, answer: str) -> None:
    """Append a finished question/answer exchange to the chat's context."""
    entry = {
        "id": _new_id(),
        "kind": "turn",
        "question": question,
        "answer": answer,
        "created_at": _now(),
    }
    chat_manager.update_context(chat_id, lambda entries: entries + [entry])


def delete_entry(chat_id: str, entry_id: str) -> bool:
    """Remove one context entry. Returns False if there was no such entry."""
    found = False

    def updater(entries):
        nonlocal found
        kept = [entry for entry in entries if entry["id"] != entry_id]
        found = len(kept) != len(entries)
        return kept

    chat_manager.update_context(chat_id, updater)
    return found


def clear(chat_id: str) -> None:
    """Remove every context entry."""
    chat_manager.update_context(chat_id, lambda entries: [])


def _fold_into_summary(pieces: List[str], focus: str = "") -> str:
    """Summarize transcript pieces, batching so no single call is too large."""
    compactor = get_context_compactor()
    focus_note = (f"Give extra attention to: {focus}. Still keep the other facts, questions and conclusions, "
                  "just more briefly.\n\n" if focus else "")
    running = ""
    batch: List[str] = []
    size = 0

    def flush():
        nonlocal running, batch, size
        transcript = "\n\n".join(batch)
        if running:
            transcript = f"Summary so far:\n{running}\n\nNext part of the conversation:\n{transcript}"
        running = compactor.invoke({"transcript": transcript, "focus": focus_note}).strip()
        batch, size = [], 0

    for piece in pieces:
        if batch and size + len(piece) > MAX_COMPACT_BATCH_CHARS:
            flush()
        batch.append(piece)
        size += len(piece)
    if batch:
        flush()
    return running


def compact(chat_id: str, focus: str = "") -> dict:
    """
    Replace the whole context with a single summary of it, optionally told
    what to concentrate on.

    The model call happens outside the context lock; the swap afterwards
    only replaces the entries that were summarized, so a turn that finished
    streaming in the meantime isn't lost.

    Returns:
        {"before": entries before, "after": entries after}

    Raises:
        ValueError: if the chat doesn't exist, or the model returned nothing
    """
    entries = chat_manager.get_context(chat_id)
    if not entries:
        return {"before": 0, "after": 0}

    summary_text = _fold_into_summary([_render_entry(entry) for entry in entries], focus.strip())
    if not summary_text:
        raise ValueError("The model returned an empty summary")

    summarized_ids = {entry["id"] for entry in entries}
    summary = _summary_entry(summary_text)
    updated = chat_manager.update_context(
        chat_id,
        lambda current: [summary] + [entry for entry in current if entry["id"] not in summarized_ids],
    )
    return {"before": len(entries), "after": len(updated)}


def _split_blocks(text: str) -> List[str]:
    """
    Split summary text into blocks that stand or fall together: a line plus
    the indented lines beneath it (a bullet and its sub-bullets).
    """
    blocks: List[List[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[0].isspace() and blocks:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return ["\n".join(block) for block in blocks]


NARROWED_PREFIX = "Context narrowed to:"


def _topic_checker(topic: str):
    """
    A function telling whether a piece of context is about `topic`.

    A literal, case-insensitive mention counts outright - the local model
    answered "No" for the topic "john" on text saying "John", so the plain
    check goes first. Only text that doesn't contain the topic's words is
    left to the model's yes/no, which catches paraphrase ("my dog" for
    "pets").
    """
    matcher = get_topic_matcher()
    needle = topic.strip().lower()

    def about_topic(text: str) -> bool:
        if needle and needle in text.lower():
            return True
        return parse_topic_match(matcher.invoke({"topic": topic, "text": text}))

    return about_topic


def keep_only(chat_id: str, topic: str) -> dict:
    """
    Narrow the context to what concerns one topic, dropping the rest.

    Selective, never generative: the model only answers yes/no on whether a
    piece of context is about the topic, and whatever is kept stays word for
    word. An exchange is kept or dropped whole; the summary entry is
    filtered block by block, so a summary that covers several subjects
    keeps just the blocks about this one.

    Returns:
        {"before": entries before, "after": entries after, "kept": how many
        of the original entries had anything on-topic}

    Raises:
        ValueError: if the chat doesn't exist
    """
    entries = chat_manager.get_context(chat_id)
    if not entries:
        return {"before": 0, "after": 0, "kept": 0}

    about_topic = _topic_checker(topic)

    kept_turn_ids = set()
    kept_blocks: List[str] = []
    for entry in entries:
        if entry["kind"] == "summary":
            for block in _split_blocks(entry["text"]):
                if block.startswith(NARROWED_PREFIX):
                    continue  # the previous narrowing's header; replaced below
                if about_topic(block):
                    kept_blocks.append(block)
        elif about_topic(_render_entry(entry)):
            kept_turn_ids.add(entry["id"])

    processed_ids = {entry["id"] for entry in entries}
    new_summary = None
    if kept_blocks:
        new_summary = _summary_entry(f"{NARROWED_PREFIX} {topic}\n" + "\n".join(kept_blocks))

    def updater(current):
        result = [new_summary] if new_summary else []
        for entry in current:
            if entry["id"] not in processed_ids:
                result.append(entry)  # arrived while the model was working
            elif entry["id"] in kept_turn_ids:
                result.append(entry)
        return result

    updated = chat_manager.update_context(chat_id, updater)
    kept = len(kept_turn_ids) + (1 if new_summary else 0)
    return {"before": len(entries), "after": len(updated), "kept": kept}


def forget(chat_id: str, topic: str) -> dict:
    """
    Drop whatever in the context concerns one topic - the opposite of
    keep_only, and selective in the same way: the model only answers yes/no
    on whether a piece of context is about the topic, and everything not
    dropped stays word for word. An exchange goes whole; the summary entry
    loses just the blocks about the topic.

    Returns:
        {"before": entries before, "after": entries after, "removed": how
        many of the original entries had anything about the topic}

    Raises:
        ValueError: if the chat doesn't exist
    """
    entries = chat_manager.get_context(chat_id)
    if not entries:
        return {"before": 0, "after": 0, "removed": 0}

    about_topic = _topic_checker(topic)

    dropped_turn_ids = set()
    remaining_blocks: List[str] = []
    summary_changed = False
    for entry in entries:
        if entry["kind"] == "summary":
            for block in _split_blocks(entry["text"]):
                if about_topic(block):
                    summary_changed = True
                else:
                    remaining_blocks.append(block)
        elif about_topic(_render_entry(entry)):
            dropped_turn_ids.add(entry["id"])

    processed_ids = {entry["id"] for entry in entries}
    new_summary = _summary_entry("\n".join(remaining_blocks)) if remaining_blocks else None

    def updater(current):
        result = [new_summary] if new_summary else []
        for entry in current:
            if entry["id"] not in processed_ids:
                result.append(entry)  # arrived while the model was working
            elif entry["kind"] == "turn" and entry["id"] not in dropped_turn_ids:
                result.append(entry)
        return result

    updated = chat_manager.update_context(chat_id, updater)
    removed = len(dropped_turn_ids) + (1 if summary_changed else 0)
    return {"before": len(entries), "after": len(updated), "removed": removed}
