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


def estimate_tokens(text: str) -> int:
    """A rough token count (about four characters each) - close enough to size a window."""
    return (len(text) + 3) // 4


def _turn_cost(entry: dict) -> int:
    return estimate_tokens(entry["question"]) + estimate_tokens(_truncate(entry["answer"], MAX_HISTORY_ANSWER_CHARS))


def _window(entries: List[dict], config: dict):
    """
    Split the context into what gets sent and what doesn't.

    The summary always goes first. After it come the most recent exchanges,
    newest first, until either the exchange limit or the token budget is
    reached - the latest exchange always goes, whatever its size.

    Returns:
        (summary entry or None, sent turns oldest first, turns left out, tokens sent)
    """
    summary = next((entry for entry in entries if entry["kind"] == "summary"), None)
    turns = [entry for entry in entries if entry["kind"] == "turn"]
    used = estimate_tokens(summary["text"]) if summary else 0
    sent: List[dict] = []
    for entry in reversed(turns):
        cost = _turn_cost(entry)
        if len(sent) >= config["max_context_entries"]:
            break
        if sent and used + cost > config["max_context_tokens"]:
            break
        sent.append(entry)
        used += cost
    sent.reverse()
    return summary, sent, turns[:len(turns) - len(sent)], used


def build_history(chat_id: str) -> List[Tuple[str, str]]:
    """
    The (question, answer) pairs to show the model alongside a new question.

    Empty when the chat's context is switched off. Otherwise the summary
    entry (if any) comes first - it stands in for everything older - then the
    recent exchanges that fit the window (see _window).
    """
    config = chat_manager.get_config(chat_id)
    if not config["context_enabled"]:
        return []

    summary, sent, _, _ = _window(chat_manager.get_context(chat_id), config)
    history = []
    if summary:
        history.append(("summary of the earlier conversation", summary["text"]))
    for entry in sent:
        history.append((entry["question"], _truncate(entry["answer"], MAX_HISTORY_ANSWER_CHARS)))
    return history


def usage(chat_id: str) -> dict:
    """
    How full the context window is: what is sent to the model per message,
    against the budget, and how much older context is being left out.
    """
    config = chat_manager.get_config(chat_id)
    entries = chat_manager.get_context(chat_id)
    summary, sent, dropped, used = _window(entries, config)
    if not config["context_enabled"]:
        used, sent = 0, []
    budget = config["max_context_tokens"]
    return {
        "enabled": config["context_enabled"],
        "used_tokens": used,
        "budget_tokens": budget,
        "percent": round(100 * used / budget) if budget else 0,
        "sent_ids": ([summary["id"]] if summary and config["context_enabled"] else []) + [e["id"] for e in sent],
        "sent_exchanges": len(sent),
        "total_exchanges": len(sent) + len(dropped),
        "left_out": len(dropped),
    }


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


def undo_turn(chat_id: str, question: str) -> bool:
    """
    Forget the most recent exchange, if it's the one for `question`.

    Used when the user regenerates or edits their last message, so the old
    answer isn't remembered alongside the new one. A question that never
    reached the context (cancelled, or failed) has nothing to undo.

    Returns:
        True if an exchange was removed
    """
    removed = False

    def updater(entries):
        nonlocal removed
        for i in range(len(entries) - 1, -1, -1):
            if entries[i]["kind"] == "turn":
                if entries[i]["question"] == question:
                    removed = True
                    return entries[:i] + entries[i + 1:]
                break   # the latest turn is about something else: leave it
        return entries

    chat_manager.update_context(chat_id, updater)
    return removed


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


# Folding a single stray exchange into the summary on every message would cost
# a model call each time; wait until a few have piled up.
AUTO_COMPACT_MIN_LEFT_OUT = 3


def compact_overflow(chat_id: str) -> dict:
    """
    Fold the exchanges that no longer fit the window into the summary, and
    drop them. Everything still inside the window stays word for word, so
    nothing the model can currently see changes.

    Returns:
        {"folded": how many exchanges went into the summary}
    """
    entries = chat_manager.get_context(chat_id)
    summary, _, dropped, _ = _window(entries, chat_manager.get_config(chat_id))
    if not dropped:
        return {"folded": 0}

    pieces = ([f"Earlier summary: {summary['text']}"] if summary else []) + [_render_entry(e) for e in dropped]
    summary_text = _fold_into_summary(pieces)
    if not summary_text:
        raise ValueError("The model returned an empty summary")

    folded_ids = {entry["id"] for entry in dropped} | ({summary["id"]} if summary else set())
    new_summary = _summary_entry(summary_text)
    chat_manager.update_context(
        chat_id,
        lambda current: [new_summary] + [entry for entry in current if entry["id"] not in folded_ids],
    )
    return {"folded": len(dropped)}


def maybe_auto_compact(chat_id: str) -> bool:
    """
    Run compact_overflow if the chat has auto-compact on and enough exchanges
    have fallen out of the window. Never raises: a failed summary just means
    the older exchanges stay where they are for next time.

    Returns:
        True if the context was compacted
    """
    try:
        config = chat_manager.get_config(chat_id)
        if not config["auto_compact"] or not config["context_enabled"]:
            return False
        if usage(chat_id)["left_out"] < AUTO_COMPACT_MIN_LEFT_OUT:
            return False
        return compact_overflow(chat_id)["folded"] > 0
    except Exception as e:
        print(f"WARNING: auto-compact failed: {e}")
        return False


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
