import chat_manager
import context_manager
import pytest


class FakeChain:
    def __init__(self, func):
        self.func = func
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return self.func(inputs)


def fake_matcher(monkeypatch, about=()):
    """A topic matcher that says 'yes' when any word in `about` is in the text (case-insensitively)."""
    chain = FakeChain(lambda i: "Yes." if any(w in i["text"].lower() for w in about) else "No.")
    monkeypatch.setattr(context_manager, "get_topic_matcher", lambda: chain)
    return chain


def fake_compactor(monkeypatch, reply="SUMMARY", during=None):
    def respond(inputs):
        if during:
            during()
        return reply
    chain = FakeChain(respond)
    monkeypatch.setattr(context_manager, "get_context_compactor", lambda: chain)
    return chain


def questions(chat_id):
    return [e.get("question") or "SUMMARY:" + e["text"] for e in chat_manager.get_context(chat_id)]


@pytest.fixture
def john_chat(chat_id):
    context_manager.record_turn(chat_id, "my name is actually John", "Got it, John.")
    context_manager.record_turn(chat_id, "what is the capital of France?", "Paris.")
    context_manager.record_turn(chat_id, "my dog is called Rex", "Nice name.")
    return chat_id


# --- what the model gets to see ---

def test_history_is_the_recent_exchanges_with_the_summary_first(chat_id):
    chat_manager.save_config(chat_id, {**chat_manager.get_config(chat_id), "max_context_entries": 2})
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry("earlier stuff")])
    for i in range(4):
        context_manager.record_turn(chat_id, f"q{i}", f"a{i}")
    history = context_manager.build_history(chat_id)
    assert history == [("summary of the earlier conversation", "earlier stuff"), ("q2", "a2"), ("q3", "a3")]


def test_history_is_empty_when_context_is_switched_off(chat_id):
    context_manager.record_turn(chat_id, "q", "a")
    chat_manager.save_config(chat_id, {**chat_manager.get_config(chat_id), "context_enabled": False})
    assert context_manager.build_history(chat_id) == []


def test_long_answers_are_trimmed_in_history_but_kept_in_storage(chat_id):
    context_manager.record_turn(chat_id, "q", "x" * 5000)
    (_, answer), = context_manager.build_history(chat_id)
    assert len(answer) <= context_manager.MAX_HISTORY_ANSWER_CHARS + 1
    assert len(chat_manager.get_context(chat_id)[0]["answer"]) == 5000


def test_delete_and_clear(chat_id):
    context_manager.record_turn(chat_id, "q1", "a1")
    context_manager.record_turn(chat_id, "q2", "a2")
    first = chat_manager.get_context(chat_id)[0]["id"]
    assert context_manager.delete_entry(chat_id, first) is True
    assert context_manager.delete_entry(chat_id, first) is False
    assert questions(chat_id) == ["q2"]
    context_manager.clear(chat_id)
    assert chat_manager.get_context(chat_id) == []


# --- compact ---

def test_compact_replaces_everything_with_one_summary(john_chat, monkeypatch):
    fake_compactor(monkeypatch, "SUMMARY")
    assert context_manager.compact(john_chat) == {"before": 3, "after": 1}
    entries = chat_manager.get_context(john_chat)
    assert [(e["kind"], e["text"]) for e in entries] == [("summary", "SUMMARY")]


def test_compact_passes_the_focus_to_the_model(john_chat, monkeypatch):
    chain = fake_compactor(monkeypatch)
    context_manager.compact(john_chat, focus="the dog")
    assert "the dog" in chain.calls[0]["focus"] and "my dog is called Rex" in chain.calls[0]["transcript"]
    chain = fake_compactor(monkeypatch)
    context_manager.compact(john_chat)
    assert chain.calls[0]["focus"] == ""


def test_compact_of_nothing_is_a_no_op(chat_id, monkeypatch):
    chain = fake_compactor(monkeypatch)
    assert context_manager.compact(chat_id) == {"before": 0, "after": 0}
    assert chain.calls == []


def test_compact_keeps_a_turn_that_finishes_while_the_model_is_working(john_chat, monkeypatch):
    fake_compactor(monkeypatch, during=lambda: context_manager.record_turn(john_chat, "late question", "late answer"))
    context_manager.compact(john_chat)
    assert questions(john_chat) == ["SUMMARY:SUMMARY", "late question"]


def test_compact_with_an_empty_model_reply_changes_nothing(john_chat, monkeypatch):
    fake_compactor(monkeypatch, reply="  ")
    with pytest.raises(ValueError):
        context_manager.compact(john_chat)
    assert len(chat_manager.get_context(john_chat)) == 3


# --- forget ---

def test_forget_drops_what_mentions_the_topic_and_keeps_the_rest(john_chat, monkeypatch):
    fake_matcher(monkeypatch)
    assert context_manager.forget(john_chat, "John") == {"before": 3, "after": 2, "removed": 1}
    assert questions(john_chat) == ["what is the capital of France?", "my dog is called Rex"]


def test_forget_matches_the_topic_literally_whatever_its_case_without_asking_the_model(john_chat, monkeypatch):
    chain = fake_matcher(monkeypatch)   # would answer 'No' to everything
    result = context_manager.forget(john_chat, "john")   # the topic is lowercase, the text says "John"
    assert result["removed"] == 1
    assert all(call["text"].lower().find("john") == -1 for call in chain.calls)


def test_forget_asks_the_model_about_paraphrases(john_chat, monkeypatch):
    fake_matcher(monkeypatch, about=("rex",))
    assert context_manager.forget(john_chat, "pets")["removed"] == 1
    assert "my dog is called Rex" not in " ".join(questions(john_chat))


def test_forget_nothing_matching_removes_nothing(john_chat, monkeypatch):
    fake_matcher(monkeypatch)
    assert context_manager.forget(john_chat, "quantum physics") == {"before": 3, "after": 3, "removed": 0}


def test_forget_filters_a_summary_block_by_block(chat_id, monkeypatch):
    summary = "- the user is called John\n  - likes tea\n- the capital of France is Paris"
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry(summary)])
    fake_matcher(monkeypatch)
    assert context_manager.forget(chat_id, "john")["removed"] == 1
    (entry,) = chat_manager.get_context(chat_id)
    assert entry["text"] == "- the capital of France is Paris"


def test_forget_removing_the_whole_summary_leaves_no_summary(chat_id, monkeypatch):
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry("- John likes tea")])
    fake_matcher(monkeypatch)
    context_manager.forget(chat_id, "john")
    assert chat_manager.get_context(chat_id) == []


def test_forget_keeps_a_turn_that_finishes_while_the_model_is_working(john_chat, monkeypatch):
    def matcher(inputs):
        context_manager.record_turn(john_chat, "late John question", "late answer")   # mentions the topic, but arrived late
        return "No."
    monkeypatch.setattr(context_manager, "get_topic_matcher", lambda: FakeChain(matcher))
    context_manager.forget(john_chat, "pets")
    assert "late John question" in questions(john_chat)


# --- keep only ---

def test_keep_only_keeps_what_mentions_the_topic(john_chat, monkeypatch):
    fake_matcher(monkeypatch)
    result = context_manager.keep_only(john_chat, "france")
    assert result == {"before": 3, "after": 1, "kept": 1}
    assert questions(john_chat) == ["what is the capital of France?"]


def test_keep_only_writes_a_narrowed_header_over_the_kept_summary_blocks(chat_id, monkeypatch):
    summary = "- the user is called John\n- the capital of France is Paris"
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry(summary)])
    fake_matcher(monkeypatch)
    context_manager.keep_only(chat_id, "france")
    (entry,) = chat_manager.get_context(chat_id)
    assert entry["text"] == f"{context_manager.NARROWED_PREFIX} france\n- the capital of France is Paris"


def test_keep_only_with_nothing_on_topic_empties_the_context(john_chat, monkeypatch):
    fake_matcher(monkeypatch)
    assert context_manager.keep_only(john_chat, "quantum physics") == {"before": 3, "after": 0, "kept": 0}


# --- undo (regenerate / edit) ---

def test_undo_forgets_the_latest_exchange_when_it_matches(chat_id):
    context_manager.record_turn(chat_id, "q1", "a1")
    context_manager.record_turn(chat_id, "q2", "a2")
    assert context_manager.undo_turn(chat_id, "q2") is True
    assert questions(chat_id) == ["q1"]


def test_undo_leaves_the_context_alone_when_the_latest_exchange_is_about_something_else(chat_id):
    context_manager.record_turn(chat_id, "q1", "a1")
    context_manager.record_turn(chat_id, "q2", "a2")
    assert context_manager.undo_turn(chat_id, "q1") is False     # q1 isn't the latest
    assert questions(chat_id) == ["q1", "q2"]


def test_undo_of_a_question_that_was_never_remembered_does_nothing(chat_id):
    assert context_manager.undo_turn(chat_id, "never answered") is False
    context_manager.record_turn(chat_id, "q1", "a1")
    assert context_manager.undo_turn(chat_id, "cancelled question") is False
    assert questions(chat_id) == ["q1"]


def test_undo_never_touches_the_summary(chat_id):
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry("earlier")])
    assert context_manager.undo_turn(chat_id, "anything") is False
    assert questions(chat_id) == ["SUMMARY:earlier"]


# --- token budget window ---

def configure(chat_id, **changes):
    chat_manager.save_config(chat_id, {**chat_manager.get_config(chat_id), **changes})


def add_turns(chat_id, count, size=400):
    for i in range(count):
        context_manager.record_turn(chat_id, f"q{i}", f"a{i} " + "x" * size)


def test_token_estimates_are_about_four_characters_each():
    assert context_manager.estimate_tokens("") == 0
    assert context_manager.estimate_tokens("abcd") == 1
    assert context_manager.estimate_tokens("abcde") == 2
    assert context_manager.estimate_tokens("x" * 4000) == 1000


def test_the_window_stops_at_the_token_budget_before_the_exchange_limit(chat_id):
    configure(chat_id, max_context_entries=20, max_context_tokens=520)      # each exchange is ~102 tokens
    add_turns(chat_id, 10)
    sent = [q for q, _ in context_manager.build_history(chat_id)]
    assert sent == ["q5", "q6", "q7", "q8", "q9"]
    info = context_manager.usage(chat_id)
    assert info["sent_exchanges"] == 5 and info["total_exchanges"] == 10 and info["left_out"] == 5
    assert info["used_tokens"] <= 520 and info["percent"] >= 90


def test_the_window_stops_at_the_exchange_limit_before_the_token_budget(chat_id):
    configure(chat_id, max_context_entries=2, max_context_tokens=32000)
    add_turns(chat_id, 6)
    assert [q for q, _ in context_manager.build_history(chat_id)] == ["q4", "q5"]


def test_the_latest_exchange_is_always_sent_even_if_it_alone_exceeds_the_budget(chat_id):
    configure(chat_id, max_context_tokens=500)
    context_manager.record_turn(chat_id, "short", "a")
    context_manager.record_turn(chat_id, "huge", "y" * 100_000)          # only 1500 chars of it are ever sent
    history = context_manager.build_history(chat_id)
    assert [q for q, _ in history][-1] == "huge"
    context_manager.record_turn(chat_id, "x" * 3000, "z" * 1500)         # ~1100 tokens: over budget by itself
    assert [q[:1] for q, _ in context_manager.build_history(chat_id)] == ["x"]


def test_the_summary_counts_against_the_budget_and_always_leads(chat_id):
    configure(chat_id, max_context_entries=20, max_context_tokens=520)
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry("s" * 800)])    # 200 tokens
    add_turns(chat_id, 6)
    history = context_manager.build_history(chat_id)
    assert history[0][0] == "summary of the earlier conversation"
    assert len(history) == 1 + 3                                          # 320 tokens left for ~102-token turns
    info = context_manager.usage(chat_id)
    assert info["sent_ids"][0] == chat_manager.get_context(chat_id)[0]["id"] and len(info["sent_ids"]) == 4


def test_usage_when_context_is_switched_off(chat_id):
    add_turns(chat_id, 2)
    configure(chat_id, context_enabled=False)
    info = context_manager.usage(chat_id)
    assert info["enabled"] is False and info["used_tokens"] == 0 and info["sent_ids"] == [] and info["percent"] == 0


def test_usage_of_an_empty_context(chat_id):
    info = context_manager.usage(chat_id)
    assert (info["used_tokens"], info["total_exchanges"], info["left_out"], info["sent_ids"]) == (0, 0, 0, [])


# --- folding what falls out of the window ---

def test_overflow_is_folded_into_the_summary_and_the_window_is_untouched(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=2)
    add_turns(chat_id, 5, size=10)
    chain = fake_compactor(monkeypatch, "FOLDED")
    before = context_manager.build_history(chat_id)
    assert context_manager.compact_overflow(chat_id) == {"folded": 3}
    assert "q0" in chain.calls[0]["transcript"] and "q2" in chain.calls[0]["transcript"] and "q3" not in chain.calls[0]["transcript"]
    assert questions(chat_id) == ["SUMMARY:FOLDED", "q3", "q4"]
    assert context_manager.build_history(chat_id)[1:] == before                 # same recent exchanges, verbatim


def test_folding_includes_the_existing_summary(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=1)
    chat_manager.update_context(chat_id, lambda e: [context_manager._summary_entry("old summary")])
    add_turns(chat_id, 3, size=10)
    chain = fake_compactor(monkeypatch, "NEW SUMMARY")
    context_manager.compact_overflow(chat_id)
    assert "old summary" in chain.calls[0]["transcript"]
    assert questions(chat_id) == ["SUMMARY:NEW SUMMARY", "q2"]


def test_folding_nothing_does_not_call_the_model(chat_id, monkeypatch):
    add_turns(chat_id, 2, size=10)
    chain = fake_compactor(monkeypatch)
    assert context_manager.compact_overflow(chat_id) == {"folded": 0}
    assert chain.calls == []


def test_a_turn_that_finishes_during_folding_is_kept(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=1)
    add_turns(chat_id, 3, size=10)
    fake_compactor(monkeypatch, "S", during=lambda: context_manager.record_turn(chat_id, "late", "late answer"))
    context_manager.compact_overflow(chat_id)
    assert questions(chat_id) == ["SUMMARY:S", "q2", "late"]


def test_auto_compact_waits_until_enough_has_piled_up(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=2, auto_compact=True)
    chain = fake_compactor(monkeypatch, "S")
    add_turns(chat_id, 4, size=10)               # 2 left out: below the threshold
    assert context_manager.maybe_auto_compact(chat_id) is False and chain.calls == []
    add_turns(chat_id, 1, size=10)               # 3 left out
    assert context_manager.maybe_auto_compact(chat_id) is True and len(chain.calls) == 1
    assert context_manager.usage(chat_id)["left_out"] == 0


def test_auto_compact_does_nothing_unless_switched_on_or_context_is_off(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=1)
    chain = fake_compactor(monkeypatch)
    add_turns(chat_id, 6, size=10)
    assert context_manager.maybe_auto_compact(chat_id) is False
    configure(chat_id, auto_compact=True, context_enabled=False)
    assert context_manager.maybe_auto_compact(chat_id) is False
    assert chain.calls == []


def test_auto_compact_never_raises_and_leaves_the_context_alone_on_failure(chat_id, monkeypatch):
    configure(chat_id, max_context_entries=1, auto_compact=True)
    add_turns(chat_id, 6, size=10)
    monkeypatch.setattr(context_manager, "get_context_compactor", lambda: FakeChain(lambda i: 1 / 0))
    assert context_manager.maybe_auto_compact(chat_id) is False
    assert len(chat_manager.get_context(chat_id)) == 6
