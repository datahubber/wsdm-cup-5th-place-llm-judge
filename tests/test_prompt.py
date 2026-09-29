from __future__ import annotations

import pytest

from wsdm_judge.prompt import (
    ANSWER_PREFIX,
    ELLIPSIS,
    QUESTION,
    SYSTEM_PROMPT,
    build_prompt,
    build_user_message,
    head_tail,
    sections,
    template_overhead,
)


def test_short_input_is_kept_whole(tok):
    msg = build_user_message("What is 2+2?", "4", "Four.", tok, max_length=4096)
    assert ELLIPSIS not in msg
    assert msg == "".join(sections("What is 2+2?", "4", "Four.")) + QUESTION


def test_sections_are_in_order(tok):
    msg = build_user_message("q", "a", "b", tok)
    positions = [msg.index(s) for s in ("<Query>", "<Response_A>", "<Response_B>", QUESTION)]
    assert positions == sorted(positions)


def test_long_response_keeps_head_and_tail(tok):
    response = "H" * 3000 + "M" * 3000 + "T" * 3000
    msg = build_user_message("q", response, "short", tok, max_length=2000)
    assert ELLIPSIS in msg
    assert "H" in msg and "T" in msg
    assert "M" not in msg  # the middle is what gets dropped
    assert "short" in msg  # a section under its budget is untouched


def test_truncated_input_fits_the_budget(tok):
    closing_tags = sum(len(tok.encode(s)) for s in sections("", "", "")[1::2])
    for max_length in (1500, 2000, 4096):
        msg = build_user_message("x" * 5000, "y" * 9000, "z" * 9000, tok, max_length)
        used = len(tok.encode(msg)) - len(tok.encode(QUESTION)) + template_overhead(tok)
        # Closing tags sit outside the head/tail budget, as in the trained format.
        assert used <= max_length + closing_tags


def test_head_tail_with_zero_tail_keeps_only_the_head():
    kept = head_tail(list(range(10)), free=10, head_frac=0.5, tail_frac=0.0, sep=[-1])
    assert kept == [0, 1, 2, 3, 4, -1]


def test_head_tail_leaves_short_sections_alone():
    assert head_tail([1, 2, 3], free=100, head_frac=0.3, tail_frac=0.1, sep=[-1]) == [1, 2, 3]


def test_too_small_max_length_raises(tok):
    with pytest.raises(ValueError, match="no room"):
        build_user_message("q" * 100, "a" * 100, "b" * 100, tok, max_length=50)


def test_build_prompt_wraps_chat_and_ends_with_answer_prefix(tok, render_chat):
    text = build_prompt("q", "a", "b", tok, render_chat)
    assert text.endswith("<|im_start|>assistant\n" + ANSWER_PREFIX)
    assert text.count(SYSTEM_PROMPT) == 1
