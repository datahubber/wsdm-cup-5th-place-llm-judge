"""Shared fixtures: a character-level tokenizer and a Qwen-like chat template (no downloads)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))  # lets tests import the scripts as modules


class CharTokenizer:
    """One token per character. Exactly invertible, so truncation is easy to check."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)


def qwen_like_chat(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


class ChatCharTokenizer(CharTokenizer):
    """CharTokenizer plus `render_chat`, the interface `infer.TransformersTokenizer` offers."""

    def render_chat(self, system: str, user: str) -> str:
        return qwen_like_chat(system, user)


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def chat_tok() -> ChatCharTokenizer:
    return ChatCharTokenizer()


@pytest.fixture
def render_chat() -> Callable[[str, str], str]:
    return qwen_like_chat


@pytest.fixture
def pairs_path() -> Path:
    return FIXTURES / "pairs.jsonl"


@pytest.fixture
def logprobs_path() -> Path:
    return FIXTURES / "logprobs.csv"


@pytest.fixture
def repo_root() -> Path:
    return ROOT
