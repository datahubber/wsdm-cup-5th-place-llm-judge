"""Judge prompt with head+tail token truncation.

The fine-tuned Qwen2.5-14B judge was trained on one fixed input format, and inference has to
rebuild it exactly: a system rubric, the query and both responses in tagged sections, and a
closing question whose answer is a single token, "A" or "B".

When the input is longer than `max_length` tokens, each section keeps its head and its tail and
drops the middle, with an ellipsis in between. The free budget is split 15 % + 5 % for the query and
30 % + 10 % for each response. The reasoning, from my 2025 write-up: readers look at the start and
the end of a long response first.

Checked: the rubric, the section tags and the budgets are identical to the judge's training-data
code (few-shot option off, as in the final training config), and on 3,000 random inputs with a
character-level tokenizer `build_prompt` returns exactly what the inference notebook's prompt
function returned.

Licence note: the rubric text (`SYSTEM_PROMPT`) is reproduced from jagatkiran's public Kaggle
notebook "vLLM_Qwen14B<>Zero_shot- logits-processor-zoo" and stays under that notebook's licence
(Apache 2.0, the default for public Kaggle notebooks).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

# Byte-identical to the rubric in the judge's training-data code. Do not edit.
SYSTEM_PROMPT = """You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query. Determine which response is more likely to be selected by a user based on the following criteria:
- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

"""  # noqa: E501

QUESTION = "Which response is more likely to be selected by a user? (A or B)\n"
ANSWER_PREFIX = "Answer:\n"
ELLIPSIS = "......"
RULE = "---" * 10


class Tokenizer(Protocol):
    """The two tokenizer calls this module needs. `infer.TransformersTokenizer` adapts one."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


# (system, user) -> chat-formatted text that ends with the assistant turn header.
RenderChat = Callable[[str, str], str]


@dataclass(frozen=True)
class Budget:
    """Fractions of the free token budget kept from the head and the tail of each section."""

    query_head: float = 0.15
    query_tail: float = 0.05
    response_head: float = 0.30
    response_tail: float = 0.10


DEFAULT_BUDGET = Budget()


def head_tail(
    ids: Sequence[int], free: int, head_frac: float, tail_frac: float, sep: Sequence[int]
) -> list[int]:
    """Keep the first `head_frac * free` and last `tail_frac * free` tokens, joined by `sep`.

    A section that already fits in `(head_frac + tail_frac) * free` tokens is returned unchanged.
    """
    if len(ids) <= int(free * (head_frac + tail_frac)):
        return list(ids)
    head, tail = int(free * head_frac), int(free * tail_frac)
    # ids[len - tail:] rather than ids[-tail:], which would return everything when tail == 0.
    return [*ids[:head], *sep, *ids[len(ids) - tail :]]


def sections(query: str, response_a: str, response_b: str) -> list[str]:
    """The six text pieces of the user turn: each body section is followed by its closing tag."""
    return [
        f"**Here is your input to process now-**\nInput:\n\n<Query>\n{query}",
        f"\n</Query>\n{RULE}\n",
        f"\n<Response_A>\n{response_a}",
        f"\n</Response_A>\n{RULE}\n",
        f"\n<Response_B>\n{response_b}",
        "\n</Response_B>\n\n",
    ]


def template_overhead(tok: Tokenizer) -> int:
    """Tokens outside the three sections, counted as the trained format counts them.

    That is the system rubric with its role line, one token each for the turn start and end
    markers, the user role line, and the closing question.
    """
    return (
        len(tok.encode("system\n" + SYSTEM_PROMPT))
        + 2
        + len(tok.encode("user\n"))
        + len(tok.encode(QUESTION))
    )


def build_user_message(
    query: str,
    response_a: str,
    response_b: str,
    tok: Tokenizer,
    max_length: int = 4096,
    budget: Budget = DEFAULT_BUDGET,
) -> str:
    """User turn with the three sections truncated to fit `max_length` tokens."""
    q, q_end, a, a_end, b, b_end = (tok.encode(s) for s in sections(query, response_a, response_b))
    overhead = template_overhead(tok)
    total = overhead + sum(len(x) for x in (q, q_end, a, a_end, b, b_end))
    if total > max_length:
        dots = tok.encode(ELLIPSIS)
        free = max_length - overhead - 3 * len(dots)
        if free <= 5:
            raise ValueError(
                f"max_length={max_length} leaves no room after {overhead} template tokens"
            )
        q = head_tail(q, free, budget.query_head, budget.query_tail, dots)
        a = head_tail(a, free, budget.response_head, budget.response_tail, dots)
        b = head_tail(b, free, budget.response_head, budget.response_tail, dots)
    return tok.decode([*q, *q_end, *a, *a_end, *b, *b_end, *tok.encode(QUESTION)])


def build_prompt(
    query: str,
    response_a: str,
    response_b: str,
    tok: Tokenizer,
    render_chat: RenderChat,
    max_length: int = 4096,
    budget: Budget = DEFAULT_BUDGET,
) -> str:
    """Full judge prompt. The next token the model generates is its verdict, "A" or "B"."""
    user = build_user_message(query, response_a, response_b, tok, max_length, budget)
    return render_chat(SYSTEM_PROMPT, user) + ANSWER_PREFIX
