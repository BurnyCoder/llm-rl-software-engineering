"""Global context: construct the sole model-visible MiniBug repair prompt.

Sources:
- https://huggingface.co/docs/transformers/main/en/chat_templating
- https://huggingface.co/docs/trl/main/en/dataset_formats
"""

from __future__ import annotations

import json
from typing import Any

from minibug_rl.schemas import RepairTask, TestCase

# A short stable system message keeps output parsing deterministic on a tiny model.
_SYSTEM = (
    "You repair one pure Python function. Return exactly one complete replacement "
    "function and nothing else. Do not import modules or use unsafe builtins."
)


def _public_case(case: TestCase) -> dict[str, Any]:
    """Render one model-visible example with its expected result."""
    # JSON avoids executable prompt fragments and represents arguments unambiguously.
    return {"args": list(case.args), "kwargs": case.kwargs, "expected": case.expected}


def build_messages(task: RepairTask) -> list[dict[str, str]]:
    """Build a conversational repair prompt containing no hidden reward evidence."""
    # Public examples demonstrate semantics; hidden cases never enter this function.
    examples = json.dumps(
        [_public_case(case) for case in task.public_tests],
        indent=2,
        ensure_ascii=False,
    )
    # Delimit source and examples explicitly for reliable small-model parsing.
    user = (
        f"Specification:\n{task.specification}\n\n"
        f"Buggy function:\n```python\n{task.buggy_code}\n```\n\n"
        f"Public examples as JSON calls:\n{examples}\n\n"
        f"Write the corrected `{task.function_name}` function now."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the selected model's native chat template for local generation."""
    # `tokenize=False` returns the exact prompt text that is also written to run logs.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
