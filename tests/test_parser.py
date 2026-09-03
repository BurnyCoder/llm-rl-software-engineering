"""Global context: test the narrow replacement-function completion contract.

Sources:
- https://docs.python.org/3/library/ast.html
- https://docs.python.org/3/library/functions.html#exec
"""

from minibug_rl.parser import parse_candidate


def test_parser_accepts_one_matching_function_in_optional_fence() -> None:
    """Accept the concise format the prompt requests and normalize its Markdown fence."""
    # A fenced function is common even when models are asked for code only.
    result = parse_candidate("```python\ndef clamp(x):\n    return max(0, x)\n````"[:-1], "clamp")

    # The sandbox receives normalized Python, never fence markers.
    assert result.valid is True
    assert result.code == "def clamp(x):\n    return max(0, x)"
    assert result.error is None


def test_parser_rejects_explanation_or_multiple_top_level_statements() -> None:
    """Prevent ambiguous prose or helper statements from reaching the executor."""
    # The response contract deliberately permits one top-level function only.
    result = parse_candidate("Here is the fix:\n\ndef clamp(x):\n    return x", "clamp")

    assert result.valid is False
    assert "Python" in (result.error or "") or "top-level" in (result.error or "")


def test_parser_rejects_wrong_name_and_imports() -> None:
    """Match the requested API and reject imports as a defense-in-depth policy."""
    # Function identity is part of every task and must remain stable for its tests.
    wrong_name = parse_candidate("def other(x):\n    return x", "clamp")
    # Imports are unnecessary for the deliberately tiny curriculum.
    imported = parse_candidate("def clamp(x):\n    import os\n    return x", "clamp")

    assert wrong_name.valid is False
    assert "clamp" in (wrong_name.error or "")
    assert imported.valid is False
    assert "Import" in (imported.error or "")


def test_parser_normalizes_conversational_trl_completion() -> None:
    """TRL may return assistant-message lists for conversational prompts."""
    # Match TRL's conversational completion representation documented by GRPOTrainer.
    completion = [{"role": "assistant", "content": "def clamp(x):\n    return x"}]

    result = parse_candidate(completion, "clamp")

    assert result.valid is True
