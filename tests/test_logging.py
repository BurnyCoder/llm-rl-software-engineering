"""Global context: prove prompts and generations are persisted without truncation.

Source: https://docs.python.org/3/library/logging.html
"""

import json
from pathlib import Path

from minibug_rl.run_logging import RunLogger


def test_generation_logging_preserves_full_prompt_and_completion(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Write exact text to JSONL and the real-time terminal stream."""
    # Long sentinel tails catch accidental display-oriented truncation.
    prompt = "PROMPT-BEGIN\n" + ("p" * 20_000) + "\nPROMPT-END"
    completion = "OUTPUT-BEGIN\n" + ("o" * 20_000) + "\nOUTPUT-END"
    logger = RunLogger.create(tmp_path, run_id="20260101T000000Z-test")

    logger.generation(
        task_id="task-1",
        split="train",
        prompt=prompt,
        completion=completion,
        metadata={"reward": 1.1},
    )
    logger.close()

    # JSONL is the authoritative machine-readable audit trail.
    record = json.loads((logger.directory / "generations.jsonl").read_text(encoding="utf-8"))
    assert record["prompt"] == prompt
    assert record["completion"] == completion
    # Terminal logging is intentionally complete per the repository requirement.
    terminal = capsys.readouterr().out  # type: ignore[attr-defined]
    assert prompt in terminal
    assert completion in terminal
