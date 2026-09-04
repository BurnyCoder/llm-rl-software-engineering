"""Global context: public command-line entry point for the complete MiniBug-RL workflow.

Sources:
- https://docs.python.org/3/library/argparse.html
- https://bbc2.github.io/python-dotenv/
"""

from __future__ import annotations

import argparse
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from minibug_rl.config import load_run_config
from minibug_rl.context import PipelineContext
from minibug_rl.pipeline import PHASES, run_phases
from minibug_rl.run_logging import RunLogger, utc_timestamp

# Resolve checked defaults from the package source tree during local uv execution.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPOSITORY_ROOT / "configs" / "local-8gb.toml"


def build_parser() -> argparse.ArgumentParser:
    """Declare one phase-oriented interface shared by users, tests, and documentation."""
    parser = argparse.ArgumentParser(
        prog="swe-rl",
        description="Train and publish tiny Python bug-repair GRPO on an 8 GB GPU.",
    )
    parser.add_argument(
        "phase",
        choices=["all", *PHASES],
        help="Run the whole lifecycle or one resumable phase.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Checked TOML profile (default: configs/local-8gb.toml).",
    )
    parser.add_argument(
        "--run-id",
        help="Reuse an existing timestamped run directory across separate commands.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a phase even when its state is already recorded.",
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="For `all`, stop after locally verified export rather than changing the Hub.",
    )
    return parser


def _load_environment() -> None:
    """Load ignored operator configuration after checking private file permissions."""
    environment_file = _REPOSITORY_ROOT / ".env"
    if environment_file.exists():
        mode = stat.S_IMODE(environment_file.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise RuntimeError(
                ".env must not be accessible by group or other users; run chmod 600 .env"
            )
        # Existing process variables take precedence over the local file.
        load_dotenv(environment_file, override=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Load one safe run context and invoke readable imported pipeline phases."""
    arguments = build_parser().parse_args(argv)
    _load_environment()
    config = load_run_config(arguments.config)
    run_id = arguments.run_id or f"{utc_timestamp()}-{config.project.name}"
    logger = RunLogger.create(config.project.run_root, run_id=run_id)
    context = PipelineContext.open(config, logger)
    logger.write_json("resolved-config.json", config.public_dict())
    logger.message(
        "run_start",
        "Starting MiniBug-RL command.",
        run_id=run_id,
        phase=arguments.phase,
    )
    try:
        if arguments.phase == "all":
            selected = list(PHASES)
            if arguments.skip_publish:
                selected.remove("publish")
        else:
            selected = [arguments.phase]
        run_phases(context, selected, force=arguments.force)
    except Exception as error:
        # Log the complete exception text but never serialize environment variables.
        logger.message("run_failed", f"{type(error).__name__}: {error}")
        logger.close()
        raise
    logger.message("run_complete", "MiniBug-RL command completed.", run_id=run_id)
    logger.close()
    print(f"Run artifacts: {logger.directory}", flush=True)
    return 0


if __name__ == "__main__":
    # `SystemExit` propagates the conventional process status to shell automation.
    sys.exit(main())
