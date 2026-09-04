"""Global context: validate and load the frozen repository-authored MiniBug curriculum.

Sources:
- https://docs.python.org/3/library/json.html
- https://docs.python.org/3/library/ast.html
- https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/utils.py
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from minibug_rl.prompts import build_messages
from minibug_rl.schemas import RepairTask, TestCase

# Frozen split names prevent typographical variants from silently leaking evaluation data.
_SPLITS = frozenset({"train", "validation", "test"})
# Difficulty is deliberately descriptive rather than an unbounded numeric score.
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


class TaskDataError(ValueError):
    """Identify malformed or leakage-prone curriculum data before training begins."""


class _BindingCollector(ast.NodeVisitor):
    """Collect function-local bindings in deterministic AST traversal order."""

    def __init__(self) -> None:
        """Start with no observed bindings."""
        self.identifiers: list[str] = []

    def _remember(self, value: str) -> None:
        """Retain each binding once at its first declaration or assignment."""
        if value not in self.identifiers:
            self.identifiers.append(value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Treat a function name as a binding before its arguments and body."""
        self._remember(node.name)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        """Record positional, keyword-only, variadic, and lambda parameters."""
        self._remember(node.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Record local assignment and deletion targets but preserve free names."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._remember(node.id)


class _StructuralNormalizer(ast.NodeTransformer):
    """Alpha-normalize local bindings and literal values for template matching."""

    def __init__(self, tree: ast.AST) -> None:
        """Create positional placeholders from one preordered binding pass."""
        collector = _BindingCollector()
        collector.visit(tree)
        self._identifiers = {
            identifier: f"name_{index}" for index, identifier in enumerate(collector.identifiers)
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Normalize a function binding while retaining decorators, arguments, and body."""
        node.name = self._identifiers[node.name]
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        """Normalize a bound parameter while preserving its annotation and default shape."""
        node.arg = self._identifiers[node.arg]
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Rename local references consistently and retain semantic free/builtin names."""
        node.id = self._identifiers.get(node.id, node.id)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Replace literal values with type-specific sentinels while preserving node shape."""
        value = node.value
        if value is None:
            normalized: Any = None
        elif isinstance(value, bool):
            normalized = False
        elif isinstance(value, int):
            normalized = 0
        elif isinstance(value, float):
            normalized = 0.0
        elif isinstance(value, complex):
            normalized = 0j
        elif isinstance(value, str):
            normalized = ""
        elif isinstance(value, bytes):
            normalized = b""
        else:
            normalized = value
        return ast.copy_location(ast.Constant(value=normalized, kind=node.kind), node)


def structural_fingerprint(source: str) -> str:
    """Hash code shape independent of identifier spellings and literal values.

    Source: https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """
    tree = ast.parse(source, mode="exec")
    normalized = _StructuralNormalizer(tree).visit(tree)
    dumped = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a JSON object at one schema location."""
    # Precise labels make hand-authored task mistakes fast to locate.
    if not isinstance(value, Mapping):
        raise TaskDataError(f"{label} must be a JSON object")
    return value


def _string(value: Any, label: str) -> str:
    """Require a non-empty string without normalizing meaningful task content."""
    # Whitespace-only identifiers and descriptions are never useful evidence.
    if not isinstance(value, str) or not value.strip():
        raise TaskDataError(f"{label} must be a non-empty string")
    return value


def _test_case(raw: Any, label: str) -> TestCase:
    """Validate one JSON-serializable call/expected record."""
    item = _mapping(raw, label)
    # Positional arguments must remain ordered arrays.
    args = item.get("args")
    if not isinstance(args, list):
        raise TaskDataError(f"{label}.args must be a JSON array")
    # Keyword arguments must remain named JSON objects.
    kwargs = item.get("kwargs")
    if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
        raise TaskDataError(f"{label}.kwargs must be an object with string keys")
    # Presence is different from a null expected value, which is a valid result.
    if "expected" not in item:
        raise TaskDataError(f"{label}.expected is required")
    return TestCase(tuple(args), dict(kwargs), item["expected"])


def _cases(raw: Any, label: str, minimum: int, maximum: int) -> tuple[TestCase, ...]:
    """Validate a bounded sequence of public or hidden function calls."""
    # Strings are sequences but cannot represent a list of structured test cases.
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TaskDataError(f"{label} must be a JSON array")
    if not minimum <= len(raw) <= maximum:
        raise TaskDataError(f"{label} must contain between {minimum} and {maximum} cases")
    return tuple(_test_case(case, f"{label}[{index}]") for index, case in enumerate(raw))


def _validate_buggy_function(code: str, function_name: str, label: str) -> None:
    """Require one import-free function with the public task name."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise TaskDataError(f"{label}.buggy_code is invalid Python: {error.msg}") from error
    # The prompt contract presents one replacement unit rather than a whole module.
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TaskDataError(f"{label}.buggy_code must contain one top-level function")
    if tree.body[0].name != function_name:
        raise TaskDataError(f"{label}.buggy_code must define {function_name!r}")
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise TaskDataError(f"{label}.buggy_code must not contain imports")


def _repair_task(raw: Any, index: int) -> RepairTask:
    """Convert one checked JSON object into an immutable domain task."""
    label = f"tasks[{index}]"
    item = _mapping(raw, label)
    task_id = _string(item.get("id"), f"{label}.id")
    split = _string(item.get("split"), f"{label}.split")
    if split not in _SPLITS:
        raise TaskDataError(f"{label}.split must be one of {sorted(_SPLITS)}")
    difficulty = _string(item.get("difficulty"), f"{label}.difficulty")
    if difficulty not in _DIFFICULTIES:
        raise TaskDataError(f"{label}.difficulty must be one of {sorted(_DIFFICULTIES)}")
    function_name = _string(item.get("function_name"), f"{label}.function_name")
    buggy_code = _string(item.get("buggy_code"), f"{label}.buggy_code")
    _validate_buggy_function(buggy_code, function_name, label)
    return RepairTask(
        id=task_id,
        split=split,
        family=_string(item.get("family"), f"{label}.family"),
        difficulty=difficulty,
        specification=_string(item.get("specification"), f"{label}.specification"),
        function_name=function_name,
        buggy_code=buggy_code,
        public_tests=_cases(item.get("public_tests"), f"{label}.public_tests", 2, 2),
        hidden_tests=_cases(item.get("hidden_tests"), f"{label}.hidden_tests", 4, 8),
    )


def load_tasks(path: str | Path) -> tuple[RepairTask, ...]:
    """Load and validate the complete curriculum from one versioned JSON file."""
    task_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TaskDataError(f"Cannot load task file {task_path}: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise TaskDataError("Task file must contain a non-empty JSON array")
    tasks = tuple(_repair_task(item, index) for index, item in enumerate(raw))
    identifiers = [task.id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise TaskDataError("Duplicate task IDs are prohibited")
    return tasks


def tasks_for_split(tasks: Iterable[RepairTask], split: str) -> tuple[RepairTask, ...]:
    """Select one frozen split without changing task order."""
    if split not in _SPLITS:
        raise TaskDataError(f"Unknown split {split!r}")
    return tuple(task for task in tasks if task.split == split)


def _case_dict(case: TestCase) -> dict[str, Any]:
    """Serialize a test case for the reward-only dataset column."""
    # Expected values remain outside the model-visible prompt object.
    return {"args": list(case.args), "kwargs": case.kwargs, "expected": case.expected}


def training_rows(tasks: Iterable[RepairTask]) -> list[dict[str, Any]]:
    """Build TRL conversational prompt rows with reward metadata in separate columns."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "prompt": build_messages(task),
                "task_id": task.id,
                "split": task.split,
                "function_name": task.function_name,
                "hidden_tests_json": json.dumps(
                    [_case_dict(case) for case in task.hidden_tests],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def validate_split_manifest(task_path: str | Path, manifest_path: str | Path) -> dict[str, int]:
    """Recompute frozen task hashes, AST fingerprints, and registered split counts."""
    source_path = Path(task_path).expanduser().resolve()
    raw_tasks = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw_tasks, list) or not isinstance(manifest, dict):
        raise TaskDataError("Task or manifest root has an invalid shape")
    entries = manifest.get("tasks")
    if manifest.get("schema_version") != 2 or not isinstance(entries, dict):
        raise TaskDataError("Split manifest schema is not version 2")
    counts = {split: 0 for split in _SPLITS}
    # Identical templates inside one split are allowed; crossing splits is leakage-prone.
    fingerprint_splits: dict[str, str] = {}
    for raw in raw_tasks:
        item = _mapping(raw, "manifest task")
        task_id = _string(item.get("id"), "manifest task id")
        entry = entries.get(task_id)
        if not isinstance(entry, dict):
            raise TaskDataError(f"Manifest has no entry for {task_id}")
        canonical = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        canonical_hash = hashlib.sha256(canonical).hexdigest()
        fingerprint = structural_fingerprint(str(item["buggy_code"]))
        if entry.get("canonical_sha256") != canonical_hash:
            raise TaskDataError(f"Canonical hash mismatch for {task_id}")
        if entry.get("ast_fingerprint") != fingerprint:
            raise TaskDataError(f"AST fingerprint mismatch for {task_id}")
        split = str(item["split"])
        if entry.get("split") != split:
            raise TaskDataError(f"Manifest split mismatch for {task_id}")
        previous_split = fingerprint_splits.get(fingerprint)
        if previous_split is not None and previous_split != split:
            raise TaskDataError(f"Cross-split structural duplicate for {task_id}")
        fingerprint_splits[fingerprint] = split
        counts[split] = counts.get(split, 0) + 1
    if set(entries) != {str(item["id"]) for item in raw_tasks}:
        raise TaskDataError("Manifest task IDs do not exactly match the curriculum")
    expected_counts = manifest.get("counts")
    actual_counts = {**counts, "total": len(raw_tasks)}
    if expected_counts != actual_counts:
        raise TaskDataError("Manifest split counts do not match the curriculum")
    return actual_counts
