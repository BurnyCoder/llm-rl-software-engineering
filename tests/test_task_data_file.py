"""Validate the immutable MiniBug JSON curriculum and its integrity manifest.

Global context: these tests keep malformed records and cross-split exact-code leakage out
of training before any model or sandbox is loaded. The implementation follows Python's
JSON, AST, and SHA-256 APIs:
https://docs.python.org/3/library/json.html
https://docs.python.org/3/library/ast.html
https://docs.python.org/3/library/hashlib.html
"""

# `ast` parses task source without importing or running it.
import ast
# `Counter` computes the required immutable split sizes.
from collections import Counter
# `hashlib` recomputes every recorded SHA-256 integrity value.
import hashlib
# `json` loads data and verifies that every case remains JSON serializable.
import json
# `Path` resolves fixtures relative to this test instead of the caller's directory.
from pathlib import Path
# `unittest` keeps this validation runnable with the Python standard library alone.
import unittest


# The repository root is the parent of the tests directory containing this file.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The curriculum path is centralized so every test examines the same bytes.
TASKS_PATH = REPOSITORY_ROOT / "data" / "minibug_tasks.json"
# The manifest path is centralized beside its corresponding curriculum.
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "split_manifest.json"
# An exact field set prevents accidental gold answers or undocumented metadata additions.
TASK_FIELDS = {
    "id",
    "split",
    "family",
    "difficulty",
    "specification",
    "function_name",
    "buggy_code",
    "public_tests",
    "hidden_tests",
}
# Test cases cross the sandbox boundary using only these three documented fields.
CASE_FIELDS = {"args", "kwargs", "expected"}
# These frozen counts implement the pre-registered 60-task split.
EXPECTED_COUNTS = {"train": 36, "validation": 12, "test": 12}


def canonical_task_bytes(task: dict[str, object]) -> bytes:
    """Return stable UTF-8 JSON bytes for a task manifest digest.

    Python documents `sort_keys=True` for stable regression comparisons and compact
    separators for whitespace-independent serialization:
    https://docs.python.org/3/library/json.html#json.dumps
    """

    # Sorting keys and fixing separators makes the digest independent of file formatting.
    canonical = json.dumps(
        task,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    # UTF-8 turns the canonical text into the bytes required by SHA-256.
    return canonical.encode("utf-8")


def source_fingerprint(source: str) -> str:
    """Hash a location-free AST representation of one buggy function.

    `include_attributes=False` excludes line and column metadata as documented at
    https://docs.python.org/3/library/ast.html#ast.dump.
    """

    # Parsing rejects syntactically invalid task programs before a training run starts.
    tree = ast.parse(source)
    # Dumping the AST normalizes whitespace while retaining semantic names and literals.
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    # The hexadecimal digest is stable, compact, and directly comparable to the manifest.
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class TaskDataFileTests(unittest.TestCase):
    """Check schema, splits, Python structure, and manifest integrity as one boundary."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load both small JSON fixtures once for this test class."""

        # Reading explicitly as UTF-8 matches the manifest's canonicalization contract.
        cls.tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        # The same decoder verifies that the manifest itself is valid JSON.
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_exact_task_and_split_counts(self) -> None:
        """Require exactly 60 records divided 36/12/12."""

        # A list is required because record order must remain explicit and reviewable.
        self.assertIsInstance(self.tasks, list)
        # The requested curriculum size is fixed rather than a minimum.
        self.assertEqual(len(self.tasks), 60)
        # Counting actual split labels rejects unknown labels as well as count drift.
        actual_counts = Counter(task["split"] for task in self.tasks)
        # Converting to a normal dict makes the equality message easy to read.
        self.assertEqual(dict(actual_counts), EXPECTED_COUNTS)

    def test_task_and_case_schema(self) -> None:
        """Require exact task fields and JSON-safe public and hidden cases."""

        # Each record is checked independently so failures identify the affected task.
        for task in self.tasks:
            # A subtest preserves the task ID in assertion output.
            with self.subTest(task_id=task.get("id")):
                # Exact fields prevent undocumented data and accidental gold solutions.
                self.assertEqual(set(task), TASK_FIELDS)
                # Text metadata must be non-empty to produce an actionable repair prompt.
                for field in ("id", "family", "difficulty", "specification", "function_name", "buggy_code"):
                    # Both type and content are checked instead of relying on truthiness alone.
                    self.assertIsInstance(task[field], str)
                    # Whitespace-only metadata is not useful to training or evaluation.
                    self.assertTrue(task[field].strip())
                # The prompt-visible suite is deliberately fixed at two compact examples.
                self.assertEqual(len(task["public_tests"]), 2)
                # Hidden suites balance behavioral coverage with cheap local execution.
                self.assertGreaterEqual(len(task["hidden_tests"]), 4)
                # Eight hidden cases is the stated upper bound for this tiny curriculum.
                self.assertLessEqual(len(task["hidden_tests"]), 8)
                # Both suites share the exact same transport schema.
                for suite_name in ("public_tests", "hidden_tests"):
                    # Every test case is validated rather than sampling one representative.
                    for case in task[suite_name]:
                        # Exact keys keep the later sandbox protocol narrow.
                        self.assertEqual(set(case), CASE_FIELDS)
                        # Positional arguments must be a JSON array.
                        self.assertIsInstance(case["args"], list)
                        # Keyword arguments must be a JSON object.
                        self.assertIsInstance(case["kwargs"], dict)
                        # Reject NaN and Infinity because strict JSON transports cannot use them.
                        json.dumps(case, allow_nan=False)

    def test_ids_are_unique(self) -> None:
        """Ensure every task can be addressed by one unambiguous identifier."""

        # Extracting once gives a direct duplicate-sensitive length comparison.
        task_ids = [task["id"] for task in self.tasks]
        # A set removes duplicates, so equal lengths prove uniqueness.
        self.assertEqual(len(task_ids), len(set(task_ids)))

    def test_buggy_code_is_one_named_import_free_function(self) -> None:
        """Require parseable, short, single-function repair targets with matching arity."""

        # Every stored program is statically inspected and is never executed by this test.
        for task in self.tasks:
            # A subtest makes malformed-source failures immediately attributable.
            with self.subTest(task_id=task["id"]):
                # `ast.parse` supplies the canonical Python grammar check.
                tree = ast.parse(task["buggy_code"])
                # Exactly one top-level statement means there is no setup code beside the function.
                self.assertEqual(len(tree.body), 1)
                # The sole statement must be a synchronous function definition.
                self.assertIsInstance(tree.body[0], ast.FunctionDef)
                # The source name must agree with the callable named by the task record.
                self.assertEqual(tree.body[0].name, task["function_name"])
                # Imports are forbidden even if nested inside a branch or function body.
                self.assertFalse(
                    any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
                )
                # Twenty source lines is a conservative ceiling for a tiny repair exercise.
                self.assertLessEqual(len(task["buggy_code"].splitlines()), 20)
                # These tasks intentionally use positional arguments and no varargs or kwargs.
                function = tree.body[0]
                # Counting declared positional parameters catches malformed case fixtures.
                positional_count = len(function.args.posonlyargs) + len(function.args.args)
                # Every case must supply the function's declared positional arity.
                for case in task["public_tests"] + task["hidden_tests"]:
                    # Empty keyword mappings make an exact positional comparison appropriate.
                    self.assertEqual(case["kwargs"], {})
                    # Equal counts prevent reward failures caused only by fixture arity mistakes.
                    self.assertEqual(len(case["args"]), positional_count)

    def test_manifest_matches_every_task(self) -> None:
        """Recompute canonical hashes and AST fingerprints for all records."""

        # The version lets future incompatible manifest formats fail explicitly.
        self.assertEqual(self.manifest["schema_version"], 1)
        # Stored counts include an explicit total for quick human inspection.
        expected_manifest_counts = {**EXPECTED_COUNTS, "total": 60}
        # The manifest count summary must match the registered split.
        self.assertEqual(self.manifest["counts"], expected_manifest_counts)
        # Index actual tasks once so manifest lookup does not depend on JSON list ordering.
        tasks_by_id = {task["id"]: task for task in self.tasks}
        # The manifest must cover every task exactly once and nothing else.
        self.assertEqual(set(self.manifest["tasks"]), set(tasks_by_id))
        # Fingerprints are collected globally to detect identical source across any splits.
        fingerprints: list[str] = []
        # Recompute both integrity values instead of trusting checked-in metadata.
        for task_id, task in tasks_by_id.items():
            # A subtest keeps a changed record's ID in the failure message.
            with self.subTest(task_id=task_id):
                # Resolve the corresponding manifest record by stable task ID.
                entry = self.manifest["tasks"][task_id]
                # Split duplication in the manifest guards against moving a task silently.
                self.assertEqual(entry["split"], task["split"])
                # SHA-256 follows the documented one-shot bytes API.
                canonical_sha256 = hashlib.sha256(canonical_task_bytes(task)).hexdigest()
                # Any task-field or test-case edit must invalidate this assertion.
                self.assertEqual(entry["canonical_sha256"], canonical_sha256)
                # The source fingerprint independently tracks normalized Python structure.
                fingerprint = source_fingerprint(task["buggy_code"])
                # Formatting-only edits retain the fingerprint while semantic edits change it.
                self.assertEqual(entry["ast_fingerprint"], fingerprint)
                # Save it for the global cross-split duplicate check below.
                fingerprints.append(fingerprint)
        # Global uniqueness is stricter than merely checking duplicates within each split.
        self.assertEqual(len(fingerprints), len(set(fingerprints)))


# Direct execution supports `python tests/test_task_data_file.py` without pytest.
if __name__ == "__main__":
    # Verbose mode names the precise invariant being checked in local terminal output.
    unittest.main(verbosity=2)
