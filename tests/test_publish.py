"""Global context: require byte-identical Safetensors before publication completes.

Sources:
- https://docs.python.org/3.12/library/hashlib.html
- https://github.com/huggingface/huggingface_hub/blob/48ef2781c2c4c2247431c97efcd5487e89d42732/src/huggingface_hub/hf_api.py
"""

from pathlib import Path

import pytest

from minibug_rl.artifacts import PUBLICATION_ALLOWLIST
from minibug_rl.phases.publish import _validate_remote_publication_files, _verify_safetensors


def _write_weight(root: Path, relative_path: str, content: bytes) -> None:
    """Create one inert file shaped like a model weight for digest-only tests."""
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_verify_safetensors_accepts_identical_recursive_weight_tree(tmp_path: Path) -> None:
    """Hash root and adapter weights using stable relative paths."""
    local = tmp_path / "local"
    downloaded = tmp_path / "downloaded"
    for root in (local, downloaded):
        _write_weight(root, "model.safetensors", b"merged-weights")
        _write_weight(root, "adapter/adapter_model.safetensors", b"adapter-weights")

    hashes = _verify_safetensors(local, downloaded)

    assert set(hashes) == {"model.safetensors", "adapter/adapter_model.safetensors"}
    assert all(len(digest) == 64 for digest in hashes.values())


@pytest.mark.parametrize(
    ("local_files", "downloaded_files", "message"),
    [
        ({}, {}, "no Safetensors"),
        (
            {"model.safetensors": b"same", "adapter/adapter_model.safetensors": b"adapter"},
            {"model.safetensors": b"same"},
            "file set differs",
        ),
        (
            {"model.safetensors": b"local"},
            {"model.safetensors": b"remote"},
            "hash mismatch",
        ),
    ],
)
def test_verify_safetensors_rejects_missing_or_mismatched_weights(
    tmp_path: Path,
    local_files: dict[str, bytes],
    downloaded_files: dict[str, bytes],
    message: str,
) -> None:
    """Fail closed when weight coverage or bytes cannot verify the exact revision."""
    local = tmp_path / "local"
    downloaded = tmp_path / "downloaded"
    local.mkdir()
    downloaded.mkdir()
    for relative_path, content in local_files.items():
        _write_weight(local, relative_path, content)
    for relative_path, content in downloaded_files.items():
        _write_weight(downloaded, relative_path, content)

    with pytest.raises(RuntimeError, match=message):
        _verify_safetensors(local, downloaded)


def test_remote_publication_tree_allows_only_reviewed_files_and_hub_metadata() -> None:
    """Treat the Hub-managed attributes file as metadata, not unreviewed model content."""
    remote_files = sorted(PUBLICATION_ALLOWLIST | {".gitattributes"})

    assert set(_validate_remote_publication_files(remote_files)) == set(remote_files)


@pytest.mark.parametrize(
    "remote_files",
    [
        sorted(PUBLICATION_ALLOWLIST - {"results.json"}),
        sorted(PUBLICATION_ALLOWLIST | {"adapter/training_args.bin"}),
    ],
)
def test_remote_publication_tree_rejects_missing_or_extra_files(
    remote_files: list[str],
) -> None:
    """Ensure a stale remote artifact cannot survive the replacement upload unnoticed."""
    with pytest.raises(RuntimeError, match="Published Hub tree differs"):
        _validate_remote_publication_files(remote_files)
