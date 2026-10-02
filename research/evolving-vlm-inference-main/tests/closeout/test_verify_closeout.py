from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluation.verify_closeout import (
    VerificationError,
    _write_no_replace,
    verify,
)


PHASE3 = {
    "campaign_id": "6beb857c-ef2c-4e0a-a6e6-34abee7f4a13",
    "completed_runs": 7,
    "planned_runs": 30,
    "failed_run_index": 8,
    "timeout_seconds": 7200,
    "automatic_retry": False,
    "gpu1_queried_or_used": False,
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> Path:
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    p2 = evidence / "phase2.json"
    p3 = evidence / "phase3.json"
    p2.write_text('{"status":"completed","children":6}\n', encoding="utf-8")
    p3.write_text('{"status":"timeout","retry":false}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "BLOCKED_PHASE3_TIMEOUT",
        "formal_complete": False,
        "phase2_complete": True,
        "phase3": PHASE3,
        "artifacts": [
            {
                "path": "evidence/phase2.json",
                "scope": "phase2",
                "sha256": _hash(p2),
                "size_bytes": p2.stat().st_size,
            },
            {
                "path": "evidence/phase3.json",
                "scope": "phase3",
                "sha256": _hash(p3),
                "size_bytes": p3.stat().st_size,
            },
        ],
        "json_assertions": [
            {
                "path": "evidence/phase2.json",
                "pointer": "/status",
                "expected": "completed",
            },
            {
                "path": "evidence/phase3.json",
                "pointer": "/status",
                "expected": "timeout",
            },
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rewrite_hash(manifest: Path, artifact: Path) -> None:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    record = next(
        item for item in document["artifacts"]
        if item["path"] == artifact.relative_to(manifest.parent).as_posix()
    )
    record["sha256"] = _hash(artifact)
    record["size_bytes"] = artifact.stat().st_size
    manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def test_phase2_pass_and_phase3_never_claims_pass(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    before = set(sys.modules)
    phase2 = verify(tmp_path, manifest, "phase2")
    phase3 = verify(tmp_path, manifest, "phase3")
    both = verify(tmp_path, manifest, "all")
    assert phase2["status"] == "pass"
    assert phase3["status"] == both["status"] == "blocked"
    assert phase3["formal_complete"] is False
    assert "torch" not in set(sys.modules) - before


def test_hash_tamper_fails_closed(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    (tmp_path / "evidence/phase2.json").write_text(
        '{"status":"completed","children":7}\n', encoding="utf-8"
    )
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        verify(tmp_path, manifest, "phase2")


def test_semantic_tamper_fails_even_with_rehashed_artifact(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "evidence/phase2.json"
    artifact.write_text('{"status":"not-completed","children":6}\n', encoding="utf-8")
    _rewrite_hash(manifest, artifact)
    with pytest.raises(VerificationError, match="JSON assertion mismatch"):
        verify(tmp_path, manifest, "phase2")


@pytest.mark.parametrize(
    "payload, message",
    [
        ('{"status":"completed","status":"forged"}\n', "duplicate JSON key"),
        ('{"status":NaN}\n', "non-finite JSON constant"),
    ],
)
def test_noncanonical_json_fails_after_outer_hash_is_updated(
    tmp_path: Path, payload: str, message: str
) -> None:
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "evidence/phase2.json"
    artifact.write_text(payload, encoding="utf-8")
    _rewrite_hash(manifest, artifact)
    with pytest.raises(VerificationError, match=message):
        verify(tmp_path, manifest, "phase2")


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "evidence/phase2.json"
    payload = artifact.read_bytes()
    artifact.unlink()
    target = tmp_path / "evidence/inside.json"
    target.write_bytes(payload)
    artifact.symlink_to(target.name)
    with pytest.raises(VerificationError, match="non-symlink"):
        verify(tmp_path, manifest, "phase2")


def test_no_replace_output_preserves_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    _write_no_replace(output, b"first\n", tmp_path)
    with pytest.raises(FileExistsError):
        _write_no_replace(output, b"second\n", tmp_path)
    assert output.read_bytes() == b"first\n"
