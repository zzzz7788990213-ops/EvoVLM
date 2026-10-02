from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

import submission_checker as checker


PROGRAM = "def vlm_inference(image_path, question):\n    return 'A'\n"


def _write_blocked_root(root: Path) -> None:
    root.mkdir(parents=True)
    for name, payload in {
        "README.md": "# blocked evidence package\n",
        "evaluate.py": "def main():\n    return 0\n",
        "requirements.lock.txt": "jsonschema==4.25.1\n",
        "manual_instruct.py": PROGRAM,
        "manual_thinking.py": PROGRAM,
        "evolved_instruct.py": PROGRAM,
        "evolved_thinking.py": PROGRAM,
        "evolve_instruct.py": "\"\"\"Instruct evolution driver.\"\"\"\n",
        "evolve_thinking.py": "\"\"\"Thinking evolution driver.\"\"\"\n",
    }.items():
        (root / name).write_text(payload, encoding="utf-8")
    reproduce = root / "reproduce.sh"
    reproduce.write_text("#!/usr/bin/env bash\nexit 2\n", encoding="utf-8")
    reproduce.chmod(0o755)
    evaluation = root / "evaluation"
    evaluation.mkdir()
    (evaluation / "closeout_manifest.json").write_text("{}\n", encoding="utf-8")
    (evaluation / "verify_closeout.py").write_text("def main():\n    return 2\n", encoding="utf-8")
    (root / "report.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 5000 + b"\n%%EOF\n")
    status_doc = {
        "schema_version": 1,
        "status": "BLOCKED_PHASE3_TIMEOUT",
        "formal_complete": False,
        "phase2_complete": True,
        "campaign_id": "6beb857c-ef2c-4e0a-a6e6-34abee7f4a13",
        "failed_run_index": 8,
        "timeout_seconds": 7200,
        "completed_runs": 7,
        "planned_runs": 30,
        "automatic_retry": False,
        "gpu1_queried_or_used": False,
        "missing_formal_artifacts": sorted(checker.FORMAL_DEPENDENT),
    }
    (root / "BLOCKED_STATUS.json").write_text(
        json.dumps(status_doc, sort_keys=True) + "\n", encoding="utf-8"
    )


def _make_zip(path: Path, members: dict[str, bytes], *, bad_hash: bool = False) -> None:
    records = []
    for index, (name, payload) in enumerate(sorted(members.items())):
        digest = hashlib.sha256(payload).hexdigest()
        if bad_hash and index == 0:
            digest = "0" * 64
        records.append({"path": name, "sha256": digest, "size_bytes": len(payload)})
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_complete": True,
        "missing_formal_artifacts": [],
        "source": {"kind": "unit_test"},
        "files": records,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        )


def test_blocked_root_is_blocked_never_pass(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_blocked_root(root)
    report = checker.check_root(root, "final", True)
    assert report["status"] == "blocked"
    assert report["formal_complete"] is False
    assert not report["failures"]
    assert {item["path"] for item in report["blockers"]} == checker.FORMAL_DEPENDENT


def test_blocked_root_without_status_fails(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_blocked_root(root)
    (root / "BLOCKED_STATUS.json").unlink()
    report = checker.check_root(root, "final", True)
    assert report["status"] == "fail"
    assert any(item["path"] == "BLOCKED_STATUS.json" for item in report["failures"])


def test_blocked_root_rejects_formal_export(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_blocked_root(root)
    (root / "best_accuracy.py").write_text(PROGRAM, encoding="utf-8")
    report = checker.check_root(root, "final", True)
    assert report["status"] == "fail"
    assert any("must not exist" in item["issue"] for item in report["failures"])


def test_safe_manifested_zip_passes_audit(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _make_zip(archive, {"README.md": b"ok\n", "src/program.py": PROGRAM.encode()})
    report = checker.audit_zip(archive)
    assert report["issues"] == []


@pytest.mark.parametrize(
    "name",
    ["BLOCKED_STATUS.json", "docs/archive/phase3_v1/BLOCKED_STATUS.v1.json", "PENDING_STATUS.json"],
)
def test_zip_rejects_obsolete_blocked_or_pending_status(
    tmp_path: Path, name: str,
) -> None:
    archive = tmp_path / "obsolete.zip"
    _make_zip(archive, {name: b"{}\n"})
    assert any("obsolete blocked/pending" in issue for issue in checker.audit_zip(archive)["issues"])


@pytest.mark.parametrize(
    ("name", "issue"),
    [
        ("../escape", "traversal"),
        ("/absolute", "traversal"),
        ("bad\\path", "backslash"),
    ],
)
def test_zip_rejects_unsafe_names(tmp_path: Path, name: str, issue: str) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(name, b"x")
    report = checker.audit_zip(archive_path, require_manifest=False)
    assert any(issue in item for item in report["issues"])


def test_zip_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    assert any("symlink" in item for item in checker.audit_zip(archive_path, require_manifest=False)["issues"])


def test_zip_rejects_duplicate_and_case_conflict(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("A.txt", b"one")
        archive.writestr("a.txt", b"two")
        archive.writestr("A.txt", b"three")
    issues = checker.audit_zip(archive_path, require_manifest=False)["issues"]
    assert any("duplicate" in item for item in issues)
    assert any("case-conflicting" in item for item in issues)


def test_zip_rejects_manifest_hash_tamper(tmp_path: Path) -> None:
    archive_path = tmp_path / "tamper.zip"
    _make_zip(archive_path, {"README.md": b"ok\n"}, bad_hash=True)
    assert any("SHA-256 mismatch" in item for item in checker.audit_zip(archive_path)["issues"])


def test_zip_rejects_compression_bomb_ratio(tmp_path: Path) -> None:
    archive_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("zeros.bin", b"0" * (2 * 1024 * 1024))
    issues = checker.audit_zip(archive_path, require_manifest=False)["issues"]
    assert any("compression ratio" in item for item in issues)


def test_json_output_is_no_replace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "root"
    _write_blocked_root(root)
    output = tmp_path / "result.json"
    args = [
        "--stage", "final", "--root", str(root), "--anchor", str(tmp_path),
        "--allow-missing-formal", "--json-out", str(output),
    ]
    assert checker.main(args) == checker.BLOCKED_EXIT
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        checker.main(args)


def test_program_interface_accepts_trusted_manual_factory(tmp_path: Path) -> None:
    program = tmp_path / "manual.py"
    program.write_text(
        "vlm_inference = make_manual_inference(SPEC)\n", encoding="utf-8"
    )
    assert checker._program_interface(program) is None


def test_program_interface_rejects_arbitrary_factory_or_alias(tmp_path: Path) -> None:
    for index, source in enumerate((
        "vlm_inference = other_factory(SPEC)\n",
        "vlm_inference = make_manual_inference(SPEC, EXTRA)\n",
        "vlm_inference = candidate\n",
    )):
        program = tmp_path / f"unsafe-{index}.py"
        program.write_text(source, encoding="utf-8")
        assert checker._program_interface(program) is not None


def test_checker_does_not_match_its_own_placeholder_rule() -> None:
    source = Path(checker.__file__).read_text(encoding="utf-8")
    assert not any(__import__("re").search(pattern, source, flags=__import__("re").IGNORECASE)
                   for pattern in checker.FORBIDDEN_PATTERNS)
