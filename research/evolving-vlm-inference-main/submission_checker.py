"""Portable fail-closed source/ZIP checker for the challenge deliverable."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

BLOCKED_EXIT = 2
MAX_MEMBERS = 5000
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0
HEX64 = set("0123456789abcdef")
OFFICIAL_STYLE_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"

GATE1_REQUIRED = [
    "README.md",
    "evaluate.py",
    "requirements.lock.txt",
    "reproduce.sh",
]
FINAL_REQUIRED = [
    "report.pdf",
    "manual_instruct.py",
    "manual_thinking.py",
    "evolved_instruct.py",
    "evolved_thinking.py",
    "evolve_instruct.py",
    "evolve_thinking.py",
    "best_accuracy.py",
    "best_speed.py",
    "best_overall.py",
    "evaluate.py",
    "reproduce.sh",
    "README.md",
    "requirements.lock.txt",
    "submission_checker.py",
    "scripts/package_submission.py",
    "evaluation/recovery_finalize.py",
    "evaluation/recovery_v2/final_manifest.json",
    "report/main.tex",
    "report/references.bib",
    "report/neurips_2025.sty",
    "report/OFFICIAL_TEMPLATE_PROVENANCE.json",
    "report/generated/formal_results.tex",
    "report/generated/report_number_trace.json",
    "model_locator.py",
    "evaluation/portable_reproduce.py",
    "evaluation/portability_identity.py",
    "evaluation/portability/amendment_manifest.json",
    "scripts/portable_full_inference.py",
    "scripts/final_portability_smoke.py",
]
HISTORICAL_BLOCKED_REQUIRED = [
    "report.pdf",
    "manual_instruct.py",
    "manual_thinking.py",
    "evolved_instruct.py",
    "evolved_thinking.py",
    "evolve_instruct.py",
    "evolve_thinking.py",
    "best_accuracy.py",
    "best_speed.py",
    "best_overall.py",
    "evaluate.py",
    "reproduce.sh",
    "README.md",
    "requirements.lock.txt",
    "evaluation/closeout_manifest.json",
    "evaluation/verify_closeout.py",
]

FORMAL_DEPENDENT = {
    "best_accuracy.py",
    "best_speed.py",
    "best_overall.py",
}
FORBIDDEN_PATTERNS = [
    r"\bTBD\b",
    r"\bTODO\b",
    r"\bPLACE" + r"HOLDER\b",
    r"xx\.x",
]


class CheckError(RuntimeError):
    """Package input is structurally unsafe."""


def _reject_constant(value: str) -> None:
    raise CheckError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(payload: str, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CheckError(f"invalid JSON {label}: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str | None:
    if not name or "\x00" in name or "\\" in name:
        return "empty, NUL, or backslash ZIP path"
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        return "absolute or traversal ZIP path"
    if any(part in {"", "."} for part in pure.parts):
        return "non-canonical ZIP path"
    if pure.as_posix() != name.rstrip("/"):
        return "non-canonical ZIP path"
    return None


def audit_zip(path: Path, *, require_manifest: bool = True) -> dict[str, Any]:
    issues: list[str] = []
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    casefolded: set[str] = set()
    total = 0
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        return {"issues": [f"invalid ZIP: {exc}"], "members": [], "total_bytes": 0}
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            issues.append(f"too many ZIP members: {len(infos)}")
        bad = archive.testzip()
        if bad:
            issues.append(f"CRC failure: {bad}")
        for info in infos:
            name = info.filename
            name_issue = _safe_member_name(name)
            if name_issue:
                issues.append(f"{name_issue}: {name!r}")
            basename = PurePosixPath(name).name.upper()
            if basename.startswith("BLOCKED_STATUS") or basename.startswith("PENDING_STATUS"):
                issues.append(f"obsolete blocked/pending state artifact: {name}")
            if name in seen:
                issues.append(f"duplicate ZIP member: {name}")
            if name.casefold() in casefolded:
                issues.append(f"case-conflicting ZIP member: {name}")
            seen.add(name)
            casefolded.add(name.casefold())
            if info.flag_bits & 0x1:
                issues.append(f"encrypted ZIP member: {name}")
            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if file_type == stat.S_IFLNK:
                issues.append(f"ZIP symlink: {name}")
            elif file_type and not (
                file_type == stat.S_IFREG or file_type == stat.S_IFDIR
            ):
                issues.append(f"ZIP special file: {name}")
            if info.is_dir():
                continue
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES:
                issues.append(f"oversized ZIP member: {name}")
            if info.file_size and info.compress_size == 0:
                issues.append(f"impossible compression size: {name}")
            elif info.file_size >= 1024 * 1024:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > MAX_COMPRESSION_RATIO:
                    issues.append(f"suspicious compression ratio {ratio:.1f}: {name}")
            members.append(
                {
                    "path": name,
                    "size_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
            )
        if total > MAX_TOTAL_BYTES:
            issues.append(f"uncompressed ZIP total exceeds limit: {total}")

        manifest_name = "PACKAGE_MANIFEST.json"
        if require_manifest and manifest_name not in seen:
            issues.append("missing PACKAGE_MANIFEST.json")
        elif require_manifest and manifest_name in seen:
            try:
                info = archive.getinfo(manifest_name)
                if info.file_size > 1024 * 1024:
                    raise CheckError("PACKAGE_MANIFEST.json is oversized")
                manifest = _loads_json(
                    archive.read(info).decode("utf-8"), manifest_name
                )
                _verify_internal_manifest(archive, manifest, members)
            except (KeyError, OSError, UnicodeError, CheckError) as exc:
                issues.append(str(exc))
    return {"issues": issues, "members": members, "total_bytes": total}


def _verify_internal_manifest(
    archive: zipfile.ZipFile,
    manifest: Any,
    members: list[dict[str, Any]],
) -> None:
    if not isinstance(manifest, dict):
        raise CheckError("PACKAGE_MANIFEST.json must be an object")
    required = {
        "schema_version",
        "status",
        "formal_complete",
        "missing_formal_artifacts",
        "files",
        "source",
    }
    if not required.issubset(manifest):
        raise CheckError("PACKAGE_MANIFEST.json missing required fields")
    if manifest["schema_version"] != 1:
        raise CheckError("unsupported PACKAGE_MANIFEST.json version")
    if manifest["status"] != "COMPLETE":
        raise CheckError("package status must be COMPLETE")
    if manifest["formal_complete"] is not True:
        raise CheckError("package does not bind formal completion")
    if manifest["missing_formal_artifacts"] != []:
        raise CheckError("complete package must have no missing formal artifacts")
    files = manifest["files"]
    if not isinstance(files, list):
        raise CheckError("package file list must be an array")
    records: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path", "sha256", "size_bytes"
        }:
            raise CheckError("invalid package manifest file record")
        name = record["path"]
        if _safe_member_name(name) or name == "PACKAGE_MANIFEST.json":
            raise CheckError(f"unsafe manifest file path: {name!r}")
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - HEX64:
            raise CheckError(f"invalid manifest SHA-256: {name}")
        if name in records:
            raise CheckError(f"duplicate package manifest path: {name}")
        records[name] = record
    archive_files = {
        item["path"] for item in members if item["path"] != "PACKAGE_MANIFEST.json"
    }
    if set(records) != archive_files:
        raise CheckError("package manifest member set mismatch")
    for name in sorted(records):
        info = archive.getinfo(name)
        if info.file_size > MAX_MEMBER_BYTES:
            raise CheckError(f"manifested member too large: {name}")
        payload = archive.read(info)
        if len(payload) != records[name]["size_bytes"]:
            raise CheckError(f"package manifest size mismatch: {name}")
        if sha256_bytes(payload) != records[name]["sha256"]:
            raise CheckError(f"package manifest SHA-256 mismatch: {name}")


def regular_nonempty(root: Path, relative: str) -> str | None:
    path = root / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if not stat.S_ISREG(mode):
        return "not_regular_file"
    if not path.resolve(strict=True).is_relative_to(root):
        return "escaped_root"
    if path.stat().st_size == 0:
        return "empty"
    return None


def _audit_tree_symlinks(root: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in list(directories) + files:
            path = base / name
            if path.is_symlink():
                failures.append(
                    {"path": path.relative_to(root).as_posix(), "issue": "symlink"}
                )
    return failures


def _blocked_status(root: Path) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    path = root / "BLOCKED_STATUS.json"
    issue = regular_nonempty(root, "BLOCKED_STATUS.json")
    if issue:
        return ([{"path": "BLOCKED_STATUS.json", "issue": issue}], None)
    try:
        document = _loads_json(path.read_text(encoding="utf-8"), str(path))
    except (OSError, CheckError) as exc:
        return ([{"path": "BLOCKED_STATUS.json", "issue": str(exc)}], None)
    expected = {
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
        "missing_formal_artifacts": sorted(FORMAL_DEPENDENT),
    }
    failures = []
    for key, value in expected.items():
        if document.get(key) != value or type(document.get(key)) is not type(value):
            failures.append(
                {"path": "BLOCKED_STATUS.json", "issue": f"invalid field: {key}"}
            )
    return failures, document


def _program_interface(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return str(exc)
    function = next(
        (
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "vlm_inference"
        ),
        None,
    )
    if function is None:
        # The two frozen manual programs expose the exact interface through the
        # trusted make_manual_inference factory; accept only that declarative AST
        # shape, never an arbitrary alias or call.
        factory_assignment = next(
            (
                node for node in tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "vlm_inference"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "make_manual_inference"
                and len(node.value.args) == 1
                and not node.value.keywords
            ),
            None,
        )
        if factory_assignment is not None:
            return None
        return "missing vlm_inference(image_path, question)"
    args = function.args
    names = [item.arg for item in args.args]
    if names != ["image_path", "question"] or args.vararg or args.kwarg:
        return "vlm_inference signature must be (image_path, question)"
    return None


def check_root(
    root: Path,
    stage: str,
    allow_missing_formal: bool,
) -> dict[str, Any]:
    required = (
        GATE1_REQUIRED
        if stage == "gate1"
        else (HISTORICAL_BLOCKED_REQUIRED if allow_missing_formal else FINAL_REQUIRED)
    )
    failures: list[dict[str, str]] = _audit_tree_symlinks(root)
    blockers: list[dict[str, str]] = []
    blocked_status: dict[str, Any] | None = None
    if allow_missing_formal:
        if stage != "final":
            failures.append(
                {"path": "--allow-missing-formal", "issue": "valid only for final"}
            )
        status_failures, blocked_status = _blocked_status(root)
        failures.extend(status_failures)

    for relative in required:
        issue = regular_nonempty(root, relative)
        if not issue:
            continue
        if allow_missing_formal and relative in FORMAL_DEPENDENT and issue == "missing":
            blockers.append({"path": relative, "issue": "NOT_RUN: Phase3 formal incomplete"})
        else:
            failures.append({"path": relative, "issue": issue})

    if allow_missing_formal:
        present_formal = sorted(
            relative for relative in FORMAL_DEPENDENT
            if regular_nonempty(root, relative) is None
        )
        if present_formal:
            failures.append(
                {
                    "path": ",".join(present_formal),
                    "issue": "formal-dependent exports must not exist in blocked package",
                }
            )

    for name in [
        "manual_instruct.py",
        "manual_thinking.py",
        "evolved_instruct.py",
        "evolved_thinking.py",
        "best_accuracy.py",
        "best_speed.py",
        "best_overall.py",
    ]:
        path = root / name
        if path.is_file() and not path.is_symlink():
            issue = _program_interface(path)
            if issue:
                failures.append({"path": name, "issue": issue})

    python_files: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories
            if name not in {".git", ".venv", ".cache", ".tmp", "__pycache__"}
        ]
        for name in files:
            if name.endswith(".py"):
                python_files.append(Path(current) / name)
    for path in sorted(python_files):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(
                {"path": path.relative_to(root).as_posix(), "issue": str(exc)}
            )

    reproduce = root / "reproduce.sh"
    if reproduce.is_file() and not reproduce.is_symlink():
        bash_check = subprocess.run(
            ["bash", "-n", str(reproduce)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if bash_check.returncode:
            failures.append(
                {"path": "reproduce.sh", "issue": bash_check.stderr.strip()}
            )
        if stage == "final" and not os.access(reproduce, os.X_OK):
            failures.append({"path": "reproduce.sh", "issue": "not_executable"})

    if stage == "final":
        if not allow_missing_formal and ((root / "BLOCKED_STATUS.json").exists() or (root / "BLOCKED_STATUS.json").is_symlink()):
            failures.append({"path": "BLOCKED_STATUS.json", "issue": "obsolete root BLOCKED marker"})
        if not allow_missing_formal:
            try:
                from evaluation.portable_reproduce import replay
                sealed = replay(verify_closure=True)
                if sealed.get("status") != "COMPLETE" or sealed.get("formal_complete") is not True:
                    raise CheckError("portable formal status differs")
            except (CheckError, ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
                failures.append(
                    {
                        "path": "evaluation/recovery_v2/final_manifest.json",
                        "issue": f"portable raw-evidence replay failed: {exc}",
                    }
                )
            try:
                from evaluation.final_fresh_evidence import verify_active_document
                fresh = verify_active_document(root, "final")
                if fresh.get("status") != "COMPLETE" or fresh.get("result_count") not in {8, 9}:
                    raise CheckError("final fresh reproduction status differs")
            except (CheckError, ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
                failures.append(
                    {
                        "path": "docs/FINAL_PORTABILITY_SMOKE.md",
                        "issue": f"active fresh reproduction verification failed: {exc}",
                    }
                )
        tex = root / "report/main.tex"
        style = root / "report/neurips_2025.sty"
        if style.is_file() and not style.is_symlink() and sha256(style) != OFFICIAL_STYLE_SHA256:
            failures.append(
                {"path": "report/neurips_2025.sty", "issue": "official style SHA-256 differs"}
            )
        if tex.is_file() and not tex.is_symlink():
            tex_source = tex.read_text(encoding="utf-8", errors="replace")
            if r"\usepackage[preprint]{neurips_2025}" not in tex_source:
                failures.append(
                    {"path": "report/main.tex", "issue": "official NeurIPS 2025 style not required"}
                )
            if "\\IfFileExists" in tex_source or "\\usepackage[margin=1in]{geometry}" in tex_source:
                failures.append(
                    {"path": "report/main.tex", "issue": "forbidden report-style fallback"}
                )
        for relative in required:
            path = root / relative
            if path.suffix in {".py", ".sh", ".md", ".txt"} and path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                for pattern in FORBIDDEN_PATTERNS:
                    if re.search(pattern, text, flags=re.IGNORECASE):
                        failures.append(
                            {"path": relative, "issue": f"place" f"holder pattern: {pattern}"}
                        )
        pdf = root / "report.pdf"
        if pdf.is_file():
            payload = pdf.read_bytes()
            if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
                failures.append({"path": "report.pdf", "issue": "invalid PDF framing"})
            if len(payload) < 4096:
                failures.append({"path": "report.pdf", "issue": "implausibly small PDF"})

    status = "fail" if failures else ("blocked" if blockers else "pass")
    return {
        "schema_version": 2,
        "stage": stage,
        "root": str(root),
        "status": status,
        "formal_complete": status == "pass" and stage == "final",
        "allow_missing_formal": allow_missing_formal,
        "required_count": len(required),
        "failures": failures,
        "blockers": blockers,
        "blocked_status": blocked_status,
        "note": (
            "BLOCKED is an honest incomplete artifact and is not submission readiness."
            if status == "blocked"
            else "Gate 1 pass does not imply final submission readiness."
        ),
    }


def _write_no_replace(path: Path, payload: bytes, anchor: Path) -> None:
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(anchor):
        raise CheckError("JSON output escaped anchor")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _default_anchor() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate1", "final"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-missing-formal", action="store_true")
    args = parser.parse_args(argv)

    anchor = (args.anchor or _default_anchor()).resolve(strict=True)
    root = args.root.resolve(strict=True)
    if not root.is_dir() or not root.is_relative_to(anchor):
        raise SystemExit(f"root must be a directory inside anchor: {root}")
    report = check_root(root, args.stage, args.allow_missing_formal)

    if args.zip_path:
        archive = args.zip_path.resolve(strict=True)
        if not archive.is_file() or not archive.is_relative_to(anchor):
            zip_report = {"issues": ["ZIP must be a regular file inside anchor"]}
        else:
            zip_report = audit_zip(archive)
        report["zip"] = zip_report
        report["failures"].extend(
            {"path": str(args.zip_path), "issue": issue}
            for issue in zip_report["issues"]
        )
        if zip_report["issues"]:
            report["status"] = "fail"
            report["formal_complete"] = False

    encoded = (
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if args.json_out:
        output = args.json_out if args.json_out.is_absolute() else anchor / args.json_out
        _write_no_replace(output, encoded, anchor)
    sys.stdout.buffer.write(encoded)
    if report["status"] == "pass":
        return 0
    if report["status"] == "blocked":
        return BLOCKED_EXIT
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
