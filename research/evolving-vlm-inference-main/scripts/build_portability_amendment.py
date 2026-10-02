"""Build the deterministic no-replace final portability amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCRIPT_PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from evaluation.portable_reproduce import (
    FINAL_MANIFEST,
    PORTABLE_EXECUTABLE_CLOSURE,
    PROJECT_ROOT,
    load_json,
    sha256_file,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/portability/amendment_manifest.json"
ALLOWED_FORMAL_DOCUMENT_CHANGES = {
    "report/main.tex": "post-formal report portability/correction source",
}
ALLOWED_POST_FORMAL_OPERATIONAL_CHANGES = {
    "evolution/local_campaign.py": (
        "completion recovery and relocation only; sealed campaign artifacts and "
        "exported candidates remain immutable"
    ),
    "scripts/build_portability_amendment.py": (
        "fail-closed amendment regeneration and provenance disclosure"
    ),
    "scripts/package_submission.py": (
        "deterministic repair packaging and restored runtime-input closure"
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}")
AMENDMENT_RELATIVE = "evaluation/portability/amendment_manifest.json"


class AmendmentBuildError(RuntimeError):
    """The pre-portability formal closure or current closure is unsafe."""


def _safe_current(project: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise AmendmentBuildError(f"unsafe closure path: {relative}")
    path = project.joinpath(*pure.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AmendmentBuildError(f"closure file is missing: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AmendmentBuildError(f"closure path is not a physical file: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(project):
        raise AmendmentBuildError(f"closure path escaped project: {relative}")
    return resolved


def _record(project: Path, relative: str) -> dict[str, object]:
    path = _safe_current(project, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_document(
    project: Path,
    final_path: Path,
    closure_paths: Sequence[str],
) -> dict[str, object]:
    project = project.resolve(strict=True)
    final = load_json(final_path)
    formal = final.get("artifacts")
    if not isinstance(formal, list) or not formal:
        raise AmendmentBuildError("formal artifact closure is absent")
    index: dict[str, Mapping[str, object]] = {}
    for item in formal:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise AmendmentBuildError("formal artifact record fields differ")
        relative = item["path"]
        if not isinstance(relative, str) or relative in index:
            raise AmendmentBuildError("formal artifact path is invalid or duplicate")
        index[relative] = item
    if len(set(closure_paths)) != len(closure_paths):
        raise AmendmentBuildError("portable closure contains duplicate paths")

    changed: list[dict[str, object]] = []
    for relative in sorted(index):
        current = _record(project, relative)
        formal_record = index[relative]
        if (
            current["sha256"] != formal_record["sha256"]
            or current["size_bytes"] != formal_record["size_bytes"]
        ):
            changed.append(
                {
                    "path": relative,
                    "formal_sha256": formal_record["sha256"],
                    "submission_sha256": current["sha256"],
                    "size_bytes": current["size_bytes"],
                }
            )

    closure = [_record(project, relative) for relative in closure_paths]
    added = [record for record in closure if record["path"] not in index]
    final_relative = final_path.resolve(strict=True).relative_to(project).as_posix()
    return {
        "manifest_version": "final-portability-amendment-v1",
        "formal_manifest": _record(project, final_relative),
        "formal_artifact_count": len(index),
        "changed_artifacts": changed,
        "added_artifacts": added,
        "portable_executable_closure": closure,
        "statements": {
            "formal_results_modified": False,
            "algorithm_or_selection_modified": False,
            "purpose": "path portability, evidence replay, and final packaging only",
        },
    }


def apply_production_policy(
    document: dict[str, object],
    previous: Mapping[str, object],
) -> dict[str, object]:
    """Bind one repair amendment without relabeling changed formal evidence."""
    changed = document.get("changed_artifacts")
    current_closure = document.get("portable_executable_closure")
    prior_closure = previous.get("portable_executable_closure")
    if not isinstance(changed, list):
        raise AmendmentBuildError("changed formal artifact list is invalid")
    changed_formal_paths = {
        str(record.get("path")) for record in changed if isinstance(record, dict)
    }
    unexpected_formal = changed_formal_paths - set(ALLOWED_FORMAL_DOCUMENT_CHANGES)
    if unexpected_formal:
        raise AmendmentBuildError(
            "formal artifact change requires new evidence, not a portability amendment: "
            + ", ".join(sorted(unexpected_formal))
        )
    if (
        previous.get("manifest_version") != "final-portability-amendment-v1"
        or not isinstance(current_closure, list)
        or not isinstance(prior_closure, list)
    ):
        raise AmendmentBuildError("previous portability amendment is invalid")

    def closure_index(
        records: Sequence[object], label: str
    ) -> dict[str, Mapping[str, object]]:
        result: dict[str, Mapping[str, object]] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "path", "sha256", "size_bytes"
            }:
                raise AmendmentBuildError(f"{label} closure record fields differ")
            relative = record["path"]
            if not isinstance(relative, str) or relative in result:
                raise AmendmentBuildError(f"{label} closure path is invalid or duplicate")
            result[relative] = record
        return result

    current = closure_index(current_closure, "current")
    prior = closure_index(prior_closure, "previous")
    if set(current) != set(prior):
        raise AmendmentBuildError("portable executable closure membership changed")
    operational: list[dict[str, object]] = []
    for relative in sorted(current):
        before = prior[relative]
        after = current[relative]
        if (
            before["sha256"] == after["sha256"]
            and before["size_bytes"] == after["size_bytes"]
        ):
            continue
        reason = ALLOWED_POST_FORMAL_OPERATIONAL_CHANGES.get(relative)
        if reason is None:
            raise AmendmentBuildError(
                "executable change requires fresh evidence or explicit policy review: "
                + relative
            )
        operational.append(
            {
                "path": relative,
                "previous_amendment_sha256": before["sha256"],
                "submission_sha256": after["sha256"],
                "size_bytes": after["size_bytes"],
                "reason": reason,
            }
        )

    result = dict(document)
    result["formal_document_change_reasons"] = [
        {
            "path": relative,
            "reason": ALLOWED_FORMAL_DOCUMENT_CHANGES[relative],
        }
        for relative in sorted(changed_formal_paths)
    ]
    result["post_formal_operational_changes"] = operational
    result["previous_amendment_sha256"] = hashlib.sha256(
        (
            json.dumps(
                previous,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    result["statements"] = {
        "formal_results_modified": False,
        "algorithm_or_selection_modified": False,
        "algorithm_or_selection_scope": (
            "sealed formal evaluation and exported candidate selection only"
        ),
        "formal_evaluated_programs_modified": False,
        "historical_evolution_artifacts_modified": False,
        "post_formal_operational_code_modified": bool(operational),
        "purpose": (
            "preserve sealed formal evidence while repairing portable completion "
            "recovery and deterministic packaging"
        ),
    }
    return result


def write_no_replace(path: Path, document: Mapping[str, object], project: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.resolve(strict=True).is_relative_to(project):
        raise AmendmentBuildError("amendment output parent escaped project")
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_previous_from_base_zip(path: Path) -> tuple[dict[str, object], str]:
    """Load the baseline amendment only when its package manifest binds it."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AmendmentBuildError(f"baseline ZIP is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AmendmentBuildError("baseline ZIP must be one physical regular file")
    try:
        with zipfile.ZipFile(path.resolve(strict=True)) as archive:
            if sum(info.filename == AMENDMENT_RELATIVE for info in archive.infolist()) != 1:
                raise AmendmentBuildError("baseline ZIP amendment member is not unique")
            if sum(info.filename == "PACKAGE_MANIFEST.json" for info in archive.infolist()) != 1:
                raise AmendmentBuildError("baseline ZIP package manifest is not unique")
            amendment_bytes = archive.read(AMENDMENT_RELATIVE)
            package_bytes = archive.read("PACKAGE_MANIFEST.json")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise AmendmentBuildError("cannot read baseline ZIP amendment closure") from exc

    def strict_object(payload: bytes, label: str) -> dict[str, object]:
        def reject(value: str) -> object:
            raise AmendmentBuildError(f"non-finite JSON in {label}: {value}")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise AmendmentBuildError(f"duplicate JSON key in {label}: {key}")
                result[key] = value
            return result

        try:
            value = json.loads(
                payload.decode("utf-8"),
                parse_constant=reject,
                object_pairs_hook=unique,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AmendmentBuildError(f"invalid strict JSON in {label}") from exc
        if not isinstance(value, dict):
            raise AmendmentBuildError(f"JSON root is not an object in {label}")
        return value

    previous = strict_object(amendment_bytes, "baseline amendment")
    package = strict_object(package_bytes, "baseline package manifest")
    files = package.get("files")
    if not isinstance(files, list):
        raise AmendmentBuildError("baseline package file inventory is invalid")
    matches = [
        record
        for record in files
        if isinstance(record, dict) and record.get("path") == AMENDMENT_RELATIVE
    ]
    digest = hashlib.sha256(amendment_bytes).hexdigest()
    if (
        len(matches) != 1
        or set(matches[0]) != {"path", "sha256", "size_bytes"}
        or matches[0]["sha256"] != digest
        or matches[0]["size_bytes"] != len(amendment_bytes)
    ):
        raise AmendmentBuildError("baseline package does not bind its amendment")
    return previous, digest


def replace_hash_bound(
    path: Path,
    document: Mapping[str, object],
    project: Path,
    expected_previous_sha256: str,
) -> None:
    """Atomically replace only the explicitly authenticated prior amendment."""
    if HEX64.fullmatch(expected_previous_sha256) is None:
        raise AmendmentBuildError("expected previous amendment SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        raise AmendmentBuildError("existing amendment is not one physical file")
    if sha256_file(path) != expected_previous_sha256:
        raise AmendmentBuildError("existing amendment hash differs from explicit expectation")
    temporary = path.parent / f".{path.name}.{uuid.uuid4()}.next"
    try:
        write_no_replace(temporary, document, project)
        if sha256_file(path) != expected_previous_sha256:
            raise AmendmentBuildError("existing amendment changed during regeneration")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--previous",
        type=Path,
        help="prior amendment whose executable hashes define the reviewed repair delta",
    )
    parser.add_argument(
        "--base-zip",
        type=Path,
        help="authenticate the prior amendment directly from the delivered baseline ZIP",
    )
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--expected-previous-sha256")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output = output.absolute()
    if args.previous is not None and args.base_zip is not None:
        raise AmendmentBuildError("choose only one of --previous or --base-zip")
    previous_path: Path | None = None
    if args.base_zip is not None:
        base_zip = (
            args.base_zip
            if args.base_zip.is_absolute()
            else Path.cwd() / args.base_zip
        ).absolute()
        previous, previous_digest = load_previous_from_base_zip(base_zip)
    else:
        previous_argument = args.previous or DEFAULT_OUTPUT
        previous_path = (
            previous_argument
            if previous_argument.is_absolute()
            else PROJECT_ROOT / previous_argument
        ).absolute()
        previous = load_json(previous_path)
        previous_digest = sha256_file(previous_path)
    document = apply_production_policy(
        build_document(PROJECT_ROOT, FINAL_MANIFEST, PORTABLE_EXECUTABLE_CLOSURE),
        previous,
    )
    document["previous_amendment_sha256"] = previous_digest
    if args.replace_existing:
        if not isinstance(args.expected_previous_sha256, str):
            raise AmendmentBuildError(
                "--replace-existing requires --expected-previous-sha256"
            )
        replace_hash_bound(
            output,
            document,
            PROJECT_ROOT,
            args.expected_previous_sha256,
        )
    else:
        write_no_replace(output, document, PROJECT_ROOT)
    print(json.dumps({
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
        "changed": len(document["changed_artifacts"]),
        "added": len(document["added_artifacts"]),
        "closure": len(document["portable_executable_closure"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
