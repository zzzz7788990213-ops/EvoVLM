"""Build a reproducible final submission ZIP from sealed Recovery-v2 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

PROJECT_NAME = "evolving-vlm-inference-main"
OUTPUT_NAME = "Zinuo_Zhang.zip"
STATUS = "COMPLETE"
MISSING: list[str] = []
TEXT_EXTENSIONS = {".bib", ".json", ".jsonl", ".lock", ".md", ".py", ".sh", ".tex", ".txt"}
ROOT_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}
TREE_ROOTS = ("charxiv", "docs", "evaluation", "evolution", "report", "scripts", "tests/closeout")
SKIP_PARTS = {".cache", ".git", ".pytest_cache", ".runtime", ".tmp", "__pycache__", "archive", "dist", "images", "official_template"}
SKIP_FILES = {
    "evaluation/provenance.py",
    "evaluation/verify_historical_v3.py",
}
SKIP_ROOT_FILES = {"PACKAGE_MANIFEST.json"}
PHASE2_MANIFESTS = tuple(
    f"data/phase2_local_search_v2/manifests/{stage}.json"
    for stage in ("warmup", "single", "search12", "selection32")
)
PHASE2_CATALOG = (
    "data/phase2_mutation_catalog/catalog.json",
    "data/phase2_mutation_catalog/catalog_manifest.json",
)
CHARXIV_IMAGE_PROVENANCE = "evaluation/provenance/charxiv_images_manifest_v4.json"
DERIVED_GENERATION_CHECKPOINT_PREFIX = (
    "results/evolution/checkpoints/phase2_local/campaign-"
)
HEX40 = re.compile(r"[0-9a-f]{40}")


class PackageError(RuntimeError):
    """The final package cannot be built safely."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise PackageError(f"non-finite JSON constant in {path}: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PackageError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject,
        object_pairs_hook=unique,
    )


def default_project() -> Path:
    script = Path(__file__).resolve(strict=True)
    candidate = script.parent.parent
    if candidate.name == PROJECT_NAME:
        return candidate
    nested = candidate / "workspace" / PROJECT_NAME
    if nested.is_dir():
        return nested.resolve(strict=True)
    raise PackageError("cannot locate project root")


def _regular_inside(project: Path, relative: str) -> Path:
    path = project / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PackageError(f"missing package source: {relative}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PackageError(f"package source is not a regular file: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(project):
        raise PackageError(f"package source escaped project: {relative}")
    return path


def _tree_files(project: Path, relative_root: str) -> set[str]:
    root = project / relative_root
    if not root.exists():
        return set()
    if root.is_symlink() or not root.is_dir():
        raise PackageError(f"unsafe package tree root: {relative_root}")
    selected: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained = []
        for name in directories:
            child = current_path / name
            if child.is_symlink():
                raise PackageError(f"symlink in package tree: {child.relative_to(project)}")
            if name not in SKIP_PARTS:
                retained.append(name)
        directories[:] = retained
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(project).as_posix()
            if relative in SKIP_FILES:
                continue
            if path.is_symlink():
                raise PackageError(f"symlink in package tree: {relative}")
            if path.suffix.lower() in TEXT_EXTENSIONS:
                selected.add(relative)
    return selected


def _canonical_nonempty_files(
    project: Path, selected: set[str], required: set[str]
) -> list[str]:
    """Return safe nonempty files, omitting only optional zero-byte tree entries."""
    canonical: list[str] = []
    total = 0
    for relative in sorted(selected):
        if relative.startswith("/") or "\\" in relative or ".." in Path(relative).parts:
            raise PackageError(f"unsafe selected path: {relative}")
        path = _regular_inside(project, relative)
        size = path.stat().st_size
        if size == 0:
            if relative in required:
                raise PackageError(f"empty required package source: {relative}")
            continue
        total += size
        canonical.append(relative)
    if len(canonical) > 5000 or total > 256 * 1024 * 1024:
        raise PackageError(f"package source budget exceeded: files={len(canonical)} bytes={total}")
    return canonical


def _add_phase2_closeout_artifacts(project: Path, selected: set[str]) -> None:
    """Add and re-hash the Phase 2 closure required by reproduce.sh check."""
    document = load_json(_regular_inside(project, "evaluation/closeout_manifest.json"))
    if not isinstance(document, dict) or document.get("phase2_complete") is not True:
        raise PackageError("Phase 2 closeout manifest is not complete")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageError("Phase 2 closeout artifact list is invalid")
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {
            "path", "scope", "sha256", "size_bytes"
        }:
            raise PackageError("invalid Phase 2 closeout artifact record")
        if record["scope"] != "phase2":
            continue
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
        ):
            raise PackageError(f"unsafe Phase 2 closeout path: {relative!r}")
        path = _regular_inside(project, relative)
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise PackageError(f"Phase 2 closeout artifact mismatch: {relative}")
        selected.add(relative)


def _add_final_fresh_artifacts(project: Path, selected: set[str]) -> dict[str, object]:
    """Require and add the receipt-bound fresh reproduction closure."""
    from evaluation.final_fresh_evidence import verify_active_document

    verified = verify_active_document(project, "final")
    if verified.get("status") != "COMPLETE":
        raise PackageError("final fresh reproduction is not complete")
    artifacts = verified.get("verified_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PackageError("final fresh artifact closure is absent")
    for relative in artifacts:
        if not isinstance(relative, str):
            raise PackageError("final fresh artifact path is invalid")
        _regular_inside(project, relative)
        selected.add(relative)
    return verified


def _add_formal_runtime_events(
    project: Path,
    selected: set[str],
    final_manifest: dict[str, Any],
) -> None:
    """Add the 26 replay-required runtime events omitted by the legacy manifest."""
    composition = final_manifest.get("composition")
    campaign_id = final_manifest.get("campaign_id")
    if not isinstance(composition, dict) or not isinstance(campaign_id, str):
        raise PackageError("formal composition identity is invalid")
    old_campaign_id = composition.get("old_campaign_id")
    old_plan_record = composition.get("retained_campaign_plan")
    new_plan_record = final_manifest.get("campaign_plan")
    if (
        not isinstance(old_campaign_id, str)
        or not isinstance(old_plan_record, dict)
        or not isinstance(new_plan_record, dict)
        or not isinstance(old_plan_record.get("path"), str)
        or not isinstance(new_plan_record.get("path"), str)
    ):
        raise PackageError("formal campaign plan records are invalid")
    plans = (
        load_json(_regular_inside(project, old_plan_record["path"])),
        load_json(_regular_inside(project, new_plan_record["path"])),
    )
    campaign_ids = (old_campaign_id, campaign_id)
    seen: set[str] = set()
    for index in range(1, 27):
        lane = 0 if index <= 11 else 1
        schedule = plans[lane].get("schedule")
        if not isinstance(schedule, list) or len(schedule) != 26:
            raise PackageError("formal campaign schedule length differs")
        entry = schedule[index - 1]
        if (
            not isinstance(entry, dict)
            or entry.get("index") != index
            or not isinstance(entry.get("result"), str)
        ):
            raise PackageError(f"formal schedule entry differs: {index}")
        result_relative = (
            f"results/phase3_recovery_v2/campaigns/{campaign_ids[lane]}/"
            f"{entry['result']}"
        )
        if result_relative not in selected:
            raise PackageError(f"formal result is absent from package closure: {result_relative}")
        result = load_json(_regular_inside(project, result_relative))
        runtime_relative = result.get("runtime_event_path")
        runtime_hash = result.get("runtime_event_sha256")
        if (
            not isinstance(runtime_relative, str)
            or not isinstance(runtime_hash, str)
            or runtime_relative in seen
        ):
            raise PackageError(f"formal runtime event reference differs: {index}")
        runtime = _regular_inside(project, runtime_relative)
        if sha256_file(runtime) != runtime_hash:
            raise PackageError(f"formal runtime event hash differs: {index}")
        selected.add(runtime_relative)
        seen.add(runtime_relative)
    if len(seen) != 26:
        raise PackageError("formal runtime event closure is incomplete")


def _add_v1_erratum_artifacts(project: Path, selected: set[str]) -> None:
    """Add every file opened while verifying the immutable Phase-3-v1 erratum."""
    erratum_relative = "evaluation/recovery_v2/phase3_v1_plan_hash_erratum.json"
    if erratum_relative not in selected:
        raise PackageError("Phase-3-v1 erratum is absent from package closure")
    document = load_json(_regular_inside(project, erratum_relative))
    plan = document.get("actual_campaign_plan")
    declarations = document.get("declarations")
    if not isinstance(plan, dict) or not isinstance(declarations, list):
        raise PackageError("Phase-3-v1 erratum artifact inventory is invalid")
    records = [plan, *declarations]
    for record in records:
        if not isinstance(record, dict):
            raise PackageError("Phase-3-v1 erratum artifact record is invalid")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        expected_size = record.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not isinstance(expected_size, int)
        ):
            raise PackageError("Phase-3-v1 erratum artifact identity is invalid")
        artifact = _regular_inside(project, relative)
        if sha256_file(artifact) != expected_hash or artifact.stat().st_size != expected_size:
            raise PackageError(f"Phase-3-v1 erratum artifact differs: {relative}")
        selected.add(relative)


def collect_files(project: Path) -> list[str]:
    selected: set[str] = set()
    for path in project.iterdir():
        if path.is_symlink():
            raise PackageError(f"symlink at project root: {path.name}")
        if (
            path.is_file()
            and path.name not in SKIP_ROOT_FILES
            and path.suffix.lower() in ROOT_EXTENSIONS
        ):
            selected.add(path.name)
    for root in TREE_ROOTS:
        selected.update(_tree_files(project, root))

    final_manifest = load_json(
        _regular_inside(project, "evaluation/recovery_v2/final_manifest.json")
    )
    if (
        final_manifest.get("status") != "COMPLETE"
        or final_manifest.get("formal_complete") is not True
        or final_manifest.get("result_count") != 26
    ):
        raise PackageError("Recovery-v2 final manifest is not formally complete")
    artifacts = final_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageError("final artifact list is invalid")
    for record in artifacts:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("size_bytes"), int)
        ):
            raise PackageError("final artifact record is invalid")
        selected.add(record["path"])

    _add_formal_runtime_events(project, selected, final_manifest)
    _add_v1_erratum_artifacts(project, selected)
    _add_phase2_closeout_artifacts(project, selected)
    _add_phase2_restart_assets(project, selected)
    _add_final_fresh_artifacts(project, selected)

    required = {
        "README.md", "evaluate.py", "requirements.lock.txt", "reproduce.sh",
        "manual_instruct.py", "manual_thinking.py", "evolved_instruct.py",
        "evolved_thinking.py", "evolve_instruct.py", "evolve_thinking.py",
        "best_accuracy.py", "best_speed.py", "best_overall.py",
        "report.pdf", "report/main.tex", "report/references.bib",
        "report/neurips_2025.sty", "report/OFFICIAL_TEMPLATE_PROVENANCE.json",
        "report/generated/formal_results.tex",
        "report/generated/report_number_trace.json", "submission_checker.py",
        "evaluation/recovery_v2/final_manifest.json",
        "evaluation/recovery_finalize.py", "scripts/package_submission.py",
        "model_locator.py", "evaluation/portable_reproduce.py",
        "evaluation/portability_identity.py",
        "evaluation/portability/amendment_manifest.json",
        "scripts/portable_full_inference.py",
        "scripts/final_portability_smoke.py",
        "evaluation/final_fresh_evidence.py",
        "scripts/build_final_fresh_inputs.py",
        "scripts/final_fresh_reproduction.py",
        "data/final_fresh/descriptive_val.json",
        "data/final_fresh/query_images_manifest.json",
        "docs/FINAL_PORTABILITY_SMOKE.md",
        *PHASE2_CATALOG,
        *PHASE2_MANIFESTS,
    }
    missing = sorted(required - selected)
    if missing:
        raise PackageError(f"required package files were not selected: {missing}")
    derived = sorted(
        relative
        for relative in selected
        if relative.startswith(DERIVED_GENERATION_CHECKPOINT_PREFIX)
        and relative.endswith("/generation2.json")
    )
    if derived:
        raise PackageError(
            f"derived generation checkpoints must be rebuilt, not packaged: {derived}"
        )

    return _canonical_nonempty_files(project, selected, required)


def _add_phase2_restart_assets(project: Path, selected: set[str]) -> None:
    """Add the deterministic Phase-2 restart inputs and their exact images."""
    from evaluate import DEFAULT_LIMIT, load_queries
    from evolution.mutation_catalog import load_frozen_catalog
    from evolution.phase2_manifests import (
        STAGE_COUNTS,
        load_private_queries,
        validate_disjoint_manifests,
    )

    load_frozen_catalog(project / "data/phase2_mutation_catalog")
    validation = validate_disjoint_manifests(
        project / "data/phase2_local_search_v2/manifests"
    )
    if validation.get("stage_counts") != STAGE_COUNTS:
        raise PackageError("Phase-2 restart manifest counts differ")
    selected.update(PHASE2_CATALOG)
    selected.update(PHASE2_MANIFESTS)

    _, default_queries = load_queries(limit=DEFAULT_LIMIT)
    query_sets = [
        [query for _, query in default_queries],
        *[
            [
                {"figure_path": str(query.image_path)}
                for query in load_private_queries(
                    stage,
                    root=project / "data/phase2_local_search_v2/manifests",
                )
            ]
            for stage in STAGE_COUNTS
        ],
    ]
    required_images: set[str] = set()
    for queries in query_sets:
        for query in queries:
            raw = Path(str(query["figure_path"]))
            candidate = raw if raw.is_absolute() else project / raw
            try:
                relative = candidate.resolve(strict=True).relative_to(project).as_posix()
            except (FileNotFoundError, ValueError) as exc:
                raise PackageError(f"Phase-2/default query image is unavailable: {raw}") from exc
            if not relative.startswith("charxiv/images/"):
                raise PackageError(f"query image escaped the CharXiv image tree: {relative}")
            required_images.add(relative)
    if len(required_images) != 76:
        raise PackageError(
            f"Phase-2/default image closure differs: {len(required_images)} != 76"
        )

    provenance = load_json(_regular_inside(project, CHARXIV_IMAGE_PROVENANCE))
    if provenance.get("schema_version") != 1:
        raise PackageError("CharXiv image provenance version differs")
    entries = provenance.get("entries")
    if not isinstance(entries, list) or provenance.get("count") != len(entries):
        raise PackageError("CharXiv image provenance inventory is invalid")
    index: dict[str, dict[str, object]] = {}
    for record in entries:
        if not isinstance(record, dict) or set(record) != {
            "image_id", "relative_path", "sha256", "size_bytes"
        }:
            raise PackageError("CharXiv image provenance record differs")
        relative = record["relative_path"]
        if not isinstance(relative, str) or relative in index:
            raise PackageError("CharXiv image provenance path is invalid or duplicate")
        index[relative] = record
    for relative in sorted(required_images):
        if relative not in index:
            raise PackageError(f"query image lacks official provenance: {relative}")
        path = _regular_inside(project, relative)
        record = index[relative]
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise PackageError(f"query image differs from official provenance: {relative}")
        selected.add(relative)


def git_identity(project: Path) -> tuple[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            raise PackageError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return completed.stdout.strip()

    dirty = run("status", "--porcelain=v1", "--untracked-files=all", "--", ".")
    if dirty:
        raise PackageError("project worktree must be clean before packaging")
    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    return commit, tree


def base_zip_identity(
    project: Path,
    base_zip: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Authenticate the original delivered ZIP used as a repair baseline."""
    try:
        metadata = base_zip.lstat()
    except FileNotFoundError as exc:
        raise PackageError(f"base ZIP is missing: {base_zip}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PackageError("base ZIP must be one physical regular file")
    base_zip = base_zip.resolve(strict=True)
    try:
        with zipfile.ZipFile(base_zip) as archive:
            payload = archive.read("PACKAGE_MANIFEST.json")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise PackageError("base ZIP PACKAGE_MANIFEST.json is unavailable") from exc
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PackageError(f"non-finite base manifest constant: {constant}")
            ),
            object_pairs_hook=lambda pairs: _unique_manifest_pairs(
                pairs, "base PACKAGE_MANIFEST.json"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError("base ZIP PACKAGE_MANIFEST.json is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("status") != STATUS
        or value.get("formal_complete") is not True
        or value.get("missing_formal_artifacts") != []
    ):
        raise PackageError("base ZIP manifest is not a complete schema-v1 package")
    source = value.get("source")
    files = value.get("files")
    if not isinstance(source, dict) or not isinstance(files, list):
        raise PackageError("base ZIP manifest source/files are invalid")
    commit = source.get("commit")
    tree = source.get("tree")
    recovery_hash = source.get("recovery_final_manifest_sha256")
    if (
        not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
        or not isinstance(tree, str)
        or HEX40.fullmatch(tree) is None
        or not isinstance(recovery_hash, str)
        or recovery_hash != sha256_file(
            project / "evaluation/recovery_v2/final_manifest.json"
        )
    ):
        raise PackageError("base ZIP source identity differs from sealed formal evidence")
    records: dict[str, dict[str, object]] = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path", "sha256", "size_bytes"
        }:
            raise PackageError("base ZIP file record differs")
        relative = record["path"]
        if not isinstance(relative, str) or relative in records:
            raise PackageError("base ZIP file path is invalid or duplicate")
        records[relative] = dict(record)
    local_manifest = project / "PACKAGE_MANIFEST.json"
    if local_manifest.is_file() and not local_manifest.is_symlink():
        if local_manifest.read_bytes() != payload:
            local = load_json(local_manifest)
            local_source = local.get("source") if isinstance(local, dict) else None
            if (
                not isinstance(local_source, dict)
                or local_source.get("kind") != "verified_repair_from_delivered_zip"
                or local_source.get("base_zip_sha256") != sha256_file(base_zip)
                or local_source.get("base_package_manifest_sha256")
                != sha256_bytes(payload)
                or local_source.get("recovery_final_manifest_sha256")
                != recovery_hash
            ):
                raise PackageError(
                    "local PACKAGE_MANIFEST.json is neither the base nor its "
                    "verified repair descendant"
                )
    return (
        {
            "kind": "verified_repair_from_delivered_zip",
            "identity_scope": "sealed_formal_base_only",
            "commit": commit,
            "tree": tree,
            "recovery_final_manifest_sha256": recovery_hash,
            "base_zip_sha256": sha256_file(base_zip),
            "base_zip_size_bytes": base_zip.stat().st_size,
            "base_package_manifest_sha256": sha256_bytes(payload),
        },
        records,
    )


def _unique_manifest_pairs(
    pairs: list[tuple[str, object]], label: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON key in {label}: {key}")
        result[key] = value
    return result


def build_manifest(
    project: Path,
    files: list[str],
    source: dict[str, object],
    base_records: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    records = []
    for relative in files:
        path = _regular_inside(project, relative)
        records.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    source = dict(source)
    source["package_file_records_sha256"] = sha256_bytes(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": STATUS,
        "formal_complete": True,
        "missing_formal_artifacts": MISSING,
        "source": source,
        "files": records,
    }
    if base_records is not None:
        current = {str(record["path"]): record for record in records}
        omitted = sorted(set(base_records) - set(current))
        if omitted:
            raise PackageError(f"repair package omitted base members: {omitted}")
        modified: list[dict[str, object]] = []
        added: list[dict[str, object]] = []
        unchanged = 0
        for relative, record in current.items():
            base = base_records.get(relative)
            if base is None:
                added.append(dict(record))
            elif (
                record["sha256"] == base["sha256"]
                and record["size_bytes"] == base["size_bytes"]
            ):
                unchanged += 1
            else:
                modified.append(
                    {
                        "path": relative,
                        "base_sha256": base["sha256"],
                        "base_size_bytes": base["size_bytes"],
                        "package_sha256": record["sha256"],
                        "package_size_bytes": record["size_bytes"],
                    }
                )
        document["repair"] = {
            "base_file_count": len(base_records),
            "package_file_count": len(records),
            "unchanged_file_count": unchanged,
            "modified_files": sorted(modified, key=lambda item: str(item["path"])),
            "added_files": sorted(added, key=lambda item: str(item["path"])),
            "omitted_files": [],
            "statements": {
                "sealed_formal_raw_results_modified": False,
                "formal_metrics_recomputed_by_repair": False,
                "derived_generation_checkpoints_included": False,
                "purpose": (
                    "restore packaged runtime inputs and completion recovery; "
                    "preserve sealed formal evidence"
                ),
            },
        }
    return document


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_zip_no_replace(
    project: Path,
    output: Path,
    files: list[str],
    manifest: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    if not parent.is_relative_to(project):
        raise PackageError("ZIP output escaped project")
    temporary = parent / f".{output.name}.{uuid.uuid4()}.partial"
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    try:
        with zipfile.ZipFile(
            temporary, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for relative in files:
                payload = _regular_inside(project, relative).read_bytes()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if relative.endswith(".sh") else 0o644)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload, compresslevel=9)
            info = zipfile.ZipInfo("PACKAGE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, manifest_bytes, compresslevel=9)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        with output.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(parent)
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(parent)


def write_json_no_replace(path: Path, document: dict[str, Any], project: Path) -> None:
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(project):
        raise PackageError("manifest output escaped project")
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--base-zip",
        type=Path,
        help=(
            "authenticate a previously delivered ZIP as the repair baseline; "
            "required when rebuilding from an extracted package without a clean Git tree"
        ),
    )
    args = parser.parse_args(argv)
    project = (args.project or default_project()).resolve(strict=True)
    if not project.is_dir():
        raise PackageError("project root is not a directory")

    sys.path.insert(0, str(project))
    from evaluation.portable_reproduce import replay
    from submission_checker import audit_zip, check_root

    verified = replay(verify_closure=True)
    if verified.get("status") != "COMPLETE" or verified.get("result_count") != 26:
        raise PackageError("portable formal evidence replay did not verify")
    root_report = check_root(project, "final", False)
    if root_report["status"] != "pass":
        raise PackageError(f"final root checker failed: {root_report['failures']}")

    base_records: dict[str, dict[str, object]] | None = None
    if args.base_zip is not None:
        base_zip = args.base_zip
        if not base_zip.is_absolute():
            base_zip = Path.cwd() / base_zip
        base_zip = base_zip.absolute()
        base_report = audit_zip(base_zip)
        if base_report["issues"]:
            raise PackageError(f"base ZIP failed audit: {base_report['issues']}")
        source, base_records = base_zip_identity(project, base_zip)
    else:
        commit, tree = git_identity(project)
        source = {
            "kind": "clean_git_worktree",
            "identity_scope": "current_package_source",
            "commit": commit,
            "tree": tree,
            "recovery_final_manifest_sha256": sha256_file(
                project / "evaluation/recovery_v2/final_manifest.json"
            ),
        }
    files = collect_files(project)
    manifest = build_manifest(project, files, source, base_records)
    output = args.output or (project / "dist" / OUTPUT_NAME)
    if not output.is_absolute():
        output = project / output
    output = output.absolute()
    write_zip_no_replace(project, output, files, manifest)
    zip_report = audit_zip(output)
    if zip_report["issues"]:
        raise PackageError(f"built ZIP failed audit: {zip_report['issues']}")

    receipt = {
        "schema_version": 1,
        "status": STATUS,
        "formal_complete": True,
        "zip_path": output.relative_to(project).as_posix(),
        "zip_sha256": sha256_file(output),
        "zip_size_bytes": output.stat().st_size,
        "member_count": len(files) + 1,
        "source_kind": source["kind"],
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "checker_status": "pass",
        "missing_formal_artifacts": MISSING,
    }
    receipt_path = output.with_suffix(output.suffix + ".manifest.json")
    write_json_no_replace(receipt_path, receipt, project)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
