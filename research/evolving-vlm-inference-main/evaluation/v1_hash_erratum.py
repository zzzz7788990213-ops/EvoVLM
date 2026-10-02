"""Verify the immutable Phase-3-v1 campaign-plan hash erratum."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

ERRATUM_RELATIVE = "evaluation/recovery_v2/phase3_v1_plan_hash_erratum.json"
INCORRECT = "effe1b6f06a78a3544444248667f9e5c5f941633424299475121cb329f526f567"
ACTUAL = "effe1b6a1741885b2ab4c48c71efeedc1735825039d837ea0533b79f13258c79"


class ErratumError(RuntimeError):
    pass


def _reject(value: str) -> object:
    raise ErratumError(f"non-finite JSON constant: {value}")


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ErratumError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject,
        object_pairs_hook=_pairs,
    )
    if not isinstance(value, dict):
        raise ErratumError("erratum JSON root is not an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(root: Path, record: Mapping[str, object]) -> Path:
    if set(record) != {"path", "sha256", "size_bytes", "locator"}:
        raise ErratumError("declaration record fields differ")
    path = root / str(record["path"])
    if path.is_symlink() or not path.is_file():
        raise ErratumError("declaration artifact is unsafe")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ErratumError("declaration artifact escaped root")
    if sha256(resolved) != record["sha256"] or resolved.stat().st_size != record["size_bytes"]:
        raise ErratumError("declaration artifact hash/size differs")
    return resolved


def verify(root: Path, erratum_path: Path | None = None) -> dict[str, object]:
    root = root.resolve(strict=True)
    path = root / ERRATUM_RELATIVE if erratum_path is None else erratum_path
    document = load(path)
    required = {
        "schema_version", "status", "created_base_commit", "discovered_at_utc",
        "incorrect_declared_sha256", "actual_campaign_plan",
        "declarations", "formal_metrics_affected", "selection_affected",
        "explanation",
    }
    if set(document) != required:
        raise ErratumError("erratum fields differ")
    if document["schema_version"] != 1 or document["status"] != "ERRATUM_VERIFIED":
        raise ErratumError("erratum status/version differs")
    if document["incorrect_declared_sha256"] != INCORRECT:
        raise ErratumError("incorrect declaration value differs")
    if document["formal_metrics_affected"] is not False or document["selection_affected"] is not False:
        raise ErratumError("erratum must not alter metrics or selection")
    plan_record = document["actual_campaign_plan"]
    if not isinstance(plan_record, dict) or set(plan_record) != {"path", "sha256", "size_bytes"}:
        raise ErratumError("actual plan record fields differ")
    plan = root / str(plan_record["path"])
    if plan.is_symlink() or not plan.is_file() or not plan.resolve(strict=True).is_relative_to(root):
        raise ErratumError("actual plan path is unsafe")
    if plan_record["sha256"] != ACTUAL or sha256(plan) != ACTUAL or plan.stat().st_size != plan_record["size_bytes"]:
        raise ErratumError("actual campaign plan hash/size differs")
    declarations = document["declarations"]
    if not isinstance(declarations, list) or len(declarations) != 2:
        raise ErratumError("declaration inventory differs")
    seen: set[str] = set()
    for record in declarations:
        if not isinstance(record, dict):
            raise ErratumError("declaration record is not an object")
        artifact = _file(root, record)
        locator = record["locator"]
        if not isinstance(locator, dict) or set(locator) != {"kind", "value"}:
            raise ErratumError("declaration locator fields differ")
        rel = str(record["path"])
        if rel in seen:
            raise ErratumError("duplicate declaration artifact")
        seen.add(rel)
        if locator["kind"] == "markdown_exact_line":
            line = str(locator["value"])
            text = artifact.read_text(encoding="utf-8")
            if text.count(line) != 1 or INCORRECT not in line:
                raise ErratumError("Markdown declaration locator differs")
        elif locator["kind"] == "json_pointer":
            if locator["value"] != "/recovery_of/campaign_plan_sha256":
                raise ErratumError("JSON declaration pointer differs")
            value = load(artifact).get("recovery_of", {}).get("campaign_plan_sha256")
            if value != INCORRECT:
                raise ErratumError("JSON declaration value differs")
        else:
            raise ErratumError("unknown declaration locator")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document["created_base_commit"])):
        raise ErratumError("created base commit is invalid")
    return {
        "status": "ERRATUM_VERIFIED",
        "actual_campaign_plan_sha256": ACTUAL,
        "incorrect_declaration_sha256": INCORRECT,
        "declaration_count": len(seen),
        "erratum_sha256": sha256(path),
    }
