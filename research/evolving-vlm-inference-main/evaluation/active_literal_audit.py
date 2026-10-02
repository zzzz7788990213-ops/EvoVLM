"""AST and byte-level guard against stale protocol literals in active code."""

from __future__ import annotations

import ast
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Iterable

from evaluation.active_paths import ACTIVE_VERSION_TAG

HISTORICAL_PREFIX_ALLOWLIST = (
    "evaluation/archive/",
)
HISTORICAL_EXACT_ALLOWLIST = frozenset(
    {
        "evaluation/build_freeze_v3.py",
        "evaluation/check_existing_v3.py",
        "evaluation/reproduce_v3.py",
        "evaluation/verify_historical_v3.py",
    }
)


def legacy_tokens() -> tuple[str, ...]:
    underscore = tuple(f"v4_{minor}" for minor in range(1, 4))
    dotted = tuple(f"v4.{minor}" for minor in range(1, 4))
    protocols = tuple(f"gate1-exact-match-v4.{minor}" for minor in range(1, 4))
    return underscore + dotted + protocols


def is_historical_allowlisted(relative: str) -> bool:
    return relative in HISTORICAL_EXACT_ALLOWLIST or any(
        relative.startswith(prefix) for prefix in HISTORICAL_PREFIX_ALLOWLIST
    )


def _safe_regular(project_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"literal-audit path is not canonical: {relative}")
    path = project_root / relative
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"literal-audit input is not a regular file: {relative}")
    if not path.resolve(strict=True).is_relative_to(project_root.resolve(strict=True)):
        raise ValueError(f"literal-audit input escaped project root: {relative}")
    return path


def audit_active_version_literals(
    project_root: Path,
    relative_paths: Iterable[str],
) -> dict[str, object]:
    tokens = legacy_tokens()
    if ACTIVE_VERSION_TAG in tokens:
        raise AssertionError("active version cannot be classified as legacy")
    scanned: list[dict[str, object]] = []
    allowed_historical: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for relative in sorted(set(relative_paths)):
        path = _safe_regular(project_root, relative)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        byte_hits = sorted(token for token in tokens if token.encode("utf-8") in data)
        ast_hits: list[str] = []
        if path.suffix == ".py":
            tree = ast.parse(data, filename=relative)
            ast_hits = sorted(
                {
                    token
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    for token in tokens
                    if token in node.value
                }
            )
        record = {
            "path": relative,
            "sha256": digest,
            "size_bytes": len(data),
            "byte_hits": byte_hits,
            "ast_string_hits": ast_hits,
        }
        if is_historical_allowlisted(relative):
            allowed_historical.append(record)
        else:
            scanned.append(record)
            if byte_hits or ast_hits:
                violations.append(record)
    if violations:
        rendered = ", ".join(
            f"{item['path']}:{'/'.join(item['byte_hits'])}" for item in violations
        )
        raise AssertionError(f"legacy active-version literals detected: {rendered}")
    return {
        "schema_version": 1,
        "status": "pass",
        "active_version_tag": ACTIVE_VERSION_TAG,
        "legacy_token_count": len(tokens),
        "active_file_count": len(scanned),
        "historical_allowlisted_file_count": len(allowed_historical),
        "historical_prefix_allowlist": list(HISTORICAL_PREFIX_ALLOWLIST),
        "historical_exact_allowlist": sorted(HISTORICAL_EXACT_ALLOWLIST),
        "active_files": scanned,
        "historical_allowlisted_files": allowed_historical,
    }
