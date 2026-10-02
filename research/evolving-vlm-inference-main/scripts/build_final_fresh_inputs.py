"""Build/verify 128 independent regular image inputs for fresh reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import evaluate  # noqa: E402
DEFAULT_OUTPUT = PROJECT_ROOT / "data/final_fresh"


class InputBuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def build(data: Path, images: Path, output: Path, limit: int = 128) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise InputBuildError("fresh input output already exists")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    image_output = output / "images"
    image_output.mkdir(mode=0o700)
    _, queries = evaluate.load_queries(
        data_path=data, images_path=images, limit=limit
    )
    records = []
    content_counts: dict[str, int] = {}
    for index, (key, query) in enumerate(queries):
        source = Path(str(query["figure_path"]))
        if source.is_symlink() or not source.is_file():
            raise InputBuildError("source image is unsafe")
        digest = sha256(source)
        name = f"{index:03d}.{key}.jpg"
        target = image_output / name
        with source.open("rb") as incoming:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush(); os.fsync(outgoing.fileno())
        if sha256(target) != digest:
            raise InputBuildError("copied image hash differs")
        content_counts[digest] = content_counts.get(digest, 0) + 1
        records.append({
            "query_key": key,
            "relative_path": name,
            "size_bytes": target.stat().st_size,
            "sha256": digest,
            "source_image_sha256": digest,
        })
    data_target = output / "descriptive_val.json"
    _write(data_target, data.read_bytes())
    order_payload = "\n".join(item["query_key"] for item in records).encode()
    document = {
        "schema_version": 1,
        "sample_count": limit,
        "source_data_sha256": sha256(data),
        "packaged_data_sha256": sha256(data_target),
        "query_order_sha256": hashlib.sha256(order_payload).hexdigest(),
        "independent_regular_file_count": len(records),
        "unique_content_sha256_count": len(content_counts),
        "duplicate_content_groups": sorted(
            count for count in content_counts.values() if count > 1
        ),
        "records": records,
    }
    _write(
        output / "query_images_manifest.json",
        (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    verify(output, limit)
    return document


def verify(output: Path, limit: int = 128) -> dict[str, Any]:
    manifest = output / "query_images_manifest.json"
    value = json.loads(
        manifest.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(InputBuildError(value)),
    )
    if value.get("schema_version") != 1 or value.get("sample_count") != limit:
        raise InputBuildError("fresh input manifest identity differs")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != limit:
        raise InputBuildError("fresh input manifest count differs")
    paths = []
    for item in records:
        path = output / "images" / str(item["relative_path"])
        if path.is_symlink() or not path.is_file():
            raise InputBuildError("fresh input image is unsafe")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to((output / "images").resolve(strict=True)):
            raise InputBuildError("fresh input image escaped")
        if resolved in paths or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise InputBuildError("fresh input file identity differs")
        paths.append(resolved)
    if len(paths) != limit:
        raise InputBuildError("fresh input physical file count differs")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if args.verify:
        result = verify(output)
    else:
        result = build(
            PROJECT_ROOT / "charxiv/data/descriptive_val.json",
            PROJECT_ROOT / "charxiv/images",
            output,
        )
    print(json.dumps({
        "sample_count": result["sample_count"],
        "query_order_sha256": result["query_order_sha256"],
        "unique_content_sha256_count": result["unique_content_sha256_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
