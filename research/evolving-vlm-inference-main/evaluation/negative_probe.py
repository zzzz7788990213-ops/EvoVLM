#!/usr/bin/env python3
"""One safe tamper probe. Exit 2 means the changed byte was detected."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from evaluation.provenance import (
    EVO_ROOT,
    GT_PATH,
    IMAGE_MANIFEST_PATH,
    PROJECT_ROOT,
    load_freeze,
    load_json_strict,
)


def source_and_expected(kind: str, summary_path: Path | None) -> tuple[Path, str]:
    freeze = load_freeze()
    if kind == "gt":
        return GT_PATH, freeze["dataset"]["ground_truth_sha256"]
    if kind == "image":
        manifest = load_json_strict(IMAGE_MANIFEST_PATH)
        entry = manifest["entries"][0]
        return PROJECT_ROOT / entry["relative_path"], entry["sha256"]
    if kind in {"model", "processor"}:
        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        filename = "config.json" if kind == "model" else "preprocessor_config.json"
        root = EVO_ROOT / freeze["models"][model_id]["snapshot_relative_to_evo"]
        return root / filename, freeze["models"][model_id]["required_file_sha256"][filename]
    if kind == "schema":
        return (
            PROJECT_ROOT / freeze["result_schema_path"],
            freeze["result_schema_sha256"],
        )
    if kind == "manifest":
        item = freeze["manifests"]["dev_128"]
        return PROJECT_ROOT / item["path"], item["sha256"]
    if kind == "evaluator":
        path = PROJECT_ROOT / "evaluate.py"
        return path, freeze["evaluator_dependency_sha256"]["evaluate.py"]
    if kind == "raw":
        if summary_path is None:
            raise ValueError("--summary is required for raw probe")
        summary = load_json_strict(summary_path)
        return (
            PROJECT_ROOT / summary["raw_log"],
            summary["provenance"]["raw_log_sha256"],
        )
    raise ValueError(f"unknown kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("gt", "image", "model", "processor", "schema", "manifest", "evaluator", "raw"),
    )
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()

    source, expected = source_and_expected(arguments.kind, arguments.summary)
    root = EVO_ROOT / ".tmp/provenance-negative-tests"
    root.mkdir(parents=True, exist_ok=True)
    sandbox = root / uuid4().hex
    sandbox.mkdir(mode=0o700)
    copy = sandbox / source.name
    try:
        shutil.copyfile(source, copy)
        with copy.open("ab") as stream:
            stream.write(b"\nGATE1_SAFE_TAMPER\n")
            stream.flush()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.verify_results",
                "--probe-path",
                str(copy),
                "--probe-expected-sha256",
                expected,
                "--probe-label",
                arguments.kind,
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(
            json.dumps(
                {
                    "kind": arguments.kind,
                    "expected_detection": True,
                    "detected": completed.returncode == 2,
                    "validator_exit_status": completed.returncode,
                    "validator_output": completed.stdout.strip(),
                    "safe_copy_relative_to_evo": copy.relative_to(EVO_ROOT).as_posix(),
                },
                sort_keys=True,
            )
        )
        return completed.returncode
    finally:
        if copy.exists():
            copy.unlink()
        sandbox.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
