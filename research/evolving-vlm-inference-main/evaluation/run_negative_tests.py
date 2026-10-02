#!/usr/bin/env python3
"""Run all eight safe v3 provenance tamper probes and record exit statuses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluation.provenance import (
    EVO_ROOT,
    PROJECT_ROOT,
    load_json_strict,
    sha256_file,
    write_json_atomic,
)

LATEST = PROJECT_ROOT / "results/evaluations/v3/latest_campaign.json"
OUTPUT = EVO_ROOT / "logs/negative_provenance_tests_v3.json"
KINDS = ("gt", "image", "model", "processor", "schema", "manifest", "evaluator", "raw")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    fixture_root: Path | None = None
    if arguments.preflight:
        fixture_root = (
            EVO_ROOT / ".tmp/provenance-negative-tests" / f"preflight-{uuid4().hex}"
        )
        fixture_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        synthetic_raw = fixture_root / "synthetic.raw.jsonl"
        descriptor = os.open(
            synthetic_raw,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b'{"phase":"synthetic-preflight"}\n')
            stream.flush()
            os.fsync(stream.fileno())
        summary = fixture_root / "synthetic.summary.json"
        write_json_atomic(
            summary,
            {
                "raw_log": str(synthetic_raw),
                "provenance": {"raw_log_sha256": sha256_file(synthetic_raw)},
            },
        )
        campaign_id = f"pre-gpu-synthetic-{uuid4()}"
        output = EVO_ROOT / "logs/negative_provenance_tests_v3_pre_gpu.json"
        fixture_mode = "pre_gpu_synthetic_raw"
    else:
        latest = load_json_strict(LATEST)
        campaign = PROJECT_ROOT / latest["campaign_path"]
        plan = load_json_strict(campaign / "campaign_plan.json")
        summary = campaign / plan["runs"][0]["output_filename"]
        campaign_id = latest["campaign_id"]
        output = OUTPUT
        fixture_mode = "latest_v3_campaign"
    summary_sha256 = sha256_file(summary)
    rows = []
    for kind in KINDS:
        command = [
            sys.executable,
            "-m",
            "evaluation.negative_probe",
            "--kind",
            kind,
        ]
        if kind == "raw":
            command.extend(["--summary", str(summary)])
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        parsed = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        rows.append(
            {
                "kind": kind,
                "expected_exit_status": 2,
                "actual_exit_status": completed.returncode,
                "detected": completed.returncode == 2,
                "probe_output": parsed,
            }
        )
    status = "pass" if all(row["detected"] for row in rows) else "fail"
    if fixture_root is not None:
        synthetic_raw.unlink()
        summary.unlink()
        fixture_root.rmdir()
    document = {
        "schema_version": 3,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "fixture_mode": fixture_mode,
        "summary_path_relative_to_evo": summary.relative_to(EVO_ROOT).as_posix(),
        "summary_sha256": summary_sha256,
        "synthetic_fixture_removed": fixture_root is not None,
        "safe_fixture_root_relative_to_evo": ".tmp/provenance-negative-tests",
        "formal_inputs_unchanged": True,
        "tests": rows,
    }
    write_json_atomic(output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
