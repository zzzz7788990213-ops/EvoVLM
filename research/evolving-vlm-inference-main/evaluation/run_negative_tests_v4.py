#!/usr/bin/env python3
"""Run safe active provenance tamper probes against Evo-internal copies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from evaluation.active_paths import ACTIVE_VERSION_TAG, LATEST_PATH
from evaluation.atomic_artifacts import write_json_noreplace
from evaluation.provenance import EVO_ROOT, PROJECT_ROOT, load_json_strict, sha256_file

LATEST = PROJECT_ROOT / LATEST_PATH
KINDS = ("gt", "image", "model", "processor", "schema", "manifest", "evaluator", "raw")


def main() -> int:
    latest = load_json_strict(LATEST)
    campaign = PROJECT_ROOT / latest["campaign_path"]
    plan = load_json_strict(campaign / "campaign_plan.json")
    summary = campaign / plan["runs"][0]["output_filename"]
    rows = []
    for kind in KINDS:
        command = [sys.executable, "-m", "evaluation.negative_probe", "--kind", kind]
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
    document = {
        "schema_version": 4,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": latest["campaign_id"],
        "fixture_mode": "safe_evo_internal_copies",
        "summary_path_relative_to_evo": summary.relative_to(EVO_ROOT).as_posix(),
        "summary_sha256": sha256_file(summary),
        "formal_inputs_unchanged": True,
        "tests": rows,
    }
    output = EVO_ROOT / "logs" / f"negative_provenance_tests_{ACTIVE_VERSION_TAG}.{uuid4()}.json"
    write_json_noreplace(output, document)
    print(
        json.dumps(
            {
                "status": status,
                "output": output.relative_to(EVO_ROOT).as_posix(),
                "output_sha256": sha256_file(output),
                "tests": rows,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
