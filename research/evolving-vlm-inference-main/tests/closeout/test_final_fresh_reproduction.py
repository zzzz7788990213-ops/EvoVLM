"""CPU-only scheduling/validation tests for final fresh reproduction."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
import sys
from pathlib import Path

import pytest

from scripts import final_fresh_reproduction as fresh


def test_final_fresh_plan_has_exact_frozen_scope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fresh, "_program_plan", lambda program, snapshots: {
        "program": program, "program_source_sha256": (
            "a" * 64 if program in {
                "evolved_instruct", "best_accuracy", "best_speed", "best_overall"
            } else (program.encode().hex() + "0" * 64)[:64]
        ),
    })
    monkeypatch.setattr(fresh, "verify_inputs", lambda root: {
        "sample_count": 128, "query_order_sha256": "b" * 64,
        "independent_regular_file_count": 128,
    })
    monkeypatch.setattr(fresh, "_sha", lambda path: "c" * 64)
    monkeypatch.setattr(fresh, "FORMAL_MANIFEST", Path(__file__))
    snapshots = {
        "Qwen/fixed": {
            "manifest_path": "evaluation/model_snapshots/fixed.json",
            "manifest_sha256": "f" * 64, "manifest_size_bytes": 1,
            "revision": "1" * 40, "processor_revision": "1" * 40,
            "weight_layout": "single_safetensors", "runtime_files": [],
        }
    }
    plan = fresh.build_plan(tmp_path / str(uuid.uuid4()), {
        "commit": "d" * 40, "tree": "e" * 40,
    }, snapshots)
    assert plan["model_snapshots"]["Qwen/fixed"]["manifest"]["sha256"] == "f" * 64
    assert plan["speed32_executed"] is False
    assert plan["evolution_executed"] is False
    assert [item["program"] for item in plan["lanes"]["0"]] == [
        "baseline_instruct", "manual_instruct", "evolved_instruct",
        "best_accuracy", "best_speed", "best_overall",
    ]
    assert [item["program"] for item in plan["lanes"]["1"]] == [
        "manual_thinking", "evolved_thinking", "baseline_thinking",
    ]


def test_source_identity_requires_real_commit_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINAL_FRESH_SOURCE_COMMIT", raising=False)
    monkeypatch.delenv("FINAL_FRESH_SOURCE_TREE", raising=False)
    monkeypatch.setattr(fresh, "PROJECT_ROOT", Path("/nonexistent"))
    with pytest.raises(fresh.FreshRunError):
        fresh._source_identity()


def test_default_output_is_unique_and_managed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVOLVING_VLM_RUNTIME_ROOT", str(tmp_path))
    first = fresh._output(None); second = fresh._output(None)
    assert first != second
    assert first.is_relative_to(tmp_path) and second.is_relative_to(tmp_path)


def test_lane_order_is_fixed_and_official_evaluator_command_is_used() -> None:
    source = Path(fresh.__file__).read_text()
    assert 'str(PROJECT_ROOT / "evaluate.py"), program' in source
    assert '"speed32"' not in repr(fresh.LANES)
    assert fresh.EXPECTED == {
        "baseline_instruct": 51, "manual_instruct": 78,
        "manual_thinking": 23, "evolved_instruct": 78,
        "evolved_thinking": 44,
    }


def test_reproduce_no_argument_routes_to_fresh_gpu_reproduction() -> None:
    source = (fresh.PROJECT_ROOT / "reproduce.sh").read_text()
    assert 'readonly MODE="${1:-fresh}"' in source
    assert '"$PYTHON" -B "$PROJECT/scripts/final_fresh_reproduction.py"' in source


def test_reproduce_no_argument_executes_fresh_scheduler_without_gpu(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    script = package / "reproduce.sh"
    script.write_bytes((fresh.PROJECT_ROOT / "reproduce.sh").read_bytes())
    script.chmod(0o700)
    fake = package / "fake-python"
    fake.write_text("""#!/bin/sh\nprintf '%s\\n' "$@" > "$TRACE"\n""")
    fake.chmod(0o700)
    trace = package / "trace.txt"
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(fake), "TRACE": str(trace),
        "EVOLVING_VLM_RUNTIME_ROOT": str(package / "runtime"),
    })
    completed = subprocess.run(
        ["bash", str(script)], cwd=package, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert trace.read_text().splitlines() == [
        "-B", str(package / "scripts/final_fresh_reproduction.py"),
    ]



def test_run_lane_cpu_fake_uses_official_entry_and_binds_attempt_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / str(uuid.uuid4())
    output.mkdir()
    program = "baseline_instruct"
    monkeypatch.setattr(fresh, "LANES", {0: ((program, "dev128", 128),)})
    plan = {
        "run_id": output.name,
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "programs": {program: {
            "program_source_sha256": "c" * 64,
            "dependency_closure_sha256": "d" * 64,
            "model_id": "Qwen/fixed", "model_revision": "e" * 40,
            "processor_revision": "e" * 40, "model_weight_sha256": "f" * 64,
        }},
        "input_manifest": {"sha256": "1" * 64},
    }
    fresh._write(output / "plan.json", plan)
    commands: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        result_path = Path(command[command.index("--output") + 1])
        result_path.write_text("{}\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fresh, "validate_result", lambda *args, **kwargs: {
        "document": {"total_time": 2.0, "avg_time_per_query": 2.0 / 128},
        "correct": 51, "tokens": 128,
    })
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    receipt = fresh.run_lane(output, 0, runner=fake_runner)
    assert len(commands) == 1
    assert commands[0][2] == str(fresh.PROJECT_ROOT / "evaluate.py")
    assert commands[0][3] == program
    result = receipt["results"][0]
    assert set(result) >= {"result", "console", "start", "end", "attempt_id"}
    for field in ("result", "console", "start", "end"):
        assert (output / result[field]["path"]).is_file()
    assert (output / "lane0/receipt.json").is_file()



def test_snapshot_failure_precedes_output_spawn_cuda_and_heavy_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    for name in ("torch", "transformers", "evaluate"):
        sys.modules.pop(name, None)
    calls: list[str] = []

    def fail_snapshots() -> dict[str, object]:
        calls.append("snapshots")
        raise fresh.SnapshotError("missing thinking snapshot")

    def forbidden_output(value: str | None) -> Path:
        calls.append("output")
        raise AssertionError("_output ran before snapshot preflight")

    def forbidden_spawn(*args: object, **kwargs: object) -> object:
        calls.append("spawn")
        raise AssertionError("spawn ran before snapshot preflight")

    monkeypatch.setattr(fresh, "validate_both_snapshots", fail_snapshots)
    monkeypatch.setattr(fresh, "_output", forbidden_output)
    monkeypatch.setattr(fresh.subprocess, "Popen", forbidden_spawn)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preflight-sentinel")
    output = tmp_path / "must-not-exist"
    assert fresh.main(["--output", str(output)]) == 2
    assert calls == ["snapshots"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "preflight-sentinel"
    assert not output.exists()
    assert all(name not in sys.modules for name in ("torch", "transformers", "evaluate"))


def test_successful_dry_run_orders_dual_snapshot_preflight_before_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    order: list[str] = []
    snapshots = {
        "Qwen/Qwen3-VL-2B-Instruct": {"validated": True},
        "Qwen/Qwen3-VL-2B-Thinking": {"validated": True},
    }

    def preflight() -> dict[str, object]:
        order.append("snapshots")
        return snapshots

    def output_path(value: str | None) -> Path:
        order.append("output")
        assert value is not None
        return tmp_path / value

    def plan(
        output: Path, source: object, checked: object,
    ) -> dict[str, object]:
        order.append("plan")
        assert checked is snapshots
        return {
            "run_id": str(uuid.uuid4()), "lanes": {},
            "input_manifest": {"sha256": "a" * 64},
        }

    monkeypatch.setattr(fresh, "validate_both_snapshots", preflight)
    monkeypatch.setattr(fresh, "_output", output_path)
    monkeypatch.setattr(
        fresh, "_source_identity",
        lambda: {"commit": "a" * 40, "tree": "b" * 40},
    )
    monkeypatch.setattr(fresh, "build_plan", plan)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preflight-sentinel")
    assert fresh.main(["--dry-run", "--output", "dry"]) == 0
    assert order == ["snapshots", "output", "plan"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "preflight-sentinel"
    assert not (tmp_path / "dry").exists()



def test_snapshot_failure_precedes_output_spawn_cuda_and_heavy_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    for name in ("torch", "transformers", "evaluate"):
        sys.modules.pop(name, None)
    calls: list[str] = []

    def fail_snapshots() -> dict[str, object]:
        calls.append("snapshots")
        raise fresh.SnapshotError("missing thinking snapshot")

    def forbidden_output(value: str | None) -> Path:
        calls.append("output")
        raise AssertionError("_output ran before snapshot preflight")

    def forbidden_spawn(*args: object, **kwargs: object) -> object:
        calls.append("spawn")
        raise AssertionError("spawn ran before snapshot preflight")

    monkeypatch.setattr(fresh, "validate_both_snapshots", fail_snapshots)
    monkeypatch.setattr(fresh, "_output", forbidden_output)
    monkeypatch.setattr(fresh.subprocess, "Popen", forbidden_spawn)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preflight-sentinel")
    output = tmp_path / "must-not-exist"
    assert fresh.main(["--output", str(output)]) == 2
    assert calls == ["snapshots"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "preflight-sentinel"
    assert not output.exists()
    assert all(name not in sys.modules for name in ("torch", "transformers", "evaluate"))


def test_successful_dry_run_orders_dual_snapshot_preflight_before_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    order: list[str] = []
    snapshots = {
        "Qwen/Qwen3-VL-2B-Instruct": {"validated": True},
        "Qwen/Qwen3-VL-2B-Thinking": {"validated": True},
    }

    def preflight() -> dict[str, object]:
        order.append("snapshots")
        return snapshots

    def output_path(value: str | None) -> Path:
        order.append("output")
        assert value is not None
        return tmp_path / value

    def plan(
        output: Path, source: object, checked: object,
    ) -> dict[str, object]:
        order.append("plan")
        assert checked is snapshots
        return {
            "run_id": str(uuid.uuid4()), "lanes": {},
            "input_manifest": {"sha256": "a" * 64},
        }

    monkeypatch.setattr(fresh, "validate_both_snapshots", preflight)
    monkeypatch.setattr(fresh, "_output", output_path)
    monkeypatch.setattr(
        fresh, "_source_identity",
        lambda: {"commit": "a" * 40, "tree": "b" * 40},
    )
    monkeypatch.setattr(fresh, "build_plan", plan)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preflight-sentinel")
    assert fresh.main(["--dry-run", "--output", "dry"]) == 0
    assert order == ["snapshots", "output", "plan"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "preflight-sentinel"
    assert not (tmp_path / "dry").exists()
