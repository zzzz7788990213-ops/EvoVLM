"""Run the four non-formal final portability single-query smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.portability_identity import (
    PortabilityIdentityError,
    compare_fresh_single,
    load_old_single_reference,
)

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
LABEL = "FINAL_PORTABILITY_SMOKE_ONLY"
LANES = {
    0: ("manual_instruct", "evolved_instruct"),
    1: ("manual_thinking", "evolved_thinking"),
}


class PortabilitySmokeError(RuntimeError):
    """A final non-formal portability smoke invariant failed."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise PortabilitySmokeError(f"JSON root is not an object: {path}")
    return value


def _write_no_replace(path: Path, document: Mapping[str, object]) -> None:
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _input_file(value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise PortabilitySmokeError(f"data input is not a physical file: {path}")
    return path.resolve(strict=True)


def _input_directory(value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise PortabilitySmokeError(f"image input is not a physical directory: {path}")
    return path.resolve(strict=True)


def _output_root(value: str) -> Path:
    managed = Path(
        os.environ.get("EVOLVING_VLM_RUNTIME_ROOT", str(PROJECT_ROOT))
    ).expanduser().resolve(strict=False)
    output = Path(value).expanduser()
    if not output.is_absolute():
        output = managed / output
    output = output.resolve(strict=False)
    if output == managed or not _inside(output, managed):
        raise PortabilitySmokeError("output must be a child of the managed runtime root")
    return output

def _sealed_input_records(images: Path) -> list[dict[str, object]]:
    records = []
    for programs in LANES.values():
        for program in programs:
            reference = load_old_single_reference(program)
            query_key = str(reference.get("query_key", ""))
            parts = query_key.rsplit("_", 1)
            if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
                raise PortabilitySmokeError(
                    f"sealed Single1 query key differs: {program}"
                )
            image = images / f"{parts[0]}.jpg"
            verified = load_old_single_reference(program, image_path=image)
            records.append({
                "program": program,
                "query_key": query_key,
                "question_sha256": verified["question_sha256"],
                "image_path": str(image.resolve(strict=True)),
                "image_sha256": verified["image_sha256"],
                "model_id": verified["model_id"],
                "model_revision": verified["model_revision"],
                "processor_revision": verified["processor_revision"],
                "model_weight_sha256": verified["weight_sha256"],
                "algorithm_identity_sha256": verified[
                    "algorithm_identity_sha256"
                ],
            })
    return records


def _lane_environment(output: Path, lane: int) -> dict[str, str]:
    if lane not in LANES:
        raise PortabilitySmokeError("unknown GPU lane")
    runtime = output / f"lane{lane}.runtime"
    runtime.mkdir(mode=0o700, exist_ok=False)
    cache = runtime / "cache"
    locations = {
        "TMPDIR": runtime / "tmp",
        "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "PIP_CACHE_DIR": cache / "pip",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "VLLM_CACHE_ROOT": cache / "vllm",
    }
    environment = os.environ.copy()
    for key, path in locations.items():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        environment[key] = str(path)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(lane)
    return environment



def _result_record(output: Path, program: str, physical_gpu: int) -> dict[str, object]:
    path = output / f"{program}.json"
    if path.is_symlink() or not path.is_file():
        raise PortabilitySmokeError(f"missing result: {program}")
    document = _load(path)
    scope = document.get("gpu_scope")
    if (
        document.get("program") != program
        or document.get("num_evaluated") != 1
        or document.get("num_errors") != 0
        or not isinstance(scope, dict)
        or scope.get("physical_gpu_index") != physical_gpu
        or scope.get("cuda_visible_devices") != str(physical_gpu)
        or scope.get("visible_device_count") != 1
        or scope.get("logical_device") != "cuda:0"
    ):
        raise PortabilitySmokeError(f"result identity/GPU facts differ: {program}")
    source = PROJECT_ROOT / f"{program}.py"
    comparison = compare_fresh_single(program, path)
    return {
        "program": program,
        "physical_gpu_index": physical_gpu,
        "result_path": f"lane{physical_gpu}/{path.name}",
        "result_sha256": _sha256(path),
        "source_sha256": _sha256(source),
        "num_errors": 0,
        "total_time": document.get("total_time"),
        "sealed_single_comparison": comparison,
    }


def run_lane(
    *, output: Path, lane: int, data: Path, images: Path,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    if lane not in LANES or os.environ.get("CUDA_VISIBLE_DEVICES") != str(lane):
        raise PortabilitySmokeError("lane does not match its single physical GPU")
    lane_output = output / f"lane{lane}"
    lane_output.mkdir(mode=0o700, exist_ok=False)
    records = []
    for program in LANES[lane]:
        command = [
            sys.executable, "-B", str(PROJECT_ROOT / "evaluate.py"), program,
            "--limit", "1", "--data", str(data), "--images", str(images),
            "--output", str(lane_output / f"{program}.json"),
        ]
        result = runner(command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=False)
        if result.returncode != 0:
            raise PortabilitySmokeError(f"{program} failed with rc={result.returncode}")
        records.append(_result_record(lane_output, program, lane))
    receipt = {"label": LABEL, "lane": lane, "formal_metrics_affected": False, "results": records}
    _write_no_replace(lane_output / "receipt.json", receipt)
    return receipt


def orchestrate(
    *, output: Path, data: Path, images: Path,
    sealed_inputs: Sequence[Mapping[str, object]],
    popen: Any = subprocess.Popen,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise PortabilitySmokeError("output already exists; refusing overwrite/resume")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    plan = {
        "project_root": str(PROJECT_ROOT),
        "sealed_inputs": [dict(item) for item in sealed_inputs],
        "label": LABEL,
        "run_id": run_id,
        "formal_metrics_affected": False,
        "lanes": {str(key): list(value) for key, value in LANES.items()},
    }
    _write_no_replace(output / "plan.json", plan)
    children = []
    log_handles = []
    failures: list[tuple[int, int]] = []
    try:
        for lane in sorted(LANES):
            environment = _lane_environment(output, lane)
            log = (output / f"lane{lane}.console.log").open("xb")
            log_handles.append(log)
            command = [
                sys.executable, "-B", "-m", "scripts.final_portability_smoke",
                "--lane", str(lane), "--output", str(output),
                "--data", str(data), "--images", str(images),
            ]
            children.append((lane, popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )))
        failures = [(lane, child.wait()) for lane, child in children]
    finally:
        for handle in log_handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    failed = [(lane, code) for lane, code in failures if code != 0]
    if failed:
        raise PortabilitySmokeError(f"GPU lane failure: {failed}")
    lanes = [_load(output / f"lane{lane}" / "receipt.json") for lane in sorted(LANES)]
    results = [record for lane in lanes for record in lane["results"]]
    receipt = {
        "label": LABEL,
        "run_id": run_id,
        "formal_metrics_affected": False,
        "result_count": len(results),
        "results": results,
        "completed_unix_seconds": time.time(),
    }
    _write_no_replace(output / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--images")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lane", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = _output_root(args.output)
        data = _input_file(args.data, PROJECT_ROOT / "charxiv/data/descriptive_val.json")
        images = _input_directory(args.images, PROJECT_ROOT / "charxiv/images")
        sealed_inputs = _sealed_input_records(images)
        if args.dry_run:
            if output.exists() or output.is_symlink():
                raise PortabilitySmokeError("output already exists")
            print(json.dumps({"label": LABEL, "output": str(output), "lanes": LANES,
                              "sealed_inputs": sealed_inputs}, sort_keys=True))
        elif args.lane is None:
            print(json.dumps(orchestrate(output=output, data=data, images=images,
                                         sealed_inputs=sealed_inputs), sort_keys=True))
        else:
            print(json.dumps(run_lane(output=output, lane=args.lane, data=data, images=images), sort_keys=True))
    except (
        OSError,
        ValueError,
        PortabilityIdentityError,
        PortabilitySmokeError,
    ) as exc:
        print(f"final portability smoke failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
