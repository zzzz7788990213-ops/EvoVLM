"""Minimal official evaluator for submitted vlm_inference programs.

Usage:
    python evaluate.py PROGRAM

The only required argument is the importable program name. By default the
first 128 descriptive queries are evaluated in the order in
charxiv/data/descriptive_val.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
EVO_ROOT = Path(
    os.environ.get("EVOLVING_VLM_RUNTIME_ROOT", str(PROJECT_ROOT))
).expanduser().resolve(strict=False)
CHARXIV_ROOT = PROJECT_ROOT / "charxiv"
DATA_PATH = CHARXIV_ROOT / "data" / "descriptive_val.json"
IMAGES_PATH = CHARXIV_ROOT / "images"
DEFAULT_LIMIT = 128

# CharXiv uses this misspelled public helper name.
sys.path.insert(0, str(CHARXIV_ROOT / "src"))
from descriptive_utils import build_descriptive_quries  # noqa: E402


class EvaluationError(RuntimeError):
    """Raised when the evaluator cannot safely start or finish."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a JSON object: {path}")
    return value


def load_queries(
    *,
    data_path: Path = DATA_PATH,
    images_path: Path = IMAGES_PATH,
    limit: int = DEFAULT_LIMIT,
    query_image_manifest: Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Load the first limit descriptive queries in source-file order."""
    if limit < 1:
        raise EvaluationError("--limit must be positive")
    ground_truth = _load_json_object(data_path)
    queries = build_descriptive_quries(ground_truth, str(images_path))
    selected = list(queries.items())[:limit]
    if len(selected) != limit:
        raise EvaluationError(
            f"requested {limit} queries but only {len(selected)} are available"
        )
    if query_image_manifest is not None:
        selected = apply_query_image_manifest(
            selected,
            images_path=images_path,
            manifest_path=query_image_manifest,
        )
    return ground_truth, selected




def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_query_image_manifest(
    selected: list[tuple[str, dict[str, Any]]],
    *,
    images_path: Path,
    manifest_path: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Bind each query to its own regular, hash-authenticated image copy."""
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise EvaluationError("query-image manifest version differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) < len(selected):
        raise EvaluationError("query-image manifest count differs")
    records = records[:len(selected)]
    keys = [key for key, _ in selected]
    if [item.get("query_key") for item in records if isinstance(item, dict)] != keys:
        raise EvaluationError("query-image manifest order differs")
    result: list[tuple[str, dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    root = images_path.resolve(strict=True)
    for (key, query), record in zip(selected, records):
        if not isinstance(record, dict) or set(record) != {
            "query_key", "relative_path", "size_bytes", "sha256", "source_image_sha256"
        }:
            raise EvaluationError("query-image manifest record fields differ")
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluationError("query-image manifest path is unsafe")
        path = images_path / relative
        if path.is_symlink() or not path.is_file():
            raise EvaluationError("query image is not a regular file")
        resolved = path.resolve(strict=True)
        if not _inside(resolved, root) or resolved in seen_paths:
            raise EvaluationError("query image escaped or was reused")
        seen_paths.add(resolved)
        if (
            path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
            or record["sha256"] != record["source_image_sha256"]
        ):
            raise EvaluationError("query image hash/size differs")
        updated = dict(query)
        updated["figure_path"] = str(resolved)
        result.append((key, updated))
    return result


def _program_path(program: str) -> Path:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", program) is None:
        raise EvaluationError("PROGRAM must be a simple Python module name")
    path = (PROJECT_ROOT / f"{program}.py").resolve(strict=False)
    if not _inside(path, PROJECT_ROOT):
        raise EvaluationError("PROGRAM resolves outside the project")
    if not path.is_file() or path.is_symlink():
        raise EvaluationError(f"program file not found or unsafe: {path}")
    return path


def import_program(program: str) -> ModuleType:
    """Import one root-level submission program after path validation."""
    expected = _program_path(program)
    module = importlib.import_module(program)
    actual_name = getattr(module, "__file__", None)
    if actual_name is None or Path(actual_name).resolve() != expected:
        raise EvaluationError("imported module does not match the requested program")
    if not callable(getattr(module, "vlm_inference", None)):
        raise EvaluationError(
            "PROGRAM must define callable vlm_inference(image_path, question)"
        )
    return module


def configure_cuda_scope(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Select one physical device before any program can import torch."""
    environment = os.environ if environ is None else environ
    if environment.get("EVO_GPU_AUTHORIZATION_SCOPE"):
        raise EvaluationError("legacy GPU authorization scopes are not accepted")
    original = environment.get("CUDA_VISIBLE_DEVICES")
    if original is None or original == "":
        selected = "0"
        mode = "default_gpu0"
    elif re.fullmatch(r"[0-9]+", original):
        selected = original
        mode = "explicit_single"
    elif re.fullmatch(r"[0-9]+(?:,[0-9]+)+", original):
        selected = original.split(",", 1)[0]
        mode = "first_from_declared_list"
    else:
        raise EvaluationError(
            "CUDA_VISIBLE_DEVICES must be empty or a comma-separated list "
            "of integer indices"
        )
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return {
        "cuda_visible_devices_original": original,
        "cuda_visible_devices": selected,
        "selection_mode": mode,
    }


def verify_gpu0_scope(
    torch_loader: Callable[[str], Any] = importlib.import_module,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless one explicitly selected GPU is logical cuda:0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"[0-9]+", visible) is None:
        raise EvaluationError("CUDA_VISIBLE_DEVICES must select one physical GPU")
    if os.environ.get("EVO_GPU_AUTHORIZATION_SCOPE"):
        raise EvaluationError("legacy GPU authorization scopes are not accepted")

    torch_module = torch_loader("torch")
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise EvaluationError(f"expected one visible CUDA device, found {count}")
    current = int(torch_module.cuda.current_device())
    if current != 0:
        raise EvaluationError(f"expected logical cuda:0, found cuda:{current}")
    result = {
        "physical_gpu_index": int(visible),
        "cuda_visible_devices": visible,
        "visible_device_count": count,
        "logical_device": "cuda:0",
    }
    if selection is not None:
        result.update(selection)
        result["cuda_visible_devices"] = visible
    return result


def verify_greedy(module: ModuleType) -> dict[str, Any]:
    """Require explicit, machine-checkable effective generation metadata."""
    config = getattr(module, "EFFECTIVE_GENERATION_CONFIG", None)
    if not isinstance(config, dict):
        raise EvaluationError(
            "PROGRAM must expose EFFECTIVE_GENERATION_CONFIG as a dict"
        )
    if config.get("do_sample") is not False:
        raise EvaluationError("greedy decoding requires do_sample=False")
    if config.get("num_beams", 1) != 1:
        raise EvaluationError("greedy decoding requires num_beams=1")
    max_new_tokens = config.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise EvaluationError("EFFECTIVE_GENERATION_CONFIG.max_new_tokens must be int")
    if max_new_tokens < 1:
        raise EvaluationError("max_new_tokens must be positive")
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
    }


def _expected_answer(
    ground_truth: dict[str, Any],
    query: dict[str, Any],
) -> str:
    figure_key = str(query["figure_id"])
    subq_idx = int(query["subq_idx"])
    figure = ground_truth.get(figure_key)
    if not isinstance(figure, dict):
        raise EvaluationError(f"missing figure in ground truth: {figure_key}")
    answers = figure.get("answers")
    if not isinstance(answers, list) or not 0 <= subq_idx < len(answers):
        raise EvaluationError(
            f"missing answer for figure {figure_key}, subquery {subq_idx}"
        )
    answer = answers[subq_idx]
    if not isinstance(answer, str):
        raise EvaluationError("ground-truth answer must be a string")
    return answer


def evaluate_program(
    *,
    module: ModuleType,
    program_name: str,
    ground_truth: dict[str, Any],
    selected_queries: list[tuple[str, dict[str, Any]]],
    gpu_metadata: dict[str, Any],
    greedy_metadata: dict[str, Any],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate a module; failed queries remain in the accuracy denominator."""
    records: list[dict[str, Any]] = []
    correct = 0
    num_errors = 0
    started = clock()

    for query_key, query in selected_queries:
        expected = _expected_answer(ground_truth, query)
        response: str | None = None
        error: str | None = None
        generation_metadata: dict[str, Any] | None = None
        item_started = clock()
        try:
            value = module.vlm_inference(
                str(query["figure_path"]),
                str(query["question"]),
            )
            if not isinstance(value, str):
                raise TypeError("vlm_inference must return str")
            response = value
            raw_metadata = getattr(
                module.vlm_inference, "last_generation_metadata", None
            )
            if raw_metadata is None:
                runtime = getattr(module, "_RUNTIME", None)
                raw_metadata = getattr(runtime, "last_generation_metadata", None)
            if isinstance(raw_metadata, dict):
                generation_metadata = dict(raw_metadata)
                _reject_nonfinite(generation_metadata)
        except Exception as exc:
            num_errors += 1
            error = f"{type(exc).__name__}: {exc}"
        latency = clock() - item_started
        is_correct = (
            response is not None
            and response.strip().casefold() == expected.strip().casefold()
        )
        correct += int(is_correct)
        records.append(
            {
                "query_key": query_key,
                "figure_id": query["figure_id"],
                "subq_idx": query["subq_idx"],
                "question": query["question"],
                "image_path": query["figure_path"],
                "expected_answer": expected,
                "response": response,
                "correct": is_correct,
                "latency_seconds": latency,
                "generation_metadata": generation_metadata,
                "error": error,
            }
        )

    total_time = clock() - started
    count = len(selected_queries)
    result = {
        "program": program_name,
        "num_evaluated": count,
        "correct": correct,
        "accuracy": correct / count,
        "num_errors": num_errors,
        "total_time": total_time,
        "avg_time_per_query": total_time / count,
        "timing_scope": "vlm_inference calls only; program/model import excluded",
        "exact_match": "prediction.strip().casefold() == expected.strip().casefold()",
        "greedy_decoding": greedy_metadata,
        "gpu_scope": gpu_metadata,
        "program_identity": {
            "model_id": getattr(module, "MODEL_ID", None),
            "model_revision": getattr(module, "MODEL_REVISION", None),
            "processor_revision": getattr(module, "PROCESSOR_REVISION", None),
            "model_weight_sha256": getattr(module, "MODEL_WEIGHT_SHA256", None),
        },
        "records": records,
    }
    _reject_nonfinite(result)
    return result


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationError("result contains NaN or Infinity")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def resolve_output(program: str, requested: str | None) -> Path:
    path = (
        PROJECT_ROOT / "results" / "evaluations" / f"{program}.json"
        if requested is None
        else Path(requested)
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=False)
    if resolved.suffix != ".json":
        raise EvaluationError("output must have a .json suffix")
    if not _inside(resolved, EVO_ROOT):
        raise EvaluationError("output must remain inside EVO_ROOT")
    return resolved


def save_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish a finite JSON result inside Evo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a submitted vlm_inference program."
    )
    parser.add_argument("program", metavar="PROGRAM")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="number of descriptive queries in source order (default: 128)",
    )
    parser.add_argument(
        "--data",
        help="optional descriptive_val.json path (default: packaged CharXiv data)",
    )
    parser.add_argument(
        "--images",
        help="optional CharXiv image directory (default: packaged images)",
    )
    parser.add_argument(
        "--query-images-manifest",
        help="optional ordered per-query image-copy manifest",
    )
    parser.add_argument(
        "--output",
        help="JSON result path inside the managed runtime root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = configure_cuda_scope()
        module = import_program(args.program)
        gpu_metadata = verify_gpu0_scope(selection=selection)
        greedy_metadata = verify_greedy(module)
        query_options: dict[str, Any] = {"limit": args.limit}
        if args.data is not None:
            query_options["data_path"] = Path(args.data).expanduser().resolve(strict=True)
        if args.images is not None:
            query_options["images_path"] = Path(args.images).expanduser().resolve(
                strict=True
            )
        if args.query_images_manifest is not None:
            query_options["query_image_manifest"] = Path(
                args.query_images_manifest
            ).expanduser().resolve(strict=True)
        ground_truth, queries = load_queries(**query_options)
        result = evaluate_program(
            module=module,
            program_name=args.program,
            ground_truth=ground_truth,
            selected_queries=queries,
            gpu_metadata=gpu_metadata,
            greedy_metadata=greedy_metadata,
        )
        output = resolve_output(args.program, args.output)
        save_json(output, result)
    except (EvaluationError, OSError, ValueError, ImportError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({key: result[key] for key in (
        "program",
        "num_evaluated",
        "correct",
        "accuracy",
        "num_errors",
        "total_time",
        "avg_time_per_query",
    )}, indent=2, allow_nan=False))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
