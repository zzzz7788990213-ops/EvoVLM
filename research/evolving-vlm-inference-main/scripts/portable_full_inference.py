"""Run a fresh, portable official-evaluator inference batch.

This is not a replay or replacement of the sealed Phase 3 formal campaign.
It writes only to a caller-selected, previously nonexistent managed directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
PROGRAMS = (
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)


class FullInferenceError(RuntimeError):
    """Portable full-inference preflight or child execution failed."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _input_file(value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise FullInferenceError(f"input JSON is not a physical regular file: {path}")
    return path.resolve(strict=True)


def _input_directory(value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise FullInferenceError(f"image input is not a physical directory: {path}")
    return path.resolve(strict=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new managed output directory")
    parser.add_argument("--data", help="external or packaged descriptive_val.json")
    parser.add_argument("--images", help="external or packaged image directory")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument(
        "--program", action="append", choices=PROGRAMS,
        help="program to run; repeat as needed (default: all four optimized programs)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if args.limit < 1:
        raise FullInferenceError("--limit must be positive")
    runtime_root = Path(
        os.environ.get("EVOLVING_VLM_RUNTIME_ROOT", str(PROJECT_ROOT))
    ).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = runtime_root / output
    output = output.resolve(strict=False)
    if not _inside(output, runtime_root) or output == runtime_root:
        raise FullInferenceError("output must be a child of the managed runtime root")
    if output.exists() or output.is_symlink():
        raise FullInferenceError("output already exists; refusing overwrite/resume")
    data = _input_file(
        args.data, PROJECT_ROOT / "charxiv/data/descriptive_val.json"
    )
    images = _input_directory(args.images, PROJECT_ROOT / "charxiv/images")
    selected = tuple(args.program or PROGRAMS)
    if len(set(selected)) != len(selected):
        raise FullInferenceError("duplicate --program is not accepted")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.dry_run and (visible is None or re.fullmatch(r"[0-9]+", visible) is None):
        raise FullInferenceError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    return {
        "runtime_root": runtime_root,
        "output": output,
        "data": data,
        "images": images,
        "programs": selected,
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
    }


def run(configuration: dict[str, object]) -> None:
    output = configuration["output"]
    assert isinstance(output, Path)
    if configuration["dry_run"]:
        printable = {
            key: ([str(item) for item in value] if isinstance(value, tuple) else str(value) if isinstance(value, Path) else value)
            for key, value in configuration.items()
        }
        print(json.dumps(printable, sort_keys=True, indent=2))
        print("portable full-inference preflight PASS (no output created)")
        return
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for program in configuration["programs"]:
        command = [
            sys.executable, "-B", str(PROJECT_ROOT / "evaluate.py"), str(program),
            "--limit", str(configuration["limit"]),
            "--data", str(configuration["data"]),
            "--images", str(configuration["images"]),
            "--output", str(output / f"{program}.json"),
        ]
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=False,
        )
        if result.returncode != 0:
            raise FullInferenceError(
                f"official evaluator failed for {program} with rc={result.returncode}"
            )
    print(f"portable full-inference complete: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        configuration = prepare(build_parser().parse_args(argv))
        run(configuration)
    except (FullInferenceError, OSError, ValueError) as exc:
        print(f"portable full-inference failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
