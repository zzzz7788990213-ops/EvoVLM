"""Deterministic trusted renderer for strict declarative candidates."""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import inspect
import os
import pprint
import secrets
import sys
from pathlib import Path
from types import ModuleType

from .dsl import MODEL_METADATA, ProgramSpec
from .process_safety import EVO_ROOT, PROJECT_ROOT


RENDERER_VERSION = "phase2-local-renderer-v2"


class RenderedProgramError(ValueError):
    pass


def _mapping_literal(spec: ProgramSpec) -> str:
    return pprint.pformat(
        spec.to_mapping(),
        width=88,
        sort_dicts=False,
        compact=False,
    )


def render_candidate_program(spec: ProgramSpec) -> str:
    mapping = _mapping_literal(spec)
    metadata = MODEL_METADATA[spec.model_id]
    return (
        '"""Deterministic renderer-generated VLM inference program."""\n\n'
        "from evolution.candidate_runtime import CandidateRuntime\n\n"
        f"RENDERER_VERSION = {RENDERER_VERSION!r}\n"
        f"CANDIDATE_SPEC = {mapping}\n"
        f"MODEL_ID = {spec.model_id!r}\n"
        f"MODEL_REVISION = {metadata['revision']!r}\n"
        f"PROCESSOR_REVISION = {metadata['revision']!r}\n"
        f"MODEL_WEIGHT_SHA256 = {metadata['weight_sha256']!r}\n"
        "DTYPE = 'bfloat16'\n"
        "ATTENTION_IMPLEMENTATION = 'sdpa'\n"
        "QUANTIZATION = 'none'\n"
        f"EFFECTIVE_GENERATION_CONFIG = {spec.generation_config!r}\n\n"
        "_RUNTIME = CandidateRuntime(CANDIDATE_SPEC)\n\n"
        "def vlm_inference(image_path: str, question: str) -> str:\n"
        "    return _RUNTIME.vlm_inference(image_path, question)\n"
    )


def rendered_program_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _extract_candidate_mapping(source: str) -> dict[str, object]:
    if not isinstance(source, str) or not source.strip():
        raise RenderedProgramError("rendered source must be nonempty")
    if len(source.encode("utf-8")) > 65536:
        raise RenderedProgramError("rendered source exceeds 65536 bytes")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise RenderedProgramError("rendered source is not valid syntax") from exc
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "CANDIDATE_SPEC"
    ]
    if len(assignments) != 1:
        raise RenderedProgramError(
            "rendered source must contain one CANDIDATE_SPEC assignment"
        )
    try:
        mapping = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise RenderedProgramError("embedded candidate spec is not literal") from exc
    if not isinstance(mapping, dict):
        raise RenderedProgramError("embedded candidate spec is not a mapping")
    return mapping


def validate_rendered_program(
    source: str,
    *,
    expected_model_id: str | None = None,
) -> ProgramSpec:
    mapping = _extract_candidate_mapping(source)
    try:
        spec = ProgramSpec(**mapping)
    except (TypeError, ValueError) as exc:
        raise RenderedProgramError("embedded candidate spec is invalid") from exc
    if expected_model_id is not None and spec.model_id != expected_model_id:
        raise RenderedProgramError("rendered program crossed the model population")
    canonical = render_candidate_program(spec)
    if source != canonical:
        raise RenderedProgramError(
            "rendered source differs from the trusted deterministic template"
        )
    return spec


def rendered_program_diff(parent_source: str, child_source: str) -> str:
    validate_rendered_program(parent_source)
    validate_rendered_program(child_source)
    return "".join(
        difflib.unified_diff(
            parent_source.splitlines(keepends=True),
            child_source.splitlines(keepends=True),
            fromfile="parent_program.py",
            tofile="child_program.py",
        )
    )


def _inside_evo(path: Path) -> bool:
    try:
        path.relative_to(EVO_ROOT)
    except ValueError:
        return False
    return True


def _physical_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RenderedProgramError("import-check directory is not physical")
    resolved = path.resolve(strict=True)
    if not _inside_evo(resolved):
        raise RenderedProgramError("import-check directory escaped Evo")
    return resolved


def import_check_rendered_program(
    source: str,
    *,
    temporary_directory: Path,
) -> dict[str, object]:
    """Import only a byte-exact trusted rendering; candidate text is never run."""

    spec = validate_rendered_program(source)
    directory = _physical_directory(temporary_directory)
    name = f"_evo_render_check_{secrets.token_hex(12)}"
    path = directory / f"{name}.py"
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    before_torch = "torch" in sys.modules
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise RenderedProgramError("cannot create import specification")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        _validate_imported_module(module, spec)
        if not before_torch and "torch" in sys.modules:
            raise RenderedProgramError("CPU import check unexpectedly imported torch")
        return {
            "renderer_version": RENDERER_VERSION,
            "program_sha256": rendered_program_sha256(source),
            "model_id": spec.model_id,
            "torch_imported": "torch" in sys.modules,
            "interface": "vlm_inference(image_path, question)",
        }
    finally:
        sys.modules.pop(name, None)
        if path.is_file() and not path.is_symlink():
            path.unlink()


def _validate_imported_module(module: ModuleType, spec: ProgramSpec) -> None:
    if getattr(module, "RENDERER_VERSION", None) != RENDERER_VERSION:
        raise RenderedProgramError("renderer version metadata mismatch")
    if getattr(module, "MODEL_ID", None) != spec.model_id:
        raise RenderedProgramError("model metadata mismatch")
    if getattr(module, "MODEL_REVISION", None) != spec.model_revision:
        raise RenderedProgramError("revision metadata mismatch")
    if getattr(module, "MODEL_WEIGHT_SHA256", None) != spec.weight_sha256:
        raise RenderedProgramError("weight metadata mismatch")
    config = getattr(module, "EFFECTIVE_GENERATION_CONFIG", None)
    if config != spec.generation_config:
        raise RenderedProgramError("generation metadata mismatch")
    inference = getattr(module, "vlm_inference", None)
    if not callable(inference):
        raise RenderedProgramError("rendered program lacks vlm_inference")
    parameters = list(inspect.signature(inference).parameters)
    if parameters != ["image_path", "question"]:
        raise RenderedProgramError("vlm_inference interface is not exact")
