"""Portable, fixed-identity model discovery for submission inference programs."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
_MODEL_ENV = {
    "Qwen/Qwen3-VL-2B-Instruct": "EVOLVING_VLM_INSTRUCT_MODEL_PATH",
    "Qwen/Qwen3-VL-2B-Thinking": "EVOLVING_VLM_THINKING_MODEL_PATH",
}


class ModelLocationError(RuntimeError):
    """The configured fixed model cannot be located or authenticated."""


@dataclass(frozen=True)
class ModelSource:
    reference: str
    revision: str
    local_files_only: bool
    local_path: Path | None
    weight_sha256: str
    weight_size_bytes: int | None

    @property
    def pretrained_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if self.local_path is None:
            values["revision"] = self.revision
        return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_local_source(
    path: Path,
    *,
    revision: str,
    expected_weight_sha256: str,
) -> ModelSource:
    if path.is_symlink() or not path.is_dir():
        raise ModelLocationError(f"local model override is not a physical directory: {path}")
    resolved = path.resolve(strict=True)
    weight = resolved / "model.safetensors"
    if weight.is_symlink() or not weight.is_file():
        raise ModelLocationError(f"local model weight is not a regular file: {weight}")
    actual = _sha256(weight)
    if actual != expected_weight_sha256:
        raise ModelLocationError(
            f"local model weight SHA-256 mismatch: expected {expected_weight_sha256}, got {actual}"
        )
    return ModelSource(
        reference=str(resolved),
        revision=revision,
        local_files_only=True,
        local_path=resolved,
        weight_sha256=actual,
        weight_size_bytes=weight.stat().st_size,
    )


def resolve_model_source(
    *,
    model_id: str,
    revision: str,
    model_directory: str,
    expected_weight_sha256: str,
    environ: dict[str, str] | None = None,
) -> ModelSource:
    """Resolve the frozen model without embedding a machine-specific path."""

    if model_id not in _MODEL_ENV:
        raise ModelLocationError(f"unsupported fixed model ID: {model_id}")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ModelLocationError("model revision must be a 40-character Git hash")
    if re.fullmatch(r"[0-9a-f]{64}", expected_weight_sha256) is None:
        raise ModelLocationError("model weight identity must be SHA-256")
    environment = os.environ if environ is None else environ
    specific = environment.get(_MODEL_ENV[model_id], "").strip()
    generic = environment.get("EVOLVING_VLM_MODEL_ROOT", "").strip()
    if specific:
        return _validated_local_source(
            Path(specific),
            revision=revision,
            expected_weight_sha256=expected_weight_sha256,
        )
    if generic:
        return _validated_local_source(
            Path(generic) / model_directory,
            revision=revision,
            expected_weight_sha256=expected_weight_sha256,
        )
    packaged = PROJECT_ROOT / "models" / model_directory
    if packaged.exists() or packaged.is_symlink():
        return _validated_local_source(
            packaged,
            revision=revision,
            expected_weight_sha256=expected_weight_sha256,
        )
    return ModelSource(
        reference=model_id,
        revision=revision,
        local_files_only=False,
        local_path=None,
        weight_sha256=expected_weight_sha256,
        weight_size_bytes=None,
    )


def resolve_readable_image(image_path: str) -> Path:
    """Resolve an evaluator-supplied image without package containment."""

    if not isinstance(image_path, str) or not image_path or "://" in image_path:
        raise ValueError("image_path must name a local readable file")
    path = Path(image_path)
    if path.is_symlink() or not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError("image must be a readable regular file")
    return path.resolve(strict=True)
