"""Fail-closed active GPU policy: physical GPU 0 only."""

from __future__ import annotations

import os


PHYSICAL_GPU_INDEX = 0
CUDA_VISIBLE_DEVICES = "0"
LOGICAL_DEVICE = "cuda:0"
VISIBLE_DEVICE_COUNT = 1
AUTHORIZATION_SCOPE = "physical-gpu0-only"
AUTHORIZATION_ENV = "EVO_GPU_AUTHORIZATION_SCOPE"
DEFAULT_FUTURE_GPU = 0


def active_gpu_metadata() -> dict[str, object]:
    """Validate GPU 0 before any torch import and return frozen metadata."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "active inference requires CUDA_VISIBLE_DEVICES=0 exactly; "
            "GPU1 authorization has expired"
        )
    if AUTHORIZATION_ENV in os.environ:
        raise RuntimeError(
            f"{AUTHORIZATION_ENV} is a retired replay mechanism and must be absent"
        )
    return {
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "logical_device": LOGICAL_DEVICE,
        "visible_device_count": VISIBLE_DEVICE_COUNT,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "default_future_gpu": DEFAULT_FUTURE_GPU,
    }


def assert_single_visible_cuda_device(actual_count: int) -> None:
    if actual_count != VISIBLE_DEVICE_COUNT:
        raise RuntimeError(
            f"expected exactly {VISIBLE_DEVICE_COUNT} visible CUDA device, got {actual_count}"
        )
