"""Shared strict IPC, process, environment, and artifact safety helpers."""

from __future__ import annotations

import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping


PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
_DEFAULT_RUNTIME_ROOT = (
    PROJECT_ROOT.parent.parent
    if PROJECT_ROOT.parent.name == "workspace"
    else PROJECT_ROOT
)
EVO_ROOT = Path(
    os.environ.get("EVOLVING_VLM_RUNTIME_ROOT", str(_DEFAULT_RUNTIME_ROOT))
).expanduser().resolve(strict=False)
VENV_PYTHON = Path(
    os.environ.get("EVOLVING_VLM_PYTHON", sys.executable)
).expanduser()
MAX_FRAME_BYTES = 64 * 1024 * 1024
SYSTEM_CUDA_HOME = "/usr/local/cuda"
SYSTEM_CUDA_RUNTIME_DIRS = (
    "/usr/local/cuda/lib64",
    "/.singularity.d/libs",
)
_FORBIDDEN_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "PROXY",
    "SSH",
)


class ProcessSafetyError(RuntimeError):
    pass


class FrameError(ValueError):
    pass


class ChildTimeout(ProcessSafetyError):
    pass


class NetworkDisabled(ProcessSafetyError):
    pass


@dataclass(frozen=True)
class ChildResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    pid: int
    elapsed_seconds: float
    reaped: bool
    shell: bool = False
    start_new_session: bool = True


class NetworkAttemptCounter:
    def __init__(self) -> None:
        self.count = 0

    def audit(self, event: str, args: tuple[object, ...]) -> None:
        del args
        if event.startswith("socket."):
            self.count += 1
            raise NetworkDisabled(f"network audit event denied: {event}")


def install_network_guard() -> NetworkAttemptCounter:
    import socket

    socket.has_ipv6 = False
    counter = NetworkAttemptCounter()
    sys.addaudithook(counter.audit)
    return counter


def _inside_evo(path: Path) -> bool:
    try:
        path.relative_to(EVO_ROOT)
    except ValueError:
        return False
    return True


def ensure_internal_directory(path: Path) -> Path:
    if not _inside_evo(path):
        raise ProcessSafetyError("directory escaped Evo")
    if EVO_ROOT.is_symlink() or not EVO_ROOT.is_dir():
        raise ProcessSafetyError("Evo root is not physical")
    current = EVO_ROOT
    for component in path.relative_to(EVO_ROOT).parts:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise ProcessSafetyError("directory chain contains a link or file")
        else:
            current.mkdir(mode=0o700)
    resolved = current.resolve(strict=True)
    if not _inside_evo(resolved):
        raise ProcessSafetyError("directory resolved outside Evo")
    return resolved


def trusted_python() -> Path:
    resolved = VENV_PYTHON.resolve(strict=True)
    if not resolved.is_file():
        raise ProcessSafetyError("configured Python is not a regular file")
    return resolved


def trusted_project() -> Path:
    if PROJECT_ROOT.is_symlink() or not PROJECT_ROOT.is_dir():
        raise ProcessSafetyError("project directory is not physical")
    resolved = PROJECT_ROOT.resolve(strict=True)
    if resolved != PROJECT_ROOT:
        raise ProcessSafetyError("project directory changed physical identity")
    return resolved


def sanitized_gpu_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Inherit nothing; writable paths are Evo-local and runtime libs are fixed."""

    del environ
    tmp = ensure_internal_directory(EVO_ROOT / ".tmp/local_evolution")
    cache = ensure_internal_directory(EVO_ROOT / ".cache/local_evolution")
    pycache = ensure_internal_directory(cache / "pycache")
    if not Path(SYSTEM_CUDA_HOME).is_dir() or not all(
        Path(item).is_dir() for item in SYSTEM_CUDA_RUNTIME_DIRS
    ):
        raise ProcessSafetyError("fixed system CUDA runtime directories are unavailable")
    environment = {
        "EVO_ROOT": str(EVO_ROOT),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(EVO_ROOT / ".cache"),
        "HF_HOME": str(EVO_ROOT / ".cache/huggingface"),
        "HF_DATASETS_CACHE": str(EVO_ROOT / ".cache/huggingface/datasets"),
        "TORCH_HOME": str(EVO_ROOT / ".cache/torch"),
        "TORCH_EXTENSIONS_DIR": str(EVO_ROOT / ".cache/torch_extensions"),
        "TRITON_CACHE_DIR": str(EVO_ROOT / ".cache/triton"),
        "CUDA_CACHE_PATH": str(EVO_ROOT / ".cache/cuda"),
        "PYTHONPYCACHEPREFIX": str(pycache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(EVO_ROOT / ".cache/matplotlib"),
        "NUMBA_CACHE_DIR": str(EVO_ROOT / ".cache/numba"),
        "VLLM_CACHE_ROOT": str(EVO_ROOT / ".cache/vllm"),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_HOME": SYSTEM_CUDA_HOME,
        "LD_LIBRARY_PATH": ":".join(SYSTEM_CUDA_RUNTIME_DIRS),
        "EVO_NETWORK_DISABLED": "1",
        "LANG": "C.UTF-8",
    }
    forbidden = [
        name
        for name in environment
        if any(marker in name.upper() for marker in _FORBIDDEN_ENV_MARKERS)
    ]
    if forbidden:
        raise ProcessSafetyError("allowlisted environment contains forbidden names")
    for name, value in environment.items():
        if name in {
            "TMPDIR",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "PYTHONPYCACHEPREFIX",
            "MPLCONFIGDIR",
            "NUMBA_CACHE_DIR",
            "VLLM_CACHE_ROOT",
        }:
            path = Path(value).resolve(strict=False)
            if not _inside_evo(path):
                raise ProcessSafetyError(f"{name} escaped Evo")
    return environment


def _duplicate_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise FrameError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _nonfinite_guard(value: str) -> object:
    raise FrameError(f"non-finite JSON value: {value}")


def encode_frame(document: Mapping[str, object]) -> bytes:
    try:
        payload = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrameError("document is not finite JSON") from exc
    if len(payload) < 2 or len(payload) > MAX_FRAME_BYTES:
        raise FrameError("frame payload size is invalid")
    return struct.pack(">I", len(payload)) + payload


def decode_frame(frame: bytes) -> dict[str, object]:
    if type(frame) is not bytes or len(frame) < 4:
        raise FrameError("frame header is missing")
    declared = struct.unpack(">I", frame[:4])[0]
    if declared < 2 or declared > MAX_FRAME_BYTES:
        raise FrameError("declared frame length is invalid")
    if len(frame) != declared + 4:
        raise FrameError("frame length mismatch or trailing data")
    try:
        value = json.loads(
            frame[4:].decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError("frame is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise FrameError("frame document must be an object")
    return value


def read_frame(handle: BinaryIO) -> dict[str, object]:
    header = handle.read(4)
    if len(header) != 4:
        raise FrameError("stream frame header is incomplete")
    declared = struct.unpack(">I", header)[0]
    if declared < 2 or declared > MAX_FRAME_BYTES:
        raise FrameError("stream declared frame length is invalid")
    payload = handle.read(declared)
    if len(payload) != declared:
        raise FrameError("stream frame payload is incomplete")
    return decode_frame(header + payload)


def write_frame(handle: BinaryIO, document: Mapping[str, object]) -> None:
    handle.write(encode_frame(document))
    handle.flush()


def concatenate_frames(documents: list[Mapping[str, object]]) -> bytes:
    return b"".join(encode_frame(document) for document in documents)


def _kill_child_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if pgid != process.pid:
        process.kill()
        process.wait(timeout=2)
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_trusted_child(
    *,
    module: str,
    function: str,
    input_bytes: bytes,
    timeout_seconds: float,
) -> ChildResult:
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 4500:
        raise ProcessSafetyError("child timeout is outside the approved range")
    if not module.startswith("evolution.") or not function.startswith("_child_"):
        raise ProcessSafetyError("child bootstrap target is not trusted")
    python = trusted_python()
    project = trusted_project()
    environment = sanitized_gpu_environment()
    bootstrap = (
        "import sys; sys.path.insert(0, "
        + repr(str(project))
        + "); from "
        + module
        + " import "
        + function
        + "; raise SystemExit("
        + function
        + "())"
    )
    argv = [str(python), "-I", "-B", "-c", bootstrap]
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        cwd=str(project),
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
        close_fds=True,
    )
    try:
        stdout, stderr = process.communicate(
            input=input_bytes,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _kill_child_group(process)
        raise ChildTimeout(
            f"trusted child timed out after {timeout_seconds:.3f} seconds"
        ) from exc
    elapsed = time.perf_counter() - started
    return ChildResult(
        stdout=stdout,
        stderr=stderr,
        returncode=int(process.returncode),
        pid=process.pid,
        elapsed_seconds=elapsed,
        reaped=process.poll() is not None,
    )


def _finite_tree(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProcessSafetyError("artifact contains NaN or Infinity")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProcessSafetyError("artifact key is not text")
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def atomic_create_json(path: Path, document: Mapping[str, object]) -> None:
    _finite_tree(dict(document))
    parent = ensure_internal_directory(path.parent)
    target = parent / path.name
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"artifact already exists: {target}")
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = parent / f".{target.name}.{os.getpid()}.{time.time_ns()}.partial"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def atomic_create_text(path: Path, text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text artifact must be a string")
    parent = ensure_internal_directory(path.parent)
    target = parent / path.name
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"artifact already exists: {target}")
    payload = text.encode("utf-8")
    temporary = parent / f".{target.name}.{os.getpid()}.{time.time_ns()}.partial"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
