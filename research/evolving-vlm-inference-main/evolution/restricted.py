"""CPU-only restricted subprocess boundary for declarative candidates.

The parent owns private labels and scoring.  The child receives one framed
request containing candidate data plus only an opaque request id, question text,
and image bytes.  Candidate source is parsed as one literal CANDIDATE
assignment and is never imported or executed.  The only child backend is an
explicit TEST_ONLY synthetic fixture.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .core import Metrics
from .process_safety import EVO_ROOT, PROJECT_ROOT
from .worker import CandidateSpec, parse_candidate_source


WORKER_TMP_ROOT = EVO_ROOT / ".tmp/restricted_worker"
WORKER_CACHE_ROOT = EVO_ROOT / ".cache/restricted_worker"
FRAME_VERSION = 1
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PREDICTION_CHARS = 16_384
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
_TEST_ONLY_MODE = "TEST_ONLY"
_FORBIDDEN_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "PROXY",
    "SSH",
)


class RestrictedFrameError(ValueError):
    pass


class RestrictedWorkerError(RuntimeError):
    pass


class RestrictedWorkerTimeout(RestrictedWorkerError):
    pass


class RestrictedWorkerUnavailable(RestrictedWorkerError):
    pass


def _is_inside_evo(path: Path) -> bool:
    try:
        path.relative_to(EVO_ROOT)
    except ValueError:
        return False
    return True


def _ensure_internal_directory(path: Path) -> Path:
    if not _is_inside_evo(path):
        raise RestrictedWorkerUnavailable("worker directory escaped Evo")
    relative = path.relative_to(EVO_ROOT)
    current = EVO_ROOT
    if current.is_symlink() or not current.is_dir():
        raise RestrictedWorkerUnavailable("Evo root is not a physical directory")
    for component in relative.parts:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise RestrictedWorkerUnavailable(
                    "worker directory chain contains a non-directory or symlink"
                )
        else:
            current.mkdir(mode=0o700)
    resolved = current.resolve(strict=True)
    if not _is_inside_evo(resolved):
        raise RestrictedWorkerUnavailable("worker directory resolved outside Evo")
    return resolved


def sanitized_worker_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an allowlisted child environment; no parent values are copied."""

    del environ
    tmp = _ensure_internal_directory(WORKER_TMP_ROOT)
    cache = _ensure_internal_directory(WORKER_CACHE_ROOT)
    pycache = _ensure_internal_directory(cache / "pycache")
    return {
        "EVO_ROOT": str(EVO_ROOT),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
        "PYTHONPYCACHEPREFIX": str(pycache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "EVO_RESTRICTED_WORKER_MODE": _TEST_ONLY_MODE,
        "LANG": "C.UTF-8",
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RestrictedFrameError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise RestrictedFrameError(f"non-finite JSON constant is forbidden: {value}")


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
        raise RestrictedFrameError("frame is not finite JSON") from exc
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise RestrictedFrameError("frame payload size is invalid")
    return struct.pack(">I", len(payload)) + payload


def decode_frame(frame: bytes) -> dict[str, object]:
    if not isinstance(frame, bytes) or len(frame) < 4:
        raise RestrictedFrameError("frame header is incomplete")
    declared = struct.unpack(">I", frame[:4])[0]
    if declared < 2 or declared > MAX_FRAME_BYTES:
        raise RestrictedFrameError("declared frame length is invalid")
    if len(frame) != 4 + declared:
        raise RestrictedFrameError("frame length mismatch or trailing bytes")
    try:
        text = frame[4:].decode("utf-8", errors="strict")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestrictedFrameError("frame is not strict UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise RestrictedFrameError("frame document must be an object")
    return document


@dataclass(frozen=True)
class RestrictedSample:
    request_id: str
    question: str
    image_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _REQUEST_ID.fullmatch(
            self.request_id
        ):
            raise ValueError("request_id must be an opaque 32-character hex id")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be nonempty text")
        if len(self.question) > 8_192 or "\x00" in self.question:
            raise ValueError("question is too long or contains NUL")
        if type(self.image_bytes) is not bytes or not self.image_bytes:
            raise ValueError("image_bytes must be nonempty bytes")
        if len(self.image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image_bytes exceeds the restricted frame limit")


@dataclass(frozen=True)
class PrivateScoredSample:
    """Parent-only label wrapper; expected_answer is never serialized."""

    sample: RestrictedSample
    expected_answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.expected_answer, str):
            raise ValueError("expected_answer must be text")


@dataclass(frozen=True)
class WorkerReply:
    request_id: str
    prediction: str
    work_units: int
    test_only: bool = True


@dataclass(frozen=True)
class SpawnAttestation:
    argv_uses_absolute_venv_python: bool
    cwd: str
    shell: bool
    start_new_session: bool
    env_keys: tuple[str, ...]


def build_request_document(
    source: str,
    sample: RestrictedSample,
) -> dict[str, object]:
    parse_candidate_source(source)
    return {
        "schema_version": FRAME_VERSION,
        "candidate_source": source,
        "request_id": sample.request_id,
        "question": sample.question,
        "image_b64": base64.b64encode(sample.image_bytes).decode("ascii"),
    }


def _parse_request(document: Mapping[str, object]) -> tuple[CandidateSpec, RestrictedSample]:
    expected = {
        "schema_version",
        "candidate_source",
        "request_id",
        "question",
        "image_b64",
    }
    if set(document) != expected:
        raise RestrictedFrameError("worker request fields are not exact")
    if document["schema_version"] != FRAME_VERSION:
        raise RestrictedFrameError("unsupported worker request version")
    source = document["candidate_source"]
    request_id = document["request_id"]
    question = document["question"]
    image_b64 = document["image_b64"]
    if not isinstance(source, str):
        raise RestrictedFrameError("candidate_source must be text")
    if not isinstance(request_id, str) or not isinstance(question, str):
        raise RestrictedFrameError("request id and question must be text")
    if not isinstance(image_b64, str):
        raise RestrictedFrameError("image_b64 must be text")
    try:
        image_bytes = base64.b64decode(
            image_b64.encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise RestrictedFrameError("image_b64 is invalid") from exc
    try:
        sample = RestrictedSample(request_id, question, image_bytes)
        spec = parse_candidate_source(source)
    except (ValueError, TypeError) as exc:
        raise RestrictedFrameError(str(exc)) from exc
    return spec, sample


def _parse_reply(document: Mapping[str, object]) -> WorkerReply:
    expected = {
        "schema_version",
        "request_id",
        "prediction",
        "error",
        "work_units",
        "test_only",
    }
    if set(document) != expected:
        raise RestrictedFrameError("worker reply fields are not exact")
    if document["schema_version"] != FRAME_VERSION:
        raise RestrictedFrameError("unsupported worker reply version")
    request_id = document["request_id"]
    prediction = document["prediction"]
    error = document["error"]
    work_units = document["work_units"]
    test_only = document["test_only"]
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise RestrictedFrameError("reply request_id is invalid")
    if test_only is not True:
        raise RestrictedFrameError("worker reply must be TEST_ONLY")
    if error is not None:
        if not isinstance(error, str) or not error or len(error) > 1_024:
            raise RestrictedFrameError("worker error is invalid")
        if prediction is not None or work_units is not None:
            raise RestrictedFrameError("errored reply contains result fields")
        raise RestrictedWorkerError(f"restricted child error: {error}")
    if not isinstance(prediction, str) or len(prediction) > MAX_PREDICTION_CHARS:
        raise RestrictedFrameError("prediction is invalid")
    if (
        isinstance(work_units, bool)
        or not isinstance(work_units, int)
        or not 1 <= work_units <= 1_000_000
    ):
        raise RestrictedFrameError("work_units is invalid")
    return WorkerReply(request_id, prediction, work_units)


def _assert_test_child_scope() -> None:
    if os.environ.get("EVO_RESTRICTED_WORKER_MODE") != _TEST_ONLY_MODE:
        raise RestrictedWorkerUnavailable("restricted child is not in TEST_ONLY mode")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RestrictedWorkerUnavailable("restricted child CVD must be 0")
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT:
        raise RestrictedWorkerUnavailable("restricted child cwd is not the project")
    for name in ("TMPDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX"):
        value = os.environ.get(name)
        if not value:
            raise RestrictedWorkerUnavailable(f"restricted child missing {name}")
        path = Path(value).resolve(strict=True)
        if not _is_inside_evo(path):
            raise RestrictedWorkerUnavailable(f"restricted child {name} escaped Evo")
    forbidden = [
        name
        for name in os.environ
        if any(marker in name.upper() for marker in _FORBIDDEN_ENV_MARKERS)
    ]
    if forbidden:
        raise RestrictedWorkerUnavailable(
            "restricted child inherited forbidden environment names"
        )


def _test_only_prediction(
    spec: CandidateSpec,
    sample: RestrictedSample,
) -> tuple[str, int]:
    """Trusted deterministic fixture; never available as a production backend."""

    _assert_test_child_scope()
    spec.render(sample.question)
    mode = spec.instruction.strip()
    words = [
        word.strip(".,?!:;")
        for word in sample.question.split()
        if word.strip(".,?!:;")
    ]
    if mode == "TEST_ONLY_SLEEP":
        time.sleep(60.0)
        raise AssertionError("unreachable")
    if mode == "TEST_ONLY_ERROR":
        raise RuntimeError("synthetic backend failure")
    if mode == "TEST_ONLY_SCOPE":
        prediction = "clean"
    elif mode == "TEST_ONLY_LAST_TOKEN":
        if not words:
            raise ValueError("synthetic question has no token")
        prediction = words[-1]
    elif mode == "TEST_ONLY_FIRST_TOKEN":
        if not words:
            raise ValueError("synthetic question has no token")
        prediction = words[0]
    elif mode == "TEST_ONLY_CONSTANT_ALPHA":
        prediction = "alpha"
    elif mode == "TEST_ONLY_IMAGE_SHA8":
        prediction = hashlib.sha256(sample.image_bytes).hexdigest()[:8]
    else:
        raise ValueError("unknown TEST_ONLY synthetic instruction")
    if spec.answer_mode == "final_tag":
        raw = f"<final>{prediction}</final>"
        prediction = raw[len("<final>") : -len("</final>")]
    return prediction, spec.max_new_tokens


def _child_process_once() -> int:
    try:
        _assert_test_child_scope()
        raw = sys.stdin.buffer.read(MAX_FRAME_BYTES + 5)
        document = decode_frame(raw)
        spec, sample = _parse_request(document)
    except Exception as exc:
        print(f"restricted request rejected: {type(exc).__name__}", file=sys.stderr)
        return 2
    try:
        prediction, work_units = _test_only_prediction(spec, sample)
        reply: dict[str, object] = {
            "schema_version": FRAME_VERSION,
            "request_id": sample.request_id,
            "prediction": prediction,
            "error": None,
            "work_units": work_units,
            "test_only": True,
        }
    except Exception as exc:
        reply = {
            "schema_version": FRAME_VERSION,
            "request_id": sample.request_id,
            "prediction": None,
            "error": type(exc).__name__,
            "work_units": None,
            "test_only": True,
        }
    sys.stdout.buffer.write(encode_frame(reply))
    sys.stdout.buffer.flush()
    return 0


def _trusted_python() -> Path:
    candidate = Path(os.environ.get("EVOLVING_VLM_PYTHON", sys.executable)).expanduser()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise RestrictedWorkerUnavailable("worker Python is not a regular file")
    return resolved


def _trusted_project() -> Path:
    if PROJECT_ROOT.is_symlink() or not PROJECT_ROOT.is_dir():
        raise RestrictedWorkerUnavailable("project root is not a physical directory")
    resolved = PROJECT_ROOT.resolve(strict=True)
    if not _is_inside_evo(resolved):
        raise RestrictedWorkerUnavailable("project root escaped Evo")
    return resolved


def _kill_own_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if pgid != process.pid:
        process.kill()
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=0.25)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1.0)


@dataclass
class RestrictedSubprocessWorker:
    """One-request-per-child TEST_ONLY restricted worker."""

    test_mode: bool = False
    timeout_seconds: float = 5.0
    last_spawn_attestation: SpawnAttestation | None = None
    last_timed_out_pgid: int | None = None

    @property
    def restricted_worker_ready(self) -> bool:
        return True

    def infer(self, source: str, sample: RestrictedSample) -> WorkerReply:
        if not self.test_mode:
            raise RestrictedWorkerUnavailable(
                "production candidate backend is not configured"
            )
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > 60
        ):
            raise ValueError("worker timeout must be finite and in (0, 60]")
        request = build_request_document(source, sample)
        frame = encode_frame(request)
        python = _trusted_python()
        project = _trusted_project()
        environment = sanitized_worker_environment()
        bootstrap = (
            "import sys;"
            f"sys.path.insert(0,{str(project)!r});"
            "from evolution.restricted import _child_process_once;"
            "raise SystemExit(_child_process_once())"
        )
        argv = [str(python), "-I", "-c", bootstrap]
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
        self.last_spawn_attestation = SpawnAttestation(
            argv_uses_absolute_venv_python=Path(argv[0]).is_absolute(),
            cwd=str(project),
            shell=False,
            start_new_session=True,
            env_keys=tuple(sorted(environment)),
        )
        try:
            stdout, stderr = process.communicate(
                input=frame,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            try:
                self.last_timed_out_pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                self.last_timed_out_pgid = process.pid
            _kill_own_process_group(process)
            raise RestrictedWorkerTimeout("restricted child timed out") from exc
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace")[-512:].strip()
            raise RestrictedWorkerError(
                f"restricted child exited {process.returncode}: {message}"
            )
        if stderr:
            raise RestrictedWorkerError("restricted child wrote unexpected stderr")
        reply = _parse_reply(decode_frame(stdout))
        if reply.request_id != sample.request_id:
            raise RestrictedFrameError("reply request_id mismatch")
        return reply


@dataclass
class ParentOwnedScorer:
    """Trusted parent-only exact-match scorer over TEST_ONLY worker predictions."""

    worker: RestrictedSubprocessWorker

    def evaluate(
        self,
        source: str,
        cases: Sequence[PrivateScoredSample],
    ) -> Metrics:
        if not cases:
            raise ValueError("at least one private scored sample is required")
        correct = 0
        errors = 0
        work_units = 0
        for case in cases:
            try:
                reply = self.worker.infer(source, case.sample)
            except RestrictedWorkerError:
                errors += 1
                work_units += 1
                continue
            work_units += reply.work_units
            correct += int(
                reply.prediction.strip().casefold()
                == case.expected_answer.strip().casefold()
            )
        count = len(cases)
        return Metrics(
            accuracy=correct / count,
            latency_seconds=max(work_units, 1) / (1_000.0 * count),
            error_rate=errors / count,
        )
