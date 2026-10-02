"""Restricted target evaluator: private scoring stays in the CPU parent."""

from __future__ import annotations

import base64
import hashlib
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from model_locator import resolve_readable_image

from .candidate_runtime import (
    CandidateRuntimeError,
    InferenceMeasurement,
    TrustedTargetEngine,
)
from .dsl import (
    MODEL_IDS,
    canonical_candidate_source,
    parse_candidate_dsl,
    seed_spec,
)
from .process_safety import (
    ChildTimeout,
    FrameError,
    atomic_create_json,
    decode_frame,
    encode_frame,
    install_network_guard,
    read_frame,
    run_trusted_child,
    write_frame,
)
from .renderer import (
    rendered_program_sha256,
    validate_rendered_program,
)


TARGET_WORKER_VERSION = "phase2-restricted-target-v3"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_TARGET_QUERIES = 32
MAX_TARGET_CANDIDATES = 8
TARGET_RECORD_FIELDS = {
    "candidate_id",
    "request_id",
    "image_sha256",
    "status",
    "prediction",
    "prediction_sha256",
    "raw_text",
    "input_token_count",
    "generated_token_count",
    "termination_reason",
    "preprocessing_seconds",
    "generation_seconds",
    "total_seconds",
    "peak_memory_bytes",
    "error_type",
    "error_message",
}


class TargetWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrivateQuery:
    request_id: str
    query_key: str
    figure_id: str
    qid: str
    question: str
    image_path: Path
    expected_answer: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id) > 128
            or not all(c.isalnum() or c in "-_" for c in self.request_id)
        ):
            raise ValueError("private query request_id is invalid")
        for name, value in (
            ("query_key", self.query_key),
            ("figure_id", self.figure_id),
            ("qid", self.qid),
            ("question", self.question),
            ("expected_answer", self.expected_answer),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"private query {name} is invalid")
        resolve_readable_image(str(self.image_path))

    def to_worker_mapping(self) -> dict[str, object]:
        data = self.image_path.read_bytes()
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("private query image byte size is invalid")
        return {
            "request_id": self.request_id,
            "question": self.question,
            "image_b64": base64.b64encode(data).decode("ascii"),
            "image_sha256": hashlib.sha256(data).hexdigest(),
        }


@dataclass(frozen=True)
class TargetCandidate:
    candidate_id: str
    model_id: str
    canonical_dsl: str
    rendered_source: str
    rendered_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or len(self.candidate_id) > 128
            or not all(c.isalnum() or c in "-_" for c in self.candidate_id)
        ):
            raise ValueError("target candidate_id is invalid")
        if self.model_id not in MODEL_IDS:
            raise ValueError("target candidate model is unsupported")
        spec = parse_candidate_dsl(
            self.canonical_dsl,
            expected_model_id=self.model_id,
        )
        if canonical_candidate_source(spec) != self.canonical_dsl:
            raise ValueError("target candidate DSL is not canonical")
        rendered_spec = validate_rendered_program(
            self.rendered_source,
            expected_model_id=self.model_id,
        )
        if rendered_spec != spec:
            raise ValueError("target candidate rendered spec differs from DSL")
        if rendered_program_sha256(self.rendered_source) != self.rendered_sha256:
            raise ValueError("target candidate rendered hash mismatch")

    def to_worker_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    sample_count: int
    correct_count: int
    accuracy: float
    error_count: int
    total_seconds: float
    mean_latency_seconds: float
    total_generated_tokens: int
    termination_counts: Mapping[str, int]
    private_records: tuple[Mapping[str, object], ...]

    def public_metrics(self, stage: str) -> dict[str, object]:
        return {
            "stage": stage,
            "samples": self.sample_count,
            "accuracy": self.accuracy,
            "errors": self.error_count,
            "mean_latency_seconds": self.mean_latency_seconds,
        }

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["termination_counts"] = dict(self.termination_counts)
        value["private_records"] = [dict(record) for record in self.private_records]
        return value


def _strict_image_bytes(raw: Mapping[str, object]) -> tuple[str, str, bytes, str]:
    if set(raw) != {"request_id", "question", "image_b64", "image_sha256"}:
        raise TargetWorkerError("target query fields are not exact")
    request_id = raw.get("request_id")
    question = raw.get("question")
    encoded = raw.get("image_b64")
    expected_hash = raw.get("image_sha256")
    if (
        not isinstance(request_id, str)
        or not request_id
        or len(request_id) > 128
        or not all(c.isalnum() or c in "-_" for c in request_id)
    ):
        raise TargetWorkerError("target request_id is invalid")
    if not isinstance(question, str) or not question or len(question) > 8192:
        raise TargetWorkerError("target question is invalid")
    if not isinstance(encoded, str) or not isinstance(expected_hash, str):
        raise TargetWorkerError("target image encoding fields are invalid")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise TargetWorkerError("target image base64 is invalid") from exc
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise TargetWorkerError("target image size is invalid")
    actual_hash = hashlib.sha256(image_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise TargetWorkerError("target image hash mismatch")
    return request_id, question, image_bytes, actual_hash


def _strict_child_document_v1_archived(
    document: Mapping[str, object],
) -> tuple[str, list[TargetCandidate], list[tuple[str, str, bytes, str]]]:
    if set(document) != {"kind", "worker_version", "model_id", "candidates", "queries"}:
        raise TargetWorkerError("target worker request fields are not exact")
    if document.get("kind") != "restricted_target_evaluation":
        raise TargetWorkerError("target worker request kind is invalid")
    if document.get("worker_version") != TARGET_WORKER_VERSION:
        raise TargetWorkerError("target worker version mismatch")
    model_id = document.get("model_id")
    if model_id not in MODEL_IDS:
        raise TargetWorkerError("target worker model is invalid")
    raw_candidates = document.get("candidates")
    raw_queries = document.get("queries")
    if (
        not isinstance(raw_candidates, list)
        or not 1 <= len(raw_candidates) <= MAX_TARGET_CANDIDATES
    ):
        raise TargetWorkerError("target candidate count is invalid")
    if (
        not isinstance(raw_queries, list)
        or not 1 <= len(raw_queries) <= MAX_TARGET_QUERIES
    ):
        raise TargetWorkerError("target query count is invalid")
    candidates: list[TargetCandidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict) or set(raw) != set(TargetCandidate.__dataclass_fields__):
            raise TargetWorkerError("target candidate fields are not exact")
        candidate = TargetCandidate(**raw)
        if candidate.model_id != model_id:
            raise TargetWorkerError("target candidate crossed model population")
        candidates.append(candidate)
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise TargetWorkerError("duplicate target candidate_id")
    queries = [_strict_image_bytes(raw) for raw in raw_queries if isinstance(raw, dict)]
    if len(queries) != len(raw_queries):
        raise TargetWorkerError("target query is not an object")
    query_ids = [item[0] for item in queries]
    if len(query_ids) != len(set(query_ids)):
        raise TargetWorkerError("duplicate target request_id")
    return str(model_id), candidates, queries
def _strict_child_document(
    document: Mapping[str, object],
) -> tuple[
    str,
    str,
    list[TargetCandidate],
    tuple[str, str, bytes, str],
    list[tuple[str, str, bytes, str]],
]:
    required = {
        "kind",
        "worker_version",
        "model_id",
        "run_manifest_sha256",
        "candidates",
        "warmup_query",
        "queries",
    }
    if set(document) != required:
        raise TargetWorkerError("target worker request fields are not exact")
    if document.get("kind") != "restricted_target_evaluation":
        raise TargetWorkerError("target worker request kind is invalid")
    if document.get("worker_version") != TARGET_WORKER_VERSION:
        raise TargetWorkerError("target worker version mismatch")
    model_id = document.get("model_id")
    if model_id not in MODEL_IDS:
        raise TargetWorkerError("target worker model is invalid")
    run_manifest_sha256 = document.get("run_manifest_sha256")
    if (
        not isinstance(run_manifest_sha256, str)
        or len(run_manifest_sha256) != 64
        or any(c not in "0123456789abcdef" for c in run_manifest_sha256)
    ):
        raise TargetWorkerError("target run manifest SHA-256 is invalid")

    raw_candidates = document.get("candidates")
    raw_warmup = document.get("warmup_query")
    raw_queries = document.get("queries")
    if (
        not isinstance(raw_candidates, list)
        or not 1 <= len(raw_candidates) <= MAX_TARGET_CANDIDATES
    ):
        raise TargetWorkerError("target candidate count is invalid")
    if not isinstance(raw_warmup, dict):
        raise TargetWorkerError("target warmup query is not an object")
    if (
        not isinstance(raw_queries, list)
        or not 1 <= len(raw_queries) <= MAX_TARGET_QUERIES
    ):
        raise TargetWorkerError("target query count is invalid")
    candidates: list[TargetCandidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict) or set(raw) != set(TargetCandidate.__dataclass_fields__):
            raise TargetWorkerError("target candidate fields are not exact")
        candidate = TargetCandidate(**raw)
        if candidate.model_id != model_id:
            raise TargetWorkerError("target candidate crossed model population")
        candidates.append(candidate)
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise TargetWorkerError("duplicate target candidate_id")
    warmup = _strict_image_bytes(raw_warmup)
    queries = [_strict_image_bytes(raw) for raw in raw_queries if isinstance(raw, dict)]
    if len(queries) != len(raw_queries):
        raise TargetWorkerError("target query is not an object")
    query_ids = [item[0] for item in queries]
    if len(query_ids) != len(set(query_ids)):
        raise TargetWorkerError("duplicate target request_id")
    if warmup[0] in query_ids:
        raise TargetWorkerError("warmup request must be disjoint from scoring queries")
    return str(model_id), run_manifest_sha256, candidates, warmup, queries


def _evaluate_child_document(
    document: Mapping[str, object],
    *,
    engine_factory: Callable[[str], object] = TrustedTargetEngine,
) -> dict[str, object]:
    model_id, run_manifest_sha256, candidates, warmup, queries = (
        _strict_child_document(document)
    )
    engine = engine_factory(model_id)
    records: list[dict[str, object]] = []
    execution_order: list[dict[str, object]] = []
    warmup_measurements: list[dict[str, object]] = []
    for candidate in candidates:
        spec = parse_candidate_dsl(
            candidate.canonical_dsl,
            expected_model_id=model_id,
        )
        prepare = getattr(engine, "prepare", None)
        if not callable(prepare):
            raise TargetWorkerError("target engine lacks explicit prepare boundary")
        prepare(spec)
        execution_order.append(
            {"candidate_id": candidate.candidate_id, "phase": "prepare"}
        )
        warmup_id, warmup_question, warmup_bytes, warmup_hash = warmup
        warmup_measurement = engine.infer(
            spec,
            image_bytes=warmup_bytes,
            question=warmup_question,
        )
        if not isinstance(warmup_measurement, InferenceMeasurement):
            raise TargetWorkerError("warmup returned wrong measurement type")
        execution_order.append(
            {
                "candidate_id": candidate.candidate_id,
                "phase": "warmup",
                "request_id": warmup_id,
            }
        )
        warmup_measurements.append(
            {
                "candidate_id": candidate.candidate_id,
                "request_id": warmup_id,
                "image_sha256": warmup_hash,
                **warmup_measurement.to_mapping(),
            }
        )
        for request_id, question, image_bytes, image_hash in queries:
            execution_order.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "phase": "scoring",
                    "request_id": request_id,
                }
            )
            try:
                measurement = engine.infer(
                    spec,
                    image_bytes=image_bytes,
                    question=question,
                )
                if not isinstance(measurement, InferenceMeasurement):
                    raise TargetWorkerError(
                        "target engine returned wrong measurement type"
                    )
                record = {
                    "candidate_id": candidate.candidate_id,
                    "request_id": request_id,
                    "image_sha256": image_hash,
                    "status": "ok",
                    "prediction": measurement.prediction,
                    "prediction_sha256": hashlib.sha256(
                        measurement.prediction.encode("utf-8")
                    ).hexdigest(),
                    "raw_text": measurement.raw_text,
                    "input_token_count": measurement.input_token_count,
                    "generated_token_count": measurement.generated_token_count,
                    "termination_reason": measurement.termination_reason,
                    "preprocessing_seconds": measurement.preprocessing_seconds,
                    "generation_seconds": measurement.generation_seconds,
                    "total_seconds": measurement.total_seconds,
                    "peak_memory_bytes": measurement.peak_memory_bytes,
                    "error_type": None,
                    "error_message": None,
                }
            except (
                CandidateRuntimeError,
                TargetWorkerError,
                ValueError,
                RuntimeError,
            ) as exc:
                record = {
                    "candidate_id": candidate.candidate_id,
                    "request_id": request_id,
                    "image_sha256": image_hash,
                    "status": "error",
                    "prediction": "",
                    "prediction_sha256": hashlib.sha256(b"").hexdigest(),
                    "raw_text": "",
                    "input_token_count": 0,
                    "generated_token_count": 0,
                    "termination_reason": "other",
                    "preprocessing_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "total_seconds": 0.0,
                    "peak_memory_bytes": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1024],
                }
            records.append(record)
    evidence = getattr(engine, "snapshot_evidence", None)
    gpu_scope = getattr(engine, "gpu_scope_evidence", None)
    if engine_factory is TrustedTargetEngine:
        if not isinstance(evidence, dict):
            raise TargetWorkerError("production target engine lacks model evidence")
        if gpu_scope != {
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
        }:
            raise TargetWorkerError(
                "production target engine lacks GPU0 attestation"
            )
    return {
        "kind": "restricted_target_evaluation_result",
        "worker_version": TARGET_WORKER_VERSION,
        "model_id": model_id,
        "run_manifest_sha256": run_manifest_sha256,
        "model_evidence": evidence,
        "gpu_scope": gpu_scope,
        "execution_order": execution_order,
        "warmup_measurements": warmup_measurements,
        "records": records,
    }


def _child_target_main() -> int:
    network = install_network_guard()
    try:
        document = read_frame(sys.stdin.buffer)
        if sys.stdin.buffer.read(1) != b"":
            raise FrameError("trailing target worker input")
        reply = _evaluate_child_document(document)
        if network.count != 0:
            raise TargetWorkerError("target worker attempted network access")
        reply["network_attempts"] = network.count
        write_frame(sys.stdout.buffer, reply)
        return 0
    except BaseException as exc:
        try:
            write_frame(
                sys.stdout.buffer,
                {
                    "kind": "restricted_target_evaluation_error",
                    "worker_version": TARGET_WORKER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2048],
                    "network_attempts": network.count,
                },
            )
        except BaseException:
            pass
        return 1


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetWorkerError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise TargetWorkerError(f"{name} is non-finite or negative")
    return number


def _score_reply(
    reply: Mapping[str, object],
    *,
    candidates: Sequence[TargetCandidate],
    queries: Sequence[PrivateQuery],
    warmup_query: PrivateQuery,
    run_manifest_sha256: str,
) -> list[CandidateScore]:
    expected = {
        "kind",
        "worker_version",
        "model_id",
        "run_manifest_sha256",
        "model_evidence",
        "gpu_scope",
        "execution_order",
        "warmup_measurements",
        "records",
        "network_attempts",
    }
    if set(reply) != expected:
        raise TargetWorkerError("target worker reply fields are not exact")
    if reply.get("kind") != "restricted_target_evaluation_result":
        raise TargetWorkerError("target worker returned an error")
    if reply.get("worker_version") != TARGET_WORKER_VERSION:
        raise TargetWorkerError("target worker version mismatch")
    if reply.get("run_manifest_sha256") != run_manifest_sha256:
        raise TargetWorkerError("target worker run manifest mismatch")

    if reply.get("network_attempts") != 0:
        raise TargetWorkerError("target worker network attempts are nonzero")
    model_ids = {candidate.model_id for candidate in candidates}
    if len(model_ids) != 1 or reply.get("model_id") not in model_ids:
        raise TargetWorkerError("target worker reply model mismatch")
    model_id = next(iter(model_ids))
    expected_model = seed_spec(model_id)
    evidence = reply.get("model_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("model_id") != model_id
        or evidence.get("revision") != expected_model.model_revision
        or evidence.get("weight_sha256") != expected_model.weight_sha256
    ):
        raise TargetWorkerError("target model provenance is invalid")
    if reply.get("gpu_scope") != {
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
    }:
        raise TargetWorkerError("target GPU scope is invalid")
    expected_order: list[dict[str, object]] = []
    for candidate in candidates:
        expected_order.append(
            {"candidate_id": candidate.candidate_id, "phase": "prepare"}
        )
        expected_order.append(
            {
                "candidate_id": candidate.candidate_id,
                "phase": "warmup",
                "request_id": warmup_query.request_id,
            }
        )
        expected_order.extend(
            {
                "candidate_id": candidate.candidate_id,
                "phase": "scoring",
                "request_id": query.request_id,
            }
            for query in queries
        )
    if reply.get("execution_order") != expected_order:
        raise TargetWorkerError("target prepare/warmup/scoring order is invalid")
    warmups = reply.get("warmup_measurements")
    if not isinstance(warmups, list) or len(warmups) != len(candidates):
        raise TargetWorkerError("target warmup evidence is incomplete")
    warmup_hash = hashlib.sha256(warmup_query.image_path.read_bytes()).hexdigest()
    metadata_fields = {"candidate_id", "request_id", "image_sha256"}
    measurement_fields = set(InferenceMeasurement.__dataclass_fields__)
    for candidate, warmup in zip(candidates, warmups, strict=True):
        if not isinstance(warmup, dict):
            raise TargetWorkerError("target warmup evidence is not an object")
        if set(warmup) != metadata_fields | measurement_fields:
            raise TargetWorkerError("target warmup evidence fields are not exact")
        if (
            warmup["candidate_id"] != candidate.candidate_id
            or warmup["request_id"] != warmup_query.request_id
            or warmup["image_sha256"] != warmup_hash
        ):
            raise TargetWorkerError("target warmup evidence identity is invalid")
        InferenceMeasurement(
            **{key: warmup[key] for key in measurement_fields}
        )
    raw_records = reply.get("records")
    if not isinstance(raw_records, list):
        raise TargetWorkerError("target records are not a list")
    expected_pairs = [
        (candidate.candidate_id, query.request_id)
        for candidate in candidates
        for query in queries
    ]
    actual_pairs = [
        (record.get("candidate_id"), record.get("request_id"))
        for record in raw_records
        if isinstance(record, dict)
    ]
    if len(actual_pairs) != len(raw_records) or actual_pairs != expected_pairs:
        raise TargetWorkerError("target record matrix/order is incomplete")
    query_by_id = {query.request_id: query for query in queries}
    scores: list[CandidateScore] = []
    for candidate in candidates:
        subset = [
            record for record in raw_records if record["candidate_id"] == candidate.candidate_id
        ]
        correct = 0
        errors = 0
        total_seconds = 0.0
        tokens = 0
        terminations: dict[str, int] = {}
        private_records: list[dict[str, object]] = []
        for record in subset:
            if not isinstance(record, dict) or set(record) != TARGET_RECORD_FIELDS:
                raise TargetWorkerError("target record fields are not exact")
            request_id = str(record["request_id"])
            query = query_by_id[request_id]
            status = record.get("status")
            if status not in {"ok", "error"}:
                raise TargetWorkerError("target record status is invalid")
            if record.get("image_sha256") != hashlib.sha256(
                query.image_path.read_bytes()
            ).hexdigest():
                raise TargetWorkerError("target record image hash mismatch")
            prediction = record.get("prediction")
            if not isinstance(prediction, str):
                raise TargetWorkerError("target prediction is not text")
            if record.get("prediction_sha256") != hashlib.sha256(
                prediction.encode("utf-8")
            ).hexdigest():
                raise TargetWorkerError("target prediction hash mismatch")
            raw_text = record.get("raw_text")
            if not isinstance(raw_text, str):
                raise TargetWorkerError("target raw text is invalid")
            for count_name in ("input_token_count", "peak_memory_bytes"):
                count = record.get(count_name)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise TargetWorkerError(f"target {count_name} is invalid")
            if status == "ok":
                if record.get("error_type") is not None or record.get("error_message") is not None:
                    raise TargetWorkerError("successful target record contains an error")
            elif (
                not isinstance(record.get("error_type"), str)
                or not isinstance(record.get("error_message"), str)
            ):
                raise TargetWorkerError("errored target record lacks error metadata")
            latency = _finite_nonnegative(record.get("total_seconds"), "total_seconds")
            generation_seconds = _finite_nonnegative(
                record.get("generation_seconds"),
                "generation_seconds",
            )
            preprocessing_seconds = _finite_nonnegative(
                record.get("preprocessing_seconds"),
                "preprocessing_seconds",
            )
            if latency + 1e-9 < generation_seconds + preprocessing_seconds:
                raise TargetWorkerError("target latency components exceed total")
            generated = record.get("generated_token_count")
            if isinstance(generated, bool) or not isinstance(generated, int) or generated < 0:
                raise TargetWorkerError("target generated token count is invalid")
            reason = record.get("termination_reason")
            if reason not in {"eos", "length", "stop_text", "other"}:
                raise TargetWorkerError("target termination reason is invalid")
            is_error = status == "error"
            is_correct = (
                not is_error
                and prediction.strip().casefold()
                == query.expected_answer.strip().casefold()
            )
            correct += int(is_correct)
            errors += int(is_error)
            total_seconds += latency
            tokens += generated
            terminations[str(reason)] = terminations.get(str(reason), 0) + 1
            private_records.append(
                {
                    "request_id": request_id,
                    "query_key": query.query_key,
                    "figure_id": query.figure_id,
                    "qid": query.qid,
                    "question": query.question,
                    "expected_answer": query.expected_answer,
                    "prediction": prediction,
                    "correct": is_correct,
                    "status": status,
                    "image_sha256": record["image_sha256"],
                    "input_token_count": record.get("input_token_count"),
                    "generated_token_count": generated,
                    "termination_reason": reason,
                    "preprocessing_seconds": preprocessing_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": latency,
                    "peak_memory_bytes": record.get("peak_memory_bytes"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "raw_text": record.get("raw_text"),
                }
            )
        sample_count = len(queries)
        scores.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                sample_count=sample_count,
                correct_count=correct,
                accuracy=correct / sample_count,
                error_count=errors,
                total_seconds=total_seconds,
                mean_latency_seconds=total_seconds / sample_count,
                total_generated_tokens=tokens,
                termination_counts=terminations,
                private_records=tuple(private_records),
            )
        )
    return scores


class RestrictedTargetEvaluator:
    evidence_class = "production_evolution"
    def evaluate(
        self,
        *,
        model_id: str,
        candidates: Sequence[TargetCandidate],
        queries: Sequence[PrivateQuery],
        warmup_query: PrivateQuery,
        artifact_path: Path,
        timeout_seconds: float,
        run_manifest_sha256: str,
    ) -> list[CandidateScore]:
        candidate_values = list(candidates)
        query_values = list(queries)
        if not isinstance(warmup_query, PrivateQuery):
            raise TargetWorkerError("target warmup query is invalid")
        if warmup_query.request_id in {item.request_id for item in query_values}:
            raise TargetWorkerError("target warmup overlaps scoring queries")

        if not candidate_values or not query_values:
            raise TargetWorkerError("target evaluation inputs are empty")
        if len(candidate_values) > MAX_TARGET_CANDIDATES:
            raise TargetWorkerError("too many target candidates")
        if len(query_values) > MAX_TARGET_QUERIES:
            raise TargetWorkerError("too many target queries")
        if any(candidate.model_id != model_id for candidate in candidate_values):
            raise TargetWorkerError("target model boundary mismatch")
        if (
            not isinstance(run_manifest_sha256, str)
            or len(run_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in run_manifest_sha256)
        ):
            raise TargetWorkerError("target run manifest SHA-256 is invalid")

        document = {
            "kind": "restricted_target_evaluation",
            "worker_version": TARGET_WORKER_VERSION,
            "model_id": model_id,
            "run_manifest_sha256": run_manifest_sha256,
            "candidates": [item.to_worker_mapping() for item in candidate_values],
            "queries": [item.to_worker_mapping() for item in query_values],
            "warmup_query": warmup_query.to_worker_mapping(),
        }
        try:
            child = run_trusted_child(
                module="evolution.target_worker",
                function="_child_target_main",
                input_bytes=encode_frame(document),
                timeout_seconds=timeout_seconds,
            )
        except ChildTimeout as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "worker_version": TARGET_WORKER_VERSION,
                    "model_id": model_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "child_reaped": True,
                    "candidate_ids": [item.candidate_id for item in candidate_values],
                    "request_ids": [item.request_id for item in query_values],
                },
            )
            raise TargetWorkerError("target child timed out and was reaped") from exc
        try:
            reply = decode_frame(child.stdout)
        except FrameError as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "worker_version": TARGET_WORKER_VERSION,
                    "model_id": model_id,
                    "error_type": type(exc).__name__,
                    "error_message": "target child stdout is not one strict frame",
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "stdout_size_bytes": len(child.stdout),
                    "stdout_sha256": hashlib.sha256(child.stdout).hexdigest(),
                },
            )
            raise TargetWorkerError("target child stdout is not one strict frame") from exc
        if not child.reaped:
            raise TargetWorkerError("target child was not reaped")
        self.last_child_elapsed_seconds = child.elapsed_seconds
        if child.returncode != 0:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "worker_version": TARGET_WORKER_VERSION,
                    "model_id": model_id,
                    "error_type": str(reply.get("error_type", "unknown")),
                    "error_message": str(reply.get("error_message", "no detail")),
                    "network_attempts": reply.get("network_attempts"),
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "candidate_ids": [item.candidate_id for item in candidate_values],
                    "request_ids": [item.request_id for item in query_values],
                },
            )
            raise TargetWorkerError(
                "target child failed: "
                + str(reply.get("error_type", "unknown"))
                + ": "
                + str(reply.get("error_message", "no detail"))
            )
        scores = _score_reply(
            reply,
            candidates=candidate_values,
            queries=query_values,
            warmup_query=warmup_query,
            run_manifest_sha256=run_manifest_sha256,
        )
        artifact = {
            "worker_version": TARGET_WORKER_VERSION,
            "model_id": model_id,
            "run_manifest_sha256": run_manifest_sha256,
            "candidate_sources": [
                {
                    "candidate_id": item.candidate_id,
                    "model_id": item.model_id,
                    "rendered_sha256": item.rendered_sha256,
                    "canonical_dsl_sha256": hashlib.sha256(
                        item.canonical_dsl.encode("utf-8")
                    ).hexdigest(),
                }
                for item in candidate_values
            ],
            "child_pid": child.pid,
            "child_returncode": child.returncode,
            "child_reaped": child.reaped,
            "child_elapsed_seconds": child.elapsed_seconds,
            "child_shell": child.shell,
            "child_start_new_session": child.start_new_session,
            "child_stderr_size_bytes": len(child.stderr),
            "child_stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
            "child_stderr_utf8": child.stderr.decode("utf-8", errors="replace")[:65536],
            "network_attempts": reply["network_attempts"],
            "model_evidence": reply["model_evidence"],
            "gpu_scope": reply["gpu_scope"],
            "warmup_request_id": warmup_query.request_id,
            "warmup_measurements": reply["warmup_measurements"],
            "execution_order": reply["execution_order"],
            "measurement_protocol": "load-excluded; fixed warmup per candidate; candidate-major fixed query order; CUDA pre/post sync",
            "query_order": [item.request_id for item in query_values],
            "worker_input_disclosure": [
                "opaque request_id",
                "question",
                "image bytes",
                "actual image SHA-256",
            ],
            "worker_input_forbidden": [
                "query_key",
                "figure_id",
                "qid",
                "split",
                "image_path",
                "expected_answer",
                "correctness",
            ],
            "candidate_execution": "byte-exact trusted renderer template only",
            "scores": [score.to_mapping() for score in scores],
        }
        atomic_create_json(artifact_path, artifact)
        return scores


__all__ = [
    "CandidateScore",
    "PrivateQuery",
    "RestrictedTargetEvaluator",
    "TARGET_WORKER_VERSION",
    "TargetCandidate",
    "TargetWorkerError",
]
