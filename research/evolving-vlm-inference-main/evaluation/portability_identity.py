"""CPU-only binding of portable programs to sealed Single1 evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Mapping

from evaluation import portable_reproduce as replay


PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
PROGRAMS = (
    "manual_instruct",
    "evolved_instruct",
    "manual_thinking",
    "evolved_thinking",
)
DEPENDENCIES = {
    "manual_instruct": ("manual_common.py", "model_locator.py"),
    "manual_thinking": ("manual_common.py", "model_locator.py"),
    "evolved_instruct": (
        "evolution/candidate_runtime.py", "evolution/dsl.py", "model_locator.py",
    ),
    "evolved_thinking": (
        "evolution/candidate_runtime.py", "evolution/dsl.py", "model_locator.py",
    ),
}
SCORING = "prediction.strip().casefold() == expected.strip().casefold()"
QUESTION = "What value is shown?"
_INSTRUCT = {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "model_revision": "78448d793a7eb2f7a987a1da76d464384aa1becd",
    "processor_revision": "78448d793a7eb2f7a987a1da76d464384aa1becd",
    "weight_sha256": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
}
_THINKING = {
    "model_id": "Qwen/Qwen3-VL-2B-Thinking",
    "model_revision": "d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3",
    "processor_revision": "d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3",
    "weight_sha256": "d889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888",
}
_FIXED = {
    "dtype": "bfloat16", "attention": "sdpa", "quantization": "none",
    "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
    "use_cache": True, "scoring": SCORING,
}


class PortabilityIdentityError(RuntimeError):
    """An inference-semantic or sealed-reference invariant failed."""


def _manual_sealed(model: Mapping[str, object], *, thinking: bool) -> dict[str, object]:
    if thinking:
        prompt = (
            "Analyze the chart carefully using the model's native reasoning. "
            "Finish with the concise answer enclosed in <final> and </final>. "
            f"Question: {QUESTION}"
        )
    else:
        prompt = (
            "Answer the chart question concisely. Return only the answer "
            "enclosed in <final> and </final>; do not add an explanation. "
            f"Question: {QUESTION}"
        )
    return {
        **model, **_FIXED, "system_instruction": "", "prompt": prompt,
        "reasoning_mode": "thinking" if thinking else "direct",
        "max_new_tokens": 384 if thinking else 48,
        "min_pixels": "processor_default", "max_pixels": "processor_default",
        "preprocessing_option": "qwen_default",
        "extraction_strategy": "terminal_final_tag_or_conservative_raw",
        "postprocessing_strategy": "strip",
    }


def _evolved_sealed(model: Mapping[str, object], *, thinking: bool) -> dict[str, object]:
    if thinking:
        prompt = (
            "Use only evidence visible in the scientific figure; do not invent "
            "missing or inapplicable chart elements. Use the model's native "
            "reasoning, then return only the concise final answer enclosed in "
            f"<final> and </final>. Question: {QUESTION}"
        )
    else:
        prompt = (
            "Answer the chart question concisely. Return only the answer "
            "enclosed in <final> and </final>; do not add an explanation. "
            f"Question: {QUESTION}"
        )
    return {
        **model, **_FIXED, "system_instruction": "", "prompt": prompt,
        "reasoning_mode": "thinking" if thinking else "direct",
        "max_new_tokens": 384 if thinking else 64,
        "min_pixels": 200704, "max_pixels": 1003520,
        "preprocessing_option": "qwen_default",
        "extraction_strategy": "final_tag" if thinking else "boxed_or_final",
        "postprocessing_strategy": (
            "strip_terminal_punctuation" if thinking else "strip"
        ),
    }


SEALED_IDENTITIES = {
    "manual_instruct": _manual_sealed(_INSTRUCT, thinking=False),
    "manual_thinking": _manual_sealed(_THINKING, thinking=True),
    "evolved_instruct": _evolved_sealed(_INSTRUCT, thinking=False),
    "evolved_thinking": _evolved_sealed(_THINKING, thinking=True),
}


def _sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(relative: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise PortabilityIdentityError(f"unsafe dependency: {relative}")
    return {
        "path": relative, "sha256": replay.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def current_identity(program: str) -> dict[str, object]:
    if program not in PROGRAMS:
        raise PortabilityIdentityError(f"unknown optimized program: {program}")
    module = importlib.import_module(program)
    if program.startswith("manual_"):
        import manual_common
        spec = (
            manual_common.INSTRUCT_SPEC
            if program.endswith("instruct") else manual_common.THINKING_SPEC
        )
        return {
            "model_id": spec.model_id, "model_revision": spec.revision,
            "processor_revision": spec.revision,
            "weight_sha256": spec.weight_sha256, "dtype": manual_common.DTYPE,
            "attention": manual_common.ATTENTION_IMPLEMENTATION,
            "quantization": manual_common.QUANTIZATION,
            "system_instruction": "",
            "prompt": manual_common.build_user_prompt(spec, QUESTION),
            "reasoning_mode": (
                "direct" if spec.mode == "instruct" else "thinking"
            ),
            "max_new_tokens": spec.max_new_tokens,
            "min_pixels": "processor_default",
            "max_pixels": "processor_default",
            "preprocessing_option": "qwen_default",
            "extraction_strategy": "terminal_final_tag_or_conservative_raw",
            "postprocessing_strategy": "strip", "do_sample": False,
            "num_beams": 1, "num_return_sequences": 1,
            "use_cache": True, "scoring": SCORING,
        }
    spec = dict(module.CANDIDATE_SPEC)
    return {
        "model_id": module.MODEL_ID, "model_revision": module.MODEL_REVISION,
        "processor_revision": module.PROCESSOR_REVISION,
        "weight_sha256": module.MODEL_WEIGHT_SHA256, "dtype": module.DTYPE,
        "attention": module.ATTENTION_IMPLEMENTATION,
        "quantization": module.QUANTIZATION,
        "system_instruction": spec["system_instruction"],
        "prompt": str(spec["prompt_template"]).replace("{question}", QUESTION),
        "reasoning_mode": spec["reasoning_mode"],
        "max_new_tokens": spec["max_new_tokens"],
        "min_pixels": spec["min_pixels"], "max_pixels": spec["max_pixels"],
        "preprocessing_option": spec["preprocessing_option"],
        "extraction_strategy": spec["extraction_strategy"],
        "postprocessing_strategy": spec["postprocessing_strategy"],
        "do_sample": spec["do_sample"], "num_beams": spec["num_beams"],
        "num_return_sequences": spec["num_return_sequences"],
        "use_cache": spec["use_cache"], "scoring": SCORING,
    }


def verify_algorithm_identity(program: str) -> dict[str, object]:
    current = current_identity(program)
    sealed = SEALED_IDENTITIES[program]
    if current != sealed:
        changed = sorted(
            key for key in set(current) | set(sealed)
            if current.get(key) != sealed.get(key)
        )
        raise PortabilityIdentityError(
            f"semantic identity differs for {program}: {changed}"
        )
    return {
        "fields": current, "identity_sha256": _sha(current),
        "dependencies": [_file_record(item) for item in DEPENDENCIES[program]],
    }


def _formal_record(final: Mapping[str, object], relative: str) -> dict[str, object]:
    record = replay._hash_index(final).get(relative)
    if record is None:
        raise PortabilityIdentityError(f"unbound formal artifact: {relative}")
    path = replay.safe_file(relative)
    if (
        replay.sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["size_bytes"]
    ):
        raise PortabilityIdentityError(f"formal artifact changed: {relative}")
    return dict(record)


def load_old_single_reference(
    program: str,
    *, image_path: Path | None = None,
) -> dict[str, object]:
    identity = verify_algorithm_identity(program)
    final = replay.load_json(replay.FINAL_MANIFEST)
    source = _formal_record(final, f"{program}.py")
    plan_path = replay._record_hash(
        final["composition"]["retained_campaign_plan"], "retained plan"
    )
    plan = replay.load_json(plan_path)
    matches = [
        item for item in plan.get("schedule", [])
        if item.get("program") == program
        and item.get("manifest") == "single1"
        and item.get("repeat") == 1
    ]
    if len(matches) != 1:
        raise PortabilityIdentityError(f"non-unique old Single1: {program}")
    relative = (
        plan_path.parent / str(matches[0]["result"])
    ).relative_to(PROJECT_ROOT).as_posix()
    result_record = _formal_record(final, relative)
    document = replay.load_json(replay.safe_file(relative))
    program_record = document.get("program_record")
    evaluation = document.get("evaluation")
    if not isinstance(program_record, dict) or not isinstance(evaluation, dict):
        raise PortabilityIdentityError("old Single1 structure differs")
    if program_record.get("source") != source:
        raise PortabilityIdentityError("old wrapper source differs")
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise PortabilityIdentityError("old Single1 record count differs")
    record = records[0]
    if image_path is not None:
        if (
            image_path.is_symlink()
            or not image_path.is_file()
            or image_path.name != f"{record['figure_id']}.jpg"
            or replay.sha256_file(image_path) != record.get("image_sha256")
        ):
            raise PortabilityIdentityError("old Single1 image hash differs")
    fields = identity["fields"]
    for key in ("model_id", "model_revision", "weight_sha256"):
        if program_record.get(key) != fields[key]:
            raise PortabilityIdentityError(f"old Single1 {key} differs")
    return {
        "program": program, "source": source,
        "old_dependencies": program_record.get("dependencies"),
        "current_dependencies": identity["dependencies"],
        "algorithm_identity_sha256": identity["identity_sha256"],
        "model_id": fields["model_id"],
        "model_revision": fields["model_revision"],
        "processor_revision": fields["processor_revision"],
        "weight_sha256": fields["weight_sha256"],
        "query_key": record.get("query_key"),
        "question": record.get("question"),
        "question_sha256": hashlib.sha256(
            str(record.get("question")).encode("utf-8")
        ).hexdigest(),
        "image_sha256": record.get("image_sha256"),
        "prediction": record.get("prediction"),
        "prediction_sha256": record.get("prediction_sha256"),
        "correct": record.get("correct"), "old_result": result_record,
    }


def compare_fresh_single(program: str, path: Path) -> dict[str, object]:
    fresh = replay.load_json(path)
    records = fresh.get("records")
    if fresh.get("program") != program or fresh.get("num_evaluated") != 1:
        raise PortabilityIdentityError("fresh Single1 identity differs")
    if not isinstance(records, list) or len(records) != 1:
        raise PortabilityIdentityError("fresh Single1 record count differs")
    record = records[0]
    image = Path(str(record.get("image_path")))
    reference = load_old_single_reference(program, image_path=image)
    comparisons = {
        "query_key": record.get("query_key") == reference["query_key"],
        "question": record.get("question") == reference["question"],
        "image": (
            image.is_file() and not image.is_symlink()
            and replay.sha256_file(image) == reference["image_sha256"]
        ),
        "output": record.get("response") == reference["prediction"],
        "correct": record.get("correct") == reference["correct"],
    }
    if not all(comparisons.values()):
        failed = sorted(key for key, value in comparisons.items() if not value)
        raise PortabilityIdentityError(
            f"fresh Single1 differs for {program}: {failed}"
        )
    response_sha = hashlib.sha256(
        str(record["response"]).encode("utf-8")
    ).hexdigest()
    if response_sha != reference["prediction_sha256"]:
        raise PortabilityIdentityError("fresh prediction hash differs")
    return {
        "program": program, "comparison": comparisons,
        "program_source": reference["source"],
        "old_dependencies": reference["old_dependencies"],
        "current_dependencies": reference["current_dependencies"],
        "algorithm_identity_sha256": reference["algorithm_identity_sha256"],
        "model_id": reference["model_id"],
        "model_revision": reference["model_revision"],
        "processor_revision": reference["processor_revision"],
        "model_weight_sha256": reference["weight_sha256"],
        "query_key": reference["query_key"],
        "question_sha256": reference["question_sha256"],
        "image_sha256": reference["image_sha256"],
        "prediction_sha256": response_sha,
        "old_result": reference["old_result"],
        "fresh_result": {
            "path": path.name, "sha256": replay.sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
    }
