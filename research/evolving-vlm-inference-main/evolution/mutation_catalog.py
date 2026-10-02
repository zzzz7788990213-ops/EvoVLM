"""Frozen catalog for LLM-guided, focus-only field-patch mutation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from .dsl import (
    INSTRUCT_MODEL_ID,
    MODEL_IDS,
    THINKING_MODEL_ID,
    ProgramSpec,
    canonical_candidate_source,
    parse_candidate_dsl,
    seed_spec,
)
from .process_safety import PROJECT_ROOT, atomic_create_json
from .renderer import (
    render_candidate_program,
    rendered_program_sha256,
    validate_rendered_program,
)


CATALOG_VERSION = "phase2-mutation-catalog-v1"
CATALOG_MANIFEST_VERSION = "phase2-mutation-catalog-manifest-v1"
CATALOG_ROOT = PROJECT_ROOT / "data/phase2_mutation_catalog"
CATALOG_PATH = CATALOG_ROOT / "catalog.json"
CATALOG_MANIFEST_PATH = CATALOG_ROOT / "catalog_manifest.json"
FOCUS_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_template": ("prompt_template",),
    "token_budget_and_extraction": ("max_new_tokens", "extraction_strategy"),
    "preprocessing_and_pixel_bounds": (
        "preprocessing_option",
        "min_pixels",
        "max_pixels",
    ),
    "answer_extraction": ("extraction_strategy", "postprocessing_strategy"),
    "system_instruction": ("system_instruction",),
    "token_budget_and_postprocessing": (
        "max_new_tokens",
        "postprocessing_strategy",
    ),
}
_CHOICE_FIELDS = {
    "choice_id",
    "catalog_version",
    "applicable_model",
    "focus",
    "description",
    "patch",
    "allowed_changed_fields",
    "prohibited_changed_fields",
    "expected_runtime_effective_change",
    "canonical_patch_sha256",
}
_SAFE_ID = re.compile(r"[A-Z0-9_]{4,64}")
_FORBIDDEN_CATALOG_MARKERS = (
    "query_key",
    "figure_id",
    "subq_idx",
    "ground_truth",
    "expected_answer",
    "sample_id",
    "question_id",
    "/scratch/",
    "http://",
    "https://",
    "file://",
    "__import__",
    "subprocess",
    "socket.",
    "open(",
    "eval(",
    "exec(",
)
_ALL_SPEC_FIELDS = tuple(ProgramSpec.__dataclass_fields__)


class MutationCatalogError(ValueError):
    """The frozen patch catalog or one selected patch is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _choice_id(model_id: str, focus: str, index: int) -> str:
    model = "I" if model_id == INSTRUCT_MODEL_ID else "T"
    focus_code = {
        "prompt_template": "PT",
        "token_budget_and_extraction": "TE",
        "preprocessing_and_pixel_bounds": "PP",
        "answer_extraction": "AE",
        "system_instruction": "SI",
        "token_budget_and_postprocessing": "TP",
    }[focus]
    return f"{model}_{focus_code}_{index + 1:02d}"


def _prompt_patches(model_id: str) -> list[tuple[str, dict[str, object]]]:
    ending = (
        "Return only the concise answer enclosed in <final> and </final>. "
        "Question: {question}"
    )
    if model_id == THINKING_MODEL_ID:
        ending = (
            "Use the model's native reasoning, then return only the concise final "
            "answer enclosed in <final> and </final>. Question: {question}"
        )
    stems = (
        "Inspect the relevant plot region, axes, legend, and annotations carefully. "
        "Align visible evidence with what the question asks. ",
        "Determine whether the question requests a value, label, list, comparison, "
        "or boolean relation, and read the corresponding chart evidence precisely. ",
        "Trace the question to the exact plotted element and verify units, scale, "
        "legend identity, and annotations before answering. ",
        "Use only evidence visible in the scientific figure; do not invent missing "
        "or inapplicable chart elements. ",
    )
    return [
        (
            description,
            {"prompt_template": stem + ending},
        )
        for description, stem in zip(
            (
                "Careful plot-region, axis, legend, and annotation alignment.",
                "Generic task-type recognition before concise answering.",
                "Exact plotted-element and scale verification.",
                "Visible-evidence-only scientific chart reading.",
            ),
            stems,
            strict=True,
        )
    ]


def _system_patches() -> list[tuple[str, dict[str, object]]]:
    values = (
        "You are a careful scientific chart reader. Base the answer only on visible figure evidence.",
        "You are a precise scientific-figure analyst. Check axes, units, legends, and annotations.",
        "Read charts conservatively: distinguish observed evidence from missing or inapplicable elements.",
        "Resolve the element named by the question before producing one concise final response.",
    )
    return [(f"Generic system instruction {index + 1}.", {"system_instruction": value}) for index, value in enumerate(values)]


def _token_extraction_patches(model_id: str) -> list[tuple[str, dict[str, object]]]:
    values = (
        (
            (64, "final_tag"),
            (80, "final_tag"),
            (64, "boxed_or_final"),
            (96, "last_nonempty_line"),
        )
        if model_id == INSTRUCT_MODEL_ID
        else (
            (320, "final_tag"),
            (448, "final_tag"),
            (384, "boxed_or_final"),
            (512, "last_nonempty_line"),
        )
    )
    return [
        (
            f"Use token cap {cap} with {strategy} extraction.",
            {"max_new_tokens": cap, "extraction_strategy": strategy},
        )
        for cap, strategy in values
    ]


def _preprocessing_patches() -> list[tuple[str, dict[str, object]]]:
    return [
        ("Convert the chart input deterministically to RGB.", {"preprocessing_option": "rgb"}),
        (
            "Apply deterministic RGB autocontrast before model processing.",
            {"preprocessing_option": "autocontrast_rgb"},
        ),
        (
            "Apply deterministic RGB sharpening before model processing.",
            {"preprocessing_option": "sharpen_rgb"},
        ),
        (
            "Use RGB with broader preapproved pixel bounds.",
            {
                "preprocessing_option": "rgb",
                "min_pixels": 100352,
                "max_pixels": 1605632,
            },
        ),
    ]


def _answer_extraction_patches() -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "Accept boxed output or the final tag.",
            {"extraction_strategy": "boxed_or_final"},
        ),
        (
            "Use final-tag extraction and normalize whitespace.",
            {
                "extraction_strategy": "final_tag",
                "postprocessing_strategy": "normalize_whitespace",
            },
        ),
        (
            "Use the last nonempty line with strict stripping.",
            {
                "extraction_strategy": "last_nonempty_line",
                "postprocessing_strategy": "strip",
            },
        ),
        (
            "Use final-tag extraction and strip terminal punctuation.",
            {
                "extraction_strategy": "final_tag",
                "postprocessing_strategy": "strip_terminal_punctuation",
            },
        ),
    ]


def _token_post_patches(model_id: str) -> list[tuple[str, dict[str, object]]]:
    values = (
        (
            (64, "strip"),
            (48, "normalize_whitespace"),
            (80, "strip_terminal_punctuation"),
            (96, "normalize_numeric"),
        )
        if model_id == INSTRUCT_MODEL_ID
        else (
            (320, "strip"),
            (384, "normalize_whitespace"),
            (448, "strip_terminal_punctuation"),
            (512, "normalize_numeric"),
        )
    )
    return [
        (
            f"Use token cap {cap} with {strategy} postprocessing.",
            {"max_new_tokens": cap, "postprocessing_strategy": strategy},
        )
        for cap, strategy in values
    ]


def _focus_options(model_id: str, focus: str) -> list[tuple[str, dict[str, object]]]:
    if focus == "prompt_template":
        return _prompt_patches(model_id)
    if focus == "system_instruction":
        return _system_patches()
    if focus == "token_budget_and_extraction":
        return _token_extraction_patches(model_id)
    if focus == "preprocessing_and_pixel_bounds":
        return _preprocessing_patches()
    if focus == "answer_extraction":
        return _answer_extraction_patches()
    if focus == "token_budget_and_postprocessing":
        return _token_post_patches(model_id)
    raise MutationCatalogError("unknown mutation focus")


def _patch_hash_material(choice: Mapping[str, object]) -> dict[str, object]:
    return {
        key: choice[key]
        for key in (
            "choice_id",
            "catalog_version",
            "applicable_model",
            "focus",
            "patch",
        )
    }


def build_catalog_document() -> dict[str, object]:
    choices: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        for focus in FOCUS_FIELDS:
            allowed_focus = set(FOCUS_FIELDS[focus])
            for index, (description, patch) in enumerate(
                _focus_options(model_id, focus)
            ):
                patch_fields = sorted(patch)
                if not set(patch_fields) <= allowed_focus:
                    raise AssertionError("built-in catalog patch escaped focus")
                choice: dict[str, object] = {
                    "choice_id": _choice_id(model_id, focus, index),
                    "catalog_version": CATALOG_VERSION,
                    "applicable_model": model_id,
                    "focus": focus,
                    "description": description,
                    "patch": dict(patch),
                    "allowed_changed_fields": patch_fields,
                    "prohibited_changed_fields": sorted(
                        set(_ALL_SPEC_FIELDS) - allowed_focus
                    ),
                    "expected_runtime_effective_change": (
                        "The trusted patcher changes at least one focus field, "
                        "passes ProgramSpec policy, renders a different program, "
                        "and is not a population duplicate."
                    ),
                }
                choice["canonical_patch_sha256"] = _sha256_bytes(
                    _canonical_bytes(_patch_hash_material(choice))
                )
                choices.append(choice)
    return {
        "catalog_version": CATALOG_VERSION,
        "selection_method": "single-token-label-argmax",
        "choices": choices,
    }


def _validate_scalar_text(value: object, label: str, *, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise MutationCatalogError(f"{label} is invalid")
    lowered = value.casefold()
    if any(marker in lowered for marker in _FORBIDDEN_CATALOG_MARKERS):
        raise MutationCatalogError(f"{label} contains a forbidden marker")
    return value


def validate_choice(choice: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(choice, Mapping) or set(choice) != _CHOICE_FIELDS:
        raise MutationCatalogError("catalog choice fields are not exact")
    value = dict(choice)
    choice_id = _validate_scalar_text(value["choice_id"], "choice_id", limit=64)
    if _SAFE_ID.fullmatch(choice_id) is None:
        raise MutationCatalogError("choice_id is unsafe")
    if value["catalog_version"] != CATALOG_VERSION:
        raise MutationCatalogError("choice catalog version is stale")
    model_id = value["applicable_model"]
    focus = value["focus"]
    if model_id not in MODEL_IDS or focus not in FOCUS_FIELDS:
        raise MutationCatalogError("choice model/focus is invalid")
    _validate_scalar_text(value["description"], "description")
    _validate_scalar_text(
        value["expected_runtime_effective_change"],
        "expected_runtime_effective_change",
    )
    patch = value["patch"]
    if not isinstance(patch, dict) or not patch:
        raise MutationCatalogError("choice patch is empty or not an object")
    if any(not isinstance(key, str) for key in patch):
        raise MutationCatalogError("choice patch keys are invalid")
    allowed = value["allowed_changed_fields"]
    prohibited = value["prohibited_changed_fields"]
    if (
        not isinstance(allowed, list)
        or allowed != sorted(patch)
        or len(allowed) != len(set(allowed))
        or not set(allowed) <= set(FOCUS_FIELDS[str(focus)])
    ):
        raise MutationCatalogError("choice allowed fields are invalid")
    expected_prohibited = sorted(set(_ALL_SPEC_FIELDS) - set(FOCUS_FIELDS[str(focus)]))
    if prohibited != expected_prohibited:
        raise MutationCatalogError("choice prohibited fields are invalid")
    actual_patch_hash = _sha256_bytes(
        _canonical_bytes(_patch_hash_material(value))
    )
    if value["canonical_patch_sha256"] != actual_patch_hash:
        raise MutationCatalogError("choice canonical patch hash mismatch")
    parent = seed_spec(str(model_id))
    child = apply_choice(parent, value)
    if canonical_candidate_source(child) == canonical_candidate_source(parent):
        raise MutationCatalogError("catalog patch is seed-equivalent")
    return value


def validate_catalog_document(document: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(document, Mapping) or set(document) != {
        "catalog_version",
        "selection_method",
        "choices",
    }:
        raise MutationCatalogError("catalog fields are not exact")
    if document["catalog_version"] != CATALOG_VERSION:
        raise MutationCatalogError("catalog version is stale")
    if document["selection_method"] != "single-token-label-argmax":
        raise MutationCatalogError("catalog selection method is invalid")
    raw_choices = document["choices"]
    if not isinstance(raw_choices, list) or not raw_choices:
        raise MutationCatalogError("catalog choices are empty")
    choices = [validate_choice(item) for item in raw_choices]
    identifiers = [str(item["choice_id"]) for item in choices]
    if len(identifiers) != len(set(identifiers)):
        raise MutationCatalogError("catalog choice_id is duplicated")
    for model_id in MODEL_IDS:
        for focus in FOCUS_FIELDS:
            count = sum(
                item["applicable_model"] == model_id and item["focus"] == focus
                for item in choices
            )
            if count < 4:
                raise MutationCatalogError("catalog focus has fewer than four choices")
    return {
        "catalog_version": CATALOG_VERSION,
        "selection_method": "single-token-label-argmax",
        "choices": choices,
    }


def build_catalog_manifest(document: Mapping[str, object]) -> dict[str, object]:
    validated = validate_catalog_document(document)
    canonical = _artifact_json_bytes(validated)
    choices = validated["choices"]
    assert isinstance(choices, list)
    return {
        "manifest_version": CATALOG_MANIFEST_VERSION,
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": _sha256_bytes(canonical),
        "choice_count": len(choices),
        "ordered_choice_ids": [item["choice_id"] for item in choices],
        "ordered_patch_sha256s": [
            item["canonical_patch_sha256"] for item in choices
        ],
    }


def write_frozen_catalog(root: Path = CATALOG_ROOT) -> tuple[Path, Path]:
    document = validate_catalog_document(build_catalog_document())
    manifest = build_catalog_manifest(document)
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    manifest_path = root / "catalog_manifest.json"
    atomic_create_json(catalog_path, document)
    actual_catalog_sha = _sha256_file(catalog_path)
    if manifest["catalog_sha256"] != actual_catalog_sha:
        raise MutationCatalogError("written catalog serialization hash mismatch")
    atomic_create_json(manifest_path, manifest)
    load_frozen_catalog(root)
    return catalog_path, manifest_path


def load_frozen_catalog(
    root: Path = CATALOG_ROOT,
) -> tuple[dict[str, object], dict[str, object]]:
    catalog_path = root / "catalog.json"
    manifest_path = root / "catalog_manifest.json"
    for path in (catalog_path, manifest_path):
        if path.is_symlink() or not path.is_file():
            raise MutationCatalogError("frozen catalog artifact is not regular")
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutationCatalogError("frozen catalog is not strict JSON") from exc
    validated = validate_catalog_document(document)
    expected_manifest = build_catalog_manifest(validated)
    if manifest != expected_manifest:
        raise MutationCatalogError("catalog manifest closure mismatch")
    if _sha256_file(catalog_path) != expected_manifest["catalog_sha256"]:
        raise MutationCatalogError("catalog file SHA-256 mismatch")
    return validated, expected_manifest


def choices_for(
    document: Mapping[str, object],
    *,
    model_id: str,
    focus: str,
) -> list[dict[str, object]]:
    validated = validate_catalog_document(document)
    values = [
        dict(item)
        for item in validated["choices"]
        if item["applicable_model"] == model_id and item["focus"] == focus
    ]
    if len(values) < 4:
        raise MutationCatalogError("catalog selection set is incomplete")
    return values


def choice_by_id(
    document: Mapping[str, object],
    choice_id: str,
) -> dict[str, object]:
    matches = [
        dict(item)
        for item in validate_catalog_document(document)["choices"]
        if item["choice_id"] == choice_id
    ]
    if len(matches) != 1:
        raise MutationCatalogError("choice_id is absent or ambiguous")
    return matches[0]


def apply_choice(parent: ProgramSpec, choice: Mapping[str, object]) -> ProgramSpec:
    if not isinstance(parent, ProgramSpec):
        raise MutationCatalogError("parent is not a ProgramSpec")
    value = dict(choice)
    if value.get("applicable_model") != parent.model_id:
        raise MutationCatalogError("catalog choice crosses model population")
    focus = value.get("focus")
    if focus not in FOCUS_FIELDS:
        raise MutationCatalogError("catalog choice focus is invalid")
    patch = value.get("patch")
    if not isinstance(patch, dict) or not patch:
        raise MutationCatalogError("catalog choice patch is invalid")
    if not set(patch) <= set(FOCUS_FIELDS[str(focus)]):
        raise MutationCatalogError("catalog patch escapes its focus")
    try:
        child = replace(parent, **patch)
    except TypeError as exc:
        raise MutationCatalogError("catalog patch fields are invalid") from exc
    changed = {
        field
        for field in _ALL_SPEC_FIELDS
        if getattr(parent, field) != getattr(child, field)
    }
    if not changed:
        raise MutationCatalogError("catalog patch is parent-equivalent")
    if not changed <= set(FOCUS_FIELDS[str(focus)]):
        raise MutationCatalogError("catalog patch changed prohibited fields")
    source = render_candidate_program(child)
    validate_rendered_program(source, expected_model_id=parent.model_id)
    if rendered_program_sha256(source) == rendered_program_sha256(
        render_candidate_program(parent)
    ):
        raise MutationCatalogError("catalog patch is not runtime-effective")
    return child


def rank_choice_ids(
    choices: Sequence[Mapping[str, object]],
    scores: Mapping[str, float],
) -> list[str]:
    identifiers = [str(item["choice_id"]) for item in choices]
    if set(scores) != set(identifiers):
        raise MutationCatalogError("ranking scores do not match legal choices")
    indexed: list[tuple[float, int, str]] = []
    for index, choice_id in enumerate(identifiers):
        score = scores[choice_id]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise MutationCatalogError("choice score is not numeric")
        numeric = float(score)
        if not (-float("inf") < numeric < float("inf")):
            raise MutationCatalogError("choice score is not finite")
        indexed.append((-numeric, index, choice_id))
    return [item[2] for item in sorted(indexed)]


def select_ranked_choice(
    *,
    parent_dsl: str,
    model_id: str,
    choices: Sequence[Mapping[str, object]],
    ranking: Sequence[str],
    population_program_sha256s: Sequence[str],
) -> tuple[dict[str, object], ProgramSpec, list[dict[str, str]]]:
    parent = parse_candidate_dsl(parent_dsl, expected_model_id=model_id)
    by_id = {str(item["choice_id"]): dict(item) for item in choices}
    if len(by_id) != len(choices) or set(ranking) != set(by_id) or len(ranking) != len(by_id):
        raise MutationCatalogError("ranking is not an exact choice permutation")
    population = set(population_program_sha256s)
    skipped: list[dict[str, str]] = []
    for choice_id in ranking:
        choice = by_id[choice_id]
        try:
            child = apply_choice(parent, choice)
        except MutationCatalogError as exc:
            if str(exc) == "catalog patch is parent-equivalent":
                skipped.append(
                    {
                        "choice_id": choice_id,
                        "reason": "parent_equivalent",
                    }
                )
                continue
            raise
        digest = rendered_program_sha256(render_candidate_program(child))
        if digest in population:
            skipped.append({"choice_id": choice_id, "reason": "population_duplicate"})
            continue
        return choice, child, skipped
    raise MutationCatalogError("no legal nonduplicate ranked choice remains")


def catalog_hashes(root: Path = CATALOG_ROOT) -> dict[str, str]:
    document, manifest = load_frozen_catalog(root)
    del document
    return {
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": str(manifest["catalog_sha256"]),
        "catalog_manifest_sha256": _sha256_file(root / "catalog_manifest.json"),
    }


def main() -> int:
    catalog_path, manifest_path = write_frozen_catalog()
    print(
        json.dumps(
            {
                "catalog": str(catalog_path),
                "catalog_sha256": _sha256_file(catalog_path),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
