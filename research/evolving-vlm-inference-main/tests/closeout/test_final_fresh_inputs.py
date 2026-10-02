"""CPU tests for independent fresh input copies."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_final_fresh_inputs as inputs


def test_real_final_fresh_input_manifest_is_128_independent_regular_files() -> None:
    document = inputs.verify(inputs.DEFAULT_OUTPUT)
    assert document["sample_count"] == 128
    assert document["independent_regular_file_count"] == 128
    assert len({item["relative_path"] for item in document["records"]}) == 128


def test_fresh_input_tamper_and_reuse_are_rejected(tmp_path: Path) -> None:
    source = inputs.DEFAULT_OUTPUT
    copied = tmp_path / "copy"
    __import__("shutil").copytree(source, copied)
    first = copied / "images" / json.loads(
        (copied / "query_images_manifest.json").read_text()
    )["records"][0]["relative_path"]
    first.write_bytes(first.read_bytes() + b"x")
    with pytest.raises(inputs.InputBuildError):
        inputs.verify(copied)
