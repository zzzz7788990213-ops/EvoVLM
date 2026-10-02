from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import package_submission as package
import submission_checker


def _manifest(project: Path, names: list[str]) -> dict[str, object]:
    records = []
    for name in names:
        payload = (project / name).read_bytes()
        records.append(
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return {
        "schema_version": 1,
        "status": package.STATUS,
        "formal_complete": True,
        "missing_formal_artifacts": [],
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "files": records,
    }


def test_zip_writer_is_deterministic_and_no_replace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_bytes(b"evidence\n")
    files = ["README.md"]
    manifest = _manifest(project, files)
    output = project / "dist" / package.OUTPUT_NAME
    package.write_zip_no_replace(project, output, files, manifest)
    first_hash = package.sha256_file(output)
    assert submission_checker.audit_zip(output)["issues"] == []
    with pytest.raises(FileExistsError):
        package.write_zip_no_replace(project, output, files, manifest)
    assert package.sha256_file(output) == first_hash


def test_zip_writer_rejects_output_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_bytes(b"evidence\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(package.PackageError, match="escaped"):
        package.write_zip_no_replace(
            project, outside / "bad.zip", ["README.md"], _manifest(project, ["README.md"])
        )
    assert not (outside / "bad.zip").exists()


def test_regular_inside_rejects_symlink(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "target"
    target.write_bytes(b"x")
    link = project / "link"
    link.symlink_to(target)
    with pytest.raises(package.PackageError, match="not a regular"):
        package._regular_inside(project, "link")


def test_json_loader_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(package.PackageError, match="duplicate"):
        package.load_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(package.PackageError, match="non-finite"):
        package.load_json(nonfinite)


def test_receipt_writer_is_no_replace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = project / "receipt.json"
    document = {"status": package.STATUS, "formal_complete": True}
    package.write_json_no_replace(output, document, project)
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        package.write_json_no_replace(output, document, project)
    assert output.read_bytes() == first
    assert json.loads(first)["formal_complete"] is True


def test_optional_empty_is_omitted_but_required_empty_fails(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "required.md").write_text("evidence\n", encoding="utf-8")
    (project / "optional.md").write_bytes(b"")
    assert package._canonical_nonempty_files(
        project, {"required.md", "optional.md"}, {"required.md"}
    ) == ["required.md"]
    with pytest.raises(package.PackageError, match="empty required package source"):
        package._canonical_nonempty_files(
            project, {"required.md", "optional.md"}, {"optional.md"}
        )


def test_phase2_closeout_artifacts_are_included_and_rehashed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "evaluation").mkdir(parents=True)
    phase2 = project / "results" / "phase2.json"
    phase2.parent.mkdir()
    phase2.write_text("{\"status\":\"PASS\"}\n", encoding="utf-8")
    phase3 = project / "results" / "phase3.json"
    phase3.write_text("{\"status\":\"PASS\"}\n", encoding="utf-8")
    def record(path: Path, scope: str) -> dict[str, object]:
        return {
            "path": path.relative_to(project).as_posix(),
            "scope": scope,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "phase2_complete": True,
        "artifacts": [record(phase2, "phase2"), record(phase3, "phase3")],
    }
    (project / "evaluation" / "closeout_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    selected: set[str] = set()
    package._add_phase2_closeout_artifacts(project, selected)
    assert selected == {"results/phase2.json"}
    phase2.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(package.PackageError, match="artifact mismatch"):
        package._add_phase2_closeout_artifacts(project, set())


def test_formal_runtime_events_are_selected_and_rehashed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    old_id = "old-campaign"
    new_id = "new-campaign"
    old_schedule = []
    new_schedule = []
    selected: set[str] = set()
    runtime_paths = []
    for index in range(1, 27):
        old_schedule.append({"index": index, "result": f"old-{index}.json"})
        new_schedule.append({"index": index, "result": f"new-{index}.json"})
        campaign = old_id if index <= 11 else new_id
        name = f"old-{index}.json" if index <= 11 else f"new-{index}.json"
        result_relative = f"results/phase3_recovery_v2/campaigns/{campaign}/{name}"
        runtime_relative = f"results/phase3/runtime_events/event-{index}.json"
        result_path = project / result_relative
        runtime_path = project / runtime_relative
        result_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps({"index": index}) + "\n", encoding="utf-8"
        )
        result_path.write_text(
            json.dumps({
                "runtime_event_path": runtime_relative,
                "runtime_event_sha256": hashlib.sha256(
                    runtime_path.read_bytes()
                ).hexdigest(),
            }) + "\n",
            encoding="utf-8",
        )
        selected.add(result_relative)
        runtime_paths.append(runtime_relative)
    plan_root = project / "evaluation"
    plan_root.mkdir()
    old_plan = plan_root / "old-plan.json"
    new_plan = plan_root / "new-plan.json"
    old_plan.write_text(json.dumps({"schedule": old_schedule}), encoding="utf-8")
    new_plan.write_text(json.dumps({"schedule": new_schedule}), encoding="utf-8")
    final = {
        "campaign_id": new_id,
        "campaign_plan": {"path": "evaluation/new-plan.json"},
        "composition": {
            "old_campaign_id": old_id,
            "retained_campaign_plan": {"path": "evaluation/old-plan.json"},
        },
    }
    package._add_formal_runtime_events(project, selected, final)
    assert set(runtime_paths).issubset(selected)
    assert len(set(runtime_paths)) == 26
    (project / runtime_paths[-1]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(package.PackageError, match="runtime event hash differs"):
        package._add_formal_runtime_events(project, selected - set(runtime_paths), final)


def test_v1_erratum_artifacts_are_selected_and_rehashed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    erratum_relative = "evaluation/recovery_v2/phase3_v1_plan_hash_erratum.json"
    records = []
    for relative in (
        "results/phase3/campaigns/v1/campaign_plan.json",
        "docs/PHASE3_V1_FAILURE_CLOSEOUT.md",
        "evaluation/recovery_v2/protocol.json",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
        records.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    records[1]["locator"] = {"kind": "markdown_exact_line", "value": "x"}
    records[2]["locator"] = {"kind": "json_pointer", "value": "/x"}
    erratum = project / erratum_relative
    erratum.parent.mkdir(parents=True, exist_ok=True)
    erratum.write_text(
        json.dumps({
            "actual_campaign_plan": records[0],
            "declarations": records[1:],
        }) + "\n",
        encoding="utf-8",
    )
    selected = {erratum_relative}
    package._add_v1_erratum_artifacts(project, selected)
    assert {record["path"] for record in records}.issubset(selected)
    (project / records[0]["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(package.PackageError, match="erratum artifact differs"):
        package._add_v1_erratum_artifacts(project, {erratum_relative})


def test_archive_tree_is_excluded_from_submission(tmp_path: Path) -> None:
    project = tmp_path / "project"
    archive = project / "docs/archive/phase3_v1"
    archive.mkdir(parents=True)
    (archive / "BLOCKED_STATUS.v1.json").write_text("{}\n", encoding="utf-8")
    current = project / "docs/current.md"
    current.write_text("complete\n", encoding="utf-8")
    selected = package._tree_files(project, "docs")
    assert selected == {"docs/current.md"}
    assert all("BLOCKED" not in path for path in selected)



def test_final_fresh_artifact_closure_is_mandatory_and_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    artifact = project / "results/fresh/receipt.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    from evaluation import final_fresh_evidence

    monkeypatch.setattr(final_fresh_evidence, "verify_active_document", lambda root, stage: {
        "status": "COMPLETE", "verified_artifacts": ["results/fresh/receipt.json"],
    })
    selected: set[str] = set()
    verified = package._add_final_fresh_artifacts(project, selected)
    assert verified["status"] == "COMPLETE"
    assert selected == {"results/fresh/receipt.json"}
    monkeypatch.setattr(final_fresh_evidence, "verify_active_document", lambda root, stage: {
        "status": "PENDING_FINAL_FRESH_REPRODUCTION",
    })
    with pytest.raises(package.PackageError, match="not complete"):
        package._add_final_fresh_artifacts(project, set())
