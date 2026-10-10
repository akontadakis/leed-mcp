# tests/test_deliver_package.py
"""The submission folder (reports spec 8) and the package command."""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import pathlib

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

import leed_daylight.drivers.package as package_module
import leed_daylight.drivers.report as report_module
import leed_daylight.drivers.run as run_module
from conftest import symlink_or_skip
from deliver_folder import CASE, Option1Fake, build_deliver_project
from leed_daylight.deliver import package
from merge_split_folder import build_split_project, prepared_model

NOW = datetime.datetime(2030, 1, 15, 9, 30)


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date())
    return folder


def _package(folder, **kwargs):
    kwargs.setdefault("out", io.StringIO())
    kwargs.setdefault("now", NOW)
    return package_module.package_folder(folder, **kwargs)


def _write_views_only_results(folder, prepared, case: str) -> None:
    """Like ``deliver_folder._write_views_results``, generalised to an
    arbitrary case name -- that helper hardcodes ``CASE = "base_case"``.
    Written locally rather than extending ``tests/deliver_folder.py``'s
    shared ``SPACES_CSV``/``build_deliver_project``, which several other
    test files (word, data, narrative and report tests) also build on; a
    second case there would need re-validating every one of them for the
    one views-only-case assertion needed here (fix round 1, item 5)."""
    import csv as _csv

    from leed_daylight.constants import TOOL_VERSION as _TOOL_VERSION
    from leed_daylight.project.config import load_project as _load_project
    from leed_daylight.project.inputs import input_fingerprints as _input_fingerprints
    from option2_folder import ROOM as _ROOM

    (grid,) = prepared.grids
    west = grid.points[:, 0] < 3.0
    share = 100.0 * float(west.sum()) / west.shape[0]
    case_dir = folder / "_run" / "views" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_version": _TOOL_VERSION, "radiance_version": None, "case": case,
        "settings": {
            "eye_height_m": 1.2, "eye_height_source": "Fictional example: seated eye height",
            "max_interior_glass": 1, "angular_step_deg": 2.0, "rays_per_point": 5400,
            "ground_z_m": 0.0, "moveable_furniture": "exclude", "rules": {},
        },
        "method_notes": ["Invented method note."],
        "quality_area_pct": share, "quality_point_pct": share,
        "type2_area_pct": 100.0, "type3_area_pct": share,
        "total_area_m2": 24.0, "quality_area_m2": 24.0 * share / 100.0,
        "score": {"points": None, "band": None, "reason": "below the 75 % threshold"},
        "spaces": [{
            "room": _ROOM, "storey": 0, "area_m2": 24.0, "points": int(west.shape[0]),
            "type2_pct": 100.0, "type3_pct": share, "quality_pct": share,
            "quality_area_m2": 24.0 * share / 100.0,
        }],
        "excluded_spaces": [], "warnings": [],
        "inputs": _input_fingerprints(_load_project(folder / "project.yaml"), folder),
    }
    (case_dir / "views_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with open(case_dir / "points.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["room", "storey", "x", "y", "z", "type2", "type3", "quality",
                         "median_outside_m", "reaching_rays", "limit_rays"])
        for i, (x, y, z) in enumerate(grid.points):
            writer.writerow([grid.room, grid.storey, f"{x:.3f}", f"{y:.3f}", "1.200",
                             1, int(west[i]), int(west[i]), "12.00", 100, 0])


@pytest.fixture
def views_only_folder(tmp_path, write_example, write_wea):
    """A project with a second case, ``viewsonly``: a ``views_viewsonly``
    column with no matching ``include_viewsonly`` column, so it is a
    Quality-Views-only case per ``project.spaces.views_case_names`` (F15;
    fix round 1, item 5)."""
    from option2_folder import build_option2_project, load_chain

    spaces_csv = (
        "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
        "view_preserving_shades,merge_into,include_base_case,views_base_case,views_viewsonly,reason\n"
        "L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,1,\n"
    )

    def mutate(d):
        d["views"] = {"eye_height_m": 1.2, "eye_height_source": "Fictional example: seated eye height"}

    project_folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=spaces_csv, mutate=mutate)
    _, prepared, _, _, _ = load_chain(project_folder)
    _write_views_only_results(project_folder, prepared, "viewsonly")
    return project_folder


def _assert_model_contents(model: pathlib.Path) -> None:
    """What ``model/`` must hold (build.prepare's own output): the scene,
    materials, blinds and grids, and the manifest -- never an octree, since
    ``build.prepare`` never runs Radiance (fix round 1, item 8)."""
    assert (model / "manifest.json").is_file()
    assert (model / "scene.rad").is_file()
    assert (model / "materials.rad").is_file()
    assert any(model.glob("blinds/*.rad"))
    assert any(model.glob("grids/*.pts"))
    assert not list(model.rglob("*.oct"))


def test_the_package_holds_every_part(folder):
    path = _package(folder)

    assert path == folder / "submission" / CASE
    for part in ("README.md", "provenance.json", "checksums.sha256",
                 "reports/option1/00_PROJECT_option1.docx", "reports/option1/glare_narrative.md",
                 "results/option2/option2_results.json", "results/views/points.csv",
                 "config/project.yaml", "config/spaces.csv", "config/credit_rules.yaml"):
        assert (path / part).is_file(), part
    _assert_model_contents(path / "model")
    assert not list((folder / "submission").glob(".*building*"))
    assert (folder / "submission" / "README.md").is_file()


def test_a_views_only_case_packages_a_model_folder(views_only_folder):
    report_module.report_folder(views_only_folder, case="viewsonly", options=("views",),
                                renderer="pillow", out=io.StringIO(), today=NOW.date())

    path = _package(views_only_folder, case="viewsonly")

    assert path == views_only_folder / "submission" / "viewsonly"
    _assert_model_contents(path / "model")


def test_provenance_records_versions_inputs_and_results(folder):
    provenance = json.loads((_package(folder) / "provenance.json").read_text())

    assert provenance["case"] == CASE
    assert provenance["created"] == "2030-01-15T09:30:00"
    assert provenance["options_included"] == ["option1", "option2", "views"]
    assert provenance["options_skipped"] == {}
    assert "spaces.csv" in provenance["inputs"]
    assert "numpy" in provenance["libraries"]
    assert provenance["summary"]["option2"]["points"] is not None
    assert provenance["summary"]["area_m2"] == 24.0
    assert any("logs/" in note for note in provenance["notes"])


def test_the_readme_explains_the_logs_folder(folder):
    readme = (_package(folder) / "README.md").read_text()
    assert "logs/<option>_radiance.log" in readme
    assert "may be from a different case" in readme


def test_checksums_verify(folder):
    path = _package(folder)

    lines = (path / "checksums.sha256").read_text().splitlines()
    assert lines == sorted(lines, key=lambda l: l.split("  ", 1)[1])
    assert not any(l.endswith("checksums.sha256") for l in lines)
    assert package.verify_checksums(path) == []


def test_checksums_verify_including_a_name_with_a_space(folder):
    path = _package(folder)
    odd = path / "reports" / "a name with spaces.txt"
    odd.write_text("invented")
    lines = (path / "checksums.sha256").read_text().splitlines()
    lines.append(f"{hashlib.sha256(b'invented').hexdigest()}  reports/a name with spaces.txt")
    (path / "checksums.sha256").write_text("\n".join(sorted(lines, key=lambda l: l.split('  ', 1)[1])) + "\n")

    assert package.verify_checksums(path) == []
    odd.write_text("changed")
    assert package.verify_checksums(path) == ["reports/a name with spaces.txt"]


def test_verify_checksums_guards_malformed_lines_and_escaping_paths(folder):
    path = _package(folder)
    checksums = path / "checksums.sha256"
    original = checksums.read_text()

    checksums.write_text(original + "not-a-valid-checksum-line\n")
    with pytest.raises(package.PackageError, match="malformed"):
        package.verify_checksums(path)

    digest = "0" * 64
    checksums.write_text(original.rstrip("\n") + f"\n{digest}  ../outside.txt\n")
    with pytest.raises(package.PackageError, match="outside"):
        package.verify_checksums(path)


def test_a_symlinked_report_file_is_refused(folder):
    manifest_path = folder / "_run" / "reports" / CASE / package.REPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    victim_rel = manifest["options"]["option2"]["files"][0]
    victim = folder / "_run" / "reports" / CASE / victim_rel
    victim.unlink()
    symlink_or_skip(victim, folder / "glare_narrative.md")

    with pytest.raises(package.PackageError, match="symlink"):
        _package(folder)


def test_packaging_twice_replaces_the_first(folder):
    first = _package(folder)
    marker = first / "old_marker.txt"
    marker.write_text("from the first package")

    second = _package(folder)

    assert second == first
    assert not marker.exists()
    assert not list((folder / "submission").glob(".*"))


def test_a_failure_keeps_the_old_package(folder, monkeypatch):
    first = _package(folder)
    before = (first / "provenance.json").read_text()

    def boom(*args, **kwargs):
        raise OSError("invented failure while writing the model")

    # Ruling (F11): the plan monkeypatches a bare ``prepare`` imported
    # straight from ``build.prepare``, but package.py must reuse
    # ``deliver.data.prepare_case_model`` instead of re-deriving its own
    # scene -> materials -> prepare chain (a certain duplicate-logic
    # defect otherwise, and the one that silently mishandled a
    # views-only case, F15). ``prepare_case_model`` is still a bare
    # module-global inside ``leed_daylight.deliver.package``, so patching
    # it here still simulates a failure while writing the model, and the
    # test's own assertions (the OSError propagates, the earlier package
    # is untouched, no ``.building`` folder survives) are unchanged.
    monkeypatch.setattr(package, "prepare_case_model", boom)
    with pytest.raises(OSError, match="invented failure"):
        _package(folder)

    assert (first / "provenance.json").read_text() == before
    assert not list((folder / "submission").glob(".*building*"))


def test_a_failed_swap_restores_the_old_package(folder, monkeypatch):
    """Fix round 1 (IMPORTANT): if the final rename -- ``.building`` into
    ``<case>/`` -- itself fails (disk full, permissions, killed), the
    package that was moved aside to ``.old`` a moment earlier must be
    restored, and no ``.building``/``.old`` left behind."""
    first = _package(folder)
    real_rename = pathlib.Path.rename

    def flaky_rename(self, target):
        if self.name == f".{CASE}.building":
            raise OSError("invented failure while renaming the building into place")
        return real_rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", flaky_rename)

    with pytest.raises(OSError, match="invented failure"):
        _package(folder)

    assert first.is_dir()
    assert (first / "provenance.json").is_file()
    submission = folder / "submission"
    assert not list(submission.glob(f".{CASE}.old"))
    assert not list(submission.glob(f".{CASE}.building"))


def test_a_leftover_old_with_no_final_is_restored(folder, monkeypatch):
    """Fix round 1 (IMPORTANT): ``.old`` is the only complete package left
    when a previous run was killed between the two renames (``final``
    missing). The next run must restore it before doing anything else --
    proven here by also forcing that next run's own build to fail, so the
    restored package is the only thing that could still be there
    afterwards; the old buggy leftover-cleanup loop deleted ``.old``
    unconditionally, which would leave nothing."""
    first = _package(folder)
    old = folder / "submission" / f".{CASE}.old"
    first.rename(old)
    assert not first.exists()

    def boom(*args, **kwargs):
        raise OSError("invented failure while writing the model")

    monkeypatch.setattr(package, "prepare_case_model", boom)

    with pytest.raises(OSError, match="invented failure"):
        _package(folder)

    assert first.is_dir()
    assert (first / "provenance.json").is_file()
    assert not old.exists()
    assert not list((folder / "submission").glob(".*building*"))


def test_no_reports_is_refused(tmp_path, write_example, write_wea):
    unreported = build_deliver_project(tmp_path, write_example, write_wea)

    with pytest.raises(package.PackageError) as excinfo:
        _package(unreported)
    message = str(excinfo.value)
    assert "leed_daylight.drivers.report" in message


def test_a_missing_report_file_is_refused(folder):
    manifest_path = folder / "_run" / "reports" / CASE / package.REPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    victim = manifest["options"]["option2"]["files"][0]
    (folder / "_run" / "reports" / CASE / victim).unlink()

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "option2" in message
    assert "leed_daylight.drivers.report" in message


def test_a_manifest_entry_without_files_is_refused(folder):
    manifest_path = folder / "_run" / "reports" / CASE / package.REPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    del manifest["options"]["option2"]["files"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(package.PackageError, match="option2"):
        _package(folder)


def test_a_stale_report_is_refused(folder):
    results = folder / "_run" / "option2" / CASE / "option2_results.json"
    results.write_text(results.read_text().replace('"case"', '"case" ', 1))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "option2" in message
    assert "leed_daylight.drivers.report" in message


def test_a_stale_points_csv_is_refused(folder):
    """Fix round 1, item 6: the report manifest also records points.csv's
    own sha256, so an option's points.csv changing after the report was
    made (results.json untouched) is refused too."""
    points = folder / "_run" / "option2" / CASE / "points.csv"
    points.write_text(points.read_text() + "\n")

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "option2" in message
    assert "leed_daylight.drivers.report" in message


def test_an_edited_narrative_is_refused(folder):
    """F6: the report manifest records the glare narrative's sha256 at
    report time; editing it afterwards must be refused by name, not
    shipped silently out of date."""
    narrative_path = folder / "glare_narrative.md"
    narrative_path.write_text(narrative_path.read_text() + "\nEdited after the report was made.\n")

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "glare_narrative.md" in message
    assert "option1" in message
    assert "leed_daylight.drivers.report" in message


def test_a_narrative_reported_for_another_case_is_refused(folder):
    """F6: the report manifest also records which case the narrative was
    rendered for; a manifest claiming a different case must be refused."""
    manifest_path = folder / "_run" / "reports" / CASE / package.REPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["options"]["option1"]["narrative_case"] = "another_case"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "option1" in message
    assert "leed_daylight.drivers.report" in message


def test_write_index_skips_a_case_with_malformed_provenance(folder):
    """Fix round 1, item 3: another, unrelated case's malformed
    provenance.json must not fail a package that already succeeded and
    was already swapped into place -- skip that row with a warning."""
    _package(folder)
    other = folder / "submission" / "other_case"
    other.mkdir()
    (other / "provenance.json").write_text(json.dumps({"case": "other_case"}))

    out = io.StringIO()
    package_module.package_folder(folder, out=out, now=NOW)

    index = (folder / "submission" / "README.md").read_text()
    assert f"| {CASE} |" in index
    assert "other_case" not in index
    printed = out.getvalue()
    assert "warning" in printed
    assert "other_case" in printed


def test_the_index_lists_every_case(folder):
    _package(folder)

    index = (folder / "submission" / "README.md").read_text()
    assert f"| {CASE} |" in index


def test_main_prints_the_path_and_the_check_command(folder, capsys):
    code = package_module.main([str(folder)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "submission" in printed
    assert "shasum -a 256 -c checksums.sha256" in printed


# -- Final review I5: a hand-edited report manifest never gives a traceback --

def _manifest_path(folder):
    return folder / "_run" / "reports" / CASE / package.REPORT_MANIFEST


def _edit_manifest(folder, change):
    path = _manifest_path(folder)
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("text", ["[]\n", '{"options": [], "skipped": {}}\n', '{"options": {"option2": 3}}\n',
                                  '{"options": {', ""])
def test_main_refuses_a_malformed_report_manifest_in_one_line(folder, capsys, text):
    _manifest_path(folder).write_text(text)

    code = package_module.main([str(folder)])

    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert package.REPORT_MANIFEST in err


@pytest.mark.parametrize("bad", ["../../escaped.docx", "/tmp/escaped.docx", 7, "option2/../../escaped.docx"])
def test_a_manifest_file_outside_the_reports_folder_is_refused(folder, bad):
    """Final review I5/T12: a manifest entry cannot steer a copy outside
    the package being built."""
    _edit_manifest(folder, lambda m: m["options"]["option2"]["files"].append(bad))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    assert package.REPORT_MANIFEST in str(excinfo.value)
    assert not (folder / "submission" / "escaped.docx").exists()
    assert not (folder / "escaped.docx").exists()


def test_fill_confines_every_destination_to_the_package(folder, tmp_path):
    """The destination check in _fill itself, independent of the manifest
    reader: a relative name that resolves outside ``building`` is refused."""
    with pytest.raises(package.PackageError, match="outside"):
        package._confined(tmp_path / "building" / "reports", "../../x.docx")


@pytest.mark.parametrize("content", [b"\xff\xfe\x00", b'{"summary": {"option1": [1, 2]}}',
                                     b'{"summary": {"views": {"quality_area_pct": "high"}}}'])
def test_write_index_skips_undecodable_or_odd_provenance(folder, content):
    _package(folder)
    other = folder / "submission" / "other_case"
    other.mkdir()
    (other / "provenance.json").write_bytes(content)

    out = io.StringIO()
    package.write_index(folder / "submission", out=out)

    index = (folder / "submission" / "README.md").read_text()
    assert f"| {CASE} |" in index and "other_case" not in index
    assert "other_case" in out.getvalue() and "warning" in out.getvalue()


def test_the_readme_points_to_the_narrative_in_the_project_folder(folder):
    """Final review M1: editing the packaged copy does nothing."""
    readme = (_package(folder) / "README.md").read_text()

    line = next(line for line in readme.splitlines() if "reports/option1/glare_narrative.md" in line)
    assert "<project folder>/glare_narrative.md" in line
    assert "(edit this" not in line


def test_a_two_storey_project_packages_every_storey_report(tmp_path, write_example, write_wea):
    """Final review I6: one docx per storey reaches the package."""
    from deliver_folder import build_two_storey_project

    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date())

    path = _package(folder)

    for option in ("option1", "option2", "views"):
        names = sorted(p.name for p in (path / "reports" / option).glob("*.docx") if p.name != "glare_narrative.docx")
        assert names == [f"00_PROJECT_{option}.docx", f"L0_{option}.docx", f"L1_{option}.docx"]
    assert package.verify_checksums(path) == []


def test_the_readme_says_how_to_reproduce_with_the_real_paths(folder):
    """CLI spec 7.3: the paths come from the copied project.yaml as written,
    and the commands are the one command, for the options that ran."""
    assert (folder / "glare_narrative.md").is_file()
    readme = (_package(folder) / "README.md").read_text()
    reproduce = readme.split("## Reproduce\n", 1)[1]
    assert "Put `config/credit_rules.yaml` at `credit_rules.yaml`" in reproduce
    assert "Put the scene files at `scene.rad`" in reproduce
    assert "Put the weather file at `weather/example.wea`" in reproduce
    # I1: the package's glare_narrative.md is the person's edited narrative,
    # never a fresh draft -- the reproduce steps must say to put it back
    # before running `report`, or the rebuilt submission silently differs.
    assert ("Copy `reports/option1/glare_narrative.md` to `glare_narrative.md` in the folder "
            "before you run `report`.") in reproduce
    commands = [line.strip() for line in reproduce.splitlines() if line.startswith("    leed-daylight ")]
    assert commands == [
        f'leed-daylight run "<folder>" --case "{CASE}"',
        f'leed-daylight option2 "<folder>" --case "{CASE}"',
        f'leed-daylight views "<folder>" --case "{CASE}"',
        f'leed-daylight report "<folder>" --case "{CASE}"',
        f'leed-daylight package "<folder>" --case "{CASE}"',
    ]
    assert "leed_daylight.drivers" not in reproduce
    assert "`python -m leed_daylight` takes the same verbs" in reproduce
    # The steps must stay numbered 1..N in order, with the narrative step
    # immediately before the "Run" step.
    numbers = [int(line.split(".", 1)[0]) for line in reproduce.splitlines() if line[:2].rstrip(".").isdigit()]
    assert numbers == list(range(1, len(numbers) + 1))


def test_the_reproduce_commands_leave_out_options_that_did_not_run(views_only_folder):
    report_module.report_folder(views_only_folder, case="viewsonly", options=("views",),
                                renderer="pillow", out=io.StringIO(), today=NOW.date())
    readme = (_package(views_only_folder, case="viewsonly") / "README.md").read_text()
    reproduce = readme.split("## Reproduce\n", 1)[1]
    commands = [line.strip() for line in readme.splitlines() if line.startswith("    leed-daylight ")]
    assert commands == [
        'leed-daylight views "<folder>" --case "viewsonly"',
        'leed-daylight report "<folder>" --case "viewsonly"',
        'leed-daylight package "<folder>" --case "viewsonly"',
    ]
    # I1: a views-only case never ran option1, so there is no narrative to
    # put back and no such step.
    assert "glare_narrative.md" not in reproduce


# -- Atlas spec 11: the package holds the floor atlas and checks it is current --

def _atlas_dir(folder):
    return folder / "_run" / "atlas" / CASE


def test_the_package_holds_the_atlas_but_never_the_blender_log(folder):
    (_atlas_dir(folder) / "blender.log").write_text("an invented Blender log naming /a/machine/path\n")

    path = _package(folder)

    assert sorted(p.name for p in (path / "atlas").iterdir()) == ["atlas.html", "bundle.npz"]
    assert (path / "atlas" / "atlas.html").read_bytes() == (_atlas_dir(folder) / "atlas.html").read_bytes()
    assert package.verify_checksums(path) == []
    assert "atlas/atlas.html" in (path / "checksums.sha256").read_text()
    readme = (path / "README.md").read_text()
    assert "| `atlas/atlas.html` |" in readme
    assert "model.blend" not in readme
    provenance = json.loads((path / "provenance.json").read_text())
    assert provenance["atlas_files"] == ["atlas.html", "bundle.npz"]


def test_a_blender_file_is_packaged_when_the_report_made_one(folder):
    _atlas_dir(folder).joinpath("model.blend").write_bytes(b"BLENDER-v501")
    _edit_manifest(folder, lambda m: m["atlas"]["files"].append("model.blend"))

    path = _package(folder)

    assert (path / "atlas" / "model.blend").read_bytes() == b"BLENDER-v501"
    assert "| `atlas/model.blend` |" in (path / "README.md").read_text()


def test_reports_made_before_the_atlas_are_refused_once(folder):
    _edit_manifest(folder, lambda m: m.pop("atlas"))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert f"the reports for case {CASE!r} have no floor atlas" in message
    assert "reports made before the atlas was added have none" in message
    assert message.endswith("run python -m leed_daylight.drivers.report again")


def test_an_atlas_built_from_other_inputs_is_refused(folder):
    _edit_manifest(folder, lambda m: m["atlas"]["inputs"].__setitem__("spaces.csv", "0" * 64))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert f"the floor atlas for case {CASE!r} was built from other inputs (spaces.csv changed since)" in message
    assert "leed_daylight.drivers.report" in message


def test_an_atlas_of_other_options_is_refused(folder):
    _edit_manifest(folder, lambda m: m["atlas"]["options"].pop("views"))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert (f"the floor atlas for case {CASE!r} shows option1, option2 but the package holds "
            "option1, option2, views") in message
    assert "leed_daylight.drivers.report" in message


@pytest.mark.parametrize("key, name", [("results_sha256", "results file"), ("points_sha256", "points.csv")])
def test_an_atlas_older_than_a_results_or_points_file_is_refused(folder, key, name):
    _edit_manifest(folder, lambda m: m["atlas"]["options"]["option2"].__setitem__(key, "0" * 64))

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert f"the floor atlas for case {CASE!r} is older than the option2 {name}" in message
    assert "leed_daylight.drivers.report" in message


def test_a_missing_atlas_file_is_refused(folder):
    (_atlas_dir(folder) / "atlas.html").unlink()

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "atlas file atlas.html" in message and "missing" in message
    assert "leed_daylight.drivers.report" in message


def test_a_symlinked_atlas_file_is_refused(folder):
    bundle = _atlas_dir(folder) / "bundle.npz"
    bundle.unlink()
    symlink_or_skip(bundle, folder / "spaces.csv")

    with pytest.raises(package.PackageError, match="symlink"):
        _package(folder)


def test_report_with_one_option_keeps_every_layer_in_the_atlas(folder):
    # Review Focus 5: `report --option views` after a full report.
    report_module.report_folder(folder, options=("views",), renderer="pillow", out=io.StringIO(),
                                today=NOW.date())

    manifest = json.loads(_manifest_path(folder).read_text())
    assert sorted(manifest["atlas"]["options"]) == ["option1", "option2", "views"]
    page = (_atlas_dir(folder) / "atlas.html").read_text(encoding="utf-8")
    for key in ("option1.da_pct", "option2.passes", "views.quality"):
        assert f'"key":"{key}"' in page, key
    path = _package(folder)
    assert (path / "atlas" / "atlas.html").is_file()


def test_a_bad_atlas_entry_is_still_refused_by_package(folder):
    # Final review M1: only the two writers of the entry pass over a bad
    # one (and replace it); package keeps the full check.
    _edit_manifest(folder, lambda m: m.__setitem__("atlas", {"files": ["../x"], "inputs": {}, "options": {}}))

    with pytest.raises(package.PackageError, match="the 'atlas' entry lists '../x'"):
        _package(folder)


# -- Merge-split Review Focus 5: a split project's package carries splits.csv --

def _split_folder(tmp_path, write_example, write_wea, *, link=False):
    """The split project of ``tests/merge_split_folder.py``, Option 1 run on
    the fake engine and reported; with ``link``, its splits.csv is a symlink
    to the same lines in another file, made before the run so the recorded
    fingerprints match it."""
    folder = build_split_project(tmp_path, write_example, write_wea)
    if link:
        real = folder / "split_lines.csv"
        (folder / "splits.csv").rename(real)
        symlink_or_skip(folder / "splits.csv", real)
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared_model(folder)), out=io.StringIO())
    report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date())
    return folder


def test_the_package_carries_splits_csv(tmp_path, write_example, write_wea):
    folder = _split_folder(tmp_path, write_example, write_wea)

    path = _package(folder)

    assert (path / "config" / "splits.csv").read_bytes() == (folder / "splits.csv").read_bytes()
    reproduce = (path / "README.md").read_text().split("## Reproduce\n", 1)[1]
    assert ("1. Copy `config/project.yaml`, `config/materials.csv`, `config/spaces.csv` and "
            "`config/splits.csv` into it.") in reproduce
    assert "config/splits.csv" in (path / "checksums.sha256").read_text()
    assert package.verify_checksums(path) == []
    # The packaged model is the split one.
    manifest = json.loads((path / "model" / "manifest.json").read_text())
    assert sorted(manifest["layout"]) == ["Room Strip", "Room West"]


def test_a_package_without_splits_csv_is_unchanged(folder):
    path = _package(folder)

    assert sorted(p.name for p in (path / "config").iterdir()) == [
        "credit_rules.yaml", "materials.csv", "project.yaml", "spaces.csv",
    ]
    readme = (path / "README.md").read_text()
    reproduce = readme.split("## Reproduce\n", 1)[1]
    assert "1. Copy `config/project.yaml`, `config/materials.csv` and `config/spaces.csv` into it." in reproduce
    assert "splits.csv" not in readme


def test_a_symlinked_splits_csv_is_refused(tmp_path, write_example, write_wea):
    folder = _split_folder(tmp_path, write_example, write_wea, link=True)

    with pytest.raises(package.PackageError, match="splits.csv: refusing to package a symlink"):
        _package(folder)


# -- Presentation sections and the sensor points file --------------------------

def test_an_edit_to_the_report_section_after_the_reports_is_refused(folder):
    path = folder / "project.yaml"
    path.write_text(path.read_text() + "report:\n  prepared_by: Someone Else\n")

    with pytest.raises(package.PackageError) as excinfo:
        _package(folder)
    message = str(excinfo.value)
    assert "report: or atlas: section" in message
    assert message.endswith("run python -m leed_daylight.drivers.report again")


def test_an_edit_to_the_atlas_section_after_the_reports_is_refused(folder):
    # Only the atlas entry carries a stale digest: the option entries are made current.
    path = folder / "project.yaml"
    path.write_text(path.read_text() + "atlas:\n  drawings: []\nreport:\n  prepared_by: A\n")
    report_module.report_folder(folder, options=("option1", "option2", "views"), renderer="pillow",
                                out=io.StringIO(), today=NOW.date())
    path.write_text(path.read_text().replace("prepared_by: A", "prepared_by: B"))

    with pytest.raises(package.PackageError, match="report: or atlas: section"):
        _package(folder)


def test_a_manifest_without_the_presentation_digest_is_refused(folder):
    _edit_manifest(folder, lambda m: m["options"]["option1"].pop("presentation_sha256"))

    with pytest.raises(package.PackageError, match="report: or atlas: section"):
        _package(folder)


def test_an_atlas_entry_without_the_presentation_digest_is_refused(folder):
    _edit_manifest(folder, lambda m: m["atlas"].pop("presentation_sha256"))

    with pytest.raises(package.PackageError, match="floor atlas.*report: or atlas: section"):
        _package(folder)


def test_the_package_copies_the_points_file_and_the_readme_says_where_it_goes(tmp_path, write_example, write_wea):
    def mutate(d):
        d["grid"]["points_file"] = "sensors.csv"

    points = "space,x,y,z\nRoom North,1.0,1.0,0.762\nRoom North,2.0,2.0,0.762\n"
    folder = build_deliver_project(tmp_path, write_example, write_wea, mutate_more=mutate,
                                   extra_files={"sensors.csv": points})
    report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date())

    path = _package(folder)

    assert (path / "config" / "sensors.csv").read_text() == (folder / "sensors.csv").read_text()
    readme = (path / "README.md").read_text()
    assert "Put `config/sensors.csv` at `sensors.csv` (the path `grid.points_file` names)." in readme
    assert "config/sensors.csv" in (path / package.CHECKSUMS).read_text()


_POINTS = "space,x,y,z\nRoom North,1.0,1.0,0.762\nRoom North,2.0,2.0,0.762\n"


def _points_folder(tmp_path, write_example, write_wea, name):
    def mutate(d):
        d["grid"]["points_file"] = name

    folder = build_deliver_project(tmp_path, write_example, write_wea, mutate_more=mutate,
                                   extra_files={name: _POINTS})
    report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date())
    return folder


def test_a_points_file_outside_the_project_folder_is_packaged(tmp_path, write_example, write_wea):
    folder = _points_folder(tmp_path, write_example, write_wea, "../outside_sensors.csv")

    path = _package(folder)

    assert (path / "config" / "outside_sensors.csv").read_text() == _POINTS
    assert "Put `config/outside_sensors.csv` at `../outside_sensors.csv`" in (path / "README.md").read_text()


def test_a_points_file_named_like_a_config_file_is_refused(tmp_path, write_example, write_wea):
    folder = _points_folder(tmp_path, write_example, write_wea, "../spaces.csv")

    with pytest.raises(package.PackageError, match="grid.points_file is named 'spaces.csv'"):
        _package(folder)


def test_a_points_file_named_like_a_config_file_in_another_case_is_refused(tmp_path, write_example, write_wea):
    folder = _points_folder(tmp_path, write_example, write_wea, "../Spaces.csv")

    with pytest.raises(package.PackageError, match="grid.points_file is named 'Spaces.csv'"):
        _package(folder)
