# tests/test_atlas_build.py
"""``build_atlas`` and the atlas entry of the report manifest (atlas spec 3,
5, 11, 12).

Blender is never started: under the test guard ``BLENDER`` names a path
that does not exist, and the tests that need a Blender point ``BLENDER`` at
an empty executable file and pass a fake runner. Every number is invented
and round.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib

import pytest
import yaml

import leed_daylight.deliver.atlas as atlas_module
from deliver_folder import CASE, build_deliver_project
from leed_daylight.build.prepare import PreparedModel
from leed_daylight.deliver.atlas import AtlasError, AtlasResult, build_atlas
from leed_daylight.deliver.atlas.drawing import DrawingError
from leed_daylight.deliver.data import DeliverError, load_case
from leed_daylight.deliver.manifest import ATLAS_KEY, atlas_entry, read_report_manifest
from leed_daylight.project.inputs import input_fingerprints


class FakeRunner:
    """Answers ``--version`` with ``version``; a build prints a line,
    saves the ``.part`` file (the last argument) and exits ``code``."""

    def __init__(self, *, version="Blender 5.1.1\n", code=0):
        self.calls = []
        self.version, self.code = version, code

    def run(self, argv, *, log_path, timeout_s):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[1:] == ["--version"]:
            pathlib.Path(log_path).write_text(self.version, newline="\n")
            return 0
        pathlib.Path(log_path).write_text("building\n", newline="\n")
        pathlib.Path(argv[-1]).write_bytes(b"BLENDER-v501")
        return self.code


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea)


@pytest.fixture
def fake_blender(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "blender"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8", newline="\n")
    monkeypatch.setenv("BLENDER", str(exe))
    return exe


def _build(folder, **kwargs) -> tuple[AtlasResult, str]:
    out = io.StringIO()
    return build_atlas(folder, out=out, **kwargs), out.getvalue()


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the manifest entry ----------------------------------------------------------

def test_the_atlas_entry_records_files_inputs_and_each_option(folder):
    results = folder / "_run" / "option1" / CASE / "results.json"
    entry = atlas_entry(["model.blend", "atlas.html"], {"b.csv": "2", "a.csv": "1"}, {"option1": results})
    assert entry == {
        "files": ["atlas.html", "model.blend"],
        "inputs": {"a.csv": "1", "b.csv": "2"},
        "options": {"option1": {"results_sha256": _sha(results),
                                "points_sha256": _sha(results.parent / "points.csv")}},
    }
    assert ATLAS_KEY == "atlas"


def _manifest(tmp_path, atlas) -> pathlib.Path:
    path = tmp_path / "report_manifest.json"
    path.write_text(json.dumps({"options": {}, "skipped": {}, "atlas": atlas}), newline="\n")
    return path


def test_the_manifest_reader_accepts_an_atlas_entry(tmp_path):
    atlas = {"files": ["atlas.html"], "inputs": {"a.csv": "1"},
             "options": {"views": {"results_sha256": "1", "points_sha256": "2"}}}
    assert read_report_manifest(_manifest(tmp_path, atlas))["atlas"] == atlas


@pytest.mark.parametrize("atlas, fault", [
    (["atlas.html"], "is a list, not an object"),
    ({"files": "atlas.html", "inputs": {}, "options": {}}, "has 'files' as a str, not a list"),
    ({"files": ["../atlas.html"], "inputs": {}, "options": {}}, "lists '../atlas.html'"),
    ({"files": ["/atlas.html"], "inputs": {}, "options": {}}, "lists '/atlas.html'"),
    ({"files": [], "inputs": ["a.csv"], "options": {}}, "has no 'inputs' object"),
    ({"files": [], "inputs": {}, "options": []}, "has no 'options' object"),
    ({"files": [], "inputs": {}, "options": {"views": {"results_sha256": "1"}}}, "for 'views'"),
])
def test_the_manifest_reader_refuses_a_bad_atlas_entry(tmp_path, atlas, fault):
    with pytest.raises(DeliverError, match="the 'atlas' entry") as info:
        read_report_manifest(_manifest(tmp_path, atlas))
    assert fault in str(info.value)
    assert "python -m leed_daylight.drivers.atlas" in str(info.value)


def test_a_writer_of_the_atlas_entry_can_read_past_a_bad_one(tmp_path):
    # Final review M1: atlas and report replace the entry, so they read the
    # manifest without it; every other part is still checked.
    path = _manifest(tmp_path, {"files": ["../x"], "inputs": {}, "options": {}})
    assert read_report_manifest(path, check_atlas=False) == {"options": {}, "skipped": {}}
    path.write_text(json.dumps({"options": [], "atlas": {}}), newline="\n")
    with pytest.raises(DeliverError, match="'options' is a list"):
        read_report_manifest(path, check_atlas=False)


# --- build_atlas ------------------------------------------------------------------

def test_the_atlas_of_a_case_with_every_result(folder):
    result, printed = _build(folder)
    target = folder / "_run" / "atlas" / CASE
    assert result.folder == target
    assert result.files == ("bundle.npz", "atlas.html")
    assert sorted(p.name for p in target.iterdir()) == ["atlas.html", "bundle.npz"]
    assert printed.splitlines() == [f"wrote {target / 'bundle.npz'}", f"wrote {target / 'atlas.html'}"]
    assert result.layers[0] == "option1.da_pct" and result.layers[-1] == "views.quality"
    assert dict(result.skipped) == {}
    assert result.entry["files"] == ["atlas.html", "bundle.npz"]
    assert sorted(result.entry["options"]) == ["option1", "option2", "views"]
    data = load_case(folder)
    assert result.entry["inputs"] == input_fingerprints(data.project, folder)
    views = folder / "_run" / "views" / CASE / "views_results.json"
    assert result.entry["options"]["views"]["results_sha256"] == _sha(views)


def test_with_no_blender_found_the_page_is_still_written(folder):
    # Under the test guard BLENDER names a path that does not exist.
    result, _ = _build(folder)
    assert result.blender.startswith("Blender skipped, no model.blend: BLENDER is set to ")
    assert not result.blender_failed
    assert (result.folder / "atlas.html").is_file()


def test_the_folder_is_emptied_first(folder):
    target = folder / "_run" / "atlas" / CASE
    target.mkdir(parents=True)
    (target / "model.blend").write_bytes(b"old")
    (target / "stray.txt").write_text("old", newline="\n")
    result, _ = _build(folder, blender=False)
    assert sorted(p.name for p in target.iterdir()) == ["atlas.html", "bundle.npz"]
    assert result.blender == ""


def test_a_blender_build_adds_model_blend_but_never_the_log(folder, fake_blender):
    runner = FakeRunner()
    result, printed = _build(folder, blender_runner=runner)
    target = result.folder
    assert result.files == ("bundle.npz", "atlas.html", "model.blend")
    assert result.entry["files"] == ["atlas.html", "bundle.npz", "model.blend"]
    assert sorted(p.name for p in target.iterdir()) == ["atlas.html", "blender.log", "bundle.npz", "model.blend"]
    assert result.blender == f"model.blend built by Blender 5.1.1 ({fake_blender})"
    assert not result.blender_failed
    assert printed.splitlines()[-1] == f"wrote {target / 'model.blend'}"
    version, build = runner.calls
    assert version == [str(fake_blender), "--version"]
    assert build[-3:] == ["--", str(target / "bundle.npz"), str(target / "model.blend.part")]
    assert build[build.index("--python") + 1] == str(atlas_module.BLENDER_SCRIPT)


def test_a_failed_blender_run_keeps_the_page(folder, fake_blender):
    result, _ = _build(folder, blender_runner=FakeRunner(code=1))
    target = result.folder
    assert result.blender_failed
    assert result.blender.startswith("Blender failed, no model.blend: Blender exited 1")
    assert str(target / "blender.log") in result.blender
    assert result.files == ("bundle.npz", "atlas.html")
    assert sorted(p.name for p in target.iterdir()) == ["atlas.html", "blender.log", "bundle.npz"]


def test_an_old_blender_is_one_line(folder, fake_blender):
    runner = FakeRunner(version="Blender 4.1.0\n")
    result, _ = _build(folder, blender_runner=runner)
    assert result.blender.startswith("Blender skipped, no model.blend: Blender 4.1.0 at ")
    assert "older than 4.2" in result.blender
    assert not result.blender_failed
    assert len(runner.calls) == 1


def test_no_blender_asked_for_means_no_blender_looked_for(folder, fake_blender):
    runner = FakeRunner()
    result, _ = _build(folder, blender=False, blender_runner=runner)
    assert runner.calls == []
    assert result.blender == "" and not result.blender_failed


def test_case_data_holding_every_option_is_used_as_it_is(folder, monkeypatch):
    data = load_case(folder)

    def no_load(*args, **kwargs):
        raise AssertionError("load_case called again")

    monkeypatch.setattr(atlas_module, "load_case", no_load)
    result, _ = _build(folder, data=data, blender=False)
    assert len(result.layers) == 12


def test_case_data_of_one_option_is_loaded_again_with_every_option(folder, monkeypatch):
    data = load_case(folder, options=("views",))
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return load_case(*args, **kwargs)

    monkeypatch.setattr(atlas_module, "load_case", spy)
    result, _ = _build(folder, data=data, blender=False)
    assert calls == [((folder, CASE), {"require_results": False})]
    assert sorted(result.entry["options"]) == ["option1", "option2", "views"]


def test_case_data_of_another_case_is_refused(folder):
    with pytest.raises(AtlasError, match="the loaded case is 'base_case', not 'other_case'"):
        build_atlas(folder, "other_case", data=load_case(folder), out=io.StringIO())


def test_the_prepared_model_is_handed_over_once(folder):
    seen = []
    _build(folder, blender=False, on_prepared=seen.append)
    assert len(seen) == 1 and isinstance(seen[0], PreparedModel)


def test_a_case_with_no_results_is_a_model_check(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    result, _ = _build(folder, blender=False)
    assert result.layers == ()
    assert sorted(result.skipped) == ["option1", "option2", "views"]
    assert result.entry["options"] == {}
    assert (result.folder / "atlas.html").is_file()


def test_a_drawing_refusal_comes_before_anything_is_written(folder):
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["atlas"] = {"drawings": [{
        "storey": "L0", "image": "drawings/level0.png",
        "points": [{"pixel": [0, 0], "model": [0.0, 0.0]}, {"pixel": [100, 0], "model": [10.0, 0.0]}],
    }]}
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    with pytest.raises(DrawingError, match="image not found"):
        build_atlas(folder, out=io.StringIO())
    assert not (folder / "_run" / "atlas").exists()


def test_an_unknown_case_is_refused(folder):
    with pytest.raises(DeliverError, match="unknown case 'nope'"):
        build_atlas(folder, "nope", out=io.StringIO())
