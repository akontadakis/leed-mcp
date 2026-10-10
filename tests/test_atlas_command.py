# tests/test_atlas_command.py
"""The ``atlas`` command (atlas spec 3, 11, 12, Review Focus 1).

No test starts a real Blender. Under the test guard ``BLENDER`` names a
path that does not exist; the tests of a Blender that runs point
``BLENDER`` at a small Python program that answers ``--version`` like
Blender 5.1.1 and then fails, or saves the file it is given. Every number
is invented and round.
"""
from __future__ import annotations

import html
import json
import os
import re

import pytest

from deliver_folder import CASE, build_deliver_project
from leed_daylight.deliver.manifest import REPORT_MANIFEST
from leed_daylight.drivers.atlas import atlas_folder, main
from stub_program import make_program

#: A stand-in for Blender: the version line, then a build that exits 1.
FAILING_BLENDER = (
    "import sys\n"
    "if sys.argv[1:2] == ['--version']:\n"
    "    print('Blender 5.1.1'); sys.exit(0)\n"
    "print('an invented failure')\n"
    "sys.exit(1)\n"
)
#: A stand-in for Blender that saves the file named by its last argument.
SAVING_BLENDER = (
    "import sys\n"
    "if sys.argv[1:2] == ['--version']:\n"
    "    print('Blender 5.1.1'); sys.exit(0)\n"
    "open(sys.argv[-1], 'wb').write(b'BLENDER-v501')\n"
)


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea)


def _blender_script(tmp_path, monkeypatch, body):
    exe = make_program(tmp_path / "bin", "blender", body)
    monkeypatch.setenv("BLENDER", str(exe))
    return exe


def _payload_layers(page: str) -> list[str]:
    match = re.search(r"<script>window\.__ATLAS__=(.*?);</script>", page, re.S)
    return [layer["key"] for layer in json.loads(match.group(1).replace("<\\/", "</"))["layers"]]


def test_atlas_before_any_run_is_a_model_check(tmp_path, write_example, write_wea, capsys):
    # Review Focus 1: a project with no results yet.
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    assert main([str(folder)]) == 0
    printed = capsys.readouterr()
    target = folder / "_run" / "atlas" / CASE
    page = (target / "atlas.html").read_text(encoding="utf-8")
    assert _payload_layers(page) == ["model"]
    for option in ("option1", "option2", "views"):
        reason = re.search(rf"^skipped {option}: (.*)$", printed.out, re.M).group(1)
        assert reason.startswith(f"no {option} results for case 'base_case'")
        # The terminal names the full path; the page, which a package
        # carries, names it inside the project folder with "/" on every
        # system (final review M3; the terminal keeps Windows' backslashes).
        inside = re.sub(re.escape(f"{folder}{os.sep}") + r"(\S*)", lambda m: m.group(1).replace(os.sep, "/"), reason)
        assert html.escape(inside) in page
    assert f"case: {CASE}" in printed.out
    assert "prepared model written to" not in printed.out
    assert "Blender skipped, no model.blend: BLENDER is set to " in printed.out
    assert printed.err == ""


def test_every_written_file_and_the_summary_are_printed(folder, capsys):
    assert main([str(folder), "--no-blender"]) == 0
    out = capsys.readouterr().out.splitlines()
    target = folder / "_run" / "atlas" / CASE
    assert out[0] == f"case: {CASE}"
    assert out.index(f"wrote {target / 'bundle.npz'}") < out.index(f"wrote {target / 'atlas.html'}")
    assert not any("Blender" in line for line in out)
    assert not any(line.startswith("skipped ") for line in out)


def test_the_entry_is_merged_into_the_report_manifest(folder):
    base = folder / "_run" / "reports" / CASE
    base.mkdir(parents=True)
    kept = {"results_sha256": "1", "points_sha256": "2", "files": ["option1/a.docx"]}
    (base / REPORT_MANIFEST).write_text(json.dumps({
        "case": CASE, "options": {"option1": kept}, "skipped": {"views": "an invented reason"},
    }), newline="\n")
    result = atlas_folder(folder, blender=False)
    manifest = json.loads((base / REPORT_MANIFEST).read_text())
    assert manifest["options"] == {"option1": kept}
    assert manifest["skipped"] == {"views": "an invented reason"}
    assert manifest["atlas"] == result.entry


def test_with_no_report_yet_the_manifest_is_created(folder, capsys):
    result = atlas_folder(folder, blender=False)
    capsys.readouterr()
    manifest = json.loads((folder / "_run" / "reports" / CASE / REPORT_MANIFEST).read_text())
    assert manifest == {"case": CASE, "options": {}, "skipped": {}, "atlas": result.entry}


def test_a_failed_blender_run_exits_1_and_keeps_the_page(folder, tmp_path, monkeypatch, capsys):
    _blender_script(tmp_path, monkeypatch, FAILING_BLENDER)
    assert main([str(folder)]) == 1
    printed = capsys.readouterr()
    target = folder / "_run" / "atlas" / CASE
    log = target / "blender.log"
    assert printed.err == f"error: Blender failed, no model.blend: Blender exited 1; its output is in {log}\n"
    assert sorted(p.name for p in target.iterdir()) == ["atlas.html", "blender.log", "bundle.npz"]
    assert log.read_text() == "an invented failure\n"
    manifest = json.loads((folder / "_run" / "reports" / CASE / REPORT_MANIFEST).read_text())
    assert manifest["atlas"]["files"] == ["atlas.html", "bundle.npz"]


def test_a_blender_that_saves_gives_model_blend(folder, tmp_path, monkeypatch, capsys):
    exe = _blender_script(tmp_path, monkeypatch, SAVING_BLENDER)
    assert main([str(folder)]) == 0
    out = capsys.readouterr().out
    target = folder / "_run" / "atlas" / CASE
    assert (target / "model.blend").read_bytes() == b"BLENDER-v501"
    assert f"wrote {target / 'model.blend'}" in out
    assert f"model.blend built by Blender 5.1.1 ({exe})" in out
    manifest = json.loads((folder / "_run" / "reports" / CASE / REPORT_MANIFEST).read_text())
    assert manifest["atlas"]["files"] == ["atlas.html", "bundle.npz", "model.blend"]


def test_an_old_blender_is_one_line_and_exit_0(folder, tmp_path, monkeypatch, capsys):
    _blender_script(tmp_path, monkeypatch, FAILING_BLENDER.replace("Blender 5.1.1", "Blender 3.6.0"))
    assert main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if "Blender" in line]) == 1
    assert "Blender 3.6.0 at " in out and "older than 4.2" in out
    assert not (folder / "_run" / "atlas" / CASE / "blender.log").exists()


def test_main_refuses_an_unknown_case_in_one_line(folder, capsys):
    assert main([str(folder), "--case", "nope"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: unknown case 'nope'") and err.count("\n") == 1
    assert not (folder / "_run" / "atlas").exists()


def test_main_refuses_a_missing_folder(tmp_path, capsys):
    assert main([str(tmp_path / "nowhere")]) == 1
    assert "not a folder" in capsys.readouterr().err


def test_main_refuses_a_malformed_report_manifest(folder, capsys):
    base = folder / "_run" / "reports" / CASE
    base.mkdir(parents=True)
    (base / REPORT_MANIFEST).write_text(json.dumps({"options": [], "skipped": {}}), newline="\n")
    assert main([str(folder), "--no-blender"]) == 1
    assert "'options' is a list, not an object" in capsys.readouterr().err


# -- Final review M1 and C5: a bad atlas entry is replaced, never a loop ------

BAD_ATLAS = {"files": ["../x"], "inputs": {}, "options": {}}


def test_a_bad_atlas_entry_is_replaced(folder, capsys):
    base = folder / "_run" / "reports" / CASE
    base.mkdir(parents=True)
    kept = {"results_sha256": "1", "points_sha256": "2", "files": ["option1/a.docx"]}
    (base / REPORT_MANIFEST).write_text(json.dumps({
        "case": CASE, "options": {"option1": kept}, "skipped": {}, "atlas": BAD_ATLAS,
    }), newline="\n")

    assert main([str(folder), "--no-blender"]) == 0

    assert capsys.readouterr().err == ""
    manifest = json.loads((base / REPORT_MANIFEST).read_text())
    assert manifest["options"] == {"option1": kept}
    assert manifest["atlas"]["files"] == ["atlas.html", "bundle.npz"]


def test_a_malformed_manifest_is_refused_before_anything_is_written(folder, capsys):
    base = folder / "_run" / "reports" / CASE
    base.mkdir(parents=True)
    (base / REPORT_MANIFEST).write_text(json.dumps({"options": [], "skipped": {}, "atlas": BAD_ATLAS}), newline="\n")

    assert main([str(folder), "--no-blender"]) == 1

    printed = capsys.readouterr()
    assert "'options' is a list, not an object" in printed.err
    assert not (folder / "_run" / "atlas").exists()
    assert "wrote " not in printed.out
