# tests/test_output_links.py
"""No verb deletes, renames or replaces through a planted parent link
(re-review OOS-1, the security review H2's second recommendation).

A received project folder can hold a symbolic link at ``_run``,
``_run/atlas``, ``_run/reports``, ``submission`` or another folder a verb
writes under. The verb's output path is then not itself a link, exists, and
resolves outside the project; the verb must refuse, naming the link, before
anything there is deleted or replaced. Every test moves the real folder
outside the project (under ``tmp_path``), leaves a link in its place, runs
the verb with the fake engines, and checks that every outside file is still
there, unchanged. Radiance and Blender are never started.
"""
from __future__ import annotations

import datetime
import io
import pathlib
import shutil

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

import leed_daylight.drivers.option2 as option2_module
import leed_daylight.drivers.package as package_module
import leed_daylight.drivers.prepare as prepare_module
import leed_daylight.drivers.report as report_module
import leed_daylight.drivers.run as run_module
from conftest import symlink_or_skip
from deliver_folder import CASE, Option1Fake, Option2Fake, build_deliver_project
from leed_daylight.deliver.atlas import build_atlas
from leed_daylight.engine.jobs import UnsafePath
from option2_folder import load_chain

NOW = datetime.datetime(2030, 1, 15, 9, 30)


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea)


def plant(folder: pathlib.Path, tmp_path: pathlib.Path, *parts: str) -> pathlib.Path:
    """Move ``<folder>/<parts>`` outside the project folder (an empty one
    when it is missing) and leave a link to it in its place; return the
    outside folder."""
    inside = folder.joinpath(*parts)
    outside = tmp_path / "outside" / "-".join(parts)
    outside.parent.mkdir(exist_ok=True)
    if inside.exists():
        shutil.move(inside, outside)
    else:
        outside.mkdir()
    inside.parent.mkdir(parents=True, exist_ok=True)
    symlink_or_skip(inside, outside)
    assert folder.resolve() not in outside.resolve().parents
    return outside


def keep(outside: pathlib.Path, *rel: str) -> None:
    """A file at ``outside/<rel>`` that no verb may delete."""
    path = outside.joinpath(*rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("outside the project; never deleted\n")


def snapshot(outside: pathlib.Path) -> dict[str, bytes]:
    return {p.relative_to(outside).as_posix(): p.read_bytes() for p in sorted(outside.rglob("*")) if p.is_file()}


def _refused(call, folder: pathlib.Path, parts: tuple[str, ...]) -> None:
    """``call`` refuses, naming the link and saying what to do."""
    with pytest.raises(UnsafePath) as caught:
        call()
    link = folder.resolve().joinpath(*parts)
    assert str(caught.value) == (
        f"{link}: is a symbolic link; nothing is written or deleted through a link; delete it and try again"
    )


# ---------------------------------------------------------------------------
# atlas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parts", [("_run", "atlas"), ("_run",)])
def test_the_atlas_never_deletes_through_a_linked_parent(folder, tmp_path, parts):
    build_atlas(folder, CASE, blender=False, out=io.StringIO())
    outside = plant(folder, tmp_path, *parts)
    rel = ("atlas", CASE) if parts == ("_run",) else (CASE,)
    keep(outside, *rel, "keep.txt")
    before = snapshot(outside)

    _refused(lambda: build_atlas(folder, CASE, blender=False, out=io.StringIO()), folder, parts)

    assert snapshot(outside) == before


def test_the_atlas_command_prints_the_refusal_in_one_line(folder, tmp_path, capsys):
    import leed_daylight.drivers.atlas as atlas_command

    outside = plant(folder, tmp_path, "_run", "atlas")
    keep(outside, CASE, "keep.txt")
    before = snapshot(outside)

    assert atlas_command.main([str(folder), "--no-blender"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert "atlas: is a symbolic link" in err and "delete it and try again" in err
    assert snapshot(outside) == before


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report(folder):
    return report_module.report_folder(folder, renderer="pillow", out=io.StringIO(), today=NOW.date(),
                                       blender=False)


@pytest.mark.parametrize("parts", [("_run", "reports"), ("_run", "reports", CASE), ("_run",)])
def test_the_report_never_deletes_through_a_linked_parent(folder, tmp_path, parts):
    _report(folder)
    outside = plant(folder, tmp_path, *parts)
    tail = {("_run",): ("reports", CASE), ("_run", "reports"): (CASE,), ("_run", "reports", CASE): ()}[parts]
    keep(outside, *tail, "option1", "keep.txt")
    before = snapshot(outside)

    _refused(lambda: _report(folder), folder, parts)

    assert snapshot(outside) == before


# ---------------------------------------------------------------------------
# package
# ---------------------------------------------------------------------------

def _package(folder):
    return package_module.package_folder(folder, out=io.StringIO(), now=NOW)


@pytest.mark.parametrize("parts", [("submission",), ("submission", CASE)])
def test_the_package_never_deletes_or_renames_through_a_link(folder, tmp_path, parts):
    _report(folder)
    _package(folder)
    outside = plant(folder, tmp_path, *parts)
    if parts == ("submission",):
        keep(outside, f".{CASE}.building", "keep.txt")
        keep(outside, f".{CASE}.old", "keep.txt")
    keep(outside, *(() if len(parts) == 2 else (CASE,)), "keep.txt")
    before = snapshot(outside)

    _refused(lambda: _package(folder), folder, parts)

    assert snapshot(outside) == before
    assert folder.joinpath(*parts).is_symlink()


def test_the_package_command_prints_the_refusal_in_one_line(folder, tmp_path, capsys):
    _report(folder)
    outside = plant(folder, tmp_path, "submission")
    keep(outside, CASE, "keep.txt")
    before = snapshot(outside)

    assert package_module.main([str(folder)]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert "submission: is a symbolic link" in err
    assert snapshot(outside) == before


# ---------------------------------------------------------------------------
# prepare, run, option2: the prepared model's stale files and the caches
# ---------------------------------------------------------------------------

# Each returns the verb's call, ready to run: the fake engine is built (its
# own prepare included) before the link is planted.

def _prepare(folder):
    return lambda: prepare_module.prepare_folder(folder, CASE, out=io.StringIO())


def _run_option1(folder):
    _, prepared, _, _, _ = load_chain(folder)
    return lambda: run_module.run_folder(folder, CASE, engine=Option1Fake(prepared), out=io.StringIO())


def _run_option2(folder):
    _, prepared, _, _, _ = load_chain(folder)
    return lambda: option2_module.option2_folder(folder, CASE, engine=Option2Fake(prepared), out=io.StringIO())


@pytest.mark.parametrize("verb", [_prepare, _run_option1, _run_option2])
@pytest.mark.parametrize("parts", [("_run",), ("_run", "blinds"), ("_run", "grids")])
def test_the_prepared_model_is_never_cleaned_through_a_link(folder, tmp_path, verb, parts):
    call = verb(folder)
    outside = plant(folder, tmp_path, *parts)
    for sub, name in (("blinds", "stale.rad"), ("grids", "stale.pts")):
        keep(outside, *((sub,) if parts == ("_run",) else ()), name)
    before = snapshot(outside)

    _refused(call, folder, parts)

    assert snapshot(outside) == before


@pytest.mark.parametrize(("verb", "parts"), [
    (_run_option1, ("_run", "engine")),
    (_run_option2, ("_run", "option2")),
    (_run_option2, ("_run", "option2", CASE)),
])
def test_a_run_never_deletes_a_cache_through_a_link(folder, tmp_path, verb, parts):
    call = verb(folder)
    outside = plant(folder, tmp_path, *parts)
    keep(outside, "keep.fingerprint.json")
    before = snapshot(outside)

    _refused(call, folder, parts)

    assert snapshot(outside) == before


# ---------------------------------------------------------------------------
# Deeper links in the cache trees (the OOS-1 report's residual)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("verb", "parts"), [
    (_run_option1, ("_run", "engine", "sky")),
    (_run_option1, ("_run", "engine", "groups")),
    (_run_option2, ("_run", "option2", CASE, "skies")),
    (_run_option2, ("_run", "option2", CASE, "engine")),
])
def test_a_run_never_deletes_a_cache_through_a_deeper_link(folder, tmp_path, verb, parts):
    call = verb(folder)
    outside = plant(folder, tmp_path, *parts)
    keep(outside, "sky.mtx.fingerprint.json")
    keep(outside, "sky.mtx")
    before = snapshot(outside)

    _refused(call, folder, parts)

    assert snapshot(outside) == before


class _NoRadiance:
    """A Radiance that must never run."""
    version = "fake"

    def run(self, *args, **kwargs):
        pytest.fail("Radiance ran through a linked cache folder")


@pytest.mark.parametrize("below", [(), ("{scope}",), ("{scope}", "open")], ids=["unit", "scope", "state"])
def test_the_coefficients_never_delete_through_a_link_in_their_cache_tree(folder, tmp_path, below):
    from leed_daylight.engine.coefficients import _scope_key, coefficients, sensors_in_scope

    project, prepared, _, _, _ = load_chain(folder)
    group = prepared.groups[0]
    scope = sensors_in_scope(group, prepared.grids, project.performance)
    key = _scope_key([g.room for g in scope])
    parts = ("_run", "engine", "groups", group.group_id, *(p.format(scope=key) for p in below))
    outside = plant(folder, tmp_path, *parts)
    # The sidecar the trace deletes first, wherever the link sits.
    keep(outside, *(p.format(scope=key) for p in ("{scope}", "open")[len(below):]), "sky.mtx.fingerprint.json")
    before = snapshot(outside)
    suns = tmp_path / "suns.mtx"

    _refused(lambda: coefficients(_NoRadiance(), prepared, group, prepared.grids, project, False, suns=suns),
             folder, parts)

    assert snapshot(outside) == before


# ---------------------------------------------------------------------------
# Single files written through a link (re-review OOS-1's table, H1's class)
# ---------------------------------------------------------------------------

def plant_file(folder: pathlib.Path, tmp_path: pathlib.Path, *parts: str) -> pathlib.Path:
    """A link at ``<folder>/<parts>`` to a file outside the project; return
    that outside file's folder."""
    outside = tmp_path / "outside-file"
    outside.mkdir(exist_ok=True)
    keep(outside, "target.txt")
    inside = folder.joinpath(*parts)
    inside.parent.mkdir(parents=True, exist_ok=True)
    if inside.exists() or inside.is_symlink():
        inside.unlink()
    symlink_or_skip(inside, outside / "target.txt")
    return outside


def _materials(folder):
    from leed_daylight.drivers.materials import materials_folder

    return lambda: materials_folder(folder, out=io.StringIO())


def _rooms(folder):
    from leed_daylight.drivers.rooms import rooms_folder

    return lambda: rooms_folder(folder, out=io.StringIO())


@pytest.mark.parametrize(("verb", "name"), [(_materials, "materials_template.csv"), (_rooms, "spaces_template.csv")])
def test_a_template_is_never_written_through_a_link_at_it(folder, tmp_path, verb, name):
    outside = plant_file(folder, tmp_path, "_run", name)
    before = snapshot(outside)

    _refused(verb(folder), folder, ("_run", name))

    assert snapshot(outside) == before


@pytest.mark.parametrize("verb", [_materials, _rooms])
def test_a_template_is_never_written_under_a_linked_run_folder(folder, tmp_path, verb):
    outside = plant(folder, tmp_path, "_run")
    before = snapshot(outside)

    _refused(verb(folder), folder, ("_run",))

    assert snapshot(outside) == before
    assert not list(outside.glob("*_template.csv"))


def test_a_template_is_written_and_rewritten(folder):
    first = _materials(folder)()
    again = _materials(folder)()
    assert first == again and again.is_file() and not again.is_symlink()
    assert not list(again.parent.glob("*.part"))
    assert _rooms(folder)().read_text(encoding="utf-8").startswith("floor,room,")


def test_the_glare_narrative_is_never_written_through_a_link(folder, tmp_path):
    _report(folder)
    outside = plant_file(folder, tmp_path, "glare_narrative.md")
    before = snapshot(outside)

    _refused(lambda: _report(folder), folder, ("glare_narrative.md",))

    assert snapshot(outside) == before


def test_the_submission_index_is_never_written_through_a_link(folder, tmp_path):
    _report(folder)
    _package(folder)
    outside = plant_file(folder, tmp_path, "submission", "README.md")
    before = snapshot(outside)

    _refused(lambda: _package(folder), folder, ("submission", "README.md"))

    assert snapshot(outside) == before


def test_the_glare_narrative_and_the_index_are_rewritten_in_place(folder):
    _report(folder)
    _package(folder)
    _report(folder)
    _package(folder)
    for path in (folder / "glare_narrative.md", folder / "submission" / "README.md"):
        assert path.is_file() and not path.is_symlink()
        assert not list(path.parent.glob("*.part"))


# ---------------------------------------------------------------------------
# NTFS junctions count as links (re-review of OOS-1, Windows)
# ---------------------------------------------------------------------------

def test_a_junction_in_the_chain_is_refused_as_a_link(folder, tmp_path, monkeypatch):
    import os

    from leed_daylight.engine.jobs import checked_dir

    junction = folder.resolve() / "_run"
    junction.mkdir(exist_ok=True)
    real = getattr(os.path, "isjunction", None)
    monkeypatch.setattr(os.path, "isjunction",
                        lambda p: os.fspath(p) == os.fspath(junction) or bool(real and real(p)), raising=False)

    _refused(lambda: checked_dir(folder, "_run", "engine", make=False), folder, ("_run",))
    _refused(lambda: checked_dir(folder, "_run", "engine"), folder, ("_run",))


def test_a_python_without_isjunction_still_checks_links(folder, tmp_path, monkeypatch):
    import os

    from leed_daylight.engine.jobs import checked_dir

    monkeypatch.delattr(os.path, "isjunction", raising=False)
    plant(folder, tmp_path, "_run")
    _refused(lambda: checked_dir(folder, "_run", "engine", make=False), folder, ("_run",))
    assert checked_dir(folder, "submission", make=False) == folder.resolve() / "submission"
