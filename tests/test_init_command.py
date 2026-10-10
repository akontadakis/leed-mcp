# tests/test_init_command.py
"""Tests for leed_daylight.drivers.init: the ``init`` verb (CLI spec 4.1).

Every folder is built fresh in ``tmp_path``. The template is checked
against the invented ``tests/fixtures/project_example.yaml``: filling the
template's FILL_ME values from it must give a project ``load_project``
accepts.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

import leed_daylight.drivers.init as init_module
from leed_daylight.constants import FILL_ME
from leed_daylight.project.config import ProjectError, load_project

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
TEMPLATES = pathlib.Path(init_module.__file__).resolve().parents[1] / "templates"

#: Keys of the invented example that the template leaves out on purpose:
#: optional lines the template's comments tell a person to add.
OPTIONAL_IN_EXAMPLE = {
    "glazing.frame_share.glazing_curtain_wall",
    "glazing.skylight_well",
    "glazing.skylight_well_reflectance",
}


def _leaves(value, path=""):
    """{dotted path: value} for every leaf; a list or an empty mapping is
    one leaf."""
    if isinstance(value, dict) and value:
        out = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{path}.{key}" if path else str(key)))
        return out
    return {path: value}


def _set(raw, path, value):
    *parents, last = path.split(".")
    for key in parents:
        raw = raw[key]
    raw[last] = value


def test_init_writes_the_folder(tmp_path, capsys):
    folder = tmp_path / "office_a"
    assert init_module.main([str(folder)]) == 0
    assert (folder / "project.yaml").is_file()
    assert (folder / "credit_rules.yaml").is_file()
    assert (folder / "model").is_dir()
    assert (folder / "weather").is_dir()
    assert not (folder / "materials.csv").exists()
    assert not (folder / "spaces.csv").exists()
    out = capsys.readouterr().out
    assert f"wrote {folder / 'project.yaml'}" in out
    assert f'leed-daylight scan "{folder}"' in out


def test_credit_rules_is_a_byte_copy_of_the_template(tmp_path):
    folder = tmp_path / "office_a"
    init_module.init_folder(folder, out=None)
    assert (folder / "credit_rules.yaml").read_bytes() == (
        TEMPLATES / "credit_rules_v41_bdc_nc.yaml"
    ).read_bytes()


def test_project_yaml_is_the_template(tmp_path):
    folder = tmp_path / "office_a"
    init_module.init_folder(folder, out=None)
    assert (folder / "project.yaml").read_bytes() == (TEMPLATES / "project_template.yaml").read_bytes()


def test_the_written_project_is_refused_until_filled(tmp_path):
    folder = tmp_path / "office_a"
    init_module.init_folder(folder, out=None)
    with pytest.raises(ProjectError, match="still say FILL_ME"):
        load_project(folder / "project.yaml")


def test_template_fill_me_paths_match_the_example():
    template = _leaves(yaml.safe_load((TEMPLATES / "project_template.yaml").read_text()))
    example = _leaves(yaml.safe_load((FIXTURES / "project_example.yaml").read_text()))
    fill_me = {path for path, value in template.items() if value == FILL_ME}
    assert set(template) - fill_me == {"standard.credit_rules"}
    assert template["standard.credit_rules"] == "credit_rules.yaml"
    assert fill_me == set(example) - OPTIONAL_IN_EXAMPLE - {"standard.credit_rules"}


def test_filling_the_template_from_the_example_loads(tmp_path, write_example):
    # write_example creates the scene, weather and credit rules files the
    # example names; then its project.yaml is replaced by the filled template.
    project_path = write_example(tmp_path)
    example = _leaves(yaml.safe_load((FIXTURES / "project_example.yaml").read_text()))
    raw = yaml.safe_load((TEMPLATES / "project_template.yaml").read_text())
    for path, value in _leaves(raw).items():
        if value == FILL_ME:
            _set(raw, path, example[path])
    project_path.write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    project = load_project(project_path)
    assert project.name == "Example Tower"


@pytest.mark.parametrize("name", ["project.yaml", "materials.csv", "spaces.csv", "credit_rules.yaml"])
def test_an_existing_config_file_is_refused_and_nothing_written(tmp_path, capsys, name):
    folder = tmp_path / "office_a"
    folder.mkdir()
    (folder / name).write_text("mine\n", newline="\n")
    assert init_module.main([str(folder)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert name in err
    assert sorted(p.name for p in folder.iterdir()) == [name]
    assert (folder / name).read_text() == "mine\n"


@pytest.mark.parametrize("name", ["model", "weather"])
def test_a_file_in_place_of_model_or_weather_is_refused_before_anything_is_written(tmp_path, capsys, name):
    """M3: a rerun must not be greeted with "already holds project.yaml"
    after a half-made folder from a failed mkdir -- refuse by name before
    project.yaml or credit_rules.yaml is written."""
    folder = tmp_path / "office_a"
    folder.mkdir()
    (folder / name).write_text("not a folder\n", newline="\n")
    assert init_module.main([str(folder)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert f"{folder / name}: exists and is not a plain folder" in err
    assert not (folder / "project.yaml").exists()
    assert not (folder / "credit_rules.yaml").exists()


def test_an_existing_empty_folder_is_accepted(tmp_path):
    folder = tmp_path / "office_a"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep\n", newline="\n")
    assert init_module.main([str(folder)]) == 0
    assert (folder / "notes.txt").read_text() == "keep\n"


def test_a_missing_parent_is_refused(tmp_path, capsys):
    folder = tmp_path / "no_such_parent" / "office_a"
    assert init_module.main([str(folder)]) == 1
    assert "parent folder" in capsys.readouterr().err
    assert not folder.parent.exists()


def test_a_file_in_place_of_the_folder_is_refused(tmp_path, capsys):
    target = tmp_path / "office_a"
    target.write_text("not a folder\n", newline="\n")
    assert init_module.main([str(target)]) == 1
    assert "not a folder" in capsys.readouterr().err


def test_a_folder_inside_the_repository_is_refused_before_it_is_made(tmp_path, monkeypatch, capsys):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(init_module, "REPO_ROOT", fake_repo.resolve())
    folder = fake_repo / "office_a"
    assert init_module.main([str(folder)]) == 1
    assert "inside the repository" in capsys.readouterr().err
    assert not folder.exists()


def test_a_relative_folder_is_made_in_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert init_module.main(["office_a"]) == 0
    assert (tmp_path / "office_a" / "project.yaml").is_file()


@pytest.mark.skipif(not hasattr(__import__("os"), "symlink") or __import__("sys").platform == "win32",
                    reason="needs unprivileged symlinks")
@pytest.mark.parametrize("name", ["project.yaml", "credit_rules.yaml", "model", "weather"])
def test_a_dangling_link_is_refused_and_not_written_through(tmp_path, capsys, name):
    outside = tmp_path / "outside" / "target"
    outside.parent.mkdir()
    folder = tmp_path / "office_a"
    folder.mkdir()
    (folder / name).symlink_to(outside)
    assert init_module.main([str(folder)]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert not outside.exists()
    assert sorted(p.name for p in folder.iterdir()) == [name]
