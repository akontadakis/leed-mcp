# tests/mcp/test_paths.py
"""Where the server may read and write (spec 5.3, 10)."""
import logging
import os

import pytest

from conftest import symlink_or_skip
from leed_mcp.paths import (
    PathOutsideRoots, inside_root, REPO_ROOT, RootsNotConfigured, contain, roots_from_env,
)


def _env(*parts):
    return {"LEED_MCP_ROOTS": os.pathsep.join(str(p) for p in parts)}


def test_good_roots_are_resolved(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    assert roots_from_env(_env(a, b), repository=tmp_path / "repo") == (a.resolve(), b.resolve())


def test_unset_or_empty_gives_no_roots(tmp_path):
    assert roots_from_env({}, repository=tmp_path / "repo") == ()
    assert roots_from_env({"LEED_MCP_ROOTS": " "}, repository=tmp_path / "repo") == ()


def test_bad_entries_are_dropped_with_a_stderr_line(tmp_path, caplog):
    good = tmp_path / "good"; good.mkdir()
    file_root = tmp_path / "a_file"; file_root.write_text("x")
    repo = tmp_path / "repo"; repo.mkdir()
    with caplog.at_level(logging.WARNING, logger="leed_mcp.paths"):
        roots = roots_from_env(_env("relative/dir", "~/somewhere", tmp_path / "missing", file_root,
                                    tmp_path, repo / "inside", good), repository=repo)
    assert roots == (good.resolve(),)
    text = caplog.text
    assert "not absolute" in text and "not an existing folder" in text
    assert "contains this repository" in text


def test_a_root_inside_the_repository_is_dropped(tmp_path):
    repo = tmp_path / "repo"; (repo / "sub").mkdir(parents=True)
    assert roots_from_env(_env(repo / "sub"), repository=repo) == ()


def test_the_repository_constant_is_this_checkout():
    assert (REPO_ROOT / "leed_mcp" / "paths.py").is_file()


def test_contain_accepts_a_path_inside_a_root(tmp_path):
    root = tmp_path / "root"; (root / "p").mkdir(parents=True)
    assert contain(str(root / "p"), (root.resolve(),)) == (root / "p").resolve()


def test_contain_refuses_without_roots(tmp_path):
    with pytest.raises(RootsNotConfigured):
        contain(str(tmp_path), ())


@pytest.mark.parametrize("bad", ["relative/p", "~/p", "/etc/passwd", "x\0y"])
def test_contain_refuses_relative_home_outside_and_nul(tmp_path, bad):
    root = tmp_path / "root"; root.mkdir()
    with pytest.raises(PathOutsideRoots):
        contain(bad, (root.resolve(),))


def test_contain_refuses_a_symlink_leading_out(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    symlink_or_skip(root / "link", outside)
    with pytest.raises(PathOutsideRoots):
        contain(str(root / "link"), (root.resolve(),))


def test_the_repository_is_found_from_a_no_editable_install(tmp_path, monkeypatch):
    """A --no-editable install puts the packages in site-packages; the
    repository is then the folder holding the running .venv."""
    import sys

    import leed_daylight.drivers._folder as folder

    repo = tmp_path / "repo"
    (repo / ".venv").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "leed-mcp"\n', encoding="utf-8")
    site = tmp_path / "site-packages" / "leed_daylight" / "drivers"
    site.mkdir(parents=True)
    monkeypatch.setattr(folder, "__file__", str(site / "_folder.py"))
    monkeypatch.setattr(sys, "prefix", str(repo / ".venv"))
    assert folder.repository_root() == repo.resolve()

    (repo / "pyproject.toml").write_text('[project]\nname = "other"\n', encoding="utf-8")
    assert folder.repository_root() == (tmp_path / "site-packages").resolve()


def test_a_root_holding_the_repository_is_dropped_when_only_the_case_differs(tmp_path):
    repo = (tmp_path / "Outer" / "repo").resolve()
    repo.mkdir(parents=True)
    swapped = repo.parent.parent / "oUTER"
    if not swapped.exists():
        pytest.skip("the file system tells names that differ in case apart")
    assert roots_from_env(_env(swapped), repository=repo) == ()


def test_a_file_system_that_reports_inode_zero_does_not_match_every_ancestor(tmp_path, monkeypatch):
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir(); other.mkdir()
    real = os.stat

    def zero_inode(path, *args, **kwargs):
        info = real(path, *args, **kwargs)
        return os.stat_result((info.st_mode, 0, info.st_dev, info.st_nlink, info.st_uid, info.st_gid,
                               info.st_size, info.st_atime, info.st_mtime, info.st_ctime))

    monkeypatch.setattr(os, "stat", zero_inode)
    assert not inside_root(other / "file.csv", root)
    assert inside_root(root / "file.csv", root)
