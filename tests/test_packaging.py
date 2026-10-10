# tests/test_packaging.py
"""Every non-Python file the package reads at run time must ship with it.

setuptools installs only ``*.py`` files unless ``package-data`` (or a
MANIFEST with ``include-package-data``) names the rest, so a non-editable
install would silently lose ``leed_daylight/templates/*.yaml``. This reads
pyproject.toml itself -- no install, no network -- and checks that every
git-tracked non-Python file under ``leed_daylight/`` is matched by a package-data
pattern of the ``leed_daylight`` package.
"""
from __future__ import annotations

import fnmatch
import pathlib
import subprocess
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "leed_daylight"


def _package_data_patterns() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return config["tool"]["setuptools"]["package-data"]["leed_daylight"]


def _data_files() -> list[str]:
    """Git-tracked non-Python files under the package: what ships. Files git
    does not track (macOS ``.DS_Store``, caches) never reach an install."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "leed_daylight"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode().split("\0")
    return sorted(
        pathlib.PurePosixPath(name).relative_to("leed_daylight").as_posix()
        for name in listed
        if name and pathlib.PurePosixPath(name).suffix != ".py"
    )


def test_the_templates_are_declared_as_package_data():
    patterns = _package_data_patterns()
    files = _data_files()
    assert "templates/credit_rules_v41_bdc_nc.yaml" in files
    unshipped = [f for f in files if not any(fnmatch.fnmatch(f, p) for p in patterns)]
    assert not unshipped, f"not declared in [tool.setuptools.package-data]: {unshipped}"


def test_the_atlas_web_files_are_declared_as_package_data():
    # Atlas spec 4: the page inlines the viewer from these files at run time.
    patterns = _package_data_patterns()
    for name in ("deliver/atlas/web/viewer.js", "deliver/atlas/web/viewer.css"):
        assert any(fnmatch.fnmatch(name, p) for p in patterns), name


def test_only_the_two_packages_are_installed():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["leed_daylight*", "leed_mcp*"]


def test_the_mcp_resources_are_declared_as_package_data():
    # leed-mcp spec 3 and 7: the guide resource ships with the server.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["leed_mcp"]
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "leed_mcp"], cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode().split("\0")
    files = sorted(
        pathlib.PurePosixPath(name).relative_to("leed_mcp").as_posix()
        for name in listed if name and pathlib.PurePosixPath(name).suffix != ".py"
    )
    assert files == ["resources/running_a_project.md"]
    assert all(any(fnmatch.fnmatch(f, p) for p in patterns) for f in files)


def test_it_says_it_runs_on_three_systems():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for classifier in ("Operating System :: MacOS", "Operating System :: Microsoft :: Windows",
                       "Operating System :: POSIX :: Linux"):
        assert f'"{classifier}"' in text
