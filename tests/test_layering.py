"""Enforce the seven-layer import rule (spec section 4).

A layer may import only layers strictly below it: 0 ``constants`` (and the
top-level ``leed_daylight/__init__.py``); 1 ``units``, ``geometry``,
``scene/`` (intake); 2 ``project/``; 3 ``build/``; 4 ``engine/``; 5
``metrics/`` (pure -- restricted to ``constants`` only, not "everything
below"); 6 ``deliver/`` (unused so far); 7 ``drivers/`` and ``edit/``.

The checking logic below (``find_*`` functions) takes a package *root*
directory -- the directory that is itself the package, e.g.
``leed_daylight`` -- and is exercised twice: first against small synthetic
packages planted under ``tmp_path`` (one per rule, so each rule can be SEEN
to fail before it is enforced), then against the real ``leed_daylight``
tree.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

import leed_daylight

REAL_ROOT = Path(leed_daylight.__file__).resolve().parent

# --------------------------------------------------------------------------
# The allowed-import table (Controller rulings, R-P39).
# --------------------------------------------------------------------------

#: Packages/modules that are directories under the package root and get
#: their own row in the coverage table.
KNOWN_SUBPACKAGE_DIRS = {
    "scene", "project", "build", "engine", "metrics", "deliver", "drivers", "edit",
}

#: Top-level standalone modules with their own row (plus the top-level
#: ``__init__`` pseudo-module, key ``"__init__"``).
KNOWN_TOP_MODULES = {"constants", "units", "geometry", "cli", "__main__"}

#: key -> set of keys it may import from *inside* the package.
ALLOWED: dict[str, set[str]] = {
    "__init__": {"constants"},
    "constants": set(),
    "units": {"constants"},
    "geometry": {"constants"},
    "scene": {"constants", "units", "geometry"},
    "project": {"constants", "units", "geometry", "scene"},
    "build": {"constants", "units", "geometry", "scene", "project"},
    "engine": {"constants", "units", "geometry", "scene", "project", "build"},
    "metrics": {"constants"},
    "deliver": {
        "constants", "units", "geometry", "scene", "project", "build",
        "engine", "metrics",
    },
    "drivers": {
        "constants", "units", "geometry", "scene", "project", "build",
        "engine", "metrics", "deliver",
    },
    # leed-mcp: the file helpers moved from the wizard; project_yaml reads
    # drivers.init's template path, so edit sits on the drivers layer.
    "edit": {
        "constants", "units", "geometry", "scene", "project", "build",
        "engine", "metrics", "deliver", "drivers",
    },
    # CLI spec 3.1: the dispatcher imports verb modules by name at run time.
    "cli": {"constants", "drivers"},
    "__main__": {"cli"},
}

#: key -> numeric layer, for messages only.
LAYER_NUM = {
    "__init__": 0, "constants": 0, "units": 1, "geometry": 1, "scene": 1,
    "project": 2, "build": 3, "engine": 4, "metrics": 5, "deliver": 6,
    "drivers": 7, "edit": 7, "cli": 7, "__main__": 7,
}

#: Every key that has a row in the layering table.
KNOWN_KEYS = set(ALLOWED)

METRICS_FORBIDDEN_STDLIB = {"os", "pathlib", "subprocess", "io", "shutil", "tempfile"}

STDLIB_NAMES = set(sys.stdlib_module_names)


# --------------------------------------------------------------------------
# File / import bookkeeping shared by every rule.
# --------------------------------------------------------------------------

def _iter_py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _dotted_name(root: Path, file: Path) -> str:
    prefix = root.name
    parts = list(file.relative_to(root).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join([prefix, *parts]) if parts else prefix


def _package_of(root: Path, file: Path) -> str:
    dotted = _dotted_name(root, file)
    if file.name == "__init__.py":
        return dotted
    return dotted.rsplit(".", 1)[0] if "." in dotted else root.name


def _importer_key(root: Path, file: Path) -> str | None:
    """The row this file belongs to, or None if no row covers it."""
    parts = file.relative_to(root).parts
    if len(parts) == 1:
        name = parts[0]
        if name == "__init__.py":
            return "__init__"
        stem = name[:-3]
        if stem in KNOWN_TOP_MODULES:
            return stem
        return None
    top = parts[0]
    return top if top in KNOWN_SUBPACKAGE_DIRS else None


def _key_from_dotted(dotted: str, prefix: str) -> str | None:
    """Which row an internal dotted module path belongs to, or None if external."""
    parts = dotted.split(".")
    if not parts or parts[0] != prefix:
        return None
    if len(parts) == 1:
        return "__init__"
    nxt = parts[1]
    if nxt in KNOWN_TOP_MODULES:
        return nxt
    return nxt


def _resolve_relative(package: str, level: int, module: str | None) -> str:
    parts = package.split(".") if package else []
    keep = max(len(parts) - (level - 1), 0)
    base = ".".join(parts[:keep])
    if module:
        return f"{base}.{module}" if base else module
    return base


class _ImportStmt:
    """One import statement, resolved to its internal dotted candidates."""

    __slots__ = ("lineno", "text", "resolved_module", "full_candidates", "node")

    def __init__(self, lineno, text, resolved_module, full_candidates, node):
        self.lineno = lineno
        self.text = text
        self.resolved_module = resolved_module  # the "from X" part, or None
        self.full_candidates = full_candidates  # per-alias full dotted paths
        self.node = node


def _imports_of(root: Path, file: Path) -> list[_ImportStmt]:
    prefix = root.name
    package = _package_of(root, file)
    tree = ast.parse(file.read_text(), filename=str(file))
    out: list[_ImportStmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(_ImportStmt(node.lineno, f"import {alias.name}", None, [alias.name], node))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved = node.module or ""
            else:
                resolved = _resolve_relative(package, node.level, node.module)
            candidates = [f"{resolved}.{a.name}" if resolved else a.name for a in node.names]
            names_text = ", ".join(a.name for a in node.names)
            dots = "." * node.level
            mod_text = node.module or ""
            out.append(_ImportStmt(
                node.lineno, f"from {dots}{mod_text} import {names_text}",
                resolved, candidates, node,
            ))
    return out


def _rel(root: Path, file: Path) -> str:
    return str(file.relative_to(root.parent)) if root.parent != root else str(file)


# --------------------------------------------------------------------------
# Rule 1: no module imports above its layer (also enforces metrics' pure,
# constants-only internal rule, since metrics' ALLOWED set is {"constants"}
# rather than "everything below").
# --------------------------------------------------------------------------

def find_layer_violations(root: Path) -> list[str]:
    prefix = root.name
    violations = []
    for file in _iter_py_files(root):
        importer_key = _importer_key(root, file)
        if importer_key is None:
            continue  # uncovered directory -- reported by the coverage rule
        allowed = ALLOWED[importer_key]
        for stmt in _imports_of(root, file):
            seen_keys = set()
            for candidate in stmt.full_candidates:
                target_key = _key_from_dotted(candidate, prefix)
                if target_key is None:
                    continue  # external import
                seen_keys.add(target_key)
            for target_key in seen_keys:
                if target_key == importer_key:
                    continue  # same package
                if target_key not in KNOWN_KEYS:
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: '{importer_key}' "
                        f"(layer {LAYER_NUM[importer_key]}) imports "
                        f"'{prefix}.{target_key}' via `{stmt.text}`, but "
                        f"'{target_key}' has no row in the layering table"
                    )
                    continue
                if target_key not in allowed:
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: '{importer_key}' "
                        f"(layer {LAYER_NUM[importer_key]}) imports "
                        f"'{prefix}.{target_key}' -> '{target_key}' "
                        f"(layer {LAYER_NUM[target_key]}) via `{stmt.text}`, "
                        f"which is above what layer {LAYER_NUM[importer_key]} may import"
                    )
    return violations


# --------------------------------------------------------------------------
# Rule 2: metrics only imports numpy from third parties.
# --------------------------------------------------------------------------

def find_metrics_third_party_violations(root: Path) -> list[str]:
    prefix = root.name
    violations = []
    metrics_dir = root / "metrics"
    if not metrics_dir.is_dir():
        return violations
    for file in _iter_py_files(root):
        if _importer_key(root, file) != "metrics":
            continue
        for stmt in _imports_of(root, file):
            for candidate in stmt.full_candidates:
                top = candidate.split(".")[0]
                if top == prefix or top in STDLIB_NAMES:
                    continue
                if top != "numpy":
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: metrics imports third-party "
                        f"'{top}' via `{stmt.text}`, but metrics (layer {LAYER_NUM['metrics']}) "
                        f"may only import numpy externally"
                    )
    return violations


# --------------------------------------------------------------------------
# Rule 3: metrics never touches file/process I/O.
# --------------------------------------------------------------------------

def find_metrics_stdlib_io_violations(root: Path) -> list[str]:
    violations = []
    metrics_dir = root / "metrics"
    if not metrics_dir.is_dir():
        return violations
    for file in _iter_py_files(root):
        if _importer_key(root, file) != "metrics":
            continue
        tree = ast.parse(file.read_text(), filename=str(file))
        for stmt in _imports_of(root, file):
            for candidate in stmt.full_candidates:
                top = candidate.split(".")[0]
                if top in METRICS_FORBIDDEN_STDLIB:
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: metrics imports forbidden "
                        f"standard-library module '{top}' via `{stmt.text}` "
                        f"(layer {LAYER_NUM['metrics']} must not touch file/process I/O)"
                    )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                violations.append(
                    f"{_rel(root, file)}:{node.lineno}: metrics calls builtin `open` "
                    f"(layer {LAYER_NUM['metrics']} must not touch file/process I/O)"
                )
    return violations


# --------------------------------------------------------------------------
# Rule 4: only engine/ runs a subprocess.
# --------------------------------------------------------------------------

def find_subprocess_violations(root: Path) -> list[str]:
    violations = []
    for file in _iter_py_files(root):
        importer_key = _importer_key(root, file)
        if importer_key == "engine":
            continue
        for stmt in _imports_of(root, file):
            for candidate in stmt.full_candidates:
                if candidate.split(".")[0] == "subprocess":
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: '{importer_key}' imports "
                        f"'subprocess' via `{stmt.text}`, but only engine/ (layer "
                        f"{LAYER_NUM['engine']}) may run a subprocess"
                    )
        tree = ast.parse(file.read_text(), filename=str(file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "os"):
                continue
            attr = func.attr
            if attr in ("system", "popen") or attr.startswith("exec") or attr.startswith("spawn"):
                violations.append(
                    f"{_rel(root, file)}:{node.lineno}: '{importer_key}' calls "
                    f"`os.{attr}` (a subprocess launch), but only engine/ (layer "
                    f"{LAYER_NUM['engine']}) may run a subprocess"
                )
    return violations


# --------------------------------------------------------------------------
# Rule 5: only engine/ converts RGB to lux.
# --------------------------------------------------------------------------

def find_rgb_to_lux_violations(root: Path) -> list[str]:
    violations = []
    for file in _iter_py_files(root):
        importer_key = _importer_key(root, file)
        if importer_key == "engine":
            continue
        tree = ast.parse(file.read_text(), filename=str(file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "rgb_to_lux":
                violations.append(
                    f"{_rel(root, file)}:{node.lineno}: '{importer_key}' calls "
                    f"`rgb_to_lux`, but only engine/ (layer {LAYER_NUM['engine']}) "
                    f"may convert Radiance RGB to lux"
                )
    return violations


# --------------------------------------------------------------------------
# Rule 6: a private (`_x.py`) module is imported only from inside its own
# package.
# --------------------------------------------------------------------------

def _private_modules(root: Path) -> dict[str, str]:
    """dotted path -> owning package key, for every ``_x.py`` (not ``__init__.py``)."""
    out = {}
    for file in _iter_py_files(root):
        if file.name == "__init__.py" or not file.name.startswith("_"):
            continue
        owner = _importer_key(root, file)
        if owner is None:
            continue
        out[_dotted_name(root, file)] = owner
    return out


def find_private_module_violations(root: Path) -> list[str]:
    private = _private_modules(root)
    if not private:
        return []
    violations = []
    for file in _iter_py_files(root):
        importer_key = _importer_key(root, file)
        for stmt in _imports_of(root, file):
            probes = list(stmt.full_candidates)
            if stmt.resolved_module:
                probes.append(stmt.resolved_module)
            # A single import statement can match a private module's own
            # dotted path via more than one probe (e.g. both the bare
            # resolved "from X" module and the alias-qualified candidate).
            # Collect the distinct private modules this ONE statement
            # touches, so each violating statement is reported once.
            matched: dict[str, str] = {}
            for probe in probes:
                for priv_path, owner in private.items():
                    if probe == priv_path or probe.startswith(priv_path + "."):
                        matched[priv_path] = owner
            for priv_path, owner in matched.items():
                if importer_key != owner:
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: '{importer_key}' "
                        f"imports private module '{priv_path}' via `{stmt.text}`, "
                        f"but a private module may be imported only from inside "
                        f"its own package ('{owner}')"
                    )
    return violations


# --------------------------------------------------------------------------
# Rule 7: every sub-package with a .py file, and every top-level .py module,
# has a row in the table.
# --------------------------------------------------------------------------

def find_coverage_violations(root: Path) -> list[str]:
    violations = []
    if not root.is_dir():
        return violations
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            has_py = any(True for _ in entry.rglob("*.py"))
            if not has_py:
                continue
            if entry.name not in KNOWN_SUBPACKAGE_DIRS:
                violations.append(
                    f"'{entry.name}/' holds a .py file but has no row in the layering table"
                )
        elif entry.is_file() and entry.suffix == ".py" and entry.name != "__init__.py":
            stem = entry.name[:-3]
            if stem not in KNOWN_TOP_MODULES:
                violations.append(
                    f"'{entry.name}' is a top-level module but has no row in the layering table"
                )
    return violations


# --------------------------------------------------------------------------
# Rule 8: only the wizard verb opens a browser (wizard spec 4: the one
# stated exception to "only engine/ starts a process").
# --------------------------------------------------------------------------

#: The one module that may import ``webbrowser``, relative to the root.
WEBBROWSER_MODULE = ("drivers", "wizard.py")


def find_webbrowser_violations(root: Path) -> list[str]:
    violations = []
    for file in _iter_py_files(root):
        if file.relative_to(root).parts == WEBBROWSER_MODULE:
            continue
        for stmt in _imports_of(root, file):
            for candidate in stmt.full_candidates:
                if candidate.split(".")[0] == "webbrowser":
                    violations.append(
                        f"{_rel(root, file)}:{stmt.lineno}: imports 'webbrowser' via "
                        f"`{stmt.text}`, but only {'/'.join(WEBBROWSER_MODULE)} may open "
                        "a browser"
                    )
    return violations


# ==========================================================================
# Red evidence: a checker guarded against a clean tree cannot be SEEN to
# fail. Plant one small synthetic package per rule and check each is
# reported by the matching finder.
# ==========================================================================

def _write(base: Path, rel: str, content: str) -> None:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def test_planted_upward_import_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "units.py", "from .constants import X\n")
    _write(pkg, "geometry.py", "")
    _write(pkg, "scene/__init__.py", "")
    _write(pkg, "scene/reader.py", "from .. import units\n")
    _write(pkg, "project/__init__.py", "")
    _write(pkg, "project/config.py", "from .. import scene\n")
    _write(pkg, "build/__init__.py", "")
    _write(pkg, "engine/__init__.py", "")
    # The violation: build/ (layer 3) reaches up into engine/ (layer 4).
    _write(pkg, "build/bad.py", "from ..engine import something\n")

    violations = find_layer_violations(pkg)

    assert any("build" in v and "engine" in v for v in violations), violations


def test_planted_metrics_stdlib_import_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "metrics/__init__.py", "")
    _write(pkg, "metrics/bad.py", "import pathlib\n")

    violations = find_metrics_stdlib_io_violations(pkg)

    assert any("pathlib" in v for v in violations), violations


def test_planted_metrics_third_party_import_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "metrics/__init__.py", "")
    _write(pkg, "metrics/bad.py", "import yaml\n")

    violations = find_metrics_third_party_violations(pkg)

    assert any("yaml" in v for v in violations), violations


def test_planted_subprocess_outside_engine_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "engine/__init__.py", "")
    _write(pkg, "build/__init__.py", "")
    _write(pkg, "build/bad.py", "import subprocess\n")

    violations = find_subprocess_violations(pkg)

    assert any("subprocess" in v and "build" in v for v in violations), violations


def test_planted_rgb_to_lux_call_outside_engine_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "engine/__init__.py", "")
    _write(pkg, "build/__init__.py", "")
    _write(pkg, "build/bad.py", textwrap.dedent("""\
        from .. import units

        def f(rgb):
            return units.rgb_to_lux(rgb)
    """))

    violations = find_rgb_to_lux_violations(pkg)

    assert any("build" in v for v in violations), violations


def test_planted_private_module_imported_across_packages_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "build/__init__.py", "")
    _write(pkg, "build/_names.py", "def slug(x):\n    return x\n")
    _write(pkg, "engine/__init__.py", "")
    # A single import statement whose resolved "from X" module is ITSELF the
    # private module's own dotted path (`pkg.build._names`), and whose
    # alias-qualified candidate (`pkg.build._names.slug`) also matches via
    # prefix -- both used to be reported as two separate violations for the
    # one bad statement.
    _write(pkg, "engine/bad.py", "from ..build._names import slug\n")

    violations = find_private_module_violations(pkg)

    matching = [v for v in violations if "_names" in v and "engine" in v]
    assert len(matching) == 1, violations


def test_planted_untabled_sibling_import_is_reported(tmp_path):
    """A covered module importing an untabled sibling must be reported, not crash."""
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "project/__init__.py", "")
    # "newthing" is a real top-level module, but it has no row in ALLOWED /
    # LAYER_NUM. Indexing LAYER_NUM[target_key] for it must not raise KeyError.
    _write(pkg, "newthing.py", "")
    _write(pkg, "project/config.py", "from .. import newthing\n")

    violations = find_layer_violations(pkg)  # must not raise KeyError

    assert any("project" in v and "newthing" in v for v in violations), violations


def test_planted_metrics_open_call_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "metrics/__init__.py", "")
    _write(pkg, "metrics/bad.py", textwrap.dedent("""\
        def f():
            return open("somefile")
    """))

    violations = find_metrics_stdlib_io_violations(pkg)

    assert any("open" in v for v in violations), violations


def test_planted_new_top_level_module_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "extra_module.py", "")

    violations = find_coverage_violations(pkg)

    assert any("extra_module" in v for v in violations), violations


def test_planted_uncovered_directory_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "extra/__init__.py", "")
    _write(pkg, "extra/thing.py", "")

    violations = find_coverage_violations(pkg)

    assert any("extra" in v for v in violations), violations


def test_planted_webbrowser_import_outside_the_wizard_verb_is_reported(tmp_path):
    pkg = tmp_path / "pkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "constants.py", "")
    _write(pkg, "drivers/__init__.py", "")
    _write(pkg, "drivers/wizard.py", "import webbrowser\n")
    _write(pkg, "drivers/wizard_app/__init__.py", "")
    _write(pkg, "drivers/wizard_app/server.py", "from webbrowser import open as show\n")

    violations = find_webbrowser_violations(pkg)

    assert len(violations) == 1, violations
    assert "wizard_app/server.py" in violations[0] and "webbrowser" in violations[0], violations


# ==========================================================================
# The real tree.
# ==========================================================================

def test_no_module_imports_above_its_layer():
    violations = find_layer_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_metrics_imports_nothing_but_constants_and_numpy():
    violations = find_layer_violations(REAL_ROOT)
    violations += find_metrics_third_party_violations(REAL_ROOT)
    metrics_related = [
        v for v in violations if "metrics" in v.split(":", 1)[0] or "'metrics'" in v
    ]
    assert metrics_related == [], "\n".join(metrics_related)


def test_metrics_imports_no_standard_library_io():
    violations = find_metrics_stdlib_io_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_only_the_engine_runs_a_subprocess():
    violations = find_subprocess_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_only_the_engine_converts_rgb_to_lux():
    violations = find_rgb_to_lux_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_no_private_module_imported_across_packages():
    violations = find_private_module_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_the_table_covers_every_package():
    violations = find_coverage_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_only_the_wizard_verb_opens_a_browser():
    violations = find_webbrowser_violations(REAL_ROOT)
    assert violations == [], "\n".join(violations)


def test_leed_daylight_imports_no_mcp():
    """leed-mcp spec 3: the pipeline never imports the MCP layer or the SDK.

    Its own AST walk: ``_imports_of`` keeps only imports inside the package,
    so it never sees ``mcp``."""
    forbidden = {"mcp", "mcp_types", "leed_mcp"}
    offenders = []
    for file in _iter_py_files(REAL_ROOT):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            offenders += [f"{_rel(REAL_ROOT, file)}:{node.lineno}: {name}"
                          for name in names if name.split(".")[0] in forbidden]
    assert not offenders, offenders
