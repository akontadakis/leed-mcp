# tests/test_regex_keywords.py
"""No module-level regex call passes ``maxsplit``, ``count`` or ``flags`` by
position (CLI spec 7.2).

Python 3.13 deprecates those positional arguments of the module functions
``re.split``, ``re.sub`` and ``re.subn`` ("DeprecationWarning: 'maxsplit'
is passed as positional argument"). A compiled pattern's own ``split``,
``sub`` and ``subn`` are not deprecated: checked with Python 3.13 and 3.14
and ``-W error`` on 25 September 2026. Python 3.12, which runs this suite,
warns about neither, so this guard reads the package source instead.

Allowed positional arguments: ``re.split(pattern, string)`` and
``re.sub(pattern, repl, string)``, ``re.subn(pattern, repl, string)``.
"""
from __future__ import annotations

import ast
import pathlib

import leed_daylight

PACKAGE = pathlib.Path(leed_daylight.__file__).resolve().parent

#: The most positional arguments each module function may take before the
#: deprecated ones begin.
LIMITS = {"split": 2, "sub": 3, "subn": 3}


def positional_regex_calls(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"):
            continue
        limit = LIMITS.get(node.func.attr)
        if limit is not None and len(node.args) > limit:
            found.append(f"line {node.lineno}: re.{node.func.attr} with {len(node.args)} positional arguments")
    return found


def test_the_guard_finds_the_deprecated_form_only():
    source = (
        "import re\n"
        "_PIPE = re.compile(r'[|]')\n"
        "a = re.split('[|]', text, 5)\n"
        "b = re.sub('x', 'y', text, 2)\n"
        "c = re.subn('x', 'y', text, 2, re.I)\n"
        "d = re.split('[|]', text, maxsplit=5)\n"
        "e = re.sub('x', 'y', text, count=2)\n"
        "f = _PIPE.split(text, 5)\n"
        "g = line.split(' ', 1)\n"
    )
    assert positional_regex_calls(source) == [
        "line 3: re.split with 3 positional arguments",
        "line 4: re.sub with 4 positional arguments",
        "line 5: re.subn with 5 positional arguments",
    ]


def test_no_package_module_passes_regex_options_by_position():
    problems = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for problem in positional_regex_calls(path.read_text(encoding="utf-8")):
            problems.append(f"{path.relative_to(PACKAGE.parent)} {problem}")
    assert problems == []
