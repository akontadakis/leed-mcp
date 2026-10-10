"""Text the pipeline writes is the same bytes on every system (spec 2026-10-09 A1)."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = ("leed_daylight", "leed_mcp")


def _calls(method):
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == method):
                    yield path.relative_to(ROOT), node.lineno, node


def _keywords(node):
    return {k.arg for k in node.keywords}


def _text_mode(node):
    """The mode argument of an open/fdopen call, when it is a literal."""
    mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
    return mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None


def test_every_write_text_names_utf8_and_lf():
    bad = [f"{p}:{n}" for p, n, node in _calls("write_text") if not {"encoding", "newline"} <= _keywords(node)]
    assert bad == []


def test_every_read_text_names_its_encoding():
    bad = [f"{p}:{n}" for p, n, node in _calls("read_text") if "encoding" not in _keywords(node)]
    assert bad == []


def test_every_text_mode_fdopen_writes_lf():
    bad = []
    for path, line, node in _calls("fdopen"):
        mode = _text_mode(node)
        if mode and "b" not in mode and mode[0] in "wa" and "newline" not in _keywords(node):
            bad.append(f"{path}:{line}")
    assert bad == []


def test_git_keeps_lf_and_leaves_the_golden_fixtures_alone():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text
    for folder in ("golden_prepare", "golden_outputs", "golden_figures"):
        assert f"tests/fixtures/{folder}/** -text" in text
