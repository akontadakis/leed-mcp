# tests/test_generic_denylist.py
"""An always-on generic confidentiality guard.

tests/test_confidentiality.py is the stricter private second layer: it
checks a client-supplied denylist of literal strings, but it skips unless
LEED_ACCEPTANCE_FIXTURE is set, so it is off in every ordinary run.

This test never skips. It does not know any client name or number; it only
knows the *shape* a leaked datasheet value takes: a float literal written
to four or more decimal places, with no stated source. Example values in
this repository are invented and round, so such a literal is either sourced
or a value that should not be here.

A matching literal is allowed only when the marker `# source:` appears on
the same line or on the line immediately above it. `#` is the comment
character in Radiance `.rad` files and in YAML alike, so this one marker
convention covers all three scanned file types unchanged. In a `.js`,
`.css` or `.html` file the marker sits inside that file's own comment
(`/* # source: ... */`).

The scan covers four sets, in this order: the package source under
`leed_daylight/` (`*.py` and `*.yaml`), the tests under `tests/` (`*.py`),
the fixture files under `tests/fixtures/` (`*.rad`, `*.yaml`, `*.yml`
and `*.csv`), and the package's web files (`*.js`, `*.css`, `*.html`, the
atlas viewer, atlas spec 10) with `PORTED.md`. leed-mcp adds the MCP layer
(`*.py`, and `*.md`: the guide resource) and the server's `README.md`. A
leaked value reaches this repository just as easily through a library
default, a checked-in template, a fixture or ported code as through a test.
"""
import pathlib
import re
import sys

LONG_FLOAT = re.compile(r"\d+\.\d{4,}")
SOURCE_MARKER = "# source:"
TESTS_DIR = pathlib.Path(__file__).parent
ROOT = TESTS_DIR.parent
PACKAGE_DIR = ROOT / "leed_daylight"
#: leed-mcp: the MCP layer is scanned too (spec 12 step 8).
MCP_DIR = ROOT / "leed_mcp"
FIXTURES_DIR = TESTS_DIR / "fixtures"


def _scan_targets():
    """Every file the always-on guard scans, in the documented order."""
    return (
        sorted(PACKAGE_DIR.rglob("*.py"))
        + sorted(PACKAGE_DIR.rglob("*.yaml"))
        + sorted(MCP_DIR.rglob("*.py"))
        + sorted(MCP_DIR.rglob("*.md"))
        + sorted(TESTS_DIR.rglob("*.py"))
        + sorted(FIXTURES_DIR.glob("*.rad"))
        + sorted(FIXTURES_DIR.glob("*.yaml"))
        + sorted(FIXTURES_DIR.glob("*.yml"))
        + sorted(FIXTURES_DIR.glob("*.csv"))
        + sorted(PACKAGE_DIR.rglob("*.js"))
        + sorted(PACKAGE_DIR.rglob("*.css"))
        + sorted(PACKAGE_DIR.rglob("*.html"))
        + [ROOT / "PORTED.md", ROOT / "README.md"]
    )


def find_unmarked_long_floats(paths):
    """Scan the given files for unsourced long-decimal float literals.

    Returns a list of (path, line_number, matched_text) findings, one per
    matching line that lacks a `# source:` marker on itself or the line
    immediately above.
    """
    findings = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            match = LONG_FLOAT.search(line)
            if not match:
                continue
            marked_here = SOURCE_MARKER in line
            marked_above = i > 0 and SOURCE_MARKER in lines[i - 1]
            if marked_here or marked_above:
                continue
            findings.append((path, i + 1, match.group()))
    return findings


def test_every_long_float_in_the_repository_names_its_source():
    findings = find_unmarked_long_floats(_scan_targets())
    assert not findings, "unmarked long-decimal float literals found:\n" + "\n".join(
        f"{path}:{line}: {value}" for path, line, value in findings
    )


def test_package_source_is_in_the_scan_set():
    # Anchored to the same _scan_targets() the guard itself runs against,
    # so this fails if a future edit drops leed_daylight/ from the set --
    # not just if this test's own expectation drifts from it.
    assert PACKAGE_DIR / "units.py" in _scan_targets()
    # units.py is top-level, so it would still pass even if rglob() above
    # were narrowed to glob(): anchor a second assertion on a subpackage
    # file so that narrowing goes red too.
    assert PACKAGE_DIR / "metrics" / "blinds.py" in _scan_targets()


def test_a_fixture_file_is_in_the_scan_set():
    # Same discrimination requirement as above, for tests/fixtures/.
    assert FIXTURES_DIR / "simple_room.rad" in _scan_targets()


def test_a_package_yaml_template_is_in_the_scan_set():
    # P5: the widened scan set adds leed_daylight/**/*.yaml.
    assert PACKAGE_DIR / "templates" / "credit_rules_v41_bdc_nc.yaml" in _scan_targets()


def test_a_fixture_csv_is_in_the_scan_set():
    # P5: the widened scan set adds tests/fixtures/*.csv.
    assert FIXTURES_DIR / "materials_example.csv" in _scan_targets()


def test_the_viewer_and_ported_md_are_in_the_scan_set():
    # Atlas spec 10: the ported viewer and the record of what was ported.
    targets = _scan_targets()
    assert PACKAGE_DIR / "deliver" / "atlas" / "web" / "viewer.js" in targets
    assert PACKAGE_DIR / "deliver" / "atlas" / "web" / "viewer.css" in targets
    assert ROOT / "PORTED.md" in targets


def test_a_package_html_file_is_in_the_scan_set(tmp_path, monkeypatch):
    # No .html file ships today; one that does is scanned too.
    page = tmp_path / "web" / "page.html"
    page.parent.mkdir()
    page.write_text("<p>1.5</p>\n", newline="\n")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE_DIR", tmp_path)
    assert page in _scan_targets()


def test_the_scanner_finds_an_unmarked_long_float(tmp_path):
    # Four decimal places, no source: the exact shape that leaked before.
    # The value itself is invented and appears nowhere else.
    sample = tmp_path / "sample.py"
    sample.write_text("x = 0.1234\n", newline="\n")  # source: invented for this test only
    findings = find_unmarked_long_floats([sample])
    assert len(findings) == 1
    path, line, value = findings[0]
    assert path == sample
    assert line == 1
    assert value == "0.1234"  # source: the literal written three lines above


def test_a_same_line_source_marker_allows_it(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("x = 0.1234  # source: made up\n", newline="\n")
    assert find_unmarked_long_floats([sample]) == []


def test_a_previous_line_source_marker_allows_it(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("# source: made up\nx = 0.1234\n", newline="\n")
    assert find_unmarked_long_floats([sample]) == []


def test_scientific_notation_is_not_matched(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("x = 1e-12\n", newline="\n")
    assert find_unmarked_long_floats([sample]) == []


def test_an_integer_is_not_matched(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("x = 3000\n", newline="\n")
    assert find_unmarked_long_floats([sample]) == []


def test_the_mcp_layer_is_scanned():
    assert MCP_DIR / "__init__.py" in _scan_targets()


def test_the_mcp_guide_and_the_readme_are_scanned():
    # leed-mcp spec 7 and 11: the packaged guide and the README hold example values too.
    targets = _scan_targets()
    assert MCP_DIR / "resources" / "running_a_project.md" in targets
    assert ROOT / "README.md" in targets
