# Changelog

## Unreleased

- Claude Code plugin: `/plugin marketplace add akontadakis/leed-mcp` then `/plugin install leed-mcp@leed-mcp`
  registers the server (through uvx) and the `running-leed-studies` skill. The README also shows a uvx setup
  without a clone for the Claude Code CLI and Claude Desktop.
- `deliver(action="calculator")` and `leed-daylight calculator`: fill USGBC's LEED v4.1 Daylight and Quality Views
  Calculator V02 from a case's results. The blank calculator is not included; the server reads the copy named by
  `LEED_MCP_CALCULATOR`. The submission folder includes the filled calculator when it is current.
- `vlt` in materials.csv: the visible light transmittance (%) of a glazing material, for the calculator.
- `calculator.units` in project.yaml: `IP` or `SI`, as the project is registered with LEED.
- Windows and Linux are supported besides macOS. CI runs the tests on all three. Radiance and Blender are found in
  each system's usual folders; Radiance runs with one process on Windows; text files are written as UTF-8 with
  LF line endings.
- New refusal code `FILE_IN_USE`: Windows, a file held open by Excel or Word. A deliver job that fails this way says so.
- Fix: the pipeline runs on Python 3.11 (unhashable dataclass defaults).
- `radiance.direct` accepts only `-d*` and `-s*` options; the ambient options go in `radiance.ambient`. The sun pass
  keeps its ambient bounces at zero.
- The stored matrices are rebuilt after a Radiance upgrade, or after an edit to a `.cal`, `.dat` or picture file a
  material uses. Radiance headers marked `BigEndian=1` are read correctly. The matrix fingerprints have new keys,
  so the first Option 1 run after upgrading rebuilds every matrix, which can take hours on an annual run.
- Option 1 gives no points unless `analysis.hours` is `08:00-18:00` and `analysis.aggregation` is `floor_area`.
  Option 2 gives no points unless `option2.times` is exactly 09:00 and 15:00 and the lux band matches the rules file.
  The results are still computed; the output says "not awarded" and why, with a warning. The calculator still
  fills such an option's values and names it in a warning, because Excel scores them.
- A calculator filled by an earlier version is left out of the submission folder as out of date; fill it again.
- A credit rules file must give `option2.lux_min` and `option2.lux_max`.
- Band edges are compared after rounding to 9 decimal places (was 6). Quality Views shares are built from exact
  point counts, as Option 1's are.
- `package` refuses when the `report:` or `atlas:` section of `project.yaml`, or a drawing the atlas uses, changed
  after `report` ran. It copies `grid.points_file` into `config/`, which may lie outside the project folder, and
  refuses a name that clashes with another file of `config/`.
- The calculator takes the Daylight areas from the scene, as every other output does. A `vlt` from 39.9 up to and
  including 40 is written as 39.9 (30 is written as 30.0). A `vlt` above 0 and at most 1 draws a warning (VLT is a
  percentage).
- Quality Views writes `results.json` and `points.csv` together or not at all. `spaces.csv` ignores a row of empty cells.
- Server: `project(action="create")` and `init` refuse a dangling link where a project file belongs; the
  repository check compares by device and inode, so another spelling of the case no longer gets past it.
- Server: scene reads are capped (`!xform` depth 32, 10,000 file reads, 2,000,000 primitives). A modifier name
  or a split-line note that would be read as a formula (or holds a control character or `<!--`) is refused.
  `project(action="files")` stops at 20,000 files and says so (`RESULT_TRUNCATED`). Jobs pass `--case=NAME`.
- Docs: the guides name the refusal codes `FOLDER_IN_REPOSITORY`, `IO_ERROR`, `SCENE_PROBLEM`, `RESULT_TRUNCATED` and
  `PIPELINE_WARNING`; the install steps use `uv sync --no-editable`; the README lists what is not covered; the
  ASE text says 1,000 lux or more; the HTML guide covers Windows and Linux and uses system fonts.

## 0.1.0 (2026-10-09)

First public release.

- Seven tools: `project`, `scene`, `inputs`, `run`, `results`, `deliver`, `job`.
- Three resources: the guide, the `project.yaml` template and the credit rules.
- Three prompts: `new_project`, `check_before_run`, `explain_results`.
- LEED v4.1 BD+C EQ Credit Daylight Option 1 and Option 2, and EQ Credit Quality Views.
- Word reports, glare narrative draft, floor atlas and checksummed submission folder.
- Long runs as background jobs, and as tasks for clients with the Tasks extension.
