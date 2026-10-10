# LEED Daylight Pipeline (as copied into leed-mcp)

This is the pipeline's own README, kept for the command-line verbs. The server's README is ../README.md.

Takes a clean Radiance scene and computes two LEED v4.1 BD+C Indoor Environmental Quality
credits:

- **EQ Credit Daylight, Option 1**: annual simulation to IES LM-83 (sDA, ASE and points).
- **EQ Credit Daylight, Option 2**: illuminance at 9:00 and 15:00 on the clearest day near each
  equinox, from the weather file (pass share and points).
- **EQ Credit Quality Views**: ray tracing from every occupied grid point (Type 2, Type 3,
  Quality View and the point).

For each case it also writes Word reports with plans and charts, a glare narrative draft, and a
checksummed submission folder. A detected room can be split by a line, and several rooms merged into
one space, from plain files (`splits.csv`, and `merge_into` in `spaces.csv`), for Option 1, Option 2
and Quality Views.

## Install

Install [uv](https://docs.astral.sh/uv/), then in the repository's folder run:

    uv sync --extra report --no-editable --reinstall-package leed-mcp

This puts the `leed-daylight` command in the repository's `.venv`, with the report libraries (python-docx,
matplotlib, Pillow and cairosvg; cairosvg also needs the system cairo library, and is needed only when
`report.renderer` is `cairosvg`). Run it again after every update of the repository: the commands run the
installed copy, not the files in the folder, and `--no-editable` keeps that working in an iCloud folder (the
README explains why). Then `.venv/bin/leed-daylight` (Windows: `.venv\Scripts\leed-daylight.exe`) works from
any folder (or `leed-daylight` with the `.venv` activated). `.venv/bin/python -m leed_daylight <verb>` takes
the same verbs.

For the tests, install the `dev` extra instead (it includes `report` and adds pytest):

    uv sync --extra dev --no-editable --reinstall-package leed-mcp

On macOS, a Homebrew `cairo` that is not on the default library search path makes the `cairosvg`
renderer refuse by name; `matplotlib` (the default) and `pillow` need no system library.

## Test

    .venv/bin/python tools/check.py

(Windows: `.venv\Scripts\python tools\check.py`.)

It runs the whole suite (`python -m pytest`; extra arguments go to pytest) and ends with one
block: `CHECK PASSED` or `CHECK FAILED`, the counts, and whether the acceptance tests and the
confidentiality scan ran or were skipped. CI runs it on macOS, Windows and Linux where the public copy is published; run it before every merge too.

The summary prints `ACCEPTANCE GATE NOT RUN: set LEED_ACCEPTANCE_FIXTURE` whenever an
acceptance test was skipped, so a green run without the fixture never reads like a passed
acceptance gate. With the fixture set, it says instead that the run has no pass mark (a run
without targets), or asks you to read the skip reasons. A separate confidentiality test runs
in the same suite: it scans every git-tracked file and every commit message for terms from a
private deny list, and skips (with its own message) when the fixture or the deny list is missing.

Acceptance tests need a private fixture and are skipped without it. Its `expected.json` holds one entry
per run; `LEED_ACCEPTANCE_RUN` names the run of the session (`tests/acceptance/conftest.py`):

    LEED_ACCEPTANCE_FIXTURE=/path/to/fixture LEED_ACCEPTANCE_RUN=A python -m pytest -m acceptance tests/acceptance/test_fixture_option1.py

For a complete step-by-step guide (folder layout, every input file, running, reading the results), see
[`RUNNING_A_PROJECT.md`](RUNNING_A_PROJECT.md).

## One command

Every step is a verb of `leed-daylight`, in the order you use them:

| Command | What it does |
|---|---|
| `leed-daylight init "<project folder>"` | Make a new project folder with a `project.yaml` to fill in |
| `leed-daylight scan "<project folder>"` | Check the Radiance scene and list its materials, storeys and rooms |
| `leed-daylight materials "<project folder>"` | Write a `materials.csv` template from the materials in the scene |
| `leed-daylight rooms "<project folder>"` | Write a `spaces.csv` template from the rooms in the scene |
| `leed-daylight prepare "<project folder>" [--case]` | Build the prepared model (frames, blinds, grid) without Radiance |
| `leed-daylight run "<project folder>" [--case] [--processes N]` | Run Option 1 (sDA and ASE) |
| `leed-daylight option2 "<project folder>" [--case] [--processes N]` | Run Option 2 |
| `leed-daylight views "<project folder>" [--case] [--processes N]` | Run Quality Views |
| `leed-daylight report "<project folder>" [--case] [--option] [--renderer] [--no-blender]` | Write the Word reports, plots and glare narrative, and the floor atlas |
| `leed-daylight atlas "<project folder>" [--case] [--no-blender]` | Write the floor atlas page and, with Blender 4.2 or newer, a Blender file |
| `leed-daylight calculator "<project folder>" [--case] --template <file>` | Fill the USGBC calculator |
| `leed-daylight package "<project folder>" [--case]` | Write the checksummed submission folder |

`leed-daylight` alone lists the verbs; `leed-daylight <verb> --help` shows a verb's arguments;
`leed-daylight --version` shows the version. Ctrl-C stops any verb; run the same command again and
a simulation resumes. The older form `.venv/bin/python -m leed_daylight.drivers.<verb>` still
works for the verbs from `rooms` to `package`, with the same arguments and results (Windows: `.venv\Scripts\python`).

`init` never overwrites anything and writes no value for you: every setting says `FILL_ME`, and
the tool refuses to run while any value still does. `materials` writes
`<project folder>/_run/materials_template.csv` with an empty role in every row and the tool's
guess in the note; it never touches `materials.csv`. `scan` writes nothing. `prepare` needs no
Radiance and prints the room, grid point and window-group counts, a quick check before a long run.

`prepare`, `atlas`, `run`, `option2`, `views`, `report` and `package` hold the lock file
`.leed-daylight.lock` in the project folder while they work, so two of them never write one folder at
once: a second one is refused (`error: run is already working on this folder (started 14:02, process
51234)`). **This is a change:** `report` during a `run` is refused. A stale lock, left by a command that
was killed, is cleared by the next command with a one-line note.

## List the rooms

Before `spaces.csv` exists, see what the tool detects as rooms and get a starter file to
edit. The folder needs `project.yaml` and `materials.csv` only (`spaces.csv` is not
required):

    leed-daylight rooms "<project folder>"

This prints one line per detected room (a placeholder label `R1`, `R2`, ..., its storey,
area, centroid and level) plus any warning `detect_rooms` raised, and writes
`<project folder>/_run/spaces_template.csv` -- one row per room, in the exact header
`spaces.csv` needs, ready to load. It never touches an existing `spaces.csv`; copy the
template over it and rename the rooms.

To split a detected room in two, draw the line in an optional `splits.csv` (header
`level_z,points,note`; `level_z` and the points in metres, whatever `scene.units`; points as `x y`
pairs separated by `;`) and run `rooms` again: it prints `split lines applied: N (splits.csv)` first
and writes one row per half. A line must cross exactly one room, end on its outline or outside it,
leave each half at least 1 m2 and end between panes of glass; each refusal names the line. To treat
several rooms on one storey as one space, keep a row per room and write the name of the row that
stands for the space in the others' `merge_into`: one grid, one result row and one blind schedule
for the space, window groups per piece (`_p1`, `_p2`). `docs/RUNNING_A_PROJECT.md` (step 5) gives
every rule.

## Run Option 1

Put a Radiance model in a folder alongside three files this tool reads: `project.yaml`
(scene, site, analysis settings), `materials.csv` (modifier -> role) and `spaces.csv`
(per-space edits, one `include_<case>` column per credit scenario). Then,
from the repository root:

    leed-daylight run "<project folder>" [--case <name>] [--processes N]

`--case` defaults to the first case in `spaces.csv`. `--processes N` (an integer, default 1)
runs `rfluxmtx`/`rcontrib` with `-n N`: faster on a multi-core machine, same method, results equal within Monte Carlo noise; it
is not part of the cache fingerprint, so changing it does not force a recompute. Everything
this run writes goes to
`<project folder>/_run/`: the prepared model (`scene.rad`, `materials.rad`, `blinds/`,
`grids/`, `manifest.json`), `radiance.log` (every Radiance command line run, with exit
status and wall time -- a command that never started is logged too, as `-> failed to start`),
an `engine/` folder of daylight-coefficient matrices, and
`option1/<case>/results.json` (the Option 1 sDA/ASE result, score, warnings and the fingerprints of
the input files, also printed to screen) and `option1/<case>/points.csv` (one row per sensor point: daylit share
of occupied hours, sDA pass, ASE hours, ASE exceeded). A stopped run resumes: files already written and hashed in the manifest are
not redone. Every scene file's own folder and the prepared model folder go on `RAYPATH`, so a
relative `.cal`, `.xml` or data-file reference in the scene resolves.

Example, for a fictional project folder `office_a/`:

    leed-daylight run office_a --case base_case

## Run Quality Views

LEED v4.1 BD+C EQ Credit Quality Views, by ray tracing from every grid point at eye height.
The project folder needs a `views:` section in `project.yaml`, at least one `views_<case>`
column in `spaces.csv`, and a `views:` section in `credit_rules.yaml` (the shipped template
has one). Then:

    leed-daylight views "<project folder>" [--case <name>] [--processes N]

It writes `<project folder>/_run/views/<case>/`: `views_results.json` (settings, per-space and
project results, score, warnings), `points.csv` (one row per point) and one SVG plan per storey
for each of `type2`, `type3` and `quality`. No annual simulation is needed. It needs only
Radiance's `oconv` and `rtrace`. `--processes N` (default 1) runs `rtrace` with `-n N`: more
processes trace the rays faster on a multi-core machine, for identical results.

Its working files go to `<project folder>/_run/views/`: `prepared/` (the framed model),
`model/` (the black, blind-free views scene), `engine/` (the octree and one cached result per
room) and `radiance.log`. A stopped run resumes, and a changed setting recomputes the rooms it
affects. Every scene file's own folder and the prepared model folder go on `RAYPATH`.

Example, for the fictional project folder `office_a/`:

    leed-daylight views office_a --case base

## Run Option 2

LEED v4.1 BD+C EQ Credit Daylight Option 2. The project folder is the same as for Option 1.
`project.yaml`'s `option2:` section must say `sky: "weather_file"` and
`time_basis: "weather_file_local"`. Then:

    leed-daylight option2 "<project folder>" [--case <name>] [--processes N]

It picks the day with the most direct sun within `clearest_day_window_days` of each
`equinox_dates` entry from the weather file. The Perez clearness index of every candidate is
recorded in the results file, and a warning is printed if it would pick another day. The
weather file's `time_zone` must equal `site.meridian_radiance`. It builds a `gendaylit` sky from the weather
file's own irradiance at each of `times` (9:00 and 15:00 as the weather file stamps them: its
own time zone, no daylight-saving shift), traces illuminance at every grid point with blinds
excluded, averages the two days, and passes a point when both averages are between `lux_min`
and `lux_max` (no upper limit in a space with `view_preserving_shades: yes`).

It writes `<project folder>/_run/option2/<case>/option2_results.json` and `points.csv`, its
working files to `<project folder>/_run/option2/<case>/` (`skies/`, `engine/`), and its log to
`<project folder>/_run/option2/radiance.log`. It needs `oconv`, `rtrace`, `gendaylit` and
`xform`. A stopped run resumes.

## Make the reports

After any of `run`, `option2` and `views`:

    leed-daylight report "<project folder>" [--case <name>] [--option option1|option2|views] [--renderer matplotlib|cairosvg|pillow]

It writes `<project folder>/_run/reports/<case>/<option>/`: `00_PROJECT_<option>.docx`, one
`<storey>_<option>.docx` per storey, and `plots/` with every figure. An option whose results are
missing, or older than the project files they came from, is skipped and named. For Option 1 it
also writes `<project folder>/glare_narrative.md`: edit it freely; a rerun replaces only the
blocks between the `<!-- auto:... -->` markers and keeps your text, including the "Glare
measures" column. A space you filled in that is no longer over the trigger (another case, or a
design change) moves to a "Spaces no longer over the trigger" table in the same block, with a
warning, and returns when it is over the trigger again. Its Word copy is
`_run/reports/<case>/option1/glare_narrative.docx`.

The optional `report:` section of `project.yaml` sets the figure renderer (`matplotlib`, the
default, `cairosvg` or `pillow`), your list of `deviations` (printed in every report) and
`prepared_by`. Editing the `report:` section never makes results out of date: `project.yaml`
is fingerprinted by its content without that section (an edit to its comments or layout alone
does not count either).

It prints every file it writes (each report `.docx` and each plot `.png`), each skipped option by
name and reason, every warning, and the count of `[ADD]`/`[CONFIRM]` marks still open in
`glare_narrative.md`.

After the Word files, `report` also builds the floor atlas of the same case (next section), always with
every current option, so `report --option views` still gives an atlas of every result. Add
`--no-blender` to leave Blender out. A Blender failure is a warning; any other atlas failure is an
`error:` line and exit 1, and `package` then refuses until the atlas is built.

## Make the USGBC calculator

    leed-daylight calculator "<project folder>" [--case <name>] --template <blank calculator .xlsm>

It fills the USGBC "LEED v4.1 Daylight and Quality Views Calculator" (V02) from the case's current
results and writes `<project folder>/_run/calculator/<case>/LEED_v4.1_Daylight_QualityViews_<case>.xlsm`
with `calculator.json` beside it (units, checksums of the template and the results, row counts, the
expected Summary values and the warnings). The blank calculator is USGBC's and is not included:
download it from USGBC and pass its path with `--template`. Any other file is refused. The command
needs `calculator.units` (`IP` or `SI`, as the project is registered with LEED) in `project.yaml` and
a `vlt` column in `materials.csv`. Excel recalculates the file when it opens it; the command prints
the Summary values Excel should show.

## See the model and the results

    leed-daylight atlas "<project folder>" [--case <name>] [--no-blender]

It writes `<project folder>/_run/atlas/<case>/`, emptied first:

- `atlas.html`: one page to open in any browser, offline (no font, script or image from the network).
  One tab per storey with the plan, a layer switch (`Model` or one result layer) and the room table,
  and one 3D tab with a switch per kind of surface and per storey. The header gives the project, the
  case and, with results, each option's project numbers. An option with no current result is listed
  at the top with the reason.
- `bundle.npz`: the data the page and the Blender file are both built from.
- `model.blend` and `blender.log`: only when Blender 4.2 or newer is found.

Before any run it is the model check: the prepared model, the rooms and the grid; `atlas` prints the
same case summary `prepare` prints. After a run it shows every current result as a layer.

Blender is looked for in this order: the `BLENDER` environment variable (when set, it must name the
Blender program, and nothing else is tried), `blender` on `PATH`, then the standard install folder: `/Applications/Blender.app/Contents/MacOS/Blender` on macOS, `%ProgramFiles%\Blender Foundation\Blender *\blender.exe` on Windows (the newest version wins), `/snap/bin/blender` and then `/opt/blender*/blender` on Linux (the newest wins). No Blender, or one older than 4.2, gives one line
and no `model.blend`; the page is still written. A Blender run that fails keeps the page, prints the
path of `blender.log` and exits 1 (`report` prints a warning instead and exits 0).

A drawing can lie under a storey plan. Add the optional `atlas:` section to `project.yaml`:

    atlas:
      drawings:
        - storey: "L1"
          image: "drawings/level1.png"
          points:
            - {pixel: [400, 1800], model: [0.0, 0.0]}
            - {pixel: [3900, 1800], model: [35.0, 0.0]}
          opacity: 0.35

`image` is a PNG or JPG file inside the project folder (a PDF drawing: save it as PNG first). The two
points pin the drawing to the model: a pixel of the image (x to the right, y down) and the same point
in model metres (y up); scale, rotation and position follow, with no stretch. `opacity` is optional
(default 0.35). Each wrong entry is refused by name: a missing image, a file that is not PNG or JPG
by its content, a storey not in `spaces.csv`, two points closer than 10 pixels or 0.5 m, two drawings
for one storey, an opacity outside 0 to 1. Editing `atlas:` never makes results out of date. The
drawing shows in the plans only, not in 3D or Blender.

## Make the submission folder

    leed-daylight package "<project folder>" [--case <name>]

It writes `<project folder>/submission/<case>/`: `reports/`, `results/`, `model/` (the model
prepared again for this case), `config/` (with `config/splits.csv` when the project has one),
`logs/`, `atlas/` (the files the report step's atlas
wrote: `atlas.html`, `bundle.npz` and, when Blender built it, `model.blend`; never `blender.log`),
`README.md`, `provenance.json` and `checksums.sha256`, and rewrites `submission/README.md`, an index
of every case. It refuses reports that are missing or older than their results. Check a package
with:

    cd "<project folder>/submission/<case>" && shasum -a 256 -c checksums.sha256

It also refuses a stale report: an option's results file or `points.csv` changed after the report
was made, or (Option 1) `glare_narrative.md` was edited after the report, or was last reported for
a different case; rerun `report` first. `logs/` holds the most recent run log for each option;
`_run/` keeps only the last run, so a log here may be from a case other than this package.

It also refuses an atlas that is missing or out of date: none built by `report`, one built from other
input files, one that shows other options than the package holds, or one older than an option's
results file or `points.csv`. Rerun `report` first.

**Changed on 27 September 2026 (package):** reports made before the floor atlas was added have no
atlas, so `package` refuses them once; run `report` again, then `package`.

## Status

Built and tested:

- Phases 1 and 2: the pure metrics layer and scene intake.
- Phase 3 (model build) and phase 4 (Radiance engine, Option 1, the `run` command).
- The `rooms` command.
- Quality Views (the `views` command). It has run end to end with Radiance 6.1a on an
  invented building.
- Option 2 (the `option2` command).
- Skylights, sloped glass, roof monitors and diffusing glazing (role `glazing_diffusing`), each with its
  own dirt factor, frame share, skylight well, blind rule, window groups and report lines. Built and
  covered by the test suite; not yet run end to end with Radiance.
- Reports and the submission folder (the report and package commands).
- The one command, `leed-daylight`, with the `init`, `scan`, `materials` and `prepare` verbs,
  and the local check `tools/check.py`.
- The floor atlas and the Blender file (the `atlas` command; `report` builds them too, and `package`
  copies them). No automatic test opens the page in a browser or starts a real Blender; an opt-in
  test (`LEED_BLENDER_SMOKE=1`) builds and reopens a Blender file.
- Splitting and merging rooms (`splits.csv`, `merge_into`) for Option 1, Option 2 and Quality Views:
  one grid and one result row per space, window groups per piece, blinds per space, and plans that draw
  every piece, label a space once and draw split lines dashed. Covered by the test suite with fake
  engines; not yet run end to end with Radiance: an opt-in test (`LEED_RADIANCE_SMOKE=1`) does that.
- The folder lock and the job log.

Not built yet: a public acceptance check. The tests exist in `tests/acceptance/` but need a private fixture (`LEED_ACCEPTANCE_FIXTURE`) and are skipped without it, so accuracy against a delivered LEED job (daylight) or a ClimateStudio run (views) is not confirmed by the public test suite.

**Changed on 23 September 2026:** the sensor-grid rule changed -- the lattice is now centred
in each room's clearance band, so a room's grid has a different point count and layout than
before. Both Option 1 and Quality Views use this grid, so any earlier result from either is
not comparable to one from this version; rerun. Option 1 also gained a residual pass: one
extra trace per room, with every window group blacked out, so light entering through a
skylight or a gap outside every window group is found and named in a warning. Each window
group's own trace already carries that light, so a room lit by several groups has the extra
copies taken off and the light counts exactly once.

**Breaking change (roof and sloped glass):** a project whose exterior glass includes any pane tilted under
85 degrees from flat is now refused until `glazing.dirt_factor` has one value per tilt class present
(`vertical`, `sloped`, `skylight`) and `glazing.frame_share` has `sloped:` (and `skylight:`); with skylights it
also needs `glazing.skylight_well`. Earlier versions read such glass as a window with one dirt factor and a
window frame. Projects with only vertical glass are unaffected: their prepared model and results are
unchanged. Option 1 results written before this version have no window-group list; `report` and `package`
skip them and ask you to run again. See `docs/RUNNING_A_PROJECT.md`, steps 2 and 4.
