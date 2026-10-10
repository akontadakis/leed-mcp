# leed-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**leed-mcp** is a local [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server. It lets an AI
assistant such as Claude Desktop or Claude Code run LEED v4.1 BD+C daylight and views calculations on your own
[Radiance](https://www.radiance-online.org) model, on your own computer.

You talk to the assistant in plain words. The assistant calls this server's tools to set up the project, check
the model, fill in the input files with values **you** give, run the simulations, explain the results and write
the reports and the submission folder.

> LEED® and USGBC® are registered trademarks of the U.S. Green Building Council. This project is not
> affiliated with, sponsored by or endorsed by USGBC or GBCI. Its results support, and do not replace, the
> judgement of a qualified daylight consultant and the LEED review.

## Install in two lines

In Claude Code:

    /plugin marketplace add akontadakis/leed-mcp
    /plugin install leed-mcp@leed-mcp

Set `LEED_MCP_ROOTS` to your projects folder and put Radiance's `bin` folder on `PATH` in your shell profile
(see [Configure](#configure)), then restart Claude Code. The tools are live, together with the
`running-leed-studies` skill. You need [uv](https://docs.astral.sh/uv/) on your `PATH`. Other clients and
the install from a clone are under [Install](#install) and [Connect](#connect).

## What it does

It computes three LEED v4.1 BD+C Indoor Environmental Quality credits from one Radiance scene:

| Credit | Method | What you get |
|---|---|---|
| **EQ Credit Daylight, Option 1** | Annual simulation to IES LM-83: daylight coefficients per window group, a direct-sun matrix and operable blinds | sDA300/50%, ASE1000,250, per space and per project, and the points |
| **EQ Credit Daylight, Option 2** | Illuminance at 9:00 and 15:00 on the clearest day near each equinox, from the weather file | Share of floor area between 300 and 3,000 lux, and the points |
| **EQ Credit Quality Views** | Ray tracing from every occupied grid point at eye height | Type 2 and Type 3 views, the Quality View share and the point |

For each case it also writes:

- Word reports with plans, charts and tables;
- a glare narrative draft for spaces with ASE above 10 %;
- a floor atlas: one self-contained HTML page with plans and a 3D view, and optionally a Blender file;
- a checksummed submission folder.

A "case" is one design scenario, such as a base design and a design with larger windows. Cases are columns in
`spaces.csv`, so you can compare them side by side.

## About the LEED credits

LEED v4.1 BD+C rewards designs that connect occupants with the outdoors, support circadian rhythms and cut
electric lighting through daylight and clear views out. It does this with two separate credits: **EQ Credit
Daylight** (up to 3 points) and **EQ Credit Quality Views** (1 point). The table shows what each requirement is
and what this server does with it. The thresholds are in the resource `leed://templates/credit-rules`.

### EQ Credit Daylight

| Requirement | What leed-mcp does |
|---|---|
| **Option 1, sDA<sub>300/50%</sub>** (spatial daylight autonomy): the share of regularly occupied floor area that gets at least 300 lux for at least 50 % of occupied hours. An average of 40 %, 55 % or 75 % earns 1, 2 or 3 points. | Computes it per point, per space and for the project, with blinds operating, and scores the points. |
| **Option 1, extra point**: with 2 points, every regularly occupied space reaches sDA 55 % or more. | Checks it and adds the point. |
| **Option 1, ASE<sub>1000,250</sub>** (annual sunlight exposure): the share of floor area that gets 1,000 lux or more of direct sun for more than 250 hours. v4.1 has no project limit. A space above 10 % needs a description of its glare control. | Computes it with blinds open, flags every space above 10 % and writes a glare narrative draft for you to complete. |
| **Option 2, illuminance**: the share of floor area between 300 and 3,000 lux at 9:00 and 15:00 on a clear-sky day near each equinox. 55 %, 75 % or 90 % earns 1, 2 or 3 points. | Picks the clearest day from the weather file, computes the share with blinds left out and scores the points. |

Project teams usually show these results with simulation tools such as ClimateStudio or other Radiance-based
programs. leed-mcp drives Radiance directly. The submission itself goes through USGBC's own forms and
calculator; this server produces the results, plans and reports that support them, and fills in USGBC's V02
calculator from a blank copy you supply (`deliver(action="calculator")`).

### EQ Credit Quality Views

| Requirement | What leed-mcp does |
|---|---|
| **Area**: at least 75 % of the regularly occupied floor area has a quality view. | Traces views from every grid point at eye height and reports the share. |
| **Clear line of sight** to vision glazing, with no permanent obstruction, within three times the head height of the glazing. | Checks it per point (Type 3). |
| **View content**: nature, urban landmarks or art, or objects at least 7.5 m (25 ft) beyond the exterior glazing. | Checks it per point (Type 2). Objects are tagged `nature`, `landmark` or `art` in `materials.csv`. Distance uses the median outside distance, the ClimateStudio method. |
| **Glazing**: vision glazing with a visible light transmittance (VLT) above 40 %, not distorted by frits, patterns or tints. | Uses the glass you tag `clear` or `blocks` in `materials.csv`. It does not detect frits or patterns itself. |
| **Atria**: views into interior atria may count for at most 30 % of the required area. | **Not checked.** Every run prints a warning, so you can check it by hand. |

## How it works

```text
 You ──chat──▶ AI client ──MCP over stdio──▶ leed-mcp server ──▶ leed_daylight pipeline ──▶ Radiance
              (Claude Desktop,               (this repository:      (the calculation:          (rfluxmtx,
               Claude Code, …)                tools, prompts,        model prep, metrics,       rcontrib,
                                              resources, jobs)       reports, package)          rtrace, …)
                                                      │
                                                      ▼
                                       your project folder, inside LEED_MCP_ROOTS
```

The repository holds two Python packages:

- **`leed_mcp`** is the MCP server. It checks every argument, keeps every path inside the folders you allow,
  reads and writes the input files safely and starts the long calculations as background jobs.
- **`leed_daylight`** is the calculation pipeline. It also works alone, as the `leed-daylight` command.
  [`docs/PIPELINE.md`](docs/PIPELINE.md) lists its commands.

A project goes through these steps:

1. **Project folder.** The server makes a folder with a `project.yaml` template. You copy the Radiance scene
   into `model/` and the weather file into `weather/`.
2. **Scene check.** The server reads the scene and lists every problem, every material and every room it finds.
   A room is one connected patch of floor.
3. **Inputs.** You give the role of each material (`materials.csv`), the name and type of each room
   (`spaces.csv`) and the project settings (`project.yaml`). The assistant asks you for each value and writes
   only what you confirm.
4. **Prepare.** The pipeline adds window frames, reveals, blinds and the sensor grid. This takes seconds and
   needs no Radiance. It is a quick check before a long run.
5. **Runs.** Option 1, Option 2 and Quality Views run as background jobs. A large Option 1 run can take hours.
   A stopped run resumes where it stopped.
6. **Results and delivery.** The assistant reads the results, flags inputs that changed after a run and writes
   the reports, the atlas and the submission folder.

Some rules hold everywhere:

- **No invented values.** Every `FILL_ME` setting and every glazing value comes from you. The server offers
  standard LEED values only as suggestions. Each change to `project.yaml` is logged with the sentence you give
  for where it came from, in `_run/input_log.jsonl`.
- **Local only.** The server runs on this computer over stdio and makes no network call.
- **Confined.** It reads and writes only inside the folders in `LEED_MCP_ROOTS`. Links that lead out, scene
  files outside the project folder and unsafe Radiance options are refused.
- **Safe writes.** A file changed by hand since it was read is not overwritten. Every write keeps a backup in
  `_run/backup/`.

## Tools

Seven tools, each with an `action` argument:

| Tool | Actions | What it does |
|---|---|---|
| `project` | `create`, `status`, `list`, `files` | Make a project folder; say what it still needs |
| `scene` | `scan`, `materials_template`, `rooms` | Check the Radiance scene; list its materials and rooms |
| `inputs` | `read`, `create`, `update`, `import_weather` | Read and write `project.yaml`, `materials.csv`, `spaces.csv`, `splits.csv`, `glare_narrative.md`; read `credit_rules.yaml` |
| `run` | `prepare`, `option1`, `option2`, `views` | Start a calculation as a job |
| `results` | `summary`, `spaces`, `points`, `compare` | Read the results of a run |
| `deliver` | `report`, `atlas`, `package`, `calculator` | Start the reports, the floor atlas, the submission folder or the USGBC calculator as a job |
| `job` | `status`, `wait`, `logs`, `cancel`, `list` | Follow a job; one job runs at a time on this computer |

More on each action:

- `project`: `create` writes `project.yaml` (every value `FILL_ME`), `credit_rules.yaml` and empty `model/` and
  `weather/` folders. `status` lists what is missing: input files, `FILL_ME` keys, cases, results and whether
  they are current, missing Radiance programs, report libraries and Blender. `list` finds project folders.
  `files` lists the scene, weather and drawing files in a folder.
- `scene`: `scan` lists scene problems, materials and storeys. `materials_template` lists every material with a
  guessed role. `rooms` lists every detected room with its outline, area and windows.
- `inputs`: `read` returns a file with its `sha256`; `create` and `update` write it only when that `sha256`
  still matches. `update` on `project.yaml` takes dotted keys and a `source` sentence. `import_weather`
  converts an `.epw` file to `.wea` with Radiance's `epw2wea`.
- `run`: `prepare` builds the prepared model; `option1`, `option2` and `views` run the credits. Each returns a
  `job_id`.
- `results`: `summary` gives the score and project metrics; `spaces` one row per space; `points` the sensor
  points; `compare` two cases side by side.
- `deliver`: `report` writes the Word reports and the glare narrative draft; `atlas` the floor atlas page;
  `package` the checksummed submission folder; `calculator` the USGBC Daylight and Quality Views Calculator,
  filled from the case.
- `job`: `wait` returns when the job ends or after up to 50 seconds; `logs` shows the end of the job's log;
  `cancel` stops the job.

A refusal is a normal tool result with `refused: true`, a reason and a code such as `FILL_ME_LEFT` or
`PATH_OUTSIDE_ROOTS`. The guide resource lists every code and what to do.

A client that declares the Tasks extension (`io.modelcontextprotocol/tasks`) on a request gets a `run` or
`deliver` job back as a task. Any other client gets a `job_id` and follows it with `job(action="wait")`.

## Resources

| URI | What it is |
|---|---|
| `leed://docs/running-a-project` | The guide: every step of a project as tool calls, and every refusal code |
| `leed://templates/project` | The `project.yaml` template, every value `FILL_ME` |
| `leed://templates/credit-rules` | The LEED v4.1 BD+C credit thresholds the tool scores against |

## Prompts

Prompts are ready-made instructions that you pick in your client, for example from the `/` menu in Claude Code.

| Prompt | Arguments | What it asks the model to do |
|---|---|---|
| `new_project` | `folder` | Set up a new project step by step, asking you for every value |
| `check_before_run` | `folder` | List what is still missing before a long run |
| `explain_results` | `folder`, `case` | Explain a case's results in plain words |

## Example

A shortened session in Claude Desktop. The folder, rooms and numbers are invented.

Every call carries the project's `folder`; below, `F` stands for `"/Users/you/LEED projects/office_a"`.

```text
You:    Use the new_project prompt for "/Users/you/LEED projects/office_a".

Claude: project(action="create", folder=F)
        I made the folder with project.yaml, credit_rules.yaml, model/ and weather/.
        Please copy the scene .rad files into model/ and the weather file into weather/.

You:    Done. Units are metres; materials.rad first, then building.rad.

Claude: inputs(action="update", file="project", folder=F,
               values={"scene.units": "m", "scene.files": ["model/materials.rad", "model/building.rad"]},
               source="typed by the user in chat")
        scene(action="scan", folder=F)
        The scene has no problems. It uses 9 materials. "glass_south" looks like glazing.
        What role does each material have?

You:    glass_south and glass_north are glazing, wall_ext is wall, …

Claude: inputs(action="create", file="materials", folder=F, rows=[…])
        scene(action="rooms", folder=F)
        I found 6 rooms on 2 storeys. R1 is 42 m² on level 0.0 with 3 windows facing south.
        What is each room's name and space type? Is it regularly occupied?

        … the remaining FILL_ME values, each asked and confirmed …

Claude: run(action="prepare", folder=F)  →  job(action="wait")
        Prepared: 6 rooms, 1,240 grid points, 8 window groups, no warnings.
        run(action="option1", folder=F, case="base", processes=8)  →  job(action="wait") …
        results(action="summary", folder=F, case="base", credit="option1")
        sDA300/50% is 62 % of the regularly occupied floor area, so the case earns 2 points.
        ASE1000,250 is above 10 % in "Office 3", so that space needs a glare narrative.

You:    Write the reports.

Claude: deliver(action="report", folder=F, case="base")
        The glare narrative has 4 [ADD] marks. What glare control does Office 3 have?
```

## Requirements

- macOS, Windows 10 or newer, or Linux. CI runs the tests on all three without Radiance or Blender; full simulations are tested on macOS. On Windows, a stop waits 10 seconds and then ends the job by force; the graceful stop that keeps finished work (exit 130) is macOS and Linux only.
- Python 3.11 or newer, installed through [uv](https://docs.astral.sh/uv/).
- [Radiance](https://www.radiance-online.org), for every run except `prepare`.
- An MCP client: Claude Desktop, Claude Code or another client that supports stdio servers.
- Optional: [Blender](https://www.blender.org) 4.2 or newer, only for the atlas's 3D Blender file.

## Install

1. Install [uv](https://docs.astral.sh/uv/), then Python: `uv python install 3.12` (3.11 or newer works).
2. Get the code:

       git clone https://github.com/akontadakis/leed-mcp.git
       cd leed-mcp

3. In the repository's folder, install the server and the report libraries:

       uv sync --extra report --no-editable --reinstall-package leed-mcp

   Run this command again after every update of the repository. The server runs the installed copy, not the
   files in the folder. `--reinstall-package leed-mcp` makes uv copy the new code even when the version
   number did not change.

### macOS

The `--no-editable` note below applies to iCloud folders on macOS.

**Why `--no-editable`.** An editable install finds the code through a `.pth` file in
`.venv/lib/python3.12/site-packages/`. When the repository sits in a folder that iCloud syncs (Desktop,
Documents), macOS keeps setting the `hidden` flag on the files in `.venv`, and Python skips a hidden `.pth`
file. The `leed-mcp` command then fails outside the repository with `ModuleNotFoundError: No module named
'leed_mcp'`. Check the flag:

    ls -lO .venv/lib/python3.12/site-packages/*.pth

`hidden` in the fifth column means the flag is set. `chflags nohidden` clears it, but iCloud sets it again.
With `--no-editable` the code is copied into `.venv`, there is no `.pth` file for it, and the flag does no
harm. Another way: keep the repository in a folder that iCloud does not sync, and the editable install
works.

### Windows

Run the same commands in PowerShell. The server program is `.venv\Scripts\leed-mcp.exe`.
Install Radiance from https://github.com/LBNL-ETA/Radiance/releases (default folder `C:\Radiance`)
and Blender from https://www.blender.org (found automatically under `C:\Program Files\Blender Foundation`).
Radiance runs with one process on Windows. Close the calculator and the reports in Excel and Word
before you ask for them again: Windows does not let the server replace a file that is open.

### Linux

Install Radiance from https://github.com/LBNL-ETA/Radiance/releases or your package manager, and
Blender from https://www.blender.org, the snap, or an archive unpacked under `/opt`. The server
program is `.venv/bin/leed-mcp`.

### All systems

Check the install from outside the repository:

    cd / && "<repository>/.venv/bin/python" -I -c "import leed_mcp; print(leed_mcp.__file__)"

It must print a path inside `<repository>/.venv/lib/` (on Windows, inside `.venv\Lib\`).

On Windows, run `.venv\Scripts\python.exe -I -c "import leed_mcp; print(leed_mcp.__file__)"` from another folder.

**Radiance.** The runs need Radiance on the server's `PATH`. Which programs each run needs comes from the
pipeline's own checks: `project(action="status")` lists the ones missing per run action, and a run is
refused with `RADIANCE_NOT_FOUND` naming them. `run(action="prepare")` needs none.

**Reports.** The default figure renderer, `matplotlib`, needs no system library. The `cairosvg` renderer
needs the system cairo library.

**Blender 4.2 or newer** is optional: only for the 3D file of the floor atlas, and only when a `deliver` call
sends `blender: true`.

## Configure

| Variable | Meaning |
|---|---|
| `LEED_MCP_ROOTS` | The folders the server may read and write, separated by `:` (macOS, Linux) or `;` (Windows). Each must be an absolute path to an existing folder, neither inside this repository nor holding it, so not the folder that holds the repository (such as Desktop when the repository is on Desktop). Without one, every project path is refused (`ROOTS_NOT_CONFIGURED`). |
| `LEED_MCP_STATE` | Optional. The job index folder; default `~/.leed-mcp/`. It must lie outside every root. |
| `PATH` | Must hold the Radiance `bin` folder, and on macOS and Linux also `/usr/bin:/bin`. Claude Desktop starts the server without your shell's environment, so its configuration sets it. |
| `LEED_MCP_CALCULATOR` | Optional. A blank copy of USGBC's "LEED v4.1 Daylight and Quality Views Calculator" V02 (.xlsm), for `deliver(action="calculator")`. USGBC publishes it; it is not included here. `project(action="status")` reports whether it is set and supported. |
| `RAYPATH` | Optional. Without it the pipeline finds Radiance's library next to `rtrace`. |
| `BLENDER` | Optional. The Blender program; without it the server looks on `PATH` and in the standard install folder of your system. |

## Connect

### Claude Code plugin (two lines)

This repository is also a Claude Code plugin marketplace, so one install registers the server and the
`running-leed-studies` skill:

    /plugin marketplace add akontadakis/leed-mcp
    /plugin install leed-mcp@leed-mcp

Restart Claude Code and check with `/mcp`. The plugin launches
`uvx --from "leed-mcp[report] @ git+https://github.com/akontadakis/leed-mcp" leed-mcp` and declares no
environment variables: the server inherits your shell's, so put `LEED_MCP_ROOTS` and Radiance's `bin`
folder on `PATH` in your shell profile, or use the CLI line below, which takes `-e`. Manage the plugin
later with `/plugin`.

### Without a clone (uvx)

uvx fetches, builds and caches the server on first run.

**Claude Code (CLI):**

    claude mcp add leed-mcp --scope user -e "LEED_MCP_ROOTS=/Users/you/LEED projects" -e "PATH=/usr/local/radiance/bin:/usr/bin:/bin" -- uvx --from "leed-mcp[report] @ git+https://github.com/akontadakis/leed-mcp" leed-mcp

**Claude Desktop and other MCP clients:**

```json
{
  "mcpServers": {
    "leed-mcp": {
      "command": "uvx",
      "args": ["--from", "leed-mcp[report] @ git+https://github.com/akontadakis/leed-mcp", "leed-mcp"],
      "env": {
        "LEED_MCP_ROOTS": "/Users/you/LEED projects",
        "PATH": "/usr/local/radiance/bin:/usr/bin:/bin"
      }
    }
  }
}
```

Claude Desktop may not find `uvx` on its own: then give its absolute path as `command` (`which uvx`
prints it). With uvx, keep `LEED_MCP_ROOTS` on a projects folder, not your home folder: the home folder
holds uv's cache, where the server is installed, and a root may not hold the server.

### From a clone

The command is the absolute path of `.venv/bin/leed-mcp` in this repository. The examples use invented
paths: replace `/Users/you/leed-mcp` with this repository's folder, `/Users/you/LEED projects` with your
projects folder and `/usr/local/radiance/bin` with Radiance's `bin` folder.

**Claude Code:**

    claude mcp add leed-mcp --scope user -e "LEED_MCP_ROOTS=/Users/you/LEED projects" -e "PATH=/usr/local/radiance/bin:/usr/bin:/bin" -- /Users/you/leed-mcp/.venv/bin/leed-mcp

`claude mcp list` then shows `leed-mcp` as connected.

**Claude Desktop:** add the server to the Claude Desktop config file,
then quit and restart the app. The config file is `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows and `~/.config/Claude/claude_desktop_config.json` on Linux. On Windows the command is `C:\\Users\\you\\leed-mcp\\.venv\\Scripts\\leed-mcp.exe` (each `\` doubled in JSON) and `PATH` is `C:\\Radiance\\bin`:

```json
{
  "mcpServers": {
    "leed-mcp": {
      "command": "/Users/you/leed-mcp/.venv/bin/leed-mcp",
      "env": {
        "LEED_MCP_ROOTS": "/Users/you/LEED projects",
        "PATH": "/usr/local/radiance/bin:/usr/bin:/bin"
      }
    }
  }
}
```

Add `"BLENDER"` and `"RAYPATH"` to `env` when you need them.

**MCP Inspector**, to try the server by hand (it needs Node.js):

    npx -y @modelcontextprotocol/inspector -e "LEED_MCP_ROOTS=/Users/you/LEED projects" -e "PATH=/usr/local/radiance/bin:/usr/bin:/bin" -- /Users/you/leed-mcp/.venv/bin/leed-mcp

The server serves MCP protocol revision 2026-07-28 and, for older clients, 2025-11-25.

## A project folder

`project(action="create")` makes the folder and its `project.yaml`. Copy the scene files into its `model/`
folder and the weather file into `weather/` with your file manager. For an `.epw` file, `inputs(action="import_weather")`
converts it to a `.wea` file. Every file `project.yaml` names must lie inside the project folder. The
`new_project` prompt walks through the rest.

```text
office_a/
├── project.yaml            settings: site, scene, glazing, grid, analysis, Radiance options, calculator units, …
├── credit_rules.yaml       the LEED thresholds (do not edit)
├── materials.csv           one role per scene material (and the glass's vlt)
├── spaces.csv              one row per room; include_<case> and views_<case> columns pick the rooms
├── splits.csv              optional: lines that split a detected room in two
├── glare_narrative.md      written by deliver(action="report"); you fill in the marks
├── model/                  your Radiance scene (.rad)
├── weather/                your weather file (.wea, or .epw to convert)
└── _run/                   everything the server writes: prepared model, results, reports, atlas, logs
```

The scene rules, every `project.yaml` setting and every refusal code are in the guide resource
(`leed://docs/running-a-project`, the file [`leed_mcp/resources/running_a_project.md`](leed_mcp/resources/running_a_project.md)).
[`docs/RUNNING_A_PROJECT.md`](docs/RUNNING_A_PROJECT.md) is the full guide for the command-line pipeline.
[`docs/daylight-and-views-guide.html`](docs/daylight-and-views-guide.html) is a plain-language
introduction to the daylight and views credits and the workflow, for people new to them.

## Limitations

- LEED v4.1 BD+C only (New Construction credit rules). Other rating systems and versions are not covered.
- The scene must be made of polygons only, with a floor polygon in every room and glass polygons in the window
  openings. The guide lists the model rules.
- Runs need Radiance on this computer. A large annual run can take hours.
- One job runs at a time on a computer.
- Not covered:
  - glare-control devices: v4.1 asks for them in regularly occupied spaces; the tool does not check them;
  - healthcare bands and points;
  - Core and Shell, Schools and the other adaptations;
  - Daylight Option 3 (measurement);
  - Quality Views multiple lines of sight and the view-factor criteria;
  - daylight saving in the southern hemisphere: use `none`; the occupied-hour mask is then one hour off
    during local summer time;
  - the Word reports and the atlas are metric only.
- The Quality Views method follows the common reading of the credit (Type 2 and Type 3 views). Check it
  against the reference guide and your reviewer's expectations.

## Test

The tests need the `dev` extra, which includes `report` and keeps pytest. Install it first:

    uv sync --extra dev --no-editable

`uv sync` removes every package it was not asked for, so the server-only install above removes pytest. Then run:

    .venv/bin/python tools/check.py

It runs the whole suite and ends with `CHECK PASSED` or `CHECK FAILED`. On a computer with Radiance, run it
through `python tools/no_radiance.py` to keep every Radiance program from starting:

    python tools/no_radiance.py .venv/bin/python tools/check.py -q

The acceptance tests compare a full run with a real project's expected results. That project is not public, so
these tests are skipped without it (`LEED_ACCEPTANCE_FIXTURE`; [`docs/PIPELINE.md`](docs/PIPELINE.md) explains
it). `tools/time_scene.py "<project folder>"` times `scene(action="scan")` and `scene(action="rooms")` on a
real project.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Report a security problem as
[SECURITY.md](SECURITY.md) says, not in a public issue.

## License

[MIT](LICENSE) © 2026 Antonis Kontadakis.

The optional `report` extra installs `cairosvg`, which is licensed under the LGPL 3.0. It is a separate
package, imported only when you pick the `cairosvg` renderer. Radiance and Blender are separate programs with
their own licences.

## Acknowledgements

- [Radiance](https://www.radiance-online.org), by Lawrence Berkeley National Laboratory, does every lighting
  calculation.
- The [Model Context Protocol](https://modelcontextprotocol.io) and its Python SDK connect the server to AI
  clients.
