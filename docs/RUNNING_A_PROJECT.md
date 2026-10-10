# Running a project: the complete guide

This guide takes you from a Radiance model to three LEED v4.1 BD+C results: EQ Credit Daylight Option 1
(sDA, ASE and points, steps 1 to 8), EQ Credit Quality Views (step 9) and EQ Credit Daylight Option 2 (step 10). All three use the same project folder.
Every example uses an invented building called "office_a". Replace the names and numbers with your own.

---

## 0. What you need once

- **Radiance** installed, with `oconv`, `gendaylit`, `gendaymtx`, `gensky`, `rcontrib`, `rfluxmtx`, `rtrace`
  and `xform` on your `PATH`. Check it: open Terminal (Windows: PowerShell) and type `which rfluxmtx` (Windows: `where.exe rfluxmtx`). It must print a path.
  Quality Views alone needs only `oconv` and `rtrace`. Option 2 needs `oconv`, `rtrace`, `gendaylit`
  and `xform`.
- **This repository** at `<repository>`, installed once with uv. Install [uv](https://docs.astral.sh/uv/),
  then in the repository folder run `uv sync --extra report --no-editable --reinstall-package leed-mcp`.
  That gives the `leed-daylight` command in the repository's `.venv`. Run the same command again after every
  update of the repository: the commands run the installed copy, not the files in the folder. Every command
  below is `leed-daylight <verb>`. Type `<repository>/.venv/bin/leed-daylight` (Windows:
  `<repository>\.venv\Scripts\leed-daylight.exe`) instead if the `.venv` is not activated. (`--no-editable`
  matters when the repository is in an iCloud folder; the README explains why.)
- **Blender 4.2 or newer**, optional: only for the Blender file of the floor atlas (section 11). The tool
  finds it through the `BLENDER` environment variable, then `blender` on your `PATH`, then the standard install folder: `/Applications/Blender.app/Contents/MacOS/Blender` on macOS, `%ProgramFiles%\Blender Foundation\Blender *\blender.exe` on Windows (the newest version wins), `/snap/bin/blender` and then `/opt/blender*/blender` on Linux (the newest wins). Without it, everything else works.

---

## 1. Make a project folder

Make one folder per project, **outside** the repository. The tool refuses a folder inside it, so client
models can never end up in the code. Let the tool make it:

```
leed-daylight init ~/Desktop/office_a
```

It writes `project.yaml` (every value says `FILL_ME`; step 4), `credit_rules.yaml` (the LEED v4.1
rules; step 4) and the empty folders `model/` and `weather/`. It never overwrites a file and refuses a link, even a dangling one, where one of its files or folders
belongs. The tool
refuses to run while any value in `project.yaml` still says `FILL_ME`.

```
~/Desktop/office_a/
├── model/
│   ├── materials.rad      your Radiance materials
│   └── geometry.rad       your Radiance geometry
├── weather/
│   └── site.wea           the weather file
├── project.yaml           the settings           (step 4)
├── materials.csv          one role per material  (step 3)
├── credit_rules.yaml      the LEED thresholds    (step 4)
└── spaces.csv             one row per room       (step 5)
```

Folder and file names are up to you. `project.yaml` points at them.

---

## 2. Prepare the Radiance model

The model must follow these rules. The tool checks them and names any surface that breaks one.

1. **Polygons only.** Every surface is a `polygon`. Spheres, cones, rings and meshes are refused.
2. **Units.** Say which units the model uses (`m`, `mm`, `cm`, `ft` or `in`) in `project.yaml`. The tool
   converts to metres once.
3. **Every room has a floor.** Each room needs a flat, upward-facing polygon whose material has the role
   `floor`. The tool finds rooms from floors. One connected patch of floor is one room. A hole in that
   patch (an atrium or a shaft) is kept as a hole in the room: no sensor point falls inside it, and the
   view plans draw its outline.
4. **Windows are glass polygons** with a `glass` material whose role is `glazing` or
   `glazing_curtain_wall`, or with a diffusing `trans` material whose role is `glazing_diffusing`
   (rule 6). Put each window in its wall opening. Do not also cover the opening with the wall.
   A duplicate or collinear vertex in the opening outline is dropped before framing, so the framed glass
   can end up with fewer vertices than the opening you drew.
5. **Interior glass** (glass partitions between rooms) uses the role `glazing_interior`. It lets light
   through, but it gets no frame, no blind and no window group. **Frosted interior glass** (acid-etched or
   satin partitions) may be a diffusing `trans` with the same role: 7 real arguments, the last one `0`
   (no see-through part), as for diffusing roof glass. It is used as modelled, with no dirt factor, and
   it always blocks a view in Quality Views.
6. **Skylights, sloped glass and roof monitors.** Glass at any tilt is accepted. The tool measures each
   pane's tilt from flat (0 degrees is flat, 90 is vertical) and puts it in one of three classes:
   **vertical** (85 to 90 degrees), **sloped** (20 to under 85) and **skylight** (under 20). Glass that the
   window rule (rule 9) places on a room's facade is facade glass, exactly as before. If the room the
   window rule finds (the nearest one within 0.6 m on the pane's storey) is not named in `spaces.csv`, the
   pane is refused as `unmatched exterior glazing`, as before: it is never passed on to a room below.
   Roof glass is every skylight-class pane, vertical or sloped glass with no room within 0.6 m on its
   storey, and glass more than 0.6 m inside a room's outline on both sides. For roof glass the tool looks
   straight down from its centre and gives it to the first room whose floor outline lies under it, on
   the pane's own storey or a storey below. That room must be named in `spaces.csv`. A skylight, or roof
   glass the look-down gives to a room `spaces.csv` does not name, is refused (`roof glass over no named
   room`). Vertical or sloped glass with no room under it either is refused as `unmatched exterior
   glazing`.
   - A skylight's outside is its upper side, whichever way you drew it. Model the skylight well (the
     shaft between the glass and the ceiling opening) yourself, or let the tool build straight vertical
     walls under the glass (`glazing.skylight_well`, step 4). Leave the ceiling open under a skylight: a
     `ceiling` surface under one is warned about, because it blocks the light.
   - Vertical glass on a roof (a roof monitor, a sawtooth) faces the side whose straight-down look does
     not land on its room's floor. If both sides land on the floor, or neither does, it is refused
     (`monitor glass reaches its room's floor on both sides` / `on neither side`): check the floors and
     roof surfaces under it.
   - Each kind gets its own window groups: `<room>_sky` (all of a room's clear skylights), `<room>_s<azimuth>`
     (sloped glass), `<room>_m<azimuth>` (roof monitors) and `<room>_dif` (all of a room's diffusing glass).
     Facade windows keep `<room>_<azimuth>`.
   - Clear roof glass gets an operable blind, like a window (LM-83-12 2.2.6). Diffusing glass gets none.
   - **Diffusing glass** (opal or translucent panels, at any tilt, on a roof or a facade) is a Radiance
     `trans` material with the role `glazing_diffusing`: 7 real arguments, the last one (the transmitted
     specularity) `0`, so nothing is seen through it. It lets in no direct sun, gets no blind and is never
     a view.
   - **Known limit.** Vertical or sloped glass on a roof whose centre is within 0.6 m of the room outline in plan
     (a north light right at the eave, say) is read as facade glass, as before: it gets a facade window group
     and is oriented by the facade rule. A skylight over a double-height space with no room above it at that
     point is given to the room on the storey below.
7. **Every surface has its own name.** Two surfaces with the same name are refused.
8. **The glass material is the product's real value.** The tool reads the glass transmissivity from your
   `glass` material and turns it back into visible transmittance, then applies the dirt factor. If your
   glass material already includes dirt, say so in `project.yaml` (`vt_includes_dirt: true`). The glass
   primitive's refractive index must be `1.52` (Radiance's own default) or the conversion is refused; a
   `glass` primitive takes 3 real arguments (r g b transmissivity) or 4 (plus that index).
9. **Windows need a room.** Each exterior window must lie within 0.6 m of the room behind it. Roof glass
   follows rule 6. That room must be named in `spaces.csv` (step 5), or the run is refused. Glass drawn
   set back from a floor outline that was traced to the exterior wall face is still read correctly, as
   long as it sits within 0.6 m of that outline on exactly one side.

The tool adds the window frames, the reveals and the blinds itself, and, if you ask, the skylight wells.
Do not model them, except a skylight well you model on purpose (`skylight_well: "modelled"`).

**Reading the `.rad` files.** The reader supports the `!xform` options a daylight model actually needs
(`-t`, `-rx`/`-ry`/`-rz`, `-s`, `-mx`/`-my`/`-mz`, `-n`), several files on one `!xform` line, `alias`
primitives, quoted words (spaces allowed inside matching quotes), and a command line continued onto the
next with a trailing backslash. `!xform`'s `-n prefix` renames only surfaces, sources and instances, as
Radiance's own `xform` does -- never a material, pattern or texture, and never a modifier reference.
A `source`, `instance`, `mesh` or any other primitive type this reader does not know is refused by name.
When the tool runs Radiance, each scene file's own folder and the prepared model folder go on
`RAYPATH`, so a relative `.cal`, `.xml` or data-file reference inside your `.rad` files resolves. If
`RAYPATH` is not set, the tool finds Radiance's library next to the `rtrace` it runs (the folder that
holds `rayinit.cal`); if it finds none, `radiance.log` says so after its version line, because
`RAYPATH` then holds only the scene folders and Radiance's built-in library path is no longer
searched: set `RAYPATH` yourself to include the library. The scene files' folders come first on
`RAYPATH`, so a file there with the name of a Radiance library file (for example `rayinit.cal`) replaces
the library's own: do not put such a file next to the scene. A surface may use a material defined in a
later file; a material that uses another material or pattern must come after it. A keyhole floor (one
polygon wrapped around a hole) is read as a floor with a hole, the same as floor pieces around it. Words
holding spaces, `#` or quotes keep their quotes in the prepared model.

**Room-detection warnings.** Finding rooms from floors (rule 3) also warns, without stopping the run: a
storey-to-storey or floor-to-ceiling height outside 2.0-10.0 m (the declared `scene.units` may be wrong,
cm vs mm or in vs ft); a sloped, non-flat ceiling; a floor or ceiling polygon that is not a polygon or is
malformed, or whose edges cross (each is skipped rather than used); a floor whose outline overlaps itself,
so its plan area differs from the drawn area (usually a hole loop wound the wrong way).

---

## 3. Fill in materials.csv

List **every** material your model uses, one per row, with its role. Once `project.yaml` is filled in
(step 4), the tool writes a first draft for you:

```
leed-daylight materials ~/Desktop/office_a
```

It writes `~/Desktop/office_a/_run/materials_template.csv`: every material the scene uses, largest area
first, with an empty role and the tool's guess in the note. Copy it to `materials.csv` and fill every
role; an empty role is refused. Exact header:

```
modifier,primitive_type,role,note
glass_clear,glass,glazing,south and east windows
glass_cw,glass,glazing_curtain_wall,curtain wall
glass_partition,glass,glazing_interior,meeting room partitions
glass_opal,trans,glazing_diffusing,rooflight diffusers
wall_paint,plastic,wall,
ceiling_tile,plastic,ceiling,
floor_carpet,plastic,floor,
louvre_metal,plastic,fixed_shading,external fins
site_context,plastic,context,neighbouring buildings
site_ground,plastic,ground,
desk_top,plastic,furniture,
helper_lines,plastic,ignore,construction helpers
```

- `modifier` is the material name exactly as in your `.rad` file.
- `primitive_type` is its Radiance type (`plastic`, `glass`, `metal`, `trans`...). If you fill it in,
  it must match the scene: a row that says `glass` for a material the scene defines as `plastic` is
  refused. Leave it empty to skip the check.
- `role` is one of: `glazing`, `glazing_curtain_wall`, `glazing_diffusing`, `glazing_interior`, `wall`,
  `ceiling`, `floor`, `fixed_shading`, `context`, `ground`, `furniture`, `ignore`. `glazing_diffusing` is
  diffusing glass (step 2, rule 6): the materials command guesses it for a `trans` whose seventh argument is
  `0`, and `fixed_shading` for any other `trans`. Its `view` value (step 9) may only be empty or `blocks`.
  `glazing_interior` takes a `glass` or a frosted (diffusing) `trans` (step 2, rule 5); on a `trans` its
  `view` may also only be empty or `blocks`. The materials command still guesses `glazing_diffusing` for
  such a `trans`: change it to `glazing_interior` for a partition inside the building.
- `note` is free text for you. It may be empty.

A material in the model with no row here is refused by name.

For Quality Views you may add a fifth column, `view`, at the end of the header. Step 9 explains it.
A file with the four columns above still loads.

Save `materials.csv` and `spaces.csv` (step 5) as UTF-8, comma-separated. A file saved in a Windows code
page, or with `;` as the delimiter (some regional Excel settings), is refused by name; re-save it as
"CSV UTF-8 (Comma delimited)".

---

## 4. Fill in project.yaml

Start from `tests/fixtures/project_example.yaml` in the repository and change the values. Every section
below is **required**. There are no hidden defaults: a missing value is refused.

Relative paths in this file are read from the project folder.

| Section | Keys | What to put |
|---|---|---|
| `project` | `name`, `credit` | Free text, printed in reports. |
| `site` | `latitude_N`, `longitude_E` | The site in degrees. North and east are positive. |
| | `meridian_radiance` | The time-zone meridian in Radiance's convention: degrees **west** positive (Central Europe is `-15.0`). A warning appears if it is more than 30° from the site's expected meridian (`-longitude_E`); it is refused instead if the signs suggest the east-positive value was written by mistake. |
| | `north_rotation_deg` | The angle between true north and the model's +Y axis, counter-clockwise positive. `0.0` means +Y is north. |
| | `north_rotation_source` | One sentence saying how you know the angle. It is printed in every report. |
| `scene` | `units`, `files` | `m`, `mm`, `cm`, `ft` or `in`; then the list of `.rad` files, materials first. |
| `standard` | `daylight`, `credit_rules` | `LM-83-12` or `LM-83-23`; the path to the LEED rules file (see below). |
| `weather` | `file` | The `.wea` file (see below). |
| `glazing` | `dirt_factor` | LM-83-12 clause 2.2.9 dirt factor, between 0 and 1: one number when every exterior pane is vertical, or one value per tilt class present, e.g. `{vertical: 0.95, sloped: 0.90, skylight: 0.85}` (step 2, rule 6). LM-83-12's minimum dirt losses are 5 %, 10 % and 15 %: a `sloped` value above 0.90 or a `skylight` value above 0.85 is warned about. With `vt_includes_dirt: true` the factor is not used, so one number is accepted whatever glass the scene has. |
| | `frame_share` | The share of each window's area that is frame, from the manufacturer. `default:` is required; add a line per glazing role that differs, e.g. `glazing_curtain_wall: 0.15`. Add `sloped:` when the scene has sloped glass and `skylight:` when it has skylights; for a pane of that class it wins over its role's line. LM-83-12 assumes 20 % for windows and 10 % for skylights when the frame is not known. |
| | `frame_reflectance` | The frame colour's reflectance, 0 to 1. |
| | `frame_style` | `constant_width` (a frame of even width) or `uniform_scale`. |
| | `reveal_depth_m`, `reveal_reflectance` | The window reveal depth in metres (`0` turns reveals off) and its reflectance. No reveal is built on a sill at the floor or a head at the ceiling (within 0.01 m), nor on a joint between two coplanar panes such as curtain-wall mullions. |
| | `skylight_well` | Required when the scene has skylight-class glass. `"modelled"`: your scene carries the well and the tool adds nothing. A number above 0.05 (metres, like `reveal_depth_m`): the tool builds straight vertical walls that deep under every skylight, from the glass's lowest corner down, and skips the edge between two skylight panes of one room. Splayed wells must be modelled. |
| | `skylight_well_reflectance` | The built well's reflectance, 0 to 1. Required when `skylight_well` is a number; refused with `"modelled"`. |
| | `vt_includes_dirt` | `true` only if your glass material already includes the dirt factor. |
| `reflectance` | `policy`, then one value per role | `keep_as_modelled` uses your materials as they are; `override` replaces them with the values here; `fill_gaps` replaces only missing ones. Role keys: `ceiling`, `wall`, `floor`, `context`, `ground`... A role you do not list is left as modelled. Under `override`, a warning names any role present in the scene with no value given here; that role keeps its modelled reflectance. A key here that is not `policy` or a role name from `materials.csv`, or a key under `glazing.frame_share` that is not `default`, `sloped`, `skylight` or such a role name, is refused by name, and the message lists the valid ones. |
| `grid` | `spacing_m`, `work_plane_m`, `min_clearance_m` | LEED: spacing at most `0.60`; work plane `0.762` (30 in); a clearance from the walls. |
| | `points_file` | Optional. Leave it out and the tool lays its own grid in every space. Give a file name to use your own sensor points instead (below). |
| `analysis` | `hours` | Occupied hours in local clock time, e.g. `"08:00-18:00"`. |
| | `daylight_saving` | `EU`, `US` or `none`. The day daylight saving ends counts as standard time. DST dates follow the year 2025 calendar (the tool's typical-meteorological-year convention); the year used is recorded in `option1/<case>/results.json` under `settings.daylight_saving_year`. |
| | `aggregation` | `floor_area` (what LEED uses) or `points`. Both are always reported; this picks the headline. |
| | `moveable_furniture` | `include` or `exclude`, as modelled. |
| `performance` | `trace_scope`, `linked_zones` | Use `trace_scope: "all"` and `linked_zones: {}`. `storey` is faster but not yet proven equal. |
| `sky` | `basis`, `MF` | `Reinhart` and `4`. |
| `radiance` | `ambient.draft`, `ambient.full`, `direct` | Radiance settings. The run uses `full`. Higher `-ab`/`-ad` is slower and more accurate. `direct` takes only `-d*` and `-s*` options (`-dj`, `-dt`, `-ds`, `-sj`, and so on); any other option there is refused ("option ... is not allowed here"), and the ambient options go in `ambient`. With diffusing glass (`glazing_diffusing`) in the scene, the last `-ab` in `ambient.full` followed by `direct` must be 2 or more, or the run is refused: light through a diffuser reaches a sensor only through ambient bounces. An `@file` of options is refused then, because its `-ab` cannot be checked. That light is sampled by ambient rays, so it is noisier than light through clear glass; a higher `-ad` reduces the noise. |
| `blinds` | `transmittance_diffuse`, `transmittance_specular`, `room_side_reflectance` | Your blind product, or the LM-83-12 clause 2.2.7 values if you use them. |
| | `trigger_area` | Optional: `room` (the default) or `grid`. The area that sets each space's LM-83-12 2.2.6 blind threshold (under 200 sf more than 1 lit point closes blinds, under 500 sf more than 3, under 1,000 sf more than 5, then more than 2 % of the points). `room` is the room area from the scene; `grid` is the area the space's sensor points stand for, the point count times `spacing_m` squared. Use `grid` only to match a study that set its blind thresholds that way; the area weighting of sDA and ASE always uses the room area. |
| `option2` | `equinox_dates`, `clearest_day_window_days`, `times`, `sky`, `lux_min`, `lux_max`, `blinds`, `time_basis` | LEED values: `["03-21", "09-21"]` (one March date, then one September date), `15`, `["09:00", "15:00"]` (quoted whole hours, no repeats), `"weather_file"`, `300`, `3000`, `"exclude"`, `"weather_file_local"`. Option 2 refuses any other `sky`, `blinds` or `time_basis`; `run` and `views` still load a file whose `time_basis` is `UNVERIFIED`. |

**Sensor points from a file.** To compare with an earlier study on its own sensor points, write them to a
CSV file in the project folder and name it in `grid.points_file` (e.g. `points_file: "points.csv"`). The
header is exactly `space,x,y,z`; each row is one point: its space's name as the `room` column of
`spaces.csv` writes it (a space row, not a piece row), and its position in metres, whatever `scene.units`.
`z` is the work plane: the floor level plus `work_plane_m`. Every regularly occupied space of the case
needs at least one point; the points are used in file order, with no clearance or spacing rule. A point
outside its space's floor (on the outline, in a hole, or in another room), off the work plane by more
than 1 mm, or naming a space `spaces.csv` does not have, is refused by its line (section 8). A point
exactly on a split line is refused too, since it must lie strictly inside its piece, while the tool's own
lattice keeps such points. Points of a space the case leaves out are not used, so one file serves every
case. The file is an input of the results: changing it makes them stale. Option 1, Option 2, the atlas
and the reports use the points; Quality Views keeps its own grid at eye height. With `grid.points_file`
set, `prepare` (which `views` also runs) still needs points for every included, regularly occupied space
of the case. The `package` command copies the points file into `config/`; it may lie outside the project
folder, but not be a link, and its file name must not be one the package writes there (`project.yaml`,
`materials.csv`, `spaces.csv`, `credit_rules.yaml`, `splits.csv`).

**Projects made before skylights were supported.** Earlier versions read glass tilted more than 5 degrees
from flat as a window, with one dirt factor and a window frame, and refused flat glass. Now a project whose
exterior glass includes any pane tilted under 85 degrees from flat (sloped or skylight class) is refused until
`glazing.dirt_factor` has one value per class present and `glazing.frame_share` has `sloped:` (and
`skylight:` where there are skylights); with skylights it also needs `glazing.skylight_well`. A project with
only vertical glass is unaffected: its prepared model and results are the same as before. Its Option 1
`results.json` has no `groups` list, though, and `report` and `package` refuse it with "run again": run
`run` (Option 1) once more before `report` or `package`.

**The LEED rules file.** Copy
`leed_daylight/templates/credit_rules_v41_bdc_nc.yaml` from the repository into your project folder as
`credit_rules.yaml`, and write `credit_rules: "credit_rules.yaml"`. It holds the LEED v4.1 thresholds
(sDA 300 lux / 50 %, ASE 1000 lux / 250 h, the point bands). Do not edit the numbers.
`option1.sda.additional_point.only_if_points` was renamed from `only_if_points_at_least`; the old name is
refused by name if it is still in your file. It also carries the optional `intent` sentence (read by the
Option 1 and Option 2 reports), `views.intent` (Quality Views) and `option1.space_rule_sda_pct` keys the
reports read (section 11); the shipped template already has all three.

**The weather file.** A Radiance/DAYSIM `.wea` file with 8,760 hourly lines in local standard time.
Make one from an EPW with Radiance's `epw2wea`:

```
epw2wea site.epw ~/Desktop/office_a/weather/site.wea
```

An `.epw` file cannot be used directly. The tool checks it before running `gendaymtx`: the six `.wea`
header lines must all be present; exactly 8,760 data lines, starting 1 January, chronological with no
gap or repeat; every hour stamped at its middle (`x.5`; an hour-ending integer stamp moves every sun 30
minutes) and no 29 February. A warning appears if the header's site is more than 1° from `project.yaml`'s
(the header's longitude is west-positive, same convention as `meridian_radiance`).

**YAML gotchas.** `1e3` reads as text in YAML -- write `1.0e3`. An unquoted clock time such as `15:00`
reads as the base-60 number 900 -- quote it: `"15:00"`. An unquoted date reads as a date, not text --
quote it. `credit_rules.yaml`'s `views.version` must be a quoted string (step 9).

---

### Check the scene

When `project.yaml` is filled in, check the model before anything else:

```
leed-daylight scan ~/Desktop/office_a
```

It prints the scene's size in metres (a wrong `scene.units` shows at once), every material with its
surface count and area, and every problem the checks in step 2 find. With `materials.csv` in place it
also prints each storey's level, room count and floor area. It writes nothing. It ends with an error
line when the scene has a problem.

With a `splits.csv` (step 5) and `materials.csv` in place, `scan` also prints `split lines: 1` (the
number of lines) before the storeys, and the storeys are those after the splits. A line that cannot
be applied is a problem like any other: `split problems: 1`, the reason on the next line, and the
error line at the end. Without `materials.csv`, `scan` does not read `splits.csv`.

---

## 5. List the rooms and fill in spaces.csv

`spaces.csv` tells the tool which rooms count and what they are called. First let the tool show you the
rooms it found:

```
leed-daylight rooms ~/Desktop/office_a
```

It prints one line per room (area, centre point, floor level). It also writes
`~/Desktop/office_a/_run/spaces_template.csv`. Copy that file to `~/Desktop/office_a/spaces.csv` and edit
it:

```
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base,reason
L0,Office North,office,yes,48.00,6.00,4.00,0.00,no,,1,
L0,Corridor,circulation,no,20.00,6.00,9.50,0.00,no,,1,not regularly occupied
L1,Meeting 1,meeting,yes,24.00,3.00,4.00,3.50,no,,1,
```

- `room`: your name for the room. Change `R1`, `R2`... to real names.
- `regularly_occupied`: `yes` for rooms LEED counts. Only these get a sensor grid.
- `area_m2`: leave as printed. Results always use the room area from the scene. If your value is
  more than 5 % away from it, `prepare` and every run print a warning: check that the row matched
  the room you meant.
- `centroid_x`, `centroid_y`, `level_z`: **leave as printed.** They link the row to the detected room
  (within 0.5 m and 0.05 m).
- `merge_into`: leave empty, unless the room is one piece of a larger space (below, **Merge rooms**).
- `include_base`: `1` to include the room in the main case, `0` to leave it out. For a scenario, add a
  column such as `include_option_b` before `reason`, and choose it with `--case option_b`.
- Every room that has an exterior window must have a row.
- For Quality Views, add `views_<case>` columns after the `include_<case>` columns, and optionally
  `views_reason` at the very end. Step 9 explains them. A file without them still loads and `run` works.

**Split a room: `splits.csv`.** When the tool finds one room where LEED counts two spaces (an open
office and a meeting area with no wall between them, say), draw the dividing line in an optional
`splits.csv` in the project folder:

```
level_z,points,note
3.50,"10.0 0.0; 10.0 6.0; 16.0 6.0",meeting area edge
```

(Invented numbers.)

- The header is exactly `level_z,points,note`. Save the file as "CSV UTF-8 (Comma delimited)", like
  the other CSV files (step 3).
- `level_z`: the storey's floor level, as `rooms` prints it (within 0.05 m).
  **Metres, whatever `scene.units`**, like `level_z` and the centre points in `spaces.csv`: a line
  copied from a millimetre drawing is refused.
- `points`: two or more `x y` pairs in metres, separated by `;`. A bend is one more point. Two
  consecutive points must be at least 0.01 m apart, and the line must not cross or overlap itself.
- `note`: your own text; the tool never reads it.
- One line per row, applied in file order. A later line may cut a half an earlier line made.
  Messages number the rows counting the header as line 1.

Draw each line so that it crosses exactly one room, with each end on the room's outline or outside
it. Running along a neighbouring room's wall or touching its corner is not crossing it. An end point
more than 0.01 m inside the room is refused; one within 0.01 m of the outline is moved onto it. The
line must cut the room into exactly two parts, each half at least 1 m2, and it must not meet the
room's exterior glass: a split must end between panes (at a joint between two facade panes, or along
a mullion between two roof panes), so that each pane lights one half. Glass set back from the room's
outline counts too. Each refusal names the `splits.csv` line, the room and what to change (step 8).

Then run `rooms` again. It prints `split lines applied: 1 (splits.csv)` first, and the template gets
one row per half, with the half's own centre point. A half's id is the room's id plus `a` (the half
on the left of the line, walking from its first point to its last) or `b` (the right); a half cut
again gets a second letter (`S1_R0ab`). Copy the halves' centre points into `spaces.csv`. A row that
still holds the old room's centre point is refused, and the message says to run `rooms` again. A half
is a room like any other, with its own ceiling height, floor names, window groups and grid. Its grid
points are the points the whole room would have had on its side of the line, so the grid runs up to
the line with no clearance there.

**Merge rooms: `merge_into`.** When the tool finds two or more rooms where LEED counts one space (two
rooms joined by a wide opening, say), keep one row per room. Choose the row that stands for the space,
the **space row**, and write its `room` name in the `merge_into` of every other row. A row with
`merge_into` set is a **piece row**. A chain (A into B, B into C) makes both A and B pieces of C.

- A piece row must have the same `floor` and the same `include_<case>` and `views_<case>` ticks as
  its space row, column by column. Otherwise it is refused, naming both rows and the first column
  that differs.
- The space's `space_type`, `regularly_occupied` and `view_preserving_shades` come from the space row.
  A piece row with another value gives a warning, and its value is not used.
- `area_m2` stays each row's own room area, and each row keeps its own room's centre point.
- The pieces must be on one storey: a piece on another storey is refused. Merging across storeys is
  not supported.
- `performance.linked_zones` names space rows only; a piece row there is refused.
- Every piece of a regularly occupied space needs at least one grid point, or the space is refused
  naming the piece (LM-83-12 2.2.6: the grid extends across the entire space).

A merged space is one space from then on: one grid (every piece's points together), one row in the
results, one name in the reports. Its window groups are formed per piece: a space of several pieces
adds `_p1`, `_p2` and so on after the space's name (`Open_Office_p1_180`), and a window equally near
two pieces of one space goes to the first. **Blinds respond to the whole space**: Option 1 works out
one blind schedule for a merged space, over all its points and its whole area, and the 929.03 m2
warning (step 7) uses the whole area too. When a merged space has a window group that can carry a
blind, the Option 1 report says so in its method section, one sentence per such space; a merged space
with no blind-capable window group, or a case's Option 2 and Quality Views reports, get the sentence
without that clause. A split and a merge can be combined: a half can be a piece of another space.

---

## 6. Run it

First, a quick check that needs no Radiance and takes seconds:

```
leed-daylight prepare ~/Desktop/office_a
```

It builds the prepared model and prints the rooms, grid points and window groups of the case (by kind:
window, sloped, monitor, skylight, diffusing, with one line for each group that is not a window), and every
warning. A split or merged space gets one more line under its room line, naming its pieces:
`    pieces (merged): S0_R2 12.00 m2, S0_R3 8.00 m2`; a half adds the `splits.csv` line that made it,
as in `S0_R0a 18.00 m2 (splits.csv line 2)`. Then run the simulation:

```
leed-daylight run ~/Desktop/office_a
```

To run a scenario instead of the first case: add `--case option_b`. Add `--processes N` (an integer,
default 1) to run `rfluxmtx`/`rcontrib` with `-n N`: the same result, faster on several cores. It is not
part of the cache, so changing it between runs does not force a recompute.

The run takes from minutes (a small building, low settings) to many hours (a large building, full
settings). You can stop it with Ctrl+C at any time: the prepared model (`scene.rad`, `materials.rad`,
`blinds/`, `grids/`, `manifest.json`) is written to a temporary name and only then renamed into place,
manifest last, so a stopped or failed build never leaves a half-written prepared model. Run the
same command again and it continues: finished results are reused, and only what changed is recomputed.

**One command at a time per folder.** `prepare`, `atlas`, `run`, `option2`, `views`, `report` and
`package` each hold the lock file `.leed-daylight.lock` in the project folder while they work, whether
you typed them. A second one on the same folder is refused:
`error: run is already working on this folder (started 14:02, process 51234)`. **This is a change:**
`report` during a `run` is refused; wait for the run to end. `scan`, `materials` and `rooms` only read,
and take no lock. A command keeps the lock file open with the operating system's file lock on it, which
ends with its process however it ends. A lock left by a command that was killed (the computer restarted,
the window was closed) is a stale lock: the next command sees that its process is gone, takes it over and
says so on one line, `note: cleared a stale lock left by run (process 51234, started 14:02)`. When the process still
exists but its start time cannot be checked, the lock is kept and the refusal names the lock file: if
that process is not a `leed-daylight` command, delete the file by hand. A folder at that name is refused:
move it away.

---

## 7. Find the results

**On screen**, at the end of the run:

```
case: base
occupied hours: 3650
sDA (floor_area): 62.5% (points: 60.1%)
ASE (floor_area): 8.0% (points: 7.5%)
score: 2 points (55.0 % or more)
  Office North: 48.0 m2, sDA 70.0%, ASE 12.0% (over 10 %: unsatisfactory); glare narrative needed
  Meeting 1: 24.0 m2, sDA 47.5%, ASE 0.0% (under 3 %: clearly acceptable)
warnings: 2
results written to ~/Desktop/office_a/_run/option1/base/results.json
```

(Invented numbers.)

- **sDA** is spatial daylight autonomy 300 lux / 50 % of hours, with blinds operating.
- **ASE** is annual sunlight exposure 1000 lux / 250 hours, with blinds open. Each room's ASE band, from
  IES LM-83-12 3.1.3, is one of "under 3 %: clearly acceptable", "3-7 %: neutral", "7-10 %: not classified
  by LM-83-12" or "over 10 %: unsatisfactory".
- **score** is the LEED v4.1 Table 1 points. A percentage is only compared to a band threshold after
  rounding both to 9 decimal places, so a result that is exactly on an edge in exact arithmetic but
  lands a hair below it in floating point still lands on the edge; floor-area weighting is built from
  each room's exact `passing / total` point counts, not from an already-rounded percentage. "not
  awarded" means no points are given, and the line says why. Besides an unverified threshold in the rules
  file, there are two cases. Option 1 gives no points unless `analysis.hours` is `08:00-18:00` and
  `analysis.aggregation` is `floor_area` (the basis of LEED Table 1); the percentages are still computed and
  reported, but they are not a LEED result. Option 2 gives no points unless `option2.times` is exactly
  `09:00` and `15:00` and `lux_min` and `lux_max` match the rules file's band; its share is still
  computed. Each case adds a warning ("..., so no points are awarded"), and the package's README
  shows "not awarded" for it.
- **glare narrative needed** marks a room with ASE above 10 %. LEED asks you to explain how that room
  handles glare.
- **warnings** are things that may be intended but you should check. They are listed in full. Among them:
  an implausible storey or ceiling height, a sloped ceiling, or a malformed floor/ceiling (see step 2); a
  room's blind analysis area over 929.03 m2 (LM-83-12 2.2.6's 10,000 sf), pooled into one schedule instead
  of split by facade; light reaching a room through a surface outside every window group, from the
  residual pass (see below); a reflectance `override` with no value for a role present in the scene.

**The residual pass.** Every room's Option 1 trace includes one extra pass with every window group's
opening blacked out and no blinds. It measures light that enters through a surface outside every window group
(glass given the role `ignore`, say, or a gap). Every window group's own trace already carries that light,
so a room lit by several groups would count it several times: the tool takes off the extra copies, and the
light counts exactly once. A blind never closes because of that light. Its share of each room's sky
illuminance is `spaces[].residual_sky_pct` in `option1/<case>/results.json`; a warning names the room when
that share is over 1 % or when direct sun through those surfaces reaches 1000 lux at a sensor point.

**In files**, in `~/Desktop/office_a/_run/`:

| File or folder | What it is |
|---|---|
| `option1/<case>/results.json` | Every number above, per room and for the project, plus the blind hours and warnings. Also records `settings.daylight_saving_year` and, per room, `residual_sky_pct` (see below). It also lists every window group under `groups`: its kind, tilt, facade azimuth (`null` for skylight and diffusing groups) and whether it has a blind. Results written before this version have no `groups`; the report and package commands skip them and ask you to run again. |
| `option1/<case>/points.csv` | One row per sensor point: daylit share of occupied hours, sDA pass, ASE hours, ASE exceeded. |
| `radiance.log` | Every Radiance command the run executed, with its exit status and time. A command that never started at all is logged too, as `-> failed to start`. |
| `manifest.json` | What the tool built: glass areas, frame widths, window groups, rooms, checksums. |
| `scene.rad`, `materials.rad` | The prepared model, with frames, reveals and dirt applied. |
| `blinds/`, `grids/` | The blind geometry per window group, and the sensor points per room. |
| `engine/` | The stored Radiance matrices. Keep it to resume; delete it to force a full rerun. They are rebuilt by themselves after a Radiance upgrade, or when a `.cal`, `.dat` or picture file a material uses has been edited. |
| `spaces_template.csv` | The starter `spaces.csv` the rooms command wrote (step 5). |
| `views/` | Quality Views results and working files (step 9). |
| `option2/` | Option 2 results and working files (step 10). |
| `reports/` | Word reports, plots and the glare narrative draft (section 11). |

---

## 8. When something is refused

The tool prints one line starting with `error:` and stops. A YAML, geometry or file-system error is
folded into that same one-line `error:` message too, rather than a raw traceback. The line names the
object and what was wrong. Common ones:

| The message mentions | What to do |
|---|---|
| `modifier ... has no role in materials.csv` | Add that material to `materials.csv`. |
| `unsupported surface type` | Convert that surface to polygons. |
| `... is not supported in the scene` | A `source`, `instance`, `mesh` or other primitive type the reader does not know. Remove it, or convert it to polygons (step 2). |
| `unmatched exterior glazing` | Add the room behind that window to `spaces.csv`. This includes a window within 0.6 m of a room `spaces.csv` does not name, even when a named room lies on a storey below (step 2, rule 6). |
| `not planar` / `degenerate polygon` | Fix that polygon in the model. |
| `polygon crosses itself` | Redraw that polygon as a simple polygon (a keyhole shape is fine; edges must not cross). |
| a modifier `is used before it is defined` / `is never defined` | A material that uses another material or pattern must come after it; move the definition up, or add the missing one. A surface may use a material from any file. |
| a modifier/name `is defined twice with different contents` | Two definitions of the same name must be identical; rename or remove one. |
| an `alias ... refers to ... which is not defined` | Point the alias at a modifier that is actually defined, or add that modifier. |
| `unterminated ... quote in a primitive` | Close the quote around that word in the `.rad` file. |
| `glass ... has refractive index ...; only 1.52 is supported` | Use a `glass` primitive with refractive index 1.52 (Radiance's own default), or omit the fourth real argument. |
| `roof glass over no named room` | No room named in `spaces.csv` lies under that roof glass. Add the room below to `spaces.csv`, or check that the glass sits over a floor (step 2, rule 6). |
| `roof glass lies on the boundary of more than one room` | The glass's plan centre sits exactly on the shared edge of two rooms' footprints. Move the glass, or adjust the room outlines, so it clearly covers one room (step 2, rule 6). |
| `monitor glass reaches its room's floor on both sides` / `on neither side` | The tool cannot tell which side of that vertical roof glass is outside. Check the floors and roof surfaces under it (step 2, rule 6). |
| `glazing.dirt_factor` with a tilt class (`vertical`, `sloped`, `skylight`) | The scene has glass of that class and `dirt_factor` has no value for it, or is one number while the scene has sloped or skylight glass. Give one value per class (step 4). |
| `glazing.frame_share.sloped` / `glazing.frame_share.skylight` | The scene has glass of that class; add the key (step 4). |
| `glazing.skylight_well` / `glazing.skylight_well_reflectance` | The scene has skylights and no well setting, or a depth without a reflectance, or a reflectance with `"modelled"` (step 4). |
| `glazing_diffusing` with `glass`, or `trans` with another glazing role, or a `trans` argument | Diffusing glass must be a `trans` with the role `glazing_diffusing` (exterior) or `glazing_interior` (frosted partitions), 7 real arguments and the last one `0`; the message names the argument that is out of range. |
| `grid.points_file line ...` | A point of the sensor points file is outside its space, off the work plane, or names no space row; the message names the line and the point (step 4, **Sensor points from a file.**). `grid.points_file has no point for space ...`: add that space's points, or leave it out of the case. |
| `grid.points_file: file does not exist or is not readable` | The sensor points file `project.yaml` names is not there; check the name and that it sits in the project folder (step 4). |
| `the header must be exactly space,x,y,z; the first line has N field(s)` | The first line of the sensor points file must be `space,x,y,z`. The message counts the fields; it does not repeat the line. |
| `line N: x ... is not a finite number; give metres` (or `y`, `z`) | That coordinate is empty, text, `nan` or `inf`; give a number in metres. A value shown in a refusal is escaped and cut to 40 characters. |
| `line N: the point x ..., y ... of space ... is already on line M` | Two rows of the sensor points file give the same space, `x` and `y`. Remove one of them; a point is counted once. |
| `line N: more than 1000000 points; a points file holds at most 1000000` | The sensor points file has more than 1,000,000 points; the message names the line where the limit is passed. Use fewer points. |
| `-ab` with diffusing glass | Set the last `-ab` in `radiance.ambient.full` plus `radiance.direct` to 2 or more, and do not use an `@file` for the options (step 4). |
| `cannot tell which side of this opening faces the room's interior` | The glass is within 0.6 m of the floor outline on both sides at equal distance, or outside it on both sides at equal distance. Move it so one side is nearer. Glass more than 0.6 m inside the outline on both sides is roof glass now (step 2, rule 6), not refused. |
| `uniform_scale cannot frame this concave opening` | No point sees the whole opening (a U-shape, say). Use `frame_style: constant_width` instead. |
| a frame refusal naming a window | Often a window outline with an extra point on a straight edge. Remove that point in the model. `frame_style: uniform_scale` also avoids it, but it draws the frame differently (a scaled copy of the outline, not an even width). |
| `inside the repository` | Move the project folder out of `<repository>`. |
| `no room found at level_z` / `outside the ... tolerance` / `matches more than one room` | A `spaces.csv` row's centre point or level does not match exactly one detected room. Run the rooms command again and copy its values. |
| `splits.csv changed the rooms: run the rooms command again` | A row still holds the centre point of a room that `splits.csv` now cuts. Run `rooms` again and copy each half's centre point (step 5). |
| `splits.csv line N: level_z ... matches no storey` | Give the storey's level in metres, as `rooms` prints it; the points are metres too, whatever `scene.units`. |
| `splits.csv line N: the line crosses no room` / `crosses 2 rooms` | Move or shorten the line so that it crosses exactly one room. |
| `splits.csv line N: its first point` / `last point ... inside` | Move that end point onto the room's outline or past it. |
| `splits.csv line N: the line cuts ... into 3 piece(s)` | Draw the line so that it enters and leaves the room once, or use two lines. |
| `splits.csv line N: the line cuts ... into halves of` | Move the line: each half at least 1 m2. |
| `splits.csv line N: the line meets pane` | Move the line so that it ends between two panes (step 5). |
| `splits.csv line N:` a point, a gap under 0.01 m, or `crosses or overlaps itself` | Fix that row: `x y` pairs in metres separated by `;`, at least 0.01 m apart, with no loop. |
| `is a piece of ... but its ... differs` | Give the piece row the same `floor` and ticks as its space row (step 5). |
| `the pieces of a space must be on one storey` | Clear `merge_into` for that row. |
| `performance.linked_zones ... is a piece of space` | Name the space row instead. |
| `every piece of a regularly occupied space needs at least one sensor point` | That piece is too small or narrow for a grid point. Clear its `merge_into` and untick it or mark it not regularly occupied, or check the room in the model. |
| `analysis.hours ... opens before 03:00` under daylight saving | The clock change falls inside such a window. Open it at 03:00 or later, or set `daylight_saving: none`. |
| `occupied_mask produced ... occupied hours, expected ...` | An internal check on the occupied-hour mask; it should never appear. Report it as a bug. |
| `Radiance binaries not found on search path` | Radiance is not on your `PATH` (step 0). The message lists the missing programs. |
| `unknown key '...': not a role name` | A `reflectance.<k>` key that is not `policy` or a role from `materials.csv`, or a `glazing.frame_share.<k>` key that is not `default`, `sloped`, `skylight` or such a role. The message lists the valid names. |
| `this key was renamed ... only_if_points` | Rename `only_if_points_at_least` to `option1.sda.additional_point.only_if_points` in `credit_rules.yaml`. |
| `weather.file must be a Radiance .wea file` / a `.wea` header, line-count or ordering message | Convert your `.epw` with `epw2wea` and check its output against the rules in step 4. |
| a CSV `is not UTF-8 text` / uses `';'` as the delimiter | Re-save `materials.csv` or `spaces.csv` as "CSV UTF-8 (Comma delimited)" (step 3). |
| `project.yaml has no views: section` | Add the `views:` section (step 9). |
| `views.eye_height_m` / `views.eye_height_source` | Give both. They have no default. |
| `views.angular_step_deg ... must divide 360` / `... must divide half the vertical span` | Use a step that divides both 360 and 30 exactly, such as `0.5`, `1.0`, `1.5`, `2.0`, `2.5`, `3.0` or `5.0`. |
| `no views_<case> column` | Add a `views_<case>` column to `spaces.csv` (step 9). |
| `ticked in 'views_...' but regularly_occupied is no` | Quality Views counts regularly occupied rooms only. Untick the room or correct `regularly_occupied`. |
| `marked 'clear' ... but its visible transmittance ... is not above` | LEED needs glass with a VLT above 40 %. Remove `clear` from that glass. |
| `no exterior glazing is see-through for views` | No exterior glass is above the VLT limit or marked `clear`. Check the glass materials. |
| `credit_rules.yaml has no Quality Views section` | Copy the `views:` section from the shipped rules template (step 9). |
| `no ground surface and no floor level; cannot place the ground plane` | Add a `ground`-role surface, or a floor, to the scene. |
| `Option 2 cannot run: option2....` | An `option2:` setting the tool cannot run, such as one still `UNVERIFIED`. Fix that setting (step 10). |
| `the weather file's time_zone ... differs from ... site.meridian_radiance` | Option 2 checks the weather file's own hours. Make both name the same time zone (step 10). |
| `no day within ... days of ... has any direct sun` | Option 2 needs a clear day near each equinox. Check the weather file. |
| `the sun is below the horizon` / `the diffuse irradiance is 0 W/m2` at a checked time | Option 2 cannot build that sky. Check `option2.times` and the weather file. |
| `atlas.drawings[<n>]` | A drawing of the `atlas:` section is wrong; the message says what (section 11). Fix the entry, or remove it. |
| `have no floor atlas` / `the floor atlas for case` | `package` found no current atlas (section 11). Run `report` again, then `package`. |

---

## 9. Quality Views

LEED v4.1 BD+C EQ Credit Quality Views: whether every regularly occupied point has a direct
line of sight to the outdoors, with something worth seeing beyond the glass. It runs by ray
tracing, separately from `run` (Option 1), and needs no annual simulation.

**Add a `views:` section to `project.yaml`:**

| Key | What to put |
|---|---|
| `eye_height_m` | Required. No default. A value such as `1.07` (seated eye height, 42 in, as used in reviewer-accepted drawings) is common; use whatever your project's drawings show. |
| `eye_height_source` | Required. One sentence saying where the eye height came from. Printed in every report. |
| `max_interior_glass` | Optional, default `1`. How many interior glass partitions a line of sight may cross and still count. |
| `angular_step_deg` | Optional, default `2.0`. The panorama's angular spacing in degrees; must divide 360 and half the vertical span (30 degrees) exactly, e.g. `0.5`, `1.0`, `1.5`, `2.0`, `2.5`, `3.0` or `5.0`. |

**The optional `view` column in `materials.csv`.** Add a fifth column, `view`, to tag what a
modifier is to a view ray. Leave it empty to use the role's own default. The five values:

- `clear` -- see-through. Glass must be above the credit's VLT limit to be marked `clear`: its
  computed transmittance is rounded to 4 decimal places before the comparison, so a glass whose
  true VT is exactly 0.40 stays "not above 0.40" and is not clear, even though the raw conversion
  can come back a hair over; any other modifier may also be marked `clear`, and the run warns
  about each one.
- `blocks` -- stops the ray. The default for anything that is not see-through glass.
- `nature`, `art`, `landmark` -- what a ray sees counts as a view when it is at least
  `method.min_outside_distance_m` beyond the glass (a `credit_rules.yaml` key, in its `views:` section, not a
  `project.yaml` key); a tagged object closer than that
  cut-off is skipped, like any other object that close. Allowed only on the roles `context`,
  `ground`, `fixed_shading` and `wall` -- never on glass, a floor, a ceiling or furniture.

Only facade glass can be vision glazing. Skylights, sloped roof glass, roof monitors and every diffusing pane
block view rays and set no head height, whatever their `view` value; a material marked `clear` that such a pane
uses is warned about.

**`spaces.csv`.** Add one `views_<case>` column per Quality Views case (the same file `run` uses;
`include_<case>` and `views_<case>` are independent tickboxes -- `views` no longer needs any
`include_<case>` column at all). A regularly occupied space left
out of every `views_<case>` should say why in the `views_reason` column. The `views_<case>`
columns go after the `include_<case>` columns; `views_reason` goes last, after `reason`:

```
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base,views_base,reason,views_reason
L0,Office North,office,yes,48.00,6.00,4.00,0.00,no,,1,1,,
L0,Corridor,circulation,no,20.00,6.00,9.50,0.00,no,,1,0,not regularly occupied,
L1,Meeting 1,meeting,yes,24.00,3.00,4.00,3.50,no,,1,0,,video conference room: may be excluded
```

- Only a regularly occupied room may be ticked. A tick on any other room is refused. A piece row
  (step 5, **Merge rooms**) is judged by its space row's `regularly_occupied`, not its own.
- LEED lets you leave out auditoriums, conference rooms used for video conferencing and
  gymnasiums. Untick them and say why in `views_reason`.
- A piece row has its space row's `views_<case>` ticks; the space is assessed as one, over every
  piece's points.

**`credit_rules.yaml`.** It needs a `views:` section. The shipped template
`leed_daylight/templates/credit_rules_v41_bdc_nc.yaml` has one: copy it if your rules file is
older. It holds the credit values (75 % of floor area, 1 point, VLT above 40 %, three times the
head height, objects at 7.5 m, and the 0.3 m column exception, which is recorded but not applied: the
tool only reports it). Do not edit them. Its `method:` part
holds three ClimateStudio defaults that are not LEED text (median distance, a 2.0 m minimum
outside distance, a 60 degree vertical panorama). Its `views.version` must be a quoted string
(YAML reads an unquoted date-like value as a date, not text).

**Run it:**

```
leed-daylight views ~/Desktop/office_a
```

`--case` picks a views case; it defaults to the first `views_<case>` column. `--processes N`
(default 1) runs `rtrace` with `-n N`: more processes trace the rays faster on several cores, for
identical results. At the end it prints
(invented numbers):

```
Quality Views, case base
quality view (floor area): 80.0% (points: 78.5%)
type 2: 85.0%, type 3: 95.0%
score: 1 point(s): result is 80.0 %, which reaches the 75.0 % band
  Office North: 48.0 m2, quality 80.0%
warnings: 1
  - the 30 % cap on views into interior atria is not assessed; a view through interior glass counts like any other
results written to ~/Desktop/office_a/_run/views/base/views_results.json
```

- **type 3** is the share of floor area that sees vision glass within three times its head
  height, with nothing in between.
- **type 2** is the share whose view beyond the glass is at least 7.5 m away (median), or shows
  something tagged `nature`, `art` or `landmark`.
- **quality view** is the share that passes both. The credit needs 75 %.
- Only the passing part of a room counts, point by point.

The results go to `~/Desktop/office_a/_run/views/<case>/`:

| File | What it is |
|---|---|
| `views_results.json` | Settings, per-space and project results, the score and warnings. Includes `settings.ground_z_m`, the elevation the ground plane was placed at. |
| `points.csv` | One row per point: its type 2, type 3 and quality pass/fail, and its median outside distance. |
| `plan_<storey>_<type2\|type3\|quality>.svg` | One plan per storey per type, points coloured pass or fail, with a north arrow turned by `site.north_rotation_deg`. |

The working files go to `~/Desktop/office_a/_run/views/`: `prepared/` (the model with frames),
`model/` (the views scene: every material black, no blinds), `engine/` (the octree and one saved
result per room) and `radiance.log`. Stop the run at any time and run it again: finished rooms
are reused. Change a setting and the rooms it affects are recomputed. Delete `engine/` to force a
full rerun. A cache written by an older version of the tool is also treated as stale and
recomputed automatically -- expect the first run after an upgrade to retrace every room.

**Warnings** it may print: the atrium cap is not assessed (always); a storey with ticked rooms
but no exterior vision glass; rays that stopped at the 8-segment limit; each non-glass material
marked `clear`; the regularly occupied spaces not ticked into a `views_<case>`, each with its
`views_reason` (or "(no reason given)"); no ground surface in the scene, in which case the ground
plane is taken at the lowest floor level (`z = ...`); a room whose floor is below that ground
plane; and the room-detection warnings from step 2 (implausible storey height, a sloped ceiling,
a malformed floor/ceiling), which now appear in Quality Views results too, not only Option 1's.

**How the method works**, beyond the credit text itself:

- Head height is the top of the vision glass above the finished floor of the point's storey.
- Eye height is `views.eye_height_m`, with its source given in `views.eye_height_source`.
- Objects at 7.5 m are judged by the solid-angle-weighted median outside distance over a
  360 x 60 degree panorama -- the ClimateStudio method, not LEED's own text.
- An object closer than `method.min_outside_distance_m` (`credit_rules.yaml`, `views:` section) beyond the glass is looked
  through -- a ClimateStudio default, not LEED's own text.
- A view may cross at most `views.max_interior_glass` interior glass partitions.
- Two glass faces along a ray count as one layer -- a double or triple pane -- when they are
  within 0.10 m of each other along the ray, or parallel and within 0.10 m measured
  perpendicular to the glass.

- Blinds and other movable shades are left out: LEED lets movable glare control be ignored.
  Furniture follows `analysis.moveable_furniture`.

**Not built yet:** the 30 % cap on views into interior atria, the healthcare and warehouse
variants of this credit, LEED v4 (only v4.1 is implemented), and checks of frits or patterns (mark
a fritted glass `blocks` or `clear` yourself).

---

## 10. Option 2

```
leed-daylight option2 ~/Desktop/office_a
```

It prepares `<folder>/_run/` for the chosen case, as `run` does. To run a scenario instead of the
first case: add `--case option_b`. Add `--processes N` (an integer, default 1) to run `rtrace` with
`-n N`: the same result, faster on several cores.

**How the two days are chosen.** For each equinox date, the tool adds up each day's direct normal
irradiance from the weather file, over the 15 days either side. The day with the most wins; a tie
goes to the day closer to the equinox, then the earlier day. It also works out each day's Perez sky
clearness index and warns if that index would pick a different day. The choice does not change.

**The time.** 9:00 and 15:00 are the weather file's own hours: its time zone, with no daylight-saving
shift. In a country on summer time in September, 9:00 in the file is 10:00 on the wall clock. The
report must say so. The weather file's `time_zone` header line must equal `site.meridian_radiance`,
or the run stops.

**The sky.** Radiance's `gendaylit` builds each sky from the weather file's direct normal and diffuse
horizontal irradiance at that hour: the mean of the two half-hour records either side of it. If the
sun is below the horizon, or the diffuse irradiance is zero, at a checked time, the run stops and
names the day and time.

**The result.** Each point's 9:00 value is the mean of its March and September 9:00 values; the same
at 15:00. A point passes if both are between 300 and 3,000 lux. A room marked
`view_preserving_shades: yes` has no upper limit. The score is the share of regularly occupied floor
area that passes: 55 % earns 1 point, 75 % earns 2, 90 % earns 3.

**The files**, in `~/Desktop/office_a/_run/option2/`:

| File or folder | What it is |
|---|---|
| `<case>/option2_results.json` | The settings, the chosen days and their runners-up with their Perez index, the four skies, every room, the score, the warnings. |
| `<case>/points.csv` | One row per point: lux under each sky, the two averages, pass or fail. |
| `<case>/skies/` | The four `gendaylit` skies. |
| `<case>/engine/` | Octrees and cached results. Keep it to resume; delete it to force a rerun. |
| `radiance.log` | Every Radiance command, with its exit status and time. |

## 11. Reports and the submission folder

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

The Option 1 plans draw facade glass as solid lines and roof glass as dashed outlines, each with its group's
name. The window-group table gives each group's type and tilt; a skylight or diffusing group has no facade
azimuth ("–"), and diffusing glass has no blind. The method section prints the readings of LM-83-12 and the
LEED Reference Guide the tool applies to the roof or diffusing glass in the scene. The glare narrative names
diffusing glass in a space over the trigger as needing no glare-control device.

Every plan, in the reports and the atlas, draws each piece of a merged space, with the space's name
once, on its largest piece, and draws split lines as dashed lines. The method section adds one sentence
per split or merged space, for example "Office 1 is two detected rooms treated as one space; each keeps
its own grid, and blinds respond to the whole space." Only the Option 1 report can add the blind clause,
and only for a space with a window group that can carry a blind; the Option 2 and Quality Views reports
never add it, since those options use no blinds.

The optional `report:` section of `project.yaml` sets the figure renderer (`matplotlib`, the
default, `cairosvg` or `pillow`), your list of `deviations` (printed in every report) and
`prepared_by`. Editing the `report:` section never makes results out of date, so you can add a
deviation or `prepared_by` and run `report` again without rerunning the options.

**See the model and the results: the floor atlas.**

    leed-daylight atlas "<project folder>" [--case <name>] [--no-blender]

It writes `<project folder>/_run/atlas/<case>/`, emptied first:

- `atlas.html`: one page to open in any browser, with no network (no font, script or image is
  fetched). One tab per storey: the plan, a layer switch above it (`Model`, or one result layer such as
  daylight autonomy, ASE hours, Option 2 illuminance per sky or the Quality View), and the room table
  (area, points, included or not, window groups and, with results, each option's per-room percentage
  from that option's results file: Option 1 sDA % and ASE %, Option 2 the percentage in band, Quality
  Views the percentage with a quality view; a merged space shows the sum of its rows' `area_m2`), and,
  under the table, the sentence of each split or merged space on that storey (with the blind clause
  only for a space the page shows Option 1 results of, and, for a merged space with no grid in this
  case, neither the blind nor the grid clause). One 3D tab: a switch for each kind of surface (walls,
  floors, each kind of glass, frames, blinds, context and so on), storey buttons, and the grid coloured
  by the chosen layer; a click on a point shows its room, storey, position and value. The header gives
  the project, the case, the date and, with results, each option's project numbers. An option with no
  current result is listed at the top with the reason, for example out of date after a model change.
- `bundle.npz`: the data the page and the Blender file are both built from.
- `model.blend`, with `blender.log` beside it: only when Blender 4.2 or newer is found (section 0).
  Its collections are `MODEL`, `ADDED` (frames, reveals, skylight wells, blinds), `CONTEXT` (hidden),
  `ROOMS` and `RESULTS`, with one view layer per storey and one for the whole building.

Before any run, `atlas` is the model check: it prints the same case summary as `prepare` and the page
shows the prepared model, the rooms and the grid. After a run it shows every current result as a layer.

`report` builds the same atlas after the Word files, always with every current option, so
`report --option views` still keeps every result layer. `--no-blender` works for both commands and
leaves Blender out. Blender outcomes:

| What happened | `atlas` | `report` |
|---|---|---|
| No Blender found, or one older than 4.2 | One line, no `model.blend`, exit 0 | Same |
| Blender ran and failed or took over 10 minutes | Page kept, the path of `blender.log` printed, exit 1 | A warning, exit 0 |
| `--no-blender` | Blender is not looked for | Same |

When `BLENDER` is set it must name the Blender program; nothing else is tried then. To use another
Blender than the one on your `PATH`:

```
export BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
```

In PowerShell (Windows): `$env:BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"`. On Linux, for example, `export BLENDER=/opt/blender-4.2/blender`.

Any other atlas failure in `report` (a wrong drawing, say) is an `error:` line and exit 1; the Word
reports are kept, and `package` refuses until the atlas is built.

**A drawing under a storey plan.** The optional `atlas:` section of `project.yaml` puts a floor plan
drawing under a storey's plan in the page:

```yaml
atlas:
  drawings:
    - storey: "L1"
      image: "drawings/level1.png"
      points:
        - {pixel: [400, 1800], model: [0.0, 0.0]}
        - {pixel: [3900, 1800], model: [35.0, 0.0]}
      opacity: 0.35        # optional, default 0.35
```

- `storey` is a `floor` value of `spaces.csv`; one drawing per storey.
- `image` is a PNG or JPG file, as a path relative to the project folder (no absolute path, no `..`, no
  link leading out). The tool reads the file's content, not its name, so a PDF renamed `.png` is
  refused: export the drawing as PNG first.
- The two points pin the drawing: for each, a pixel of the image (x to the right, y down, as an image
  editor shows it) and the same point in the model in metres (y up). Pick two points far apart, such
  as two grid-line crossings. The tool scales, turns and moves the drawing to match; it never
  stretches it. The points must be at least 10 pixels and 0.5 m apart.
- `opacity` is between 0 and 1.

Each wrong entry is refused by name, `atlas.drawings[<n>]`, with the reason. Editing `atlas:` never
makes results out of date: like `report:`, it is left out of the `project.yaml` fingerprint. The
drawing shows in the 2D plans only, not in 3D or in Blender.

**Make the USGBC calculator.**

    leed-daylight calculator "<project folder>" [--case <name>] --template <blank calculator .xlsm>

It fills the USGBC "LEED v4.1 Daylight and Quality Views Calculator" (V02) from the case's current
results and writes `<project folder>/_run/calculator/<case>/LEED_v4.1_Daylight_QualityViews_<case>.xlsm`
and `calculator.json`. The blank calculator is USGBC's and is not included; any other file is refused.
It needs `calculator.units` (`IP` or `SI`) in `project.yaml` and a `vlt` column in `materials.csv`.
The Daylight areas it writes are the rooms' areas from the scene, as every other output uses them
(`spaces.csv` only for a space no option holds). A `vlt` from 39.9 up to and including 40 is written as 39.9, because the
tool passes glass above 40 % and the calculator passes it at 40 % or more. A `vlt` above 0 and at most 1
(0.62, say) draws a warning, in `calculator` and in `views`: VLT is a percentage, so it is probably meant as 62.

**Make the submission folder.**

    leed-daylight package "<project folder>" [--case <name>]

It writes `<project folder>/submission/<case>/`: `reports/`, `results/`, `model/` (the model
prepared again for this case), `config/` (`project.yaml`, `materials.csv`, `spaces.csv`, the credit
rules and, when the project has one, `config/splits.csv`; the package's `README.md` then names it in
its reproduce steps), `logs/`, `atlas/` (the files the report step's atlas wrote:
`atlas.html`, `bundle.npz` and, when Blender built it, `model.blend`; never `blender.log`, which holds
paths of your machine), `README.md`, `provenance.json` and `checksums.sha256`, and rewrites
`submission/README.md`, an index of every case. It refuses reports that are missing or older than
their results. Check a package with:

    cd "<project folder>/submission/<case>" && shasum -a 256 -c checksums.sha256

Order of work: run the options you need (`run`, `option2`, `views`), then `report`, then read
the reports and fill `glare_narrative.md`, then `report` again, then `package`. If you change
any input file (`project.yaml`, `materials.csv`, `spaces.csv`, `splits.csv`, the credit rules,
the weather file or a scene file; adding or removing `splits.csv` counts too), rerun the options
first: the report and package steps refuse results
computed from older files, and name the file that changed. `project.yaml` is compared by its
content without the `report:` section: a change to `report:`, or to comments, spacing or key order
only, does not count as a change to the results. The reports and the atlas are made from the `report:` and
`atlas:` sections and from each drawing the atlas names, so `package` also refuses when those changed after
`report` ran (or when `report` was an older version that did not record them): run `report` again.
For the same reason the `project.yaml` entry under `inputs` in
`provenance.json` is that content hash, not the file's sha256 in `checksums.sha256`.

The `report` command prints every file it writes (each report `.docx` and each plot `.png`),
each skipped option by name and reason, every warning, and the count of `[ADD]`/`[CONFIRM]` marks
still open in `glare_narrative.md`. The `package` command also refuses a stale report: an
option's results file or `points.csv` changed after the report was made, or (Option 1)
`glare_narrative.md` was edited after the report, or was last reported for a different case;
rerun `report` first. `logs/` holds the most recent run log for each option; `_run/` keeps only
the last run, so a log here may be from a case other than this package.

`package` also refuses an atlas that is missing or out of date: none built by `report`, one made before the
`report:` or `atlas:` section or a drawing last changed, one built
from other input files, one that shows other options than the package holds, or one older than an
option's results file or `points.csv`. Run `report` again first. **Reports made before the floor atlas
was added have no atlas**, so an existing project's first `package` after this version asks for
`report` once more.

**The renderer.** `matplotlib` (the default) and `pillow` need no system library. `cairosvg`
needs the system cairo library; on macOS, a Homebrew `cairo` that is not on the default library
search path makes the `cairosvg` renderer refuse by name (choose `matplotlib` or `pillow`
instead -- do not set `DYLD_LIBRARY_PATH` or similar to work around it).

## 12. What the tool does not do yet

- **The floor atlas** shows drawings in the 2D plans only (not in 3D or Blender), reads no PDF drawing
  (export it as PNG), does not cut surfaces taller than one storey, renders no images in Blender, and
  cannot edit rooms. No automatic test opens the page in a browser: check it by eye once.
- **Splitting and merging rooms** is done in files (`splits.csv`, `merge_into`). Split lines are straight segments (bends, no curves); rooms on
  different storeys cannot be merged; a merged space keeps its pieces' own outlines, with the walls
  between them; room detection itself is unchanged.
- **Roof glass**: the tool does not build splayed skylight wells (model them and set
  `skylight_well: "modelled"`), takes no BSDF data for skylights, and does not apply the LM-83-12 2.2.6
  exemption that leaves the blind off clear glass. It also does not check that a skylight's ceiling opening
  matches its glass, and gives no extra dirt warning on vertical glass (only sloped and skylight glass are,
  step 4). Closing "view" window groups before "clerestory" groups (LM-83-12 2.2.6's discussion) is not
  done either: the existing most-lit-first closing order stays. The Quality Views atrium cap is also not
  assessed (step 9).
- **Accuracy** against a delivered LEED job has not been confirmed yet. That is the acceptance test,
  run separately with a private fixture (`LEED_ACCEPTANCE_FIXTURE`) whose `expected.json` holds one
  entry per run; `LEED_ACCEPTANCE_RUN` picks the run (`tests/acceptance/conftest.py` describes the
  folder). Quality Views has not been compared with a ClimateStudio run yet.
- **Speed on large models**: `rtrace` runs on one processor core for Quality Views by default; pass
  `--processes N` (step 9) to use several. A small building takes seconds; a large one has not been
  timed yet.
