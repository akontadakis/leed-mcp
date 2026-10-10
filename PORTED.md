# Ported assets

Assets ported from the author's earlier work, stripped of every client string, logo,
coordinate and sample number before entering this repository.

| Asset | Ported | Notes |
|---|---|---|
| Six-stop blue-to-red colour ramp | `leed_daylight/deliver/figures/spec.py` (`_STOPS`, `ramp`) | Six RGB stops and a linear blend. No client string or number. |
| Word layout kit (page, tables, key/value table, bullets, caption, note) | `leed_daylight/deliver/word/kit.py` | Layout only; reads no results. Names changed, no client string, footer text supplied by the caller. |
| Markdown-subset to Word renderer | `leed_daylight/deliver/narrative.py` (`_runs`, `_blocks`, `_table`, `render_docx`) | Rendering rules only. The narrative template text is new; no project name, product, drawing or result from the earlier work. |
| Checksum list and provenance pattern (`<sha256>  <path>`, sorted, the list itself excluded; build under a temporary name, then rename) | `leed_daylight/deliver/package.py` | The pattern only; the README text, provenance keys and folder layout are new. |
| 3D viewer (WebGL 1 set-up, shaders with storey solo and highlight, orbit, zoom and pan, click picking, int16 positions with one offset and scale, the WebGL-missing sentence) | `leed_daylight/deliver/atlas/web/viewer.js`, `leed_daylight/deliver/atlas/web/viewer.css` | Code and layout only. Every project value removed: no project name, no storey codes, default storey or storey names, no work-plane height or grid spacing, no hour count or fixed scale ticks, no colour ramp, no web fonts. Colours, legends, storeys, the grid spacing and the work-plane height come from the page's data; the viewer reads only `window.__ATLAS__` and `#atlas-3d`. The earlier page itself is not ported. |
| Ear-clipping triangulation of a planar polygon, and the byte buffer with 4-byte aligned parts | `leed_daylight/deliver/atlas/page.py` (`_earclip`, `_triangulate`, `_Blob`) | Algorithm only; no geometry, name or number from the earlier work. |
| Blender builder structure (per-storey objects and sub-collections, a per-corner colour attribute on each result tile, near-transparent glass materials, viewport colour shown through the material and through a vertex colour) | `leed_daylight/deliver/atlas/blender_build.py` | Structure and code only. Stripped of the project name, the fixed storey code list and storey names, the fixed slab heights, the work-plane height and grid spacing, the hard-coded LEED point thresholds and the fixed modifier names (replaced by the geometry kinds); every value in the file is invented and round. |

## Copied into leed-mcp

The pipeline (`leed_daylight/`) was copied into this repository from its standalone repository, which is not
public. What changed in the copy, and why:

| Path here | Source | Change |
|---|---|---|
| `leed_daylight/` | `leed_daylight/` | Copied. The browser wizard is left out. |
| `leed_daylight/edit/files.py` | the wizard's `files.py` | Moved; imports made two-dot; backups go to `_run/backup/`. |
| `leed_daylight/edit/project_yaml.py` | the wizard's `project_yaml.py` | Moved; imports `..drivers.init`. |
| `leed_daylight/cli.py` | same | `VERBS` loses `wizard` and gains `calculator`. |
| `leed_daylight/engine/jobs.py` | same | `VERBS` gains `prepare` and `calculator`; two messages no longer name the wizard. |
| `leed_daylight/project/config.py` | same | One message no longer names the wizard. The `radiance:` option strings accept only numeric Radiance options, so a value cannot name a file or a command. The optional `calculator:` section (`units`: `IP` or `SI`, no default) is read into `Project.calculator`. |
| `leed_daylight/project/materials.py` | same | The optional `vlt` column (visible light transmittance in %, 0 to 100), after `view`, allowed only on `glazing` and `glazing_curtain_wall` rows; no calculation reads it. |
| `leed_daylight/project/inputs.py` | same | The run fingerprint leaves out the `calculator:` section of project.yaml, and materials.csv is digested without its `vlt` column (`materials_digest`), so neither marks results out of date. |
| `leed_daylight/drivers/_folder.py` | same | `REPO_ROOT` finds the repository also from a `--no-editable` install. |
| `leed_daylight/engine/radiance.py` | same | A child run without input gets `stdin=DEVNULL` (`_stdin_kwargs`). |
| `docs/PIPELINE.md` | `README.md` | Renamed; wizard text removed. |
| `docs/RUNNING_A_PROJECT.md` | same | Wizard section and mentions removed. |
| `tests/` | `tests/` | Wizard tests dropped. |

## Ported into leed_mcp

Logic ported, not copied, from the browser wizard and the CLI drivers, and the underscore-private pipeline
names `leed_mcp` imports on purpose.

| Here | Source | Note |
|---|---|---|
| `leed_mcp/adapters/scene.py` `scan` | the wizard's scene step (`_table`, `_problems`); storeys as `leed_daylight/drivers/scan.py` `_print_storeys` prints them | Reads `load_scene_settings`, never `load_project`. |
| `leed_mcp/adapters/scene.py` `_detect` | the wizard's rooms step (`_detect`, `detected_rooms`) | |
| `leed_mcp/adapters/scene.py` `rooms`, `rings`, `_opening` | the wizard's rooms step (`plan`, `glazed_rooms`) | Panes through `build.groups.window_groups` with every detected room named. |
| `leed_mcp/adapters/scene.py` `labelled`, `_write_spaces_template` | `leed_daylight/drivers/rooms.py` `_labelled_rooms`, `_write_template` | Same bytes as the CLI. |
| `leed_mcp/adapters/scene.py` `_note`, `materials_template` | `leed_daylight/drivers/materials.py` `_note`, `materials_folder` | Same bytes as the CLI. |
| `leed_mcp/adapters/project_state.py` `fill_me_keys` | `leed_daylight/project/config.py` `_fill_me_paths` | |
| `leed_mcp/adapters/materials_file.py` `table_bytes` | the wizard's scene step (`table_bytes`) | The `view` column only when a row has a view. |
| `leed_mcp/adapters/spaces_file.py` `table_bytes`, `matches`, `_centroid`, `_float` | the wizard's rooms step (same names, `_matches`) | `matches` takes the header and rows, not the wizard's `SpacesTable`. |
| `leed_mcp/adapters/splits_file.py` `splits_bytes`, `_number` | the wizard's rooms step (same names) | |
| `leed_mcp/adapters/narrative_file.py` `MARK`, `count_marks`, `split_row`, `is_space_row`, `measures`, `block_state`, `body_span` | `leed_daylight/deliver/narrative.py` `_MARK`, `count_marks`, `_PIPE`, `_split_row`, `_measures`, `_block_state`, `_body_span` | Ported, not imported: `deliver/narrative.py` imports `python-docx`, which the optional `report` extra installs. `test_the_ported_narrative_rules_equal_the_pipelines` holds them equal. |

Underscore-private pipeline names `leed_mcp` imports, each pinned by a test in `tests/mcp/`:

- `leed_daylight.scene.reader._tokens_of` and `_parse_xform`, in `leed_mcp/adapters/scene.py`: the
  `!xform` containment check reads the targets exactly as `read_scene` will
  (`test_an_xform_target_outside_the_folder_is_refused_before_reading`).
- `leed_daylight.project.inputs._project_yaml_digest`, in `leed_mcp/adapters/results_files.py`:
  project.yaml is compared on its own when another input cannot be read
  (`test_an_input_that_cannot_be_read_counts_as_changed`).
