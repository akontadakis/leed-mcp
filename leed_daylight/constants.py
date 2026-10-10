# leed_daylight/constants.py
"""Layer 0: shared sentinels and version string.  Imports nothing from
inside the package."""
from __future__ import annotations

UNVERIFIED: str = "UNVERIFIED"

#: The tool's own version. Defined once here (M3) and re-exported as
#: ``leed_daylight.__version__`` so a reader (or a layering test) never
#: needs a row for the package root itself; ``build.prepare`` imports this
#: name directly for the manifest.
TOOL_VERSION: str = "0.1.0"

#: The marker ``leed-daylight init`` writes for every value a person must
#: fill in. ``project.config.load_project`` refuses any value still equal
#: to it (CLI spec 5), so a template value can never reach a report.
FILL_ME: str = "FILL_ME"

#: The environment variable that names a job's end record (wizard spec
#: 8.1). The wizard sets it for every job it starts; ``cli.main`` then
#: writes the verb's exit code and end time to that path however the verb
#: ends. Layer 0, so both ``cli`` and ``engine.jobs`` read the one name.
JOB_RECORD_ENV: str = "LEED_JOB_RECORD"
