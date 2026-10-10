"""tools/no_radiance.py strips Radiance from PATH on every system (spec 2026-10-09 A1)."""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_helper_strips_path_entries_and_smoke_vars(tmp_path):
    env = {**os.environ, "PATH": os.pathsep.join([str(tmp_path / "Radiance" / "bin"), str(tmp_path / "ok")]),
           "LEED_RADIANCE_SMOKE": "1"}
    code = "import os; print(os.environ['PATH']); print(os.environ.get('LEED_RADIANCE_SMOKE', 'unset'))"
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "no_radiance.py"), sys.executable, "-c", code],
                         env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [str(tmp_path / "ok"), "unset"]
