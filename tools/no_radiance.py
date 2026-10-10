# tools/no_radiance.py -- run a command with every Radiance folder removed from PATH.
"""Radiance may be installed on the build machine and no test may start it
(spec section 13, N5); the pipeline's skipif rules then skip the Radiance tests.

    python tools/no_radiance.py .venv/bin/python tools/check.py -q
"""
import os
import subprocess
import sys

SMOKE = ("LEED_RADIANCE_SMOKE", "LEED_SKYLIGHT_SMOKE", "LEED_BLENDER_SMOKE")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python tools/no_radiance.py <command> [args...]", file=sys.stderr)
        return 2
    env = {k: v for k, v in os.environ.items() if k not in SMOKE}
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                  if p and "radiance" not in p.lower())
    return subprocess.call(argv, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
