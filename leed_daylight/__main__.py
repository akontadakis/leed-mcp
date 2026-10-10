# leed_daylight/__main__.py
"""``python -m leed_daylight <verb>``: the same command as ``leed-daylight``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
