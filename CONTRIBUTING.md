# Contributing

Thank you for helping. Issues and pull requests are welcome.

## Before you start

- Open an issue first for a large change, so we can agree on the approach.
- Never commit client data: no real project names, addresses, coordinates, weather files or product values.
  Example values in this repository are invented and round. `tests/test_generic_denylist.py` refuses an
  unsourced number written to four or more decimal places.

## Set up

    uv sync --extra dev --no-editable --reinstall-package leed-mcp

## Test

    .venv/bin/python tools/check.py

The run must end with `CHECK PASSED`. On a computer with Radiance, run it through `python tools/no_radiance.py` so no
Radiance program starts:

    python tools/no_radiance.py .venv/bin/python tools/check.py -q

Add a test with every fix and every feature. A pull request with a failing check is not merged.

## Style

- Match the code around your change: its naming, comments and error messages.
- An error message says what is wrong and what to do, in plain words, and names the setting or file.
- The server never invents a project value. A new input is either the user's value or a suggestion the user
  confirms.
