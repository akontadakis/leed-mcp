# tests/stub_program.py
"""A tiny program for tests that start one, on every system (spec 2026-10-09 A2)."""
import pathlib
import sys


def make_program(folder: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """A program ``name`` in ``folder`` that runs the Python ``body``; the path to start it."""
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        script = folder / f"{name}_stub.py"
        script.write_text(body, encoding="utf-8", newline="\n")
        program = folder / f"{name}.bat"
        program.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8", newline="")
        return program
    program = folder / name
    program.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8", newline="\n")
    program.chmod(0o755)
    return program
