# tests/test_confidentiality.py
"""Spec section 12 rule 6: fail the build if a client string reaches the repository.

The deny list itself is confidential, so it lives in the private fixture folder
and is never committed.  With no fixture configured the test skips and says so.

"The repository" is exactly what git would publish: every git-tracked file
(``git ls-files``) and every commit message on every ref
(``git log --all --format=%B``). Untracked working-tree content -- a
virtual environment, caches, ``.git`` itself, gitignored ledgers such as
``.superpowers/`` -- is never scanned: it is not in the repository, and
walking it would fail on material that is meant to stay local.
"""
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        pytest.fail(
            f"git {' '.join(args)} failed ({result.stderr.strip()}); the "
            "confidentiality scan needs a git checkout of the repository"
        )
    return result.stdout


def repo_text_files():
    """Every git-tracked file that exists on disk, as an absolute path."""
    for name in _git("ls-files", "-z").split("\0"):
        if name:
            path = REPO / name
            if path.is_file():
                yield path


def commit_messages() -> str:
    """Every commit message reachable from any ref."""
    return _git("log", "--all", "--format=%B")


def test_denylist_file_is_not_committed():
    assert not (REPO / "denylist.txt").exists(), (
        "denylist.txt must live in the private fixture folder, not the repository"
    )


def test_the_scan_covers_exactly_the_tracked_files():
    tracked = {line for line in _git("ls-files").splitlines() if line}
    scanned = {path.relative_to(REPO).as_posix() for path in repo_text_files()}
    assert scanned == {name for name in tracked if (REPO / name).is_file()}
    assert "tests/test_confidentiality.py" in scanned
    assert not any(
        part in {".git", ".venv", ".pytest_cache", ".superpowers", "__pycache__"}
        for name in scanned for part in pathlib.PurePosixPath(name).parts
    )


def test_the_scan_includes_commit_messages():
    # This file's own history always has at least one message.
    assert commit_messages().strip()


def test_no_client_string_in_repository():
    fixture = os.environ.get("LEED_ACCEPTANCE_FIXTURE")
    if not fixture:
        pytest.skip("LEED_ACCEPTANCE_FIXTURE not set; confidentiality scan skipped")
    denylist = pathlib.Path(fixture) / "denylist.txt"
    if not denylist.exists():
        pytest.skip(f"no denylist.txt in {fixture}; confidentiality scan skipped")
    terms = [
        line.strip().lower()
        for line in denylist.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert terms, "denylist.txt is empty; the scan would pass vacuously"
    hits = []
    for path in repo_text_files():
        lowered = path.read_bytes().decode("utf-8", errors="ignore").lower()
        hits += [
            f"{path.relative_to(REPO)}: {term}" for term in terms if term in lowered
        ]
    messages = commit_messages().lower()
    hits += [f"commit messages: {term}" for term in terms if term in messages]
    assert not hits, "client strings found in the repository:\n" + "\n".join(hits)
