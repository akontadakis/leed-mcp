# tests/mcp/test_mcp_errors.py
"""Mapping pipeline errors to codes (spec 9) and the worker-thread boundary (spec 5.1)."""
import threading

import pytest
import yaml
from shapely.errors import ShapelyError

from leed_daylight.drivers.scan import ScanError
from leed_daylight.edit.files import ChangedOutside, PathOutside
from leed_daylight.engine.jobs import JobError, LockHeld, UnsafePath
from leed_daylight.engine.radiance import RadianceError
from leed_daylight.project.config import ProjectError
from leed_daylight.scene.input_file import InputFileError
from leed_mcp.envelope import refusal
from leed_mcp.errors import CHANGED_ON_DISK, Refused, in_worker, one_line, refusal_for
from leed_mcp.paths import PathOutsideRoots, RootsNotConfigured
from mcp_helpers import anyio_backend  # noqa: F401 - the fixture


@pytest.mark.parametrize("exc, code", [
    (RootsNotConfigured("none"), "ROOTS_NOT_CONFIGURED"),
    (PathOutsideRoots("out"), "PATH_OUTSIDE_ROOTS"),
    (LockHeld("run is already working on this folder"), "LOCK_HELD"),
    (PathOutside("outside"), "PATH_OUTSIDE_ROOTS"),
    (UnsafePath("a link"), "PATH_UNSAFE"),
    (InputFileError("too large"), "PATH_UNSAFE"),
    (JobError("busy"), "JOB_RUNNING"),
    (RadianceError("rtrace failed", argv=["rtrace"]), "RADIANCE_FAILED"),
    (ProjectError("project.yaml: bad"), "INPUT_INVALID"),
    (ScanError("bad scene"), "INPUT_INVALID"),
    (ValueError("bad"), "INPUT_INVALID"),
    (yaml.YAMLError("bad yaml"), "INPUT_INVALID"),
    (ShapelyError("bad shape"), "INPUT_INVALID"),
    (FileNotFoundError("gone"), "IO_ERROR"),
    (PermissionError("denied"), "IO_ERROR"),
], ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else v)
def test_each_pipeline_class_maps_to_its_code(exc, code):
    result = refusal_for(exc)
    assert result.is_error is True
    assert result.structured_content["advisories"][0]["code"] == code
    assert result.structured_content["reason"] == one_line(exc)


def test_changed_outside_carries_this_servers_reason_not_the_pipelines():
    result = refusal_for(ChangedOutside("notes.csv was changed outside the wizard; reload the step"))
    assert result.structured_content["advisories"][0]["code"] == "INPUT_CHANGED_ON_DISK"
    assert result.structured_content["reason"] == CHANGED_ON_DISK
    assert "wizard" not in CHANGED_ON_DISK


def test_an_unexpected_class_is_left_to_the_server():
    assert refusal_for(RuntimeError("x")) is None
    assert refusal_for(KeyError("x")) is None


def test_one_line_joins_lines_and_falls_back_to_the_type():
    assert one_line(ValueError("a\n  b\tc")) == "a b c"
    assert one_line(ValueError("")) == "ValueError"


@pytest.mark.anyio
async def test_the_body_runs_off_the_event_loop_thread():
    loop_thread = threading.get_ident()
    seen = []
    await in_worker(lambda: seen.append(threading.get_ident()) or refusal("X", "y"))
    assert seen and seen[0] != loop_thread


@pytest.mark.anyio
async def test_refused_and_mapped_errors_answer_and_others_propagate():
    def refused():
        raise Refused(refusal("FILE_MISSING", "no file"))

    def mapped():
        raise ProjectError("project.yaml: bad")

    def unexpected():
        raise RuntimeError("boom")

    assert (await in_worker(refused)).structured_content["advisories"][0]["code"] == "FILE_MISSING"
    assert (await in_worker(mapped)).structured_content["advisories"][0]["code"] == "INPUT_INVALID"
    with pytest.raises(RuntimeError):
        await in_worker(unexpected)


def test_a_file_in_use_refused_before_the_start_is_its_own_code():
    from leed_daylight.engine.in_use import FileInUse
    from leed_mcp.errors import refusal_for

    result = refusal_for(FileInUse("close calc.xlsm in Excel or Word, then try again"))
    assert result.structured_content["advisories"][0]["code"] == "FILE_IN_USE"
