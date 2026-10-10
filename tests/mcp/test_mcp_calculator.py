"""deliver(action="calculator") and the status line (calculator spec 4, 5.1)."""
import pytest

from calculator_standin import add_vlt as _vlt, set_units as _units, write_standin
from conftest import symlink_or_skip
from job_fakes import FakeLauncher
from leed_daylight.deliver.data import result_path
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    """As tests/mcp/test_mcp_deliver.py's folder, with Option 1 and Views results files."""
    folder = build_option2_project(tmp_path, write_example, write_wea)
    for credit in ("option1", "views"):
        path = result_path(folder, credit, "base_case")
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    return folder


async def _calculator(tmp_path, folder, launcher=None):
    server = tool_server(tmp_path, launcher=launcher or FakeLauncher())
    return await call(server, "deliver", {"action": "calculator", "folder": str(folder)})


async def test_an_unset_template_is_not_configured(tmp_path, folder, monkeypatch):
    monkeypatch.delenv("LEED_MCP_CALCULATOR", raising=False)
    assert codes(await _calculator(tmp_path, folder)) == [("CALCULATOR_NOT_CONFIGURED", None)]


async def test_another_file_is_unsupported(tmp_path, folder, monkeypatch):
    other = tmp_path / "x.xlsm"
    other.write_bytes(b"not a workbook")
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(other))
    assert codes(await _calculator(tmp_path, folder))[0][0] == "CALCULATOR_UNSUPPORTED"


async def test_units_are_asked_for(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(write_standin(tmp_path / "blank.xlsm")))
    assert codes(await _calculator(tmp_path, folder)) == [("FILL_ME_LEFT", "calculator.units")]


async def test_vlt_is_asked_for_when_views_ran(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(write_standin(tmp_path / "blank.xlsm")))
    _units(folder)
    refused = await _calculator(tmp_path, folder)
    assert codes(refused) == [("INPUT_INVALID", "file")]
    assert "glass_clear" in refused.structured_content["reason"]


async def test_a_ready_folder_starts_the_job_with_the_template(tmp_path, folder, monkeypatch):
    template = write_standin(tmp_path / "blank.xlsm")
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(template))
    _units(folder)
    _vlt(folder)
    launcher = FakeLauncher()
    result = await _calculator(tmp_path, folder, launcher)
    assert result.is_error is False and result.structured_content["state"] == "started"
    assert launcher.calls[0]["argv"][-5:] == ["calculator", str(folder), "--case=base_case",
                                              "--template", str(template)]


async def test_status_names_the_calculator(tmp_path, folder, monkeypatch):
    server = tool_server(tmp_path, launcher=FakeLauncher())
    monkeypatch.delenv("LEED_MCP_CALCULATOR", raising=False)
    status = await call(server, "project", {"action": "status", "folder": str(folder)})
    assert status.structured_content["calculator"] == "not configured"
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(write_standin(tmp_path / "blank.xlsm")))
    status = await call(server, "project", {"action": "status", "folder": str(folder)})
    assert status.structured_content["calculator"] == "V02"


def _corrupt(path):
    """A stand-in whose xl/workbook.xml is not UTF-8."""
    import zipfile
    source = write_standin(path.with_name("good.xlsm"))
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            data = b"\xff\xfe\x00bad" if info.filename == "xl/workbook.xml" else zin.read(info.filename)
            zout.writestr(info, data)
    return path


async def test_a_corrupt_template_is_unsupported_in_deliver_and_status(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(_corrupt(tmp_path / "bad.xlsm")))
    assert codes(await _calculator(tmp_path, folder))[0][0] == "CALCULATOR_UNSUPPORTED"
    status = await call(tool_server(tmp_path, launcher=FakeLauncher()), "project",
                        {"action": "status", "folder": str(folder)})
    assert status.structured_content["calculator"].startswith("unsupported: bad.xlsm")


async def test_status_names_an_unsupported_file(tmp_path, folder, monkeypatch):
    other = tmp_path / "x.xlsm"
    other.write_bytes(b"not a workbook")
    monkeypatch.setenv("LEED_MCP_CALCULATOR", str(other))
    status = await call(tool_server(tmp_path, launcher=FakeLauncher()), "project",
                        {"action": "status", "folder": str(folder)})
    assert status.structured_content["calculator"].startswith("unsupported: x.xlsm")


@pytest.mark.parametrize("kind", ["relative", "symlink", "directory", "missing"])
async def test_a_bad_template_path_is_not_configured(tmp_path, folder, monkeypatch, kind):
    real = write_standin(tmp_path / "blank.xlsm")
    value = {"relative": "blank.xlsm", "symlink": None, "directory": str(tmp_path), "missing": str(tmp_path / "no.xlsm")}[kind]
    if kind == "symlink":
        link = tmp_path / "link.xlsm"
        symlink_or_skip(link, real)
        value = str(link)
    monkeypatch.setenv("LEED_MCP_CALCULATOR", value)
    assert codes(await _calculator(tmp_path, folder)) == [("CALCULATOR_NOT_CONFIGURED", None)]
