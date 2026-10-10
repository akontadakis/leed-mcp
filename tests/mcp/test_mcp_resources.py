# tests/mcp/test_mcp_resources.py
"""The three static resources (spec 7), on both protocol eras."""
import re

import pytest
from mcp.client import Client
from mcp.shared.exceptions import MCPError

from leed_mcp.server import build_server, register_tools
from leed_mcp.static_resources import GUIDE, TEMPLATES, register_resources
from mcp_helpers import anyio_backend, call, wire  # noqa: F401 - anyio_backend is the fixture

pytestmark = pytest.mark.anyio

URIS = ["leed://docs/running-a-project", "leed://templates/project", "leed://templates/credit-rules"]
FILES = {URIS[0]: GUIDE, URIS[1]: TEMPLATES / "project_template.yaml",
         URIS[2]: TEMPLATES / "credit_rules_v41_bdc_nc.yaml"}


def _server(*roots):
    server = build_server(roots=roots)
    register_tools(server)
    register_resources(server)
    return server


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_three_resources_are_listed_in_the_spec_order(mode):
    async with Client(_server(), mode=mode) as client:
        listed = (await client.list_resources()).resources
    assert [(r.uri, r.name, r.mime_type) for r in listed] == [
        (URIS[0], "running-a-project", "text/markdown"),
        (URIS[1], "project-template", "application/yaml"),
        (URIS[2], "credit-rules-template", "application/yaml"),
    ]
    assert all(r.title and r.description for r in listed)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize("uri", URIS)
async def test_each_resource_reads_as_text_from_its_file(mode, uri):
    async with Client(_server(), mode=mode) as client:
        (content,) = (await client.read_resource(uri)).contents
    assert content.uri == uri and content.text == FILES[uri].read_text(encoding="utf-8")


async def test_a_modern_read_carries_the_sdk_cache_defaults():
    async with wire(_server()) as send:
        read = await send("resources/read", {"uri": URIS[1]})
        listed = await send("resources/list")
    for answer in (read, listed):
        assert (answer["ttlMs"], answer["cacheScope"]) == (0, "private")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("uri", ["leed://docs/nothing", URIS[0] + "/", "LEED://templates/project",
                                 "file:///etc/hosts", "leed://templates/../templates/project"])
async def test_an_unknown_uri_is_invalid_params(legacy, uri):
    async with wire(_server(), legacy=legacy) as send:
        with pytest.raises(MCPError) as raised:
            await send("resources/read", {"uri": uri})
    assert raised.value.error.code == -32602


async def test_the_project_template_is_what_project_create_writes(tmp_path):
    server = _server(tmp_path.resolve())
    await call(server, "project", {"action": "create", "folder": str(tmp_path / "p")})
    async with Client(server) as client:
        for uri, written in ((URIS[1], "project.yaml"), (URIS[2], "credit_rules.yaml")):
            (content,) = (await client.read_resource(uri)).contents
            assert content.text == (tmp_path / "p" / written).read_text(encoding="utf-8")


async def test_no_resource_template_is_served():
    async with Client(_server()) as client:
        assert (await client.list_resource_templates()).resource_templates == []


def test_the_guide_speaks_in_tool_calls_not_commands():
    text = GUIDE.read_text(encoding="utf-8")
    assert text.startswith("# ")
    for command in ("leed-daylight ", "pip ", "python -m", ".venv/bin"):
        assert command not in text, command


async def test_every_tool_call_in_the_guide_names_a_real_action():
    async with Client(_server()) as client:
        tools = {t.name: set(t.input_schema["properties"]["action"]["enum"]) for t in (await client.list_tools()).tools}
    calls = re.findall(r"\b(\w+)\(action=\"(\w+)\"", GUIDE.read_text(encoding="utf-8"))
    assert len(calls) > 30
    assert [c for c in calls if c[1] not in tools.get(c[0], set())] == []
    assert {name for name, _ in calls} == set(tools)


def test_every_code_in_the_guide_is_one_the_server_raises():
    source = "\n".join(p.read_text(encoding="utf-8") for p in GUIDE.parents[1].rglob("*.py"))
    codes = set(re.findall(r"`([A-Z][A-Z_]{4,})`", GUIDE.read_text(encoding="utf-8")))
    assert len(codes) > 20
    assert sorted(code for code in codes if f'"{code}"' not in source) == []


def test_every_resource_the_guide_names_exists():
    named = set(re.findall(r"leed://[\w/-]+", GUIDE.read_text(encoding="utf-8")))
    assert named and named <= set(URIS)
