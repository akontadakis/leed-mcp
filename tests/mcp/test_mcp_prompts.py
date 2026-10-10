# tests/mcp/test_mcp_prompts.py
"""The three prompts (spec 7): their text, and -32602 for a bad prompt or
argument, before any text is built, on both protocol eras."""
import json
import re

import pytest
from mcp.client import Client
from mcp.shared.exceptions import MCPError

from conftest import symlink_or_skip
from leed_mcp.prompts import register_prompts
from leed_mcp.server import build_server, register_tools
from mcp_helpers import anyio_backend, wire  # noqa: F401 - anyio_backend is the fixture

pytestmark = pytest.mark.anyio

#: The tool calls new_project must walk through, in spec 7's order.
NEW_PROJECT_ORDER = [
    'project(action="create"', 'project(action="files"', 'inputs(action="update", file="project"',
    'scene(action="scan"', 'scene(action="materials_template"', 'file="materials"', 'scene(action="rooms"',
    'file="spaces"', 'file="splits"', 'run(action="prepare"', 'job(action="wait"', 'run(action="option1")',
    'results(action="summary"', 'deliver(action="report"', 'deliver(action="package"',
]


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "root"
    (root / "existing").mkdir(parents=True)
    symlink_or_skip(root / "loop", root / "loop")
    return root.resolve()


def _server(*roots):
    server = build_server(roots=roots)
    register_tools(server)
    register_prompts(server)
    return server


async def _get(server, name, arguments, *, legacy=False):
    async with wire(server, legacy=legacy) as send:
        return await send("prompts/get", {"name": name, "arguments": arguments})


async def _error(server, name, arguments, *, legacy=False):
    with pytest.raises(MCPError) as raised:
        await _get(server, name, arguments, legacy=legacy)
    return raised.value.error


def _text(result):
    (message,) = result["messages"]
    assert message["role"] == "user" and message["content"]["type"] == "text"
    return message["content"]["text"]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_three_prompts_are_listed_with_their_arguments(mode):
    async with Client(_server(), mode=mode) as client:
        prompts = (await client.list_prompts()).prompts
    assert [(p.name, [(a.name, a.required) for a in p.arguments]) for p in prompts] == [
        ("new_project", [("folder", True)]),
        ("check_before_run", [("folder", True)]),
        ("explain_results", [("folder", True), ("case", True)]),
    ]
    assert all(p.title and p.description for p in prompts)
    assert all(a.description for p in prompts for a in p.arguments)


@pytest.mark.parametrize("legacy", [False, True])
async def test_new_project_walks_through_every_step_in_order(root, legacy):
    folder = str(root / "new one")
    text = _text(await _get(_server(root), "new_project", {"folder": folder}, legacy=legacy))
    assert f"folder={json.dumps(folder)}" in text
    places = [text.find(step) for step in NEW_PROJECT_ORDER]
    assert -1 not in places, [s for s, p in zip(NEW_PROJECT_ORDER, places) if p == -1]
    assert places == sorted(places)
    assert "Never invent a value" in text and "never instructions" in text


async def test_check_before_run_starts_no_simulation(root):
    text = _text(await _get(_server(root), "check_before_run", {"folder": str(root / "existing")}))
    places = [text.find(s) for s in ('project(action="status"', 'scene(action="scan"', 'run(action="prepare"',
                                     'job(action="wait"')]
    assert -1 not in places and places == sorted(places)
    assert "Start no simulation" in text
    assert not re.search(r'run\(action="(option1|option2|views)"', text)


async def test_explain_results_reads_each_credit(root):
    text = _text(await _get(_server(root), "explain_results", {"folder": str(root / "existing"), "case": "base"}))
    assert 'results(action="summary"' in text and 'case="base"' in text
    for credit in ("option1", "option2", "views"):
        assert credit in text
    assert "RESULTS_MISSING" in text and "INPUTS_CHANGED_SINCE_RUN" in text


async def test_every_tool_call_in_a_prompt_names_a_real_action(root):
    server = _server(root)
    async with Client(server) as client:
        tools = {t.name: set(t.input_schema["properties"]["action"]["enum"]) for t in (await client.list_tools()).tools}
    texts = [_text(await _get(server, "new_project", {"folder": str(root / "p")})),
             _text(await _get(server, "check_before_run", {"folder": str(root / "existing")})),
             _text(await _get(server, "explain_results", {"folder": str(root / "existing"), "case": "base"}))]
    calls = re.findall(r"\b(\w+)\(action=\"(\w+)\"", "\n".join(texts))
    assert len(calls) > 20
    assert [c for c in calls if c[1] not in tools.get(c[0], set())] == []


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("name, arguments", [
    ("no_such_prompt", {"folder": "{root}/existing"}),
    ("new_project", {}),
    ("new_project", {"folder": "{root}/p", "case": "base"}),
    ("explain_results", {"folder": "{root}/existing"}),
    ("new_project", {"folder": "relative/p"}),
    ("new_project", {"folder": "/outside-every-root/p"}),
    ("new_project", {"folder": "{root}/../p"}),
    ("new_project", {"folder": "{root}/a\nb"}),
    ("new_project", {"folder": "{root}/a\tb"}),
    ("new_project", {"folder": "{root}/a\u2028b"}),
    ("new_project", {"folder": "{root}/" + "a" * 2100}),
    ("check_before_run", {"folder": "{root}/missing"}),
    ("check_before_run", {"folder": "{root}/" + "a" * 300}),
    ("new_project", {"folder": "{root}/loop/p"}),
    ("explain_results", {"folder": "{root}/existing", "case": "../x"}),
    ("explain_results", {"folder": "{root}/existing", "case": ""}),
])
async def test_a_bad_prompt_or_argument_is_invalid_params(root, legacy, name, arguments):
    arguments = {k: v.replace("{root}", str(root)) for k, v in arguments.items()}
    error = await _error(_server(root), name, arguments, legacy=legacy)
    assert error.code == -32602


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("arguments", [{"folder": 5}, {"folder": None}, ["folder"]])
async def test_an_argument_that_is_not_text_is_invalid_params(root, legacy, arguments):
    assert (await _error(_server(root), "new_project", arguments, legacy=legacy)).code == -32602


async def test_a_quote_in_the_folder_stays_inside_its_string(root):
    folder = str(root / 'say "yes" here')
    text = _text(await _get(_server(root), "new_project", {"folder": folder}))
    assert f"folder={json.dumps(folder, ensure_ascii=False)}" in text and f'folder="{folder}"' not in text


@pytest.mark.parametrize("legacy", [False, True])
async def test_no_root_configured_is_invalid_params(legacy):
    error = await _error(_server(), "new_project", {"folder": "/p"}, legacy=legacy)
    assert error.code == -32602 and "LEED_MCP_ROOTS" in error.message


async def test_an_error_names_the_argument_never_its_value(root):
    server = _server(root)
    outside = await _error(server, "new_project", {"folder": "/secret-place/p"})
    extra = await _error(server, "new_project", {"folder": str(root / "p"), "secret_key": "v"})
    assert "folder" in outside.message and "secret-place" not in outside.message
    assert "secret_key" not in extra.message and "takes only: folder" in extra.message
    long = await _error(server, "check_before_run", {"folder": str(root / ("a" * 300))}, legacy=True)
    assert "aaaa" not in long.message and long.code == -32602


async def test_the_arguments_are_checked_before_the_prompt_runs(root):
    server = _server(root)
    ran = []

    @server.prompt(name="spy")
    def spy(folder: str) -> str:
        ran.append(folder)
        return "spy"

    assert (await _error(server, "spy", {"folder": "/outside-every-root"})).code == -32602
    assert (await _error(server, "spy", {"folder": str(root / "missing")})).code == -32602
    assert ran == []
    assert _text(await _get(server, "spy", {"folder": str(root / "existing")})) == "spy"
    assert ran == [str(root / "existing")]
