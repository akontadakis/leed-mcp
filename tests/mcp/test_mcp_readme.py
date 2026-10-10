# tests/mcp/test_mcp_readme.py
"""README.md says how to install, configure and connect the server (spec 11, 13 N3)."""
import json
import pathlib
import re
import shlex

import pytest
from mcp.client import Client

from leed_mcp.paths import ROOTS_ENV_VAR
from leed_mcp.prompts import register_prompts
from leed_mcp.server import build_server, register_tools
from leed_mcp.static_resources import RESOURCES
from mcp_helpers import anyio_backend  # noqa: F401 - the fixture
from rawwire import HERE

README = (HERE.parents[1] / "README.md").read_text(encoding="utf-8")


#: What the plugin, the uvx examples and `.mcp.json` all launch.
UVX_ARGS = ["--from", "leed-mcp[report] @ git+https://github.com/akontadakis/leed-mcp", "leed-mcp"]


def _servers():
    """Each JSON example's leed-mcp server: the clone's first, then uvx's."""
    blocks = re.findall(r"```json\n(.*?)```", README, re.S)
    return [json.loads(block)["mcpServers"]["leed-mcp"] for block in blocks]


def _clone_server():
    (server,) = [s for s in _servers() if s["command"] != "uvx"]
    return server


def _commands(start):
    return [shlex.split(line.strip()) for line in README.splitlines() if line.strip().startswith(start)]


def _command(start):
    """The command line for a clone: it ends with the clone's server program."""
    (words,) = [w for w in _commands(start) if w[-1].endswith("/.venv/bin/leed-mcp")]
    return words


@pytest.mark.anyio
async def test_it_names_every_tool_action_resource_and_prompt():
    server = build_server(roots=())
    register_tools(server)
    register_prompts(server)
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        prompts = (await client.list_prompts()).prompts
    for tool in tools:
        actions = tool.input_schema["properties"]["action"]["enum"]
        assert f"| `{tool.name}` | {', '.join(f'`{a}`' for a in actions)} |" in README, tool.name
    for prompt in prompts:
        assert f"| `{prompt.name}` |" in README, prompt.name
    for resource in RESOURCES:
        assert f"| `{resource.uri}` |" in README, resource.uri
    assert "`io.modelcontextprotocol/tasks`" in README


def test_the_install_needs_no_pth_file():
    assert "    uv sync --extra report --no-editable --reinstall-package leed-mcp\n" in README
    assert "    uv sync --extra dev --no-editable\n" in README
    assert "chflags nohidden" in README and "ls -lO .venv/lib/python3.12/site-packages/*.pth" in README
    assert "pip install" not in README


def test_the_claude_desktop_config_is_valid_json_with_an_absolute_command():
    server = _clone_server()
    assert server["command"].startswith("/") and server["command"].endswith("/.venv/bin/leed-mcp")
    assert set(server["env"]) == {ROOTS_ENV_VAR, "PATH"}
    assert server["env"]["PATH"].endswith(":/usr/bin:/bin")


@pytest.mark.parametrize("start", ["claude mcp add", "npx -y @modelcontextprotocol/inspector"])
def test_a_command_line_passes_the_roots_and_path_and_the_server(start):
    words = _command(start)
    env = [words[i + 1] for i, word in enumerate(words) if word == "-e"]
    assert [item.split("=", 1)[0] for item in env] == [ROOTS_ENV_VAR, "PATH"]
    assert words[-2] == "--" and words[-1].endswith("/.venv/bin/leed-mcp")
    assert _clone_server()["command"] == words[-1]


def test_the_uvx_examples_and_the_plugin_launch_the_same_server():
    (server,) = [s for s in _servers() if s["command"] == "uvx"]
    assert server["args"] == UVX_ARGS
    assert set(server["env"]) == {ROOTS_ENV_VAR, "PATH"}
    (words,) = [w for w in _commands("claude mcp add") if "uvx" in w]
    assert [words[i + 1].split("=", 1)[0] for i, w in enumerate(words) if w == "-e"] == [ROOTS_ENV_VAR, "PATH"]
    assert words[words.index("--") + 1:] == ["uvx", *UVX_ARGS]
    root = HERE.parents[1]
    plugin_server = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["leed-mcp"]
    assert plugin_server == {"command": "uvx", "args": UVX_ARGS}
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == market["name"] == market["plugins"][0]["name"] == "leed-mcp"
    assert "/plugin install leed-mcp@leed-mcp" in README
    assert (root / "skills" / "running-leed-studies" / "SKILL.md").is_file()


def test_every_configuration_variable_is_explained():
    for variable in (ROOTS_ENV_VAR, "LEED_MCP_STATE", "PATH", "RAYPATH", "BLENDER"):
        assert f"| `{variable}` |" in README, variable


def test_the_install_covers_three_systems():
    for heading in ("### macOS", "### Windows", "### Linux"):
        assert heading in README, heading
    assert r".venv\Scripts\leed-mcp.exe" in README
    assert r"%APPDATA%\Claude\claude_desktop_config.json" in README
    assert "~/.config/Claude/claude_desktop_config.json" in README
    assert "separated by `:` (macOS, Linux) or `;` (Windows)" in README
    assert "Finder" not in README
    assert "python tools/no_radiance.py" in README


def test_the_prompts_name_no_mac_program():
    from leed_mcp import prompts

    assert "Finder" not in pathlib.Path(prompts.__file__).read_text(encoding="utf-8")
