# leed_mcp/static_resources.py
"""The three static resources (spec 7, D7). Project files are not served.

- ``leed://docs/running-a-project``: the guide written for this server's
  tools, packaged as ``leed_mcp/resources/running_a_project.md``.
- ``leed://templates/project`` and ``leed://templates/credit-rules``: the
  pipeline's own templates, read from ``leed_daylight/templates/`` (package
  data), so they are always the files ``project(action="create")`` writes.

Each file is read when a client reads the resource. The SDK serves them
(``FileResource``): an unknown URI is the JSON-RPC error -32602, and the cache
fields are its defaults (``ttlMs`` 0, ``cacheScope`` ``"private"``).
``FileResource`` serves a MIME type it does not know as text (YAML) as a
base64 blob unless an ``encoding`` is given, so every resource names UTF-8.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp.server.mcpserver.resources import FileResource

import leed_daylight

if TYPE_CHECKING:
    from leed_mcp.server import LeedMCPServer

__all__ = ["GUIDE", "RESOURCES", "StaticResource", "TEMPLATES", "register_resources"]

#: The guide, packaged with the server (``[tool.setuptools.package-data]``).
GUIDE = pathlib.Path(__file__).resolve().parent / "resources" / "running_a_project.md"
#: The pipeline's templates folder.
TEMPLATES = pathlib.Path(leed_daylight.__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class StaticResource:
    """One resource of spec 7's table."""

    uri: str
    name: str
    title: str
    description: str
    mime_type: str
    path: pathlib.Path


RESOURCES = (
    StaticResource(
        uri="leed://docs/running-a-project", name="running-a-project", title="Running a project",
        description="How to take a Radiance model to the LEED v4.1 BD+C daylight and views results with this "
                    "server's tools, step by step: the project folder, the scene, the input files, the runs, the "
                    "results, the reports and the submission folder.",
        mime_type="text/markdown", path=GUIDE),
    StaticResource(
        uri="leed://templates/project", name="project-template", title="project.yaml template",
        description="The project.yaml that project(action=\"create\") writes: every value is FILL_ME, with "
                    "comments saying what each one means.",
        mime_type="application/yaml", path=TEMPLATES / "project_template.yaml"),
    StaticResource(
        uri="leed://templates/credit-rules", name="credit-rules-template", title="LEED v4.1 BD+C credit rules",
        description="The credit_rules.yaml that project(action=\"create\") writes: the LEED v4.1 BD+C "
                    "thresholds and point bands of the daylight and views credits.",
        mime_type="application/yaml", path=TEMPLATES / "credit_rules_v41_bdc_nc.yaml"),
)


def register_resources(server: LeedMCPServer) -> None:
    """Add the three resources (spec 7), in the spec's order."""
    for resource in RESOURCES:
        server.add_resource(FileResource(
            uri=resource.uri, name=resource.name, title=resource.title, description=resource.description,
            mime_type=resource.mime_type, path=resource.path, encoding="utf-8"))
