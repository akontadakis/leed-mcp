# tests/test_deliver_data_require_results.py
"""``load_case(require_results=False)``: the atlas model check (atlas spec 5, 15).

With the default ``require_results=True``, ``load_case`` behaves exactly as
before: a case with no usable result is refused. With ``False`` it hands
back the case with an empty ``options`` and every reason in ``skipped``, so
the atlas can draw the model before the first run.
"""
from __future__ import annotations

import pytest
import yaml

from deliver_folder import build_deliver_project
from leed_daylight.deliver.data import OPTIONS, DeliverError, load_case


def test_a_case_with_no_results_loads_as_a_model_check(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())

    case = load_case(folder, require_results=False)

    assert dict(case.options) == {}
    assert sorted(case.skipped) == sorted(OPTIONS)
    assert all("not found" in reason for reason in case.skipped.values())
    assert [storey.label for storey in case.storeys] == ["L0"]
    assert case.groups


def test_without_results_the_default_still_refuses(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())

    with pytest.raises(DeliverError, match="no usable results for case 'base_case'"):
        load_case(folder)
    with pytest.raises(DeliverError, match="no usable results for case 'base_case'"):
        load_case(folder, require_results=True)


def test_with_some_results_both_modes_load_the_same(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))

    required = load_case(folder)
    optional = load_case(folder, require_results=False)

    assert list(required.options) == list(optional.options) == ["option1"]
    assert dict(required.skipped) == dict(optional.skipped)
    assert sorted(optional.skipped) == ["option2", "views"]


def test_stale_results_are_listed_with_their_reason_not_refused(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["project"]["name"] = "Example Tower Two"
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    case = load_case(folder, require_results=False)

    assert dict(case.options) == {}
    assert "option1 results for case 'base_case' are out of date: project.yaml changed" in case.skipped["option1"]
