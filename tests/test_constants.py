# tests/test_constants.py
import leed_daylight
from leed_daylight.constants import TOOL_VERSION, UNVERIFIED as constants_UNVERIFIED
from leed_daylight.metrics.option1 import ASE_HOURS, ASE_LUX, SDA_FRACTION, SDA_LUX
from leed_daylight.metrics.scoring import UNVERIFIED as scoring_UNVERIFIED
from leed_daylight.project.rules import UNVERIFIED as rules_UNVERIFIED


def test_unverified_is_defined_once():
    assert constants_UNVERIFIED is scoring_UNVERIFIED
    assert constants_UNVERIFIED is rules_UNVERIFIED


def test_the_package_version_is_defined_once():
    # M3: leed_daylight.__version__ is a re-export of constants.TOOL_VERSION,
    # not a second copy of the string, so build.prepare's manifest and the
    # package's own version can never drift apart.
    assert leed_daylight.__version__ is TOOL_VERSION


def test_the_lm83_reference_values_are_still_importable():
    assert SDA_LUX == 300.0
    assert SDA_FRACTION == 0.50
    assert ASE_LUX == 1000.0
    assert ASE_HOURS == 250
