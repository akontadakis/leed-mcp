"""The engine and the pure metrics layer share four numbers; they cannot
import each other, so this test keeps the copies equal."""
from leed_daylight.engine import view_trace
from leed_daylight.metrics import views


def test_the_shared_numbers_agree():
    assert view_trace.MISS == views.KIND_MISS
    assert view_trace.UNUSED == views.KIND_UNUSED
    assert view_trace.LAYER_MERGE_M == views.LAYER_MERGE_M
    assert view_trace.LAYER_PARALLEL_COS == views.LAYER_PARALLEL_COS
