# tests/mcp/test_ratelimit.py
"""Tools spec: servers MUST rate limit tool invocations (spec 9)."""
from leed_mcp.ratelimit import DEFAULT_PER_MINUTE, LIMITS, RateLimiter, rate_key


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_chosen_values():
    assert DEFAULT_PER_MINUTE == 120
    assert LIMITS == {"job.wait": 30}


def test_the_key_splits_job_wait_from_other_job_calls():
    assert rate_key("job", {"action": "wait", "job_id": "x"}) == "job.wait"
    assert rate_key("job", {"action": "status"}) == "job"
    assert rate_key("results", {"action": "wait"}) == "results"


def test_calls_over_the_limit_wait_until_the_oldest_leaves_the_window():
    clock = _Clock()
    limiter = RateLimiter(default=2, window_s=60.0, clock=clock)
    assert limiter.check("results") is None
    clock.now += 10
    assert limiter.check("results") is None
    clock.now += 5
    assert limiter.check("results") == 45.0
    clock.now += 45
    assert limiter.check("results") is None


def test_a_refused_call_is_not_counted():
    clock = _Clock()
    limiter = RateLimiter(default=1, clock=clock)
    assert limiter.check("a") is None
    for _ in range(5):
        assert limiter.check("a") is not None
    clock.now += 60
    assert limiter.check("a") is None


def test_keys_are_counted_apart_and_a_key_may_have_its_own_limit():
    clock = _Clock()
    limiter = RateLimiter(limits={"job.wait": 1}, default=3, clock=clock)
    assert limiter.check("job.wait") is None
    assert limiter.check("job.wait") is not None
    assert limiter.check("job") is None
