# leed_mcp/ratelimit.py
"""Per-tool rate limit (spec 9). Tools spec, Security: *"Servers MUST ...
Rate limit tool invocations."* A sliding window per key. The numbers are
chosen, not measured: generous for a person working in chat, low enough to
stop a runaway loop of calls.
"""
from __future__ import annotations

import collections
import time
from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_PER_MINUTE = 120
#: `job(action="wait")` blocks up to 50 s per call, so 30 a minute is already more than a client needs.
LIMITS: dict[str, int] = {"job.wait": 30}
WINDOW_S = 60.0


def rate_key(tool: str, arguments: Mapping[str, Any]) -> str:
    """The counter a call goes to."""
    if tool == "job" and arguments.get("action") == "wait":
        return "job.wait"
    return tool


class RateLimiter:
    """At most `limits.get(key, default)` calls per key in any `window_s` seconds."""

    def __init__(self, *, limits: Mapping[str, int] | None = None, default: int = DEFAULT_PER_MINUTE,
                 window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._limits = dict(LIMITS if limits is None else limits)
        self._default = default
        self._window = window_s
        self._clock = clock
        self._calls: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

    def check(self, key: str) -> float | None:
        """None if the call may go ahead (it is counted); else the seconds to wait."""
        now = self._clock()
        calls = self._calls[key]
        while calls and calls[0] <= now - self._window:
            calls.popleft()
        if len(calls) >= self._limits.get(key, self._default):
            return round(calls[0] + self._window - now, 3)
        calls.append(now)
        return None
