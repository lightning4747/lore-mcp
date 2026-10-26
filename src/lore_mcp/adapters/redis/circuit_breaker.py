"""Circuit breaker for Redis adapter connection resilience (Task 12.1)."""

import logging
import time
from typing import Literal

logger = logging.getLogger(__name__)

CircuitState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Three-state circuit breaker preventing cascading failures on Redis outages.

    States:
    - closed: Normal operation. Requests pass through. Failures increment counter.
    - open: Redis is unavailable. Requests fast-fail immediately without network calls.
    - half_open: Cooldown elapsed. A single probe request is permitted to test recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be positive")

        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._state: CircuitState = "closed"
        self._failure_count: int = 0
        self._opened_at: float | None = None
        self._probe_in_flight: bool = False

    @property
    def state(self) -> CircuitState:
        """Current operational state of the circuit breaker."""
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._reset_timeout:
                self._state = "half_open"
                self._probe_in_flight = False
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    def can_execute(self) -> bool:
        """Check whether an operation is allowed to proceed."""
        current_state = self.state
        if current_state == "closed":
            return True
        if current_state == "half_open":
            # Allow exactly one probe request while in half-open state
            if not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False
        # State is open
        return False

    def record_success(self) -> None:
        """Record a successful operation, resetting the circuit to closed."""
        if self._state != "closed":
            logger.info(
                "Redis circuit breaker reset: transitioning from %s to closed.",
                self._state,
            )
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed operation. Trips to open if threshold is reached."""
        self._failure_count += 1
        self._probe_in_flight = False

        if self._failure_count >= self._failure_threshold or self._state == "half_open":
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.warning(
                "Redis circuit breaker tripped OPEN after %d consecutive failures. "
                "Fast-failing for %.1fs. Last error: %s",
                self._failure_count,
                self._reset_timeout,
                error,
            )
