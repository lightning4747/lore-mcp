"""Unit tests for CircuitBreaker (Task 12.1)."""

import time

import pytest

from lore_mcp.adapters.redis.circuit_breaker import CircuitBreaker


@pytest.mark.unit
class TestCircuitBreaker:
    """Verifies state transitions, trip thresholds, and cooldown probe logic."""

    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_failures_increment_until_trip(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)

        cb.record_failure(RuntimeError("err 1"))
        assert cb.state == "closed"
        assert cb.failure_count == 1
        assert cb.can_execute() is True

        cb.record_failure(RuntimeError("err 2"))
        assert cb.state == "closed"
        assert cb.failure_count == 2
        assert cb.can_execute() is True

        # Third failure trips the breaker to open
        cb.record_failure(RuntimeError("err 3"))
        assert cb.state == "open"
        assert cb.failure_count == 3
        assert cb.can_execute() is False

    def test_record_success_resets_closed_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
        cb.record_failure(RuntimeError("err 1"))
        cb.record_failure(RuntimeError("err 2"))
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_cooldown_transitions_to_half_open_and_resets_on_success(self) -> None:
        # Short timeout for testing
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        cb.record_failure(RuntimeError("1"))
        cb.record_failure(RuntimeError("2"))
        assert cb.state == "open"
        assert cb.can_execute() is False

        # Sleep past reset timeout
        time.sleep(0.06)

        # State should now be half_open
        assert cb.state == "half_open"
        # First probe request is allowed
        assert cb.can_execute() is True
        # Subsequent requests while probe in flight are blocked
        assert cb.can_execute() is False

        # Successful probe resets to closed
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_failed_probe_in_half_open_reopens_immediately(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        cb.record_failure(RuntimeError("1"))
        cb.record_failure(RuntimeError("2"))
        assert cb.state == "open"

        time.sleep(0.06)
        assert cb.state == "half_open"
        assert cb.can_execute() is True

        # Probe fails
        cb.record_failure(RuntimeError("probe failed"))
        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="reset_timeout must be positive"):
            CircuitBreaker(reset_timeout=0.0)
