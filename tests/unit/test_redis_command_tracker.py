"""Unit tests for CommandTracker (Task 12.5)."""

from datetime import UTC, datetime

import pytest

from lore_mcp.adapters.redis.command_tracker import CommandTracker


@pytest.mark.unit
class TestCommandTracker:
    """Verifies command counting, monthly usage projection, and 80% quota shedding."""

    def test_initial_state(self) -> None:
        tracker = CommandTracker(monthly_budget=500000, shedding_ratio=0.80)
        assert tracker.command_count == 0
        assert tracker.shedding_threshold == 400000.0
        assert tracker.is_shedding_active() is False

    def test_record_command_increments(self) -> None:
        tracker = CommandTracker(monthly_budget=1000, shedding_ratio=0.80)
        tracker.record_command()
        assert tracker.command_count == 1
        tracker.record_command(5)
        assert tracker.command_count == 6

    def test_direct_count_trips_shedding(self) -> None:
        tracker = CommandTracker(monthly_budget=100, shedding_ratio=0.80)
        tracker.record_command(79)
        # End of 30-day month -> projected == current_count (79 < 80)
        test_now = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
        assert tracker.is_shedding_active(now=test_now) is False

        tracker.record_command(1)  # Reaches 80 (>= 80)
        assert tracker.is_shedding_active(now=test_now) is True

    def test_projected_monthly_usage_trips_shedding(self) -> None:
        # On Day 3 of a 30-day month, if 50,000 commands are used,
        # projected is ~500,000, which exceeds 80% of 500,000 (400,000)
        tracker = CommandTracker(monthly_budget=500000, shedding_ratio=0.80)
        tracker.record_command(50000)

        # Day 3 at 00:00 -> elapsed = 2.0 days / 30 = 1/15th month -> projected ~750,000
        now = datetime(2026, 4, 3, 0, 0, tzinfo=UTC)
        projected = tracker.get_projected_monthly_usage(now)
        assert projected > 400000.0
        assert tracker.is_shedding_active(now=now) is True

    def test_reset_clears_counter(self) -> None:
        tracker = CommandTracker(monthly_budget=100, shedding_ratio=0.80)
        tracker.record_command(90)
        assert tracker.command_count == 90
        tracker.reset()
        assert tracker.command_count == 0
        assert tracker.is_shedding_active() is False

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="monthly_budget must be positive"):
            CommandTracker(monthly_budget=0)

        with pytest.raises(
            ValueError, match=r"shedding_ratio must be between 0\.0 and 1\.0"
        ):
            CommandTracker(shedding_ratio=0.0)

        with pytest.raises(
            ValueError, match=r"shedding_ratio must be between 0\.0 and 1\.0"
        ):
            CommandTracker(shedding_ratio=1.5)
