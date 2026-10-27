"""In-process Redis command counter and monthly quota shedding tracker (Task 12.5)."""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class CommandTracker:
    """Tracks in-process Redis command invocations against free-tier monthly budget.

    The managed free-tier provides ~500,000 commands/month [S15].
    When projected monthly usage crosses ~80% (400,000 commands), caching operations
    are shed (skipped with a warning) to preserve Redis capacity for essential rate
    guards and daily web budgets.
    """

    def __init__(
        self,
        monthly_budget: int = 500000,
        shedding_ratio: float = 0.80,
    ) -> None:
        if monthly_budget <= 0:
            raise ValueError("monthly_budget must be positive")
        if not (0.0 < shedding_ratio <= 1.0):
            raise ValueError("shedding_ratio must be between 0.0 and 1.0")

        self._monthly_budget = monthly_budget
        self._shedding_ratio = shedding_ratio
        self._shedding_threshold = monthly_budget * shedding_ratio
        self._command_count: int = 0
        self._current_year_month: tuple[int, int] | None = None
        self._shedding_logged: bool = False

    @property
    def command_count(self) -> int:
        """Total in-process command count in current month window."""
        return self._command_count

    @property
    def shedding_threshold(self) -> float:
        """Command count threshold triggering shedding."""
        return self._shedding_threshold

    def record_command(self, count: int = 1) -> None:
        """Record one or more Redis command invocations."""
        if count > 0:
            self._command_count += count

    def get_projected_monthly_usage(self, now: datetime | None = None) -> float:
        """Calculate projected monthly Redis command usage based on month progress."""
        current_time = now or datetime.now(UTC)
        year_month = (current_time.year, current_time.month)

        if self._current_year_month is None:
            self._current_year_month = year_month
        elif self._current_year_month != year_month:
            # New month: reset tracker
            self._current_year_month = year_month
            self._command_count = 0
            self._shedding_logged = False
            return 0.0

        day = current_time.day
        hour = current_time.hour
        minute = current_time.minute

        # Approximate days in current month (28 to 31)
        if current_time.month in {1, 3, 5, 7, 8, 10, 12}:
            days_in_month = 31
        elif current_time.month in {4, 6, 9, 11}:
            days_in_month = 30
        else:
            is_leap = current_time.year % 4 == 0 and (
                current_time.year % 100 != 0 or current_time.year % 400 == 0
            )
            days_in_month = 29 if is_leap else 28

        elapsed_days = (day - 1) + (hour / 24.0) + (minute / 1440.0)
        # Clamp minimum elapsed days to 0.5 to avoid large multiplier spikes at start
        effective_elapsed = max(elapsed_days, 0.5)
        fraction_elapsed = effective_elapsed / days_in_month

        return self._command_count / fraction_elapsed

    def is_shedding_active(self, now: datetime | None = None) -> bool:
        """Check whether caching operations should be shed to preserve budget."""
        # 1. Direct count threshold check
        if self._command_count >= self._shedding_threshold:
            if not self._shedding_logged:
                logger.warning(
                    "Redis command budget threshold reached (%d >= %.0f, 80%% of %d). "
                    "Shedding cache operations to preserve monthly free-tier quota.",
                    self._command_count,
                    self._shedding_threshold,
                    self._monthly_budget,
                )
                self._shedding_logged = True
            return True

        # 2. Projected monthly usage check
        projected = self.get_projected_monthly_usage(now)
        if projected >= self._shedding_threshold:
            if not self._shedding_logged:
                logger.warning(
                    "Redis projected monthly usage (%.0f commands) exceeds 80%% limit "
                    "(%.0f of %d). Shedding cache operations for free-tier quota.",
                    projected,
                    self._shedding_threshold,
                    self._monthly_budget,
                )
                self._shedding_logged = True
            return True

        # If usage dropped below threshold, allow logging again if it trips later
        if (
            self._shedding_logged
            and self._command_count < self._shedding_threshold * 0.9
        ):
            self._shedding_logged = False

        return False

    def reset(self) -> None:
        """Reset counter (primarily for testing)."""
        self._command_count = 0
        self._shedding_logged = False
