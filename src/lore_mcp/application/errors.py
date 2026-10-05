"""Application error taxonomy."""

from collections.abc import Sequence


class ApplicationError(Exception):
    """Base exception for all application layer errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class InvalidArgument(ApplicationError):
    """Raised when an argument violates semantic or structural constraints."""

    def __init__(self, message: str, argument_name: str | None = None) -> None:
        super().__init__(message)
        self.argument_name: str | None = argument_name


class InvalidScope(ApplicationError):
    """Raised when a requested series scope is unknown or unsupported."""

    def __init__(
        self,
        scope: str,
        suggestions: Sequence[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.scope: str = scope
        self.suggestions: list[str] = list(suggestions[:3]) if suggestions else []
        if message is None:
            if self.suggestions:
                formatted = ", ".join(repr(s) for s in self.suggestions)
                msg = (
                    f"Unknown or invalid series scope '{scope}'. "
                    f"Did you mean: {formatted}?"
                )
            else:
                msg = f"Unknown or invalid series scope '{scope}'."
        else:
            msg = message
        super().__init__(msg)


class DataUnavailable(ApplicationError):
    """Raised when requested data is not found in the index or storage."""

    def __init__(self, message: str, resource_id: str | None = None) -> None:
        super().__init__(message)
        self.resource_id: str | None = resource_id


class RateLimited(ApplicationError):
    """Raised when a request exceeds daily quota or rate limit guardrails."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds: float | None = retry_after_seconds


class DependencyUnavailable(ApplicationError):
    """Raised when an external backing dependency is unreachable or fails."""

    def __init__(self, service_name: str, message: str) -> None:
        self.service_name: str = service_name
        full_message = f"Dependency '{service_name}' is unavailable: {message}"
        super().__init__(full_message)


class IncompatibleData(ApplicationError):
    """Raised when schema version or embedding dimension fails compatibility checks."""

    def __init__(
        self,
        message: str,
        expected_version: int | None = None,
        actual_version: int | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_version: int | None = expected_version
        self.actual_version: int | None = actual_version
