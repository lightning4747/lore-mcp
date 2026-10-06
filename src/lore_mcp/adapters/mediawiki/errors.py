"""Exceptions for the MediaWiki adapter."""

from typing import Any


class MediaWikiError(Exception):
    """Base exception for all MediaWiki adapter errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class MediaWikiAPIError(MediaWikiError):
    """Raised when MediaWiki returns an API error payload (even with HTTP 200)."""

    def __init__(
        self,
        code: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code: str = code
        self.info: str = info
        self.details: dict[str, Any] = details or {}
        super().__init__(f"MediaWiki API error [{code}]: {info}")


class MediaWikiMaxlagError(MediaWikiAPIError):
    """Raised when MediaWiki server replication lag exceeds threshold."""

    def __init__(
        self,
        info: str,
        lag: float | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.lag: float | None = lag
        self.retry_after: float | None = retry_after
        super().__init__(
            code="maxlag",
            info=info,
            details=details,
        )


class MediaWikiHTTPError(MediaWikiError):
    """Raised when MediaWiki returns an unhandled HTTP error status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code: int = status_code
        super().__init__(f"HTTP error {status_code}: {message}")


class MediaWikiNetworkError(MediaWikiError):
    """Raised when network transport or timeout failures persist."""

    def __init__(self, message: str) -> None:
        super().__init__(f"MediaWiki network error: {message}")
