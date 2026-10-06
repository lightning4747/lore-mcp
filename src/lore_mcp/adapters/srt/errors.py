"""Exceptions for the SRT subtitle adapter."""


class SrtError(Exception):
    """Base exception for all SRT adapter errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class SrtParseError(SrtError):
    """Raised when an SRT subtitle file cannot be parsed or contains malformed data."""

    def __init__(self, path: str, reason: str) -> None:
        self.path: str = path
        self.reason: str = reason
        super().__init__(f"Failed to parse SRT at '{path}': {reason}")


class SrtEncodingError(SrtError):
    """Raised when an SRT subtitle file cannot be decoded with supported encodings."""

    def __init__(self, path: str, attempted_encodings: list[str]) -> None:
        self.path: str = path
        self.attempted_encodings: list[str] = attempted_encodings
        enc_str = ", ".join(attempted_encodings)
        super().__init__(
            f"Failed to decode SRT file at '{path}' using encodings: {enc_str}"
        )
