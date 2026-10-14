"""MCP server interface."""

from lore_mcp.interfaces.mcp.errors import (
    McpErrorCode,
    McpErrorResponse,
    handle_mcp_errors,
    map_exception_to_mcp_error,
)
from lore_mcp.interfaces.mcp.models import (
    DialogueSnippet,
    EntityDetails,
    EventComparison,
    EventComparisonSide,
    EventDetails,
    EventTrace,
    SeriesSummary,
)
from lore_mcp.interfaces.mcp.readiness import ReadinessChecker
from lore_mcp.interfaces.mcp.server import create_http_app, create_mcp_server

__all__ = [
    "DialogueSnippet",
    "EntityDetails",
    "EventComparison",
    "EventComparisonSide",
    "EventDetails",
    "EventTrace",
    "McpErrorCode",
    "McpErrorResponse",
    "ReadinessChecker",
    "SeriesSummary",
    "create_http_app",
    "create_mcp_server",
    "handle_mcp_errors",
    "map_exception_to_mcp_error",
]
