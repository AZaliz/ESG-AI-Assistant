"""MCP-compatible server for ESG RAG assistant tools.

Provides a lightweight JSON-RPC style interface for tool discovery and execution.
"""

from __future__ import annotations

from app.mcp.schemas import ALL_TOOLS
from app.mcp.tools import TOOL_REGISTRY, call_tool, list_tools

__all__ = ["list_tools", "call_tool", "ALL_TOOLS", "TOOL_REGISTRY"]
