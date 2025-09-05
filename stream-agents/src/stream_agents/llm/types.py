"""
Type definitions for LLM function calling and MCP integration.

This module contains shared type definitions to avoid circular imports
and provide a clean interface for the LLM system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FunctionDefinition:
    """Definition of a function that can be called by an LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    is_async: bool
    return_type: Optional[type] = None


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None
