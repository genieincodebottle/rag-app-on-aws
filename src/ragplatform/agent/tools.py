"""Tools the agent can call, with explicit permissions and schema-validated arguments.

Rules the registry enforces:

* A tool is called only by its registered name; unknown names are refused.
* Arguments are validated against the tool's pydantic model before the
  handler runs. The model cannot pass extra fields (``extra="forbid"``).
* Identity and tenant come from the ``ToolContext`` built by the server, never
  from arguments, so a model cannot ask a tool to read another tenant's data.
* ``write`` tools run only if an approver callback returns True for that exact
  call. No write tool is registered by default; the gate exists so adding one
  cannot skip human approval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragplatform.domain.models import AccessScope, RetrievedChunk, SearchFilters


class ToolDenied(Exception):
    pass


class ToolArgumentsInvalid(Exception):
    pass


@dataclass
class ToolContext:
    scope: AccessScope
    filters: SearchFilters | None
    request_id: str


@dataclass
class ToolOutput:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    snippets: dict[str, str] = field(default_factory=dict)
    note: str = ""


Approver = Callable[[str, BaseModel], bool]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    permission: Literal["read", "write"]
    handler: Callable[[ToolContext, Any], ToolOutput]


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=500)


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=300)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(
        self,
        name: str,
        raw_args: dict[str, Any],
        ctx: ToolContext,
        approver: Approver | None = None,
    ) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolDenied(f"unknown tool {name!r}")
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            raise ToolArgumentsInvalid(f"{name}: {exc.error_count()} invalid argument(s)") from exc
        if tool.permission == "write" and (approver is None or not approver(name, args)):
            raise ToolDenied(f"{name} changes data and needs approval")
        return tool.handler(ctx, args)
