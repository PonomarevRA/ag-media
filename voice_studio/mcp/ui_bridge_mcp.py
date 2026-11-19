"""MCP access to shared semantic tools in a selected live Studio page."""

import asyncio
import json
import urllib.error
import urllib.request
from uuid import uuid4
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field

from mcp.types import ToolAnnotations

ElementId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[^\s\x00-\x1f]+$")]


class MeasurementPair(BaseModel):
    """Two literal DOM IDs, not CSS selectors or code; preserve the public from/to shape."""

    model_config = ConfigDict(extra="forbid")
    from_id: ElementId = Field(alias="from")
    to: ElementId


class NoBridgeRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward UI commands or their arguments outside the configured local endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Let urllib surface a redirect as an HTTP error rather than following it."""
        return None


def register_ui_bridge_tools(mcp, studio):
    """Bind the local transport to this MCP server's explicit Studio origin."""
    marker = uuid4().hex
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoBridgeRedirect())

    def request(operation, data):
        """Send a bounded non-browser request; the marker is routing, not authentication."""
        req = urllib.request.Request(studio.origin + "/api/ui-bridge/" + operation,
            data=json.dumps(data, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "X-Voice-Studio-MCP": marker})
        try:
            with opener.open(req, timeout=22) as response:
                body = response.read(1_048_577)
                if len(body) > 1_048_576:
                    return {"status": "error", "error": "ui_bridge_response_too_large"}
                value = json.loads(body)
                return value if isinstance(value, dict) else {"status": "error", "error": "ui_bridge_invalid_response"}
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read(65_537)).get("error", "ui_bridge_http_error")
            except (ValueError, AttributeError):
                message = "ui_bridge_http_error"
            return {"status": "error", "error": message}
        except (ValueError, RecursionError):
            return {"status": "error", "error": "ui_bridge_invalid_response"}
        except OSError:
            return {"status": "error", "error": "ui_bridge_service_unavailable"}

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def ui_windows() -> dict:
        """List connected browser/native pages, their IDs, contexts and available semantic tools. No implicit active-window choice."""
        return await asyncio.to_thread(request, "windows", {})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def ui_inspect(window_id: str) -> dict:
        """Read live UI state and a fresh revision from a selected window; inspect again after context errors."""
        return await asyncio.to_thread(request, "call", {"windowId": window_id, "name": "ui_inspect", "arguments": {}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    async def ui_execute(window_id: str, revision: str, name: str, arguments: dict) -> dict:
        """Execute a tool/schema returned by ui_windows, with the last inspected revision. Only shared UI tools are allowed, never scripts or selectors. Timeout means unknown outcome: inspect instead of replaying."""
        return await asyncio.to_thread(request, "call", {"windowId": window_id, "revision": revision,
                                                       "name": name, "arguments": arguments})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def ui_measure(window_id: str, revision: str,
                         targets: Annotated[list[ElementId], Field(max_length=32)] | None = None,
                         pairs: Annotated[list[MeasurementPair], Field(max_length=32)] | None = None,
                         tolerance_px: Annotated[float, Field(ge=0, le=16, allow_inf_nan=False)] = 1,
                         include_text: bool = True,
                         offset: Annotated[int, Field(ge=0, le=4000)] = 0,
                         limit: Annotated[int, Field(ge=1, le=50)] = 32) -> dict:
        """Measure exact element IDs and pair gaps/alignment in a selected connected window using viewport CSS pixels.
        Omit targets/pairs to discover rendered IDs. Returns boxes,padding,DOM-text insets and explicit unknown states,
        never field values, arbitrary selectors/scripts, glyph ink or model predictions. Inspect again after context errors.
        """
        arguments = {"targets": targets or [], "pairs": [pair.model_dump(by_alias=True) for pair in pairs or []],
                     "tolerance_px": tolerance_px, "include_text": include_text, "offset": offset, "limit": limit}
        return await asyncio.to_thread(request, "call", {"windowId": window_id, "revision": revision,
                                                       "name": "ui_measure", "arguments": arguments})
