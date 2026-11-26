"""Check canonical-only discovery and the real SDK's bounded measurement dispatch."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from voice_studio.mcp.voice_catalog import PROFILES, read_catalog
from voice_studio.mcp.voice_mcp import StudioClient, build_server


def structured(result):
    """Read the actual FastMCP structured return rather than assuming a mocked transport shape."""
    return result[1] if isinstance(result, tuple) else json.loads(next(block.text for block in result if block.type == "text"))


class MeasurementMcpTests(IsolatedAsyncioTestCase):
    """Only scratch graph directories and an intercepted bridge transport are used."""

    async def test_all_registered_names_categories_and_graph_references(self):
        """Each registered tool has an intentional category; the graph uses canonical IDs once."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            server = build_server(StudioClient(1, root / "out"), studio_root=root / "studio")
            rows = await server.list_tools()
            names = [row.name for row in rows]
            self.assertCountEqual(names, PROFILES)
            self.assertEqual(len(set(names)), len(names))
            self.assertFalse(any(name.startswith("voice_") for name in names))
            self.assertTrue({"ui_measure", "ui_execute", "ui_windows", "screen_analyze", "audio_denoise", "speech_generate"} <= set(names))
            self.assertTrue(all(row.description for row in rows))
            catalog = await read_catalog(server, include_graph=True)
            self.assertEqual(catalog["uncategorized"], [])
            by_name = {row["name"]: row for row in catalog["tools"]}
            self.assertEqual(by_name["media_mods_list"]["categories"], ["models"])
            self.assertEqual(by_name["ui_measure"]["categories"], ["ui"])
            graph = catalog["graph"]
            self.assertEqual({row["id"] for row in graph["tools"]}, set(names))
            categories = {row["id"] for row in graph["categories"]}
            links = [(row["toolId"], row["categoryId"]) for row in graph["links"]]
            self.assertEqual(len(links), len(set(links)))
            self.assertTrue(all(tool in names and category in categories for tool, category in links))
            self.assertEqual({tool for tool, _ in links}, set(names))
            @server.tool()
            def future_extension() -> dict:
                """Represent a future unclassified provider without silently assigning it to sound."""
                return {}
            unclassified = await read_catalog(server)
            self.assertEqual(unclassified["uncategorized"], ["future_extension"])

    async def test_removed_names_are_neither_advertised_nor_callable(self):
        """Historical tool names are removed, with no hidden aliases or naming profile."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            server = build_server(StudioClient(1, root / "out"), studio_root=root / "studio")
            catalog = await read_catalog(server, include_graph=True)
            self.assertNotIn("namingMode", catalog)
            self.assertTrue(all("legacyName" not in row and "replacement" not in row for row in catalog["tools"]))
            for old in ("voice_ui_windows", "voice_ui_measure", "voice_generate", "voice_catalog", "voice_mods_list"):
                with self.subTest(old=old), self.assertRaisesRegex(Exception, "Unknown tool"):
                    await server.call_tool(old, {})

    async def test_dedicated_measurement_sdk_dispatch_preserves_pairs_and_rejects_bad_batches(self):
        """The ordinary MCP tool forwards to the same page operation with no model or selector execution."""
        sent = []

        class Opener:
            """Intercept only the bridge HTTP request after actual SDK schema validation."""

            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return io.BytesIO(b'{"status":"ok","result":{"coordinateSpace":"viewport-css-px"}}')

        with TemporaryDirectory() as folder, patch("voice_studio.mcp.ui_bridge_mcp.urllib.request.build_opener", return_value=Opener()):
            root = Path(folder)
            server = build_server(StudioClient(1, root / "out"), studio_root=root / "studio")
            args = {"window_id": "b" * 32, "revision": "a" * 32, "targets": ["image-detail", "image-run"],
                    "pairs": [{"from": "image-detail", "to": "image-run"}], "tolerance_px": .5}
            result = structured(await server.call_tool("ui_measure", args))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(sent[0]["name"], "ui_measure")
            self.assertEqual(sent[0]["arguments"]["pairs"], args["pairs"])
            self.assertEqual(sent[0]["revision"], args["revision"])
            for delta in ({"targets": ["a"] * 33}, {"targets": ["a b"]}, {"pairs": [{"from": "a", "to": "b", "script": "no"}]},
                          {"tolerance_px": 17}, {"offset": -1}):
                with self.subTest(delta=delta), self.assertRaises(Exception):
                    await server.call_tool("ui_measure", {**args, **delta})
            self.assertEqual(len(sent), 1)
