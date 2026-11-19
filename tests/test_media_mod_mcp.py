"""Verify the shipped MCP process exposes the real mod settings and test operations."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


class MediaModMcpTests(IsolatedAsyncioTestCase):
    async def test_fresh_mcp_client_can_create_enable_configure_and_execute_template(self):
        """Run actual typed tools across stdio with isolated external configuration."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / 'voice_mcp.py'), '--graph-root', str(work / 'graphs'), '--studio-data', str(work / 'studio')],
                cwd=ROOT, env={**os.environ, 'VOICE_MEDIA_MODS_DIR': str(work / 'mods')})
            async with stdio_client(params) as streams, ClientSession(*streams) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                self.assertTrue({'media_mods_pipeline', 'media_mods_contracts', 'media_mods_list', 'media_mods_rescan', 'media_mods_enable', 'media_mods_select',
                                 'media_mods_configure', 'media_mods_template', 'media_mods_test'} <= names)
                async def call(name, arguments=None):
                    result = await session.call_tool(name, arguments or {})
                    self.assertFalse(result.isError, result.content)
                    return json.loads(next(item.text for item in result.content if hasattr(item, 'text')))
                pipeline = await call('media_mods_pipeline')
                self.assertEqual(len(pipeline['nodes']), 8)
                changed = await call('media_mods_pipeline', {'dependencies': {}, 'expectedRevision': 0})
                self.assertEqual(changed['revision'], 1)
                contracts = await call('media_mods_contracts')
                self.assertTrue(any(item['id'] == 'studio.asr' for item in contracts['contracts']))
                converted = await call('media_mods_template', {'id': 'mcp.adapter', 'template': 'asr-seconds'})
                adapter = next(row for row in converted['mods'] if row['id'] == 'mcp.adapter')
                self.assertFalse(adapter['enabled'])
                self.assertEqual(adapter['manifest']['capabilities'][0]['contract'], {'id': 'studio.asr', 'version': 1})
                catalog = await call('media_mods_template', {'id': 'mcp.fixture'})
                row = next(row for row in catalog['mods'] if row['id'] == 'mcp.fixture')
                await call('media_mods_enable', {'id': row['id'], 'fingerprint': row['fingerprint']})
                await call('media_mods_configure', {'id': row['id'], 'settings': {'minimumLength': 4}})
                value = await call('media_mods_test', {'id': row['id'], 'capability': 'analysis.word-count',
                                                      'payload': {'transcript': 'one seven thirty'}})
                self.assertEqual(value['result']['wordCount'], 2)
                item = next(item for item in value['modProvenance'] if item['modId'] == 'mcp.fixture')
                self.assertEqual(item['settings'], {'minimumLength': 4})
