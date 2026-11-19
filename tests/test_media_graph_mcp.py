"""Exercise the published MCP contract across a real process restart."""

import asyncio
from base64 import b64decode
from contextlib import asynccontextmanager
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, skipUnless
import wave

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from voice_studio.graph.media_graph_interval_wire import unpack_interval

ROOT = Path(__file__).resolve().parents[1]


@skipUnless(which("ffmpeg") and which("ffprobe"), "FFmpeg and FFprobe are required")
class MediaGraphMcpTests(IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def connect(self, work):
        """Start the shipped server with a private catalog and no Studio history."""
        params = StdioServerParameters(command=sys.executable, args=[
            str(ROOT / "voice_mcp.py"), "--graph-root", str(work / "graphs"),
            "--studio-data", str(work / "studio"),
        ], cwd=ROOT)
        async with stdio_client(params) as streams, ClientSession(*streams) as session:
            await session.initialize()
            yield session

    async def graph_call(self, session, action, **arguments):
        """
        Read JSON and binary attachments while explicitly preserving legacy interval checks.

        session is the initialized MCP connection; action is the graph suffix;
        arguments supplies its tool parameters, defaulting interval wire to expanded.
        Returns parsed JSON and remaining content; a tool error fails the test.
        """
        if action == "interval":
            arguments.setdefault("wire", "expanded")
        response = await session.call_tool(
            f"media_graph_{action}", arguments, read_timeout_seconds=timedelta(seconds=10),
        )
        self.assertFalse(response.isError, response.content)
        return json.loads(response.content[0].text), response.content[1:]

    async def build(self, session, source, report, **settings):
        """Poll until the isolated import publishes its snapshot or reports an error."""
        queued, _ = await self.graph_call(session, "build", file_path=str(source),
                                          analysis_path=str(report), language="en", **settings)
        async with asyncio.timeout(12):
            while True:
                job, _ = await self.graph_call(session, "job", job_id=queued["jobId"])
                self.assertNotEqual(job["status"], "failed", job)
                if job["status"] == "done":
                    return job["snapshot"]["snapshotId"]
                await asyncio.sleep(.05)

    def write_fixture(self, work):
        """Generate a playable source and a deterministic timed report without loading models."""
        source = work / "synthetic.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=16000",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-t", "1", "-c:v", "mpeg4", "-threads", "1", "-c:a", "aac", str(source),
        ], check=True, timeout=10)
        analysis = {
            "sourceDigest": sha256(source.read_bytes()).hexdigest(),
            "language": "en", "durationMs": 1000,
            "shotColors": [{"startMs": 0, "average": "#000000", "sampleCount": 10},
                           {"startMs": 500, "average": "#000000", "sampleCount": 10}],
            "segments": [
                {"startMs": 200, "endMs": 600, "text": "A timed reply", "speakerNumber": 2},
                {"startMs": 640, "endMs": 900, "text": "The next reply", "speakerNumber": 1},
            ],
            "videoEvidence": {"editorTimeline": {"durationMs": 1000, "shots": [{"startMs": 0}],
                "personMotion": [
                    {"startMs": 140, "endMs": 440, "faceId": 2, "track": 3, "horizontal": "left",
                     "vertical": "still", "depth": "foreground", "basis": "camera-compensated"},
                ]}},
            "sceneEvidence": {"status": "ok", "frames": [{
                "timeMs": 240, "text": [{"text": "Example sign", "box": [.1, .1, .3, .2]}],
                "objects": [{"label": "cup", "score": .9, "box": [.02, .02, .04, .04]}],
                "interfaces": [{
                    "box": [.1, .1, .8, .8], "basis": "frame-layout",
                    "elements": [
                        {"text": "Example sign", "role": "button_candidate", "basis": "ocr", "box": [.2, .2, .2, .1]},
                        {"role": "field_candidate", "basis": "outline", "box": [.6, .2, .2, .1]},
                    ],
                    "connections": [{"from": 0, "to": 1, "segment": [[.4, .25], [.6, .25]]}],
                }],
            }], "cameraMotion": [
                {"startMs": 100, "endMs": 460, "horizontal": "right", "vertical": "still", "zoom": "still"},
                {"startMs": 420, "endMs": 620, "horizontal": "right", "vertical": "still", "zoom": "still"},
            ]},
        }
        report = work / "analysis.json"
        report.write_text(json.dumps(analysis), encoding="utf-8")
        return source, report

    async def test_import_reopen_and_navigate_original_media(self):
        """Verify English search, stable pages, linked speakers, timed PNG and WAV."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            source, report = self.write_fixture(work)
            async with self.connect(work) as session:
                snapshot = await self.build(session, source, report)
            async with self.connect(work) as session:
                await self.check_persisted_snapshot(session, snapshot, source)
                partial = json.loads(report.read_text())
                partial.pop("sceneEvidence")
                report.write_text(json.dumps(partial))
                combined = await self.build(session, source, report, visual=False,
                                            reuse_scene_snapshot_id=snapshot)
                merged, _ = await self.graph_call(session, "overview", snapshot_id=combined)
                self.assertEqual(merged["provenance"]["sceneSnapshotId"], snapshot)
                self.assertEqual(merged["nodesByCategory"]["ui"], 4)
                self.assertEqual(merged["nodesByKind"]["utterance"], 2)
                self.assertEqual(merged["nodesByKind"]["shot"], 2)
                self.assertEqual(merged["nodesByKind"]["object_track"], 1)
                self.assertTrue(merged["coverage"]["speech"])
                self.assertTrue(merged["coverage"]["shots"])
                source.write_bytes(b"replacement")
                changed, _ = await self.graph_call(session, "overview", snapshot_id=snapshot)
                self.assertEqual(changed["source"]["status"], "signature-changed")
                changed_interval, changed_images = await self.graph_call(
                    session, "interval", snapshot_id=snapshot, start_ms=0, end_ms=1000)
                self.assertEqual(changed_interval["frames"]["status"], "source-signature-changed")
                self.assertFalse(changed_images)
                source.unlink()
                missing, _ = await self.graph_call(session, "overview", snapshot_id=snapshot)
                self.assertEqual(missing["source"]["status"], "unavailable")
                self.assertEqual(missing["source"]["path"], str(source.resolve()))
                missing_interval, missing_images = await self.graph_call(
                    session, "interval", snapshot_id=snapshot, start_ms=0, end_ms=1000)
                self.assertEqual(missing_interval["frames"]["status"], "source-unavailable")
                self.assertFalse(missing_images)

    async def check_persisted_snapshot(self, session, snapshot, source):
        """
        Verify saved evidence, legacy interval JSON, compact restoration and media attachments.

        session is the reconnected MCP client; snapshot pins the saved graph; source
        is its unchanged original file. Assertion failures report contract regressions.
        """
        catalog, _ = await self.graph_call(session, "list")
        self.assertEqual([row["snapshotId"] for row in catalog["snapshots"]], [snapshot])
        args = {"snapshot_id": snapshot}
        overview, _ = await self.graph_call(session, "overview", **args)
        self.assertEqual(overview["frameSize"], [160, 120])
        self.assertEqual(overview["nodesByCategory"]["ui"], 4)
        ui, _ = await self.graph_call(session, "timeline", **args, start_ms=200, end_ms=400, kind="ui")
        self.assertEqual([item["kind"] for item in ui["nodes"]],
                         ["screen_region", "ui_element", "ui_element", "ui_connection"])
        found, _ = await self.graph_call(session, "search", **args, query="Example sign", kind="ui")
        element, = found["nodes"]
        self.assertEqual(element["kind"], "ui_element")
        detail, _ = await self.graph_call(session, "node", **args, node_id=element["id"])
        self.assertTrue({"in_screen", "contained_in"} <= {edge["relation"] for edge in detail["neighbors"]})
        catalog = await session.call_tool("media_catalog", {"category": "ui"})
        self.assertFalse(catalog.isError, catalog.content)
        discovered = json.loads(catalog.content[0].text)
        self.assertTrue({"screen_analyze", "media_graph_search", "media_graph_timeline"}
                        <= {tool["name"] for tool in discovered["tools"]})
        self.assertNotIn("browser_ml_capabilities", {tool["name"] for tool in discovered["tools"]})
        browser_tools = await session.call_tool("browser_ml_capabilities", {})
        self.assertFalse(browser_tools.isError, browser_tools.content)
        browser_plan = json.loads(browser_tools.content[0].text)
        self.assertEqual(browser_plan["browserCapability"], "unmeasured")
        self.assertEqual(browser_plan["tools"][0]["name"], "browser_ml_probe")

        interval, grid = await self.graph_call(session, "interval", **args, start_ms=0, end_ms=1000)
        self.assertEqual(interval["schemaVersion"], "media-interval/1")
        self.assertEqual(interval["sourceSha256"], overview["sourceSha256"])
        self.assertEqual(interval["sections"]["speech"]["total"], 2)
        self.assertEqual(interval["frames"]["requested"], 4)
        self.assertEqual([sample["requestedMs"] for sample in interval["frames"]["samples"]], [0, 250, 500, 750])
        self.assertEqual(interval["frames"]["returned"], 4)
        self.assertIn("not word-aligned", interval["frames"]["context"]["basis"])
        self.assertEqual(interval["frames"]["context"]["samples"][1]["status"], "quoted-whole-cue")
        self.assertEqual(interval["frames"]["context"]["samples"][0]["status"], "no-returned-cue-at-request")
        self.assertIn("ASR near request", interval["frames"]["samples"][1]["caption"])
        self.assertTrue(all(sample["requestedMs"] <= sample["actualMs"] < 1000
                            for sample in interval["frames"]["samples"]))
        self.assertEqual(grid[0].mimeType, "image/png")
        self.assertTrue(b64decode(grid[0].data).startswith(b"\x89PNG\r\n\x1a\n"))
        no_frames, binary = await self.graph_call(session, "interval", **args, start_ms=0, end_ms=1000, frames=False)
        self.assertEqual(no_frames["frames"]["status"], "disabled")
        self.assertFalse(binary)
        compact = await session.call_tool("media_graph_interval",
                                          {**args, "start_ms": 0, "end_ms": 1000, "frames": False})
        self.assertFalse(compact.isError, compact.content)
        compact_packet = json.loads(compact.content[0].text)
        self.assertEqual(compact_packet["schemaVersion"], "media-interval/2")
        expanded = unpack_interval(compact_packet)
        expanded.pop("timingMs")
        no_frames.pop("timingMs")
        self.assertEqual(expanded, no_frames)
        for invalid in ({"wire": "opaque"}, {"start_ms": True}, {"start_ms": 0.0}, {"frame_count": True},
                        {"frame_count": 9}, {"end_ms": 1001}, {"frames": 1}, {"detail": "fine"}):
            parameters = {**args, "start_ms": 0, "end_ms": 1000, **invalid}
            rejected = await session.call_tool("media_graph_interval", parameters)
            self.assertTrue(rejected.isError, parameters)
        self.assertIn("media_graph_interval", {tool["name"] for tool in discovered["tools"]})

        original = overview["source"]
        self.assertEqual(original["path"], str(source.resolve()))
        self.assertEqual(original["sha256"], sha256(source.read_bytes()).hexdigest())
        self.assertEqual(original["status"], "available")
        self.assertEqual(original["tracks"], [
            {"id": "video_track:0", "kind": "video_track", "streamIndex": 0},
            {"id": "audio_track:1", "kind": "audio_track", "streamIndex": 1},
            {"id": "audio_track:2", "kind": "audio_track", "streamIndex": 2},
        ])
        camera, _ = await self.graph_call(session, "timeline", **args, start_ms=0, end_ms=900, kind="camera_motion")
        self.assertEqual([item["horizontal"] for item in camera["nodes"]], ["right"])
        person, _ = await self.graph_call(session, "node", **args, node_id="person_motion:2:3:140:440")
        self.assertEqual((person["data"]["depth"], person["data"]["horizontal"]), ("foreground", "left"))
        self.assertTrue({edge["relation"] for edge in person["neighbors"]} >= {"in_shot", "instance_of"})
        window = {**args, "start_ms": 0, "end_ms": 1000, "kind": "utterance", "limit": 1}
        first, _ = await self.graph_call(session, "timeline", **window)
        second, _ = await self.graph_call(session, "timeline", **window, after=first["nextAfter"])
        self.assertEqual([row["startMs"] for row in first["nodes"] + second["nodes"]], [200, 640])
        self.assertIsNone(second["nextAfter"])
        speech_id = first["nodes"][0]["id"]
        speech, _ = await self.graph_call(session, "node", **args, node_id=speech_id)
        self.assertTrue(any(edge["relation"] == "spoken_by" for edge in speech["neighbors"]))
        audio, wav = await self.graph_call(session, "audio", **args, node_id=speech_id, duration_ms=200)
        self.assertEqual((audio["startMs"], audio["endMs"]), (200, 400))
        other, other_wav = await self.graph_call(session, "audio", **args, node_id=speech_id,
                                                 duration_ms=200, track_id="audio_track:2")
        self.assertEqual(other["streamIndex"], 2)
        self.assertNotEqual(wav[0].data, other_wav[0].data)
        with wave.open(BytesIO(b64decode(wav[0].data))) as decoded:
            self.assertEqual(decoded.getnframes() * 1000 // decoded.getframerate(), 200)
        sign, _ = await self.graph_call(session, "search", **args, query="Example sign", kind="text_observation")
        sign_id = sign["nodes"][0]["id"]
        technical, png = await self.graph_call(session, "frame", **args, node_id=sign_id, technical=True)
        self.assertEqual(technical["timeMs"], 240)
        self.assertEqual(technical["palette"]["average"], "#000000")
        self.assertEqual(technical["palette"]["caption"]["foreground"], "#FFFFFF")
        self.assertEqual(technical["shotColor"], {"average": "#000000", "sampleCount": 10})
        plain, _ = await self.graph_call(session, "frame", **args, at_ms=240)
        self.assertEqual(plain["palette"], technical["palette"])
        self.assertEqual(technical["overlay"]["caption"], "Speaker 2: A timed reply")
        self.assertTrue(any(box["label"] == "Example sign" for box in technical["overlay"]["boxes"]))
        self.assertEqual(png[0].mimeType, "image/png")
        self.assertTrue(b64decode(png[0].data).startswith(b"\x89PNG\r\n\x1a\n"))
