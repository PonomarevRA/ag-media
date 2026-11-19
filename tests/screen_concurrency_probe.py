"""Fresh stdio MCP proof: real screenshot OCR completes while controlled video HTTP waits.

The video backend is an explicit loopback HTTP fixture held by threading.Event;
this proves dispatch isolation, not that a recognition model processed the dummy video.
Screenshot parsing uses the actual native worker and source bytes, without mocks.
"""

import argparse
import asyncio
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from voice_studio.vision.video_revolver import PYTHON, VISION


def make_screenshot(path):
    """Draw a deterministic generic interface whose labels can be read by actual OCR."""
    script = """
import cv2,numpy as np,sys
image=np.full((480,800,3),245,np.uint8)
ink=(20,20,20)
cv2.rectangle(image,(15,15),(785,465),ink,2)
for text,point in [('Project Settings',(40,60)),('Open project',(65,145)),
                   ('Search documents',(60,255)),('Document 01',(50,355)),('Document 02',(50,410))]:
    cv2.putText(image,text,point,cv2.FONT_HERSHEY_SIMPLEX,.9,ink,2)
cv2.rectangle(image,(40,100),(350,170),ink,2)
cv2.rectangle(image,(40,210),(740,275),ink,2)
if not cv2.imwrite(sys.argv[1],image):raise OSError('Cannot write controlled screenshot')
"""
    subprocess.run([str(PYTHON), '-c', script, str(path)],
                   env={**os.environ, 'PYTHONPATH':str(VISION)}, check=True, timeout=10)


def data_of(result):
    """Read SDK structuredContent or its JSON text fallback, raising on tool errors."""
    if result.isError:
        raise ValueError(next(item.text for item in result.content if item.type == 'text'))
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(next(item.text for item in result.content if item.type == 'text'))


async def probe(root, output):
    """Run real client/server transport and bounded local HTTP/OCR work, then write evidence."""
    entered, release = Event(), Event()
    http_evidence = {}

    class ControlledVideoBackend(BaseHTTPRequestHandler):
        """Hold the actual StudioClient HTTP exchange at an explicit analysis boundary."""

        def log_message(self, format, *args):
            """Keep the fixture server quiet so output remains machine-readable."""

        def do_POST(self):
            """Read uploaded fixture bytes, signal work, and await the controller's release."""
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            http_evidence.update(path=self.path, sourceSha256=sha256(body).hexdigest(), bytes=len(body))
            entered.set()
            if not release.wait(20):
                self.send_error(504, 'Controlled video boundary timed out')
                return
            payload = json.dumps({'language':'en', 'segments':[], 'srt':'',
                                  'controlledVideoBoundary':True}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    backend = ThreadingHTTPServer(('127.0.0.1',0), ControlledVideoBackend)
    thread = Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with TemporaryDirectory() as folder:
        scratch = Path(folder)
        image, video = scratch/'interface.png', scratch/'controlled-video.mov'
        make_screenshot(image)
        video.write_bytes(b'Explicit controlled video HTTP fixture; no decoder/model proof.')
        parameters = StdioServerParameters(command=str(root/'.venv/bin/python'),
            args=[str(root/'ag_media_studio_mcp.py'), '--port', str(backend.server_port),
                  '--output', str(scratch/'output'), '--studio-data', str(scratch/'studio'),
                  '--graph-root', str(scratch/'graphs')], env=dict(os.environ))
        try:
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    assert {'media_transcribe','screen_analyze','screen_scroll'} <= names
                    started = perf_counter()
                    transcript = asyncio.create_task(session.call_tool('media_transcribe',
                        {'file_path':str(video), 'language':'en', 'visual':True, 'scene':True}))
                    try:
                        assert await asyncio.to_thread(entered.wait, 3), 'Video HTTP fixture was not entered.'
                        before = perf_counter()
                        screenshot = data_of(await asyncio.wait_for(session.call_tool('screen_analyze',
                            {'file_path':str(image), 'language':'en', 'mode':'overview'}), 15))
                        screen_elapsed = perf_counter()-before
                        during = entered.is_set() and not release.is_set() and not transcript.done()
                        assert during, 'UI did not complete while video HTTP remained held.'
                        assert screenshot['source']['sha256'] == sha256(image.read_bytes()).hexdigest()
                        assert screenshot['status'] == 'draft'
                        assert screenshot['ocr'], 'Actual OCR did not return any interface text.'
                        # A second SDK roundtrip remains live before the video boundary is released.
                        await asyncio.wait_for(session.list_tools(), 2)
                    finally:
                        release.set()
                    video_result = data_of(await asyncio.wait_for(transcript, 5))
                    retained = output.parent/'ui-concurrency-source.png'
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    retained.write_bytes(image.read_bytes())
                    result = {'status':'pass', 'transport':'fresh stdio MCP ClientSession',
                              'uiCompletedWhileControlledVideoHttpHeld':during,
                              'uiElapsedSeconds':round(screen_elapsed,3),
                              'totalSeconds':round(perf_counter()-started,3),
                              'videoBackend':http_evidence,
                              'videoResultControlledBoundary':video_result.get('controlledVideoBoundary'),
                              'screenshot':{'source':screenshot['source'], 'retainedFixturePath':str(retained),
                                  'status':screenshot['status'],
                                  'schemaVersion':screenshot['schemaVersion'], 'ocrJobs':screenshot['ocrJobs'],
                                  'ocrStatuses':screenshot['ocrStatuses'],
                                  'ocrText':[row['text'] for row in screenshot['ocr']][:20],
                                  'draftDescription':screenshot['draftDescription']},
                              'qualification':'Real screenshot worker/OCR and real stdio/HTTP transport; video backend is an explicit controlled Event fixture, not real model enrichment.'}
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2))
                    return result
        finally:
            release.set()
            backend.shutdown()
            backend.server_close()
            thread.join(timeout=2)


def main():
    """Accept an explicit repository root and evidence output path for local reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.root.resolve(), args.output.resolve())), indent=2))


if __name__ == '__main__':
    main()
