"""Keep measured dialogue pauses anchored to the original audio clock."""

from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
from unittest import TestCase
import wave

from voice_studio.audio.dialogue_pauses import measure_dialogue_pauses
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


class DialoguePauseTests(TestCase):
    def test_measured_offset_classification_and_overlap_veto(self):
        """A loud region separating cues is not treated as a pause."""
        with TemporaryDirectory() as folder:
            wav = Path(folder) / "clip.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                for amplitude, milliseconds in [(0, 400), (4000, 400), (0, 600), (4000, 400),
                                                (0, 400), (4000, 400), (0, 600)]:
                    audio.writeframes(pack("<h", amplitude) * (16 * milliseconds))
            cues = [{"startMs": 1400, "endMs": 1800}, {"startMs": 2400, "endMs": 3600}]
            pauses = measure_dialogue_pauses(wav, cues, offset_ms=1000)
            self.assertEqual([(p["startMs"], p["cutMs"], p["endMs"], p["position"]) for p in pauses],
                             [(1800, 2100, 2400, "between-cues"), (2800, 3000, 3200, "near-cue")])
            overlap = {"startMs": 2810, "endMs": 2900}
            self.assertEqual(measure_dialogue_pauses(wav, cues, 1000, [overlap]), pauses[:1])

    def test_voice_gap_with_background_is_distinguished_from_quiet(self):
        """A model gap remains a tentative cue even when the soundtrack keeps playing."""
        with TemporaryDirectory() as folder:
            wav = Path(folder) / "music.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                audio.writeframes(pack("<h", 4000) * (16_000 * 2))
            cues = [{"startMs": 0, "endMs": 2000}]
            turns = [{"startMs": 0, "endMs": 500}, {"startMs": 450, "endMs": 650},
                     {"startMs": 880, "endMs": 1700}]
            pause, = measure_dialogue_pauses(wav, cues, turns=turns)
            self.assertEqual((pause["startMs"], pause["cutMs"], pause["endMs"], pause["basis"]),
                             (650, 765, 880, "speaker-gap"))

    def test_graph_timeline_carries_midpoint_and_neighbor_links(self):
        """Give graph readers timed metadata without requiring a per-pause node lookup."""
        with TemporaryDirectory() as folder:
            file = Path(folder) / "source.wav"
            file.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            report = {"language": "en", "segments": [{"startMs": 0, "endMs": 600, "text": "First"},
                                                     {"startMs": 1100, "endMs": 1700, "text": "Second"}],
                      "dialoguePauses": [{"startMs": 600, "endMs": 1100, "cutMs": 0,
                                          "position": "between-cues", "basis": "pcm-quiet"}]}
            graph = build_graph(report, kind="audio", duration_ms=2000)
            snapshot = store.publish(identify(file), {"kind": "audio", "durationMs": 2000},
                                     graph, "en", {})["snapshotId"]
            pause, = store.timeline(snapshot, 0, 2000, "pause")["nodes"]
            self.assertEqual(pause["cutMs"], 850)
            node = store.node(snapshot, pause["id"])
            links = {link["relation"] for link in node["neighbors"]}
            self.assertTrue({"after_utterance", "before_utterance", "on_track"}.issubset(links))
