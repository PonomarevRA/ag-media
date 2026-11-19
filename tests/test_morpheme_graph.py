"""Exercise alignment, contextual retry, and mixed-take PCM assembly."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from array import array
import wave

from voice_studio.synthesis.block_assembly import checked_blocks, render_blocks
from voice_studio.prosody.morpheme_graph import fragment, word_graph
from voice_studio.prosody.revolver import review_words
from voice_studio.app.studio import Studio
from test_studio import sample


class WordGraphTests(TestCase):
    def test_alignment_preserves_stress_and_reports_uncertain_timing(self):
        segments = [
            {"text": "вот", "offsets": {"from": 0, "to": 120}},
            {"text": "пять шариков", "offsets": {"from": 120, "to": 400}},
            {"text": "лежали", "offsets": {"from": 400, "to": 650}},
            {"text": "очень", "offsets": {"from": 650, "to": 800}},
            {"text": "дома", "offsets": {"from": 800, "to": 900}},
        ]
        graph = word_graph("Пять ша́риков лежат очень тихо дома.", segments)
        self.assertEqual([r["kind"] for r in graph["rows"]],
                         ["extra", "match", "match", "substitution", "match", "missing", "match"])
        word = graph["rows"][2]
        self.assertEqual(word["expected"], "ша́риков")
        self.assertIsNone(word["heard"]["startMs"])
        self.assertIsNone(word["heard"]["timingSource"])
        self.assertEqual([part for part, _ in word["morphemes"]], ["шар", "ик", "ов"])
        self.assertEqual([syllable["text"] for syllable in word["syllables"]], ["ша", "ри", "ков"])
        self.assertEqual([syllable["stressed"] for syllable in word["syllables"]], [True, False, False])
        self.assertEqual(word["stress"], {"index": 1, "source": "manual"})
        self.assertAlmostEqual(graph["wordErrorRate"], 3 / 6)

    def test_text_stress_and_boundary_pause_are_independent(self):
        """Keep acoustic intervals and source labels separate from estimated syllables."""
        segments = [
            {"text": "Надежный", "offsets": {"from": 0, "to": 420}},
            {"text": "замок", "offsets": {"from": 580, "to": 920}},
            {"text": "лежит", "offsets": {"from": 920, "to": 1230}},
        ]
        graph = word_graph("Надёжный замок лежит.", segments, quiet=[
            {"start": .42, "end": .58}, {"start": 1.60, "end": 1.84}]
        )
        first, second, _ = graph["rows"]
        self.assertEqual(first["stress"], {"index": 3, "source": "yo"})
        self.assertIsNone(second["stress"]["index"])
        self.assertEqual(first["heard"]["timingSource"], "whisper")
        self.assertNotIn("startMs", first["syllables"][0])
        self.assertEqual(graph["pauses"][0], {"startMs": 420, "endMs": 580, "durationMs": 160,
                                            "source": "pcm-rms", "afterIndex": 0, "beforeIndex": 1})
        self.assertIsNone(graph["pauses"][1]["afterIndex"])

        overlapping = [segments[0], {**segments[1], "offsets": {"from": 400, "to": 920}}]
        graph = word_graph("Надёжный замок.", overlapping, quiet=[{"start": .42, "end": .58}])
        self.assertIsNone(graph["pauses"][0]["afterIndex"])

    def test_word_review_measures_a_short_pause_in_the_whisper_input(self):
        """The graph captures a short PCM pause after normalization, even below the transcript cutoff."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                voiced = b"\x01\x20" * 8000
                audio.writeframes(voiced + bytes(160 * 32) + voiced)
            segments = [
                {"text": "Лапши", "offsets": {"from": 0, "to": 500}},
                {"text": "друг", "offsets": {"from": 660, "to": 1160}},
            ]
            with patch('voice_studio.speech.recognition.recognize_file', return_value=segments) as recognize, \
                 patch('voice_studio.prosody.prosody.resources', return_value=({"лапши": "лапш+и"}, {}, {})):
                graph = review_words(root, "ru", "Лапши, друг!", source.read_bytes())
            self.assertEqual(recognize.call_args.args[0].name, "input.wav")
            self.assertEqual(graph["rows"][0]["stress"], {"index": 4, "source": "dictionary"})
            self.assertEqual(graph["pauses"][0]["durationMs"], 160)
            self.assertEqual(graph["pauses"][0]["beforeIndex"], 1)

    def test_excerpt_and_retry_keep_parent_engine_and_context(self):
        text = "Утром мы нашли пять ша́риков, одну чашку и ключ."
        self.assertEqual(fragment(text, 4), "нашли пять ша́риков, одну чашку")
        self.assertEqual(fragment(text, 4, radius=4), "Утром мы нашли пять ша́риков, одну чашку и ключ.")
        with TemporaryDirectory() as directory:
            studio = Studio(Path(directory), demo=True)
            try:
                reference = studio.record("ru", text, sample())
                parent, = studio.synthesize("ru", {"refs": [reference], "models": ["f5"], "text": text,
                                                  "comparison": reference})
                studio.worker.submit(lambda: None).result()
                with patch('voice_studio.prosody.revolver.recognize', return_value=([{"text": "ша́риков"}], [])):
                    self.assertEqual(studio.inspect_words("ru", parent)["version"], 2)
                children = studio.retry_words("ru", parent, 4, 2)
                studio.worker.shutdown()
                jobs = {item["id"]: item for item in studio.state("ru")["jobs"]}
                self.assertEqual(len({jobs[child]["seed"] for child in children}), 2)
                for key in children:
                    item = jobs[key]
                    self.assertEqual((item["status"], item["model"], item["variant"], item["baseline_id"]),
                                     ("done", "f5", "word-retry", parent))
                    self.assertEqual(item["text"], fragment(text, 4))
                    self.assertEqual(item["comparison_id"], reference)
                    self.assertEqual(item["retry_focus"], 2)
                    with patch.object(studio, "cached_word_graph", return_value={"wordErrorRate": 0.25}), \
                         patch.object(studio, "cached_fragment_score", return_value={"cosine": 0.8}) as score:
                        studio.score_fragment(key, "ru", item["run_id"])
                    self.assertEqual(score.call_args.args[-2:], (2, 2))
                    metrics = studio.tracking.client.get_run(item["run_id"]).data.metrics
                    self.assertAlmostEqual(metrics["eres2net_context_cosine"], 0.8)
            finally:
                studio.worker.shutdown()

    def test_mix_block_order_fade_and_bounds(self):
        parts = [{"id": "b", "startMs": 100, "endMs": 200}, {"id": "a", "startMs": 200, "endMs": 325}]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in [("a", 1200), ("b", -1200)]:
                with wave.open(str(root / f"{key}.wav"), "wb") as audio:
                    audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    audio.writeframes(value.to_bytes(2, "little", signed=True) * 24000)
            sources = {"a": {"seconds": 1}, "b": {"seconds": 1}}
            output = root / "mix.wav"
            self.assertAlmostEqual(render_blocks(checked_blocks(parts, sources), root, output), .225)
            with wave.open(str(output), "rb") as audio:
                pcm = array("h", audio.readframes(5400))
            self.assertEqual([pcm[i] for i in (0, 500, 2399, 2400, 2900, 5399)], [0, -1200, 0, 0, 1200, 0])
            with self.assertRaises(ValueError):
                checked_blocks([parts[0], {**parts[1], "endMs": 1500}], sources)
