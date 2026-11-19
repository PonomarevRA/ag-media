"""Verify the offline documentation build and its narrow HTTP file boundary."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.app import wiki_assets

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("build_wiki", ROOT / "scripts/build_wiki.py")
builder = module_from_spec(spec)
spec.loader.exec_module(builder)


class WikiTests(TestCase):
    """Keep generated files, source coverage and escaped prose independently checkable."""

    def test_generated_outputs_and_module_coverage_are_current(self):
        """A source/doc change must be followed by a deterministic rebuild."""
        files = builder.render_all()
        for name, expected in files.items():
            with self.subTest(name=name):
                self.assertEqual((ROOT / "docs/wiki" / name).read_bytes(), expected)
        index = json.loads(files["module-index.json"])["modules"]
        actual = {row["path"] for row in index}
        expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / "voice_studio").rglob("*.py") if path.name != "__init__.py"}
        expected.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "web").glob("*.js"))
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(index))

    def test_html_and_attribute_content_is_escaped(self):
        """Prose and code are data, including script-looking text."""
        html, _ = builder.block_markup({"type": "p", "text": '<script>alert("x")</script> `a < b`'}, set(), set())
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<code>a &lt; b</code>", html)
        with self.assertRaises(ValueError):
            builder.block_markup({"type": "links", "items": [{"target": "javascript:alert(1)", "label": "bad"}]}, set(), set())
        with self.assertRaises(ValueError):
            builder.block_markup({"type": "links", "items": [{"target": "missing-page", "label": "bad"}]}, set(), set())

    def test_browser_responsibility_is_not_an_arbitrary_function_comment(self):
        """A new browser file needs an explicit module description, not a nearby function summary."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "voice_studio").mkdir()
            (root / "web").mkdir()
            (root / "web/app.js").write_text('/** Format a timestamp. */\nfunction format() {}')
            with patch.object(builder, "ROOT", root):
                with self.assertRaises(ValueError):
                    builder.module_index({})
                rows = builder.module_index({"app.js": {"ru": "Управление приложением.", "en": "Application orchestration."}})
            self.assertEqual(rows[0]["summary"], "Application orchestration.")
            self.assertEqual(rows[0]["summaryOrigin"], "reviewed-wiki-module-responsibility")

    def test_schematics_reject_embedded_or_active_media(self):
        """Illustrations cannot silently become a real frame or an executable SVG."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "schematic.svg"
            for fragment in ['<image href="photo.png"/>', '<script/>', '<rect onclick="x()"/>', '<foreignObject/>']:
                path.write_text('<svg xmlns="http://www.w3.org/2000/svg">' + fragment + '</svg>')
                with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                    builder.safe_svg(path)

    def test_asset_boundary_rejects_unregistered_and_symlink_paths(self):
        """The docs endpoint cannot expose source, models, recordings or parent directories."""
        self.assertEqual(wiki_assets.wiki_asset("/wiki", "en")[0].name, "Home.html")
        self.assertEqual(wiki_assets.wiki_asset("/wiki", "ru")[0].name, "Home-ru.html")
        for url in ["/wiki/../README.md", "/wiki/content.json", "/wiki/%2e%2e/README.md", "/wiki/source/secret", "/wiki//Home.html"]:
            with self.subTest(url=url):
                self.assertIsNone(wiki_assets.wiki_asset(url))
        with TemporaryDirectory() as folder:
            base = Path(folder)
            site = base / "site"
            site.mkdir()
            (base / "private.txt").write_text("private")
            (site / "linked.html").symlink_to(base / "private.txt")
            (site / "manifest.json").write_text(json.dumps({"linked.html": "text/html", "../private.txt": "text/plain"}))
            with patch.object(wiki_assets, "SITE", site):
                self.assertIsNone(wiki_assets.wiki_asset("/wiki/linked.html"))
                self.assertIsNone(wiki_assets.wiki_asset("/wiki/../private.txt"))
