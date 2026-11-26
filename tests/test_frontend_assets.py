"""Guard the Studio entry module against unserved browser dependencies."""

import posixpath
import re
from unittest import TestCase

from voice_studio.app.static_assets import ROOT, STATIC

IMPORT = re.compile(r"""\b(?:from\s+|import\s+)["'](\.?\.?/[^"']+\.js)["']""")


class FrontendAssetsTests(TestCase):
    def test_all_startup_modules_have_javascript_routes(self):
        pending = ["/app.js"]
        checked = set()
        while pending:
            route = pending.pop()
            if route in checked:
                continue
            checked.add(route)
            self.assertIn(route, STATIC, f"Unserved Studio module: {route}")
            path, mime = STATIC[route]
            self.assertTrue(mime.startswith("text/javascript"), route)
            source = (ROOT / "web" / path).read_text(encoding="utf-8")
            for dependency in IMPORT.findall(source):
                pending.append(posixpath.normpath(posixpath.join(posixpath.dirname(route), dependency)))
