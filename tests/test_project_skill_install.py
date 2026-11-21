"""Exercise installation into a separate user project from staged desktop files."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]


class ProjectSkillInstallTests(unittest.TestCase):
    """Validate project scoping and preservation without creating real user notes."""

    def setUp(self):
        """Stage the same skill payload shared by the three desktop packagers."""
        self.scratch = Path(self.enterContext(TemporaryDirectory()))
        self.staged = self.scratch / "bundled-app"
        with patch.object(sys, "path", [str(REPOSITORY / "packaging"), *sys.path]):
            from build_native import copy_agent_skills
        copy_agent_skills(REPOSITORY, self.staged)
        self.project = self.scratch / "user-project"
        self.project.mkdir()

    def install(self, *options, expected_code=0):
        """Run an isolated installer process from outside both directories."""
        script = self.staged / "skills/install.py"
        result = subprocess.run(
            (sys.executable, "-B", str(script), "--project", str(self.project), *options),
            cwd=self.scratch, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, expected_code, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_package_installs_selected_clients_and_preserves_project_contents(self):
        """Preview writes nothing and subsequent runs preserve existing findings and skills."""
        self.assertEqual(set(self.install("--client", "all", "--check").values()), {"would-create"})
        self.assertEqual(list(self.project.iterdir()), [])
        self.assertEqual(set(self.install("--client", "all").values()), {"created"})
        for client in (".agents", ".claude", ".cursor"):
            skills = self.project / client / "skills"
            self.assertTrue((skills / "agent-memory/references/ag-media-analysis.md").is_file())
            self.assertTrue((skills / "ag-media-interval-review/SKILL.md").is_file())
        local = self.project / ".ag-media-memory"
        subprocess.run(("git", "init", "-q", str(self.project)), check=True)
        subprocess.run(("git", "check-ignore", "--quiet", ".ag-media-memory/registry.a2a.yaml"),
                       cwd=self.project, check=True)
        existing = local / "example.md"
        existing.write_text("User note.", encoding="utf-8")
        edited = self.project / ".agents/skills/ag-media-ui-review/SKILL.md"
        edited.write_text("User scenario.", encoding="utf-8")
        result = self.install()
        self.assertEqual(result[".agents/skills/ag-media-ui-review"], "preserved")
        self.assertEqual(result[".agents/skills/agent-memory"], "current")
        self.assertEqual(existing.read_text(encoding="utf-8"), "User note.")
        self.assertEqual(edited.read_text(encoding="utf-8"), "User scenario.")

    def test_redirected_client_path_writes_nothing_outside_project(self):
        """An existing project client symlink cannot redirect skill writes."""
        outside = self.scratch / "outside"
        outside.mkdir()
        try:
            (self.project / ".agents").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Creating directory symlinks is unavailable")
        result = self.install(expected_code=1)
        self.assertEqual(result[".agents/skills/agent-memory"], "conflict")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.project / ".claude").exists())
