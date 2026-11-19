"""Exercise archive boundaries, Windows dependency markers and native binary identity without Windows."""

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import shutil
import subprocess
import os
import zlib
import unittest
import zipfile

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
sys.path.insert(0, str(PACKAGING))
SPEC = importlib.util.spec_from_file_location("media_build_windows", PACKAGING / "build_windows.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class WindowsPackagingTests(unittest.TestCase):
    """Protect real package inputs from traversal, architecture mistakes and host-platform resolution gaps."""

    def test_archive_paths_reject_windows_and_posix_escape(self):
        """Neither slash style nor drive-relative members may escape the package root."""
        for name in ("../secret", "/tmp/secret", "C:/secret", "folder\\secret", "folder/../../secret"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BUILDER.checked_member(name)
        self.assertEqual(str(BUILDER.checked_member("safe/module.py")), "safe/module.py")

    def test_wheel_data_layout_and_collision_are_checked(self):
        """Wheel platlib payloads land at import roots and never silently overwrite differing bytes."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "one.whl"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("demo.data/platlib/demo.py", "value = 1\n")
            BUILDER.extract_archive(archive, root / "libraries", wheel=True)
            self.assertEqual((root / "libraries/demo.py").read_text(), "value = 1\n")
            (root / "libraries/demo.py").write_text("unrelated\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                BUILDER.extract_archive(archive, root / "libraries", wheel=True)

    def test_windows_dependency_marker_is_evaluated_on_mac(self):
        """A Windows-only transitive requirement must fail even when this build runs on another OS."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_wheel(root, "demo", 'Requires-Dist: missing>=2; sys_platform == "win32"\n')
            with self.assertRaisesRegex(ValueError, "Unsatisfied Windows dependency"):
                BUILDER.validate_wheels(root)
            self.write_wheel(root, "missing", "", version="2.1")
            self.assertEqual(len(BUILDER.validate_wheels(root)), 2)

    def test_non_windows_wheel_and_wrong_binary_fail(self):
        """A macOS binary or wheel cannot enter an allegedly Windows x64 package."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "demo-1.0-cp313-cp313-macosx_11_0_arm64.whl").write_bytes(b"unused")
            with self.assertRaisesRegex(ValueError, "Non-Windows"):
                BUILDER.validate_wheels(root)
            executable = root / "wrong.exe"
            executable.write_bytes(b"\xcf\xfa\xed\xfe")
            with self.assertRaisesRegex(ValueError, "Not a Windows executable"):
                BUILDER.pe_machine(executable)
            content = bytearray(128)
            content[:2] = b"MZ"
            struct.pack_into("<I", content, 0x3C, 64)
            content[64:68] = b"PE\0\0"
            struct.pack_into("<H", content, 68, 0x8664)
            executable.write_bytes(content)
            self.assertEqual(BUILDER.pe_machine(executable), 0x8664)

    def test_binary_privacy_scan_rejects_split_marker_without_leaking_it(self):
        """A compiler identity crossing a scan-block boundary must fail with only its artifact path."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = b"private-build-account"
            path = root / "application.dll"
            path.write_bytes(b"x" * (1024 * 1024 - 5) + marker + b"tail")
            with self.assertRaisesRegex(ValueError, "application.dll") as caught:
                BUILDER.check_payload_privacy(root, markers=[marker])
            self.assertNotIn(marker.decode(), str(caught.exception))
            path.write_bytes(b"RSDS/_/normalized-build/application.pdb")
            self.assertEqual(BUILDER.check_payload_privacy(root, markers=[marker]), 1)

    def test_embedded_debug_metadata_is_privacy_checked(self):
        """Compression must not hide an identity-bearing compiler document from the package gate."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = b"private-build-account"
            symbols = b"document/" + marker + b"/OwnedBackend.cs"
            compressed = zlib.compressobj(wbits=-15)
            payload = compressed.compress(symbols) + compressed.flush()
            (root / "AgMediaStudio.dll").write_bytes(b"MZ...MPDB" + struct.pack("<I", len(symbols)) + payload)
            with self.assertRaisesRegex(ValueError, "embedded debug metadata"):
                BUILDER.check_payload_privacy(root, markers=[marker])

    @unittest.skipUnless(shutil.which("dotnet"), "The pure Windows-origin policy check requires the .NET SDK")
    def test_actual_csharp_origin_and_microphone_policy(self):
        """Run the actual UI-independent C# policy for owned navigation and microphone-consent boundaries."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(PACKAGING / "windows/OriginPolicy.cs", root / "OriginPolicy.cs")
            (root / "PolicyCheck.csproj").write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>')
            (root / "Program.cs").write_text('''using AgMediaStudio;
var owned = new Uri("http://127.0.0.1:18766");
if (!OriginPolicy.IsOwned(owned, "http://127.0.0.1:18766/editor")) throw new Exception("Owned navigation rejected");
if (!OriginPolicy.MayAskMicrophone(owned, "http://127.0.0.1:18766", true)) throw new Exception("Microphone consent unavailable");
if (OriginPolicy.MayAskMicrophone(owned, "http://127.0.0.1:18766", false)) throw new Exception("Non-microphone consent allowed");
foreach (string other in new[] { "https://127.0.0.1:18766", "http://127.0.0.1:18767", "http://localhost:18766", "http://example.invalid:18766", "http://user@127.0.0.1:18766", "file:///local", "invalid" }) {
    if (OriginPolicy.IsOwned(owned, other) || OriginPolicy.MayAskMicrophone(owned, other, true)) throw new Exception("Untrusted request allowed");
}
if (OriginPolicy.MayAskMicrophone(null, "http://127.0.0.1:18766", true)) throw new Exception("Missing owner allowed");
Console.WriteLine("origin-and-microphone-policy:PASS");
''')
            environment = {**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_CLI_HOME": str(root / "dotnet-home")}
            result = subprocess.run(["dotnet", "run", "--project", str(root / "PolicyCheck.csproj"), "--configuration", "Release", "--verbosity", "quiet"], text=True, capture_output=True, env=environment, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("origin-and-microphone-policy:PASS", result.stdout)

    @staticmethod
    def write_wheel(root, name, requirements, version="1.0"):
        """Create only the metadata needed by dependency-contract tests."""
        filename = root / f"{name}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(filename, "w") as package:
            package.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requirements}\n")


if __name__ == "__main__":
    unittest.main()
