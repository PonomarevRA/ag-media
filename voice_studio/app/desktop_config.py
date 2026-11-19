"""Explicit native-instance discovery and MCP launch configuration, with no development-port fallback."""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

from voice_paths import application_data


LINUX_PACKAGE_SOURCE = Path("/opt/ag-media-studio/studio")
ADD_IMPORT_PATH = "sys.path.insert(0,sys.argv.pop(1));"
RUN_CONFIG = "runpy.run_module('voice_studio.app.desktop_config',run_name='__main__')"


def instance(root):
    """Require matching live metadata and the nonce returned by this instance's owned origin.

    root is an external workspace. Missing, stale, malformed or mismatched metadata
    raises ValueError. No request is ever sent to the development default port.
    """
    root = Path(root).expanduser().resolve()
    try:
        value = json.loads((root / "desktop-instance.json").read_text())
        port = value["port"]
        pid = value["pid"]
        if value.get("schema") != "ag-media-desktop-ready-v1" or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid native instance metadata")
        if type(pid) is not int or pid <= 0:
            raise ValueError("Invalid native process identity")
        expected = f"http://127.0.0.1:{port}"
        if value.get("origin") != expected or not isinstance(value.get("instanceId"), str):
            raise ValueError("Invalid native origin or instance identity")
        if os.name != "nt":
            os.kill(pid, 0)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(expected + "/api/desktop-instance", timeout=3) as response:
            content = response.read(16385)
        if len(content) > 16384:
            raise ValueError("Invalid native instance response")
        live = json.loads(content)
        if any(live.get(key) != value.get(key) for key in ("schema", "pid", "origin", "port", "instanceId")):
            raise ValueError("Native instance identity does not match")
        if Path(live["data"]).resolve() != root:
            raise ValueError("Native instance workspace does not match")
        return live
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Open this workspace in the native app before using its HTTP-backed MCP tools") from error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep instance validation on the one literal loopback origin."""

    def redirect_request(self, *args, **kwargs):
        """Reject all redirect targets rather than trusting a second server identity."""
        return None


def configuration(root, live):
    """Return copyable stdio configuration whose mutable paths are all outside the app bundle."""
    source = Path(__file__).resolve().parents[2]
    vendor = source.parent / "python"
    import_paths = [str(source)]
    # The Debian launcher adds vendored dependencies ahead of system packages.
    if sys.platform == "linux" and source == LINUX_PACKAGE_SOURCE and vendor.is_dir() and not vendor.is_symlink():
        import_paths.insert(0, str(vendor))
    bootstrap = "import runpy,sys;" + ADD_IMPORT_PATH * len(import_paths) + RUN_CONFIG
    return {"mcpServers": {"ag-media-studio": {
        "command": sys.executable,
        "args": ["-I", "-B", "-c", bootstrap, *import_paths, "--mcp", "--data", str(root)],
        "env": {key: os.environ[key] for key in ("VOICE_MODELS_DIR", "VOICE_DEPENDENCIES_DIR", "VOICE_TEMP_DIR") if key in os.environ},
    }}, "instance": {key: live[key] for key in ("schema", "pid", "port", "origin", "instanceId")},
        "paths": {"studio": str(root), "graphs": str(root / "media-graphs"), "output": str(root / "mcp-output")}}


def main():
    """Print validated configuration or attach the existing MCP server to that exact GUI instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("VOICE_STUDIO_DATA", application_data() / "studio")))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mcp", action="store_true")
    mode.add_argument("--print-config", action="store_true")
    options = parser.parse_args()
    root = options.data.expanduser().resolve()
    try:
        live = instance(root)
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    if options.print_config:
        print(json.dumps(configuration(root, live), indent=2))
        return
    from voice_studio.mcp.voice_mcp import StudioClient, build_server
    output = root / "mcp-output"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    build_server(StudioClient(live["port"], output), studio_root=root, graph_root=root / "media-graphs").run(transport="stdio")


if __name__ == "__main__":
    main()
