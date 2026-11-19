"""Prepare local transcription from a checkout, then open the Studio."""

import argparse
import os
from pathlib import Path
from shutil import which
import subprocess
import sys
import venv

from voice_paths import PROJECT_ROOT, application_data
from voice_studio.speech.whisper_assets import install_base, whisper_cli


def missing_tools():
    """Report which executables the Studio will need for media transcription."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not which(tool)]
    cli = whisper_cli()
    if not cli.is_file() or not os.access(cli, os.X_OK):
        missing.append("whisper-cli")
    return missing


def prepare_tools():
    """Reuse available tools and install only missing dependencies through Homebrew."""
    if sys.platform == "darwin":
        system_bins = ("/opt/homebrew/bin", "/usr/local/bin")
        os.environ["PATH"] = os.pathsep.join((os.environ.get("PATH", ""), *system_bins)).strip(os.pathsep)
    missing = missing_tools()
    brew = which("brew") if missing and sys.platform == "darwin" else None
    if brew:
        packages = []
        if "ffmpeg" in missing or "ffprobe" in missing:
            packages.append("ffmpeg")
        if "whisper-cli" in missing and not os.environ.get("VOICE_WHISPER_CLI"):
            packages.append("whisper.cpp")
        if packages:
            subprocess.run([brew, "install", *packages], check=True)
            missing = missing_tools()
    if missing:
        raise ValueError(f"Нужны программы {', '.join(missing)}. См. docs/GETTING-STARTED.md.")


def prepare_python():
    """Install the core project dependencies in the checkout's dedicated environment."""
    environment = PROJECT_ROOT / ".venv"
    relative = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = environment / relative
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
    pip = [str(python), "-m", "pip", "--disable-pip-version-check"]
    if subprocess.run([*pip, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run([str(python), "-m", "ensurepip"], check=True)
    subprocess.run([*pip, "install", "-r", str(PROJECT_ROOT / "requirements-core.txt")], check=True)
    return python


def prepare_weights():
    """Show download progress and the persistent path used for the next launch."""
    last_percent = -10

    def progress(received, total):
        nonlocal last_percent
        percent = received * 100 // total
        if percent >= last_percent + 10 or percent == 100:
            print(f"Whisper Base · {percent}%", flush=True)
            last_percent = percent

    path, installed = install_base(progress)
    action = "Модель загружена" if installed else "Используется установленная модель"
    print(f"{action} · {path}", flush=True)


def main():
    """Prepare requested resources and keep the Studio in this terminal session."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Set up dependencies without launching the app.")
    modes.add_argument("--models-only", action="store_true", help="Download Whisper Base without installing tools.")
    parser.add_argument("--data", type=Path, default=application_data() / "studio")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local URL automatically.")
    options = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Требуется Python 3.12 или новее.")
    try:
        if options.models_only:
            prepare_weights()
            return 0
        prepare_tools()
        python = prepare_python()
        prepare_weights()
        if options.prepare_only:
            return 0
        command = [str(python), str(PROJECT_ROOT / "ag_media_studio.py"), "--data", str(options.data), "--port", "0"]
        if not options.no_browser:
            command.append("--open-browser")
        return subprocess.call(command)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Настройка студии · {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
