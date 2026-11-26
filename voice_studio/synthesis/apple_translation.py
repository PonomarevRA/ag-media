"""Run the macOS Translation framework with installed language assets only."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock

from voice_paths import PROJECT_ROOT, RUNTIME_DIR, ensure_temporary_root

MOD_ID = "builtin.apple-translation"
CAPABILITY = "translation.text"
HELPER = "apple-translation"
SWIFT_SOURCE = Path(__file__).with_name("native") / "apple_translation.swift"
COMPILE_LOCK = Lock()
ERRORS = {
    "unavailable": "Перевод Apple доступен в macOS 26 и новее с установленным адаптером студии.",
    "unsupported-pair": "Перевод Apple не поддерживает выбранную пару языков.",
    "needs-languages": "Загрузи оба языка в macOS: Системные настройки → Основные → Язык и регион → Языки перевода.",
    "failed": "Системный перевод macOS не завершился. Повтори попытку.",
}


def supported_platform():
    """Check only the OS version, without starting or provisioning the translator."""
    return sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0] or 0) >= 26


def bundled_helper():
    """Locate the prebuilt signed adapter next to the bundled Studio source."""
    bundle = PROJECT_ROOT.parent
    if bundle.name == "Resources" and PROJECT_ROOT.name == "studio":
        binary = bundle / "translation" / HELPER
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    return None


def adapter_available():
    """Expose static availability without compiling or asking Apple about languages."""
    return supported_platform() and bool(bundled_helper() or (SWIFT_SOURCE.is_file() and shutil.which("swiftc")))


def native_helper():
    """Use the bundled executable or build once in the private source-development cache."""
    if binary := bundled_helper():
        return binary
    swift = shutil.which("swiftc")
    if not swift or not SWIFT_SOURCE.is_file():
        raise ValueError(ERRORS["unavailable"])
    ensure_temporary_root()
    cache = RUNTIME_DIR / "translation"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    version = sha256(SWIFT_SOURCE.read_bytes()).hexdigest()[:16]
    target = cache / f"{HELPER}-{version}"
    with COMPILE_LOCK:
        if not target.is_file():
            try:
                with TemporaryDirectory(dir=cache) as folder:
                    output = Path(folder) / HELPER
                    subprocess.run([swift, "-O", "-parse-as-library", "-module-cache-path", str(cache / "swift-modules"),
                                    str(SWIFT_SOURCE), "-o", str(output)],
                                   capture_output=True, check=True, timeout=45)
                    output.replace(target)
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(ERRORS["unavailable"]) from error
    return target


def request_translation(payload):
    """Read status or translate text over stdin without writing any media or transcripts."""
    if not supported_platform():
        return {"status": "unavailable"}
    try:
        executable = native_helper()
    except (OSError, ValueError):
        return {"status": "unavailable"}
    try:
        output = subprocess.run([str(executable)], input=json.dumps(payload).encode(),
                                capture_output=True, check=True, timeout=180 if payload["texts"] else 15)
        reply = json.loads(output.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "failed"}
    return reply if isinstance(reply, dict) else {"status": "failed"}
