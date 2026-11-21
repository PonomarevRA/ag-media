"""Coordinate explicit builtin model installs requested from Studio settings."""

from dataclasses import dataclass
import os
import platform
import subprocess
import sys
from threading import Lock, Thread

from voice_paths import MODELS_DIR, dependency_path
from voice_studio.app.first_run import missing_tools, prepare_tools
from voice_studio.core.optional_runtime import runtime_environment, runtime_ready, sherpa_ready
from voice_studio.core.worker_process import worker_command
from voice_studio.core.workspace_lock import workspace_lock
from voice_studio.provision.assets import DUB_BACKGROUND, LIGHT_ASD, MEDIAPIPE, SOUNDS, SPEAKERS, VISION, install_assets
from voice_studio.provision.runtime import install_runtime
from voice_studio.speech.whisper_assets import install_base, whisper_base_model


@dataclass(frozen=True)
class ModelGroup:
    """Describe a task and its files without accepting caller-supplied commands."""

    assets: tuple = ()
    runtime: str = ""
    voice: bool = False


GROUPS = {
    "transcription": ModelGroup(),
    "speakers": ModelGroup(SPEAKERS, "sherpa"),
    "sounds": ModelGroup(SOUNDS, "sherpa"),
    "vision": ModelGroup(VISION, "vision"),
    "mediapipe": ModelGroup(MEDIAPIPE, "mediapipe"),
    "light-asd": ModelGroup(LIGHT_ASD, "light-asd"),
    "dub-background": ModelGroup(DUB_BACKGROUND, "dub-background"),
    "qwen": ModelGroup(runtime="mlx", voice=True),
    "chatterbox": ModelGroup(runtime="mlx", voice=True),
}


def supported(group):
    """Use the platforms with binary wheels for each pinned runtime."""
    machine = platform.machine().lower()
    major = platform.mac_ver()[0].split(".")[0]
    apple_silicon = sys.platform == "darwin" and machine == "arm64" and major.isdecimal()
    if group.runtime == "light-asd":
        if sys.version_info[:2] not in {(3, 12), (3, 13)}:
            return False
        return ((apple_silicon and int(major) >= 14)
                or (sys.platform in {"linux", "win32"} and machine in {"x86_64", "amd64"}))
    if group.runtime == "mediapipe":
        return apple_silicon or (sys.platform in {"linux", "win32"} and machine in {"x86_64", "amd64"})
    if group.runtime == "dub-background":
        return sys.version_info[:2] in {(3, 12), (3, 13)} and apple_silicon and int(major) >= 26
    return not group.voice or (apple_silicon and int(major) >= 26)


def group_ready(key, group):
    """Describe actual file and runtime prerequisites for the normal inference path."""
    if key == "transcription":
        return whisper_base_model().is_file() and not missing_tools() and initial_background_ready()
    if group.voice:
        from voice_studio.synthesis.commercial_voice import ready
        return ready(key) and (key != "qwen" or initial_background_ready())
    if key == "dub-background":
        from voice_studio.synthesis.dub_background import separator_ready
        return separator_ready() and all(asset.present() for asset in group.assets)
    if key == "light-asd" and not group_ready("vision", GROUPS["vision"]):
        return False
    runtime = sherpa_ready() if group.runtime == "sherpa" else runtime_ready(group.runtime)
    if group.runtime == "vision" and not runtime:
        from voice_studio.vision.video_revolver import LEGACY_PYTHON, VISION as legacy_vision
        runtime = LEGACY_PYTHON.is_file() and (legacy_vision / "cv2").is_dir()
    return runtime and all(asset.present() for asset in group.assets)


def initial_background_ready():
    """Require the dubbing background pack only on supported platforms."""
    group = GROUPS["dub-background"]
    return not supported(group) or group_ready("dub-background", group)


def install_initial_background(progress):
    """Provision HDemucs for supported users as part of their first model install."""
    if not initial_background_ready():
        install_group("dub-background", GROUPS["dub-background"], progress)


def install_group(key, group, progress):
    """Install the capability's dependencies and weight files in dependency order."""
    if key == "transcription":
        progress("runtime")
        prepare_tools()
        install_base(lambda received, total: progress("weights", received, total))
        install_initial_background(progress)
        return
    if key == "light-asd" and not group_ready("vision", GROUPS["vision"]):
        install_group("vision", GROUPS["vision"], progress)
    if group.runtime != "sherpa" or not sherpa_ready():
        progress("runtime")
        install_runtime(group.runtime)
    if group.voice:
        progress("weights")
        env = runtime_environment("mlx")
        env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1")
        env.pop("HF_TOKEN", None)
        subprocess.run(worker_command(sys.executable, "voice_studio.synthesis.setup_commercial_voices", key),
                       env=env, check=True, timeout=1800)
        if key == "qwen":
            install_initial_background(progress)
    else:
        install_assets(group.assets, lambda name, received, total: progress("weights", received, total, name))


class ModelInstalls:
    """Keep one background installation and its visible status per studio process."""

    def __init__(self):
        self._state = {}
        self._active = None
        self._lock = Lock()

    def inventory(self):
        """Read statuses without opening model weights or contacting providers."""
        with self._lock:
            items = []
            for key, group in GROUPS.items():
                row = {"id": key, "supported": supported(group), "ready": group_ready(key, group),
                       **self._state.get(key, {})}
                items.append(row)
            return {"active": self._active, "items": items}

    def start(self, key):
        """Start only a known builtin group and coalesce a repeated click."""
        if not isinstance(key, str) or key not in GROUPS:
            raise ValueError("Неизвестный набор моделей.")
        group = GROUPS[key]
        if not supported(group):
            raise ValueError("Набор не поддерживается на этой платформе.")
        with self._lock:
            if self._active == key or group_ready(key, group):
                return
            if self._active:
                raise ValueError("Дождитесь завершения текущей установки.")
            self._active = key
            self._state[key] = {"stage": "runtime"}
        Thread(target=self._run, args=(key, group), name=f"model-install-{key}", daemon=True).start()

    def _run(self, key, group):
        """Publish stage updates and release the slot after either result."""
        def progress(stage, received=None, total=None, artifact=None):
            with self._lock:
                self._state[key] = {"stage": stage, "received": received, "total": total, "artifact": artifact}

        error = None
        try:
            guard = dependency_path("model-install")
            guard.mkdir(mode=0o700, parents=True, exist_ok=True)
            with workspace_lock(guard):
                install_group(key, group, progress)
            if not group_ready(key, group):
                raise ValueError("Проверка готовности модели после установки не прошла.")
        except BlockingIOError:
            error = "Установка уже выполняется в другом окне студии."
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            print(f"Model install {key}: {exc}", flush=True)
            error = str(exc)[:300] if isinstance(exc, ValueError) else "Установка прервалась. Повторите попытку."
        except Exception as exc:
            print(f"Model install {key}: {exc}", flush=True)
            error = "Установка прервалась. Повторите попытку."
        finally:
            with self._lock:
                self._state[key] = {"error": error} if error else {}
                self._active = None
