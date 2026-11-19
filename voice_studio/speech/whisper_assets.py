"""Resolve and provision the multilingual Whisper Base runtime assets."""

from hashlib import file_digest, sha256
from http.client import HTTPException
import os
from pathlib import Path
import re
from shutil import which
from ssl import SSLError
import sys
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import Request, urlopen

from voice_paths import PROJECT_ROOT, dependency_path, model_path


BASE_NAME = "ggml-base.bin"
BASE_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
BASE_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{BASE_REVISION}/{BASE_NAME}"
BASE_BYTES = 147_951_465
BASE_SHA256 = "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"


def whisper_cli():
    """Honor explicit and legacy locations, then find a provisioned or system CLI."""
    override = os.environ.get("VOICE_WHISPER_CLI")
    if override:
        return Path(override).expanduser()
    filename = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
    provisioned = dependency_path("whisper", filename)
    legacy = PROJECT_ROOT / ".engines" / "whisper" / "build-arm" / "bin" / filename
    for path in (provisioned, legacy):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    system = which(filename)
    return Path(system).absolute() if system else legacy


def whisper_base_model():
    """Use the same path for setup and inference, including operator overrides."""
    override = os.environ.get("VOICE_WHISPER_MODEL")
    return Path(override).expanduser() if override else model_path("whisper", BASE_NAME)


def verify_base(path):
    """Accept the pinned upstream file and preserve any pre-existing different file."""
    if path.is_file() and path.stat().st_size == BASE_BYTES:
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() == BASE_SHA256:
                return
    raise ValueError(f"Файл {path} отличается от Whisper Base. Укажите путь к своей модели или освободите этот путь.")


def resume_response(source, stream, received):
    """Require the exact requested range or restart if the server responds with a full file."""
    if source.status == 200:
        stream.seek(0)
        stream.truncate()
        return False
    content_range = source.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range)
    if source.status != 206 or not match:
        raise ValueError("Сервер не подтвердил позицию возобновления Whisper Base.")
    start, end, total = map(int, match.groups())
    if start != received or not start <= end < total or total != BASE_BYTES:
        raise ValueError("Сервер вернул несовместимый диапазон Whisper Base.")
    return True


def fetch_base(stream, progress):
    """Resume interrupted transfers within this run and verify complete weights."""
    digest = sha256()
    received = 0
    for attempt in range(3):
        headers = {"Range": f"bytes={received}-"} if received else {}
        request = Request(BASE_URL, headers=headers)
        try:
            with urlopen(request, timeout=45) as source:
                if received and not resume_response(source, stream, received):
                    digest = sha256()
                    received = 0
                while chunk := source.read(1024 * 1024):
                    received += len(chunk)
                    if received > BASE_BYTES:
                        raise ValueError("Сервер вернул слишком большой файл Whisper Base.")
                    digest.update(chunk)
                    stream.write(chunk)
                    progress(received, BASE_BYTES)
            if received < BASE_BYTES:
                raise ConnectionError("Скачивание прервано до конца файла.")
        except (ConnectionError, TimeoutError, HTTPException, URLError, SSLError) as error:
            if attempt == 2:
                raise ValueError("Скачивание Whisper Base прервано. Проверьте подключение и повторите запуск.") from error
            continue
        if digest.hexdigest() != BASE_SHA256:
            raise ValueError("Контрольная сумма Whisper Base не совпала.")
        return


def download_base(target, progress):
    """Publish only verified weights and leave the previous file intact."""
    staging = NamedTemporaryFile(dir=target.parent, prefix=".whisper-base-", delete=False)
    staged_path = Path(staging.name)
    try:
        with staging:
            fetch_base(staging, progress)
        try:
            os.link(staged_path, target)
        except FileExistsError:
            verify_base(target)
            return False
        return True
    finally:
        staged_path.unlink(missing_ok=True)


def install_base(progress):
    """Reuse verified weights or download the selected model once by default."""
    target = whisper_base_model()
    if os.environ.get("VOICE_WHISPER_MODEL"):
        if not target.is_file():
            raise FileNotFoundError(f"Модель из VOICE_WHISPER_MODEL недоступна. {target}")
        return target, False
    if target.exists() or target.is_symlink():
        verify_base(target)
        return target, False
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return target, download_base(target, progress)
