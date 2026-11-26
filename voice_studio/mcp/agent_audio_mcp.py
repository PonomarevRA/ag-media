"""Publish bounded, pollable audio work to MCP clients."""

from voice_studio.audio.agent_audio import (Language, clean_audio, inspect_voice_quality,
                                           render_voice_restoration, separate_audio, separation_ready)
from voice_studio.audio.voice_quality import RestorationMode
from voice_studio.mcp.media_graph_mcp import GraphBuildQueue


def register_audio_tools(mcp, output):
    """Register serial audio jobs and return their queue to the owning MCP lifespan."""
    queue = GraphBuildQueue(result_key="result", label="дорожек")

    @mcp.tool()
    def audio_separate(file_path: str, start_ms: int = 0, end_ms: int | None = None,
                             language: Language = "auto", transcription: bool = True,
                             speakers: bool = False, export_cues: bool = False) -> dict:
        """Separate local audio or video into aligned dialogue, music and effects WAVs.

        Returns a jobId for audio_job. Dialogue cues use source milliseconds.
        Set export_cues=True for individual dialogue WAVs.
        Use an absolute path and select a range of up to 15 minutes on longer files.
        """
        if not separation_ready():
            raise ValueError("Установи MRX через scripts/install-mrx-audio.py в каталоге ag-media-studio.")
        if export_cues and not transcription:
            raise ValueError("Для отдельных WAV реплик включи транскрипцию.")
        return queue.submit(separate_audio, output, file_path, start_ms, end_ms, language,
                            transcription, speakers, export_cues)

    @mcp.tool()
    def audio_denoise(file_path: str, start_ms: int = 0, end_ms: int | None = None) -> dict:
        """Queue a non-destructive noise-reduced WAV and an aligned reference for A/B.

        Pass the dialogue WAV from audio_separate or a local media file.
        Timestamps are relative to file_path. Poll the returned jobId with audio_job.
        """
        return queue.submit(clean_audio, output, file_path, start_ms, end_ms)

    @mcp.tool()
    def audio_voice_quality(file_path: str, start_ms: int = 0, end_ms: int | None = None,
                            speech_intervals_ms: list[tuple[int, int]] | None = None) -> dict:
        """Measure speech level, quiet frames, relative frequency bands, and steady noise.

        Pass a local dialogue stem, WAV or video. Provide speaking intervals in
        file-relative milliseconds to measure noise in the gaps. With no intervals,
        the range is treated as dialogue and no steady-noise estimate is made.
        Poll audio_job for the measurements and recommendedModes.
        """
        return queue.submit(inspect_voice_quality, file_path, start_ms, end_ms, speech_intervals_ms)

    @mcp.tool()
    def audio_voice_restore(file_path: str, start_ms: int = 0, end_ms: int | None = None,
                            speech_intervals_ms: list[tuple[int, int]] | None = None,
                            modes: list[RestorationMode] | None = None) -> dict:
        """Render an aligned reference and a restored WAV with chosen corrections.

        Omit modes for automatic selection, or combine clarity, quiet-speech and
        denoise. Explicit denoise uses a conservative preset when no stable noise
        was measured. Poll audio_job for applied filters and WAV paths.
        """
        if modes is not None and (not modes or len(modes) != len(set(modes))):
            raise ValueError("Выбери один или несколько разных режимов обработки.")
        return queue.submit(render_voice_restoration, output, file_path, start_ms, end_ms,
                            speech_intervals_ms, modes)

    @mcp.tool()
    def audio_job(job_id: str) -> dict:
        """Read progress, measurements, or local outputs for an audio job."""
        return queue.status(job_id)

    return queue
