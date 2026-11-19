"""Publish bounded, pollable audio work to MCP clients."""

from voice_studio.audio.agent_audio import Language, clean_audio, separate_audio, separation_ready
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
    def audio_job(job_id: str) -> dict:
        """Read status and local WAV paths for a separation or denoise job."""
        return queue.status(job_id)

    return queue
