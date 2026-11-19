"""Explicit format conversion; model inference and both hooks share the same bounded worker."""


def prepare_input(payload, settings):
    """Translate Studio named ASR jobs into a vendor audio-list request without loading a model."""
    audio = []
    for index, job in enumerate(payload.get("jobs", [])):
        if not isinstance(job, list) or len(job) != 2 or not isinstance(job[0], str):
            raise ValueError(f"provider input.jobs[{index}]: expected [audioPath, outputPrefix]")
        audio.append({"uri": job[0]})
    return {"audio": audio, "language": payload.get("language", "auto")}


def convert_output(payload, settings):
    """Convert validated vendor seconds to integer Studio offsets; preserve ordering for validation."""
    results = []
    for vendor in payload:
        transcription = []
        for utterance in vendor["utterances"]:
            transcription.append({"text": utterance["content"], "offsets": {
                "from": round(utterance["begin_seconds"] * 1000),
                "to": round(utterance["end_seconds"] * 1000)}})
        results.append({"transcription": transcription, "result": {"language": vendor["language"]}})
    return results
