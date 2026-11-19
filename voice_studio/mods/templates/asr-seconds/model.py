"""Synthetic vendor response fixture; replace run with an already installed local model."""


def run(payload, settings):
    """Return vendor-shaped timestamps in seconds, once for each prepared audio job."""
    results = []
    for audio in payload["audio"]:
        results.append({"language": payload["language"], "utterances": [
            {"begin_seconds": .125, "end_seconds": 1.375, "content": "Example vendor transcript"}]})
    return results
