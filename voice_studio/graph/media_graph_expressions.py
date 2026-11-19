"""Keep facial-expression classifier evidence queryable with explicit temporal provenance."""

from collections import defaultdict

from voice_studio.vision.facial_expressions import finalize_expressions


def add_facial_expressions(graph, report, video_track):
    """Add classifier measurements and question windows without creating speaker assignments."""
    evidence = report.get("videoEvidence") or {}
    if not isinstance(evidence.get("facialExpressions"), dict):
        return
    # Recompute from checked measurements and geometry rather than trust imported summary totals.
    checked = dict(evidence)
    layer = finalize_expressions(checked, report.get("segments") or [], graph.duration)
    provenance = {key: layer[key] for key in ("basis", "model", "modelSha256", "confidenceThreshold", "sampleStepMs",
                                            "probabilityBasis", "temporalSupport")}
    summary = {key: value for key, value in layer["statistics"].items() if key != "faces"}
    if video_track:
        graph.nodes[video_track]["data"]["facialExpressions"] = {
            **provenance, "status": layer["status"], "statistics": summary}
    for facts in layer["statistics"]["faces"]:
        face = graph.face(facts["faceId"])
        graph.nodes[face]["data"]["facialExpressions"] = facts
    samples = {}
    for row in layer["samples"]:
        stamp, track, face = row["timeMs"], row["track"], row["faceId"]
        key = f"expression_sample:{stamp}:{track}"
        graph.node(key, "expression_sample", row["label"], (stamp, stamp + 1), {**provenance, **row})
        graph.connect(key, "observed_face", graph.face(face))
        graph.connect(key, "at_frame", graph.frame(stamp, "facial-expression-classifier"))
        samples[stamp, track] = key
    for row in layer["intervals"]:
        start, end, track = row["startMs"], row["endMs"], row["track"]
        key = f"facial_expression:{start}:{track}"
        graph.node(key, "facial_expression", row["label"], (start, end), {**provenance, **row})
        graph.connect(key, "observed_face", graph.face(row["faceId"]))
        graph.connect(key, "supported_by", samples[start, track])
        if video_track:
            graph.connect(key, "on_track", video_track)
        graph.link_shots(key, start, end)
    utterances = defaultdict(list)
    for node in graph.nodes.values():
        if node["kind"] == "utterance":
            utterances[node["startMs"], node["endMs"], node["label"]].append(node["id"])
    for row in layer["reactions"]:
        start, end = row["windowStartMs"], row["windowEndMs"]
        key = f"expression_question_window:{row['questionCue']}:{start}"
        graph.node(key, "expression_question_window", row["text"], (start, end), {**provenance, **row})
        graph.link_shots(key, start, end)
        if video_track:
            graph.connect(key, "on_track", video_track)
        # Link by cue time and text; temporal context is not a causal claim about the question.
        cue_key = row["questionStartMs"], row["questionEndMs"], row["text"]
        for utterance in utterances.get(cue_key, ()):
            graph.connect(key, "question_context", utterance)
        for face in row["faces"]:
            graph.connect(key, "visible_during_window", graph.face(face["faceId"]))
