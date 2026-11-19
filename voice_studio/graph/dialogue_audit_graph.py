"""Publish vector-free voice identities and review-required dialogue issues beside raw evidence."""

from voice_studio.speech.voice_profile_public import public_voice_profile, public_voice_evidence


def add_dialogue_review(graph, report, source_interval):
    """Add source-clock references and structural findings; never turn a candidate into a fact."""
    profiles = (report.get('voiceProfiles') or {}).get('profiles') or []
    for row in profiles[:64]:
        profile = public_voice_profile(row)
        if not profile:
            continue
        key = 'voice_profile:' + profile['id']
        graph.node(key, 'voice_profile', 'Voice profile ' + str(profile.get('number')), (0, graph.duration), profile)
        if type(profile.get('number')) is int and profile['number'] > 0:
            graph.connect(key, 'describes', graph.speaker(profile['number']))
        for index, reference in enumerate(profile['references']):
            bounds = source_interval(reference, graph.duration)
            if bounds:
                ref_key = key + ':reference:' + str(index)
                graph.node(ref_key, 'voice_reference', 'Clean voice reference', bounds, reference)
                graph.connect(ref_key, 'supports_profile', key)
    by_start = {row['startMs']: row for row in report.get('segments', [])}
    for key, node in list(graph.nodes.items()):
        if node['kind'] != 'utterance':
            continue
        cue = by_start.get(node['startMs']) or {}
        if cue.get('voiceProfileId'):
            graph.connect(key, 'matched_voice_profile', 'voice_profile:' + cue['voiceProfileId'])
        if 'voiceProfileEvidence' in cue:
            node['data']['voiceProfileEvidence'] = public_voice_evidence(cue['voiceProfileEvidence'])
            node['data']['rawSpeaker'] = cue.get('speaker')
    audit = report.get('dialogueAudit') or {}
    for item in (audit.get('candidates') or [])[:2000]:
        bounds = source_interval(item, graph.duration)
        if not bounds or not isinstance(item.get('id'), str):
            continue
        key = 'dialogue_issue:' + item['id'][:80]
        reasons = [value[:100] for value in item.get('reasons', [])[:12] if isinstance(value, str)]
        data = {'reasons': reasons, 'priority': str(item.get('priority', 'unknown'))[:20],
                'meaning': 'review-required-structural-candidate', 'notConfirmedError': True}
        evidence = item.get('evidence') or {}
        data['cueIndices'] = [value for value in evidence.get('cueIndices', [])[:128] if type(value) is int]
        graph.node(key, 'dialogue_issue', '; '.join(reasons), bounds, data)
        graph.link_shots(key, *bounds)
