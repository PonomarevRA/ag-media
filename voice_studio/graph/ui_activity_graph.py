"""Publish raw sampled UI anchors without turning their changes into speech identity."""

from math import isfinite

BASES = {'sampled-green-outline', 'sampled-rectangular-outline', 'agent-supplied-roi'}
CHANGES = {'initial', 'stable', 'border-appearance', 'border-disappearance',
           'border-color-change', 'content-change', 'unknown'}


def _number(value, minimum, maximum):
    """Validate finite imported numeric evidence; booleans are not measurements."""
    return type(value) in (int, float) and isfinite(value) and minimum <= value <= maximum


def _sample_facts(sample, duration):
    """Copy only valid optional v2 appearance and temporal facts from untrusted imports."""
    facts = {'contentState': 'unknown', 'changeType': 'unknown', 'identityStatus': 'unknown'}
    if sample.get('contentState') in ('static', 'dynamic', 'unknown'):
        facts['contentState'] = sample['contentState']
    if isinstance(sample.get('changeType'), str) and sample['changeType'] in CHANGES:
        facts['changeType'] = sample['changeType']
    for name in ('staticRunMs', 'dynamicRunMs'):
        if type(sample.get(name)) is int and 0 <= sample[name] <= duration:
            facts[name] = sample[name]
    color = sample.get('outlineColorBgr')
    if isinstance(color, (list, tuple)) and len(color) == 3 and all(_number(v, 0, 255) for v in color):
        facts['outlineColorBgr'] = list(color)
    if _number(sample.get('outlineContrast'), 0, 255):
        facts['outlineContrast'] = sample['outlineContrast']
    if sample.get('outlineType') == 'thin-rectangle':
        facts['outlineType'] = 'thin-rectangle'
    return facts


def _panel_facts(panel, duration):
    """Preserve bounded numeric panel summaries without trusting imported semantics."""
    facts = {'id': panel['id'], 'identityStatus': 'unknown'}
    for name in ('shot', 'firstMs', 'lastMs', 'samples', 'staticComparisons', 'comparisons',
                 'retiredMs', 'lastOutlineMs', 'longestStaticMs', 'longestDynamicMs'):
        value = panel.get(name)
        if value is None:
            facts[name] = None
        elif type(value) is int and 0 <= value <= max(duration, 100000):
            facts[name] = value
    if panel.get('staticFraction') is None or _number(panel.get('staticFraction'), 0, 1):
        facts['staticFraction'] = panel.get('staticFraction')
    if type(panel.get('seeded')) is bool:
        facts['seeded'] = panel['seeded']
    return facts


def add_ui_activity(graph, report, box_coordinates):
    """Keep up to 64 regions and 100000 samples as separate source-clock evidence nodes.

    Legacy green-outline imports remain readable. Their color contract never
    generalizes to a speaking claim; all exported region identities stay unknown.
    """
    layer = (report.get('sceneEvidence') or {}).get('uiActivity') or {}
    if not isinstance(layer, dict):
        return
    panel_rows, sample_rows = layer.get('panels'), layer.get('samples')
    if not isinstance(panel_rows, list) or not isinstance(sample_rows, list):
        return
    panels = {}
    for panel in panel_rows[:64]:
        if not isinstance(panel, dict) or not isinstance(panel.get('id'), str) or not 1 <= len(panel['id']) <= 80:
            continue
        box = box_coordinates(panel.get('box'))
        first, last = panel.get('firstMs'), panel.get('lastMs')
        if not box or type(first) is not int or type(last) is not int or not 0 <= first <= last < graph.duration:
            continue
        if panel['id'] in panels:
            continue
        basis = panel.get('basis')
        if not isinstance(basis, str) or basis not in BASES:
            basis = 'sampled-green-outline' if layer.get('version') == 1 else 'sampled-rectangular-outline'
        key = 'ui_panel:' + panel['id']
        facts = _panel_facts(panel, graph.duration)
        graph.node(key, 'ui_panel', 'Область интерфейса' if graph.language != 'en' else 'Interface region',
                   (first, last + 1), {**facts, 'box': box, 'basis': basis,
                    'truncated': layer.get('truncated') is True or len(panel_rows) > 64 or len(sample_rows) > 100000,
                    'regionMeaning': 'sampled-roi-not-participant'})
        graph.link_shots(key, first, last + 1)
        panels[panel['id']] = key, box, first, last, basis
    for sample in sample_rows[:100000]:
        if not isinstance(sample, dict) or not isinstance(sample.get('panelId'), str) or sample['panelId'] not in panels:
            continue
        stamp, strength = sample.get('timeMs'), sample.get('outlineCoverage')
        if (type(stamp) is not int or not 0 <= stamp < graph.duration or type(sample.get('active')) is not bool
                or not _number(strength, 0, 1)):
            continue
        panel_id, box, first, last, basis = panels[sample['panelId']]
        if not first <= stamp <= last:
            continue
        key = f"ui_activity:{stamp}:{sample['panelId'][:80]}"
        delta = sample.get('contentDelta')
        if delta is not None and not _number(delta, 0, 255):
            continue
        data = {'timeMs': stamp, 'panelId': sample['panelId'], 'active': sample['active'],
                'outlineCoverage': strength, 'contentDelta': delta, 'box': box,
                'meaning': 'visual-activity-candidate', 'basis': basis, 'notSpeechProbability': True,
                'geometryStatus': 'observed-outline' if sample['active'] else 'retained-roi',
                'panelPresence': 'observed' if sample['active'] else 'unverified',
                **_sample_facts(sample, graph.duration)}
        label = ('Наблюдаемая рамка' if sample['active'] else 'Рамка не найдена') if graph.language != 'en' else ('Observed outline' if sample['active'] else 'Outline not found')
        graph.node(key, 'ui_activity', label, (stamp, stamp + 1), data)
        graph.connect(key, 'in_panel', panel_id)
        graph.connect(key, 'at_frame', graph.frame(stamp, 'ui-activity'))
        graph.link_shots(key, stamp, stamp + 1)
