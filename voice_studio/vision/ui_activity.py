"""Track sampled visual anchors without inferring speech, people, or identity."""

from hashlib import sha256
from math import isfinite

from voice_studio.vision.ui_static_regions import analysis_view, outline_at_box, outline_candidates

MAX_PANELS = 64
MAX_SAMPLES = 100000
MAX_GAP_MS = 1500


def same_box(left, right):
    """Match geometry within a shot, including a modestly clipped lower edge."""
    x = max(left[0], right[0]); y = max(left[1], right[1])
    area = max(0, min(left[0]+left[2], right[0]+right[2])-x) * max(0, min(left[1]+left[3], right[1]+right[3])-y)
    union = left[2]*left[3] + right[2]*right[3] - area
    if not union:
        return False
    if area / union >= .92:
        return True
    smallest = min(left[2]*left[3], right[2]*right[3])
    return (smallest > 0 and area / smallest >= .98
            and abs(left[0]-right[0]) <= .008 and abs(left[2]-right[2]) <= .008
            and abs(left[1]-right[1]) <= .008
            and min(left[3], right[3]) >= .65 * max(left[3], right[3]))


def _valid_box(box):
    """Accept bounded normalized ROI seeds, rejecting booleans and nonfinite coordinates."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(type(value) not in (int, float) or not isfinite(value) for value in box):
        return False
    x, y, width, height = box
    return 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0 and x+width <= 1 and y+height <= 1


class UIActivity:
    """Bounded per-shot ROI measurements; optional seeds never assert observed presence."""

    def __init__(self, region_seeds=(), *, discovery_interval_ms=2000):
        """Keep 24x24 signatures; region_seeds are normalized boxes supplied by a caller.

        Seeds allow static arbitrary interfaces to be studied before a border appears.
        A seed is an unverified measurement ROI, not a detected window or participant.
        """
        self.panels = []
        self.samples = []
        self.previous = {}
        self.truncated = False
        self.region_seeds = []
        for box in region_seeds:
            if _valid_box(box) and len(self.region_seeds) < MAX_PANELS:
                self.region_seeds.append(list(box))
        self.last_shot = None
        self.last_discovery = None
        self.discovery_interval_ms = max(0, int(discovery_interval_ms))

    def _new_panel(self, box, stamp, shot, *, seeded=False):
        """Create a track and retire conflicting layouts without transferring identity."""
        if len(self.panels) >= MAX_PANELS:
            self.truncated = True
            return None
        key = sha256(f"{shot}:{stamp}:{box}:{seeded}".encode()).hexdigest()[:16]
        panel = {'id': key, 'shot': shot, 'box': box, 'firstMs': stamp, 'lastMs': stamp,
                 'lastOutlineMs': None, 'samples': 0, 'staticComparisons': 0, 'comparisons': 0,
                 'longestStaticMs': 0, 'longestDynamicMs': 0, 'seeded': seeded}
        for old in self.panels:
            if old['shot'] != shot or 'retiredMs' in old:
                continue
            a, b = old['box'], box
            area = max(0, min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
            if area > .25 * min(a[2]*a[3],b[2]*b[3]):
                old['retiredMs'] = stamp
                self.previous.pop(old['id'], None)
        self.panels.append(panel)
        return panel

    def add(self, image, sample):
        """Measure one decoded frame at local timeMs/shot; gaps retain unknown state.

        active remains a compatibility alias for outline observation, never speech.
        Missing outlines leave retained ROI presence unverified. Shot/layout changes
        retire old tracks; content comparisons never bridge more than 1500 ms.
        """
        stamp, shot = sample['timeMs'], sample['shot']
        if self.last_shot != shot:
            for panel in self.panels:
                if 'retiredMs' not in panel:
                    panel['retiredMs'] = stamp
            self.previous.clear()
            for box in self.region_seeds:
                self._new_panel(box, stamp, shot, seeded=True)
            self.last_shot = shot
            self.last_discovery = None
        observed = {}
        view = analysis_view(image)
        live = [row for row in self.panels if row['shot'] == shot and 'retiredMs' not in row]
        discover = not live or self.last_discovery is None or stamp-self.last_discovery >= self.discovery_interval_ms
        candidates = outline_candidates(view) if discover else []
        if discover:
            self.last_discovery = stamp
        for candidate in candidates:
            panel = next((row for row in self.panels if row['shot'] == shot and 'retiredMs' not in row
                          and same_box(row['box'], candidate['box'])), None)
            if panel is None:
                panel = self._new_panel(candidate['box'], stamp, shot)
            if panel is not None:
                panel['lastOutlineMs'] = stamp
                observed[panel['id']] = candidate
        for panel in self.panels:
            if panel['shot'] == shot and 'retiredMs' not in panel:
                if panel['id'] not in observed:
                    candidate = outline_at_box(view, panel['box'])
                    if candidate is not None:
                        panel['lastOutlineMs'] = stamp
                        observed[panel['id']] = candidate
                self._measure(image, panel, stamp, observed.get(panel['id']))

    def _measure(self, image, panel, stamp, candidate):
        """Compare the inset ROI with the preceding tiny signature and record raw changes."""
        import cv2
        height, width = image.shape[:2]
        x, y, w, h = panel['box']
        crop = image[round((y+h*.08)*height):round((y+h*.90)*height),
                     round((x+w*.04)*width):round((x+w*.96)*width)]
        if not crop.size:
            return
        signature = cv2.resize(crop, (24,24), interpolation=cv2.INTER_AREA)
        prior = self.previous.get(panel['id'])
        delta, state, run_ms = None, 'unknown', 0
        contiguous = prior is not None and 0 < stamp-prior['stamp'] <= MAX_GAP_MS
        if contiguous:
            channel_deltas = cv2.mean(cv2.absdiff(signature, prior['signature']))[:3]
            delta = round(sum(channel_deltas) / 3, 3)
            state = 'static' if delta <= 2 else 'dynamic'
            run_ms = stamp - prior['stamp']
            if prior['state'] == state:
                run_ms += prior['runMs']
            panel['comparisons'] += 1
            panel['staticComparisons'] += int(state == 'static')
            field = 'longestStaticMs' if state == 'static' else 'longestDynamicMs'
            panel[field] = max(panel[field], run_ms)
        change = self._change_type(candidate, prior, contiguous, state)
        self.previous[panel['id']] = {'stamp': stamp, 'signature': signature, 'state': state,
                                      'runMs': run_ms, 'outline': candidate}
        panel['lastMs'] = stamp
        panel['samples'] += 1
        if len(self.samples) >= MAX_SAMPLES:
            self.truncated = True
            return
        row = {'timeMs': stamp, 'panelId': panel['id'], 'active': candidate is not None,
               'outlineCoverage': candidate['outlineCoverage'] if candidate else 0,
               'contentDelta': delta, 'contentState': state, 'changeType': change,
               'staticRunMs': run_ms if state == 'static' else 0,
               'dynamicRunMs': run_ms if state == 'dynamic' else 0,
               'geometryStatus': 'observed-outline' if candidate else 'retained-roi',
               'panelPresence': 'observed' if candidate else 'unverified', 'identityStatus': 'unknown'}
        for name in ('outlineColorBgr', 'outlineContrast', 'outlineType'):
            row[name] = candidate[name] if candidate else None
        self.samples.append(row)

    @staticmethod
    def _change_type(candidate, prior, contiguous, state):
        """Name observed border/content changes only when adjacent sampling supports them."""
        if prior is None:
            return 'initial'
        if not contiguous:
            return 'unknown'
        previous = prior['outline']
        if candidate and not previous:
            return 'border-appearance'
        if previous and not candidate:
            return 'border-disappearance'
        if candidate and previous:
            distance = max(abs(a-b) for a,b in zip(candidate['outlineColorBgr'], previous['outlineColorBgr']))
            if distance > 35:
                return 'border-color-change'
        return 'content-change' if state == 'dynamic' else 'stable'

    def summary(self):
        """Export sampled evidence; temporal runs are comparisons, not continuous presence proof."""
        panels = []
        for panel in self.panels:
            row = dict(panel)
            comparisons = row['comparisons']
            row['staticFraction'] = round(row['staticComparisons']/comparisons, 4) if comparisons else None
            row['basis'] = 'agent-supplied-roi' if row['seeded'] and row['lastOutlineMs'] is None else 'sampled-rectangular-outline'
            row['regionMeaning'] = 'sampled-roi-not-participant'
            row['identityStatus'] = 'unknown'
            panels.append(row)
        return {'version': 2, 'truncated': self.truncated, 'meaning': 'visual-activity-candidate',
                'geometryDiscoveryIntervalMs': self.discovery_interval_ms,
                'panels': panels, 'samples': self.samples}
