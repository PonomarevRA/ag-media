"""General UI anchors retain raw evidence across colors, time, and graph storage."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from voice_studio.vision.ui_activity import UIActivity, outline_candidates
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


class GenericUIActivityTests(unittest.TestCase):
    def frame(self, color=(255, 90, 30), *, filled=False, box=(20, 40, 300, 210), background=0):
        """Create a source-independent thin rectangle on a constant background."""
        image = np.full((360, 640, 3), background, np.uint8)
        if color is not None:
            cv2.rectangle(image, box[:2], box[2:], color, -1 if filled else 3)
        return image

    def test_detects_blue_white_orange_green_and_dark_outlines(self):
        """Different hues and neutral luminance obey the same geometric contract."""
        for color, background in [((255,90,30),0), ((250,250,250),0), ((20,140,250),0),
                                  ((40,220,60),0), ((30,30,30),240)]:
            with self.subTest(color=color):
                found = outline_candidates(self.frame(color, background=background))
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]['outlineColorBgr'], list(color))
                self.assertGreater(found[0]['outlineContrast'], 100)

    def test_filled_panels_text_grid_and_weak_borders_do_not_qualify(self):
        """Negative structures cannot establish a sufficiently thin, isolated border."""
        for color in [(255,90,30), (250,250,250), (20,140,250), (40,220,60)]:
            self.assertEqual(outline_candidates(self.frame(color, filled=True)), [])
        text = self.frame(None)
        for y in range(30, 300, 40):
            cv2.putText(text, 'text without UI outline', (20,y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1)
        self.assertEqual(outline_candidates(text), [])
        grid = self.frame(None)
        for y in (20, 170, 320):
            cv2.line(grid, (20,y), (620,y), (255,255,255), 2)
        for x in (20, 220, 420, 620):
            cv2.line(grid, (x,20), (x,320), (255,255,255), 2)
        self.assertEqual(outline_candidates(grid), [])
        self.assertEqual(outline_candidates(self.frame((12,12,12))), [])

    def test_same_hue_subpixel_brightness_variation_remains_one_outline(self):
        """Compressed vertical and horizontal strokes need not have equal brightness."""
        image = self.frame(None)
        cv2.line(image, (20,40), (300,40), (90,190,25), 2)
        cv2.line(image, (20,210), (300,210), (90,190,25), 2)
        cv2.line(image, (20,40), (20,210), (60,130,15), 2)
        cv2.line(image, (300,40), (300,210), (60,130,15), 2)
        self.assertEqual(len(outline_candidates(image)), 1)

    def test_content_comparison_retains_chromatic_changes(self):
        """Equally bright grayscale colors still produce changed region evidence."""
        tracker = UIActivity(region_seeds=[[.1,.1,.5,.5]])
        first = np.zeros((360,640,3), np.uint8); first[:] = (0,0,100)
        second = np.zeros_like(first); second[:] = (0,51,0)
        tracker.add(first, {'timeMs':0,'shot':0})
        tracker.add(second, {'timeMs':500,'shot':0})
        self.assertEqual(tracker.summary()['samples'][-1]['contentState'], 'dynamic')

    def test_frame_clipped_outline_can_be_observed(self):
        """A frame boundary may hide one side without inventing the missing side."""
        found = outline_candidates(self.frame(box=(20,210,300,400)))
        self.assertEqual(len(found), 1)
        self.assertGreater(found[0]['outlineCoverage'], .7)

    def test_border_changes_keep_color_and_content_separate(self):
        """Border transitions remain raw observations while interior stability is independent."""
        tracker = UIActivity()
        colors = [(255,90,30), (255,90,30), (20,140,250), None, (250,250,250)]
        for index, color in enumerate(colors):
            tracker.add(self.frame(color), {'timeMs':index*500, 'shot':0})
        data = tracker.summary()
        self.assertEqual(data['version'], 2)
        self.assertEqual(len(data['panels']), 1)
        self.assertEqual([row['changeType'] for row in data['samples']],
                         ['initial','stable','border-color-change','border-disappearance','border-appearance'])
        self.assertEqual(data['samples'][3]['panelPresence'], 'unverified')
        self.assertTrue(all(row['identityStatus'] == 'unknown' for row in data['samples']))
        self.assertEqual(data['panels'][0]['longestStaticMs'], 2000)

    def test_seeded_static_regions_do_not_claim_geometry_or_presence(self):
        """Caller-supplied crops can stay static without claiming a detected region."""
        tracker = UIActivity(region_seeds=[[.1,.1,.5,.5]])
        for stamp in (0, 500, 1000, 4000, 4500):
            tracker.add(self.frame(None), {'timeMs':stamp, 'shot':0})
        data = tracker.summary()
        self.assertEqual(data['panels'][0]['basis'], 'agent-supplied-roi')
        self.assertIsNone(data['panels'][0]['lastOutlineMs'])
        self.assertEqual(data['panels'][0]['longestStaticMs'], 1000)
        self.assertEqual(data['samples'][3]['contentState'], 'unknown')
        self.assertEqual(data['samples'][4]['staticRunMs'], 500)
        self.assertTrue(all(row['panelPresence'] == 'unverified' for row in data['samples']))
        self.assertFalse(any(row['active'] for row in data['samples']))

    def test_content_change_produces_dynamic_run_then_new_static_run(self):
        """Temporal runs reset when sampled content switches between change and stability."""
        tracker = UIActivity(region_seeds=[[.1,.1,.5,.5]])
        for index, value in enumerate((0,100,200,200)):
            tracker.add(self.frame(None, background=value), {'timeMs':index*500,'shot':0})
        rows = tracker.summary()['samples']
        self.assertEqual([row['contentState'] for row in rows], ['unknown','dynamic','dynamic','static'])
        self.assertEqual(rows[2]['dynamicRunMs'], 1000)
        self.assertEqual(rows[3]['staticRunMs'], 500)
        self.assertEqual(rows[1]['changeType'], 'content-change')

    def test_invalid_seeds_are_ignored_without_claiming_regions(self):
        """Out-of-range, boolean and nonfinite coordinates do not become ROI evidence."""
        tracker = UIActivity(region_seeds=[[0,0,2,1], [True,0,1,1], [0,float('nan'),1,1], 'bad'])
        tracker.add(self.frame(None), {'timeMs':0,'shot':0})
        self.assertEqual(tracker.summary()['panels'], [])

    def test_layout_and_shot_changes_retire_signatures(self):
        """Conflicting geometry and shot boundaries retire tracks without identity transfer."""
        tracker = UIActivity(discovery_interval_ms=0)
        tracker.add(self.frame(), {'timeMs':0,'shot':0})
        tracker.add(self.frame(box=(60,40,340,210)), {'timeMs':500,'shot':0})
        tracker.add(self.frame(box=(60,40,340,210)), {'timeMs':1000,'shot':1})
        data = tracker.summary()
        self.assertEqual(len(data['panels']), 3)
        self.assertEqual([row['retiredMs'] for row in data['panels'][:2]], [500,1000])
        self.assertEqual(len(tracker.previous), 1)
        self.assertTrue(all(row['contentDelta'] is None for row in data['samples']))

    def test_known_roi_border_changes_are_checked_between_geometry_scans(self):
        """A short border pulse in an existing ROI survives the discovery throttle."""
        with patch('voice_studio.vision.ui_activity.outline_candidates', wraps=outline_candidates) as discovery:
            tracker = UIActivity(discovery_interval_ms=5000)
            for stamp, color in ((0,(255,90,30)), (500,None), (1000,(250,250,250)), (1500,None)):
                tracker.add(self.frame(color), {'timeMs':stamp,'shot':0})
            self.assertEqual(discovery.call_count, 1)
            self.assertEqual([row['active'] for row in tracker.summary()['samples']], [True,False,True,False])

    def test_new_layout_discovery_has_explicit_sampling_latency(self):
        """Geometry searches occur at the first supplied sample after the configured interval."""
        tracker = UIActivity(discovery_interval_ms=2000)
        tracker.add(self.frame(), {'timeMs':0,'shot':0})
        shifted = self.frame(box=(60,40,340,210))
        tracker.add(shifted, {'timeMs':500,'shot':0})
        self.assertEqual(len(tracker.summary()['panels']), 1)
        tracker.add(shifted, {'timeMs':2000,'shot':0})
        data = tracker.summary()
        self.assertEqual(len(data['panels']), 2)
        self.assertEqual(data['panels'][0]['retiredMs'], 2000)
        self.assertEqual(data['geometryDiscoveryIntervalMs'], 2000)

    def test_sample_and_panel_caps_are_explicit(self):
        """Collection limits retain truncation instead of growing pixel or sample history."""
        with patch('voice_studio.vision.ui_activity.MAX_PANELS', 2), patch('voice_studio.vision.ui_activity.MAX_SAMPLES', 2):
            tracker = UIActivity()
            for index in range(4):
                tracker.add(self.frame(), {'timeMs':index*500,'shot':index})
            data = tracker.summary()
            self.assertEqual((len(data['panels']),len(data['samples'])), (2,2))
            self.assertTrue(data['truncated'])

    def graph(self, layer):
        """Build source-time graph evidence for a short synthetic source."""
        return build_graph({'coverage':{'scene':[{'startMs':0,'endMs':3000,'status':'sampled'}]},
                            'sceneEvidence':{'frames':[],'uiActivity':layer}}, kind='video', duration_ms=3000)

    def test_graph_preserves_generic_facts_and_refuses_imported_identity(self):
        """Imported claims cannot promote raw border evidence into a speaker identity."""
        tracker = UIActivity()
        for stamp in (0,500):
            tracker.add(self.frame(), {'timeMs':stamp,'shot':0})
        data = tracker.summary()
        data['samples'][1]['identityStatus'] = 'confirmed-speaker'
        graph = self.graph(data)
        sample = next(row['data'] for row in graph['nodes'] if row['kind']=='ui_activity' and row['data']['timeMs']==500)
        self.assertEqual(sample['basis'], 'sampled-rectangular-outline')
        self.assertEqual(sample['outlineColorBgr'], [255,90,30])
        self.assertEqual(sample['staticRunMs'], 500)
        self.assertEqual(sample['identityStatus'], 'unknown')
        self.assertFalse(any(row['kind']=='speaker' for row in graph['nodes']))

    def test_malformed_optional_facts_cannot_escape_as_trusted_measurements(self):
        """Invalid optional appearance and temporal fields are removed at graph import."""
        layer = {'version':2,'panels':[{'id':'p','box':[0,0,.5,.5],'firstMs':0,'lastMs':500,'basis':[]}],
                 'samples':[{'panelId':'p','timeMs':0,'active':True,'outlineCoverage':1,
                             'outlineColorBgr':[True, 5, float('nan')], 'outlineContrast':float('inf'),
                             'contentState':[], 'changeType':{}, 'staticRunMs':-10}]}
        graph = self.graph(layer)
        row = next(row['data'] for row in graph['nodes'] if row['kind']=='ui_activity')
        self.assertNotIn('outlineColorBgr', row)
        self.assertNotIn('outlineContrast', row)
        self.assertNotIn('staticRunMs', row)
        self.assertEqual(row['contentState'], 'unknown')
        self.assertEqual(row['changeType'], 'unknown')

    def test_graph_storage_roundtrip_keeps_appearance_and_temporal_evidence(self):
        """Frozen graph reuse retains generic colors and sampled static durations."""
        tracker = UIActivity()
        for stamp in (0,500):
            tracker.add(self.frame(), {'timeMs':stamp,'shot':0})
        with TemporaryDirectory() as folder:
            source = Path(folder)/'source.mov'; source.write_bytes(b'synthetic-source')
            store = MediaGraphStore(Path(folder)/'graphs')
            manifest = store.publish(identify(source), {'kind':'video','durationMs':3000}, self.graph(tracker.summary()), 'ru', {})
            scene, _ = store.scene_evidence(manifest['snapshotId'], identify(source)['sha256'], 'ru')
            sample = scene['uiActivity']['samples'][-1]
            self.assertEqual(sample['outlineColorBgr'], [255,90,30])
            self.assertEqual(sample['staticRunMs'], 500)
            self.assertEqual(sample['contentState'], 'static')
            again = self.graph(scene['uiActivity'])
            self.assertEqual(sum(row['kind']=='ui_activity' for row in again['nodes']), 2)


if __name__ == '__main__':
    unittest.main()
