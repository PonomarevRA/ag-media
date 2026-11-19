"""Call UI evidence must remain distinct from a claimed voice identity."""
from importlib.util import find_spec
import unittest
if find_spec("cv2") is None or find_spec("numpy") is None:
    raise unittest.SkipTest("Requires the optional vision environment")

import cv2
import numpy as np
from voice_studio.vision.ui_activity import UIActivity, outline_candidates
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import node_kind_filter

class UIActivityTests(unittest.TestCase):
    def frame(self, active=True, filled=False):
        """Two synthetic call tiles; only the outline is an activity observation."""
        image=np.zeros((360,640,3),np.uint8)
        if active: cv2.rectangle(image,(20,40),(300,210),(40,220,60),-1 if filled else 3)
        return image

    def test_outline_not_foliage_or_filled_region(self):
        self.assertEqual(len(outline_candidates(self.frame())),1)
        self.assertEqual(outline_candidates(self.frame(filled=True)),[])
        self.assertEqual(outline_candidates(self.frame(False)),[])

    def test_tracks_inactive_static_content_without_bridging_gap(self):
        tracker=UIActivity()
        for stamp,active in [(0,True),(500,False),(3000,True)]:
            tracker.add(self.frame(active),{'timeMs':stamp,'shot':0})
        data=tracker.summary()
        self.assertEqual(len(data['panels']),1)
        self.assertEqual([s['active'] for s in data['samples']],[True,False,True])
        self.assertIsNone(data['samples'][-1]['contentDelta'])
        self.assertEqual(data['panels'][0]['comparisons'],1)

    def test_shot_changes_break_geometric_identity(self):
        tracker=UIActivity()
        for shot in [0,1]:tracker.add(self.frame(),{'timeMs':shot*1000,'shot':shot})
        self.assertEqual(len(tracker.summary()['panels']),2)

    def test_graph_keeps_activity_separate_from_speech(self):
        tracker=UIActivity();tracker.add(self.frame(),{'timeMs':0,'shot':0})
        result=build_graph({'language':'ru','sceneEvidence':{'frames':[],'uiActivity':tracker.summary()}},kind='video',duration_ms=1000)
        kinds=[n['kind'] for n in result['nodes']]
        self.assertIn('ui_panel',kinds);self.assertIn('ui_activity',kinds)
        self.assertNotIn('speaker',kinds)
        predicate,args=node_kind_filter('ui')
        self.assertEqual(predicate.count('?'),len(args))
        self.assertIn('ui_activity',args)


    def test_partial_panel_does_not_create_nested_duplicate(self):
        tracker=UIActivity()
        tracker.add(self.frame(),{'timeMs':0,'shot':0})
        image=np.zeros((360,640,3),np.uint8)
        cv2.rectangle(image,(20,40),(300,165),(40,220,60),3)
        tracker.add(image,{'timeMs':500,'shot':0})
        self.assertEqual(len(tracker.summary()['panels']),1)

    def test_import_rejects_invalid_collections_and_panel_references(self):
        for layer in ({'panels':{},'samples':[]},{'panels':[],'samples':[{'panelId':[]}]}):
            graph=build_graph({'sceneEvidence':{'frames':[],'uiActivity':layer}},kind='video',duration_ms=1000)
            self.assertFalse(any(n['kind']=='ui_activity' for n in graph['nodes']))


    def test_retirement_and_outline_times_follow_window_offset(self):
        from voice_studio.graph.media_graph_ingest import apply_scene_evidence
        report={'sceneEvidence':{'frames':[]},'coverage':{'scene':[]}}
        scene={'status':'ok','frames':[],'uiActivity':{'panels':[{'id':'p','firstMs':0,'lastMs':500,'retiredMs':500,'lastOutlineMs':200}], 'samples':[]}}
        apply_scene_evidence(report,scene,30000,31000)
        panel=report['sceneEvidence']['uiActivity']['panels'][0]
        self.assertEqual((panel['firstMs'],panel['lastMs'],panel['retiredMs'],panel['lastOutlineMs']),(30000,30500,30500,30200))

    def test_duplicate_import_cannot_move_sample_to_a_different_panel(self):
        panel={'id':'p','box':[0,0,.4,.4],'firstMs':0,'lastMs':10}
        layer={'panels':[panel,{**panel,'box':[.5,.5,.4,.4],'firstMs':900,'lastMs':999}],
               'samples':[{'panelId':'p','timeMs':900,'active':True,'outlineCoverage':1}]}
        graph=build_graph({'sceneEvidence':{'frames':[],'uiActivity':layer}},kind='video',duration_ms=1000)
        self.assertFalse(any(n['kind']=='ui_activity' for n in graph['nodes']))

    def test_missing_outline_retains_only_an_unverified_roi(self):
        tracker=UIActivity()
        tracker.add(self.frame(),{'timeMs':0,'shot':0})
        tracker.add(self.frame(False),{'timeMs':1000,'shot':0})
        data=tracker.summary()
        self.assertEqual(data['panels'][0]['lastOutlineMs'],0)
        self.assertEqual(data['samples'][-1]['panelPresence'],'unverified')
        self.assertEqual(data['samples'][-1]['geometryStatus'],'retained-roi')

    def test_truncation_survives_snapshot_roundtrip_even_without_panels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from voice_studio.graph.media_graph_store import MediaGraphStore,identify
        report={'coverage':{'scene':[{'startMs':0,'endMs':1000,'status':'sampled'}]},
                'sceneEvidence':{'frames':[],'uiActivity':{'panels':[],'samples':[],'truncated':True}}}
        with TemporaryDirectory() as folder:
            source=Path(folder)/'source.mov';source.write_bytes(b'test-source')
            graph=build_graph(report,kind='video',duration_ms=1000)
            store=MediaGraphStore(Path(folder)/'graphs')
            manifest=store.publish(identify(source),{'kind':'video','durationMs':1000},graph,'ru',{})
            scene,_=store.scene_evidence(manifest['snapshotId'],identify(source)['sha256'],'ru')
            self.assertTrue(scene['uiActivity']['truncated'])
            again=build_graph({'sceneEvidence':scene},kind='video',duration_ms=1000)
            self.assertTrue(again['uiActivity']['truncated'])

    def test_non_mapping_import_layer_is_ignored_safely(self):
        for layer in ([{'id':'bad'}], 'invalid', True):
            graph=build_graph({'sceneEvidence':{'frames':[],'uiActivity':layer}},kind='video',duration_ms=1000)
            self.assertFalse(graph['uiActivity']['truncated'])

if __name__=='__main__':unittest.main()
