from pathlib import Path
import json,math,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import nmcd_pc as P
import nmcd_review_extensions as E
class TestV03(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=E.canonicalise_reference();cls.d=cls.m.dictionary
        cls.tr=P.draw_counts(np.random.default_rng(191),4096,.48)
        cls.te=P.draw_counts(np.random.default_rng(292),8192,.48)
    def test_count_177(self):self.assertEqual(self.d.size,177)
    def test_parent_289_unchanged(self):self.assertEqual(P.D.size,289)
    def test_complete_function_set(self):self.assertEqual(set(map(tuple,P.D.predictions)),set(map(tuple,self.d.predictions)))
    def test_complete_support_required(self):
        with self.assertRaises(ValueError):E.canonicalise_reference(support='observed_test_rows')
    def test_mapping_all_entries(self):
        np.testing.assert_array_equal(P.D.predictions,self.d.predictions[list(self.m.old_to_new)])
    def test_class_sets_preserved(self):
        for k in (0,1,2):self.assertEqual(set(map(tuple,P.D.predictions[P.D.classes[k]])),set(map(tuple,self.d.predictions[self.d.classes[k]])))
    def test_class_counts(self):self.assertEqual([len(self.d.classes[k]) for k in (0,1,2)],[1,17,177])
    def test_idempotent(self):self.assertEqual(E.canonicalise_reference(self.d).dictionary.fingerprint,self.d.fingerprint)
    def test_frozen_predictions(self):self.assertFalse(self.d.predictions.flags.writeable)
    def test_oracle_risks_preserved(self):
        for theta,kappa in [(0,0),(.30,0),(.50,0),(.75,0),(.25,.5),(.75,.25)]:
            self.assertEqual(P.exact_truth(theta,kappa,(2,1,0)),P.exact_truth(theta,kappa,(2,1,0),d=self.d))
    def test_erm_tie_order_preserved(self):
        for t in P.HORIZONS:
            a=P.fit(self.tr[t],(2,1,0));b=P.fit(self.tr[t],(2,1,0),self.d)
            for k in a:np.testing.assert_array_equal(P.D.predictions[a[k]],self.d.predictions[b[k]])
    def test_scalar_bounds_unchanged(self):
        for t in P.HORIZONS:
            a=P.fit(self.tr[t],(2,1,0));b=P.fit(self.tr[t],(2,1,0),self.d)
            for k in a:
                for m in ('max_eb','max_hoeffding'):
                    x=P.certificate(self.te[t],a[k],k,m);y=P.certificate(self.te[t],b[k],k,m,d=self.d)
                    self.assertAlmostEqual(x,y,places=13)
    def test_union_and_two_stage_bounds_not_larger(self):
        for t in P.HORIZONS:
            a=P.fit(self.tr[t],(2,1,0));b=P.fit(self.tr[t],(2,1,0),self.d)
            for k in a:
                for m in ('union_eb','union_hoeffding','two_stage'):
                    x=P.certificate(self.te[t],a[k],k,m,train=self.tr[t]);y=P.certificate(self.te[t],b[k],k,m,d=self.d,train=self.tr[t])
                    self.assertLessEqual(y,x+1e-14)
    def test_original_floors(self):
        z=E.planning_floors(289);self.assertEqual((z['union_H_minimum'],z['scalar_H_minimum'],z['two_stage_zero_difference_minimum']),(4436,1534,13200))
    def test_canonical_floors(self):
        z=E.planning_floors(177);self.assertEqual((z['union_H_minimum'],z['scalar_H_minimum'],z['two_stage_zero_difference_minimum']),(4185,1534,12801))
    def test_scalar_floor_dictionary_independent(self):self.assertEqual(E.planning_floors(177)['scalar_H_minimum'],E.planning_floors(289)['scalar_H_minimum'])
    def test_environment_pin_rejects_other_version(self):
        with self.assertRaises(ValueError):E.require_replay_numpy('2.5.3')
    def test_environment_pin_accepts_recorded_version(self):E.require_replay_numpy('2.3.5')
    def test_actual_rng_recorded(self):self.assertEqual(E.environment_record()['bit_generator'],'PCG64')
    def test_no_retrospective_primary_promotion(self):
        p=json.loads((ROOT/'contracts/audit_plan.json').read_text());self.assertEqual(p['primary_prespecified_candidate'],'union_eb')
    def test_new_primary_only_candidate(self):
        p=json.loads((ROOT/'contracts/qualification_draft.json').read_text());self.assertEqual(p['primary_certificate_candidate'],'max_eb');self.assertFalse(p['confirmatory_execution_authorised']);self.assertIsNone(p['independent_mathematical_review'])
    def test_cosmology_group_law_still_blocked(self):
        with self.assertRaises(ValueError):P.validate_provenance(['a'],['b'],'a','b',{'X':0},group_law='independent_simulation_realisations')
if __name__=='__main__':unittest.main()
