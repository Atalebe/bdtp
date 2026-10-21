from __future__ import annotations
import dataclasses
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import nmcd_pc as P

class TestReference(unittest.TestCase):
    def setUp(self):
        self.tr=P.draw_counts(np.random.default_rng(17),128,.5)
        self.te=P.draw_counts(np.random.default_rng(31),257,.5)
        self.i=P.fit(self.tr[1])[1]
    def test_dictionary_dimensions(self):
        P.validate_dictionary(P.D)
        self.assertEqual(P.D.size,289)
        self.assertEqual([len(P.D.classes[k]) for k in (0,1,2)],[1,17,289])
    def test_original_risk_identities(self):
        for th in (0.,.25,.5,.75,1.):
            for t in P.HORIZONS:
                risks=P.exact_risks(th,t);b=Fraction(str(th))/t/2
                self.assertEqual(min(risks[i] for i in P.D.classes[1]),Fraction(1,4))
                self.assertEqual(min(risks),Fraction(1,4)-b*b)
    def test_strict_boundary(self):
        t=P.exact_truth(.5)
        self.assertEqual(t['C_weak'],1);self.assertEqual(t['C_strict'],2)
    def test_zero_coordinate_anchor(self):
        t=P.exact_truth(.25,dimensions=(2,1,0))
        self.assertEqual(t['C_strict'],0)
    def test_informative_current_coordinate(self):
        t=P.exact_truth(.25,.5,(2,1,0))
        self.assertEqual(t['C_strict'],1)
        self.assertEqual(t['gap_fractions'][1][0],'5/64')
        self.assertEqual(t['gap_fractions'][1][1],'1/64')
    def test_extension_history_needed(self):
        self.assertEqual(P.exact_truth(.75,.25,(2,1,0))['C_strict'],2)
    def test_off_grid_is_not_bayes_formula(self):
        t=P.exact_truth(.30)
        self.assertAlmostEqual(t['gaps'][1][1],.021875)
        self.assertNotAlmostEqual(t['gaps'][1][1],.30**2/4)
    def test_loss_from_explicit_episodes(self):
        cnt=self.te[1];cell=[];ys=[]
        for c in range(4):
            for y in (0,1):
                cell.extend([c]*int(cnt[c,y]));ys.extend([y]*int(cnt[c,y]))
        ep=P.D.losses[:,np.array(cell),np.array(ys)]
        np.testing.assert_allclose(ep.mean(1),P.mean_losses(cnt),rtol=1e-13)
        mean,var=P.diff_moments(cnt,self.i)
        diff=ep[self.i][None]-ep
        np.testing.assert_allclose(mean,diff.mean(1),atol=1e-14)
        np.testing.assert_allclose(var,diff.var(1,ddof=1),atol=1e-14)
    def test_public_fast_paths_agree(self):
        for t in P.HORIZONS:
            fast,fit=P.fast_all(self.tr[t],self.te[t],dimensions=(2,1,0))
            for m in P.METHODS:
                for k in (2,1,0):
                    checked=P.certificate(self.te[t],fit[k],k,m,train=self.tr[t])
                    self.assertAlmostEqual(checked,fast[m][k],places=13)
    def test_union_bounds_dominate_scalar_bounds(self):
        for t in P.HORIZONS:
            x,_=P.fast_all(self.tr[t],self.te[t])
            for k in (2,1):
                self.assertGreaterEqual(x['union_hoeffding'][k],x['max_hoeffding'][k])
                self.assertGreaterEqual(x['union_eb'][k],x['max_eb'][k])
    def test_horizon_episode_counts_shared(self):
        self.assertEqual(P.validate_shared_horizons(self.te),257)
    def test_saturation_absolute_risk(self):
        cnt=P.draw_counts(np.random.default_rng(6),4096,1.)[1]
        idx=P.fit(cnt)[2]
        self.assertEqual(P.mean_losses(cnt)[idx],0.)
        absolute=math.sqrt(math.log(1/.05)/(2*4096))
        self.assertLess(absolute,P.EPS)
        self.assertTrue(math.isfinite(P.certificate(cnt,idx,2,'max_eb')))
    def test_two_stage_current_ceiling(self):
        for n in (8192,13200):
            eta=2*math.sqrt(math.log(2*289/.02)/(2*n))
            slack=eta+math.sqrt(2*math.log(1/.03)/n)
            self.assertEqual(slack<P.EPS,n==13200)
    def test_analytic_sample_floors(self):
        self.assertEqual(math.floor(2*math.log(289/.05)/P.EPS**2)+1,4436)
        self.assertEqual(math.floor(2*math.log(1/.05)/P.EPS**2)+1,1534)
    def test_stop_and_equality(self):
        b={1:{2:.01,1:P.EPS},2:{2:.01,1:.01}}
        self.assertEqual(P.decide(b),2)
        b[1][1]=np.nextafter(P.EPS,-math.inf);self.assertEqual(P.decide(b),1)
        b[2][2]=P.EPS;self.assertIsNone(P.decide(b))
    def test_no_certificate_not_infinite_truth(self):
        b={t:{2:math.inf,1:0.} for t in P.HORIZONS}
        self.assertIsNone(P.decide(b))
    def test_weak_history_is_not_strong_history(self):
        rr=P.exact_risks(.75,1)
        weak=min(rr[i] for i in P.D.classes[1]);strong=min(rr)
        self.assertEqual(weak-weak,0)
        self.assertEqual(weak-strong,Fraction(9,64))
    def test_seed_roles_distinct_replay_reproducible(self):
        a,ak=P.rng_for(42,'case','train');b,bk=P.rng_for(42,'case','test');c,ck=P.rng_for(42,'case','train')
        self.assertNotEqual(ak,bk);self.assertEqual(ak,ck)
        np.testing.assert_array_equal(a.integers(0,100,20),c.integers(0,100,20))
    def test_identical_observations_not_duplicate_ids(self):
        P.validate_provenance(['a','b'],['c','d'],'seedA','seedB',{'U':-1,'X':0})
    def test_original_two_stage_floor_is_not_universal(self):
        # A component can be below eps despite a larger planning slack.
        # In the original sequence k=2 still stops because its fitted difference is exactly zero.
        slack=2*math.sqrt(math.log(2*289/.02)/(2*8192))+math.sqrt(2*math.log(1/.03)/8192)
        self.assertGreater(slack,P.EPS)
        self.assertLess(-.1+slack,P.EPS)
    def test_fixed_sequence_event_subset_exhaustive(self):
        # Validity is an event-inclusion proof, not an independence assumption.
        import itertools
        for true_C in (0,1,2,3):
            for bit in itertools.product((False,True),repeat=6):
                outcomes=np.array(bit).reshape(2,3)
                b={t:{k:(0 if outcomes[t-1,2-k] else 1) for k in (2,1,0)} for t in (1,2)}
                c=P.decide(b,dimensions=(2,1,0),epsilon=.5)
                if c is not None and c<true_C:
                    k0=min(true_C,3)-1
                    self.assertTrue(all(b[t][k0]<.5 for t in (1,2)))
    def test_zero_event_interval(self):
        self.assertAlmostEqual(P.zero_event_upper(2000),.0014967448951882965)

# One separate test for each rejected invalid input, rather than merely naming hazards.
class TestInvalidInputs(unittest.TestCase):
    def setUp(self):
        self.c=P.draw_counts(np.random.default_rng(12),100,.5)[1]
        self.i=P.fit(self.c)[1]
    def test_negative_counts(self):
        c=self.c.copy();c[0,0]=-1
        with self.assertRaises(ValueError):P.validate_counts(c)
    def test_fractional_counts(self):
        with self.assertRaises(ValueError):P.validate_counts(self.c.astype(float))
    def test_nan_counts(self):
        c=self.c.astype(float);c[0,0]=np.nan
        with self.assertRaises(ValueError):P.validate_counts(c)
    def test_empty_counts(self):
        with self.assertRaises(ValueError):P.validate_counts(np.zeros((4,2),dtype=int))
    def test_size_mismatch(self):
        with self.assertRaises(ValueError):P.validate_counts(self.c,32768)
    def test_bad_shape(self):
        with self.assertRaises(ValueError):P.validate_counts(np.ones((8,),dtype=int))
    def test_out_of_range_index(self):
        with self.assertRaises(ValueError):P.check_member(10**6,1)
    def test_negative_index(self):
        with self.assertRaises(ValueError):P.check_member(-1,2)
    def test_boolean_index(self):
        with self.assertRaises(ValueError):P.check_member(True,1)
    def test_false_state_member(self):
        i=next(i for i in range(P.D.size) if i not in P.D.classes[1])
        with self.assertRaises(ValueError):P.check_member(i,1)
    def test_unknown_dimension(self):
        with self.assertRaises(ValueError):P.fit(self.c,(3,))
    def test_small_n(self):
        c=np.zeros((4,2),int);c[0,0]=1
        with self.assertRaises(ValueError):P.certificate(c,144,1,'max_eb')
    def test_bad_alpha(self):
        for a in (0,1,-.1,1.1,np.nan,True):
            with self.assertRaises(ValueError):P.certificate(self.c,self.i,1,'max_eb',a)
    def test_missing_training(self):
        with self.assertRaises(ValueError):P.certificate(self.c,self.i,1,'two_stage')
    def test_bad_method(self):
        with self.assertRaises(ValueError):P.certificate(self.c,self.i,1,'pick_best_after_test')
    def test_empty_horizons(self):
        with self.assertRaises(ValueError):P.decide({})
    def test_adaptive_order(self):
        with self.assertRaises(ValueError):P.decide({1:{1:0,2:0},2:{1:0,2:0}},dimensions=(1,2))
    def test_missing_horizon(self):
        with self.assertRaises(ValueError):P.decide({1:{1:0,2:0}})
    def test_nan_bound(self):
        with self.assertRaises(ValueError):P.decide({1:{1:np.nan,2:0},2:{1:0,2:0}})
    def test_invalid_law(self):
        with self.assertRaises(ValueError):P.law(.75,1,.5)
    def test_unknown_horizon(self):
        with self.assertRaises(ValueError):P.law(.5,3)
    def test_dictionary_missing_state_class(self):
        cls=dict(P.D.classes);cls[2]=np.array([0],dtype=int)
        with self.assertRaises(ValueError):P.validate_dictionary(dataclasses.replace(P.D,classes=cls))
    def test_corrupt_loss(self):
        loss=P.D.losses.copy();loss[0,0,0]=1.1
        with self.assertRaises(ValueError):P.validate_dictionary(dataclasses.replace(P.D,losses=loss))
    def test_mismatched_horizon_samples(self):
        b=P.draw_counts(np.random.default_rng(1),100,.5);b[2]=b[2][::-1].copy()
        with self.assertRaises(ValueError):P.validate_shared_horizons(b)

class TestProvenance(unittest.TestCase):
    def check_bad(self,**kwargs):
        args=dict(train_ids=['a','b'],test_ids=['c','d'],train_seed='s1',test_seed='s2',available_times={'X':0,'U':-1})
        args.update(kwargs)
        with self.assertRaises(ValueError):P.validate_provenance(**args)
    def test_duplicate_episode_ids(self):self.check_bad(train_ids=['a','a'])
    def test_train_test_overlap(self):self.check_bad(test_ids=['b','c'])
    def test_future_feature(self):self.check_bad(available_times={'X':0,'Y_next':1})
    def test_test_selected_dictionary(self):self.check_bad(dictionary_exposure='test_data')
    def test_reused_seed(self):self.check_bad(test_seed='s1')
    def test_prior_confirmatory_seed(self):self.check_bad(prior_seed_keys=['s2'])
    def test_unsupported_dependence(self):self.check_bad(group_law='spatial_blocks_from_one_box')
    def test_unequal_weights(self):self.check_bad(weighting='population_size')
    def test_empty_partition(self):self.check_bad(test_ids=[])
    def test_no_feature_provenance(self):self.check_bad(available_times={})

if __name__=='__main__': unittest.main(verbosity=2)
