from pathlib import Path
import sys,unittest,math,json,copy,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import dependence_core as C
P=C.P
class TestDependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=C.draw_group_counts(np.random.default_rng(1234),64,16,.5,.5)
        cls.b=C.draw_group_counts(np.random.default_rng(4321),64,16,.5,.5)
        cls.tr=C.group_stats(cls.a);cls.te=C.group_stats(cls.b)
        cls.fit=C.fit_groups(cls.tr)
    def test_canonical_177(self):self.assertEqual(C.D.size,177)
    def test_parent_unchanged_289(self):self.assertEqual(P.D.size,289)
    def test_group_shape(self):self.assertEqual(self.a.shape,(2,64,4,2))
    def test_derived_n(self):self.assertEqual((self.te.G,self.te.m),(64,16))
    def test_fixed_world_totals(self):self.assertTrue(np.all(self.a.sum(axis=(2,3))==16))
    def test_shared_horizon_inputs(self):np.testing.assert_array_equal(self.a[0].sum(axis=2),self.a[1].sum(axis=2))
    def test_c0_counts(self):self.assertEqual(C.group_stats(C.draw_group_counts(np.random.default_rng(3),12,7,.5,0)).m,7)
    def test_c1_pure_duplicates(self):
        x=C.draw_group_counts(np.random.default_rng(4),12,9,.5,1)
        self.assertTrue(np.all((x==0)|(x==9)))
    def test_seed_roles_disjoint(self):
        case={'theta':'.5','copy_probability':'1','groups_per_split':64,'rows_per_group':16}
        self.assertNotEqual(C.stream_seed(case,0,'train',123),C.stream_seed(case,0,'test',123))
    def test_c1_seed_paired_over_m(self):
        a={'theta':'.5','copy_probability':'1','groups_per_split':64,'rows_per_group':1};b=dict(a,rows_per_group=256)
        self.assertEqual(C.stream_seed(a,3,'test',123),C.stream_seed(b,3,'test',123))
    def test_other_c_m_stream_distinct(self):
        a={'theta':'.5','copy_probability':'.5','groups_per_split':64,'rows_per_group':1};b=dict(a,rows_per_group=256)
        self.assertNotEqual(C.stream_seed(a,3,'test',123),C.stream_seed(b,3,'test',123))
    def test_duplicate_group_certificate_invariance(self):
        results=[]
        for m in (1,16,256):
            tr=C.draw_stats(np.random.default_rng(1),256,m,.5,1)
            te=C.draw_stats(np.random.default_rng(2),256,m,.5,1)
            results.append(C.all_bounds(te,C.fit_groups(tr))['group_max_eb'])
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
    def test_duplicate_fit_invariance(self):
        a=C.draw_stats(np.random.default_rng(55),64,1,.5,1);b=C.draw_stats(np.random.default_rng(55),64,256,.5,1)
        self.assertEqual(C.fit_groups(a),C.fit_groups(b))
    def test_one_row_matches_parent(self):
        tr=C.draw_stats(np.random.default_rng(1),256,1,.5,.5);te=C.draw_stats(np.random.default_rng(2),256,1,.5,.5)
        f=C.fit_groups(tr);b=C.all_bounds(te,f)
        for h in (1,2):
            for k in (2,1):
                expected=P.certificate(te.total_counts[h-1],f[h][k],k,'max_eb',d=C.D)
                self.assertAlmostEqual(b['group_max_eb'][h][k],expected,places=13)
    def test_one_row_group_row_equal(self):
        tr=C.draw_stats(np.random.default_rng(1),128,1,.5,.5);te=C.draw_stats(np.random.default_rng(2),128,1,.5,.5)
        b=C.all_bounds(te,C.fit_groups(tr))
        for h in (1,2):
            for k in (2,1):self.assertAlmostEqual(b['group_max_eb'][h][k],b['pooled_row_max_eb'][h][k],places=13)
    def test_group_mean_matches_expanded_rows(self):
        for h in (0,1):
            expanded=[]
            for g in self.b[h]:
                for cell in range(4):
                    for y in (0,1):expanded.extend([float(C.D.losses[1,cell,y])]*int(g[cell,y]))
            s=self.b[h].sum(axis=0)
            self.assertAlmostEqual(float(np.mean(expanded)),P.mean_losses(s,C.D)[1],places=14)
    def test_group_variance_matches_direct_losses(self):
        for h in (0,1):
            ix=self.fit[h+1][1];mu,var,rv=C.pair_moments(self.te,ix,h)
            df=C.LOSS[ix][None,:]-C.LOSS
            scores=self.b[h].reshape(64,8)/16@df.T
            np.testing.assert_allclose(mu,scores.mean(axis=0),atol=1e-14)
            np.testing.assert_allclose(var,scores.var(axis=0,ddof=1),atol=1e-14)
    def test_row_variance_matches_parent(self):
        for h in (0,1):
            ix=self.fit[h+1][1];mu,var,rv=C.pair_moments(self.te,ix,h)
            m2,v2=P.diff_moments(self.te.total_counts[h],ix,C.D)
            np.testing.assert_allclose(mu,m2,atol=1e-14);np.testing.assert_allclose(rv,v2,atol=1e-14)
    def test_boundary_exact(self):
        f=C.oracle_facts(.5,.5,64,16)
        self.assertEqual(f['mean_fraction'],'1/16');self.assertEqual(f['row_variance_fraction'],'3/64')
    def test_c0_variance_reduction(self):
        f=C.oracle_facts(.5,0,64,16);self.assertEqual(f['variance_equivalent_count_diagnostic_only'],1024)
    def test_c1_variance_floor(self):
        a=C.oracle_facts(.5,1,64,1);b=C.oracle_facts(.5,1,64,256)
        self.assertEqual(a['grand_mean_variance_fraction'],b['grand_mean_variance_fraction'])
    def test_analytic_variance_factor(self):
        f=C.oracle_facts(.5,.5,64,16);self.assertEqual(f['design_effect'],4.75)
        self.assertAlmostEqual(f['grand_mean_variance'],(3/64)*4.75/(64*16))
    def test_offgrid_exact_risk(self):self.assertEqual(P.exact_truth(.48,d=C.D)['gap_fractions'][1][1],'23/400')
    def test_real_data_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['te1','te2'],law='real_cosmology')
    def test_shared_phases_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['te1','te2'],law='iid_copy_mixture_realizations',phase_shared=True)
    def test_duplicate_ids_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['te','te'],law='iid_copy_mixture_realizations')
    def test_split_overlap_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['te','tr'],law='iid_copy_mixture_realizations')
    def test_seed_reuse_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['t1','t2'],law='iid_copy_mixture_realizations',train_seed='same',test_seed='same')
    def test_future_feature_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['t1','t2'],law='iid_copy_mixture_realizations',future_feature=True)
    def test_test_dictionary_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['t1','t2'],law='iid_copy_mixture_realizations',dictionary_exposure='test')
    def test_unequal_weights_blocked(self):
        with self.assertRaises(ValueError):C.validate_provenance(['tr'],['t1','t2'],law='iid_copy_mixture_realizations',weights='pooled_size_biased')
    def test_empty_counts_blocked(self):
        with self.assertRaises(ValueError):C.group_stats(np.zeros((2,0,4,2),dtype=int))
    def test_one_group_blocked(self):
        with self.assertRaises(ValueError):C.group_stats(np.zeros((2,1,4,2),dtype=int))
    def test_zero_size_group_blocked(self):
        x=self.a.copy();x[:,0]=0
        with self.assertRaises(ValueError):C.group_stats(x)
    def test_unequal_sizes_blocked(self):
        x=self.a.copy();x[:,0]*=2
        with self.assertRaises(ValueError):C.group_stats(x)
    def test_negative_count_blocked(self):
        x=self.a.copy();x[0,0,0,0]=-1
        with self.assertRaises(ValueError):C.group_stats(x)
    def test_float_counts_blocked(self):
        with self.assertRaises(ValueError):C.group_stats(self.a.astype(float))
    def test_horizon_mismatch_blocked(self):
        x=self.a.copy();x[1]=x[1,::-1]
        with self.assertRaises(ValueError):C.group_stats(x)
    def test_unregistered_horizon_blocked(self):
        with self.assertRaises(ValueError):C.pair_moments(self.te,0,2)
    def test_nonmember_index_blocked(self):
        f=copy.deepcopy(self.fit);f[1][1]=176
        with self.assertRaises(ValueError):C.all_bounds(self.te,f)
    def test_nan_probability_blocked(self):
        with self.assertRaises(ValueError):C.copy_probability(float('nan'))
    def test_probability_range_blocked(self):
        for x in (-.1,1.1,True):
            with self.assertRaises(ValueError):C.copy_probability(x)
    def test_eb_n_invalid(self):
        with self.assertRaises(ValueError):C.eb_from_moments(np.array([0.]),np.array([0.]),1)
    def test_eb_missing_moment(self):
        with self.assertRaises(ValueError):C.eb_from_moments(np.array([]),np.array([]),10)
    def test_eb_negative_var(self):
        with self.assertRaises(ValueError):C.eb_from_moments(np.array([0.]),np.array([-.1]),10)
    def test_row_invalid_label(self):self.assertEqual(C.row_control_status(.5,16),'INVALID_ROW_IID_CONTROL')
    def test_row_valid_at_zero_copy(self):self.assertEqual(C.row_control_status(0,16),'IID_ROW_REFERENCE')
    def test_row_valid_one_per_group(self):self.assertEqual(C.row_control_status(1,1),'IID_ROW_REFERENCE')
    def test_valid_public_api(self):
        ans=C.certify_group_counts(self.a,self.b,[f'a{i}' for i in range(64)],[f'b{i}' for i in range(64)],law='iid_copy_mixture_realizations')
        self.assertEqual(ans['independent_units'],64);self.assertEqual(ans['rows'],1024);self.assertFalse(ans['cosmology_admitted'])
    def test_public_wrong_ids_length(self):
        with self.assertRaises(ValueError):C.certify_group_counts(self.a,self.b,['a'],['b','c'],law='iid_copy_mixture_realizations')
    def test_plan_frozen_62_cases(self):
        plan=json.loads((ROOT/'contracts/development_plan.json').read_text());self.assertEqual(len(plan['cases']),62)
        self.assertFalse(plan['authorisation']['programme_enactment'])
if __name__=='__main__':unittest.main()
