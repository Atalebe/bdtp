from pathlib import Path
from dataclasses import replace
import itertools,math,sys,unittest
import numpy as np
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import inference_core as I

class TestPath(unittest.TestCase):
    def test_bank_direct_prefixes(self):
        rng=np.random.default_rng(7101)
        for i in range(20):
            t=np.cumsum(rng.uniform(.01,.09,size=12));u=rng.uniform(-.08,.08,size=12)
            for n in range(1,13):
                num,den,m=I.recursive_bank(t[:n],u[:n])
                np.testing.assert_allclose(m,[I.direct_coordinate(t[:n],u[:n],g)for g in I.GAMMAS],atol=2e-14,rtol=0)
    def test_origin_cancels(self):
        self.assertEqual(I.direct_coordinate([.1,.5],[.1,.3],1.8,1.),I.direct_coordinate([.1,.5],[.1,.3],1.8,4.))
    def test_future_blocked(self):
        with self.assertRaises(ValueError):I.direct_coordinate([.1,.5],[.1,.2],1.,.4)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):I.recursive_bank([],[])
    def test_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):I.recursive_bank([1,1],[1,2])
    def test_negative_gamma_rejected(self):
        with self.assertRaises(ValueError):I.direct_coordinate([1],[1],-1)
    def test_nonfinite_increment_rejected(self):
        with self.assertRaises(ValueError):I.direct_coordinate([1],[np.nan],1)
    def test_constant_cadence(self):
        for n in [4,8,16]:self.assertAlmostEqual(I.direct_coordinate(np.arange(1,n+1)/n,np.full(n,1/n),1.8),1/n)
    def test_denominator_not_omittable(self):
        self.assertNotEqual((.2+1)/2,(2*.2+1)/3)
    def test_no_samples_is_not_zero_increment(self):
        self.assertNotEqual(I.direct_coordinate([.1],[.4],2),I.direct_coordinate([.1,.2],[.4,0],2))
    def test_gamma_change_not_reuse(self):
        t=np.array([.1,.2,.8]);u=np.array([.03,.08,.01]);num,den,_=I.recursive_bank(t[:2],u[:2],[.5])
        wrong=(np.exp(-4.5*.6)*num[0]+u[2])/(np.exp(-4.5*.6)*den[0]+1)
        self.assertGreater(abs(wrong-I.direct_coordinate(t,u,4.5)),1e-6)
    def test_derivative(self):
        t=np.array([.1,.3,.7]);u=np.array([.03,.08,.01]);g=1.8
        w=np.exp(g*t);w/=w.sum();d=w@(t*u)-(w@t)*(w@u)
        fd=(I.direct_coordinate(t,u,g+1e-5)-I.direct_coordinate(t,u,g-1e-5))/2e-5
        self.assertAlmostEqual(d,fd,places=11)

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=I.FiniteModel();cls.ix=cls.m.parameter_index(.3,8,1.8)
        cls.tr=cls.m.draw(np.random.default_rng(882),8,4,cls.ix,'training','tr')
    def test_grid_60(self):self.assertEqual(len(self.m.params),60)
    def test_prior_normalized(self):self.assertAlmostEqual(self.m.prior.sum(),1)
    def test_support_192(self):self.assertEqual(len(self.m.latent),192)
    def test_censored_support_104(self):self.assertEqual(len(I.FiniteModel(observation='mass_truncated').latent),104)
    def test_summary_partition(self):self.assertEqual(len(self.m.summary_groups),36)
    def test_summary_has_current_structure(self):
        for ix in self.m.summary_groups:
            r=self.m.representatives[ix]
            self.assertEqual(len(set(self.m.c[r])),1)
            self.assertLess(np.ptp(self.m.u[r,-1]),1e-14)
            self.assertLess(np.ptp(self.m.u[r].sum(1)),1e-12)
    def test_truncation_masks(self):
        m=I.FiniteModel(observation='mass_truncated')
        for c,n in [(-1,3),(0,5),(1,6)]:self.assertTrue(np.all(m.obs_mask[m.obs_c==c].sum(1)==n))
    def test_posterior_normalization(self):
        p=self.m.posterior(self.tr);self.assertAlmostEqual(p.sum(),1);self.assertTrue(np.all(p>=0))
    def test_gamma_flat_constant(self):
        m=I.FiniteModel('constant_gamma');ix=m.parameter_index(.3,8,1.8)
        tr=m.draw(np.random.default_rng(7),32,4,ix,'training','a');p=m.posterior(tr)
        for g in I.GAMMAS:self.assertAlmostEqual(p[m.params[:,2]==g].sum(),1/6,places=13)
    def test_summary_model_equivalence(self):
        m=I.FiniteModel('summary_only');a=m.parameter_index(.3,0,1.8);b=m.parameter_index(0,8,1.8)
        # Equal horizon-1 response; horizon-2 deliberately distinguishes it.
        np.testing.assert_allclose(m.mu[0,:,a],m.mu[0,:,b],atol=1e-16)
        self.assertGreater(np.max(abs(m.mu[1,:,a]-m.mu[1,:,b])),.01)
    def test_summary_no_information_beyond_current(self):
        m=I.FiniteModel('summary_only');np.testing.assert_allclose(m.p,m.p_summary,atol=1e-15)
    def test_zero_amplitude_gamma_equal(self):
        a=self.m.parameter_index(.3,0,.5);b=self.m.parameter_index(.3,0,4.5)
        np.testing.assert_array_equal(self.m.mu[:,:,a],self.m.mu[:,:,b])
    def test_marginalization_preserves_mean_risk(self):
        m=I.FiniteModel(observation='mass_truncated')
        a=self.m.parameter_index(.3,8,1.8)
        np.testing.assert_allclose(self.m.obs_prob@self.m.p[0,:,a],m.obs_prob@m.p[0,:,a],atol=1e-14)
    def test_missing_has_bayes_information_cost(self):
        m=I.FiniteModel(observation='mass_truncated');i=self.ix
        rc=self.m.exact_brier_risk(self.m.p[:,:,i],i);rt=m.exact_brier_risk(m.p[:,:,i],i)
        self.assertTrue(np.all(rt>=rc-1e-15))
    def test_shared_vs_iid_same_one_object(self):
        tr=self.m.draw(np.random.default_rng(1),4,1,self.ix,'training','x')
        np.testing.assert_allclose(self.m.log_likelihood(tr,'shared'),self.m.log_likelihood(tr,'iid_wrong'),atol=1e-12)
    def test_shared_vs_iid_not_same_four_objects(self):
        self.assertGreater(np.max(abs(self.m.log_likelihood(self.tr)-self.m.log_likelihood(self.tr,'iid_wrong'))),1.)
    def test_zero_offset_restores_iid(self):
        m=I.FiniteModel(offsets=[0.],offset_probs=[1.]);tr=m.draw(np.random.default_rng(1),4,4,self.ix,'training','x')
        np.testing.assert_allclose(m.log_likelihood(tr),m.log_likelihood(tr,'iid_wrong'),atol=1e-12)
    def test_marginal_shared_offset_variance(self):
        variance=np.sum(I.OFFSET_PROBS*I.OFFSETS**2)
        self.assertAlmostEqual(variance,.0072)
        self.assertGreater(variance+I.SIGMA**2/16,(variance+I.SIGMA**2)/16)
    def test_true_grid_required(self):
        with self.assertRaises(ValueError):self.m.parameter_index(.3,7,1.8)
    def test_bad_observation_law(self):
        with self.assertRaises(ValueError):I.FiniteModel(observation='unrecorded')
    def test_malformed_noise(self):
        with self.assertRaises(ValueError):I.FiniteModel(sigma=0)
    def test_recursion_forecast_matches(self):
        for design,obs in [('excited','complete'),('excited','mass_truncated'),('summary_only','complete'),('constant_gamma','complete')]:
            m=I.FiniteModel(design,obs);tr=m.draw(np.random.default_rng(18),4,2,self.ix,'training','a');p=m.posterior(tr)
            np.testing.assert_allclose(m.forecast(p),m.retained_state_forecast(p),atol=2e-14,rtol=0)
    def test_prefix_no_future_data(self):
        p=self.m.posterior(self.tr)
        changed=replace(self.tr,responses=self.tr.responses.copy());changed.responses[:,:,1]+=999
        # likelihood uses horizon 1 only; withheld horizon 2 has no influence
        np.testing.assert_array_equal(p,self.m.posterior(changed))
    def test_test_role_cannot_fit_posterior(self):
        with self.assertRaises(ValueError):self.m.posterior(replace(self.tr,role='test'))
    def test_latent_truth_not_inference_input(self):
        changed=replace(self.tr,hidden_support=np.zeros_like(self.tr.hidden_support),hidden_offset=np.full_like(self.tr.hidden_offset,999.))
        np.testing.assert_array_equal(self.m.posterior(self.tr),self.m.posterior(changed))
    def test_empty_batch_rejected(self):
        bad=replace(self.tr,observed=np.zeros((0,4),dtype=int),responses=np.zeros((0,4,2)),world_ids=())
        with self.assertRaises(ValueError):self.m.posterior(bad)
    def test_duplicate_world_blocked(self):
        with self.assertRaises(ValueError):self.m.posterior(replace(self.tr,world_ids=('x',)*8))
    def test_law_mismatch_blocked(self):
        with self.assertRaises(ValueError):I.FiniteModel('constant_gamma').posterior(self.tr)
    def test_prior_transport(self):
        ll=self.m.log_likelihood(self.tr);pr=self.m.prior
        alternate=np.arange(1,61,dtype=float);alternate/=alternate.sum()
        pa=I.normalize_log_weights(ll,alternate)
        restored=pa*pr/alternate;restored/=restored.sum()
        np.testing.assert_allclose(restored,I.normalize_log_weights(ll,pr),atol=2e-14)
    def test_bad_prior_blocked(self):
        with self.assertRaises(ValueError):I.normalize_log_weights([1,2],[.1,.1])
    def test_no_likelihood_support_blocked(self):
        with self.assertRaises(ValueError):I.normalize_log_weights([-np.inf,-np.inf],[.5,.5])
    def test_pit_ties_uniform(self):
        self.assertAlmostEqual(I.randomized_pit([1,1],[.3,.7],0,.412),.412)
    def test_pit_atoms(self):
        self.assertAlmostEqual(I.randomized_pit([0,1,2],[.2,.5,.3],1,.4),.4)
    def test_conditional_risk_decomposition(self):
        q=self.m.forecast(self.m.prior);p=self.m.p[:,:,self.ix]
        risk=self.m.exact_brier_risk(q,self.ix);br=self.m.exact_brier_risk(p,self.ix)
        np.testing.assert_allclose(risk-br,np.sum((q-p)**2*self.m.obs_prob[None,:],axis=1),atol=1e-15)

class TestLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=I.FiniteModel();cls.ix=cls.m.parameter_index(.3,8,1.8)
        cls.tr=cls.m.draw(np.random.default_rng(4),8,4,cls.ix,'training','train')
        cls.te=cls.m.draw(np.random.default_rng(5),128,4,cls.ix,'test','test')
        cls.lib=I.make_library(cls.m,cls.tr)
    def test_nested_library(self):
        self.assertTrue(set(self.lib.classes[1])<=set(self.lib.classes[2])<=set(self.lib.classes[3]))
    def test_131_predictors(self):self.assertEqual(len(self.lib.names),131)
    def test_nonwriteable_predictors(self):self.assertFalse(self.lib.predictions.flags.writeable)
    def test_class_membership(self):
        for k,ix in self.lib.chosen.items():self.assertTrue(set(ix)<=set(self.lib.classes[k]))
    def test_test_fitting_blocked(self):
        with self.assertRaises(ValueError):I.make_library(self.m,self.te)
    def test_overlap_blocked(self):
        bad=replace(self.te,world_ids=self.tr.world_ids+tuple(f'te{i}'for i in range(120)))
        with self.assertRaises(ValueError):I.validate_split(self.m,self.tr,bad)
    def test_positive_split(self):I.validate_split(self.m,self.tr,self.te)
    def test_exact_nesting_gap(self):
        r=self.m.exact_brier_risk(self.lib.predictions,self.ix)
        gaps=[r[:,list(self.lib.classes[k])].min(axis=1)for k in (1,2,3)]
        self.assertTrue(np.all(gaps[0]>=gaps[1]-1e-14));self.assertTrue(np.all(gaps[1]>=gaps[2]-1e-14))
    def test_bayes_in_dictionary(self):
        r=self.m.exact_brier_risk(self.lib.predictions,self.ix).min(axis=1)
        np.testing.assert_allclose(r,self.m.exact_brier_risk(self.m.p[:,:,self.ix],self.ix),atol=1e-14)
    def test_bounds_no_small_group_certificate(self):
        e=I.evaluate_library(self.m,self.lib,self.te,self.ix)
        self.assertIsNone(e['certified_level'])
    def test_counter_uses_worlds_not_rows(self):
        e=I.evaluate_library(self.m,self.lib,self.te,self.ix)
        self.assertEqual(e['independent_worlds'],128);self.assertEqual(e['rows'],512)
    def test_exact_group_duplication_invariance(self):
        te=replace(self.te,observed=np.repeat(self.te.observed,3,axis=1),responses=np.repeat(self.te.responses,3,axis=1))
        a=I.evaluate_library(self.m,self.lib,self.te,self.ix);b=I.evaluate_library(self.m,self.lib,te,self.ix)
        for k in a['bounds']:np.testing.assert_allclose(a['bounds'][k],b['bounds'][k],atol=5e-14,rtol=0)
    def test_unchanged_dep_formula(self):
        mu=np.array([0,.04,-.06]);v=np.array([0,.015,.013]);g=4096;alpha=.05;z=math.log(2/alpha)
        expected=max(mu+np.sqrt(2*v*z/g)+14*z/(3*(g-1)))
        self.assertAlmostEqual(I.scalar_eb(mu,v,g,alpha),expected,places=14)
    def test_zero_groups_blocked(self):
        with self.assertRaises(ValueError):I.scalar_eb([0],[0],0)
    def test_zero_alpha_blocked(self):
        with self.assertRaises(ValueError):I.scalar_eb([0],[0],10,0)
    def test_nan_moments_blocked(self):
        with self.assertRaises(ValueError):I.scalar_eb([np.nan],[0],10)
    def test_negative_variance_blocked(self):
        with self.assertRaises(ValueError):I.scalar_eb([0],[-.1],10)
    def test_empty_moments_blocked(self):
        with self.assertRaises(ValueError):I.scalar_eb([],[],10)
    def test_seed_roles_disjoint(self):
        seeds=[I.seed_for(2026,'a',i,r)for i in range(10)for r in ('training','test','inverse')]
        self.assertEqual(len(seeds),len(set(seeds)))

if __name__=='__main__':unittest.main()
