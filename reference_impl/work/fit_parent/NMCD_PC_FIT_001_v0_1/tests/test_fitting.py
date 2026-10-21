import unittest,sys,math,itertools
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.special import logsumexp
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import fitting_core as F
I=F.I;E=F.E

class LikelihoodTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=I.FiniteModel(observation='mass_truncated');cls.t=cls.m.parameter_index(.3,16.,1.8)
  cls.b=cls.m.draw(np.random.default_rng(719),4,2,cls.t,'training','likelihood_fixture')
 def test_single_horizon_matches_parent(self):
  np.testing.assert_allclose(F.joint_log_likelihood(self.m,self.b,(0,)),self.m.log_likelihood(self.b),rtol=0,atol=3e-12)
 def test_direct_joint_finite_sum(self):
  m=self.m;b=self.b;ll=[]
  for p in range(60):
   total=0.
   for g in range(len(b.world_ids)):
    sb=0.
    for off,w in zip(m.offsets,m.offset_probs):
     pp=1.
     for i,o in enumerate(b.observed[g]):
      inner=0.
      for u in m.latent[o]:
       z=1.
       for h in (0,1):
        d=(b.responses[g,i,h]-m.mu[h,u,p]-off)/m.sigma
        z*=math.exp(-d*d/2)/(m.sigma*math.sqrt(2*math.pi))
       inner+=z/len(m.latent[o])
      pp*=inner
     sb+=w*pp
    total+=math.log(sb)
   ll.append(total)
  np.testing.assert_allclose(ll,F.joint_log_likelihood(m,b),rtol=0,atol=5e-12)
 def test_two_horizons_not_marginal_sum(self):
  self.assertGreater(np.max(np.abs(F.joint_log_likelihood(self.m,self.b)-2*F.joint_log_likelihood(self.m,self.b,(0,)))),1.)
 def test_joint_horizon2_matters(self):
  y=self.b.responses.copy();y[:,:,1]+=.1
  self.assertGreater(np.max(np.abs(F.joint_log_likelihood(self.m,self.b)-F.joint_log_likelihood(self.m,replace(self.b,responses=y)))),.1)
 def test_shared_not_independent_offset(self):
  self.assertGreater(np.max(np.abs(F.joint_log_likelihood(self.m,self.b)-F.joint_log_likelihood(self.m,self.b,shared=False))),.1)
 def test_no_test_input(self):
  with self.assertRaises(ValueError):F.joint_log_likelihood(self.m,replace(self.b,role='test'))
 def test_no_inverse_role(self):
  with self.assertRaises(ValueError):F.joint_log_likelihood(self.m,replace(self.b,role='inverse'))
 def test_empty_horizons(self):
  with self.assertRaises(ValueError):F.joint_log_likelihood(self.m,self.b,())
 def test_hidden_data_unused(self):
  b=replace(self.b,hidden_support=np.full_like(self.b.hidden_support,-100),hidden_offset=np.full_like(self.b.hidden_offset,np.nan))
  np.testing.assert_array_equal(F.joint_log_likelihood(self.m,b),F.joint_log_likelihood(self.m,self.b))
 def test_gamma_constant_identical(self):
  m=I.FiniteModel(design='constant_gamma');b=m.draw(np.random.default_rng(98),3,2,0,'training','constant')
  a=F.joint_log_likelihood(m,b).reshape(2,5,6)
  np.testing.assert_array_equal(a,np.repeat(a[:,:,:1],6,axis=2))
 def test_prior_mass_preserved(self):
  p=I.normalize_log_weights(np.zeros(60),self.m.prior);np.testing.assert_allclose(p,1/60,atol=2e-17,rtol=0)
 def test_nonfinite_response(self):
  y=self.b.responses.copy();y[0,0,0]=np.nan
  with self.assertRaises(ValueError):F.joint_log_likelihood(self.m,replace(self.b,responses=y))

class FitTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=I.FiniteModel(observation='mass_truncated');cls.t=cls.m.parameter_index(.3,16.,1.8)
  cls.tr=cls.m.draw(np.random.default_rng(1001),12,4,cls.t,'training','fit_fixture')
  cls.lib=I.make_library(cls.m,cls.tr);cls.fit=F.fit_rules(cls.m,cls.lib,cls.tr)
  cls.te=cls.m.draw(np.random.default_rng(1002),64,4,cls.t,'test','test_fixture')
 def test_sizes_unchanged(self):self.assertEqual([len(self.lib.classes[k])for k in (1,2,3)],[9,70,131])
 def test_baseline_exact_choices(self):
  for k in (1,2,3):self.assertEqual(tuple(self.fit.choices['erm'][k]),self.lib.chosen[k])
 def test_membership(self):
  for method,ch in self.fit.choices.items():
   for k in (1,2,3):self.assertTrue(set(ch[k])<=set(self.lib.classes[k]))
 def test_scores_are_minimized(self):
  for method in self.fit.choices:
   for k,ix in self.lib.classes.items():
    for h in (0,1):
     sc=self.fit.scores[method][h];sc=sc[k]if method=='variance_penalized'else sc
     self.assertLessEqual(sc[self.fit.choices[method][k][h]]-min(sc[list(ix)]),2e-12)
 def test_no_extra_predictor(self):self.assertEqual(len(self.lib.names),131)
 def test_library_immutable(self):self.assertFalse(self.lib.predictions.flags.writeable)
 def test_posterior_expected_risk_linear_mixture(self):
  for h in (0,1):
   risks=np.stack([self.m.exact_brier_risk(self.lib.predictions,p)[h]for p in range(60)])
   np.testing.assert_allclose(F.expected_brier(self.m,self.lib,self.fit.posterior_joint,h),self.fit.posterior_joint@risks,atol=3e-15,rtol=0)
 def test_point_prior_recovers_known_risk(self):
  p=np.zeros(60);p[self.t]=1
  np.testing.assert_allclose(F.expected_brier(self.m,self.lib,p,0),self.m.exact_brier_risk(self.lib.predictions,self.t)[0],atol=3e-15,rtol=0)
 def test_oracle_arrays_cannot_change_fit(self):
  tr=replace(self.tr,hidden_support=np.full_like(self.tr.hidden_support,0),hidden_offset=np.full_like(self.tr.hidden_offset,1e6))
  ff=F.fit_rules(self.m,self.lib,tr);self.assertEqual(ff.choices,self.fit.choices)
 def test_test_labels_cannot_change_fit(self):
  y=self.te.responses.copy();y[:]=-y
  a=F.evaluation_from_cov(self.m,self.lib,self.fit,replace(self.te,responses=y))
  self.assertEqual(F.fit_rules(self.m,self.lib,self.tr).choices,self.fit.choices)
 def test_test_fit_blocked(self):
  with self.assertRaises(ValueError):F.fit_rules(self.m,self.lib,self.te)
 def test_wrong_training_ids(self):
  with self.assertRaises(ValueError):F.fit_rules(self.m,self.lib,replace(self.tr,world_ids=tuple('other'+x for x in self.tr.world_ids)))
 def test_group_losses_direct(self):
  got=F.world_losses(self.m,self.lib,self.te,0)
  want=((self.lib.predictions[0,self.te.observed]-(self.te.responses[:,:,0]>0)[...,None])**2).mean(1)
  np.testing.assert_array_equal(got,want)
 def test_duplicate_world_rejected(self):
  with self.assertRaises(ValueError):F.world_losses(self.m,self.lib,replace(self.te,world_ids=(self.te.world_ids[0],)*64),0)
 def test_train_test_overlap(self):
  ids=list(self.te.world_ids);ids[0]=self.tr.world_ids[0]
  with self.assertRaises(ValueError):F.evaluation_from_cov(self.m,self.lib,self.fit,replace(self.te,world_ids=tuple(ids)))
 def test_mutated_dictionary_rejected(self):
  q=self.lib.predictions.copy();q[0,0,0]+=.01
  with self.assertRaises(ValueError):F.evaluation_from_cov(self.m,replace(self.lib,predictions=q),self.fit,self.te)
 def test_all_methods_share_covariance(self):
  ans=F.evaluation_from_cov(self.m,self.lib,self.fit,self.te)
  for method in F.METHODS:
   ref=F.evaluate_isolated(self.m,self.lib,self.fit,self.te,method)
   self.assertEqual(ref['certified_level'],ans[method]['certified_level'])
   for k in (1,2,3):np.testing.assert_allclose(ref['bounds'][k],ans[method]['bounds'][k],atol=3e-12,rtol=0)
 def test_parent_eval_class_parity(self):
  a=F.evaluation_from_cov(self.m,self.lib,self.fit,self.te);b=E.evaluate_batch(self.m,self.lib,self.te)
  for k in (1,2,3):
   np.testing.assert_allclose(a['evaluation_class']['bounds'][k],b['reviewer_class_min']['bounds'][str(k)],atol=3e-12,rtol=0)
   np.testing.assert_allclose(a['erm']['bounds'][k],b['historical_fixed']['bounds'][str(k)],atol=3e-12,rtol=0)
 def test_prefix_preserves_original(self):
  p=F.prefix(self.tr,5);np.testing.assert_array_equal(p.responses,self.tr.responses[:5]);self.assertEqual(p.world_ids,self.tr.world_ids[:5])
 def test_prefix_outside(self):
  with self.assertRaises(ValueError):F.prefix(self.tr,13)
 def test_prefix_zero(self):
  with self.assertRaises(ValueError):F.prefix(self.tr,0)
 def test_prefix_bool(self):
  with self.assertRaises(ValueError):F.prefix(self.tr,True)
 def test_separate_seeds(self):
  self.assertNotEqual(F.seed_for(1,'a',0,'training_pool'),F.seed_for(1,'a',0,'test_pool'))
 def test_unknown_role(self):
  with self.assertRaises(ValueError):F.seed_for(1,'a',0,'train')
 def test_payload_counts(self):
  r=F.resource_payload(self.m,self.lib,64,512)
  self.assertEqual(r['candidate_object_loss_values_test'],2*512*4*131)
  self.assertEqual(r['full_covariance_products']//r['selected_cross_products_upper'],131//3)
 def test_hidden_is_not_a_covariate(self):self.assertNotIn('hidden',str(self.lib.names))

class DecisionTests(unittest.TestCase):
 def test_all_below(self):self.assertEqual(F.certify({1:[0.,0.],2:[0.,0.],3:[0.,0.]}),1)
 def test_equality_is_nonrejection(self):self.assertEqual(F.certify({1:[.01,0.],2:[0.,0.],3:[0.,0.]}),2)
 def test_top_fails_unresolved(self):self.assertIsNone(F.certify({1:[0.,0.],2:[0.,0.],3:[.02,0.]}))
 def test_missing_level(self):
  with self.assertRaises(ValueError):F.certify({2:[0.,0.],3:[0.,0.]})
 def test_empty_horizon(self):
  with self.assertRaises(ValueError):F.certify({1:[],2:[0.,0.],3:[0.,0.]})
 def test_nan_bound(self):
  with self.assertRaises(ValueError):F.certify({1:[0.,np.nan],2:[0.,0.],3:[0.,0.]})
 def test_members_union_required(self):self.assertGreater(1-(1-.05)**70,.05)
 def test_corrected_union_bound(self):self.assertLessEqual(1-(1-.05/70)**70,.05)
 def test_literal_budget_matching(self):
  import json
  p=json.loads((R/'contracts/development_plan.json').read_text())
  for n,t in itertools.product(p['world_budgets'],p['training_budgets']):self.assertGreater(n-t,0);self.assertEqual(t+(n-t),n)
 def test_plan_no_live_mutations(self):
  import json
  p=json.loads((R/'contracts/development_plan.json').read_text())
  self.assertFalse(any(p[k]for k in ['real_data','git_mutation','parent_mutation','mash_mutation']))
class StabilityTests(unittest.TestCase):
 def test_near_duplicate_variance_large_n(self):
  rng=np.random.default_rng(2301);x=rng.random((32704,6))*.8+.1
  x[:,1]=x[:,0]+1e-10*np.sin(np.arange(len(x)))
  d,v,n=F.stable_pair_moments(x)
  diff=x[:,0]-x[:,1]
  self.assertAlmostEqual(v[0,1],diff.var(ddof=1),delta=1e-28)
  self.assertAlmostEqual(d[0,1],diff.mean(),delta=1e-22)
 def test_exact_duplicate_variance_zero(self):
  rng=np.random.default_rng(2302);x=rng.random((16320,5));x[:,1]=x[:,0]
  d,v,n=F.stable_pair_moments(x)
  self.assertEqual(v[0,1],0.);self.assertEqual(d[0,1],0.)
 def test_stable_all_pair_direct_bound(self):
  rng=np.random.default_rng(2303);x=rng.random((1024,7));x[:,1]=x[:,0]+1e-11*(1-x[:,0])
  d,v,n=F.stable_pair_moments(x);a=F.bounds_from_pair_moments(d,v,n,.05)
  for p,q in itertools.product(range(7),repeat=2):
   dd=x[:,p]-x[:,q];z=math.log(40)
   direct=dd.mean()+math.sqrt(2*dd.var(ddof=1)*z/n)+14*z/(3*(n-1))
   self.assertAlmostEqual(a[p,q],direct,delta=3e-14)

if __name__=='__main__':unittest.main(verbosity=2)
