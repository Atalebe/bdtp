import unittest,sys,math,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import review_extension as E
I=E.I
class TestCertificates(unittest.TestCase):
 def setUp(self):self.x=np.random.default_rng(71).random((80,7))
 def test_one_member_matches_parent(self):
  d=self.x[:,2,None]-self.x
  b=I.scalar_eb(d.mean(0),d.var(0,ddof=1),len(d),.05)
  self.assertAlmostEqual(b,E.class_min_bound(self.x,[2])['bound'],places=13)
 def test_all_pair_moments(self):
  m,c,n=E.group_loss_moments(self.x);d,v=E.pairwise_moments(m,c)
  for p,q in itertools.product(range(7),repeat=2):
   a=self.x[:,p]-self.x[:,q]
   self.assertAlmostEqual(d[p,q],a.mean(),places=14);self.assertAlmostEqual(v[p,q],a.var(ddof=1),places=14)
 def test_union_matches_loop(self):
  ix=[1,3,5];got=E.class_min_bound(self.x,ix)
  bb=[]
  for p in ix:
   d=self.x[:,p,None]-self.x
   bb.append(I.scalar_eb(d.mean(0),d.var(0,ddof=1),len(d),.05/len(ix)))
  np.testing.assert_allclose(got['per_member_bounds'],bb,rtol=0,atol=2e-13)
 def test_bigger_penalty_not_cheaper(self):
  m,c,n=E.group_loss_moments(self.x)
  self.assertTrue(np.all(E.eb_pair_bounds(m,c,n,.05/7)>=E.eb_pair_bounds(m,c,n,.05)))
 def test_reordering_invariant(self):
  a=E.class_min_bound(self.x,[1,3,5]);b=E.class_min_bound(self.x,[5,3,1]);self.assertEqual(a['bound'],b['bound'])
 def test_empty_class(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x,[])
 def test_duplicate_class(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x,[1,1])
 def test_member_outside(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x,[7])
 def test_float_index(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x,[1.0])
 def test_bool_index(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x,[True])
 def test_outside_loss(self):
  z=self.x.copy();z[0,0]=1.01
  with self.assertRaises(ValueError):E.class_min_bound(z,[1])
 def test_negative_loss(self):
  with self.assertRaises(ValueError):E.class_min_bound(-self.x,[1])
 def test_nan_loss(self):
  z=self.x.copy();z[0,0]=np.nan
  with self.assertRaises(ValueError):E.class_min_bound(z,[1])
 def test_too_few(self):
  with self.assertRaises(ValueError):E.class_min_bound(self.x[:1],[1])
 def test_false_alpha(self):
  for a in [0,1,-1,np.nan,True]:
   with self.assertRaises(ValueError):E.class_min_bound(self.x,[1],a)
 def test_variance_symmetric(self):
  m,c,n=E.group_loss_moments(self.x);d,v=E.pairwise_moments(m,c)
  np.testing.assert_allclose(d,-d.T);np.testing.assert_allclose(v,v.T);np.testing.assert_array_equal(np.diag(v),0)
 def test_bad_covariance(self):
  with self.assertRaises(ValueError):E.pairwise_moments(np.ones(2)/2,np.array([[.1,1],[0,.1]]))
 def test_negative_contrast_variance(self):
  with self.assertRaises(ArithmeticError):E.pairwise_moments(np.ones(2)/2,np.array([[.1,.2],[.2,.1]]))
 def test_self_term_floor(self):
  z=np.zeros((100,4));b=E.class_min_bound(z,[0,1])['bound']
  self.assertAlmostEqual(b,14*math.log(80)/(3*99))
 def test_floor_1(self):self.assertEqual(E.class_floor(1),1723)
 def test_floor_70(self):self.assertEqual(E.class_floor(70),3706)
 def test_floor_131(self):self.assertEqual(E.class_floor(131),3998)
 def test_floor_strict(self):
  for q in [1,70,131]:
   n=E.class_floor(q);c=14*math.log(2*q/.05)/3
   self.assertLess(c/(n-1),.01);self.assertGreaterEqual(c/(n-2),.01)
 def test_union_is_needed(self):
  # Abstract exact-coverage events, not an EB distributional sharpness test.
  Q=70;a=.05
  self.assertGreater(1-(1-a)**Q,a)
  self.assertLessEqual(1-(1-a/Q)**Q,a)
 def test_class_membership_failure(self):
  m,c,n=E.group_loss_moments(self.x)
  with self.assertRaises(ValueError):E.evaluate_moments(m,c,n,{1:[0],2:[0,1],3:list(range(7))},{1:2,2:0,3:0})
 def test_nesting_failure(self):
  m,c,n=E.group_loss_moments(self.x)
  with self.assertRaises(ValueError):E.evaluate_moments(m,c,n,{1:[1],2:[0],3:list(range(7))},{1:1,2:0,3:0})

class TestModelAndSelection(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=I.FiniteModel(observation='mass_truncated');cls.t=cls.m.parameter_index(.3,16.,1.8)
  cls.train=cls.m.draw(np.random.default_rng(91),8,4,cls.t,'training','R1/train');cls.lib=I.make_library(cls.m,cls.train)
 def test_true_class_size(self):self.assertEqual(len(self.lib.classes[2]),70)
 def test_iid_increment_law(self):
  m=I.FiniteModel();self.assertEqual(len(np.unique(m.u,axis=0)),64)
  for j in range(6):self.assertAlmostEqual((m.u[:,j]>.05).mean(),.5)
 def test_missingness_rules(self):
  np.testing.assert_array_equal(self.m.obs_mask[self.m.obs_c==-1,:3],False)
  np.testing.assert_array_equal(self.m.obs_mask[self.m.obs_c==0,:1],False)
 def test_adapter_fixed_parity(self):
  z=self.m.draw(np.random.default_rng(27),100,4,self.t,'test','R1/test')
  old=I.evaluate_library(self.m,self.lib,z,self.t);new=E.evaluate_batch(self.m,self.lib,z)
  for k in ['1','2','3']:np.testing.assert_allclose(old['bounds'][k],new['historical_fixed']['bounds'][k],rtol=0,atol=2e-12)
 def test_overlap_blocked(self):
  z=self.m.draw(np.random.default_rng(27),8,4,self.t,'test','R1/train')
  with self.assertRaises(ValueError):E.evaluate_batch(self.m,self.lib,z)
 def test_training_blocked(self):
  with self.assertRaises(ValueError):E.evaluate_batch(self.m,self.lib,self.train)
 def test_exact_group_mean(self):
  mu,c=E.exact_group_loss_moments(self.m,self.lib,self.t)
  expected=self.m.exact_brier_risk(self.lib.predictions,self.t)[0]
  np.testing.assert_allclose(mu,expected,rtol=0,atol=2e-14)
 def test_group_covariance_psd(self):
  mu,c=E.exact_group_loss_moments(self.m,self.lib,self.t)
  self.assertGreater(np.linalg.eigvalsh(c).min(),-2e-12)
 def test_group_cov_decreases_with_objects(self):
  mu,c1=E.exact_group_loss_moments(self.m,self.lib,self.t,1)
  mu,c4=E.exact_group_loss_moments(self.m,self.lib,self.t,4)
  self.assertGreater(np.linalg.eigvalsh(c1-c4).min(),-2e-12)
 def test_seed_distinct(self):self.assertNotEqual(E.seed_for(5,'a',0),E.seed_for(5,'a',1))
 def test_PCA_does_not_bound_functional_residual(self):
  cov=np.diag([.35,.35,.30]);w=np.array([.05,.05,.9]);v=w@cov@w
  self.assertAlmostEqual((.35+.35)/np.trace(cov),.7)
  self.assertGreater((w[2]**2*.30)/v,.99)
 def test_DKW_not_exact_KS(self):
  from scipy.stats import kstwo
  t=math.sqrt(math.log(4800)/2000)
  self.assertAlmostEqual(t,.06510134866842615)
  self.assertLess(kstwo.sf(t,1000),.01/24)
if __name__=='__main__':unittest.main(verbosity=2)
