import unittest,sys,math
from pathlib import Path
from fractions import Fraction
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'analysis_src'))
from range_tools import envelope,eb,F,I

class RangeTests(unittest.TestCase):
 def test_exact_binary_float_enclosures(self):
  q=np.random.default_rng(17).random((19,7));lo,hi,w=envelope(q)
  for o in range(len(q)):
   for p in range(7):
    for j in range(7):
     x,y=Fraction(float(q[o,p])),Fraction(float(q[o,j]))
     for z in [0,1]:
      d=(z-x)**2-(z-y)**2
      self.assertLessEqual(Fraction(float(lo[p,j])),d);self.assertGreaterEqual(Fraction(float(hi[p,j])),d)
  self.assertTrue(np.all(w<=2))
 def test_duplicate_functions_zero(self):
  q=np.array([[0.,0.,.1],[1.,1.,.7],[.5,.5,.3]])
  l,h,w=envelope(q);self.assertEqual(w[0,1],0);self.assertTrue(np.array_equal(np.diag(w),np.zeros(3)))
 def test_edges(self):
  q=np.array([[0.,1.,.5],[1.,0.,.5]]);lo,hi,w=envelope(q)
  self.assertEqual(lo[0,1],-1);self.assertEqual(hi[0,1],1);self.assertEqual(w[0,1],2)
 def test_input_order_invariant(self):
  q=np.random.default_rng(4).random((40,9))
  for a,b in zip(envelope(q),envelope(q[::-1])):np.testing.assert_equal(a,b)
 def test_duplicate_inputs_no_narrowing(self):
  q=np.random.default_rng(4).random((12,9))
  for a,b in zip(envelope(q),envelope(np.repeat(q,15,axis=0))):np.testing.assert_equal(a,b)
 def test_input_sample_can_underestimate_range(self):
  q=np.array([[.5,.5],[.1,.9]])
  self.assertEqual(envelope(q[:1])[2][0,1],0);self.assertGreater(envelope(q)[2][0,1],1.5)
 def test_group_average_containment(self):
  rng=np.random.default_rng(22);q=rng.random((7,5));lo,hi,w=envelope(q)
  oi=rng.integers(0,7,(32,4));y=rng.integers(0,2,(32,4))
  loss=((q[oi]-y[...,None])**2).mean(1)
  d=loss[:,:,None]-loss[:,None,:]
  self.assertTrue(np.all(d>=lo-1e-15));self.assertTrue(np.all(d<=hi+1e-15))
 def test_generic_equivalence(self):
  d=np.array([[0,.02],[-.02,0]]);v=np.array([[0,.08],[.08,0]])
  np.testing.assert_allclose(eb(d,v,1024,.05),eb(d,v,1024,.05,np.full((2,2),2.)),atol=1e-15)
 def test_zero_width_bound_exact(self):
  z=np.zeros((2,2));np.testing.assert_equal(eb(z,z,2,.05,z),z)
 def test_invalid_zero_width_rejected(self):
  with self.assertRaises(ValueError):eb(np.ones((2,2)),np.ones((2,2)),20,.05,np.zeros((2,2)))
 def test_range_tighter(self):
  q=np.random.default_rng(3).random((21,5));w=envelope(q)[2]
  d=np.zeros((5,5));v=np.full((5,5),.01);np.fill_diagonal(v,0)
  self.assertTrue(np.all(eb(d,v,1024,.05,w)<=eb(d,v,1024,.05)))
 def test_invalid_probabilities(self):
  for q in [[],[[np.nan]],[[1.1]],[[float('inf')]], [[-.1]]]:
   with self.assertRaises(ValueError):envelope(q)
 def test_invalid_sample_sizes(self):
  for n in [1,0,True,2.5]:
   with self.assertRaises(ValueError):eb(np.zeros((1,1)),np.zeros((1,1)),n,.05)
 def test_invalid_levels(self):
  for a in [0,1,-.1,float('nan')]:
   with self.assertRaises(ValueError):eb(np.zeros((1,1)),np.zeros((1,1)),20,a)
 def test_width_metadata_invalid(self):
  for w in [np.ones((1,2)),np.array([[-.1]]),np.array([[2.1]]),np.array([[np.nan]])]:
   with self.assertRaises(ValueError):eb(np.zeros((1,1)),np.zeros((1,1)),20,.05,w)
 def test_floor_values(self):
  floors=[math.floor(14*math.log(2*q/.05)/.03)+2 for q in [1,70,131]]
  self.assertEqual(floors,[1723,3706,3998])
  self.assertGreater(14*math.log(40)/(3*255),1/16)
 def test_scalar_and_class_monotonic(self):
  q=np.random.default_rng(3).random((21,5));w=envelope(q)[2];v=np.ones((5,5))*.05;np.fill_diagonal(v,0);d=np.zeros((5,5))
  b=eb(d,v,1024,.05,w);c=eb(d,v,1024,.01,w)
  self.assertTrue(np.all(c>=b))
 def test_brier_excess_identity(self):
  p=np.random.default_rng(5).random(50);q=np.random.default_rng(7).random(50)
  np.testing.assert_allclose(p*(1-q)**2+(1-p)*q*q-p*(1-p),(p-q)**2,atol=3e-16)
 def test_frozen_table_without_test_argument(self):
  import inspect
  self.assertEqual(tuple(inspect.signature(envelope).parameters),('probabilities',))
 def test_population_planning_not_sample_floor(self):
  # The sample mean is random: a positive population planning bound need not
  # prevent a below-tolerance sample upper bound.
  mu=.0075276576;G=16384;v=.03737**2
  z=math.log(2*70/.05);r=14*z/(3*(G-1));s=math.sqrt(2*v*z/G)
  self.assertGreater(mu+s+r,.01);self.assertLess(.005+s+r,.01)
if __name__=='__main__':unittest.main(verbosity=2)
