"""Inspect original library planning, not a new performance experiment."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
import mpmath as mp
from scipy.special import ndtr
R=Path(__file__).resolve().parents[1]
P=R/'baseline_runtime/parent/Finite_Dictionary_Predictive_Adequacy_v0_02'
sys.path.insert(0,str(P/'analysis_src'))
from range_tools import F,I,envelope

def population_pairs(model,lib,truth,p,h=0):
 q=lib.predictions[h]; weights=model.obs_prob
 vals=ndtr((model.mu[h,model.latent_index,truth,None]+model.offsets)/model.sigma)
 prob_b=np.sum(vals*np.exp(model.latent_log_weight)[...,None],axis=1)
 d0=q[:,p,None]**2-q**2; d1=(1-q[:,p,None])**2-(1-q)**2
 means=np.stack([weights@((1-prob_b[:,b,None])*d0+prob_b[:,b,None]*d1)for b in range(len(model.offsets))])
 seconds=np.stack([weights@((1-prob_b[:,b,None])*d0**2+prob_b[:,b,None]*d1**2)for b in range(len(model.offsets))])
 mean=model.offset_probs@means
 row_second=model.offset_probs@seconds
 shared_second=model.offset_probs@(means**2)
 variance=(row_second-shared_second)/4+shared_second-mean**2
 if np.min(variance)<-1e-14:raise ArithmeticError('negative group variance')
 return mean,np.maximum(variance,0.)

def planning(G,d,v,w,Q=70):
 z=math.log(2*Q/.05)
 return d+np.sqrt(2*v*z/G)+7*w*z/(3*(G-1))

def first_crossing(d,v,w,Q,include_variance=True):
 mp.mp.dps=80;dd=mp.mpf(str(d));vv=mp.mpf(str(v));ww=mp.mpf(str(w));eps=mp.mpf('.01');z=mp.log(mp.mpf(40)*Q)
 def val(n):return dd+(mp.sqrt(2*vv*z/n)if include_variance else 0)+7*ww*z/(3*(n-1))
 lo=2;hi=2
 while val(hi)>=eps:
  hi*=2
  if hi>10**9:raise ValueError('planning crossing outside declared search')
 while lo<hi:
  mid=(lo+hi)//2
  if val(mid)<eps:hi=mid
  else:lo=mid+1
 return {'first_integer':lo,'previous_value':float(val(lo-1))if lo>2 else None,'crossing_value':float(val(lo)),'threshold_arithmetic_digits':80}

def audit():
 m=I.FiniteModel('excited','mass_truncated');th=m.parameter_index(.3,16,1.8);p=9+th;q=70+th
 tr=m.draw(np.random.default_rng(F.seed_for(20261016190217,'truncated_strong',0,'training_pool')),4096,4,th,'training','FIT001/truncated_strong/0/train')
 details=[];primary=None
 for T in (64,512,4096):
  lib=I.make_library(m,F.prefix(tr,T));d,v=population_pairs(m,lib,th,p);w=envelope(lib.predictions[0])[2][p]
  pb=planning(16384,d,v,w);pb[w==0]=0;order=np.argsort(-pb,kind='stable')
  rec={'T':T,'library_digest':F.library_digest(lib),'posterior_true_mass':float(lib.posterior[th]),'posterior_max':float(lib.posterior.max()),'posterior_mode':int(lib.posterior.argmax()),'max_forecast_difference_mixture_true_h1':float(np.max(abs(lib.predictions[0,:,130]-lib.predictions[0,:,q]))),'planning_true':float(pb[q]),'planning_mixture':float(pb[130]),'planning_difference_mixture_minus_true':float(pb[130]-pb[q]),'argmax':int(order[0]),'top_comparators':[{'index':int(i),'planning_value':float(pb[i]),'risk_gap':float(d[i]),'group_sd':float(np.sqrt(v[i])),'width':float(w[i])}for i in order[:5]]}
  details.append(rec)
  if T==512:
   primary={'d':float(d[q]),'variance':float(v[q]),'sd':float(np.sqrt(v[q])),'width':float(w[q]),'p':p,'q':q,'width_median':float(np.median(w)),'width_max':float(w.max())}
   curves=[]
   for G in sorted(set(list(range(4096,32769,256))+[6477,13463,14984,16384,24388,32768,12288,15872])):
    Bpair=planning(G,d,v,w);Bpair[w==0]=0
    Bgen=planning(G,d,v,np.full_like(w,2.))
    curves.append({'G':G,'pair_witness':float(Bpair[q]),'generic_witness':float(Bgen[q]),'pair_all_comparator_max_for_p':float(Bpair.max()),'generic_all_comparator_max_for_p':float(Bgen.max())})
 crossings={str(Q):{label:{'remainder_only':first_crossing(primary['d'],primary['variance'],w_,Q,False),'full_formula':first_crossing(primary['d'],primary['variance'],w_,Q,True)}for label,w_ in [('pair',primary['width']),('generic',2.)]}for Q in (1,70,131)}
 return {'scope':'One original training pool at three prefixes; known population moments; not a new retraining-performance study','alpha':.05,'epsilon':.01,'primary_training_groups':512,'primary':primary,'original_library_comparisons':details,'crossings':crossings,'planning_curves':curves,'rounding_note':'80-digit formula evaluation from numerical population moments, not exact or interval-certified source moments'}
if __name__=='__main__':
 dest=Path(sys.argv[1]) if len(sys.argv)>1 else R/'evidence/planning_audit.json';result=audit();dest.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='planning_curves'},indent=2))
