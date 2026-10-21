from pathlib import Path
import sys,json,math
import numpy as np
from scipy.special import ndtr
from range_tools import F,I,R,envelope

def main():
 m=I.FiniteModel('excited','mass_truncated');th=m.parameter_index(.3,16,1.8)
 tr=m.draw(np.random.default_rng(F.seed_for(20261016190217,'truncated_strong',0,'training_pool')),4096,4,th,'training','FIT001/truncated_strong/0/train')
 lib=I.make_library(m,F.prefix(tr,512));h=0;q=lib.predictions[h];O,J=q.shape
 # Conditional event probability per observed input and common offset, with hidden paths integrated.
 vals=ndtr((m.mu[h,m.latent_index,th,None]+m.offsets)/m.sigma)
 pb=np.sum(vals*np.exp(m.latent_log_weight)[...,None],axis=1)
 weights=m.obs_prob
 loss0=q*q;loss1=(1-q)**2
 allmeans=[];allseconds=[]
 for b in range(len(m.offsets)):
  p=pb[:,b];Erow=weights@((1-p[:,None])*loss0+p[:,None]*loss1)
  second=loss0.T@(weights[:,None]*(1-p[:,None])*loss0)+loss1.T@(weights[:,None]*p[:,None]*loss1)
  allmeans.append(Erow);allseconds.append(second)
 allmeans=np.array(allmeans);mean=m.offset_probs@allmeans
 rowsecond=sum(m.offset_probs[b]*allseconds[b]for b in range(len(m.offsets)))
 cross=sum(m.offset_probs[b]*np.outer(allmeans[b],allmeans[b])for b in range(len(m.offsets)))
 cov=rowsecond/4+3*cross/4-np.outer(mean,mean)
 d=mean[:,None]-mean[None,:];v=np.maximum(0,np.diag(cov)[:,None]+np.diag(cov)[None,:]-2*cov)
 risk=m.exact_brier_risk(lib.predictions,th)[h];assert np.max(abs(risk-mean))<1e-12
 w=envelope(q)[2];state=9+th;hist=70+th;gap=risk[state]-risk[hist]
 ans={'scope':'population moment planning, not a sample confidence bound or power curve','trained_library_digest':F.library_digest(lib),'training_count':512,'replicate':0,'state_index':state,'history_index':hist,'gap':float(gap),'witness_width':float(w[state,hist]),'witness_group_sd':float(np.sqrt(v[state,hist])),'generic_self_remainder_G256_epsilon_1_16':14*math.log(40)/(3*255),'integer_self_floors_epsilon_01':{},'planning':[]}
 for Q in (1,70,131):ans['integer_self_floors_epsilon_01'][str(Q)]=math.floor(1+14*math.log(2*Q/.05)/(3*.01))+1
 for label,W in [('generic',2.),('pair',w[state,hist])]:ans[f'{label}_witness_zero_variance_planning_integer']=math.floor(1+7*W*math.log(2*70/.05)/(3*(.01-gap)))+1
 for G in (12288,15872,16384,32768):
  z=math.log(2*70/.05)
  for lab,W in [('generic',np.full_like(w,2)),('range',w)]:
   B=d+np.sqrt(2*v*z/G)+7*W*z/(3*(G-1));B=np.where(W==0,0,B)
   vals=B[:70].max(axis=1);p=int(vals.argmin());comp=int(B[p].argmax())
   ans['planning'].append({'G':G,'rule':lab,'class_bound_from_population_moments':float(vals.min()),'state':p,'comparator':comp,'witness_state_bound':float(B[state].max())})
 out=R/'evidence/planning.json';out.write_text(json.dumps(ans,indent=2,sort_keys=True)+'\n');print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
