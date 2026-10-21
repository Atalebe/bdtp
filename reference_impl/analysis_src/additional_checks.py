"""Separate post-review power/coverage references; no modification of parent results."""
from pathlib import Path
import sys,math,json,time,hashlib,traceback
import numpy as np
from scipy.stats import beta,binom
from scipy.special import logsumexp
from fractions import Fraction
import mpmath as mp
from range_tools import R,envelope,eb,F
DP=R/'work/dep_parent/NMCD_PC_DEP_001_v0_1';sys.path.insert(0,str(DP/'src'))
import dependence_core as C

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def ci(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def dep_study(out):
 w=envelope(C.D.predictions.T)[2];master=202610180023;results=[]
 for theta in [0.,.5]:
  for G in [256,1024]:
   for m in [1,16,256]:
    cfg={'theta':str(theta),'copy_probability':'1','groups_per_split':G,'rows_per_group':m}
    records=[]
    for rep in range(2000):
     tr=C.draw_stats(np.random.default_rng(C.stream_seed(cfg,rep,'train',master)),G,m,theta,1.)
     te=C.draw_stats(np.random.default_rng(C.stream_seed(cfg,rep,'test',master)),G,m,theta,1.)
     fit=C.fit_groups(tr);ans={f'{u}_{v}':{h:{}for h in(1,2)}for u in('generic','range')for v in('group','row')}
     for h in [1,2]:
      for k in [2,1]:
       p=fit[h][k];mean,var,rowvar=C.pair_moments(te,p,h-1)
       for rule in ['generic','range']:
        ww=None if rule=='generic'else w[p]
        ans[rule+'_group'][h][k]=float(eb(mean,var,G,.05,ww).max())
        ans[rule+'_row'][h][k]=float(eb(mean,rowvar,G*m,.05,ww).max())
     records.append({'rep':rep,'fit':fit,'bounds':ans,'levels':{meth:C.P.decide(b)for meth,b in ans.items()}})
    s={meth:{'certified_1':sum(r['levels'][meth]==1 for r in records),'n':2000}for meth in ans}
    for d in s.values():d['rate']=d['certified_1']/2000;d['ci95']=ci(d['certified_1'],2000)
    result={'case':cfg,'summary':s,'records':records,'null_for_level1':theta==.5,'generic_empty_region_by_self_floor':G==256,'range_from_complete_domain':True}
    write(out/f'theta{theta}_G{G}_m{m}.json',result);results.append({k:v for k,v in result.items()if k!='records'})
    print('DEP',theta,G,m,{k:v['certified_1']for k,v in s.items()},flush=True)
 # Counterexample to reviewer inference that no observed null rejection = empty region.
 # Balanced 16 joint cells -> balanced 8 cells at each horizon, positive probability at theta=.5.
 p0=int(C.D.classes[0][0]);loss=C.LOSS[p0][None,:]-C.LOSS;freq=np.full(8,1/8)
 mean=loss@freq;var=(loss*loss)@freq-mean*mean;var*=1024/1023
 b=float(eb(mean,var,1024,.05).max())
 write(out/'summary.json',{'cases':results,'new_independent_stream_pairs':8000,'case_replicate_evaluations':24000,'copies_paired':True,'balanced_sample_G1024_upper':b,'balanced_sample_rejects':b<1/16,
  'interpretation':'Balanced counts have positive null probability. Empirical minimum over simulations is not a proof of an empty rejection region.'})

def exact_power(out):
 # Fixed state p=.5, history .5+.25 U. D in {-5/16,3/16}; P(D=3/16)=(1+theta)/2.
 # D dictionary contains the state; gap=max(0,theta/4-1/16).
 mp.mp.dps=80;eps=mp.mpf(1)/16;alpha=mp.mpf(1)/20;z=mp.log(2/alpha)
 theta_grid=[Fraction(x,100)for x in [0,20,25,30,35,40,45,47,49,50,51,55,60,75,100]]
 rec=[];cross=[]
 for G in [256,1024,4096]:
  inds={rule:[]for rule in ['generic','range']}
  for k in range(G+1):
   mean=-mp.mpf(5)/16+mp.mpf(k)/(2*G);v=mp.mpf(k)*(G-k)/(4*G*(G-1))
   for rule in inds:
    width=mp.mpf(2)if rule=='generic'else mp.mpf(1)/2
    b=mean+mp.sqrt(2*v*z/G)+7*width*z/(3*(G-1));selfbound=14*z/(3*(G-1))if rule=='generic'else mp.mpf(0)
    if max(b,selfbound)<eps:inds[rule].append(k)
  for th in theta_grid:
   p=(1+th)/2;d=max(Fraction(0),th/4-Fraction(1,16))
   for rule,ks in inds.items():
    # Sum exact integer binomial weights; numerical log-PMF sums can exceed one by rounding.
    a,b=p.numerator,p.denominator;den=b**G
    if a==0:num=int(0 in ks)
    elif a==b:num=int(G in ks)
    else:
     term=(b-a)**G;num=0;selected=set(ks)
     for kk in range(G+1):
      if kk in selected:num+=term
      if kk<G:term=term*(G-kk)*a//((kk+1)*(b-a))
    val=Fraction(num,1 if a in(0,b)else den);prob=float(val)
    if not 0<=val<=1:raise ArithmeticError('probability outside unit interval')
    numeric=0. if not ks else float(np.exp(logsumexp(binom.logpmf(ks,G,float(p)))))
    if p==0:numeric=float(0 in ks)
    if p==1:numeric=float(G in ks)
    cross.append({'G':G,'theta':str(th),'rule':rule,'rational_float':prob,'numeric_difference':abs(prob-numeric)})
    rec.append({'G':G,'theta':str(th),'theta_float':float(th),'true_gap':float(d),'gap_over_epsilon':float(d/ Fraction(1,16)),'rule':rule,'probability':prob,'true_null':d>=Fraction(1,16),'rejection_counts':len(ks)})
 write(out/'power.json',{'records':rec,'rational_crosschecks':cross,'probabilities':'finite binomial sums; rejection masks at 80-digit arithmetic, all probability weights summed with exact integer arithmetic; not interval-certified','predictors':'p=.5; q=.5+.25U; no training, no finite-class selector','no_Monte_Carlo':True})

def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 try:dep_study(out/'dep');exact_power(out/'power')
 except BaseException:(out/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
