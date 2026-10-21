from pathlib import Path
import sys,json,time,hashlib,os,traceback,resource
from collections import Counter
import numpy as np
from scipy.stats import beta,binom
from range_tools import F,I,FP,R,envelope,evaluate

def enc(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,sort_keys=True,indent=2,default=enc,allow_nan=False)+'\n')
def ci(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def summary(records):
 out={}
 for var in ('generic','range'):
  out[var]={}
  for met in F.METHODS:
   n=len(records);cert=sum(r['truth'][var]['methods'][met]['certifies_summary']for r in records)
   err=sum(r['truth'][var]['methods'][met]['false_class_certification']for r in records)
   out[var][met]={'n':n,'certificates':cert,'rate':cert/n,'ci95_pointwise':ci(cert,n),'false_class_certificates':err,
    'levels':dict(Counter(str(r['answers'][var][met]['certified_level'])for r in records)),
    'mean_bounds':np.mean([r['answers'][var][met]['bounds'][2]for r in records],axis=0),
    'adequate_selected':sum(max(r['truth'][var]['methods'][met]['selected_gaps'][2])<.01 for r in records)}
 return out

def main(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 plan=json.loads((R/'contracts/range_sensitivity_plan.json').read_text());parent=json.loads((FP/'contracts/development_plan.json').read_text())
 start=time.monotonic();cells={};fit_log=[];checks=[]
 try:
  for case in parent['cases']+parent['controls']:
   cid=case['id'];model=I.FiniteModel(case['design'],case['observation']);truth=model.parameter_index(*case['truth'])
   reps=case.get('replicates',parent['replicates']);Ts=[case['training_worlds']]if 'training_worlds'in case else parent['training_budgets'];Ns=[case['total_worlds']]if 'total_worlds'in case else parent['world_budgets']
   for rep in range(reps):
    st=F.seed_for(parent['master_seed'],cid,rep,'training_pool');se=F.seed_for(parent['master_seed'],cid,rep,'test_pool')
    trpool=model.draw(np.random.default_rng(st),max(Ts),4,truth,'training',f'FIT001/{cid}/{rep}/train')
    tepool=model.draw(np.random.default_rng(se),max(Ns)-min(Ts),4,truth,'test',f'FIT001/{cid}/{rep}/test')
    for T in Ts:
     tr=F.prefix(trpool,T);lib=I.make_library(model,tr);fit=F.fit_rules(model,lib,tr)
     widths=[envelope(lib.predictions[h])[2]for h in (0,1)] # pre-evaluation functions, whole support
     fit_log.append({'case':cid,'rep':rep,'T':T,'training_digest':fit.training_digest,'library_digest':fit.library_digest,'choices':fit.choices,
                     'width_hashes':[F.array_digest(w)for w in widths],'joint_posterior':fit.posterior_joint,'library_posterior':lib.posterior})
     for N in Ns:
      G=N-T;te=F.prefix(tepool,G);ans=evaluate(model,lib,fit,te,widths)
      diag={v:F.diagnostic_truth(model,lib,truth,ans[v])for v in ans}
      key=f'{cid}_N{N}_T{T}'
      # Preserve identical old RNG IDs, library, fitting, groups; only range differs.
      cells.setdefault(key,{'case':cid,'T':T,'N':N,'G':G,'records':[]})['records'].append({'rep':rep,'answers':ans,'truth':diag,'library_digest':fit.library_digest,'width_hashes':[F.array_digest(w)for w in widths]})
    if rep%20==19:print(cid,rep+1,round(time.monotonic()-start,1),flush=True)
   # Checkpoint each completed design, not a success or scientific logbook.
   for key,c in cells.items():
    if c['case']==cid:write(out/'cells'/f'{key}.json',{**c,'summary':summary(c['records'])})
  original=json.loads((FP/'evidence/run_002/campaign/summary.json').read_text())
  old={f"{c['case']['id']}_N{c['N']}_T{c['T']}":c for c in original['cells']};baseline=[];summaries=[]
  for key,c in cells.items():
   s=summary(c['records']);summaries.append({k:v for k,v in c.items()if k!='records'}|{'summary':s})
   for met in F.METHODS:
    x=s['generic'][met];o=old[key]['methods'][met]
    baseline.append({'cell':key,'method':met,'count_match':x['certificates']==o['summary_certificates'],'levels_match':x['levels']==o['certified_counts'],
       'mean_bound_max_abs':float(np.max(abs(x['mean_bounds']-o['mean_summary_bound_by_horizon'])))})
    y=s['range'][met]
    # A sensitivity does not select whichever rule succeeds. Separate per-rule results.
    if y['false_class_certificates']:
     checks.append({'case':key,'method':met,'gross_error_alarm_pass':float(binom.sf(y['false_class_certificates']-1,y['n'],.05))>=.01/80})
   for rec in c['records']:
    for met in F.METHODS:
     for k in (1,2,3):
      if np.max(np.array(rec['answers']['range'][met]['bounds'][k])-rec['answers']['generic'][met]['bounds'][k])>1e-11:raise AssertionError('range bound cannot exceed generic with common moments')
  if not all(x['count_match']and x['levels_match']and x['mean_bound_max_abs']<3e-11 for x in baseline):raise AssertionError('historical parity')
  if not all(x['gross_error_alarm_pass']for x in checks):raise AssertionError('gross errors')
  write(out/'summary.json',{'cells':summaries,'historical_comparison':baseline,'case_replicates':sum(len(c['records'])for c in cells.values()),'training_fits':len(fit_log),'independent_pool_pairs':400,'fresh_data':False,'range_is_post_review_sensitivity':True})
  write(out/'fits.json',fit_log)
  write(out/'execution.json',{'status':'EXECUTION_PASS_REPLAY_PENDING','elapsed':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'formal_logbook':False})
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(sys.argv[1])
