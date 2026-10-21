from __future__ import annotations
import argparse,json,sys,math,hashlib,time,resource
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import beta,binom,binomtest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import fitting_core as F
I=F.I;E=F.E

def enc(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=enc,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(n,N):return [float(beta.ppf(.025,n,N-n+1))if n else 0.,float(beta.ppf(.975,n+1,N-n))if n<N else 1.]

def summarize(records):
 methods={}
 for meth in F.METHODS:
  n=len(records);cert=sum(r['truth']['methods'][meth]['certifies_summary']for r in records)
  errors=sum(r['truth']['methods'][meth]['false_class_certification']for r in records)
  choices=Counter(str(r['answers'][meth]['certified_level'])for r in records)
  methods[meth]={'certified_counts':dict(choices),'summary_certificates':cert,'replicates':n,'summary_rate':cert/n,'pointwise_95_ci':ci(cert,n),
    'false_class_certifications':errors,
    'mean_summary_gap_by_horizon':np.mean([r['truth']['methods'][meth]['selected_gaps'][2]for r in records],axis=0).tolist(),
    'summary_regret_median_h1':float(np.median([r['truth']['methods'][meth]['fitting_regret'][2][0]for r in records])),
    'summary_gap_p90_h1':float(np.quantile([r['truth']['methods'][meth]['selected_gaps'][2][0]for r in records],.9)),
    'selected_summary_truly_adequate_count':sum(max(r['truth']['methods'][meth]['selected_gaps'][2])<.01 for r in records),
    'mean_summary_bound_by_horizon':np.mean([r['answers'][meth]['bounds'][2]for r in records],axis=0).tolist()}
 plus=sum(r['truth']['methods']['posterior_risk']['certifies_summary']and not r['truth']['methods']['evaluation_class']['certifies_summary']for r in records)
 minus=sum(r['truth']['methods']['evaluation_class']['certifies_summary']and not r['truth']['methods']['posterior_risk']['certifies_summary']for r in records)
 p=binomtest(plus,plus+minus,.5).pvalue if plus+minus else 1.
 return {'methods':methods,'paired_comparison':{'posterior_only_success':plus,'class_only_success':minus,'rate_difference':(plus-minus)/len(records),'exact_discordance_p_two_sided':float(p)}}

def experiment(plan,out):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 cells={};fit_records=[];costs=[]
 all_cases=plan['cases']+plan['controls']
 for case in all_cases:
  cid=case['id'];m=I.FiniteModel(case['design'],case['observation']);truth=m.parameter_index(*case['truth'])
  control='replicates'in case
  RR=case['replicates']if control else plan['replicates']
  Ts=[case['training_worlds']]if control else plan['training_budgets']
  Ns=[case['total_worlds']]if control else plan['world_budgets']
  for rep in range(RR):
   st=F.seed_for(plan['master_seed'],cid,rep,'training_pool');se=F.seed_for(plan['master_seed'],cid,rep,'test_pool')
   training_pool=m.draw(np.random.default_rng(st),max(Ts),4,truth,'training',f'FIT001/{cid}/{rep}/train')
   test_pool=m.draw(np.random.default_rng(se),max(Ns)-min(Ts),4,truth,'test',f'FIT001/{cid}/{rep}/test')
   for T in Ts:
    train=F.prefix(training_pool,T);t=time.process_time();lib=I.make_library(m,train);tbuild=time.process_time()-t
    t=time.process_time();fit=F.fit_rules(m,lib,train);tf=time.process_time()-t
    if max(tbuild,tf)>plan['resource_ceiling']['standalone_method_cpu_seconds']:raise RuntimeError('training resource cap')
    fit_records.append({'case':cid,'rep':rep,'T':T,'training_digest':fit.training_digest,'library_digest':fit.library_digest,
       'choices':fit.choices,'joint_posterior':fit.posterior_joint,'horizon1_library_posterior':lib.posterior})
    for N in Ns:
     G=N-T;test=F.prefix(test_pool,G)
     t=time.process_time();ans=F.evaluation_from_cov(m,lib,fit,test,plan['alpha'],plan['epsilon']);dt=time.process_time()-t
     if dt>plan['resource_ceiling']['standalone_method_cpu_seconds']:raise RuntimeError('shared evaluation resource cap')
     diag=F.diagnostic_truth(m,lib,truth,ans,plan['epsilon'])
     key=f'{cid}_N{N}_T{T}'
     cells.setdefault(key,{'case':case,'T':T,'G':G,'N':N,'records':[]})['records'].append({
       'rep':rep,'training_seed':str(st),'test_seed':str(se),'library_digest':fit.library_digest,
       'test_prefix_digest':F.array_digest(test.observed)+'|'+F.array_digest(test.responses),
       'answers':ans,'truth':diag})
     costs.append({'case':cid,'rep':rep,'T':T,'N':N,'G':G,'common_library_cpu_s':tbuild,
                   'all_fit_rules_cpu_s':tf,'shared_evaluation_cpu_s':dt})
   if rep%20==19:print(cid,'replicates',rep+1,'elapsed',round(time.monotonic()-start,1),flush=True)
  print('CASE COMPLETE',cid,flush=True)
 result=[];checks=[]
 for key,cell in cells.items():
  s=summarize(cell['records']);entry={k:v for k,v in cell.items()if k!='records'};entry.update(s);result.append(entry)
  write(out/'cells'/f'{key}.json',{**entry,'records':cell['records']})
  for meth,x in s['methods'].items():
   p=float(binom.sf(x['false_class_certifications']-1,x['replicates'],plan['alpha']))if x['false_class_certifications']else 1.
   checks.append({'check':'gross_false_certificate','cell':key,'method':meth,'p_value':p,'threshold':.01/(len(cells)*len(F.METHODS)),'pass':p>=.01/(len(cells)*len(F.METHODS))})
  checks.append({'check':'matched_budget','cell':key,'pass':cell['T']+cell['G']==cell['N']})
  checks.append({'check':'replicate_count','cell':key,'pass':len(cell['records'])==(cell['case'].get('replicates',plan['replicates']))})
  print(key,{method:xx['summary_rate']for method,xx in s['methods'].items()},flush=True)
 primary=next(x for x in result if x['case']['id']==plan['primary_contrast']['case']and x['N']==plan['primary_contrast']['total_worlds']and x['T']==plan['primary_contrast']['training_worlds'])
 payload={'cells':result,'checks':checks,'all_checks_pass':all(x['pass']for x in checks),
          'primary_contrast':primary,'case_replicate_evaluations':sum(len(c['records'])for c in cells.values()),
          'distinct_training_prefix_fits':len(fit_records),'independent_pool_pairs':sum(c.get('replicates',plan['replicates'])for c in all_cases),
          'pool_pairing':'same case/rep pool reused through prefixes across N and T; methods share worlds; not independent across cells'}
 write(out/'summary.json',payload);write(out/'fits.json',fit_records)
 write(out/'operation_counts.json',[{'case':c['id'],'T':T,'N':N,**F.resource_payload(I.FiniteModel(c['design'],c['observation']),I.make_library(I.FiniteModel(c['design'],c['observation']),I.FiniteModel(c['design'],c['observation']).draw(np.random.default_rng(77),2,4,0,'training','accounting_fixture')),T,N-T)}for c in plan['cases']for T in plan['training_budgets']for N in plan['world_budgets']])
 write(out.parent/'campaign_timings.json',{'per_cell':costs,'wall_seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'scope':'shared campaign; not standalone per-method timing'})
 if not payload['all_checks_pass']:raise RuntimeError('operating checks failed')


def resource_audit(plan,out):
 out.mkdir(parents=True,exist_ok=False);records=[]
 for cfg in plan['resource_audit_cases']:
  case=next(c for c in plan['cases']if c['id']==cfg['case']);m=I.FiniteModel(case['design'],case['observation']);truth=m.parameter_index(*case['truth'])
  T,N=cfg['T'],cfg['N'];G=N-T
  tr=m.draw(np.random.default_rng(F.seed_for(plan['master_seed'],case['id'],0,'resource_training')),T,4,truth,'training','resource_train')
  te=m.draw(np.random.default_rng(F.seed_for(plan['master_seed'],case['id'],0,'resource_test')),G,4,truth,'test','resource_test')
  lib=I.make_library(m,tr);fit=F.fit_rules(m,lib,tr)
  reference=F.evaluation_from_cov(m,lib,fit,te)
  timings={method:[]for method in F.METHODS}
  # Benchmark selectors separately: shared table construction is common and recorded.
  ft={method:[]for method in F.METHODS[:-1]}
  for rr in range(plan['resource_audit_repetitions']):
   for method in F.METHODS[:-1]:
    t=time.process_time()
    if method=='posterior_risk':
     post=I.normalize_log_weights(F.joint_log_likelihood(m,tr),m.prior)
     for h in (0,1):F.choose_in_class(F.expected_brier(m,lib,post,h),lib.classes)
    else:
     for h in (0,1):
      loss=F.world_losses(m,lib,tr,h);mean=loss.mean(0)
      for k,ix in lib.classes.items():
       score=mean if method=='erm'else mean+np.sqrt(2*loss.var(0,ddof=1)*math.log(2*len(ix)/.05)/T)
       int(np.asarray(ix)[np.argmin(score[list(ix)])])
    ft[method].append(time.process_time()-t)
   for method in F.METHODS:
    t=time.process_time();a=F.evaluate_isolated(m,lib,fit,te,method);dt=time.process_time()-t;timings[method].append(dt)
    if dt>plan['resource_ceiling']['standalone_method_cpu_seconds']:raise RuntimeError('isolated resource cap exceeded')
    for k in (1,2,3):
     if np.max(np.abs(np.array(a['bounds'][k])-reference[method]['bounds'][k]))>3e-11:raise ArithmeticError('resource/paired core parity')
  records.append({'config':cfg,'G':G,'fit_cpu_seconds':ft,'evaluation_cpu_seconds':timings,'payload':F.resource_payload(m,lib,T,G),
                  'common_build_note':'Horizon-1 posterior and 131-entry table are supplied equally, charged as common work; isolated timings do not multiply shared construction by method count.'})
 write(out/'resource_timings.json',{'records':records,'cpu_seconds_not_wall_time':True,'includes_likelihood_assumptions':True,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'not_universal_benchmark':True})
 print('RESOURCE AUDIT COMPLETE',flush=True)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['campaign','resources'],required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 plan=json.loads((R/'contracts/development_plan.json').read_text());out=Path(a.output)
 if a.stage=='campaign':experiment(plan,out)
 else:resource_audit(plan,out)
if __name__=='__main__':main()
