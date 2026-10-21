"""Pre-specified review-development diagnostics; never overwrite existing output."""
from pathlib import Path
import argparse,json,sys,math
from collections import Counter
import numpy as np
from scipy.stats import kstwo,binom,beta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import review_extension as E
I=E.I


def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else x.tolist())+'\n')


def describe(a):
    x=np.asarray(a,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1))if len(x)>1 else 0.,
            'min':float(x.min()),'q10':float(np.quantile(x,.1)),'q25':float(np.quantile(x,.25)),
            'median':float(np.median(x)),'q75':float(np.quantile(x,.75)),'q90':float(np.quantile(x,.9)),
            'max':float(x.max()),'fraction_gt_0_9':float((x>.9).mean())}


def gamma_distributions(plan,out):
    results=[]
    for case in plan['gamma_distribution_cases']:
        model=I.FiniteModel(case['design'],case['observation']);records=[]
        for rep in range(case['replicates']):
            rng=np.random.default_rng(E.seed_for(plan['master_seed'],case['id'],rep))
            ti=int(rng.choice(60,p=model.prior))if case['truth']=='prior' else model.parameter_index(*case['truth'])
            data=model.draw(rng,case['worlds'],case['objects'],ti,'inverse',case['id']+f'/{rep}')
            pp=model.posterior(data);mass=I.posterior_mass_summary(model,pp,ti)
            records.append({'truth':model.params[ti].tolist(),'gamma_mass':mass['true_gamma_probability'],
                'gamma_probabilities':mass['gamma_probabilities'],'gamma_entropy_fraction':mass['gamma_entropy_fraction']})
        rr={'case':case,'overall':describe([x['gamma_mass']for x in records]),
            'by_abs_amplitude':{str(a):describe([x['gamma_mass']for x in records if abs(x['truth'][1])==a])
              for a in [0,8,16]if any(abs(x['truth'][1])==a for x in records)},'records':records,
            'interpretation':'finite-grid true-label posterior mass, not parameter coverage or continuous identifiability'}
        write(out/(case['id']+'.json'),rr);results.append({k:v for k,v in rr.items()if k!='records'})
        print('gamma complete',case['id'],flush=True)
    write(out/'summary.json',{'cases':results,'datasets':sum(c['replicates']for c in plan['gamma_distribution_cases']),
                           'checks':[{'name':'constant gamma flat','pass':max(abs(v-1/6) for r in records for v in r['gamma_probabilities'])<2e-12}]})


def reproduce_library(case_id):
    pp=json.loads((ROOT/'parent/contracts/development_plan.json').read_text())
    case=next(c for c in pp['predictive_cases']if c['id']==case_id)
    m=I.FiniteModel(case['design'],case['observation']);truth=m.parameter_index(*case['truth'])
    s=I.seed_for(pp['master_seed'],case_id,0,'frozen_library_training')
    train=m.draw(np.random.default_rng(s),64,4,truth,'training',case_id+':training');lib=I.make_library(m,train)
    return m,truth,train,lib,case


def class_certificates(plan,out):
    checks=[];cells=[]
    for cell in plan['class_certificate_cases']:
        cid=cell['parent_id'];G=cell['groups'];R=cell['replicates'];m,ti,train,lib,case=reproduce_library(cid)
        risk=m.exact_brier_risk(lib.predictions,ti);best=risk.min(axis=1)
        gap={str(k):(risk[:,list(ix)].min(axis=1)-best).tolist()for k,ix in lib.classes.items()}
        records=[]
        for rep in range(R):
            s=E.seed_for(plan['master_seed'],cid+f':class:G{G}',rep)
            data=m.draw(np.random.default_rng(s),G,4,ti,'test',f'INF001-R1/{cid}/{G}/{rep}')
            ans=E.evaluate_batch(m,lib,data,plan['alpha_per_case'],plan['epsilon'])
            for name in ['historical_fixed','reviewer_class_min']:
                c=ans[name]['certified_level'];ans[name]['false_certification']=c is not None and max(gap[str(c)])>=plan['epsilon']
            records.append(ans)
        methods={}
        for name in ['historical_fixed','reviewer_class_min']:
            cc=Counter(str(r[name]['certified_level'])for r in records);err=sum(r[name]['false_certification']for r in records)
            methods[name]={'certified_counts':{k:cc.get(k,0)for k in ['1','2','3','None']},'errors':int(err),
             'error_upper_pointwise_95':float(beta.ppf(.95,err+1,R-err))if err<R else 1.,
             'mean_bounds':{k:np.mean([r[name]['bounds'][k]for r in records],axis=0).tolist()for k in ['1','2','3']}}
            pv=float(binom.sf(err-1,R,.05))if err else 1.
            checks.append({'cell':cid+f':G{G}','method':name,'alarm_pvalue':pv,'alarm_threshold':.01/(2*len(plan['class_certificate_cases'])),'pass':pv>=.01/(2*len(plan['class_certificate_cases']))})
        rr={'design':cell,'source_case':case,'model_class_gaps':gap,'methods':methods,'records':records,
            'scope':'fresh test worlds conditional on the ORIGINAL exposed training library; development only, not retrospective parent evidence'}
        write(out/(cid+f'_G{G}.json'),rr);cells.append({k:v for k,v in rr.items()if k!='records'})
        print('class complete',cid,G,{k:v['certified_counts']for k,v in methods.items()},flush=True)
    write(out/'summary.json',{'cells':cells,'checks':checks,'all_checks_pass':all(c['pass']for c in checks),
                             'test_batches':sum(c['replicates']for c in plan['class_certificate_cases']),'frozen_parent_libraries':3})
    if not all(c['pass']for c in checks):raise RuntimeError('class certificate operating alarm')


def arithmetic(out):
    a=.01;F=24;n=1000;t=math.sqrt(math.log(2*F/a)/(2*n));m,ti,tr,lib,c=reproduce_library('truncated_strong')
    risk=m.exact_brier_risk(lib.predictions,ti);gap=float((risk[:,list(lib.classes[2])].min(axis=1)-risk.min(axis=1))[0]);room=.01-gap
    data={'DKW':{'threshold':t,'tail_upper':a/F,'finite_n_KS_tail':float(kstwo.sf(t,n)),
       'finite_n_KS_times_24':float(F*kstwo.sf(t,n)),'exact_KS_critical_numeric':float(kstwo.isf(a/F,n))},
      'class_sizes':{str(k):len(ix)for k,ix in lib.classes.items()},
      'floors':{str(q):{'zero_gap_integer_floor':E.class_floor(q),'zero_gap_real_threshold':1+14*math.log(2*q/.05)/(.03)}for q in [1,70,131]},
      'truncated':{'gap':gap,'margin_left':room,'class70_zero_variance_population_slack_planning_integer':math.floor(14*math.log(2800)/(3*room))+2},
      'means_and_variances_planning_only':[]}
    for cid in ['history_moderate','history_strong','truncated_strong']:
        m,ti,tr,lib,case=reproduce_library(cid);pp=[]
        for G in [4096,8192,16384,32768,65536]:
            hh=[]
            for h in [0,1]:
                mu,cov=E.exact_group_loss_moments(m,lib,ti,4,h)
                hh.append(E.evaluate_moments(mu,cov,G,lib.classes,{k:lib.chosen[k][h]for k in [1,2,3]}))
            pp.append({'G':G,'horizons':hh})
        data['means_and_variances_planning_only'].append({'case':case,'values':pp,'warning':'population moments inserted for planning only; NOT a confidence statement'})
    write(out/'arithmetic.json',data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['arithmetic','gamma','classes'],required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'contracts/review_plan.json').read_text())
    if args.stage=='arithmetic':arithmetic(out)
    elif args.stage=='gamma':gamma_distributions(plan,out)
    else:class_certificates(plan,out)
if __name__=='__main__':main()
