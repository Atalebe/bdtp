"""Frozen local calibration panels; all output paths must be new.
No downloads, git actions, modifications to parents, or outcome-driven retries.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import beta as beta_dist,binom
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import inference_core as I


def json_default(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(f'Unsupported JSON type {type(x)!r}')
def write(p,obj):p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False,default=json_default)+'\n')
def cp(k,n):
    return [0. if k==0 else float(beta_dist.ppf(.025,k,n-k+1)),1. if k==n else float(beta_dist.ppf(.975,k+1,n-k))]
def tv(a,b):return float(.5*np.abs(a-b).sum())


def inverse_case(case,plan):
    model=I.FiniteModel(case['design'],case['observation'])
    R=plan['inverse_replicates_per_case']; names=['shared','prior_return','iid_wrong']
    if case['observation']=='mass_truncated':names+=['drop_missing_wrong']
    methods={n:{'pit':{q:[] for q in ['curvature','amplitude','gamma','joint_loglik']},'coverage':[], 'included_mass':[],
                'tv_to_exact':[],'mean_squared_probability_error_h1':[], 'gamma_entropy_fraction':[]} for n in names}
    for rep in range(R):
        rng=np.random.default_rng(I.seed_for(plan['master_seed'],case['id'],rep,'inverse_data'))
        theta=int(rng.choice(len(model.prior),p=model.prior))
        data=model.draw(rng,case['worlds'],case['objects'],theta,'inverse',case['id']+f':{rep}')
        ll=model.log_likelihood(data);exact=I.normalize_log_weights(ll,model.prior)
        pp={'shared':exact,'prior_return':model.prior,'iid_wrong':model.posterior(data,'iid_wrong')}
        if 'drop_missing_wrong' in names:pp['drop_missing_wrong']=model.posterior(data,'drop_missing_wrong')
        quantities={'curvature':model.params[:,0],'amplitude':model.params[:,1],'gamma':model.params[:,2], 'joint_loglik':np.round(ll,10)}
        rng_tie=np.random.default_rng(I.seed_for(plan['master_seed'],case['id'],rep,'tie_randomization'))
        ties=rng_tie.random(len(quantities))
        for n,p in pp.items():
            rec=methods[n]
            for (key,val),v in zip(quantities.items(),ties):rec['pit'][key].append(I.randomized_pit(val,p,theta,v))
            order=np.argsort(-p,kind='stable'); stop=int(np.searchsorted(np.cumsum(p[order]),.9,side='left'))+1; region=order[:stop]
            rec['coverage'].append(int(theta in region));rec['included_mass'].append(float(p[region].sum()))
            rec['tv_to_exact'].append(tv(p,exact))
            pred=model.forecast(p)[0];pt=model.p[0,:,theta]
            rec['mean_squared_probability_error_h1'].append(float(model.obs_prob@((pred-pt)**2)))
            rec['gamma_entropy_fraction'].append(I.posterior_mass_summary(model,p,theta)['gamma_entropy_fraction'])
    results={}
    for n,v in methods.items():
        results[n]={
          'pit_KS':{q:I.ks_uniform(x) for q,x in v['pit'].items()},
          'pit_decile_counts':{q:np.histogram(x,bins=np.linspace(0,1,11))[0].tolist()for q,x in v['pit'].items()},
          'joint_HPD_coverage':float(np.mean(v['coverage'])),
          'joint_HPD_actual_mean_mass':float(np.mean(v['included_mass'])),
          'mean_TV_to_exact':float(np.mean(v['tv_to_exact'])),
          'median_TV_to_exact':float(np.median(v['tv_to_exact'])),
          'mean_probability_MSE_h1':float(np.mean(v['mean_squared_probability_error_h1'])),
          'mean_gamma_entropy_fraction':float(np.mean(v['gamma_entropy_fraction'])),
          'raw':v}
    return {'case':case,'replicates':R,'methods':results,'truth_sampling':'uniform 60-label model prior','roles':'inverse calibration, not adequacy evidence'}


def predictive_case(case,plan,out):
    model=I.FiniteModel(case['design'],case['observation']);truth=model.parameter_index(*case['truth'])
    seed=I.seed_for(plan['master_seed'],case['id'],0,'frozen_library_training')
    train=model.draw(np.random.default_rng(seed),plan['predictive_training_worlds'],plan['predictive_training_objects'],truth,'training',case['id']+':training')
    lib=I.make_library(model,train)
    naive_prior=model.forecast(model.prior);pred=model.forecast(lib.posterior);recpred=model.retained_state_forecast(lib.posterior)
    wrong_post=model.posterior(train,'iid_wrong')
    trunc_post=model.posterior(train,'drop_missing_wrong') if model.observation=='mass_truncated' else None
    risks=model.exact_brier_risk(lib.predictions,truth);bayes=risks.min(axis=1)
    gaps={str(k):(risks[:,list(ix)].min(axis=1)-bayes).tolist()for k,ix in lib.classes.items()}
    param=I.posterior_mass_summary(model,lib.posterior,truth)
    descriptive={'case':case,'training_worlds':len(train.world_ids),'training_rows':int(train.observed.size),'posterior':lib.posterior.tolist(),
      'parameters':model.params.tolist(),'identifiability':param,
      'gamma_prior':[1/6]*6,'gamma_posterior':param['gamma_probabilities'],
      'exact_class_gaps':gaps,'class_sizes':{str(k):len(x)for k,x in lib.classes.items()},
      'true_strict_level':next((k for k in (1,2,3)if max(gaps[str(k)])<plan['epsilon']),None),
      'posterior_forecast_brier_risk':model.exact_brier_risk(pred,truth).tolist(),
      'prior_forecast_brier_risk':model.exact_brier_risk(naive_prior,truth).tolist(),
      'posterior_probability_MSE':(model.exact_brier_risk(pred,truth)-bayes).tolist(),
      'bayes_brier_risk':bayes.tolist(),
      'observed_support_size':len(model.latent),'summary_support_size':len(model.summary_groups),
      'recurrence_forecast_max_error':float(np.max(abs(pred-recpred))),
      'iid_wrong_TV_to_exact_posterior':tv(lib.posterior,wrong_post),
      'drop_missing_wrong_TV_to_exact_posterior':None if trunc_post is None else tv(lib.posterior,trunc_post),
      'chosen_predictor_indices':{str(k):list(v)for k,v in lib.chosen.items()},
      'chosen_predictor_names':{str(k):[lib.names[x]for x in v]for k,v in lib.chosen.items()},
      'response_mean_predictions_h1':(np.sum(model.mu[0]*lib.posterior[None,:],axis=1)).tolist(),
      'not_intrinsic_dimension':True}
    write(out/(case['id']+'_fit.json'),descriptive)
    R=plan['predictive_replicates_per_case']
    cells=[]
    for G in plan['test_world_counts']:
        records=[]
        for rep in range(R):
            s=I.seed_for(plan['master_seed'],case['id']+f':G{G}',rep,'heldout_worlds')
            test=model.draw(np.random.default_rng(s),G,plan['test_objects_per_world'],truth,'test',case['id']+f':G{G}:rep{rep}')
            I.validate_split(model,train,test)
            records.append(I.evaluate_library(model,lib,test,truth,plan['alpha_per_fixed_case'],plan['epsilon']))
        counts=Counter(str(r['certified_level'])for r in records);false=sum(r['false_adequacy']for r in records)
        res={'case_id':case['id'],'groups':G,'objects_per_group':plan['test_objects_per_world'],'replicates':R,
             'certified_level_counts':{k:counts.get(k,0)for k in ['1','2','3','None']},
             'false_adequacy_count':int(false),'false_adequacy_pointwise_95_interval':cp(false,R),
             'false_adequacy_alarm_pvalue':float(binom.sf(false-1,R,plan['alpha_per_fixed_case'])) if false>0 else 1.,
             'mean_component_bounds':{str(k):np.mean([r['bounds'][str(k)]for r in records],axis=0).tolist() for k in (1,2,3)},
             'exact_class_gaps':gaps,'true_strict_level':descriptive['true_strict_level'],
             'replicate_records':records}
        write(out/(case['id']+f'_G{G}.json'),res);cells.append(res)
        print('completed predictive',case['id'],G,flush=True)
    return descriptive,cells


def numerical_checks(out):
    """Independent direct likelihood/posterior verification at 65 decimal digits."""
    import mpmath as mp
    mp.mp.dps=65
    checks=[]
    for design,obs in [('excited','complete'),('excited','mass_truncated'),('constant_gamma','complete')]:
        m=I.FiniteModel(design,obs);truth=m.parameter_index(.3,8,1.8)
        data=m.draw(np.random.default_rng(87231),2,2,truth,'fixture','mp')
        got=m.posterior(data)
        vals=[]
        for p in range(len(m.params)):
            val=mp.mpf(1)/len(m.params)
            for world in range(2):
                wg=mp.mpf(0)
                for offset,wp in zip(m.offsets,m.offset_probs):
                    prod=mp.mpf(str(wp))
                    for obj in range(2):
                        o=int(data.observed[world,obj]);total=mp.mpf(0)
                        # original full latent index with duplicate mass, not optimized array
                        for ix in m.latent[o]:
                            e=(mp.mpf(str(data.responses[world,obj,0]))-mp.mpf(str(m.mu[0,ix,p]))-mp.mpf(str(offset)))/mp.mpf(str(m.sigma))
                            total+=mp.exp(-e*e/2)/(mp.mpf(str(m.sigma))*mp.sqrt(2*mp.pi))
                        prod*=total/len(m.latent[o])
                    wg+=prod
                val*=wg
            vals.append(val)
        norm=sum(vals);reference=np.array([float(x/norm)for x in vals]);err=float(np.max(abs(reference-got)))
        checks.append({'design':design,'observation':obs,'posterior_max_abs_error':err,'pass':err<2e-11})
    # Independent event probability computation using mpmath erf and exact finite weights.
    m=I.FiniteModel(observation='mass_truncated');err=0.
    for o in [0,4,8,35,100]:
        for p in [0,12,29,45,59]:
            ans=mp.mpf(0)
            for ix in m.latent[o]:
                for b,w in zip(m.offsets,m.offset_probs):
                    z=(mp.mpf(str(m.mu[0,ix,p]))+mp.mpf(str(b)))/mp.mpf(str(m.sigma))
                    ans+=mp.mpf(str(w))*(1+mp.erf(z/mp.sqrt(2)))/2/len(m.latent[o])
            err=max(err,abs(float(ans)-m.p[0,o,p]))
    checks.append({'check':'25 finite-mixture event probabilities','max_abs_error':err,'pass':err<2e-11})
    # Rank-free information checks on all candidate mean-response vectors.
    ident=[]
    for design in ['excited','summary_only','constant_gamma']:
        m=I.FiniteModel(design)
        keys={}
        for p in range(60):keys.setdefault(tuple(np.round(m.mu[0,:,p],12)),[]).append(p)
        ident.append({'design':design,'joint_parameter_labels':60,'distinct_horizon1_mean_functions_rounded_12':len(keys),
          'equivalence_classes':list(keys.values()),'note':'function equality reference on finite support; rounding tolerance tested separately, not latent-dimension theorem'})
    write(out/'high_precision_checks.json',{'checks':checks,'all_pass':all(x['pass']for x in checks),'identity_diagnostics':ident,
       'precision_digits':65,'not_interval_arithmetic':True})
    return checks


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--plan',default=str(ROOT/'contracts/development_plan.json'))
    a.add_argument('--stage',choices=['numerical','inverse','predictive'],required=True);args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.plan).read_text())
    if args.stage=='numerical':
        good=all(c['pass']for c in numerical_checks(out));return 0 if good else 1
    if args.stage=='inverse':
        results=[]
        for c in plan['inverse_cases']:
            r=inverse_case(c,plan);results.append(r);write(out/(c['id']+'.json'),r);print('completed inverse',c['id'],flush=True)
        R=plan['inverse_replicates_per_case'];F=len(results)*4;threshold=math.sqrt(math.log(2*F/.01)/(2*R))
        checks=[]
        for r in results:
            for quantity,d in r['methods']['shared']['pit_KS'].items():checks.append({'case':r['case']['id'],'quantity':quantity,'D':d,'threshold':threshold,'pass':d<=threshold})
            difference=abs(r['methods']['shared']['joint_HPD_coverage']-r['methods']['shared']['joint_HPD_actual_mean_mass'])
            # Bound indicator-minus-included-mass in [-1,1], independently over datasets.
            bound=math.sqrt(2*math.log(2*len(results)/.01)/R)
            checks.append({'case':r['case']['id'],'check':'HPD calibration actual mass','difference':difference,'bound':bound,'pass':difference<=bound})
        # Prior return detection is a sensitivity result, not a necessary coverage theorem.
        summary={'all_checks_pass':all(x['pass']for x in checks),'checks':checks,'rank_family_alarm':.01,'cases':len(results),'datasets':len(results)*R,
            'brief':[{'case':r['case'],'methods':{n:{k:v for k,v in z.items()if k!='raw'}for n,z in r['methods'].items()}}for r in results]}
        write(out/'summary.json',summary);return 0 if summary['all_checks_pass'] else 1
    fits=[];cells=[]
    for c in plan['predictive_cases']:
        f,rr=predictive_case(c,plan,out);fits.append(f);cells+=rr
    alarm=.01/len(cells)
    checks=[{'cell':c['case_id']+':'+str(c['groups']),'check':'false_adequacy_gross_alarm','pvalue':c['false_adequacy_alarm_pvalue'],'threshold':alarm,'pass':c['false_adequacy_alarm_pvalue']>=alarm}for c in cells]
    checks +=[{'case':f['case']['id'],'check':'same forecast direct and accumulator','error':f['recurrence_forecast_max_error'],'pass':f['recurrence_forecast_max_error']<plan['numeric_absolute_tolerance']}for f in fits]
    for f in fits:
        if f['case']['design']in ('constant_gamma','summary_only'):
            checks.append({'case':f['case']['id'],'check':'unidentified gamma remains prior','pass':max(abs(np.asarray(f['gamma_posterior'])-1/6))<1e-12})
    summary={'all_checks_pass':all(x['pass']for x in checks),'checks':checks,'fits':fits,
        'cells':[{k:v for k,v in c.items()if k!='replicate_records'}for c in cells],
        'test_datasets':len(cells)*plan['predictive_replicates_per_case'],
        'training_libraries':len(fits),'inference_scope':'conditional on five frozen training libraries; not 3000 retraining experiments; nominal alpha per cell only'}
    write(out/'summary.json',summary);return 0 if summary['all_checks_pass']else 1

if __name__=='__main__':sys.exit(main())
