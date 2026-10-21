"""Frozen synthetic-development calibration. No network, Git, or source modification."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,os,sys,time,zipfile,resource
import numpy as np
from scipy.stats import beta,binom
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import dependence_core as C
P=C.P

def dump(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rate_interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def save_npz(path,**arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,arr in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,arr,allow_pickle=False)
            zi=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(zi,b.getvalue(),compresslevel=6)

def one_case(case,plan,out,truth):
    R=plan['replicates_per_case'];G=case['groups_per_split'];m=case['rows_per_group']
    theta=float(case['theta']);c=float(case['copy_probability']);master=plan['master_seed'];alpha=plan['alpha']
    facts=C.oracle_facts(theta,c,G,m)
    # columns: 0:12 (method,horizon,dimension) bounds; 12:15 certified dimensions (3 means no cert);
    # 15:19 fitted predictor indices h1k2,h1k1,h2k2,h2k1; 19 fixed contrast mean; 20 empirical group variance; 21 row variance.
    raw=np.empty((R,22),dtype=np.float64)
    for rep in range(R):
        tr=C.draw_stats(np.random.Generator(np.random.PCG64(C.stream_seed(case,rep,'train',master))),G,m,theta,c)
        te=C.draw_stats(np.random.Generator(np.random.PCG64(C.stream_seed(case,rep,'test',master))),G,m,theta,c)
        fitted=C.fit_groups(tr);b=C.all_bounds(te,fitted,alpha)
        raw[rep,:12]=[b[name][h][k] for name in C.METHODS for h in (1,2) for k in (2,1)]
        raw[rep,12:15]=[3 if (v:=P.decide(b[name])) is None else v for name in C.METHODS]
        raw[rep,15:19]=[fitted[h][k] for h in (1,2) for k in (2,1)]
        raw[rep,19]=C.oracle_contrast_mean(te,facts)
        _,var,rowvar=C.pair_moments(te,facts['state_index'],0)
        raw[rep,20]=var[facts['comparator_index']];raw[rep,21]=rowvar[facts['comparator_index']]
    if not np.isfinite(raw).all():raise ArithmeticError('nonfinite calibration output')
    result={'case':case,'replicates':R,'truth':truth,'oracle_contrast':facts,'methods':{}}
    for j,name in enumerate(C.METHODS):
        dim=raw[:,12+j];null_event=(dim<truth['C_strict'])
        num=int(null_event.sum());calib=[];coverage=[]
        for hi,h in enumerate((1,2)):
            for ki,k in enumerate((2,1)):
                u=raw[:,4*j+2*hi+ki];gap=truth['gaps'][h][k]
                coverage.append({'horizon':h,'dimension':k,'misses':int((u<gap).sum()),'mean_upper_bound':float(u.mean())})
                if gap>=P.EPS:
                    nfalse=int((u<P.EPS).sum())
                    calib.append({'horizon':h,'dimension':k,'false_rejections':nfalse,'binomial_nominal_upper_tail':float(binom.sf(nfalse-1,R,alpha))})
        result['methods'][name]={'inferential_status':C.row_control_status(c,m) if j==2 else 'IID_REALISATION_CERTIFICATE',
            'certified_1':int((dim==1).sum()),'certified_2':int((dim==2).sum()),'unresolved':int((dim==3).sum()),
            'certified_1_rate':float((dim==1).mean()),'certified_1_interval_95_pointwise':rate_interval(int((dim==1).sum()),R),
            'false_under_dimension':num,'false_under_dimension_rate':num/R,
            'false_interval_95_pointwise':rate_interval(num,R),'binomial_nominal_upper_tail':float(binom.sf(num-1,R,alpha)),
            'true_null_components':calib,'component_coverage':coverage}
    measured_var=float(raw[:,19].var(ddof=1))
    result['variance']={'replicate_variance_of_mean':measured_var,'exact_variance_of_mean':facts['grand_mean_variance'],
       'empirical_to_exact_ratio':measured_var/facts['grand_mean_variance'],
       'mean_reported_group_variance_of_mean':float(raw[:,20].mean()/G),
       'mean_naive_row_variance_of_mean':float(raw[:,21].mean()/(G*m)),
       'naive_to_true_ratio':float(raw[:,21].mean()/(G*m)/facts['grand_mean_variance'])}
    rawfile=out/(case['case_id']+'.npz');save_npz(rawfile,replicates=raw)
    result['raw_sha256']=sha(rawfile);dump(out/(case['case_id']+'.json'),result)
    return result

def run(plan_path,out):
    start=time.monotonic();plan=json.loads(plan_path.read_text())
    plan_sha=sha(plan_path)
    expected=(ROOT/'contracts/PLAN_SHA256.txt').read_text().split()[0]
    if plan_sha!=expected:raise ValueError('development plan changed')
    if np.__version__!='2.3.5':raise ValueError('NumPy 2.3.5 required for this development record')
    out.mkdir(parents=True,exist_ok=False);results=[];truths={}
    for case in plan['cases']:
        if time.monotonic()-start>plan['resource_caps']['wall_seconds']:raise TimeoutError('registered wall-clock cap reached')
        th=float(case['theta'])
        if th not in truths:truths[th]=P.exact_truth(th,d=C.D)
        cs=time.monotonic();result=one_case(case,plan,out,truths[th]);results.append(result)
        with (out/'timing.jsonl').open('a') as f:f.write(json.dumps({'case_id':case['case_id'],'seconds':time.monotonic()-cs,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})+'\n')
        print(case['case_id'],case,'finished',round(time.monotonic()-cs,2),'s',flush=True)
    checks=[];M=6*len(results);alarm=.01/M
    for r in results:
        for name in C.METHODS[:2]:
            z=r['methods'][name]
            okay=z['binomial_nominal_upper_tail']>=alarm and all(v['binomial_nominal_upper_tail']>=alarm for v in z['true_null_components'])
            checks.append({'check':'nominal_false_certification_gross_alarm','case_id':r['case']['case_id'],'method':name,'pass':okay})
        vr=r['variance']['empirical_to_exact_ratio']
        checks.append({'check':'variance_identity_gross_check','case_id':r['case']['case_id'],'pass':.75<=vr<=1.25,'ratio':vr})
    sent=next(r for r in results if float(r['case']['theta'])==.5 and float(r['case']['copy_probability'])==1 and r['case']['groups_per_split']==256 and r['case']['rows_per_group']==256)
    k=sent['methods']['pooled_row_max_eb']['certified_1'];R=plan['replicates_per_case']
    lo=0. if k==0 else float(beta.ppf(.05,k,R-k+1))
    checks.append({'check':'declared_invalid_control_sensitivity','pass':lo>.05,'count':k,'lower_95_one_sided':lo,'case_id':sent['case']['case_id']})
    # Paired copy-only cases must have identical realisation analyses. Only the row-level fiction changes.
    groups={}
    for r in results:
        c=r['case']
        if float(c['copy_probability'])==1:groups.setdefault((c['theta'],c['groups_per_split']),[]).append(r)
    for key,rr in groups.items():
        if len(rr)<2:continue
        mats=[np.load(out/(r['case']['case_id']+'.npz'))['replicates'] for r in rr]
        keep=list(range(8))+[12,13]+list(range(15,21))
        eq=all(np.array_equal(mats[0][:,keep],v[:,keep]) for v in mats[1:])
        checks.append({'check':'exact_duplicate_group_information_invariance','pass':eq,'cases':[r['case']['case_id'] for r in rr]})
    for r in results:
        if r['case']['rows_per_group']==1:
            a=np.load(out/(r['case']['case_id']+'.npz'))['replicates']
            eq=np.allclose(a[:,:4],a[:,8:12],rtol=0,atol=1e-13) and np.array_equal(a[:,12],a[:,14])
            checks.append({'check':'one_row_same_certificate','case_id':r['case']['case_id'],'pass':bool(eq)})
    summary={'experiment':'NMCD-PC-DEP-001','plan_sha256':plan_sha,'number_cases':len(results),'replicates_per_case':R,
             'dataset_pairs':len(results)*R,'paired_across_m_when_copy1':True,'primary':'group_max_eb',
             'status':'LOCAL_DEVELOPMENT_CHECKS_PASS' if all(c['pass'] for c in checks) else 'HOLD',
             'checks':checks,'scope':'Synthetic iid worlds with specified within-world dependence only; no general dependent-world or cosmological certificate.'}
    dump(out/'summary.json',summary);dump(out/'all_cases.json',results)
    columns=['case_id','theta','copy_probability','groups','rows_per_group','rows_per_split','method','status','certified_1','certified_2','unresolved','false_under_dimension','variance_ratio']
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns)
        for r in results:
            c=r['case']
            for name,z in r['methods'].items():w.writerow([c['case_id'],c['theta'],c['copy_probability'],c['groups_per_split'],c['rows_per_group'],c['groups_per_split']*c['rows_per_group'],name,z['inferential_status'],z['certified_1'],z['certified_2'],z['unresolved'],z['false_under_dimension'],r['variance']['empirical_to_exact_ratio']])
    if summary['status']=='HOLD':raise RuntimeError('At least one frozen development requirement failed; no milestone entry')
    print('CALIBRATION_COMPLETE',len(results),'cases',len(results)*R,'dataset pairs',round(time.monotonic()-start,2),'seconds',flush=True)
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,default=ROOT/'contracts/development_plan.json');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run(a.plan,a.output)
