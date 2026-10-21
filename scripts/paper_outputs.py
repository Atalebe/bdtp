"""Rebuild main-paper numerical tables and four figures from checked references.
No simulation, parameter fitting, network access, or source overwrite.
"""
from __future__ import annotations
import argparse, collections, csv, json, re, shutil, sys, traceback
from pathlib import Path
from paper_campaign import ROOT, verify_imports
METHODS=('erm','variance_penalized','posterior_risk','evaluation_class')
ORDER={'history_moderate':0,'history_strong':1,'truncated_strong':2,'summary_only':3,'constant_gamma':4}

def load(path):
    return json.loads(path.read_text())

def write_csv(path,rows):
    if not rows:
        raise ValueError('Empty table')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)

def table_data():
    base=ROOT/'reference_impl/evidence'
    fit=load(base/'matched_run01/summary.json')
    cells=sorted(fit['cells'],key=lambda c:(ORDER[c['case']],c['N'],c['T']))
    checks=[]; fitrows=[]
    for cell in cells:
        key=f"{cell['case']}_N{cell['N']}_T{cell['T']}"
        raw=load(base/'matched_run01/cells'/f'{key}.json')['records']
        for variant in ('generic','range'):
            for method in METHODS:
                s=cell['summary'][variant][method]
                ct=sum(r['truth'][variant]['methods'][method]['certifies_summary'] for r in raw)
                er=sum(r['truth'][variant]['methods'][method]['false_class_certification'] for r in raw)
                levels=dict(collections.Counter(str(r['answers'][variant][method]['certified_level']) for r in raw))
                if (ct,er,len(raw),levels)!=(s['certificates'],s['false_class_certificates'],s['n'],s['levels']):
                    raise ValueError('Fitting reference mismatch '+key)
                checks.append({'item':key,'method':method,'envelope':variant,'pass':True})
                fitrows.append({'case':cell['case'],'N':cell['N'],'T':cell['T'],'G':cell['G'],'method':method,'envelope':variant,'certificates':ct,'replicates':len(raw),'false_certificates':er})
    dep=load(base/'additional01/dep/summary.json')['cases']; deprows=[]
    for cell in sorted(dep,key=lambda c:(c['case']['groups_per_split'],float(c['case']['theta']),c['case']['rows_per_group'])):
        c=cell['case'];key=f"theta{float(c['theta'])}_G{c['groups_per_split']}_m{c['rows_per_group']}"
        raw=load(base/'additional01/dep'/f'{key}.json')['records']
        row={'G':c['groups_per_split'],'theta':float(c['theta']),'m':c['rows_per_group'],'replicates':len(raw)}
        for method in ('generic_group','generic_row','range_group','range_row'):
            ct=sum(r['levels'][method]==1 for r in raw)
            if ct!=cell['summary'][method]['certified_1'] or len(raw)!=cell['summary'][method]['n']:
                raise ValueError('Dependence reference mismatch '+key)
            row[method]=ct;checks.append({'item':key,'method':method,'pass':True})
        deprows.append(row)
    inf=load(ROOT/'reference/inference/predictive/summary.json');risks=load(ROOT/'reference/reference_risks.json')
    lookup={x['case']:x for x in risks}; designs=[];riskrows=[]
    for f in inf['fits']:
        c=f['case']; ident=c['id']; rr=load(ROOT/f'reference/inference/predictive/{ident}_G8192.json')['replicate_records'][0]
        gap=rr['exact_fitted_risks']['2'][0]-rr['oracle_risks'][0]
        row={'case':ident,'affine_gap':f['exact_class_gaps']['1'][0],'summary_gap':f['exact_class_gaps']['2'][0],'fitted_summary_gap':gap,'bayes_risk':f['bayes_brier_risk'][0]}
        expected=lookup[ident]
        for actual,key in [(row['affine_gap'],'gap_affine_h1'),(row['summary_gap'],'gap_summary_h1'),(gap,'selected_gap_h1'),(row['bayes_risk'],'bayes_risk_h1')]:
            if abs(actual-expected[key])>2e-14:
                raise ValueError('Risk extraction differs '+ident+' '+key)
        riskrows.append(row)
        designs.append({'case':ident,'B':c['truth'][0],'A':c['truth'][1],'gamma':c['truth'][2],'design':c['design'],'observation':c['observation']})
    power=load(base/'power_final/power.json')['records'];boundary=[]
    for G in (256,1024,4096):
        x={r['rule']:r['probability'] for r in power if r['G']==G and r['theta']=='1/2'}
        boundary.append({'G':G,'generic_probability':x['generic'],'pair_probability':x['range'],'generic_percent':100*x['generic'],'pair_percent':100*x['range']})
    tables={'table1_designs':designs,'table2_risk_gaps':riskrows,'table3_dependence':deprows,'table4_truncated_allocation':[r for r in fitrows if r['case']=='truncated_strong'],'table5_attained_error':boundary,'table6_other_allocations':[r for r in fitrows if r['case']!='truncated_strong']}
    return tables,checks,cells,power

def numeric_rows(text):
    rows=[]
    for line in text.splitlines():
        if '&' not in line or not line.rstrip().endswith('\\\\'):
            continue
        first=line.split('&')[0].strip()
        if not (re.fullmatch(r'[0-9,]+',first) or first in ('Moderate','Strong','Summary-only','Constant')):
            continue
        rows.append(re.findall(r'(?<![A-Za-z])(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?',line))
    return rows

def validate_typeset_tables(tables):
    # Test the frozen displayed integer cells independently of stored summary labels.
    asset=ROOT/'reference/paper_assets/generated';results=[]
    for filename,rows in [('allocation_table.tex',tables['table4_truncated_allocation']),('all_remaining_table.tex',tables['table6_other_allocations'])]:
        cases=[]
        for r in rows:
            key=(r['case'],r['N'],r['T'],r['G'],r['replicates'])
            if key not in cases:cases.append(key)
        shown=numeric_rows((asset/filename).read_text())
        expected=[]
        for case,N,T,G,n in cases:
            prefix=[N,T,G] if filename=='allocation_table.tex' else [N,T,n]
            vals=[]
            for method in METHODS:
                for variant in ('generic','range'):
                    vals.append(next(r['certificates'] for r in rows if (r['case'],r['N'],r['T'],r['method'],r['envelope'])==(case,N,T,method,variant)))
            expected.append(prefix+vals)
        got=[[int(x.replace(',','')) for x in row]for row in shown]
        if got!=expected:raise ValueError('Typeset fitting cells mismatch '+filename)
        results.append({'table':filename,'rows_checked':len(expected)})
    got=[[float(x.replace(',',''))for x in row]for row in numeric_rows((asset/'dependence_table.tex').read_text())]
    expected=[[r['G'],r['theta'],r['m'],r['generic_group'],r['generic_row'],r['range_group'],r['range_row']]for r in tables['table3_dependence']]
    if got!=expected:raise ValueError('Typeset dependence cells mismatch')
    results.append({'table':'dependence_table.tex','rows_checked':len(expected)})
    shown=[]
    for line in (asset/'attained_error_table.tex').read_text().splitlines():
        if '&' not in line or not re.fullmatch(r'[0-9,]+',line.split('&')[0].strip()):
            continue
        fields=[v.strip().rstrip('\\\\').strip() for v in line.split('&')]
        def probability(value):
            if value=='$0$':return 0.
            match=re.fullmatch(r'\$([0-9.]+)\\times10\^\{(-?\d+)\}\$',value)
            if not match:raise ValueError('Unrecognized probability cell: '+value)
            return float(match[1])*10**int(match[2])
        shown.append([int(fields[0].replace(',','')),probability(fields[1]),probability(fields[2]),float(fields[3]),float(fields[4])])
    if len(shown)!=3:raise ValueError('Attained-error row count')
    for got,row in zip(shown,tables['table5_attained_error']):
        want=[row['G'],float(f"{row['generic_probability']:.6e}"),float(f"{row['pair_probability']:.6e}"),round(row['generic_percent'],6),round(row['pair_percent'],6)]
        if any(abs(x-y)>1e-15 for x,y in zip(got,want)):raise ValueError('Typeset attained-error mismatch')
    results.append({'table':'attained_error_table.tex','rows_checked':3})
    return results

def figures(out,tables,cells,power):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    fdir=out/'figures'; fdir.mkdir()
    def save(fig,name):
        fig.tight_layout(); fig.savefig(fdir/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig(fdir/(name+'.png'),dpi=160);plt.close(fig)
    rows={r['case']:r for r in tables['table2_risk_gaps']};keys=['history_moderate','history_strong','truncated_strong'];x=np.arange(3)
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    ax.bar(x-.18,[rows[k]['summary_gap']for k in keys],.36,label='Best member of summary class')
    ax.bar(x+.18,[rows[k]['fitted_summary_gap']for k in keys],.36,label='Training-selected member')
    ax.axhline(.01,linestyle='--',label='Tolerance 0.01');ax.set_xticks(x,['Moderate history','Strong history','Truncated strong']);ax.set_ylim(0,.018);ax.set_ylabel('Excess population Brier risk');ax.legend(fontsize=8.5);save(fig,'class_and_fit')
    cs=sorted([c for c in cells if c['case']=='truncated_strong'and c['N']==16384],key=lambda c:c['T'])
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    for j,(v,m,lab)in enumerate([('generic','erm','ERM, generic'),('range','erm','ERM, pair-specific'),('generic','evaluation_class','Class selection, generic'),('range','evaluation_class','Class selection, pair-specific')]):
        y=np.array([c['summary'][v][m]['rate']for c in cs]);ci=np.array([c['summary'][v][m]['ci95_pointwise']for c in cs])
        ax.errorbar(np.arange(3)+(j-1.5)*.11,y,yerr=np.vstack((y-ci[:,0],ci[:,1]-y)),fmt=['o','s','^','D'][j],capsize=3,label=lab)
    ax.set_xticks(range(3),['64 / 16,320','512 / 15,872','4,096 / 12,288']);ax.set_ylim(-.035,1.06);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Training / evaluation groups; total 16,384');ax.set_ylabel('Summary-class certificates');ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False);save(fig,'range_allocation')
    fig,ax=plt.subplots(figsize=(7.1,4.2))
    for rule,label in [('generic','Generic width 2'),('range','Pair width 1/2')]:
        rr=sorted([r for r in power if r['G']==1024 and .25<=r['theta_float']<=.6 and r['rule']==rule],key=lambda r:r['theta_float'])
        ax.plot([r['gap_over_epsilon']for r in rr],[r['probability']for r in rr],marker='o',label=label)
    ax.axvline(1,linestyle='--',label='Adequacy-null boundary');ax.axhline(.05,linestyle=':',label='Nominal error level');ax.set_ylim(-.025,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('True class gap / tolerance');ax.set_ylabel('Certificate probability');ax.legend(fontsize=8);save(fig,'binomial_power')
    audit=load(ROOT/'reference/planning_audit.json');rr=audit['planning_curves'];fig,ax=plt.subplots(figsize=(7.1,4.4))
    ax.plot([r['G']for r in rr],[r['pair_witness']for r in rr],label='Pair-specific width, including variance')
    ax.plot([r['G']for r in rr],[r['generic_witness']for r in rr],linestyle='--',label='Generic width, including variance')
    ax.axhline(.01,linestyle=':',label='Tolerance: 0.01');ax.axvline(6477,linestyle='-.',label='Pair remainder-only count: 6,477')
    for key in ('pair','generic'):
        c=audit['crossings']['70'][key]['full_formula'];ax.plot([c['first_integer']],[c['crossing_value']],marker='o',linestyle='none');ax.annotate(f"{c['first_integer']:,}",(c['first_integer'],c['crossing_value']),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlim(4096,32768);ax.set_ylim(.0075,.03);ax.set_xlabel('Independent evaluation groups');ax.set_ylabel('Upper-bound formula at population moments');ax.legend(fontsize=8);save(fig,'variance_inclusive_planning')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--figures',action='store_true');a=ap.parse_args();out=a.output.resolve()
    if out==ROOT or ROOT in out.parents and ROOT/'.local_runs' not in out.parents:raise ValueError('Use a fresh directory under .local_runs or outside the checkout')
    out.mkdir(parents=True,exist_ok=False)
    try:
        n=verify_imports();tables,checks,cells,power=table_data();typeset=validate_typeset_tables(tables)
        for name,rows in tables.items():write_csv(out/(name+'.csv'),rows)
        (out/'figure_data.json').write_text(json.dumps({'risk_gaps':tables['table2_risk_gaps'],'allocation':cells,'power':power,'planning':load(ROOT/'reference/planning_audit.json')},indent=2,sort_keys=True)+'\n')
        if a.figures:figures(out,tables,cells,power)
        (out/'COMPLETE.json').write_text(json.dumps({'status':'PASS','source_files_verified':n,'record_to_summary_checks':len(checks),'typeset_checks':typeset,'tables':6,'figures_rendered':4 if a.figures else 0,'new_simulations':0,'plot_note':'Same numerical series; regenerated appearance and PDF bytes are not claimed identical to frozen assets'},indent=2)+'\n')
        print('PAPER_OUTPUTS_PASS',len(checks),'record checks, six tables',flush=True)
    except BaseException:
        (out/'FAILURE.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':
    main()
