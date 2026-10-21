"""Extract all published cells from retained replicate-level records; no new simulations."""
from pathlib import Path
import json,csv,hashlib,collections,sys
R=Path(__file__).resolve().parents[1];P=R/'parent/Finite_Dictionary_Predictive_Adequacy_v0_02';OUT=R/'manuscript/generated'
METHODS=('erm','variance_penalized','posterior_risk','evaluation_class')
ORDER={'history_moderate':0,'history_strong':1,'truncated_strong':2,'summary_only':3,'constant_gamma':4}
LABEL={'history_moderate':'Moderate','history_strong':'Strong','truncated_strong':'Truncated','summary_only':'Summary-only','constant_gamma':'Constant'}
def load(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cells=load(P/'evidence/matched_run01/summary.json')['cells'];cells=sorted(cells,key=lambda c:(ORDER[c['case']],c['N'],c['T']))
 checks=[];rows=[]
 for c in cells:
  key=f"{c['case']}_N{c['N']}_T{c['T']}";path=P/'evidence/matched_run01/cells'/(key+'.json');raw=load(path)
  for variant in ('generic','range'):
   for met in METHODS:
    s=c['summary'][variant][met];rr=raw['records'];n=len(rr)
    count=sum(r['truth'][variant]['methods'][met]['certifies_summary']for r in rr)
    false=sum(r['truth'][variant]['methods'][met]['false_class_certification']for r in rr)
    levels=dict(collections.Counter(str(r['answers'][variant][met]['certified_level'])for r in rr))
    assert (count,false,n,levels)==(s['certificates'],s['false_class_certificates'],s['n'],s['levels'])
    rows.append({'case':c['case'],'N':c['N'],'T':c['T'],'G':c['G'],'method':met,'envelope':variant,'certificates':count,'replicates':n,'false_certificates':false})
    checks.append({'cell':key,'method':met,'envelope':variant,'pass':True})
 def mk_table(cs,appendix):
  desc=(r'Complete remaining allocations and controls. Entries are generic $\to$ pair-envelope certificate counts on identical fitted libraries and evaluation groups. '
        r'$n_{\rm rep}$ is the number of trials, not the group count. All four rules are shown; no cell is omitted for an unfavorable outcome.' if appendix else
        r'Truncated strong-history reference: summary-class or smaller certificates out of 100 paired trials. Every arrow is generic $\to$ pair envelope. ``Variance'' is the registered training penalty, not an alternative evaluation bound.')
  label='tab:other' if appendix else 'tab:allocation'
  s='\\begin{table}['+('ht' if appendix else 'tb')+']\n\\centering\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\caption{'+desc+'}\\label{'+label+'}\n'
  cols='lrrr rrrr' if appendix else 'rrr rrrr'
  s+='\\begin{tabular}{@{}'+cols+'@{}}\n\\toprule\n'
  s+=(r'Design & $N$ & $T$ & $n_{\rm rep}$' if appendix else r'Total $N$ & Train $T$ & Evaluate $G$')+r' & ERM & Variance & Posterior & Class\\'+'\n\\midrule\n'
  prev=None
  for c in cs:
   if appendix and prev is not None and prev!=c['case']:s+='\\midrule\n'
   vals=([LABEL[c['case']],f"{c['N']:,}",f"{c['T']:,}",str(c['summary']['generic']['erm']['n'])] if appendix else [f"{c[z]:,}"for z in ('N','T','G')])
   vals += [f"${c['summary']['generic'][met]['certificates']}\\to {c['summary']['range'][met]['certificates']}$"for met in METHODS]
   s+=' & '.join(vals)+r'\\'+'\n';prev=c['case']
  return s+'\\bottomrule\n\\end{tabular}\n\\end{table}\n'
 (OUT/'allocation_table.tex').write_text(mk_table([c for c in cells if c['case']=='truncated_strong'],False))
 (OUT/'all_remaining_table.tex').write_text(mk_table([c for c in cells if c['case']!='truncated_strong'],True))
 dep=load(P/'evidence/additional01/dep/summary.json')['cases'];dep=sorted(dep,key=lambda c:(c['case']['groups_per_split'],float(c['case']['theta']),c['case']['rows_per_group']))
 ss=r'''\begin{table}[tb]
\centering\small
\caption{All dependence-extension cells, with 2,000 trials each. At $\theta=0$ certificates measure power; at $\theta=.5$ they are false certifications. Groups are independent; copied rows are not. Every generic-group entry at $G=256$ is structurally zero. The three copy factors within each $(G,\theta)$ family use the same independent episodes.}\label{tab:dup}
\begin{tabular}{@{}rrr rrrr@{}}
\toprule
&&&\multicolumn{2}{c}{Generic envelope}&\multicolumn{2}{c}{Pair envelope}\\
$G$&$\theta$&Copies $m$&Group&Row&Group&Row\\
\midrule
'''
 for c in dep:
  cfg=c['case'];raw=load(P/'evidence/additional01/dep'/f"theta{float(cfg['theta'])}_G{cfg['groups_per_split']}_m{cfg['rows_per_group']}.json")
  for meth in ('generic_group','generic_row','range_group','range_row'):
   ct=sum(r['levels'][meth]==1 for r in raw['records'])
   assert ct==c['summary'][meth]['certified_1'] and len(raw['records'])==2000
   checks.append({'dependence_case':cfg,'method':meth,'pass':True})
  ss+=' & '.join([f"{cfg['groups_per_split']:,}",str(cfg['theta']),str(cfg['rows_per_group'])]+[f"{c['summary'][m]['certified_1']:,}"for m in ('generic_group','generic_row','range_group','range_row')])+r'\\'+'\n'
 (OUT/'dependence_table.tex').write_text(ss+'\\bottomrule\\end{tabular}\\end{table}\n')
 power=load(P/'evidence/power_final/power.json')['records'];boundary=[v for v in power if v['theta']=='1/2']
 s=r'''\begin{table}[tb]
\centering\small
\caption{Attained false-certification probability at $\Delta=\varepsilon=1/16$ in the fixed two-predictor reference. Nominal $\alpha=0.05$. These are finite-binomial probabilities, not estimated error rates or the size of the full dictionary procedure.}\label{tab:attained}
\begin{tabular}{@{}rrrrr@{}}
\toprule
&\multicolumn{2}{c}{Probability}&\multicolumn{2}{c}{Percentage}\\
Groups $G$&Generic&Pair&Generic&Pair\\
\midrule
'''
 def sci(x):
  if x==0:return '$0$'
  a,b=f'{x:.6e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
 for G in (256,1024,4096):
  vals={z['rule']:z['probability']for z in boundary if z['G']==G}
  s+=f"{G:,} & {sci(vals['generic'])} & {sci(vals['range'])} & {100*vals['generic']:.6f} & {100*vals['range']:.6f}"+r'\\'+'\n'
 (OUT/'attained_error_table.tex').write_text(s+'\\bottomrule\\end{tabular}\\end{table}\n')
 write(R/'evidence/completeness_checks.json',{'checks':checks,'all_passed':all(x['pass']for x in checks),'counts':{'fitting_cells':len(cells),'method_envelope_fitting_cells':len(rows),'dependence_cells':len(dep),'dependence_method_cells':4*len(dep),'boundary_entries':len(boundary)},'new_Monte_Carlo':False})
 write(R/'manuscript/data/all_results.json',{'fitting':rows,'dependence':dep,'boundary':boundary})
 with (R/'manuscript/data/all_fitting_results.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Verified',len(checks),'record-to-summary checks; all20fitting/12dependence cells extracted.')
if __name__=='__main__':main()
