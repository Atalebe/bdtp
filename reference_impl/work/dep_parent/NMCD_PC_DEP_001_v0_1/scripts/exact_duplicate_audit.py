"""Two-predictor duplication control: exact binomial weights, 80-digit thresholds."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,math,sys
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];mp.mp.dps=80

def audit(output):
    plan=json.loads((ROOT/'contracts/development_plan.json').read_text());cfg=plan['separate_exact_control']
    eps=mp.mpf(1)/16;alpha=mp.mpf(1)/20;z=mp.log(2/alpha);results=[]
    for G in cfg['G']:
        for m in cfg['m']:
            for kind in ('group_max_eb','group_max_hoeffding','pooled_row_max_eb'):
                event=[];near=mp.inf
                for k in range(G+1):
                    p=mp.mpf(k)/G;mean=p/2-mp.mpf(5)/16
                    n=G if kind.startswith('group_') else G*m
                    var=(p*(1-p)/4)*n/(n-1)
                    if kind.endswith('hoeffding'):bound=max(mp.mpf(0),mean)+mp.sqrt(2*mp.log(1/alpha)/n)
                    else:
                        rem=14*z/(3*(n-1));bound=max(rem,mean+mp.sqrt(2*var*z/n)+rem)
                    near=min(near,abs(bound-eps))
                    if bound<eps:event.append(k)
                prob=Fraction(sum(math.comb(G,k)*3**k for k in event),4**G)
                results.append({'G':G,'m':m,'rows':G*m,'method':kind,'nominal_alpha':.05,'exact_false_certification_probability':float(prob),
                                'probability_80_digits':mp.nstr(mp.mpf(prob.numerator)/prob.denominator,75),
                                'probability_numerator':str(prob.numerator),'probability_denominator':str(prob.denominator),
                                'rejection_k':event,'minimum_threshold_distance':mp.nstr(near,30),
                                'claim':'exact rational probabilities conditional on high-precision, not interval-certified, threshold comparison'})
    obj={'experiment':'NMCD-PC-DEP-001','method':'two-predictor fixed-oracle boundary law; independent worlds, perfect duplicates within world',
         'count':len(results),'results':results,
         'valid_group_max_error':max(r['exact_false_certification_probability'] for r in results if r['method'].startswith('group_')),
         'invalid_row_max_error':max(r['exact_false_certification_probability'] for r in results if r['method']=='pooled_row_max_eb' and r['m']>1)}
    if obj['valid_group_max_error']>.05:raise ArithmeticError('valid group reference violates nominal level')
    if obj['invalid_row_max_error']<=.05:raise ArithmeticError('predeclared severe duplication fixture did not expose invalid inference')
    output.parent.mkdir(exist_ok=True,parents=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');print('EXACT_DUPLICATE_CHECKS',obj['count'],'valid_max',obj['valid_group_max_error'],'invalid_max',obj['invalid_row_max_error'])
    return obj
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();audit(a.output)
