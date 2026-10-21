"""Pre-evaluation range sensitivity. No input mutations or external operations."""
from __future__ import annotations
from pathlib import Path
import sys, math, hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
FP=R/'work/fit_parent/NMCD_PC_FIT_001_v0_1'
sys.path.insert(0,str(FP/'src'))
import fitting_core as F
I=F.I

def envelope(probabilities):
    """Enclose object Brier contrasts on ALL allowed inputs and both binary labels.

    Predictions are deployed binary64 values. Outward one-ULP arithmetic on
    each elementary operation bounds the real-valued factorisation
    (p-q)*(p+q-2*y). Group averaging preserves these same bounds, not width/m.
    Values do not depend on labels or input frequencies in an evaluation sample.
    """
    q=np.asarray(probabilities,dtype=float)
    if q.ndim!=2 or len(q)<1 or not np.isfinite(q).all() or q.min()<0 or q.max()>1:
        raise ValueError('finite complete-domain probabilities in [0,1] required')
    lo=np.full((q.shape[1],q.shape[1]),np.inf);hi=-lo
    # Input chunks avoid a large O x J x J workspace.
    for start in range(0,len(q),16):
        p=q[start:start+16,:,None];s=q[start:start+16,None,:]
        diff=p-s;dl=np.nextafter(diff,-np.inf);dh=np.nextafter(diff,np.inf)
        summ=p+s;sl=np.nextafter(summ,-np.inf);sh=np.nextafter(summ,np.inf)
        for y in (0.,1.):
            a=np.nextafter(sl-2*y,-np.inf);b=np.nextafter(sh-2*y,np.inf)
            v0=dl*a;v1=dl*b;v2=dh*a;v3=dh*b
            low=np.nextafter(np.minimum.reduce([v0,v1,v2,v3]),-np.inf).min(axis=0)
            high=np.nextafter(np.maximum.reduce([v0,v1,v2,v3]),np.inf).max(axis=0)
            lo=np.minimum(lo,low);hi=np.maximum(hi,high)
    # Exact identical functions have a provably zero contrast on this domain.
    equal=np.all(q[:,:,None]==q[:,None,:],axis=0)
    lo=np.maximum(lo,-1.);hi=np.minimum(hi,1.)
    lo[equal]=0.;hi[equal]=0.
    width=np.minimum(2.,np.nextafter(hi-lo,np.inf));width[equal]=0.
    if np.any(width<0) or not np.isfinite(width).all():raise ArithmeticError('range')
    return lo,hi,width

def eb(d,v,G,delta,width=None):
    if isinstance(G,bool) or not isinstance(G,(int,np.integer)) or G<2:raise ValueError('G')
    if not 0<delta<1:raise ValueError('delta')
    d=np.asarray(d);v=np.asarray(v)
    if d.shape!=v.shape or not np.isfinite(d).all() or not np.isfinite(v).all() or np.any(v<0):raise ValueError('moments')
    z=math.log(2/delta)
    if width is None:return d+np.sqrt(2*v*z/G)+14*z/(3*(G-1))
    w=np.asarray(width)
    if w.shape!=d.shape or not np.isfinite(w).all() or np.any(w<0) or np.any(w>2):raise ValueError('width')
    ans=d+np.sqrt(2*v*z/G)+7*w*z/(3*(G-1))
    # Zero width here only denotes identical probability functions and D=0.
    if np.any((w==0)&((abs(d)>1e-12)|(v>1e-12))):raise ValueError('zero width inconsistent with moments')
    ans=np.where(w==0,0.,ans)
    return ans

def evaluate(model,lib,fit,test,widths,alpha=.05,epsilon=.01):
    F.check_evaluation(model,lib,fit,test)
    answers={}
    for variant in ('generic','range'):
        answers[variant]={m:{'bounds':{k:[]for k in (1,2,3)},'choices':{k:[]for k in (1,2,3)}}for m in F.METHODS}
    for h in (0,1):
        loss=F.world_losses(model,lib,test,h)
        d,v,G=F.stable_pair_moments(loss)
        for variant in ('generic','range'):
            w=None if variant=='generic' else widths[h]
            pair=eb(d,v,G,alpha,w)
            for method in F.METHODS[:-1]:
                for k in (1,2,3):
                    p=fit.choices[method][k][h]
                    answers[variant][method]['bounds'][k].append(float(pair[p].max()))
                    answers[variant][method]['choices'][k].append(p)
            for k,ix in lib.classes.items():
                adj=eb(d,v,G,alpha/len(ix),w)[list(ix)].max(axis=1)
                pos=int(np.argmin(adj))
                answers[variant]['evaluation_class']['bounds'][k].append(float(adj[pos]))
                answers[variant]['evaluation_class']['choices'][k].append(int(ix[pos]))
    for variant in answers:
        for a in answers[variant].values():a['certified_level']=F.certify(a['bounds'],epsilon)
    return answers
