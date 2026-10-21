"""INF-001 review extension. Parent numeric sources are imported unchanged.

The finite-class certificate is conditional on a frozen training-only dictionary.
The low-level loss API cannot establish independence; the model/batch adapter
checks declared identities and the reference generator supplies independent worlds.
No file writes, network access or Git actions occur in this module.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import hashlib,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'parent/src'))
import inference_core as I


def seed_for(master:int,case:str,rep:int)->int:
    I.integer(master,'master',0);I.integer(rep,'rep',0)
    return int.from_bytes(hashlib.sha256(f'INF001-R1|{master}|{case}|{rep}'.encode()).digest()[:16],'little')


def _indices(indices:Sequence[int],size:int)->np.ndarray:
    if not isinstance(indices,(tuple,list,np.ndarray)) or len(indices)==0:raise ValueError('nonempty frozen class required')
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer))for x in indices):raise ValueError('integer member IDs required')
    a=np.asarray(indices,dtype=int)
    if a.ndim!=1 or len(set(a.tolist()))!=len(a) or np.any(a<0) or np.any(a>=size):raise ValueError('invalid/duplicate class members')
    return a


def group_loss_moments(losses:np.ndarray)->tuple[np.ndarray,np.ndarray,int]:
    """One row per independent world; weights within the world are fixed upstream."""
    x=np.asarray(losses,dtype=float)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1 or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
        raise ValueError('bounded finite world-by-predictor losses with >=2 worlds required')
    mean=x.mean(axis=0);center=x-mean
    covariance=(center.T@center)/(len(x)-1)
    return mean,covariance,len(x)


def pairwise_moments(mean:np.ndarray,cov:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    m=np.asarray(mean,dtype=float);c=np.asarray(cov,dtype=float)
    if m.ndim!=1 or len(m)<1 or c.shape!=(len(m),len(m)) or not np.isfinite(m).all() or not np.isfinite(c).all():raise ValueError('invalid mean/covariance shapes')
    if np.any(m<0) or np.any(m>1) or not np.allclose(c,c.T,rtol=0,atol=2e-12):raise ValueError('invalid bounded means/symmetry')
    d=np.diag(c)
    if np.any(d < -2e-12):raise ValueError('negative variance')
    variance=d[:,None]+d[None,:]-2*c
    if np.min(variance)<-2e-12:raise ArithmeticError('negative contrast variance beyond rounding allowance')
    # Algebraic covariance subtraction may leave sub-ulp negative zeros.
    variance=np.maximum(variance,0.)
    np.fill_diagonal(variance,0.)
    return m[:,None]-m[None,:],variance


def eb_pair_bounds(mean:np.ndarray,cov:np.ndarray,n:int,alpha:float)->np.ndarray:
    n=I.integer(n,'independent worlds',2);a=I.real(alpha,'alpha')
    if not 0<a<1:raise ValueError('alpha in (0,1) required')
    d,v=pairwise_moments(mean,cov);ell=math.log(2/a)
    return d+np.sqrt(2*v*ell/n)+14*ell/(3*(n-1))


def class_min_bound(losses:np.ndarray,indices:Sequence[int],alpha:float=.05)->dict:
    """Finite class-level upper bound. No test-selected dictionary is allowed.

    Union only over state members. Each row-wise maximum uses the scalar
    population-optimal history comparator lemma; it is NOT a pairwise band.
    """
    m,c,n=group_loss_moments(losses);ix=_indices(indices,len(m))
    a=I.real(alpha,'alpha')
    if not 0<a<1:raise ValueError('alpha in (0,1) required')
    per_member=eb_pair_bounds(m,c,n,a/len(ix))[ix,:].max(axis=1)
    pos=int(np.argmin(per_member))
    return {'bound':float(per_member[pos]),'selected_member':int(ix[pos]),
            'per_member_bounds':per_member.tolist(),'class_size':len(ix),
            'independent_worlds':n,'member_alpha':a/len(ix)}


def evaluate_moments(mean:np.ndarray,cov:np.ndarray,n:int,classes:dict,chosen:dict,alpha:float=.05,epsilon:float=.01)->dict:
    if set(classes)!={1,2,3} or set(chosen)!={1,2,3}:raise ValueError('registered levels are 1,2,3')
    eps=I.real(epsilon,'epsilon')
    if eps<=0:raise ValueError('epsilon positive required')
    ix={k:_indices(classes[k],len(mean))for k in (1,2,3)}
    if not set(ix[1])<=set(ix[2])<=set(ix[3]) or set(ix[3])!=set(range(len(mean))):raise ValueError('class nesting or benchmark missing')
    base=eb_pair_bounds(mean,cov,n,alpha)
    out={}
    for k in (1,2,3):
        if chosen[k] not in ix[k]:raise ValueError('chosen predictor not in its class')
        adjusted=eb_pair_bounds(mean,cov,n,alpha/len(ix[k]))[ix[k]].max(axis=1)
        pos=int(np.argmin(adjusted))
        out[k]={'fixed_bound':float(base[chosen[k]].max()),'class_bound':float(adjusted[pos]),
                'test_selected_bound_minimizer':int(ix[k][pos]),'class_size':len(ix[k])}
    return out


def evaluate_batch(model:I.FiniteModel,library:I.FrozenLibrary,test:I.WorldBatch,alpha=.05,epsilon=.01)->dict:
    """Validate evidence metadata; compute moments without large G x M x J tensors."""
    model.validate_batch(test,allow_test=True)
    if test.role!='test' or library.table_id!=model.table_id or set(test.world_ids)&set(library.training_ids):raise ValueError('test role/identity overlap')
    G,M=test.observed.shape;J=library.predictions.shape[-1]
    if G<2:raise ValueError('>=2 worlds required')
    if library.predictions.shape[:2]!=model.p.shape[:2] or not np.isfinite(library.predictions).all():raise ValueError('prediction shape/values')
    all_h=[]
    for h in (0,1):
        loss=np.empty((G,J))
        for start in range(0,G,1024):
            sl=slice(start,min(start+1024,G));o=test.observed[sl]
            loss[sl]=((library.predictions[h,o]-(test.responses[sl,:,h]>0)[...,None])**2).mean(axis=1)
        m,c,n=group_loss_moments(loss)
        all_h.append(evaluate_moments(m,c,n,library.classes,{k:library.chosen[k][h]for k in (1,2,3)},alpha,epsilon))
    result={}
    for name,field in [('historical_fixed','fixed_bound'),('reviewer_class_min','class_bound')]:
        bounds={str(k):[a[k][field]for a in all_h] for k in (1,2,3)}
        certified=None
        for k in (3,2,1):
            if max(bounds[str(k)])<epsilon:certified=k
            else:break
        result[name]={'bounds':bounds,'certified_level':certified}
    result['class_bound_minimizers']={str(k):[a[k]['test_selected_bound_minimizer']for a in all_h]for k in (1,2,3)}
    return result


def exact_group_loss_moments(model:I.FiniteModel,library:I.FrozenLibrary,truth:int,objects:int=4,h:int=0):
    """Population group moments for planning, not sample confidence bounds.

    Conditional on the shared offset, objects are iid. Integrate the finite
    observation/path/offset law and the Gaussian binary event probabilities.
    """
    from scipy.special import ndtr
    objects=I.integer(objects,'objects');truth=I.integer(truth,'truth',0)
    if truth>=len(model.params) or h not in (0,1):raise ValueError('unknown truth/horizon')
    q=library.predictions[h];l0=q*q;l1=(1-q)**2;po=model.obs_prob
    means=[];conditional_cov=[]
    for b in model.offsets:
        plat=ndtr((model.mu[h,:,truth]+b)/model.sigma)
        p=np.array([plat[ix].mean()for ix in model.latent])
        w1=po*p;w0=po*(1-p)
        mean=w1@l1+w0@l0
        second=l1.T@(l1*w1[:,None])+l0.T@(l0*w0[:,None])
        means.append(mean);conditional_cov.append(second-np.outer(mean,mean))
    means=np.asarray(means);mu=model.offset_probs@means;cov=np.zeros((len(mu),len(mu)))
    for w,m,c in zip(model.offset_probs,means,conditional_cov):cov+=w*(c/objects+np.outer(m-mu,m-mu))
    return mu,cov


def class_floor(class_size:int,epsilon:float=.01,alpha:float=.05)->int:
    q=I.integer(class_size,'class size');e=I.real(epsilon,'epsilon');a=I.real(alpha,'alpha')
    if e<=0 or not 0<a<1:raise ValueError('bad tolerance or alpha')
    # Strict criterion r_n < epsilon => n > 1 + c/epsilon.
    return math.floor(14*math.log(2*q/a)/(3*e))+2
