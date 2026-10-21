"""NMCD-PC v0.2: auditable synthetic finite-dictionary calibration.

Independent implementation; does not import the reviewer's numerical routines.
The public certificate interface validates counts, finite losses, membership and levels.
Count arrays do not establish independence: certified use additionally requires the
provenance contract. The present caller generates iid episodes by construction.

Methods: legacy two-stage; reviewer simultaneous-union H/EB; pointwise-maximum H/EB.
The last two bound ONE scalar gap; they do NOT provide simultaneous pairwise coverage.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Mapping, Sequence
import numpy as np

METHODS = ('two_stage', 'union_hoeffding', 'union_eb', 'max_hoeffding', 'max_eb')
HORIZONS = (1, 2)
RTAU = {1: 1., 2: .5}
UCELL = np.array([-1., -1., 1., 1.])
XCELL = np.array([-1., 1., -1., 1.])
EPS = 1/16


def positive_int(x: object, name: str, minimum: int=1) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(x)


def finite_real(x: object, name: str) -> float:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(float(x)):
        raise ValueError(f'{name} must be finite and real')
    return float(x)


def probability(x: object, name: str) -> float:
    x = finite_real(x, name)
    if not 0 < x < 1: raise ValueError(f'{name} must be in (0,1)')
    return x


@dataclass(frozen=True)
class Dictionary:
    grid: tuple[float, ...]
    predictions: np.ndarray
    losses: np.ndarray
    classes: Mapping[int, np.ndarray]
    fingerprint: str

    @property
    def size(self) -> int: return len(self.predictions)


def make_dictionary(step: int=16) -> Dictionary:
    step = positive_int(step, 'step', 2)
    if step % 2: raise ValueError('step must be even')
    grid = tuple(float(j/step) for j in range(-step//2, step//2+1))
    coef = np.array([(a,b) for a in grid for b in grid])
    q = np.clip(.5 + coef[:,0,None]*XCELL + coef[:,1,None]*UCELL, 0, 1)
    loss = np.stack([q*q, (1-q)**2], axis=2)
    cls = {0: np.flatnonzero((coef[:,0]==0)&(coef[:,1]==0)),
           1: np.flatnonzero(coef[:,1]==0), 2: np.arange(len(q))}
    for arr in (q, loss, *cls.values()): arr.setflags(write=False)
    fp = hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()
    return Dictionary(grid,q,loss,cls,fp)


D = make_dictionary()


def validate_dictionary(d: Dictionary) -> None:
    q, loss = np.asarray(d.predictions), np.asarray(d.losses)
    if q.ndim!=2 or q.shape[1]!=4 or not len(q): raise ValueError('dictionary shape')
    if loss.shape!=(len(q),4,2) or not np.isfinite(loss).all(): raise ValueError('loss array shape/finiteness')
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)): raise ValueError('prediction range')
    if np.any((loss<0)|(loss>1)): raise ValueError('loss outside [0,1]')
    if not np.array_equal(loss,np.stack([q*q,(1-q)**2],axis=2)): raise ValueError('loss/predictor mismatch')
    previous: set[int] = set()
    for k in (0,1,2):
        a=np.asarray(d.classes[k])
        if a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a): raise ValueError('class indices')
        cur=set(map(int,a))
        if len(cur)!=len(a) or min(cur)<0 or max(cur)>=len(q): raise ValueError('invalid/duplicate class index')
        if not previous.issubset(cur): raise ValueError('non-nested classes')
        previous=cur
    q0=q[d.classes[0]]; q1=q[d.classes[1]]
    if np.any(q0!=.5): raise ValueError('constant anchor must predict one half')
    if not np.array_equal(q1[:,0],q1[:,2]) or not np.array_equal(q1[:,1],q1[:,3]):
        raise ValueError('history leaked into current-state class')
    if hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()!=d.fingerprint: raise ValueError('dictionary fingerprint')


def law(theta: float, tau: int, kappa: float=0.) -> np.ndarray:
    theta=finite_real(theta,'theta'); kappa=finite_real(kappa,'kappa')
    if isinstance(tau,bool) or tau not in HORIZONS: raise ValueError('unregistered horizon')
    if abs(theta)+abs(kappa)>1+1e-15: raise ValueError('law needs |theta|+|kappa| <= 1; no probability clipping')
    p=.5+.5*(kappa*XCELL+theta*RTAU[tau]*UCELL)
    if np.any((p<0)|(p>1)): raise ValueError('invalid Bernoulli probability')
    return p


def validate_counts(counts: np.ndarray, expected_n: int | None=None) -> int:
    c=np.asarray(counts)
    if c.shape!=(4,2) or c.dtype.kind not in 'iu': raise ValueError('counts must be a 4x2 integer array')
    if (c<0).any(): raise ValueError('negative count')
    n=sum(int(v) for v in c.flat)
    if not 1<=n<=2**40: raise ValueError('count total outside supported range')
    if expected_n is not None and n!=positive_int(expected_n,'expected_n'):
        raise ValueError('counts do not match declared sample size')
    return n


def validate_shared_horizons(batch: Mapping[int,np.ndarray]) -> int:
    if set(batch)!=set(HORIZONS): raise ValueError('missing/extra horizons')
    n=validate_counts(batch[1]); validate_counts(batch[2],n)
    if not np.array_equal(batch[1].sum(axis=1),batch[2].sum(axis=1)):
        raise ValueError('horizons do not share the same episode-cell counts')
    return n


def draw_counts(rng: np.random.Generator, n: int, theta: float, kappa: float=0.) -> dict[int,np.ndarray]:
    n=positive_int(n,'n')
    probs={t:law(theta,t,kappa) for t in HORIZONS}
    nc=rng.multinomial(n,[.25]*4)
    out={}
    for t in HORIZONS:
        ones=rng.binomial(nc,probs[t]);out[t]=np.stack([nc-ones,ones],axis=1)
    return out


def mean_losses(counts: np.ndarray, d: Dictionary=D) -> np.ndarray:
    n=validate_counts(counts)
    return np.sum(d.losses*counts[None,:,:],axis=(1,2))/n


def fit(train: np.ndarray, dimensions: Sequence[int]=(2,1), d: Dictionary=D) -> dict[int,int]:
    r=mean_losses(train,d); ans={}
    for k in dimensions:
        if isinstance(k,bool) or k not in d.classes: raise ValueError('unknown dimension')
        indices=d.classes[k]; ans[k]=int(indices[np.argmin(r[indices])])
    return ans


def check_member(index: int, k: int, d: Dictionary=D) -> None:
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral) or not 0<=index<d.size:
        raise ValueError('predictor index out of range')
    if isinstance(k,bool) or k not in d.classes or index not in d.classes[k]:
        raise ValueError('predictor not in declared state class')


def diff_moments(counts: np.ndarray,index: int,d: Dictionary=D) -> tuple[np.ndarray,np.ndarray]:
    n=validate_counts(counts)
    if n<2: raise ValueError('sample variance requires n>=2')
    check_member(index,2,d)
    v=d.losses[index][None,:,:]-d.losses
    mu=np.sum(v*counts[None,:,:],axis=(1,2))/n
    second=np.sum(v*v*counts[None,:,:],axis=(1,2))/n
    vv=(second-mu*mu)*n/(n-1)
    if np.min(vv)<-1e-12: raise ArithmeticError('negative variance beyond floating tolerance')
    return mu,np.maximum(vv,0.)


def certificate(counts: np.ndarray,index: int,k: int,method: str,alpha: float=.05,
                d: Dictionary=D, train: np.ndarray | None=None) -> float:
    """Return a one-sided upper bound; input/assumption validation is mandatory."""
    validate_dictionary(d);check_member(index,k,d);alpha=probability(alpha,'alpha')
    n=validate_counts(counts)
    if n<2: raise ValueError('certification requires n>=2')
    if method not in METHODS: raise ValueError('unknown certificate')
    r=mean_losses(counts,d)
    if method=='two_stage':
        if train is None: raise ValueError('two-stage method needs actual training counts')
        ntr=validate_counts(train); ih=fit(train,(2,),d)[2]
        beta=.4*alpha; gamma=.6*alpha
        eta=2*math.sqrt(math.log(2*d.size/beta)/(2*ntr))
        v=float(r[index]-r[ih]+eta+math.sqrt(2*math.log(1/gamma)/n))
    elif method in ('union_hoeffding','max_hoeffding'):
        factors=d.size if method=='union_hoeffding' else 1
        v=float(r[index]-r.min()+math.sqrt(2*math.log(factors/alpha)/n))
    else:
        mu,var=diff_moments(counts,index,d)
        factors=d.size if method=='union_eb' else 1
        z=math.log(2*factors/alpha)
        v=float(np.max(mu+np.sqrt(2*var*z/n)+14*z/(3*(n-1))))
    if not math.isfinite(v): raise ArithmeticError('nonfinite certificate')
    return float(np.nextafter(v,math.inf)) # one conservative rounding step; not arbitrary clipping


def fast_all(train: np.ndarray,test: np.ndarray,dimensions=(2,1),d: Dictionary=D,alpha=.05):
    """Internal fast path after dictionary/design and generated counts have been checked.
    Uses the same formulas as the validated public API; equality tested in fixtures.
    """
    ntr=int(train.sum());n=int(test.sum()); rt=np.sum(d.losses*train[None],axis=(1,2))/ntr
    re=np.sum(d.losses*test[None],axis=(1,2))/n
    ih=int(np.argmin(rt)); fitted={k:int(d.classes[k][np.argmin(rt[d.classes[k]])]) for k in dimensions}
    eta=2*math.sqrt(math.log(2*d.size/(.4*alpha))/(2*ntr))
    out={m:{} for m in METHODS}
    for k,index in fitted.items():
        out['two_stage'][k]=float(re[index]-re[ih]+eta+math.sqrt(2*math.log(1/(.6*alpha))/n))
        mu,var=diff_moments(test,index,d)
        for prefix,factor in [('union',d.size),('max',1)]:
            out[prefix+'_hoeffding'][k]=float(mu.max()+math.sqrt(2*math.log(factor/alpha)/n))
            z=math.log(2*factor/alpha)
            out[prefix+'_eb'][k]=float(np.max(mu+np.sqrt(2*var*z/n)+14*z/(3*(n-1))))
        for m in METHODS: out[m][k]=float(np.nextafter(out[m][k],math.inf))
    return out,fitted


def decide(bounds: Mapping[int,Mapping[int,float]],dimensions=(2,1),horizons=HORIZONS,epsilon=EPS):
    epsilon=finite_real(epsilon,'epsilon')
    if epsilon<=0: raise ValueError('positive adequacy margin required')
    if tuple(dimensions) not in ((2,1),(2,1,0)): raise ValueError('dimension sequence not registered')
    if tuple(horizons)!=HORIZONS or set(bounds)!=set(horizons): raise ValueError('nonempty registered horizon set required')
    for t in horizons:
        if set(bounds[t])!=set(dimensions): raise ValueError('missing/extra dimension bound')
        for v in bounds[t].values():
            if math.isnan(float(v)) or float(v)==-math.inf: raise ValueError('invalid bound')
    ans=None
    for k in dimensions:
        if all(bounds[t][k]<epsilon for t in horizons): ans=k
        else: break
    return ans  # None = no certificate, not proof of infinite complexity


def exact_risks(theta: float,tau: int,kappa: float=0.,d: Dictionary=D) -> list[Fraction]:
    """Exact rational enumeration of all four input cells and both binary outcomes."""
    law(theta,tau,kappa)
    th=Fraction(str(theta));kap=Fraction(str(kappa));rr=Fraction(1,tau)
    vals=[]
    for q in d.predictions:
        risk=Fraction(0)
        for j,(u,x) in enumerate(zip(UCELL,XCELL)):
            p=Fraction(1,2)+(kap*int(x)+th*rr*int(u))/2
            qq=Fraction(float(q[j]))
            risk+=(p*(1-qq)**2+(1-p)*qq**2)/4
        vals.append(risk)
    return vals


def exact_truth(theta: float,kappa: float=0.,dimensions=(2,1),d: Dictionary=D,epsilon=EPS) -> dict:
    risks={t:exact_risks(theta,t,kappa,d) for t in HORIZONS}
    gaps={t:{k:min(risks[t][i] for i in d.classes[k])-min(risks[t]) for k in dimensions} for t in HORIZONS}
    eps=Fraction(str(epsilon)); ascending=sorted(dimensions)
    weak=next((k for k in ascending if max(gaps[t][k] for t in HORIZONS)<=eps),None)
    strict=next((k for k in ascending if max(gaps[t][k] for t in HORIZONS)<eps),None)
    return {'gaps':{t:{k:float(v) for k,v in g.items()} for t,g in gaps.items()},
            'gap_fractions':{t:{k:str(v) for k,v in g.items()} for t,g in gaps.items()},
            'C_weak':weak,'C_strict':strict}


def validate_provenance(train_ids: Sequence[str],test_ids: Sequence[str],train_seed: str,test_seed: str,
                        available_times: Mapping[str,float],origin=0.,group_law='iid_generated_episodes',
                        weighting='equal',dictionary_exposure='pre_test',prior_seed_keys=()) -> None:
    if not train_ids or not test_ids: raise ValueError('empty partition')
    if any(not isinstance(i,str) or not i for i in (*train_ids,*test_ids)): raise ValueError('invalid group identity')
    if len(set(train_ids))!=len(train_ids) or len(set(test_ids))!=len(test_ids): raise ValueError('duplicated group id')
    if set(train_ids)&set(test_ids): raise ValueError('train/test overlap')
    if not train_seed or not test_seed or train_seed==test_seed: raise ValueError('seed-role reuse')
    if train_seed in prior_seed_keys or test_seed in prior_seed_keys: raise ValueError('fresh evidence reuses seed key')
    if not available_times: raise ValueError('no feature provenance')
    if any(not math.isfinite(float(v)) or v>origin for v in available_times.values()): raise ValueError('future/untraceable feature')
    if group_law!='iid_generated_episodes': raise ValueError('iid theorem unavailable for this group law')
    if weighting!='equal': raise ValueError('unequal weights unsupported by reference certificate')
    if dictionary_exposure!='pre_test': raise ValueError('test-selected comparator dictionary')


def rng_for(seed:int,label:str,role:str):
    seed=positive_int(seed,'seed',0)
    key=f'{seed}|{label}|{role}'
    digest=hashlib.sha256(key.encode()).digest()
    seq=np.random.SeedSequence([seed]+list(np.frombuffer(digest,dtype='<u4')))
    return np.random.Generator(np.random.PCG64(seq)),hashlib.sha256(key.encode()).hexdigest()


def wilson(success:int,total:int,z:float=1.959963984540054) -> list[float]:
    n=positive_int(total,'total');p=success/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-rad),min(1.,mid+rad)]


def zero_event_upper(n:int,alpha=.05) -> float:
    return -math.expm1(math.log(probability(alpha,'alpha'))/positive_int(n,'n'))
