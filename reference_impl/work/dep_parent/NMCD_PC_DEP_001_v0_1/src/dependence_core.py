"""NMCD-PC-DEP-001: independent-realisation reference with known within-group dependence.

Pure local calculations. The original v0.3 core is vendored byte-for-byte.
The pooled-row result is a deliberately invalid control when c>0 and m>1.
Identifiers detect contradictions; the registered synthetic law, not IDs, supplies iid worlds.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping
import hashlib, math, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/v03/src'))
import nmcd_pc as P
import nmcd_review_extensions as E
D=E.canonicalise_reference().dictionary
LOSS=D.losses.reshape(D.size,8)
METHODS=('group_max_eb','group_max_hoeffding','pooled_row_max_eb')

@dataclass(frozen=True)
class GroupStats:
    G: int
    m: int
    total_counts: np.ndarray  # [horizon, four input cells, two outcomes]
    second_frequencies: np.ndarray  # [horizon,8,8], mean outer product of group frequency vectors


def copy_probability(x: object)->float:
    c=P.finite_real(x,'copy_probability')
    if not 0<=c<=1:raise ValueError('copy probability outside [0,1]')
    return c


def validate_provenance(train_ids,test_ids,*,law,weights='equal_world_equal_row',dictionary_exposure='pre_test',
                        future_feature=False,phase_shared=False,train_seed='train',test_seed='test'):
    if law!='iid_copy_mixture_realizations':raise ValueError('unregistered group law; no real cosmology admission')
    if weights!='equal_world_equal_row':raise ValueError('weighting changes estimand')
    if dictionary_exposure!='pre_test':raise ValueError('test-informed dictionary')
    if future_feature:raise ValueError('future feature leakage')
    if phase_shared:raise ValueError('shared phases violate independent worlds')
    if not train_ids or len(test_ids)<2:raise ValueError('nonempty training and >=2 test groups required')
    for ids in (train_ids,test_ids):
        if any(not isinstance(x,str) or not x for x in ids):raise ValueError('invalid group ID')
        if len(set(ids))!=len(ids):raise ValueError('duplicate independent-world ID')
    if set(train_ids)&set(test_ids):raise ValueError('world overlap between train and evaluation')
    if not train_seed or not test_seed or train_seed==test_seed:raise ValueError('stream role reused')


def group_stats(counts:np.ndarray)->GroupStats:
    """Derive G and m from actual equal-sized whole-world count arrays, never metadata n."""
    a=np.asarray(counts)
    if a.ndim!=4 or a.shape[0]!=2 or a.shape[2:]!=(4,2) or a.dtype.kind not in 'iu':
        raise ValueError('counts require shape (2,G,4,2), integer dtype')
    G=a.shape[1]
    if G<2 or (a<0).any():raise ValueError('>=2 nonnegative groups required')
    totals=a.sum(axis=(2,3),dtype=np.int64)
    m=int(totals[0,0])
    if not 1<=m<=2**20 or not np.all(totals==m):raise ValueError('empty/unequal/oversized groups')
    if not np.array_equal(a[0].sum(axis=2),a[1].sum(axis=2)):
        raise ValueError('horizons do not share group input counts')
    f=a.reshape(2,G,8).astype(float)/m
    ss=np.einsum('hgi,hgj->hij',f,f,optimize=True)/G
    return GroupStats(G,m,a.sum(axis=1,dtype=np.int64),ss)


def draw_group_counts(rng:np.random.Generator,G:int,m:int,theta:float,c:float)->np.ndarray:
    G=P.positive_int(G,'G',2);m=P.positive_int(m,'m');c=copy_probability(c)
    probs=np.stack([P.law(theta,t) for t in P.HORIZONS])
    if m>2**20:raise ValueError('m outside supported range')
    # One shared complete base episode per world, plus independently selected fresh episodes.
    if c==0:
        nc=rng.multinomial(m,[.25]*4,size=G)
        out=[]
        for h in range(2):
            y=rng.binomial(nc,probs[h]);out.append(np.stack((nc-y,y),axis=2))
        return np.stack(out)
    base=rng.integers(0,4,size=G)
    base_y=np.stack([rng.binomial(1,probs[h,base]) for h in range(2)])
    copied=np.full(G,m,dtype=np.int64) if c==1 else rng.binomial(m,c,size=G)
    fresh=rng.multinomial(m-copied,[.25]*4)
    nc=fresh.copy();nc[np.arange(G),base]+=copied
    out=[]
    for h in range(2):
        y=rng.binomial(fresh,probs[h])
        y[np.arange(G),base]+=copied*base_y[h]
        out.append(np.stack((nc-y,y),axis=2))
    return np.stack(out)


def draw_stats(rng,G,m,theta,c)->GroupStats:
    """Optimized exact-duplicate / one-row case; same law, documented call sequence."""
    G=P.positive_int(G,'G',2);m=P.positive_int(m,'m');c=copy_probability(c)
    if c==1 or m==1:
        raw=P.draw_counts(rng,G,theta)
        total=np.stack([raw[t] for t in P.HORIZONS])*m
        ss=np.stack([np.diag(raw[t].reshape(8)/G) for t in P.HORIZONS])
        return GroupStats(G,m,total,ss)
    return group_stats(draw_group_counts(rng,G,m,theta,c))


def fit_groups(s:GroupStats)->dict[int,dict[int,int]]:
    return {t:P.fit(s.total_counts[t-1],(2,1),D) for t in P.HORIZONS}


def pair_moments(s:GroupStats,index:int,h:int):
    P.check_member(index,2,D)
    if h not in (0,1):raise ValueError('unknown horizon index')
    mu_freq=s.total_counts[h].reshape(8)/(s.G*s.m)
    diff=LOSS[index][None,:]-LOSS
    mu=diff@mu_freq
    cov=s.second_frequencies[h]-np.outer(mu_freq,mu_freq)
    var=np.einsum('qi,ij,qj->q',diff,cov,diff,optimize=True)*s.G/(s.G-1)
    row_second=(diff*diff)@mu_freq
    row_var=(row_second-mu*mu)*(s.G*s.m)/(s.G*s.m-1)
    if min(var.min(),row_var.min()) < -2e-12:raise ArithmeticError('variance cancellation beyond tolerance')
    return mu,np.maximum(var,0),np.maximum(row_var,0)


def eb_from_moments(mu,var,n,alpha=.05):
    n=P.positive_int(n,'n',2);alpha=P.probability(alpha,'alpha')
    mu=np.asarray(mu);var=np.asarray(var)
    if mu.ndim!=1 or mu.shape!=var.shape or not len(mu):raise ValueError('empty/mismatched moments')
    if not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var<0):raise ValueError('invalid moments')
    z=math.log(2/alpha)
    out=float(np.max(mu+np.sqrt(2*var*z/n)+14*z/(3*(n-1))))
    return float(np.nextafter(out,math.inf))


def all_bounds(s:GroupStats,fitted:Mapping,alpha=.05):
    alpha=P.probability(alpha,'alpha')
    if set(fitted)!={1,2}:raise ValueError('missing fitted horizon')
    bounds={name:{} for name in METHODS}
    for t in P.HORIZONS:
        if set(fitted[t])!={1,2}:raise ValueError('missing fitted dimension')
        for name in METHODS:bounds[name][t]={}
        for k in (2,1):
            ix=fitted[t][k];P.check_member(ix,k,D)
            mu,var,rv=pair_moments(s,ix,t-1)
            bounds['group_max_eb'][t][k]=eb_from_moments(mu,var,s.G,alpha)
            bounds['group_max_hoeffding'][t][k]=float(np.nextafter(mu.max()+math.sqrt(2*math.log(1/alpha)/s.G),math.inf))
            bounds['pooled_row_max_eb'][t][k]=eb_from_moments(mu,rv,s.G*s.m,alpha)
    return bounds


def certify_group_counts(train,test,train_ids,test_ids,**metadata):
    validate_provenance(train_ids,test_ids,**metadata)
    tr=group_stats(train);te=group_stats(test)
    if (len(train_ids),len(test_ids))!=(tr.G,te.G):raise ValueError('group IDs/count arrays mismatch')
    if tr.m!=te.m:raise ValueError('train/test configurations differ')
    fitted=fit_groups(tr);bounds=all_bounds(te,fitted)
    b=bounds['group_max_eb']
    return {'status':'LOCAL_SYNTHETIC_CLASS_RELATIVE_CERTIFICATE','bound':b,
            'certified_dimension':P.decide(b),'independent_units':te.G,'rows':te.G*te.m,
            'not_simultaneous_pairwise_band':True,'cosmology_admitted':False}


def row_control_status(c,m):
    c=copy_probability(c);m=P.positive_int(m,'m')
    return 'IID_ROW_REFERENCE' if c==0 or m==1 else 'INVALID_ROW_IID_CONTROL'


def stream_seed(case:dict,rep:int,role:str,master:int)->int:
    if role not in ('train','test'):raise ValueError('unknown stream role')
    rep=P.positive_int(rep,'replicate',0);master=P.positive_int(master,'master')
    c=float(case['copy_probability'])
    # At c=1, deliberately pair different replication factors over identical independent episodes.
    m='paired_duplicates' if c==1 else case['rows_per_group']
    text=f"NMCD-PC-DEP-001|{master}|{case['theta']}|{case['copy_probability']}|{case['groups_per_split']}|{m}|{rep}|{role}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16],'little')


def oracle_facts(theta:float,c:float,G:int,m:int)->dict:
    copy_probability(c);P.positive_int(G,'G',2);P.positive_int(m,'m')
    risks=P.exact_risks(theta,1,d=D)
    ix0=int(D.classes[0][0]);ixstar=min(range(D.size),key=lambda j:risks[j])
    expected=Fraction(0);second=Fraction(0)
    th=Fraction(str(theta))
    for cell,u in enumerate((-1,-1,1,1)):
        prob=Fraction(1,2)+th*u/2
        for y in (0,1):
            pp=(prob if y else 1-prob)/4
            dd=Fraction(float(D.losses[ix0,cell,y]-D.losses[ixstar,cell,y]))
            expected+=pp*dd;second+=pp*dd*dd
    variance=second-expected*expected
    factor=Fraction(str(c))**2+(1-Fraction(str(c))**2)/m
    return {'state_index':ix0,'comparator_index':ixstar,'mean_fraction':str(expected),'mean':float(expected),
            'row_variance_fraction':str(variance),'row_variance':float(variance),
            'group_mean_variance_fraction':str(variance*factor),'group_mean_variance':float(variance*factor),
            'grand_mean_variance_fraction':str(variance*factor/G),'grand_mean_variance':float(variance*factor/G),
            'intraclass_correlation':c*c,'design_effect':1+(m-1)*c*c,
            'variance_equivalent_count_diagnostic_only':G*m/(1+(m-1)*c*c)}


def oracle_contrast_mean(s:GroupStats,facts:dict)->float:
    df=LOSS[facts['state_index']]-LOSS[facts['comparator_index']]
    return float(df@(s.total_counts[0].reshape(8)/(s.G*s.m)))
