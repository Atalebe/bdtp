"""Prospective v03 helpers. Never modify the v02 dictionary, plan or evidence.

Exact canonicalisation is valid on the ENTIRE declared four-state input domain.
It is not empirical deduplication on an observed sample and must not be used as such.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math, platform, sys
import numpy as np
import nmcd_pc as P

@dataclass(frozen=True)
class Canonicalisation:
    dictionary: P.Dictionary
    old_to_new: tuple[int, ...]
    representative_indices: tuple[int, ...]
    support: str = 'complete_binary_U_X_domain'

def canonicalise_reference(d: P.Dictionary=P.D, *, support: str='complete_binary_U_X_domain') -> Canonicalisation:
    if support != 'complete_binary_U_X_domain':
        raise ValueError('Deduplication needs equality on the complete declared input domain, not observed rows')
    P.validate_dictionary(d)
    keys={}; reps=[]; mapping=[]
    for i,row in enumerate(d.predictions):
        # All four points constitute the whole domain. Values are dyadic in this reference.
        key=tuple(float(x) for x in row)
        if key not in keys:
            keys[key]=len(reps);reps.append(i)
        mapping.append(keys[key])
    q=d.predictions[reps].copy();loss=d.losses[reps].copy()
    cls={k:np.array(list(dict.fromkeys(mapping[int(i)] for i in ix)),dtype=np.int64)
         for k,ix in d.classes.items()}
    for a in (q,loss,*cls.values()):a.setflags(write=False)
    fp=hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()
    cd=P.Dictionary(d.grid,q,loss,cls,fp);P.validate_dictionary(cd)
    return Canonicalisation(cd,tuple(mapping),tuple(reps))

def planning_floors(j: int, alpha: float=.05, epsilon: float=1/16, loss_bound: float=1.) -> dict:
    j=P.positive_int(j,'j'); alpha=P.probability(alpha,'alpha')
    eps=P.finite_real(epsilon,'epsilon');L=P.finite_real(loss_bound,'loss_bound')
    if eps<=0 or L<=0:raise ValueError('positive margin and loss range required')
    h=2*L*L*math.log(j/alpha)/eps**2
    s=2*L*L*math.log(1/alpha)/eps**2
    t=2*L*L*(math.sqrt(math.log(2*j/(.4*alpha)))+math.sqrt(math.log(1/(.6*alpha))))**2/eps**2
    return {'J':j,'union_H_threshold':h,'union_H_minimum':math.floor(h)+1,
            'scalar_H_threshold':s,'scalar_H_minimum':math.floor(s)+1,
            'two_stage_zero_difference_threshold':t,'two_stage_zero_difference_minimum':math.floor(t)+1,
            'qualification':'H floors require a state predictor included in dictionary; two-stage is an equal-split zero-difference planning floor. Counts mean independent groups. EB has a data-dependent variance term.'}

def environment_record() -> dict:
    return {'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,
            'platform':platform.platform(),'machine':platform.machine(),'libc':platform.libc_ver(),
            'bit_generator':'PCG64','byte_replay_numpy':'2.3.5',
            'policy':'same code, arguments/call order, data, build, machine and full environment still matter; a version pin alone is not a universal byte-identity guarantee'}

def require_replay_numpy(version: str | None=None) -> None:
    actual=np.__version__ if version is None else version
    if actual!='2.3.5':
        raise ValueError(f'Historical byte replay requires NumPy 2.3.5; got {actual}. Other versions require a separately labelled compatibility run.')
