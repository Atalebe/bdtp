"""Training-only selection versus finite-class evaluation selection.

Model, candidate construction and certificates are inherited, not altered.
The new joint likelihood integrates each hidden path across BOTH endpoints,
then objects conditional on a shared world offset. No oracle labels enter fit.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import hashlib, math, sys
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/inf001_v02/src'))
import review_extension as E
I=E.I
METHODS=('erm','variance_penalized','posterior_risk','evaluation_class')


def seed_for(master, case, rep, role):
    I.integer(master,'master seed',0); I.integer(rep,'replicate',0)
    if role not in ('training_pool','test_pool','resource_training','resource_test','fixture'):
        raise ValueError('unregistered stream role')
    return int.from_bytes(hashlib.sha256(f'FIT001|{master}|{case}|{rep}|{role}'.encode()).digest()[:16],'little')


def prefix(batch,n):
    n=I.integer(n,'prefix size')
    if n>len(batch.world_ids):raise ValueError('prefix exceeds allocated worlds')
    return I.WorldBatch(batch.observed[:n], batch.responses[:n], batch.world_ids[:n],
                        batch.role,batch.table_id,batch.hidden_support[:n],batch.hidden_offset[:n])


def joint_log_likelihood(model, train, horizons=(0,1), shared=True):
    """Exact finite joint likelihood for registered observed training responses.

    The missing path and nuisance offset are shared across horizons. In contrast
    to adding two separately marginalized likelihoods, this preserves that law.
    shared=False is a deterministic misspecification fixture only, not an arm.
    """
    o,y=model.validate_batch(train)
    if train.role!='training':raise ValueError('training data only')
    if tuple(horizons) not in ((0,),(0,1)):raise ValueError('unregistered fitting horizons')
    total=np.zeros(len(model.params))
    for start in range(0,len(o),16):
        oo=o[start:start+16]; yy=y[start:start+16]
        ix=model.latent_index[oo]
        ld=np.zeros(ix.shape+(len(model.params),len(model.offsets)))
        for h in horizons:
            mean=model.mu[h,ix,:]
            resid=(yy[:,:,h,None,None,None]-mean[...,None]-model.offsets)/model.sigma
            ld+=-.5*resid*resid-math.log(model.sigma*math.sqrt(2*math.pi))
        object_ll=logsumexp(ld+model.latent_log_weight[oo][...,None,None],axis=2)
        if shared:
            world_ll=logsumexp(object_ll.sum(axis=1)+np.log(model.offset_probs),axis=-1)
        else:
            world_ll=logsumexp(object_ll+np.log(model.offset_probs),axis=-1).sum(axis=1)
        total+=world_ll.sum(axis=0)
    if not np.isfinite(total).all():raise ArithmeticError('nonfinite joint likelihood')
    return total


def array_digest(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()


def library_digest(lib):return array_digest(lib.predictions)


def world_losses(model,lib,batch,h):
    model.validate_batch(batch,allow_test=True)
    if lib.table_id!=model.table_id or h not in (0,1):raise ValueError('table/horizon mismatch')
    G,M=batch.observed.shape; J=lib.predictions.shape[-1]
    out=np.empty((G,J))
    for first in range(0,G,512):
        sl=slice(first,min(first+512,G))
        out[sl]=((lib.predictions[h,batch.observed[sl]]-(batch.responses[sl,:,h]>0)[...,None])**2).mean(axis=1)
    if not np.isfinite(out).all() or out.min()<0 or out.max()>1:raise ArithmeticError('invalid group losses')
    return out


def expected_brier(model,lib,posterior,h):
    post=I.probabilities(posterior,'posterior')
    if post.shape!=model.prior.shape or h not in (0,1):raise ValueError('posterior/horizon mismatch')
    p=model.p[h]@post; q=lib.predictions[h]
    return model.obs_prob@(q*q-2*p[:,None]*q+p[:,None])


@dataclass(frozen=True)
class FitRecord:
    choices: dict
    posterior_joint: np.ndarray
    training_ids: tuple
    training_digest: str
    library_digest: str
    scores: dict


def choose_in_class(scores,classes):
    out={}
    for k,ix in classes.items():
        ind=E._indices(ix,len(scores))
        if not np.isfinite(scores[ind]).all():raise ArithmeticError('nonfinite fitting score')
        out[k]=int(ind[np.argmin(scores[ind])])
    return out


def fit_rules(model,lib,train):
    """Each selector is a function of the SAME admitted training data only.

    No truth index, hidden paths/offsets, test outcome, or population oracle is
    supplied. The posterior-risk selector assumes the inherited model law.
    """
    model.validate_batch(train)
    if train.role!='training' or lib.training_ids!=train.world_ids:raise ValueError('training/library mismatch')
    if set(lib.classes)!={1,2,3} or [len(lib.classes[k])for k in (1,2,3)]!=[9,70,131]:raise ValueError('frozen family changed')
    if not set(lib.classes[1])<=set(lib.classes[2])<=set(lib.classes[3]):raise ValueError('nesting')
    if len(train.world_ids)<2:raise ValueError('two training worlds required for variance')
    post=I.normalize_log_weights(joint_log_likelihood(model,train),model.prior)
    choices={name:{k:[]for k in (1,2,3)}for name in METHODS[:-1]}
    scores={'erm':[],'variance_penalized':[],'posterior_risk':[]}
    for h in (0,1):
        loss=world_losses(model,lib,train,h)
        mean=loss.mean(0); var=loss.var(0,ddof=1)
        pm=expected_brier(model,lib,post,h)
        for k,ix in lib.classes.items():
            ind=np.asarray(ix);vp=mean+np.sqrt(2*var*math.log(2*len(ix)/.05)/len(train.world_ids))
            choices['erm'][k].append(int(ind[np.argmin(mean[ind])]))
            choices['variance_penalized'][k].append(int(ind[np.argmin(vp[ind])]))
            choices['posterior_risk'][k].append(int(ind[np.argmin(pm[ind])]))
        scores['erm'].append(mean)
        scores['variance_penalized'].append({k:mean+np.sqrt(2*var*math.log(2*len(ix)/.05)/len(train.world_ids)) for k,ix in lib.classes.items()})
        scores['posterior_risk'].append(pm)
    for k in (1,2,3):
        if tuple(choices['erm'][k])!=lib.chosen[k]:
            # Predicted means agree up to reduction order; a genuine tie can
            # choose another equal-scoring member. Do not silently retune ties.
            for h in (0,1):
                if abs(scores['erm'][h][choices['erm'][k][h]]-scores['erm'][h][lib.chosen[k][h]])>2e-12:
                    raise ArithmeticError('inherited ERM score mismatch')
        # Preserve inherited deterministic tie order for historical baseline.
        choices['erm'][k]=list(lib.chosen[k])
    digest=array_digest(train.observed)+'|'+array_digest(train.responses)
    return FitRecord(choices,post,train.world_ids,digest,library_digest(lib),scores)


def certify(bounds,epsilon=.01):
    if set(bounds)!={1,2,3}:raise ValueError('all fixed levels required')
    eps=I.real(epsilon,'epsilon')
    if eps<=0:raise ValueError('positive margin')
    for k in (1,2,3):
        a=np.asarray(bounds[k],dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError('two finite horizons required')
    chosen=None
    for k in (3,2,1):
        if max(bounds[k])<eps:chosen=k
        else:break
    return chosen


def check_evaluation(model,lib,fit,test):
    model.validate_batch(test,allow_test=True)
    if test.role!='test' or lib.table_id!=model.table_id:raise ValueError('test identity/role')
    if set(test.world_ids)&set(fit.training_ids):raise ValueError('train/test overlap')
    if fit.library_digest!=library_digest(lib):raise ValueError('library changed after fitting')
    if len(test.world_ids)<2:raise ValueError('at least two test worlds')
    for choices in fit.choices.values():
        for k,ix in choices.items():
            if len(ix)!=2 or any(p not in lib.classes[k]for p in ix):raise ValueError('fitted member outside class')


def stable_pair_moments(loss):
    """Recompute near-zero contrast moments directly, without covariance cancellation.

    This changes no statistical bound. The 1e-12 switch is numerical only: it
    chooses an algebraically identical variance calculation, not zero clipping.
    Diagonal and reverse-pair identities are exact. Q and predictor classes stay
    unchanged; this is NOT test-driven dictionary deduplication.
    """
    mean,cov,G=E.group_loss_moments(loss)
    d,v=E.pairwise_moments(mean,cov)
    aa,bb=np.where(np.triu(v<1e-12,1))
    for first in range(0,len(aa),32):
        a=aa[first:first+32];b=bb[first:first+32]
        diff=loss[:,a]-loss[:,b]
        dm=diff.mean(axis=0);vv=diff.var(axis=0,ddof=1)
        d[a,b]=dm;d[b,a]=-dm;v[a,b]=vv;v[b,a]=vv
    return d,v,G


def bounds_from_pair_moments(d,v,G,alpha):
    G=I.integer(G,'world count',2);alpha=I.real(alpha,'alpha')
    if not 0<alpha<1:raise ValueError('alpha')
    z=math.log(2/alpha)
    return d+np.sqrt(2*v*z/G)+14*z/(3*(G-1))


def evaluation_from_cov(model,lib,fit,test,alpha=.05,epsilon=.01):
    """Shared sufficient-statistic implementation for the numerical comparison.

    Isolated resource audit below separately times efficient fixed-member versus
    full covariance calculations. Shared campaign timing is NOT per-arm cost.
    """
    check_evaluation(model,lib,fit,test)
    bounds={method:{k:[]for k in (1,2,3)}for method in METHODS}
    choices={method:{k:list(fit.choices[method][k])for k in (1,2,3)}for method in METHODS[:-1]}
    choices['evaluation_class']={k:[]for k in (1,2,3)}
    for h in (0,1):
        loss=world_losses(model,lib,test,h)
        d,v,G=stable_pair_moments(loss)
        pair=bounds_from_pair_moments(d,v,G,alpha)
        for method in METHODS[:-1]:
            for k in (1,2,3):bounds[method][k].append(float(pair[fit.choices[method][k][h]].max()))
        for k,ix in lib.classes.items():
            adj=bounds_from_pair_moments(d,v,G,alpha/len(ix))[list(ix)].max(1)
            pos=int(np.argmin(adj));bounds['evaluation_class'][k].append(float(adj[pos]))
            choices['evaluation_class'][k].append(int(ix[pos]))
    return {method:{'bounds':bounds[method],'choices':choices[method],
                    'certified_level':certify(bounds[method],epsilon)}for method in METHODS}


def evaluate_isolated(model,lib,fit,test,method,alpha=.05,epsilon=.01):
    """No unnecessary full covariance for a training-selected predictor."""
    if method not in METHODS:raise ValueError('unknown method')
    check_evaluation(model,lib,fit,test)
    bounds={k:[]for k in (1,2,3)};chosen={k:[]for k in (1,2,3)}
    for h in (0,1):
        loss=world_losses(model,lib,test,h)
        if method=='evaluation_class':
            d,v,G=stable_pair_moments(loss)
            for k,ix in lib.classes.items():
                adj=bounds_from_pair_moments(d,v,G,alpha/len(ix))[list(ix)].max(1)
                pos=int(np.argmin(adj));bounds[k].append(float(adj[pos]));chosen[k].append(int(ix[pos]))
        else:
            G=len(loss);mean=loss.mean(0);center=loss-mean;diag=np.sum(center*center,axis=0)/(G-1)
            unique=sorted(set(fit.choices[method][k][h]for k in (1,2,3)))
            cross=center[:,unique].T@center/(G-1)
            for k in (1,2,3):
                p=fit.choices[method][k][h];v=diag[p]+diag-2*cross[unique.index(p)]
                if v.min() < -2e-12:raise ArithmeticError('invalid contrast variance')
                v=np.maximum(v,0.);v[p]=0.
                dm=mean[p]-mean
                near=np.flatnonzero(v<1e-12)
                if len(near):
                    diff=loss[:,p,None]-loss[:,near]
                    dm[near]=diff.mean(0);v[near]=diff.var(0,ddof=1)
                bounds[k].append(float(I.scalar_eb(dm,v,G,alpha)));chosen[k].append(p)
    return {'bounds':bounds,'choices':chosen,'certified_level':certify(bounds,epsilon)}


def diagnostic_truth(model,lib,truth,answers,epsilon=.01):
    """Post-decision scoring only. This routine is NEVER called by a fitter."""
    risks=model.exact_brier_risk(lib.predictions,truth);best=risks.min(axis=1)
    minima={k:risks[:,list(ix)].min(axis=1)for k,ix in lib.classes.items()}
    gaps={k:minima[k]-best for k in (1,2,3)}
    out={}
    for name,res in answers.items():
        c=res['certified_level'];picked={k:np.array([risks[h,res['choices'][k][h]]for h in (0,1)])for k in (1,2,3)}
        out[name]={'selected_gaps':{k:(picked[k]-best).tolist()for k in (1,2,3)},
                   'fitting_regret':{k:(picked[k]-minima[k]).tolist()for k in (1,2,3)},
                   'false_class_certification':bool(c is not None and max(gaps[c])>=epsilon),
                   'certifies_summary':bool(c is not None and c<=2)}
    return {'class_gaps':{k:gaps[k].tolist()for k in (1,2,3)},'benchmark_risk':best.tolist(),'methods':out}


def resource_payload(model,lib,T,G,objects=4):
    """Numerical payload accounting, not Python process RSS or a bit lower bound."""
    O=len(model.latent);J=lib.predictions.shape[-1]
    return {'dtype':'binary64 probabilities','shared_candidate_table_bytes':lib.predictions.nbytes,
        'shared_original_posterior_bytes':lib.posterior.nbytes,'joint_fitting_posterior_bytes':60*8,
        'coefficient_metadata_excluded':True,'raw_numeric_training_bytes':T*objects*(8+2*8),
        'raw_numeric_test_bytes':G*objects*(8+2*8),'world_id_python_overhead_excluded':True,
        'world_loss_matrix_bytes_per_horizon':G*J*8,'full_covariance_bytes':J*J*8,
        'selected_cross_moments_bytes_upper':3*J*8,
        'candidate_object_loss_values_training':2*T*objects*J,
        'candidate_object_loss_values_test':2*G*objects*J,
        'full_covariance_products':2*G*J*J,
        'selected_cross_products_upper':2*G*3*J,
        'shared_predictive_table_shape':[2,O,J],
        'not_equal_flops':'Counts distinguish operations; Gaussian likelihood terms, lookups and products are not equated.',
        'per_history_min_numeric_payload':'six increments plus clock/mask/current context; no new compression claim'}
