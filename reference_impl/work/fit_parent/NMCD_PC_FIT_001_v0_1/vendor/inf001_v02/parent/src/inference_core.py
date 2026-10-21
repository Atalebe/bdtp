"""Likelihood-controlled accretion-state reference, NMCD-PC-INF-001.

Finite latent spaces are summed exactly; Gaussian densities/CDFs use float64.
This is a synthetic conditional-response model, not a halo evolution simulator.
No file, network, fitting-on-test, or repository side effects occur in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import hashlib
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/v03/src'))
sys.path.insert(0,str(ROOT/'vendor/dep001/src'))
from dependence_core import eb_from_moments as inherited_group_eb
import numpy as np
from scipy.special import logsumexp, ndtr

TIMES = np.array([0.15, 0.27, 0.43, 0.62, 0.80, 1.00])
GAMMAS = np.array([0.5, 1.2, 1.8, 2.6, 3.0, 4.5])
AMPLITUDES = np.array([-16., -8., 0., 8., 16.])
CURVATURES = np.array([0., 0.3])
OFFSETS = np.array([-0.12, 0., 0.12])
OFFSET_PROBS = np.array([0.25, 0.5, 0.25])
SIGMA = 0.08
DECAYS = np.array([1., 0.5])
EPSILON = 0.01  # dimensionless Brier gap, a new synthetic task's registered margin
ALPHA = 0.05


def real(x, name):
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
        raise ValueError(f'{name}: finite real scalar required')
    y=float(x)
    if not math.isfinite(y): raise ValueError(f'{name}: nonfinite')
    return y


def integer(x, name, minimum=1):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<minimum:
        raise ValueError(f'{name}: integer >= {minimum} required')
    return int(x)


def probabilities(x, name):
    p=np.asarray(x,dtype=float)
    if p.ndim!=1 or len(p)==0 or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1.,rtol=0,atol=2e-12):
        raise ValueError(f'{name}: normalized nonnegative finite probability vector required')
    return p


def validate_path(times, increments, origin=None):
    t=np.asarray(times,dtype=float); u=np.asarray(increments,dtype=float)
    if t.ndim!=1 or u.shape!=t.shape or len(t)==0 or not np.isfinite(t).all() or not np.isfinite(u).all():
        raise ValueError('nonempty finite one-dimensional equal-length path required')
    if np.any(np.diff(t)<=0):raise ValueError('strictly increasing admission times required')
    if origin is not None and t[-1]>real(origin,'origin'):raise ValueError('future increment not available at origin')
    return t,u


def direct_coordinate(times, increments, gamma, origin=None):
    t,u=validate_path(times,increments,origin)
    g=real(gamma,'gamma')
    if g<0:raise ValueError('gamma must be nonnegative')
    # shift by latest time for numerical stability; reference origin cancels
    w=np.exp(-g*(t[-1]-t))
    return float(w@u/w.sum())


def recursive_bank(times, increments, gammas=GAMMAS, origin=None):
    t,u=validate_path(times,increments,origin)
    gs=np.asarray(gammas,dtype=float)
    if gs.ndim!=1 or len(gs)==0 or not np.isfinite(gs).all() or np.any(gs<0):
        raise ValueError('invalid fixed gamma bank')
    num=np.zeros(len(gs));den=np.zeros(len(gs));last=t[0]
    for tj,uj in zip(t,u):
        r=np.exp(-gs*(tj-last));num=r*num+uj;den=r*den+1;last=tj
    return num,den,num/den


def normalize_log_weights(log_w, prior):
    p=probabilities(prior,'prior'); ll=np.asarray(log_w,dtype=float)
    if ll.shape!=p.shape or np.isnan(ll).any() or np.isposinf(ll).any():raise ValueError('invalid log likelihood')
    lp=np.full_like(p,-np.inf); pos=p>0;lp[pos]=np.log(p[pos])
    z=logsumexp(lp+ll)
    if not np.isfinite(z):raise ValueError('likelihood has no prior support')
    w=np.exp(lp+ll-z)
    return w/w.sum()


@dataclass(frozen=True)
class WorldBatch:
    """Only observed ids and response labels enter inference; hidden support/offset is audit truth."""
    observed: np.ndarray  # [world, object]
    responses: np.ndarray  # [world, object, horizon]
    world_ids: tuple[str,...]
    role: str
    table_id: str
    hidden_support: np.ndarray
    hidden_offset: np.ndarray


class FiniteModel:
    """A declared 60-label gamma/amplitude/curvature model, with independent worlds.

    Latent accretion paths use 64 equally likely bit patterns at each of three
    current strata. Multiple labels may imply the same distribution. They are
    NOT deduplicated in the posterior: their prior mass must remain intact.
    """
    def __init__(self, design='excited', observation='complete', sigma=SIGMA,
                 offsets=OFFSETS, offset_probs=OFFSET_PROBS):
        if design not in ('excited','summary_only','constant_gamma'):raise ValueError('unknown history design')
        if observation not in ('complete','mass_truncated'):raise ValueError('unknown observation law')
        self.design=design;self.observation=observation
        self.sigma=real(sigma,'sigma')
        if self.sigma<=0:raise ValueError('sigma must be positive')
        self.offsets=np.asarray(offsets,dtype=float);self.offset_probs=probabilities(offset_probs,'offset prior')
        if self.offsets.shape!=self.offset_probs.shape or not np.isfinite(self.offsets).all():raise ValueError('invalid offsets')
        self.params=np.array(list(product(CURVATURES,AMPLITUDES,GAMMAS)))
        self.prior=np.full(len(self.params),1/len(self.params))
        self.c=np.repeat(np.array([-1.,0.,1.]),64)
        bits=np.tile(np.array(list(product([0,1],repeat=6))), (3,1))
        self.u=.02+.06*bits
        self.phi=self.c**2-2/3
        if design=='summary_only':self.u=np.repeat((.05+.0375*self.phi)[:,None],6,axis=1)
        if design=='constant_gamma':self.u=np.repeat(self.u[:,0,None],6,axis=1)
        self.mask=np.ones_like(self.u,dtype=bool)
        if observation=='mass_truncated':
            self.mask[self.c==-1,:3]=False
            self.mask[self.c==0,:1]=False
        keys=[]
        for i in range(len(self.c)):
            keys.append((int(self.c[i]),tuple(self.mask[i].tolist()),tuple(np.round(self.u[i,self.mask[i]],12))))
        mapping={};obs=[];latent=[]
        for i,key in enumerate(keys):
            if key not in mapping:mapping[key]=len(mapping);latent.append([])
            j=mapping[key];obs.append(j);latent[j].append(i)
        self.support_to_obs=np.asarray(obs)
        self.latent=[np.asarray(x,dtype=int) for x in latent]
        self.obs_prob=np.array([len(x)/len(self.c) for x in latent])
        reps=np.array([x[0] for x in latent]);self.representatives=reps
        self.obs_c=self.c[reps]
        self.obs_mask=self.mask[reps]
        self.obs_u=np.where(self.obs_mask,self.u[reps],np.nan)
        self.summary_keys=[]
        for i in reps:
            self.summary_keys.append((int(self.c[i]),int(self.mask[i].sum()),
                round(float(self.u[i,self.mask[i]].sum()),12),round(float(self.u[i,-1]),12)))
        self.summary_groups=[]
        for key in dict.fromkeys(self.summary_keys):self.summary_groups.append(np.array([i for i,k in enumerate(self.summary_keys) if k==key]))
        weights=np.exp(-GAMMAS[:,None]*(TIMES[-1]-TIMES)[None,:]);weights/=weights.sum(axis=1,keepdims=True)
        self.m=self.u@weights.T-.05
        # Ensure true constant-increment equivalence is respected at machine precision.
        # The mathematical m is exactly u_1-.05 for these designs; do not create
        # spurious differences between gamma labels through summation order.
        if design!='excited':self.m=np.repeat((self.u[:,0]-.05)[:,None],len(GAMMAS),axis=1)
        gindex=np.tile(np.arange(len(GAMMAS)),len(CURVATURES)*len(AMPLITUDES))
        self.mu=np.stack([self.phi[:,None]*self.params[:,0]+r*self.m[:,gindex]*self.params[:,1] for r in DECAYS])
        p_lat=np.stack([sum(w*ndtr((mu+b)/self.sigma) for w,b in zip(self.offset_probs,self.offsets)) for mu in self.mu])
        self.p=np.stack([np.stack([pp[ix].mean(axis=0) for ix in self.latent]) for pp in p_lat]) # H,O,P
        self.p_summary=np.empty_like(self.p)
        for ix in self.summary_groups:
            ww=self.obs_prob[ix]/self.obs_prob[ix].sum()
            self.p_summary[:,ix,:]=np.sum(self.p[:,ix,:]*ww[None,:,None],axis=1)[:,None,:]
        self.table_id=hashlib.sha256((design+'|'+observation+'|'+repr(self.sigma)+'|'+repr(self.offsets.tolist())+'|'+repr(self.offset_probs.tolist())).encode()).hexdigest()
        # Uniform-prior mixture over compatible, unobserved path fragments.
        # Collapse repeated latent paths, not parameter labels; preserve prior mass.
        latent_unique=[]
        for ix in self.latent:
            _, first, counts=np.unique(self.u[ix],axis=0,return_index=True,return_counts=True)
            latent_unique.append((ix[first],counts/counts.sum()))
        nlat=max(len(x[0]) for x in latent_unique)
        self.latent_index=np.zeros((len(self.latent),nlat),dtype=int)
        self.latent_log_weight=np.full((len(self.latent),nlat),-np.inf)
        for o,(ix,pr) in enumerate(latent_unique):
            self.latent_index[o,:len(ix)]=ix
            self.latent_log_weight[o,:len(ix)]=np.log(pr)

    def parameter_index(self, curvature, amplitude, gamma):
        target=np.array([curvature,amplitude,gamma],dtype=float)
        hits=np.flatnonzero(np.all(self.params==target,axis=1))
        if len(hits)!=1:raise ValueError('truth not on registered parameter grid')
        return int(hits[0])

    def draw(self,rng,worlds,objects,theta_index,role,namespace):
        worlds=integer(worlds,'worlds');objects=integer(objects,'objects')
        theta_index=integer(theta_index,'theta_index',0)
        if theta_index>=len(self.params) or role not in ('training','test','inverse','fixture'):raise ValueError('invalid draw role or parameter')
        h=rng.integers(0,len(self.c),size=(worlds,objects))
        bi=rng.choice(len(self.offsets),size=worlds,p=self.offset_probs)
        mu=np.moveaxis(self.mu[:,h,theta_index],0,-1)
        y=mu+self.offsets[bi,None,None]+rng.normal(0,self.sigma,size=(worlds,objects,2))
        return WorldBatch(self.support_to_obs[h],y,tuple(f'{namespace}:world:{i}' for i in range(worlds)),role,self.table_id,h,self.offsets[bi])

    def validate_batch(self,batch,allow_test=False):
        if not isinstance(batch,WorldBatch) or batch.table_id!=self.table_id:raise ValueError('observation-law identity mismatch')
        o=np.asarray(batch.observed);y=np.asarray(batch.responses)
        if o.ndim!=2 or o.dtype.kind not in 'iu' or min(o.shape)<1 or o.shape+(2,)!=y.shape:raise ValueError('invalid batch shapes')
        if not np.isfinite(y).all() or np.any(o<0) or np.any(o>=len(self.latent)):raise ValueError('invalid observations')
        if len(batch.world_ids)!=o.shape[0] or len(set(batch.world_ids))!=o.shape[0] or any(not x for x in batch.world_ids):raise ValueError('invalid/duplicate world ids')
        if batch.role not in ('training','inverse','fixture','test'):raise ValueError('unregistered role')
        if batch.role=='test' and not allow_test:raise ValueError('future evaluation outcomes cannot inform the predictor')
        return o,y

    def log_likelihood(self,batch,mode='shared',allow_test=False):
        """p(training horizon-1 responses | permitted histories, theta).

        Integrate the shared world offset AFTER multiplying per-object likelihoods.
        Integrate unavailable path entries separately per object inside that product.
        """
        o,y=self.validate_batch(batch,allow_test=allow_test)
        if mode not in ('shared','iid_wrong','drop_missing_wrong'):raise ValueError('unknown likelihood mode')
        # Chunk worlds to cap memory for incomplete-path mixtures.
        total=np.zeros(len(self.params))
        for start in range(0,len(o),16):
            oo=o[start:start+16]; yy=y[start:start+16,:,0]
            if mode=='drop_missing_wrong':
                mm=[]
                for j in oo.reshape(-1):
                    keep=self.obs_mask[j]
                    mm.append([direct_coordinate(TIMES[keep],self.obs_u[j,keep],g)-.05 for g in GAMMAS])
                mm=np.asarray(mm).reshape(oo.shape+(len(GAMMAS),))
                gi=np.tile(np.arange(len(GAMMAS)),len(CURVATURES)*len(AMPLITUDES))
                mean=(self.obs_c[oo]**2-2/3)[...,None]*self.params[:,0]+mm[...,gi]*self.params[:,1]
                mean=mean[...,None,:]
                lw=np.zeros(oo.shape+(1,))
            else:
                mean=self.mu[0,self.latent_index[oo],:]
                lw=self.latent_log_weight[oo]
            r=(yy[...,None,None,None]-mean[...,None]-self.offsets[None,None,None,None,:])/self.sigma
            logdens=-.5*r*r-math.log(self.sigma*math.sqrt(2*math.pi))
            object_ll=logsumexp(logdens+lw[...,None,None],axis=2) # G,M,P,B
            if mode=='iid_wrong':
                group_ll=logsumexp(object_ll+np.log(self.offset_probs),axis=-1).sum(axis=1)
            else:
                group_ll=logsumexp(object_ll.sum(axis=1)+np.log(self.offset_probs),axis=-1)
            total+=group_ll.sum(axis=0)
        return total

    def posterior(self,batch,mode='shared',prior=None):
        pp=self.prior if prior is None else probabilities(prior,'prior')
        return normalize_log_weights(self.log_likelihood(batch,mode),pp)

    def forecast(self,posterior,representation='history'):
        pp=probabilities(posterior,'posterior')
        if pp.shape!=self.prior.shape:raise ValueError('posterior grid mismatch')
        if representation not in ('history','summary'):raise ValueError('unknown information representation')
        return (self.p if representation=='history' else self.p_summary)@pp

    def exact_brier_risk(self,predictions,truth):
        q=np.asarray(predictions,dtype=float)
        if q.shape[:2]!=self.p.shape[:2] or q.ndim not in (2,3) or not np.isfinite(q).all() or np.any(q<0) or np.any(q>1):raise ValueError('invalid event predictions')
        pt=self.p[:,:,truth]
        if q.ndim==3:pt=pt[...,None]
        loss=pt*(1-q)**2+(1-pt)*q*q
        return np.sum(loss*self.obs_prob[None,:,None] if q.ndim==3 else loss*self.obs_prob[None,:],axis=1)

    def retained_state_forecast(self,posterior):
        """Independent feature route: per-gamma accumulator for every compatible path.

        With censoring this is a finite mixture over reconstructed paths, NOT a
        claim that only two numbers suffice for arbitrary missing observations.
        """
        pp=probabilities(posterior,'posterior'); m=[]
        for u in self.u:m.append(recursive_bank(TIMES,u)[2]-.05)
        m=np.asarray(m)
        gi=np.tile(np.arange(len(GAMMAS)),len(CURVATURES)*len(AMPLITUDES))
        out=[]
        for r in DECAYS:
            mu=self.phi[:,None]*self.params[:,0]+r*m[:,gi]*self.params[:,1]
            probabilities_lat=sum(w*ndtr((mu+b)/self.sigma) for w,b in zip(self.offset_probs,self.offsets))
            p_obs=np.stack([probabilities_lat[ix].mean(0) for ix in self.latent])
            out.append(p_obs@pp)
        return np.stack(out)


def posterior_mass_summary(model,post,truth):
    post=probabilities(post,'posterior');theta=model.params[truth]
    gamma=np.array([post[model.params[:,2]==g].sum() for g in GAMMAS])
    amp=np.array([post[model.params[:,1]==a].sum() for a in AMPLITUDES])
    ent=lambda p:float(-np.sum(p[p>0]*np.log(p[p>0])))
    return {'gamma_probabilities':gamma.tolist(),'amplitude_probabilities':amp.tolist(),
        'gamma_entropy_fraction':ent(gamma)/math.log(len(GAMMAS)),
        'amplitude_entropy_fraction':ent(amp)/math.log(len(AMPLITUDES)),
        'true_label_probability':float(post[truth]),'true_gamma_probability':float(gamma[np.where(GAMMAS==theta[2])[0][0]])}


def randomized_pit(values,weights,index,v):
    val=np.asarray(values);w=probabilities(weights,'weights');v=real(v,'tie uniform')
    index=integer(index,'index',0)
    if val.shape!=w.shape or index>=len(w) or not np.isfinite(val).all() or not 0<=v<=1:raise ValueError('invalid PIT inputs')
    x=val[index]
    z=float(w[val<x].sum()+v*w[val==x].sum())
    # At an exact 0/1 endpoint, normalized float sums can miss by a few ulps.
    # Correct only this representational error; do not clip genuine bad inputs.
    tol=8*np.finfo(float).eps
    if z < -tol or z > 1+tol:raise ArithmeticError('PIT outside [0,1] beyond rounding tolerance')
    return min(1.,max(0.,z))


def ks_uniform(x):
    a=np.sort(np.asarray(x,dtype=float));n=len(a)
    if n<1 or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('invalid rank sample')
    return float(max(np.max(np.arange(1,n+1)/n-a),np.max(a-np.arange(n)/n)))


def seed_for(master,case,rep,role):
    return int.from_bytes(hashlib.sha256(f'NMCD-PC-INF-001|{master}|{case}|{rep}|{role}'.encode()).digest()[:16],'little')


def validate_split(model,train,test):
    model.validate_batch(train);model.validate_batch(test,allow_test=True)
    if train.role!='training' or test.role!='test':raise ValueError('split roles not training/test')
    if set(train.world_ids)&set(test.world_ids):raise ValueError('training/test world overlap')
    if len(test.world_ids)<2:raise ValueError('certificate needs >=2 independent worlds')


@dataclass(frozen=True)
class FrozenLibrary:
    predictions:np.ndarray # [H,O,J]
    classes:dict[int,tuple[int,...]]
    chosen:dict[int,tuple[int,int]]  # two horizon-specific fitted indices for each level
    names:tuple[str,...]
    posterior:np.ndarray
    table_id:str
    training_ids:tuple[str,...]


def make_library(model,train):
    """Finite dictionary fixed using training only; no certification outcomes enter."""
    o,y=model.validate_batch(train)
    if train.role!='training':raise ValueError('library requires training role')
    prior=model.prior;post=model.posterior(train)
    # Weak comparator: affine functions of the current stratum only.
    affine=[];names=[]
    for intercept,slope in product([-.15,0.,.15],[-.3,0.,.3]):
        mu=intercept+slope*model.obs_c
        p=sum(w*ndtr((mu+b)/model.sigma) for b,w in zip(model.offsets,model.offset_probs))
        affine.append(np.stack([p,p]));names.append(f'affine:{intercept}:{slope}')
    q=np.stack(affine,axis=-1);lev1=list(range(q.shape[-1]))
    summary_start=q.shape[-1]
    q=np.concatenate([q,model.p_summary,model.forecast(post,'summary')[...,None]],axis=2)
    names += [f'summary_grid:{i}' for i in range(len(prior))]+['summary_posterior']
    lev2=list(range(q.shape[-1]))
    q=np.concatenate([q,model.p,model.forecast(post)[...,None]],axis=2)
    names += [f'history_grid:{i}' for i in range(len(prior))]+['history_posterior']
    lev3=list(range(q.shape[-1]))
    classes={1:tuple(lev1),2:tuple(lev2),3:tuple(lev3)}
    chosen={}
    for k,ixs in classes.items():
        chosen[k]=tuple(int(ixs[np.argmin(np.mean((q[h,o,:][:,:,ixs]-(y[:,:,h]>0)[...,None])**2,axis=(0,1)))]) for h in range(2))
    q.setflags(write=False);post.setflags(write=False)
    return FrozenLibrary(q,classes,chosen,tuple(names),post,model.table_id,train.world_ids)


def scalar_eb(mu,var,n,alpha=ALPHA):
    """Same unaltered scalar-max formula as DEP-001, range D in [-1,1]."""
    n=integer(n,'independent worlds',2);a=real(alpha,'alpha')
    mm=np.asarray(mu,dtype=float);v=np.asarray(var,dtype=float)
    if not 0<a<1 or mm.shape!=v.shape or mm.ndim!=1 or len(mm)==0 or not np.isfinite(mm).all() or not np.isfinite(v).all() or np.any(v<0):raise ValueError('invalid certificate moments')
    return inherited_group_eb(mm,v,n,a)


def evaluate_library(model,library,test,truth,alpha=ALPHA,epsilon=EPSILON):
    model.validate_batch(test,allow_test=True)
    if test.role!='test' or library.table_id!=model.table_id or set(test.world_ids)&set(library.training_ids):raise ValueError('test identity/exposure mismatch')
    eps=real(epsilon,'epsilon')
    if eps<=0:raise ValueError('epsilon positive required')
    q=library.predictions; o=test.observed;G,M=o.shape
    if G<2:raise ValueError('>=2 independent worlds required')
    risk=model.exact_brier_risk(q,truth)
    oracle=np.min(risk,axis=1)
    gaps={k:(np.min(risk[:,list(ix)],axis=1)-oracle) for k,ix in library.classes.items()}
    bounds={k:[] for k in (1,2,3)};fitted_risk={k:[]for k in (1,2,3)}
    lossmean={}
    for h in range(2):
        losses=np.mean((q[h,o,:]-(test.responses[:,:,h]>0)[...,None])**2,axis=1) # G,J
        lossmean[h]=losses.mean(axis=0)
        for k in (1,2,3):
            ix=library.chosen[k][h]
            diff=losses[:,ix,None]-losses
            bounds[k].append(scalar_eb(diff.mean(axis=0),diff.var(axis=0,ddof=1),G,alpha))
            fitted_risk[k].append(float(risk[h,ix]))
    certified=None
    for k in (3,2,1):
        if max(bounds[k])<eps:certified=k
        else:break
    ctrue=next((k for k in (1,2,3)if max(gaps[k])<eps),None)
    return {'independent_worlds':G,'rows':G*M,'certified_level':certified,'true_strict_level':ctrue,
        'false_adequacy':certified is not None and max(gaps[certified])>=eps,
        'exact_class_gaps':{str(k):v.tolist()for k,v in gaps.items()},
        'bounds':{str(k):v for k,v in bounds.items()},'exact_fitted_risks':{str(k):v for k,v in fitted_risk.items()},
        'oracle_risks':oracle.tolist(), 'not_simultaneous_pairwise_band':True,'intrinsic_dimension_claim':False}
