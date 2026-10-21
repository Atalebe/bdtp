"""Offline campaign adapters. Original numerical sources remain byte-identical.
Full campaigns replay frozen experiments; they do not create independent evidence.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FP = Path('work/fit_parent/NMCD_PC_FIT_001_v0_1')
IP = FP/'vendor/inf001_v02/parent'
DP = Path('work/dep_parent/NMCD_PC_DEP_001_v0_1')
CAMPAIGNS = ('fit-pair','fit-generic','class-selection','inference-predictive','inference-inverse','inference-numerical','dependence-paper','dependence-original','power','planning')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def verify_imports():
    checked = 0
    for label in ('001','002'):
        record = json.loads((ROOT/f'provenance/IMPORT_{label}.json').read_text())
        for item in record['files']:
            p = ROOT/item['destination']
            if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Source/reference identity failed: '+str(p))
            checked += 1
    return checked

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def prepare_runtime(out):
    target = out/'runtime/reference_impl'
    for p in sorted((ROOT/'reference_impl').rglob('*')):
        if not p.is_file() or p.suffix not in ('.py','.json','.txt'):
            continue
        rel = p.relative_to(ROOT/'reference_impl')
        if 'evidence' in rel.parts and str(rel) != str(FP/'evidence/run_002/campaign/summary.json'):
            continue
        if '__pycache__' in rel.parts:
            continue
        q = target/rel; q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p,q)
    (target/'evidence').mkdir(parents=True, exist_ok=True)
    return target

def compare(actual, reference, patterns):
    paths = sorted({p for pattern in patterns for p in actual.glob(pattern) if p.is_file()})
    expected = sorted({p.relative_to(reference).as_posix() for pattern in patterns for p in reference.glob(pattern) if p.is_file()})
    found = [p.relative_to(actual).as_posix() for p in paths]
    if not paths or found != expected:
        raise ValueError('Missing/extra reproduction outputs')
    rows = []
    for p in paths:
        relative = p.relative_to(actual); ref = reference/relative
        got = hashlib.sha256(p.read_bytes()).hexdigest(); want = hashlib.sha256(ref.read_bytes()).hexdigest()
        rows.append({'file':str(relative),'expected':want,'actual':got,'byte_identical':got==want})
    return {'files':rows,'all_byte_identical':all(x['byte_identical'] for x in rows)}

def execute(name, out):
    runtime = prepare_runtime(out)
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    result = out/'results'
    if name in ('fit-pair','fit-generic','class-selection','inference-predictive','inference-inverse','inference-numerical','dependence-original'):
        if name == 'fit-pair':
            cmd = [sys.executable,str(runtime/'analysis_src/run_matched.py'),str(result)]
            ref = ROOT/'reference_impl/evidence/matched_run01'; patterns=['cells/*.json','fits.json','summary.json']
        elif name == 'fit-generic':
            cmd = [sys.executable,str(runtime/FP/'scripts/run_experiment.py'),'--stage','campaign','--output',str(result)]
            ref = ROOT/'reference_impl'/FP/'evidence/run_002/campaign'; patterns=['summary.json']
        elif name == 'class-selection':
            cmd=[sys.executable,str(runtime/FP/'vendor/inf001_v02/scripts/run_extensions.py'),'--stage','classes','--output',str(result)]
            ref=ROOT/'reference/inference/classes'; patterns=['*.json']
        elif name.startswith('inference-'):
            stage = name.split('-',1)[1]
            cmd = [sys.executable,str(runtime/IP/'scripts/run_experiments.py'),'--stage',stage,'--output',str(result)]
            ref = ROOT/'reference/inference'/stage; patterns=['*.json']
        else:
            cmd = [sys.executable,str(runtime/DP/'scripts/calibrate_dependence.py'),'--output',str(result)]
            ref = ROOT/'reference/dependence_original'; patterns=['all_cases.json','summary.json']
        subprocess.run(cmd,cwd=runtime,env=env,check=True)
    else:
        sys.path.insert(0,str(runtime/'analysis_src'))
        result.mkdir()
        if name in ('power','dependence-paper'):
            module = load_module(runtime/'analysis_src/additional_checks.py','bdtp_additional')
            if name == 'power':
                module.exact_power(result); ref=ROOT/'reference_impl/evidence/power_final'; patterns=['power.json']
            else:
                module.dep_study(result); ref=ROOT/'reference_impl/evidence/additional01/dep'; patterns=['*.json']
        else:
            module=load_module(ROOT/'experiments/original/planning_audit.py','bdtp_planning')
            write(result/'planning_audit.json',module.audit())
            ref=ROOT/'reference'; patterns=['planning_audit.json']
    verification=compare(result,ref,patterns)
    write(out/'REPLAY_COMPARISON.json',verification)
    if not verification['all_byte_identical']:
        raise RuntimeError('Replay differs from stored references; no completion. Inspect comparison without overwriting references.')
    return len(verification['files'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',choices=CAMPAIGNS,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inject-failure',action='store_true')
    a=ap.parse_args(); out=a.output.expanduser().resolve()
    if out==ROOT or ROOT in out.parents and ROOT/'.local_runs' not in out.parents:
        raise ValueError('Inside the checkout, outputs must be beneath .local_runs/')
    out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    try:
        if a.inject_failure:
            raise RuntimeError('Deliberate failure before campaign or completion')
        count=verify_imports()
        import numpy,scipy,mpmath
        if (numpy.__version__,scipy.__version__,mpmath.__version__)!=('2.3.5','1.17.0','1.3.0'):
            raise RuntimeError('Use the pinned numerical environment')
        write(out/'ENVIRONMENT.json',{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'mpmath':mpmath.__version__,'campaign':a.campaign,'source_files_verified':count,'purpose':'Historical replay; no new independent confirmation'})
        n=execute(a.campaign,out)
        verify_imports()
        write(out/'COMPLETE.json',{'status':'PASS','campaign':a.campaign,'byte_identical_reference_files':n,'seconds':time.monotonic()-start,'scientific_rule_changes':0,'new_independent_evidence':False,'git_actions':False,'formal_logbook_written':False})
        print('CAMPAIGN_REPLAY_PASS',a.campaign,n,flush=True)
    except BaseException:
        (out/'FAILURE.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__':
    main()
