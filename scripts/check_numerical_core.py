"""Verify imported bytes and run selected deterministic suites. No network or Git."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,platform
ROOT=Path(__file__).resolve().parents[1]
record=json.loads((ROOT/'provenance/IMPORT_001.json').read_text())
for item in record['files']:
    p=ROOT/item['destination']
    if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise RuntimeError('Imported source identity failed: '+item['destination'])
import numpy,scipy,mpmath
for module,version in [(numpy,'2.3.5'),(scipy,'1.17.0'),(mpmath,'1.3.0')]:
    if module.__version__!=version:
        raise RuntimeError(f'{module.__name__} {module.__version__}; expected {version}')
print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'mpmath':mpmath.__version__,'source_files_verified':len(record['files']),'scope':'deterministic tests, not Monte Carlo reproduction'},indent=2),flush=True)
env=os.environ.copy()
env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
paths=[str(p) for p in (ROOT/'reference_impl').rglob('src')]+[str(ROOT/'reference_impl/analysis_src')]
env['PYTHONPATH']=os.pathsep.join(paths)
tests=[item['destination'] for item in record['files'] if Path(item['destination']).name.startswith('test_')]
if len(tests)!=7:raise RuntimeError('Unexpected selected-suite inventory')
for relative in tests:
    print('SUITE '+relative,flush=True)
    subprocess.run([sys.executable,str(ROOT/relative)],cwd=ROOT,env=env,check=True)
print('ORIGINAL_CORE_CHECKS_PASSED',flush=True)
