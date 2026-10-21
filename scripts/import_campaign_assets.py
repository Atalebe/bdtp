"""Selective, byte-preserving import of original campaign sources and results.
No source execution, numerical changes, Git operations, or automatic publication.
"""
from pathlib import Path, PurePosixPath
import argparse, ast, hashlib, io, json, stat, zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = '49fb52caf751ff14b2a8095486b36da4456c490ef8dee2f538a175cc1a308174'
ASSET_HASH = 'ede617b3e2b44418846bbf9db7ecae02bdf01a71c457f46c2771bfc6df40e620'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def open_verified(data, label, chain):
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive member')
    for info in z.infolist():
        p = PurePosixPath(info.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('Unsafe archive member')
    root = names[0].split('/')[0] + '/'
    manifest = root + 'SHA256SUMS.txt'
    checked = set()
    for line in z.read(manifest).decode().splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        name = name.lstrip('* ')
        member = root + name
        if member in checked or digest(z.read(member)) != sha:
            raise ValueError('Manifest mismatch: ' + name)
        checked.add(member)
    files = {i.filename for i in z.infolist() if not i.is_dir()}
    if checked | {manifest} != files:
        raise ValueError('Manifest does not cover archive')
    chain.append({'archive': label, 'sha256': digest(data), 'manifest_entries': len(checked)})
    return z, root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--paper-assets', type=Path, required=True)
    a = ap.parse_args()
    raw = a.source.read_bytes(); assets = a.paper_assets.read_bytes()
    if digest(raw) != SOURCE_HASH or digest(assets) != ASSET_HASH:
        raise ValueError('Top-level source identity mismatch')
    receipt = ROOT/'provenance/IMPORT_002.json'
    if receipt.exists():
        raise FileExistsError('Import already recorded; verify rather than overwrite')
    chain = []
    z5, r5 = open_verified(raw, a.source.name, chain)
    n4 = 'Finite_Dictionary_Predictive_Adequacy_v0_04_Reviewed_Bundle.zip'
    z4, r4 = open_verified(z5.read(r5+'upstream/'+n4), n4, chain)
    n3 = 'Finite_Dictionary_Predictive_Adequacy_v0_03_Reviewed_Bundle.zip'
    z3, r3 = open_verified(z4.read(r4+'upstream/'+n3), n3, chain)
    pr = r3+'parent/Finite_Dictionary_Predictive_Adequacy_v0_02/'
    nf = 'NMCD_PC_FIT_001_v0_1_Implementation_Bundle.zip'
    f, fr = open_verified(z3.read(pr+'upstream/'+nf), nf, chain)
    nd = 'NMCD_PC_DEP_001_v0_1_Implementation_Bundle.zip'
    d, dr = open_verified(z3.read(pr+'upstream/'+nd), nd, chain)
    az, ar = open_verified(assets, a.paper_assets.name, chain)
    selected = {}; records = []
    def take(z, label, member, destination):
        data = z.read(member)
        if destination in selected:
            raise ValueError('Duplicate destination')
        if destination.endswith('.py'):
            ast.parse(data.decode())
        selected[destination] = data
        records.append({'source_archive': label, 'source_member': member,
                        'destination': destination, 'sha256': digest(data),
                        'bytes': len(data), 'source_bytes_changed': False})
    def tree(z, label, prefix, destination):
        for name in sorted(z.namelist()):
            if name.startswith(prefix) and not name.endswith('/'):
                take(z, label, name, destination+name[len(prefix):])
    base = 'reference_impl/work/fit_parent/NMCD_PC_FIT_001_v0_1/'
    ip = 'vendor/inf001_v02/parent/'
    for name in ['analysis_src/run_matched.py','analysis_src/additional_checks.py','analysis_src/planning_check.py','contracts/range_sensitivity_plan.json']:
        take(z3, n3, pr+name, 'reference_impl/'+name)
    take(f,nf,fr+'scripts/run_experiment.py',base+'scripts/run_experiment.py')
    for name in ['scripts/run_experiments.py','contracts/development_plan.json']:
        take(f,nf,fr+ip+name,base+ip+name)
    vp = 'vendor/inf001_v02/'
    for name in ['scripts/run_extensions.py','contracts/review_plan.json']:
        take(f,nf,fr+vp+name,base+vp+name)
    tree(f,nf,fr+vp+'evidence/extensions/classes/','reference/inference/classes/')
    take(f,nf,fr+'evidence/run_002/campaign/summary.json',base+'evidence/run_002/campaign/summary.json')
    take(f,nf,fr+'evidence/run_002/resources/resource_timings.json','reference/resources/generic_timing.json')
    tree(f,nf,fr+ip+'evidence/development_final/predictive/','reference/inference/predictive/')
    for stage in ['inverse','numerical']:
        tree(f,nf,fr+ip+'evidence/development_final/'+stage+'/','reference/inference/'+stage+'/')
    for name in ['src/dependence_core.py','scripts/calibrate_dependence.py','scripts/exact_duplicate_audit.py','contracts/development_plan.json','contracts/PLAN_SHA256.txt','vendor/v03/src/nmcd_pc.py','vendor/v03/src/nmcd_review_extensions.py']:
        take(d,nd,dr+name,'reference_impl/work/dep_parent/NMCD_PC_DEP_001_v0_1/'+name)
    for name in ['all_cases.json','summary.json']:
        take(d,nd,dr+'evidence/development_run/calibration/'+name,'reference/dependence_original/'+name)
    tree(z3,n3,pr+'evidence/matched_run01/cells/','reference_impl/evidence/matched_run01/cells/')
    for name in ['summary.json','fits.json']:
        take(z3,n3,pr+'evidence/matched_run01/'+name,'reference_impl/evidence/matched_run01/'+name)
    tree(z3,n3,pr+'evidence/additional01/dep/','reference_impl/evidence/additional01/dep/')
    take(z3,n3,pr+'evidence/power_final/power.json','reference_impl/evidence/power_final/power.json')
    take(z3,n3,pr+'manuscript/data/reference_risks.json','reference/reference_risks.json')
    take(z4,n4,r4+'evidence/planning_audit.json','reference/planning_audit.json')
    take(z4,n4,r4+'scripts/audit_math.py','experiments/original/planning_audit.py')
    take(z3,n3,r3+'scripts/complete_results.py','experiments/original/complete_results.py')
    for name in az.namelist():
        relative = name.removeprefix(ar)
        if relative.startswith(('figures/','generated/')) and not name.endswith('/'):
            take(az,a.paper_assets.name,name,'reference/paper_assets/'+relative)
    for name,data in selected.items():
        p = ROOT/name
        if p.is_symlink() or p.exists():
            raise FileExistsError('Refusing to overwrite: '+name)
    for name,data in selected.items():
        p = ROOT/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    report = {'schema':'bdtp.campaign-assets-import.v1','date':'2026-10-21',
              'source_chain':chain,'files':records,'source_bytes_changed':False,
              'scope':'Original main-paper campaigns and stored results; manuscript and external reconstructions excluded',
              'performance_replay_in_this_import':False,'licence':None,'doi':None}
    receipt.write_text(json.dumps(report,indent=2)+'\n')
    text = ''.join(digest(data)+'  '+name+'\n' for name,data in sorted(selected.items()))
    (ROOT/'provenance/IMPORT_002_SHA256SUMS.txt').write_text(text)
    print(json.dumps({'files':len(records),'bytes':sum(len(x)for x in selected.values()),'source_chain':chain,'manifest_sha256':digest(text.encode())},indent=2))

if __name__ == '__main__':
    main()
