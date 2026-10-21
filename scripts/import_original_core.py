"""Selective byte-preserving import from the author's downloaded archive. No execution."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, zipfile, ast
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/c/Users/satal/Downloads/Finite_Dictionary_Predictive_Adequacy_v0_05_Reviewed_Bundle.zip')
EXPECTED='49fb52caf751ff14b2a8095486b36da4456c490ef8dee2f538a175cc1a308174'
PAYLOAD='323b04a90766fcd32fdfdf59e1d80213fd92b17b8f138cef9fbbf6530d384d8d'
records=[]; chain=[]
def digest(b): return hashlib.sha256(b).hexdigest()
def opened(data,label):
    z=zipfile.ZipFile(io.BytesIO(data));names=z.namelist()
    if len(names)!=len(set(names)): raise ValueError('duplicate member')
    for item in z.infolist():
        p=PurePosixPath(item.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in item.filename or (item.external_attr>>16)&0o170000==0o120000: raise ValueError('unsafe member')
    root=names[0].split('/')[0]+'/'
    manifest=root+'SHA256SUMS.txt'; count=0
    for line in z.read(manifest).decode().splitlines():
        if not line.strip(): continue
        h,n=line.split(maxsplit=1);n=n.lstrip('* ')
        if digest(z.read(root+n))!=h:raise ValueError('manifest mismatch '+n)
        count+=1
    chain.append({'archive':label,'sha256':digest(data),'root_manifest_entries_verified':count})
    return z,root
raw=SOURCE.read_bytes()
if digest(raw)!=EXPECTED:raise ValueError('downloaded archive identity mismatch')
z5,r5=opened(raw,SOURCE.name)
name4='Finite_Dictionary_Predictive_Adequacy_v0_04_Reviewed_Bundle.zip'
z4,r4=opened(z5.read(r5+'upstream/'+name4),name4)
name3='Finite_Dictionary_Predictive_Adequacy_v0_03_Reviewed_Bundle.zip'
z3,r3=opened(z4.read(r4+'upstream/'+name3),name3)
pr=r3+'parent/Finite_Dictionary_Predictive_Adequacy_v0_02/'
fitname='NMCD_PC_FIT_001_v0_1_Implementation_Bundle.zip'
fraw=z3.read(pr+'upstream/'+fitname)
if digest(fraw)!='c6b24a88d5bf680ee19a0fccde5aa7b057a14c6197858c5293471a432c5a388f':raise ValueError('FIT identity')
f,fr=opened(fraw,fitname)
depname='NMCD_PC_DEP_001_v0_1_Implementation_Bundle.zip'
draw=z3.read(pr+'upstream/'+depname)
if digest(draw)!='95b7f040e764c3c47826b8a885145a103570842743e58bab2741cebe1d544f34':raise ValueError('DEP identity')
d,dr=opened(draw,depname)
base='reference_impl/work/fit_parent/NMCD_PC_FIT_001_v0_1/'
files=['src/fitting_core.py','tests/test_fitting.py','contracts/development_plan.json','vendor/inf001_v02/src/review_extension.py','vendor/inf001_v02/tests/test_review_extension.py','vendor/inf001_v02/parent/src/inference_core.py','vendor/inf001_v02/parent/tests/test_inference.py','vendor/inf001_v02/parent/vendor/dep001/src/dependence_core.py','vendor/inf001_v02/parent/vendor/dep001/tests/test_dependence.py','vendor/inf001_v02/parent/vendor/v03/src/nmcd_pc.py','vendor/inf001_v02/parent/vendor/v03/src/nmcd_review_extensions.py','vendor/inf001_v02/parent/vendor/v03/tests/test_core.py','vendor/inf001_v02/parent/vendor/v03/tests/test_v03.py','vendor/inf001_v02/parent/vendor/v03/contracts/audit_plan.json','vendor/inf001_v02/parent/vendor/v03/contracts/qualification_draft.json']
selected=[]
def take(z,label,member,dest):
    data=z.read(member)
    if dest.endswith('.py'):ast.parse(data.decode())
    selected.append((dest,data))
    records.append({'source_archive':label,'source_member':member,'destination':dest,'sha256':digest(data),'bytes':len(data),'changed':False})
for name in files:take(f,fitname,fr+name,base+name)
take(d,depname,dr+'contracts/development_plan.json',base+'vendor/inf001_v02/parent/vendor/dep001/contracts/development_plan.json')
for name in ['analysis_src/range_tools.py','tests/test_range.py']:
    take(z3,name3,pr+name,'reference_impl/'+name)
take(z3,name3,pr+'evidence/planning.json','reference/planning.json')
ordered=''.join(digest(data)+'  '+name+'\n' for name,data in sorted(selected))
if digest(ordered.encode())!=PAYLOAD:raise ValueError('selected payload differs from locally tested snapshot')
for name,data in selected:
    p=ROOT/name
    if p.exists() or p.is_symlink():raise FileExistsError(p)
receipt=ROOT/'provenance/IMPORT_001.json'
if receipt.exists():raise FileExistsError(receipt)
for name,data in selected:
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
report={'schema':'bdtp.original-core-import.v1','date':'2026-10-21','scope':'Original numerical methods and selected deterministic tests only; not complete paper reproduction','source_chain':chain,'files':records,'payload_manifest_sha256':PAYLOAD,'source_bytes_unchanged':True,'excluded':['reviewer reconstructions','review reports','correspondence','manuscripts','credentials','nested archives','failed source variants','Monte Carlo campaign runners'],'packaging_repair':'Initial local preparation omitted two legacy JSON contracts required by tests. They were restored unchanged before import; numerical code and tests unchanged. No failed attempt was committed.'}
receipt.write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'provenance/IMPORT_001_SHA256SUMS.txt').write_text(ordered)
print(json.dumps({'files_imported':len(selected),'payload_sha256':PAYLOAD,'source_chain':chain,'scientific_campaigns_run':0,'git_operations':False},indent=2))
