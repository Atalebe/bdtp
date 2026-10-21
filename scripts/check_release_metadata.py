"""Offline checks for the approved v1.0.0 release; no Git writes or simulations."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
VERSION='1.0.0'
ORCID='0000-0003-2317-4268'

def digest(path: Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(test: bool, message: str)->None:
    if not test:
        raise ValueError(message)

def licence_for(relative: str)->str:
    p=PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
    if relative in ('LICENSE','LICENSES/CC-BY-4.0.txt'):
        return 'STANDARD-LICENCE-TEXT'
    if p.suffix in ('.py','.sh') or 'contracts' in p.parts:
        return 'MIT'
    if relative.startswith('.github/workflows/') or relative in ('.gitignore','.gitattributes'):
        return 'MIT'
    if len(p.parts)==1 and p.name.startswith('requirements') and p.suffix=='.txt':
        return 'MIT'
    return 'CC-BY-4.0'

def validate_metadata(c: dict,z: dict)->None:
    require(c.get('cff-version')=='1.2.0','CFF version')
    require(c.get('version')==z.get('version')==VERSION,'Version mismatch')
    require(c.get('title')==z.get('title') and c['title'].startswith('Finite-Dictionary'),'Title mismatch')
    require(c.get('type')==z.get('upload_type')=='software','Software type')
    require(c.get('license')=='MIT' and z.get('license')=='mit','Primary software licence')
    require(z.get('access_right')=='open','Access right')
    require(c.get('date-released')==z.get('publication_date')=='2026-10-21','Release date')
    require('CC BY 4.0' in c.get('abstract','') and 'CC BY 4.0' in z.get('description',''),'Missing content licence scope')
    require(c['authors'][0]['orcid']=='https://orcid.org/'+ORCID,'CFF author ORCID')
    require(z['creators'][0]['orcid']==ORCID,'Zenodo author ORCID')
    require(c['authors'][0]['given-names']=='Stephen' and c['authors'][0]['family-names']=='Atalebe','CFF author')
    require(z['creators'][0]['name']=='Atalebe, Stephen','Zenodo author')
    require(c['repository-code']=='https://github.com/Atalebe/bdtp','Repository')
    require('doi' not in c and 'doi' not in z,'Unverified DOI in frozen first-release metadata')

def run(root: Path=ROOT, require_validation: bool=False)->dict:
    c=json.loads((root/'CITATION.cff').read_text())
    z=json.loads((root/'.zenodo.json').read_text())
    validate_metadata(c,z)
    sources=json.loads((root/'provenance/LICENCE_SOURCES.json').read_text())
    for record in sources.values():
        require(digest(root/record['file'])==record['sha256'],'Licence text identity')
    require('Copyright (c) 2026 Stephen Atalebe' in (root/'LICENSE').read_text(),'MIT copyright')
    require('Attribution 4.0 International' in (root/'LICENSES/CC-BY-4.0.txt').read_text(),'CC licence text')
    scope=json.loads((root/'provenance/LICENCE_SCOPE.json').read_text())
    require(scope['software']=='MIT' and scope['research_content']=='CC-BY-4.0','Licence scope')
    a=json.loads((root/'provenance/RELEASE_v1_0_0_AUTHORIZATION.json').read_text())
    require(a['public_release_authorized'] is True and a['tag']=='v1.0.0','Release not approved')
    require(a['new_scientific_runs_authorized'] is False,'Unapproved science expansion')
    count=0
    for label in ('001','002'):
        for item in json.loads((root/f'provenance/IMPORT_{label}.json').read_text())['files']:
            p=root/item['destination']
            require(not p.is_symlink() and digest(p)==item['sha256'],'Imported identity: '+item['destination'])
            count+=1
    require(count==124,'Imported source inventory')
    require((root/'docs/releases/v1.0.0.md').is_file(),'Release notes missing')
    if require_validation:
        report=json.loads((root/'provenance/RELEASE_v1_0_0_VALIDATION.json').read_text())
        require(report['status']=='LOCAL_RELEASE_GATES_PASS','Local qualification incomplete')
        require(report['reverified_replay_files']==22,'Replay evidence incomplete')
        require(report['new_scientific_runs']==0,'Unexpected new experiments')
        require(report['release_metadata_schema_validated'] is True,'Metadata schema unchecked')
        for path,sha in report['release_file_sha256'].items():
            require(digest(root/path)==sha,'Qualified release file changed: '+path)
    return {'status':'PASS','version':VERSION,'imported_files':count,'licences':{'software':'MIT','research_content':'CC-BY-4.0'},'require_validation':require_validation,'new_scientific_runs':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-validation',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(require_validation=args.require_validation),indent=2))
