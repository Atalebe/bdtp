"""Release-only tests; no network, statistical campaigns, or repository writes."""
from pathlib import Path
import copy, importlib.util, json, unittest
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_metadata',R/'scripts/check_release_metadata.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((R/'CITATION.cff').read_text());self.z=json.loads((R/'.zenodo.json').read_text())
    def test_metadata(self):
        M.validate_metadata(self.c,self.z)
    def test_complete_release_files(self):
        self.assertEqual(M.run()['imported_files'],124)
    def test_versions_agree(self):
        self.z['version']='2.0.0'
        with self.assertRaises(ValueError):M.validate_metadata(self.c,self.z)
    def test_primary_licence(self):
        self.z['license']='cc-by-4.0'
        with self.assertRaises(ValueError):M.validate_metadata(self.c,self.z)
    def test_content_scope_required(self):
        self.z['description']='Software only'
        with self.assertRaises(ValueError):M.validate_metadata(self.c,self.z)
    def test_author_identity(self):
        self.c['authors'][0]['orcid']='https://orcid.org/0000-0000-0000-0000'
        with self.assertRaises(ValueError):M.validate_metadata(self.c,self.z)
    def test_unverified_doi_rejected(self):
        self.z['doi']='10.0000/placeholder'
        with self.assertRaises(ValueError):M.validate_metadata(self.c,self.z)
    def test_code_mit(self):
        for p in ['scripts/x.py','tests/test_x.py','scripts/x.sh','reference_impl/vendor/x.py']:
            self.assertEqual(M.licence_for(p),'MIT')
    def test_execution_configuration_mit(self):
        for p in ['reference_impl/contracts/plan.json','.github/workflows/publish-v1.yml','requirements-numeric.txt']:
            self.assertEqual(M.licence_for(p),'MIT')
    def test_data_content_cc(self):
        for p in ['reference/x.json','reference/figures/x.pdf','reference/tables/x.csv','reference_impl/evidence/results.json','docs/x.md','CITATION.cff','.zenodo.json']:
            self.assertEqual(M.licence_for(p),'CC-BY-4.0')
    def test_legal_texts_distinct(self):
        for p in ['LICENSE','LICENSES/CC-BY-4.0.txt']:
            self.assertEqual(M.licence_for(p),'STANDARD-LICENCE-TEXT')
    def test_path_traversal_rejected(self):
        for p in ['../x','/tmp/x','foo/../../x']:
            with self.assertRaises(ValueError):M.licence_for(p)

if __name__=='__main__':unittest.main()
