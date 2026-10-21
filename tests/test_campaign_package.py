"""Packaging tests: no full campaigns or network calls."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import paper_campaign as C
import paper_outputs as O

class CampaignPackageTests(unittest.TestCase):
    def test_original_identities(self):
        self.assertEqual(C.verify_imports(),124)
    def test_import_counts(self):
        x=json.loads((ROOT/'provenance/IMPORT_002.json').read_text())
        self.assertEqual(len(x['files']),105)
        self.assertTrue(all(not r['source_bytes_changed'] for r in x['files']))
    def test_no_review_or_manuscript_import(self):
        x=json.loads((ROOT/'provenance/IMPORT_002.json').read_text())
        for r in x['files']:
            self.assertNotIn('/reviewer/',r['source_member'])
            self.assertFalse(r['destination'].endswith('Author_Copy.tex'))
    def test_sources_exist(self):
        for p in ('analysis_src/run_matched.py','analysis_src/additional_checks.py','analysis_src/planning_check.py'):
            self.assertTrue((ROOT/'reference_impl'/p).is_file())
    def test_ten_campaign_choices(self):
        self.assertEqual(len(C.CAMPAIGNS),10)
    def test_static_constants(self):
        plan=json.loads((ROOT/'reference_impl'/C.FP/'contracts/development_plan.json').read_text())
        self.assertEqual(plan['alpha'],.05);self.assertEqual(plan['epsilon'],.01)
        self.assertEqual(plan['training_budgets'],[64,512,4096])
        self.assertEqual(plan['world_budgets'],[16384,32768])
    def test_compare_identical(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            (a/'x.json').write_text('{}');(b/'x.json').write_text('{}')
            self.assertTrue(C.compare(a,b,['*.json'])['all_byte_identical'])
    def test_compare_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            (a/'x.json').write_text('{}');(b/'x.json').write_text('[]')
            self.assertFalse(C.compare(a,b,['*.json'])['all_byte_identical'])
    def test_compare_empty_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):C.compare(Path(d),Path(d),['*.json'])
    def test_compare_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir();(b/'x.json').write_text('{}')
            with self.assertRaises(ValueError):C.compare(a,b,['*.json'])
    def test_compare_extra_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir();(a/'x.json').write_text('{}');(b/'x.json').write_text('{}');(a/'y.json').write_text('{}')
            with self.assertRaises(ValueError):C.compare(a,b,['*.json'])
    def test_runtime_does_not_copy_reference_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            p=C.prepare_runtime(Path(d))
            self.assertFalse((p/'evidence/matched_run01/summary.json').exists())
            self.assertTrue((p/C.FP/'evidence/run_002/campaign/summary.json').is_file())
    def test_standalone_dependence_import(self):
        cmd=[sys.executable,'-c',"import sys;sys.path.insert(0,sys.argv[1]);import dependence_core;assert dependence_core.D.size==177",str(ROOT/'reference_impl'/C.DP/'src')]
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def test_driver_help(self):
        paths=[ROOT/'reference_impl'/C.FP/'scripts/run_experiment.py',ROOT/'reference_impl'/C.IP/'scripts/run_experiments.py',ROOT/'reference_impl'/C.DP/'scripts/calibrate_dependence.py']
        for p in paths:
            result=subprocess.run([sys.executable,str(p),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(result.returncode,0,result.stderr.decode())
    def test_full_result_record_checks(self):
        tables,checks,cells,power=O.table_data()
        self.assertEqual(len(checks),208);self.assertEqual(len(cells),20);self.assertEqual(len(power),90)
        self.assertEqual(len(tables),6)
    def test_complete_fitting_table(self):
        tables,*_=O.table_data()
        self.assertEqual(len(tables['table4_truncated_allocation']),48)
        self.assertEqual(len(tables['table6_other_allocations']),112)
    def test_all_duplication_factors(self):
        tables,*_=O.table_data()
        self.assertEqual({r['m']for r in tables['table3_dependence']},{1,16,256})
        self.assertEqual(len(tables['table3_dependence']),12)
    def test_typeset_integer_cells(self):
        tables,*_=O.table_data()
        self.assertEqual(sum(x['rows_checked']for x in O.validate_typeset_tables(tables)),35)
    def test_inference_risks(self):
        tables,*_=O.table_data();rows={r['case']:r for r in tables['table2_risk_gaps']}
        self.assertLess(rows['truncated_strong']['summary_gap'],.01)
        self.assertGreater(rows['truncated_strong']['fitted_summary_gap'],.01)
        self.assertGreater(rows['history_strong']['summary_gap'],.01)
    def test_truths_explicit(self):
        tables,*_=O.table_data();rows={r['case']:r for r in tables['table1_designs']}
        self.assertEqual(rows['history_moderate']['A'],8)
        self.assertEqual(rows['history_strong']['A'],16)
        self.assertTrue(all(r['gamma']==1.8 for r in rows.values()))
    def test_planning_crossing(self):
        x=O.load(ROOT/'reference/planning_audit.json')
        self.assertEqual(x['crossings']['70']['pair']['full_formula']['first_integer'],13464)
        self.assertEqual(x['crossings']['70']['generic']['full_formula']['first_integer'],24388)
    def test_reference_inference_inventory(self):
        self.assertEqual(len(list((ROOT/'reference/inference/predictive').glob('*.json'))),21)
        self.assertEqual(len(list((ROOT/'reference/inference/inverse').glob('*.json'))),7)
    def test_reference_figures_preserved(self):
        self.assertEqual(len(list((ROOT/'reference/paper_assets/figures').glob('*.pdf'))),4)
    def test_original_plan_checksum_is_staged(self):
        rel=C.DP/'contracts/PLAN_SHA256.txt'
        checksum=(ROOT/'reference_impl'/rel).read_text().split()[0]
        source=ROOT/'reference_impl'/C.DP/'contracts/development_plan.json'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),checksum)
        with tempfile.TemporaryDirectory() as d:
            staged=C.prepare_runtime(Path(d))
            self.assertEqual((staged/rel).read_bytes(),(ROOT/'reference_impl'/rel).read_bytes())
    def test_all_main_items_mapped(self):
        record=json.loads((ROOT/'reference/table_figure_map.json').read_text())
        self.assertEqual(len(record['main_items']),10)
        for item in record['main_items']:
            self.assertIn(item['campaign'],C.CAMPAIGNS)
            for p in item['reference_inputs']:self.assertTrue((ROOT/p).is_file(),p)
    def test_derived_tables_equal_fresh_extraction(self):
        tables,*_=O.table_data()
        record=json.loads((ROOT/'provenance/PAPER_TABLES.json').read_text())
        self.assertEqual(len(record['files']),6)
        with tempfile.TemporaryDirectory() as d:
            for item in record['files']:
                p=ROOT/item['path']; fresh=Path(d)/p.name
                O.write_csv(fresh,tables[p.stem])
                self.assertEqual(fresh.read_bytes(),p.read_bytes())
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
    def test_injected_failure_keeps_caller_no_completion(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'fail';env=os.environ.copy();env['PYTHON']=sys.executable
            command=['bash','-c','bash "$1" --campaign planning --output "$2" --inject-failure; s=$?; printf "CALLER_SURVIVED status=%s\\n" "$s"; test "$s" -ne 0','check',str(ROOT/'scripts/run_paper_campaign_no_exit.sh'),str(output)]
            r=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(r.returncode,0);self.assertIn(b'CALLER_SURVIVED status=1',r.stdout)
            self.assertTrue((output/'FAILURE.txt').is_file());self.assertFalse((output/'COMPLETE.json').exists());self.assertFalse((output/'runtime').exists())
    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'marker').write_text('KEEP')
            r=subprocess.run([sys.executable,str(ROOT/'scripts/paper_campaign.py'),'--campaign','planning','--output',d],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(r.returncode,0);self.assertEqual((p/'marker').read_text(),'KEEP');self.assertEqual(len(list(p.iterdir())),1)
    def test_protected_output_rejected(self):
        p=ROOT/'reference/forbidden_run'
        r=subprocess.run([sys.executable,str(ROOT/'scripts/paper_campaign.py'),'--campaign','planning','--output',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertNotEqual(r.returncode,0);self.assertFalse(p.exists())

if __name__=='__main__':
    unittest.main()
