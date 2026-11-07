"""Filesystem/notebook recovery guards; no fake training or disease metrics."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.bootstrap_v6_family_teacher import paths,needs_recovery,completed_bundle,bootstrap,write_json


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def files(self,root):
        root.mkdir(parents=True,exist_ok=True);bundle=paths(root)
        # IO-only sentinels,never passed to a model or reported as trained weights.
        for key,p in bundle.items():p.write_text('filesystem test '+key)
        return bundle
    def manifest(self,root,bundle,complete=True):
        write_json(root/'recovery_manifest.json',dict(complete=complete,paths={k:str(v) for k,v in bundle.items()},
            sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in bundle.items()}))
    def test_record_aliases_and_missing_internal_inputs(self):
        bundle=self.files(self.root);self.assertFalse(needs_recovery(bundle))
        bundle['checkpoint'].unlink();self.assertTrue(needs_recovery(bundle))
        bundle['checkpoint'].write_text('sentinel');bundle['partition'].unlink();self.assertTrue(needs_recovery(bundle))
        bundle['records'].unlink();(self.root/'records.json').write_text('[]')
        self.assertEqual(paths(self.root)['records'].name,'records.json')
    def test_existing_bundle_skips_retraining_and_creates_family(self):
        bundle=self.files(self.root)
        with patch('scripts.bootstrap_v6_family_teacher.recover') as train:
            result=bootstrap(self.root);train.assert_not_called()
        self.assertEqual(result,bundle);self.assertTrue((self.root/'family/teacher_bundle.json').is_file())
    def test_complete_recovery_reused_when_root_checkpoint_still_missing(self):
        folder=self.root/'family/teacher_recovery_v1';bundle=self.files(folder);self.manifest(folder,bundle)
        with patch('scripts.bootstrap_v6_family_teacher.recover') as train:
            result=bootstrap(self.root);train.assert_not_called()
        self.assertEqual(result,bundle)
    def test_missing_inputs_trigger_recovery_not_filenotfound(self):
        # Stub only the GPU operation; verify automatic orchestration without training.
        bundle={k:self.root/k for k in paths(self.root)}
        with patch('scripts.bootstrap_v6_family_teacher.recover',return_value=bundle) as train:
            result=bootstrap(self.root)
        train.assert_called_once();self.assertEqual(result,bundle)
    def test_incomplete_or_changed_bundle_not_claimed_complete(self):
        bundle=self.files(self.root);self.manifest(self.root,bundle,False)
        self.assertIsNone(completed_bundle(self.root))
        self.manifest(self.root,bundle);self.assertEqual(completed_bundle(self.root),bundle)
        bundle['partition'].write_text('changed')
        self.assertIsNone(completed_bundle(self.root))
    def test_notebook_python_and_recovery_order(self):
        nb=json.loads(Path('notebooks/v6_family_distill.ipynb').read_text());code=[]
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                text=''.join(cell['source']);ast.parse(text);code.append(text)
                self.assertFalse(cell['outputs'])
        text='\n'.join(code)
        self.assertNotIn('raise FileNotFoundError',text)
        self.assertIn('FAMILY.mkdir(parents=True,exist_ok=True)',text)
        self.assertIn('BATCH_SIZE = 8',text);self.assertIn('USE_SYNTHETIC=False',text)
        self.assertLess(text.index('run_live(bootstrap)'),text.index('load_context(RECORDS'))
        self.assertLess(text.index('run_live(bootstrap)'),text.index("'-m','src.training.distill'"))
    def test_notebook_budget_is_five_cu_and_two_hours_forwarded_to_trainer(self):
        nb=json.loads(Path('notebooks/v6_family_distill.ipynb').read_text())
        values={};text=''
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            code=''.join(cell['source']);text+=code+'\n'
            for node in ast.parse(code).body:
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id in ('CU_ALLOWANCE','WALL_SECONDS'):
                            values[target.id]=ast.literal_eval(node.value)
        self.assertEqual(values['CU_ALLOWANCE'],5.0)
        self.assertEqual(values['WALL_SECONDS'],7200)
        self.assertIn('remaining=int(WALL_SECONDS-(time.monotonic()-JOB_START))',text)
        self.assertIn("'--max-seconds',str(remaining)",text)
        self.assertNotIn('WALL_SECONDS=40*60',text)

    def test_teacher_is_full50_and_outputs_versioned(self):
        source=Path('scripts/bootstrap_v6_family_teacher.py').read_text()
        self.assertIn('epochs=50,patience=51',source)
        self.assertIn('batch_size=8',source)
        self.assertIn("self.subjects=sorted({str(s) for s in features['subject']})",source)
        self.assertIn("family/'teacher_recovery_v1'",source)
        self.assertIn("recovery_complete=True,epochs_completed=50",source)
        self.assertIn("export_pipeline([ck_path]",source)
        self.assertNotIn('n_splits=5',source) # one complete member,not5x50epochs

if __name__=='__main__':unittest.main()
