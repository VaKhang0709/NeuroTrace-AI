"""One-time, versioned recovery of LOST family teacher inputs.

Explicitly authorized NEW 50epoch run, not reconstruction of the old teacher.
Old production ONNX/scaler are never overwritten. Completed recovery is reused.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def paths(root):
    root=Path(root)
    record=next((root/name for name in ('v6_prepared_records.json','prepared_records.json','records.json') if (root/name).is_file()),root/'prepared_records.json')
    return dict(checkpoint=root/'fold_0_best.pth',partition=root/'partition.json',records=record,
                teacher_onnx=root/'neurotrace_v6_tremorformer.onnx',scaler=root/'scaler_config_v6.json')


def needs_recovery(bundle):
    return any(not Path(p).is_file() for p in bundle.values())


def completed_bundle(root):
    """Only use a committed bundle whose five content hashes still match."""
    from hashlib import sha256
    manifest=Path(root)/'recovery_manifest.json'
    if not manifest.is_file(): return None
    data=json.loads(manifest.read_text())
    if not data.get('complete'): return None
    for key,path in data['paths'].items():
        p=Path(path)
        if not p.is_file() or sha256(p.read_bytes()).hexdigest()!=data['sha256'][key]: return None
    if set(data['paths'])!=set(paths(root)): return None
    return {k:Path(v) for k,v in data['paths'].items()}


def bootstrap(production,local_uci=None,work='/content/v6_family_teacher_work'):
    production=Path(production);family=production/'family';family.mkdir(parents=True,exist_ok=True)
    recovery=family/'teacher_recovery_v1';recovery.mkdir(parents=True,exist_ok=True)
    # Prefer the completed new lineage on subsequent runtimes.
    cached=completed_bundle(recovery)
    original=paths(production)
    if cached is not None:
        print('Reusing completed 50epoch recovery; no download/training.',flush=True);bundle=cached
    elif not needs_recovery(original):
        print('Existing complete teacher inputs found; no recovery training.',flush=True);bundle=original
    else:
        bundle=recover(production,recovery,local_uci,Path(work))
    pointer=family/'teacher_bundle.json'
    write_json(pointer,{k:str(Path(v).resolve()) for k,v in bundle.items()})
    print('Active matched teacher bundle:',pointer,flush=True)
    return bundle


def recover(production,recovery,local_uci,work):
    import numpy as np
    import torch
    from torch.utils.data import Dataset,DataLoader
    from sklearn.model_selection import train_test_split
    from src.data_loaders.uci395_loader import build
    from src.data_loaders.schema import Trajectory,TASKS
    from src.data_ingestion.auto_downloader import acquire
    from src.training.family_data import windows,sha256,load_context
    from src.training.v6_train import train_fold,seed_all,validate_partition
    from src.augmentation.v6_augmentations import DEFAULT_PROBABILITIES
    from src.models.v6_tremorformer import TremorFormer
    from scripts.export_v6_onnx_pipeline import export_pipeline

    if not torch.cuda.is_available(): raise RuntimeError('Select T4 GPU before automatic teacher recovery; no CPU training substitute')
    seed_all(42);work.mkdir(parents=True,exist_ok=True)
    bundle=paths(recovery);record_path=bundle['records'];partition_path=bundle['partition'];ck_path=bundle['checkpoint']
    print('RECOVERY: NEW 50epoch full PRO lineage; original exported artifacts preserved.',flush=True)
    # Persist records/partition BEFORE training, so a runtime disconnect cannot
    # lose them again. Preserve an existing original partition if one survived.
    if not record_path.exists() or not partition_path.exists():
        source=paths(production)
        if source['partition'].is_file() and not partition_path.exists(): shutil.copyfile(source['partition'],partition_path)
        if source['records'].is_file() and partition_path.is_file():
            if not record_path.exists(): shutil.copyfile(source['records'],record_path)
        else:
            local=Path(local_uci) if local_uci else None
            if local is not None and local.is_dir(): raw=local
            else:
                outcomes=acquire(work/'raw',local_uci=local_uci,report=work/'acquisition.md')
                raw=work/'raw/uci_395'
                if not raw.is_dir(): raise RuntimeError('Real UCI download failed; set LOCAL_UCI to original zip/directory. No mock fallback. '+json.dumps(outcomes))
            build(raw,record_path,partition_path,time_unit='milliseconds',seed=42,feature_contract='v6-dwt19-v1')
        write_json(recovery/'lineage.json',dict(status='NEW_RECOVERY_NOT_ORIGINAL_EXPERIMENT',
            partition_origin='preserved original' if source['partition'].is_file() else 'new deterministic subject-stratified recovery split,seed42',
            epochs=50,batch_size=8,USE_SYNTHETIC=False,ssl_epochs=0,
            time_unit='milliseconds is explicit interpretation; original mirrored README does not document clock units',
            optimization='cached real features; no online raw augmentation; local SSD checkpoints; no data/epoch cap'))
    records=[Trajectory.from_dict(t) for t in json.loads(record_path.read_text())]
    partition=json.loads(partition_path.read_text());dev,holdout,_=validate_partition(records,partition)
    if len(dev)!=62 or len(holdout)!=15: raise ValueError('Expected genuine 77subject UCI cohort,62dev/15holdout; no fabricated subjects')
    before=dict(partition=sha256(partition_path),records=sha256(record_path))
    reusable=False
    if ck_path.is_file():
        saved=torch.load(ck_path,map_location='cpu',weights_only=True)
        reusable=(saved.get('recovery_complete') is True and saved.get('epochs_completed')==50
                  and saved.get('recovery_input_hashes')==before)
        if not reusable: raise ValueError('Existing recovery checkpoint is incomplete/mismatched; refusing to silently replace it')
    if not reusable:
        labels={t.subject_id:t.label for t in records}
        subjects=sorted(dev)
        train,valid=train_test_split(subjects,test_size=.2,stratify=[labels[s] for s in subjects],random_state=42)
        assert not set(train)&set(valid) and not (set(train)|set(valid))&holdout
        # Resample/derive19channels ONCE, not once/window/epoch. Holdout untouched.
        train_features=windows(records,train);valid_features=windows(records,valid)
        class Cached(Dataset):
            def __init__(self,features): self.data=features;self.subjects=sorted({str(s) for s in features['subject']})
            def __len__(self): return len(self.data['X'])
            def __getitem__(self,i):
                f=self.data
                return dict(x=torch.from_numpy(f['X'][i]),domain=torch.tensor(f['device_id'][i]),
                    label=torch.tensor(f['label'][i]),subject=str(f['subject'][i]),task=torch.tensor(-1),
                    age=torch.tensor(-1),updrs=torch.tensor(float('nan')),synthetic=torch.tensor(False))
        model=TremorFormer();model.feature_contract='v6-dwt19-v1';model.ssl_subjects=[]
        probabilities={key:0. for key in DEFAULT_PROBABILITIES}
        folder=work/'fold_0'
        print('FULL PRO50epochs,one grouped member; batch8,AMP,real-only cached features.',flush=True)
        best=train_fold(model,DataLoader(Cached(train_features),batch_size=8,shuffle=True,pin_memory=True),
                        DataLoader(Cached(valid_features),batch_size=8,pin_memory=True),folder,
                        epochs=50,patience=51,effective_batch=256,device='cuda',seed=42,probabilities=probabilities)
        rows=list(csv.DictReader((folder/'epochs.csv').open()))
        if len(rows)!=50: raise RuntimeError('Recovery did not finish all50epochs; no complete teacher published')
        checkpoint=torch.load(best,map_location='cpu',weights_only=True)
        # Compact final selected weights/identity metadata, not50epochs of optimizer
        # writes to Drive. Full optimizer/top3 remain on local SSD during training.
        for key in ('optimizer_state','scheduler_state','scaler_state','rng_state'): checkpoint.pop(key,None)
        checkpoint.update(recovery_complete=True,epochs_completed=50,recovery_input_hashes=before)
        tmp=ck_path.with_suffix('.pth.partial');torch.save(checkpoint,tmp);tmp.replace(ck_path)
        shutil.copyfile(folder/'epochs.csv',recovery/'teacher_epochs.csv')
        del model,train_features,valid_features
        torch.cuda.empty_cache()
    else: print('50epoch checkpoint already saved; finishing export only,not retraining.',flush=True)
    if before!=dict(partition=sha256(partition_path),records=sha256(record_path)): raise ValueError('Locked recovery inputs changed during training')
    # New teacher MUST have a matching new ONNX/scaler. Never pair new weights
    # with the old production ONNX or its teacher-specific calibration map.
    export_pipeline([ck_path],records,partition,recovery,threshold_policy='fixed_0.5')
    load_context(record_path,partition_path,bundle['scaler'],ck_path,bundle['teacher_onnx'])
    write_json(recovery/'recovery_manifest.json',dict(complete=True,epochs_completed=50,
        paths={k:str(v.resolve()) for k,v in bundle.items()},sha256={k:sha256(v) for k,v in bundle.items()}))
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--production',required=True)
    p.add_argument('--local-uci');p.add_argument('--work',default='/content/v6_family_teacher_work')
    args=p.parse_args();bootstrap(args.production,args.local_uci,args.work)

if __name__=='__main__': main()
