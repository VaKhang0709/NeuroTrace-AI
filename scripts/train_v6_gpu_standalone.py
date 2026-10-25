"""Colab/Kaggle orchestration. No substitute data, no automatic test partitioning.

Run as python -m scripts.train_v6_gpu_standalone. --prepare-only allows CPU data
preparation; actual GPU training aborts without CUDA. Mirrored datasets need
reviewed explicit trajectory conversion. Production code rejects mock manifests.
"""
import argparse
import json
from pathlib import Path
import torch
from src.data_ingestion.auto_downloader import acquire
from src.data_loaders.public_trajectory_parser import load_uci,append_metadata_summary
from src.data_loaders.schema import Trajectory
from src.augmentation.tremor_injector import generate,eligible_rows
from src.preprocessing.v6_pipeline import BASE_CONTRACT,DWT_CONTRACT,feature_names
from src.training.v6_train import run,validate_partition


def gpu_info(require=True):
    if not torch.cuda.is_available():
        print('CUDA unavailable. GPU training NOT STARTED.')
        if require: raise RuntimeError('Select A100/T4/P100 GPU runtime; --prepare-only is data-only')
        return None
    p=torch.cuda.get_device_properties(0)
    info=dict(name=p.name,vram_gib=round(p.total_memory/1024**3,2),
              capability=list(torch.cuda.get_device_capability(0)),torch=torch.__version__)
    print(json.dumps(info,indent=2)); return info


def execute(args):
    gpu_info(require=not args.prepare_only)
    mirrors=json.loads(Path(args.mirrors).read_text()) if args.mirrors else ()
    acquire(args.raw_root,args.local_uci,mirrors)
    records=[]
    if args.records:
        records=[Trajectory.from_dict(r) for r in json.loads(Path(args.records).read_text())]
    if not any(r.source=='uci395' for r in records):
        records.extend(load_uci(Path(args.raw_root)/'uci_395',args.subject_map))
    if any('mock' in r.source.lower() for r in records): raise ValueError('Mock records are forbidden in this production runner')
    append_metadata_summary(records,'reports/v6_ingested_datasets_summary.md')
    partition=json.loads(Path(args.partition).read_text()); dev,_,_=validate_partition(records,partition)
    manifest=None
    if not args.skip_generation:
        path=Path(args.expanded_manifest)
        if not path.exists():
            summary=generate(records,dev,args.synthetic_output,path,args.per_domain,args.seed)
            print(json.dumps(summary,indent=2))
        else:
            # Read all provenance rows and validate mocked evidence is absent.
            rows=eligible_rows(path,dev)
            from src.augmentation.tremor_injector import digest
            if any(not Path(r['path']).is_file() or digest(r['path'])!=r['sha256'] for r in rows):
                raise ValueError('Existing generated file is missing or changed')
            if len(rows)<2*args.per_domain: raise ValueError('Existing manifest has fewer eligible variants than requested')
        manifest=str(path)
    # Private prepared snapshot makes download+local records reproducible for export.
    from dataclasses import asdict
    prepared=[]
    for tr in records:
        item=asdict(tr); item['samples']=[[float(v) if __import__('math').isfinite(v) else None for v in row] for row in tr.samples]
        item['updrs']=float(tr.updrs) if __import__('math').isfinite(tr.updrs) else None
        item.update(schema_version='neurotrace-trajectory-1',time_units='seconds'); prepared.append(item)
    prepared_path=Path(args.prepared_records); prepared_path.parent.mkdir(parents=True,exist_ok=True)
    prepared_path.write_text(json.dumps(prepared,allow_nan=False))
    names=feature_names(args.feature_contract)
    print('Feature contract:',args.feature_contract,'channels:',list(names))
    if args.prepare_only: return
    config=json.loads(Path(args.config).read_text())
    results=run(records,partition,args.output,epochs=args.epochs,ssl_epochs=args.ssl_epochs,
                batch_size=args.batch_size,seed=args.seed,device='cuda',model_config=config['model'],
                augmentation_probabilities=config.get('augmentation_probabilities'),
                feature_contract=args.feature_contract,expanded_manifest=manifest if args.use_synthetic else None,
                synthetic_fraction=args.synthetic_fraction)
    print('Five-fold checkpoints:',json.dumps([r['best_checkpoint'] for r in results],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-records',default='data/processed/v6_unified_records.json')
    p.add_argument('--records',help='Authorized unified JSON list; can include UCI to bypass local raw parsing')
    p.add_argument('--partition',required=True); p.add_argument('--subject-map')
    p.add_argument('--raw-root',default='data/raw'); p.add_argument('--local-uci'); p.add_argument('--mirrors')
    p.add_argument('--config',default='configs/v6_training.json'); p.add_argument('--output',default='models/v6/checkpoints')
    p.add_argument('--synthetic-output',default='data/processed/v6_synthetic_10k')
    p.add_argument('--expanded-manifest',default='data/processed/v6_expanded_10k_manifest.csv')
    p.add_argument('--per-domain',type=int,default=5001); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--ssl-epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-generation',action='store_true',help='Explicit baseline-only run; no synthetic count claim')
    p.add_argument('--use-synthetic',action='store_true',help='Opt-in pseudo-PD ablation, never clinical evidence')
    p.add_argument('--synthetic-fraction',type=float,default=.25)
    p.add_argument('--feature-contract',choices=[BASE_CONTRACT,DWT_CONTRACT],default=DWT_CONTRACT)
    args=p.parse_args()
    if args.use_synthetic and args.skip_generation: p.error('Cannot use synthetic data while skipping generation')
    execute(args)

if __name__=='__main__': main()
