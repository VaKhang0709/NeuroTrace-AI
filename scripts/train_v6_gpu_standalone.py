"""Recorded-data UCI preparation before training. CUDA is needed only to train.

Missing unified records/partition are built by the real UCI395 loader. Existing
records are validated, NEVER repaired by a notebook adapter or replaced with mocks.
Clock interpretation is explicit. Wacom-only baseline does not synthesize phone PD.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from src.data_ingestion.auto_downloader import acquire
from src.data_loaders.public_trajectory_parser import append_metadata_summary
from src.data_loaders.schema import Trajectory
from src.data_loaders.uci395_loader import build,partition_subjects,to_record
from src.augmentation.tremor_injector import generate,eligible_rows
from src.preprocessing.v6_pipeline import BASE_CONTRACT,DWT_CONTRACT,feature_names,transform
from src.training.v6_train import run,validate_partition


def gpu_info(require=True):
    if not torch.cuda.is_available():
        print('CUDA unavailable. GPU training NOT STARTED.',flush=True)
        if require: raise RuntimeError('Select A100/T4/P100 GPU runtime; --prepare-only is data-only')
        return None
    p=torch.cuda.get_device_properties(0)
    info=dict(name=p.name,vram_gib=round(p.total_memory/1024**3,2),
              capability=list(torch.cuda.get_device_capability(0)),torch=torch.__version__)
    print(json.dumps(info,indent=2),flush=True); return info


def read_records(path):
    raw=json.loads(Path(path).read_text())
    if not isinstance(raw,list) or not raw: raise ValueError('Expected nonempty unified record LIST, not a raw table/wrapper')
    records=[]; failures=[]
    for i,value in enumerate(raw):
        try:
            if not isinstance(value,dict): raise ValueError('Expected a versioned record dictionary')
            records.append(Trajectory.from_dict(value))
        except (ValueError,TypeError,KeyError) as exc: failures.append(dict(record_index=i,error=str(exc)))
    if failures:
        print('CANONICAL RECORD FAILURES (no replacement/repair):',json.dumps(failures,indent=2),flush=True)
        raise ValueError(f'{len(failures)}/{len(raw)} records violate schema.py; use the UCI loader to rebuild from raw data')
    return records


def execute(args):
    gpu_info(require=not args.prepare_only)
    records_path=Path(args.records); partition_path=Path(args.partition)
    if not records_path.exists():
        local=Path(args.local_uci) if args.local_uci else None
        if local is not None and not local.exists():
            raise FileNotFoundError(f'Specified local UCI input does not exist: {local}; no substitute download into that path')
        if local is not None and local.is_dir():
            root=local; print('Using real local UCI directory directly (no download/copy):',root,flush=True)
        else:
            mirrors=json.loads(Path(args.mirrors).read_text()) if args.mirrors else ()
            rows=acquire(args.raw_root,args.local_uci,mirrors,report=args.ingestion_report)
            root=Path(args.raw_root)/'uci_395'
            if not root.is_dir():
                raise FileNotFoundError('Real UCI acquisition failed. Supply --local-uci-dir /path/to/hw_dataset or an authorized --local-uci zip. No mock fallback. Outcomes: '+json.dumps(rows))
        records,partition,_=build(root,records_path,partition_path,args.uci_time_unit,args.uci_azimuth_unit,
                                  args.seed,args.subject_map,args.feature_contract)
    else:
        records=read_records(records_path)
        if not partition_path.exists():
            # QC BEFORE generating a new subject partition; no drawing-level split.
            for i,tr in enumerate(records):
                try: transform(tr,contract=args.feature_contract)
                except ValueError as exc: raise ValueError(f'Feature readiness failed at existing record {i}: {exc}. Review raw QC before locking a partition') from exc
            partition=partition_subjects(records,args.seed)
            partition_path.parent.mkdir(parents=True,exist_ok=True); partition_path.write_text(json.dumps(partition,indent=2))
        else: partition=json.loads(partition_path.read_text())
    if any('mock' in r.source.lower() for r in records): raise ValueError('Mock records forbidden in production runner')
    dev,holdout,unlabeled=validate_partition(records,partition)
    if {r.subject_id for r in records}!=dev|holdout|unlabeled: raise ValueError('Every record parent must belong to EXACTLY one locked partition')
    ingestion=records_path.with_suffix('.ingestion.json')
    if ingestion.exists(): print('Stored raw conversion/QC:',ingestion.read_text(),flush=True)
    else: print('Existing canonical input declares seconds. Upstream raw clock interpretation UNKNOWN; no reconversion.',flush=True)
    print('Canonical layout: [x,y,seconds,pressure,azimuth_radians]; age unknown=-1, UPDRS missing=null.',flush=True)
    print(json.dumps(dict(records=len(records),subjects_by_label=dict(Counter(t.label for t in {r.subject_id:r for r in records}.values())),
                         domains=dict(Counter(r.domain for r in records)),partition_sizes={k:len(v) for k,v in partition.items()}),indent=2),flush=True)
    report=Path(args.ingestion_report)
    if not report.exists():
        report.parent.mkdir(parents=True,exist_ok=True); report.write_text('# V6 supplied recorded input metadata\n')
    append_metadata_summary(records,report)
    # This is actual feature preparation, not merely re-saving a raw JSON file.
    tensors=[]; labels=[]; parents=[]; domains=[]; record_indices=[]
    for i,tr in enumerate(records):
        try: x,_=transform(tr,contract=args.feature_contract)
        except ValueError as exc:
            raise ValueError(f'Feature preparation failed at record {i}: {exc}. No silent locked-holdout exclusion; review QC before partitioning') from exc
        if x.shape[1:]!=(19,512) or not np.isfinite(x).all(): raise ValueError('Invalid derived feature tensor')
        tensors.append(x); labels.extend([tr.label]*len(x)); parents.extend([tr.subject_id]*len(x))
        domains.extend([tr.domain]*len(x)); record_indices.extend([i]*len(x))
    prepared=Path(args.prepared_records); prepared.parent.mkdir(parents=True,exist_ok=True)
    prepared.write_text(json.dumps([to_record(t) for t in records],allow_nan=False))
    features=Path(args.prepared_features) if args.prepared_features else prepared.with_suffix('.features.npz')
    features.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(features,X=np.concatenate(tensors),label=np.array(labels),subject=np.array(parents),
                        domain=np.array(domains),record_index=np.array(record_indices),feature_contract=np.array(args.feature_contract))
    print('Prepared',len(parents),'real windows, shape [N,19,512]; contract=',args.feature_contract,
          'cache=',features,flush=True)
    manifest=None
    # UCI-only preparation MUST NOT require scarce mobile/mouse HC inputs.
    if args.use_synthetic:
        path=Path(args.expanded_manifest)
        if not path.exists():
            print(json.dumps(generate(records,dev,args.synthetic_output,path,args.per_domain,args.seed),indent=2),flush=True)
        else:
            rows=eligible_rows(path,dev)
            from src.augmentation.tremor_injector import digest
            if len(rows)<2*args.per_domain or any(not Path(r['path']).is_file() or digest(r['path'])!=r['sha256'] for r in rows):
                raise ValueError('Invalid/incomplete recorded-source synthetic manifest')
        manifest=str(path)
        print('SYNTHETIC PSEUDO-LABEL ABLATION ONLY; not evidence of mobile-PD detection.',flush=True)
    else: print('Real-only baseline: no synthetic generation requested. UCI Wacom cannot validate mobile/touch sensitivity.',flush=True)
    if args.prepare_only: return
    config=json.loads(Path(args.config).read_text())
    results=run(records,partition,args.output,epochs=args.epochs,ssl_epochs=args.ssl_epochs,
                batch_size=args.batch_size,seed=args.seed,device='cuda',model_config=config['model'],
                augmentation_probabilities=config.get('augmentation_probabilities'),feature_contract=args.feature_contract,
                expanded_manifest=manifest,synthetic_fraction=args.synthetic_fraction)
    print('Five-fold checkpoints:',json.dumps([r['best_checkpoint'] for r in results],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',default='data/processed/v6_unified_records.json')
    p.add_argument('--partition',default='data/processed/partition.json')
    p.add_argument('--prepared-records',default='data/processed/v6_prepared_records.json')
    p.add_argument('--prepared-features'); p.add_argument('--subject-map')
    p.add_argument('--local-uci','--local-uci-dir',dest='local_uci',help='Real extracted directory or authorized zip')
    p.add_argument('--raw-root',default='data/raw'); p.add_argument('--mirrors')
    p.add_argument('--uci-time-unit',choices=['seconds','milliseconds','microseconds'],default='milliseconds',
                   help='Explicit raw timestamp interpretation; default ms is an assumption when source units absent')
    p.add_argument('--uci-azimuth-unit',choices=['unknown','degrees','radians'],default='unknown')
    p.add_argument('--ingestion-report',default='reports/v6_ingested_datasets_summary.md')
    p.add_argument('--config',default='configs/v6_training.json'); p.add_argument('--output',default='models/v6/checkpoints')
    p.add_argument('--synthetic-output',default='data/processed/v6_synthetic_10k')
    p.add_argument('--expanded-manifest',default='data/processed/v6_expanded_10k_manifest.csv')
    p.add_argument('--per-domain',type=int,default=5001); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--ssl-epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-generation',action='store_true',help='Backward-compatible explicit real-only baseline flag')
    p.add_argument('--use-synthetic',action='store_true',help='Explicit experimental pseudo-PD ablation, never mobile clinical evidence')
    p.add_argument('--synthetic-fraction',type=float,default=.25)
    p.add_argument('--feature-contract',choices=[BASE_CONTRACT,DWT_CONTRACT],default=DWT_CONTRACT)
    a=p.parse_args()
    if a.use_synthetic and a.skip_generation: p.error('Cannot request and skip synthesis simultaneously')
    execute(a)

if __name__=='__main__': main()
