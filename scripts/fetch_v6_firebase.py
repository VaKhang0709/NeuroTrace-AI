"""Authorized Firebase ingestion: python -m scripts.fetch_v6_firebase --config ...

Each v6_sessions document: device_uuid, subject_id (stable consented participant
ID if known), session_id, tasks=[{task_id, trajectory: unified-record OR
storage_path: bucket-relative JSON path}]. No credentials embedded or printed.
Storage blobs must contain the versioned unified schema; unclear records rejected.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.data_loaders.schema import Trajectory


def component(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError('Unsafe/empty path identifier')
    return value


def ingest(config, report='reports/v6_data_ingestion_report.md'):
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    cred = Path(config['service_account'])
    if not cred.is_file():
        raise FileNotFoundError('Authorized service account missing; no ingestion performed')
    app = firebase_admin.initialize_app(credentials.Certificate(str(cred)),
          {'storageBucket': config['storage_bucket']}, name='neurotrace-v6-ingestion')
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    rows, quality, dropped = [], defaultdict(list), 0
    seen = set()
    try:
        for doc in firestore.client(app).collection(config.get('collection', 'v6_sessions')).stream():
            record = doc.to_dict()
            try:
                device = component(record['device_uuid'])
                session = component(record.get('session_id', doc.id))
                subject = str(record.get('subject_id') or 'device::'+device)
                # Cross-device same participant requires stable subject_id supplied upstream.
                tasks = record['tasks']
                if not isinstance(tasks, list) or not tasks:
                    raise ValueError('Missing tasks')
            except (KeyError, ValueError, TypeError):
                dropped += 1; continue
            valid_session = False
            for task in tasks:
                try:
                    tid = component(task['task_id'])
                    key = (device, session, tid)
                    if key in seen:
                        raise ValueError('Duplicate task path')
                    if 'storage_path' in task:
                        blob = storage.bucket(app=app).blob(task['storage_path'])
                        blob.reload()
                        if blob.size is None or blob.size > config.get('max_blob_bytes', 50*1024**2):
                            raise ValueError('Oversized/unbounded blob')
                        raw = blob.download_as_bytes(if_generation_match=blob.generation)
                        payload = json.loads(raw)
                    else:
                        payload = task['trajectory']
                    payload = dict(payload, subject_id=subject, session_id=session)
                    tr = Trajectory.from_dict(payload)
                    path = output/device/session/(tid+'.json')
                    if path.exists():
                        # Never overwrite a prior collection without explicit operator review.
                        raise ValueError('Task output already exists')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, allow_nan=False))
                    seen.add(key); valid_session = True
                    dt = np.diff(tr.samples[:, 2]); duration = np.ptp(tr.samples[:, 2])
                    quality[device].append((1/np.median(dt), duration, len(tr.samples)))
                    rows.append(dict(device_uuid=device, subject_id=subject, session_id=session,
                         task_id=tid, label=tr.label, domain=tr.domain, path=str(path),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                except (KeyError, ValueError, TypeError, OSError):
                    continue
            if not valid_session:
                dropped += 1
    finally:
        firebase_admin.delete_app(app)
    fields = ['device_uuid','subject_id','session_id','task_id','label','domain','path','sha256']
    with (output/'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # Private per-device QC beside manifest; public report uses aggregates only.
    private = {device: dict(median_hz=float(np.median(np.array(v)[:,0])),
                total_duration=float(np.array(v)[:,1].sum()), points=int(np.array(v)[:,2].sum()))
                for device,v in quality.items()}
    (output/'quality_by_device.json').write_text(json.dumps(private, indent=2))
    text = f'# V6 ingestion\n\nAccepted tasks: {len(rows)}. Devices: {len(quality)}. Dropped/invalid sessions: {dropped}.\n'
    text += 'Private per-device sampling rate, duration, points: quality_by_device.json beside manifest.\n'
    text += 'Device UUID is a grouping fallback, not proof of unique cross-device human identity.\n'
    Path(report).parent.mkdir(parents=True, exist_ok=True); Path(report).write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    a=p.parse_args(); ingest(json.loads(Path(a.config).read_text()))

if __name__=='__main__': main()
