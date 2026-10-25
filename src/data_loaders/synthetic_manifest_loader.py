"""Lazy, checksum-verified TRAIN-ONLY generator output. Never trusts pseudo labels.

Both donor+recipient parents must be in training. Subject grouping on recipient
alone is NOT enough. No generated record is allowed in clinical validation/test.
"""
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
from src.augmentation.tremor_injector import eligible_rows,materialize
from src.preprocessing.v6_pipeline import transform,BASE_CONTRACT
from src.data_loaders.schema import DOMAINS,TASKS

class SyntheticManifestDataset(Dataset):
    def __init__(self,manifest,train_subjects,contract=BASE_CONTRACT,allow_mock=False):
        self.rows=eligible_rows(manifest,train_subjects,allow_mock)
        self.subjects=sorted({r['subject_id'] for r in self.rows})
        self.provenance_subjects=sorted({r[k] for r in self.rows for k in ('subject_id','donor_subject_id')})
        self.contract=contract; self.index=[]
        for i,row in enumerate(self.rows):
            duration=float(row['duration_seconds']); n=int(np.floor(duration*100))+1
            if n<512: continue
            starts=list(range(0,n-511,128))
            if starts[-1]!=n-512: starts.append(n-512)
            self.index.extend((i,j) for j in range(len(starts)))
    def __len__(self): return len(self.index)
    @lru_cache(maxsize=8)
    def _windows(self,i): return transform(materialize(self.rows[i]),contract=self.contract)[0]
    def __getitem__(self,k):
        i,j=self.index[k]; row=self.rows[i]; x=self._windows(i)[j]
        return dict(x=torch.from_numpy(x.copy()),label=torch.tensor(1.,dtype=torch.float32),
                    domain=torch.tensor(DOMAINS.index(row['domain'])),task=torch.tensor(-1),age=torch.tensor(-1),
                    updrs=torch.tensor(float('nan')),synthetic=torch.tensor(True),subject=row['subject_id'])

class CombinedTrainingDataset(Dataset):
    def __init__(self,*datasets):
        self.datasets=datasets; self.ends=np.cumsum([len(d) for d in datasets])
        self.subjects=sorted({p for d in datasets for p in d.subjects})
        self.provenance_subjects=sorted({p for d in datasets for p in getattr(d,'provenance_subjects',d.subjects)})
    def __len__(self): return int(self.ends[-1])
    def __getitem__(self,k):
        i=int(np.searchsorted(self.ends,k,side='right')); offset=0 if i==0 else self.ends[i-1]
        return self.datasets[i][int(k-offset)]
