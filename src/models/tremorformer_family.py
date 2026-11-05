"""One architecture, three presets. PRO state-dict layout is unchanged.

All presets consume the EXISTING v6-dwt19-v1 feature tensor. No tier-specific
normalization, thresholds or new frontend feature extraction are introduced.
"""
import torch
from torch import nn
from src.models.v6_tremorformer import TremorFormer as OriginalTremorFormer

CONTRACT='v6-dwt19-v1'
PRESETS={
    'nano':dict(d_model=128,blocks=2,heads=4,expansion=3),
    'base':dict(d_model=192,blocks=4,heads=6,expansion=4),
    'pro':dict(d_model=256,blocks=6,heads=8,expansion=8),
}

class TremorFormer(OriginalTremorFormer):
    """Same FiLM/Conformer/GRL heads; NANO uses separable multi-scale stems."""
    def __init__(self,tier='base',**kwargs):
        tier=tier.lower()
        if tier not in PRESETS: raise ValueError(f'Unknown tier: {tier}')
        config={**PRESETS[tier],**kwargs}
        if config.get('in_channels',19)!=19: raise ValueError('Family contract requires 19 channels')
        super().__init__(**config)
        if tier=='nano':
            self.stems=nn.ModuleList([nn.Sequential(
                nn.Conv1d(19,19,k,stride=4,padding=k//2,groups=19),
                nn.Conv1d(19,self.config['d_model']//4,1)) for k in (3,7,15,31)])
        self.tier=tier; self.feature_contract=CONTRACT


def load_student(path,device='cpu',require_complete=True):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('family_tier') not in ('nano','base') or ck.get('feature_contract')!=CONTRACT:
        raise ValueError('Not a DWT19 student checkpoint')
    if require_complete and (not ck.get('complete') or ck.get('epochs_completed')!=ck.get('epochs_requested')):
        raise ValueError('Distillation incomplete; do not promote a budget-aborted checkpoint')
    model=TremorFormer(ck['family_tier'],**ck['model_config'])
    model.load_state_dict(ck['model_state_dict'],strict=True)
    return model.to(device).eval(),ck


def parameter_counts():
    return {tier:sum(p.numel() for p in TremorFormer(tier).parameters()) for tier in PRESETS}

if __name__=='__main__':
    for tier,n in parameter_counts().items(): print(f'{tier.upper()}: {n:,} parameters')
