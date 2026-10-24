"""Fold-scoped masked autoencoder. Never pretrain on future val/test subjects."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.models.v6_tremorformer import TremorFormer

class MaskedAutoencoder(nn.Module):
    def __init__(self,backbone,patch=16,mask_ratio=.5):
        super().__init__()
        if 512%patch or not 0<mask_ratio<1: raise ValueError('Invalid patch/mask ratio')
        self.backbone=backbone; self.patch=patch; self.mask_ratio=mask_ratio
        d=backbone.config['d_model']; c=backbone.config['in_channels']
        layer=nn.TransformerEncoderLayer(d,8 if d%8==0 else 1,dim_feedforward=d*2,
                                         dropout=.1,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.reconstruct=nn.Linear(d,c)

    def forward(self,x,domain):
        b,c,l=x.shape; n=l//self.patch
        order=torch.rand(b,n,device=x.device).argsort(dim=1)
        patch_mask=torch.zeros(b,n,device=x.device)
        patch_mask.scatter_(1,order[:,:int(n*self.mask_ratio)],1)
        mask=patch_mask.repeat_interleave(self.patch,dim=1)
        # Mask before CNN/attention: ground-truth hidden patches cannot leak in.
        tokens=self.backbone.encode(x*(1-mask[:,None,:]),domain)
        decoded=self.reconstruct(self.decoder(tokens)).transpose(1,2)
        reconstructed=F.interpolate(decoded,size=l,mode='linear',align_corners=False)
        loss=((reconstructed-x).square()*mask[:,None,:]).sum()/(mask.sum()*c).clamp_min(1)
        return loss


def pretrain(backbone,loader,allowed_subjects,output,epochs=100,patience=10,device='cpu',seed=42):
    """Loader dataset must expose its actual parent IDs; no trust in filenames."""
    allowed=set(allowed_subjects)
    if not set(loader.dataset.subjects)<=allowed:
        raise ValueError('SSL dataset includes a non-training parent')
    torch.manual_seed(seed); model=MaskedAutoencoder(backbone).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    best=float('inf'); stale=0
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train(); losses=[]
        for batch in loader:
            x=batch['x'].to(device); domain=batch['domain'].to(device)
            opt.zero_grad(set_to_none=True)
            loss=model(x,domain); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); losses.append(float(loss.detach()))
        score=float(np.mean(losses)) if losses else float('inf')
        if score<best-1e-6:
            best=score; stale=0
            torch.save(dict(model_state_dict=backbone.state_dict(),model_config=backbone.config,
                            training_subjects=sorted(allowed),epoch=epoch,ssl_loss=score,seed=seed),path)
        else: stale+=1
        if stale>=patience: break
    if not path.exists(): raise RuntimeError('No valid SSL batches')
    backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model_state_dict'])
    return backbone
