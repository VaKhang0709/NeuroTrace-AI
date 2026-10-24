"""TremorFormer V6 research architecture. No pretrained/performance claims.

6 Conformer blocks, d_model=256, FF expansion=8 (15.78M parameters, verified in tests).
Clinical head returns RAW logits for BCEWithLogitsLoss; sigmoid at inference only.
Device-conditioned FiLM is an ablatable hypothesis, not proof of invariance.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.network_v5 import GradientReversalLayer

class FeedForward(nn.Module):
    def __init__(self,d,expansion,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*expansion),nn.SiLU(),
                               nn.Dropout(dropout),nn.Linear(d*expansion,d),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self,d=256,heads=8,expansion=8,dropout=.1):
        super().__init__()
        self.ff1=FeedForward(d,expansion,dropout); self.ff2=FeedForward(d,expansion,dropout)
        self.attn_norm=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.conv_norm=nn.LayerNorm(d)
        self.conv=nn.Sequential(nn.Conv1d(d,2*d,1),nn.GLU(dim=1),
             nn.Conv1d(d,d,31,padding=15,groups=d),nn.GroupNorm(1,d),nn.SiLU(),
             nn.Conv1d(d,d,1),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(dropout)
        self.film=nn.Linear(32,2*d)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
    def forward(self,x,device):
        gamma,beta=self.film(device).chunk(2,-1)
        x=x*(1+gamma[:,None,:])+beta[:,None,:]
        x=x+.5*self.ff1(x)
        q=self.attn_norm(x)
        x=x+self.drop(self.attn(q,q,q,need_weights=False)[0])
        x=x+self.conv(self.conv_norm(x).transpose(1,2)).transpose(1,2)
        return self.norm(x+.5*self.ff2(x))

class TremorFormer(nn.Module):
    def __init__(self,in_channels=19,d_model=256,blocks=6,heads=8,expansion=8,
                 dropout=.1,age_buckets=5,grl_lambda=.1,device_film=True):
        super().__init__()
        if d_model%4 or d_model%heads: raise ValueError('d_model must divide four branches and heads')
        self.config=dict(in_channels=in_channels,d_model=d_model,blocks=blocks,heads=heads,
                         expansion=expansion,dropout=dropout,age_buckets=age_buckets,
                         grl_lambda=grl_lambda,device_film=device_film)
        self.stems=nn.ModuleList([nn.Conv1d(in_channels,d_model//4,k,stride=4,padding=k//2) for k in (3,7,15,31)])
        self.projection=nn.Sequential(nn.Linear(d_model,d_model),nn.LayerNorm(d_model),nn.SiLU())
        self.position=nn.Parameter(torch.zeros(1,128,d_model)); nn.init.normal_(self.position,std=.02)
        position=torch.arange(128).float()[:,None]
        divisor=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        sinusoid=torch.zeros(128,d_model)
        sinusoid[:,0::2]=torch.sin(position*divisor); sinusoid[:,1::2]=torch.cos(position*divisor)
        self.register_buffer('sinusoid',sinusoid[None],persistent=False)
        self.device_embedding=nn.Linear(5,32)
        self.blocks=nn.ModuleList([ConformerBlock(d_model,heads,expansion,dropout) for _ in range(blocks)])
        self.pool=nn.Sequential(nn.Linear(d_model,d_model//2),nn.Tanh(),nn.Linear(d_model//2,1))
        self.pd_head=nn.Linear(d_model,1); self.updrs_head=nn.Linear(d_model,1)
        self.task_head=nn.Linear(d_model,3); self.age_head=nn.Linear(d_model,age_buckets)
        self.grl=GradientReversalLayer(grl_lambda)
        self.device_head=nn.Sequential(nn.Linear(d_model,128),nn.SiLU(),nn.Linear(128,5))

    def encode(self,x,domain):
        h=self.projection(torch.cat([stem(x) for stem in self.stems],dim=1).transpose(1,2))
        h=h+self.position+self.sinusoid
        onehot=F.one_hot(domain.to(torch.int64),num_classes=5).to(x.dtype)
        device=self.device_embedding(onehot)
        if not self.config['device_film']: device=torch.zeros_like(device)
        for block in self.blocks: h=block(h,device)
        return h

    def extract_features(self,x,domain):
        h=self.encode(x,domain)
        return (h*torch.softmax(self.pool(h),dim=1)).sum(dim=1)

    def forward(self,x,domain):
        h=self.extract_features(x,domain)
        return dict(pd_logits=self.pd_head(h).squeeze(-1),updrs=self.updrs_head(h).squeeze(-1),
                    device_logits=self.device_head(self.grl(h)),task_logits=self.task_head(h),
                    age_logits=self.age_head(h),features=h)

    @torch.no_grad()
    def predict_proba(self,x,domain):
        self.eval()
        return torch.sigmoid(self(x,domain)['pd_logits'])
