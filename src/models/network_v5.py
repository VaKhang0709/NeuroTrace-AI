"""Checkpoint-compatible V5 CNN/BiLSTM with adversarial device head."""
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self, lam: float = 0.3) -> None:
        super().__init__()
        self.lam = float(lam)

    def forward(self, x: torch.Tensor, lam: Optional[float] = None) -> torch.Tensor:
        active_lam = self.lam if lam is None else float(lam)
        return GradientReversalFn.apply(x, active_lam)


class NeuroTraceDualHeadV5(nn.Module):
    """Dual-head 1D-CNN + BiLSTM diagnostic network with Gradient Reversal Layer.
    `forward(x)` returns RAW LOGITS `(B,)` strictly for `nn.BCEWithLogitsLoss`
    and clean standard-operator ONNX export (`opset_version=17`, `dynamo=False`).
    """

    def __init__(
        self,
        in_channels: int = 5,
        conv1: int = 32,
        conv2: int = 64,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        emb_dim: int = 4,
        dropout: float = 0.25,
        grl_lambda: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv1 = int(conv1)
        self.conv2 = int(conv2)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.emb_dim = int(emb_dim)
        self.dropout = float(dropout)
        self.grl_lambda = float(grl_lambda)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 4, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.clinical_classifier = nn.Linear(emb_dim, 1)
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.device_classifier = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(x)).transpose(1, 2)
        out, _ = self.lstm(h)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=-1)
        return self.feature_proj(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RAW clinical logits `(B,)` — NO sigmoid inside `forward()`."""
        feat = self.extract_features(x)
        return self.clinical_classifier(feat).squeeze(-1)

    def forward_dual(self, x: torch.Tensor, lam: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        clin_logits = self.clinical_classifier(feat).squeeze(-1)
        dev_logits = self.device_classifier(self.grl(feat, lam=lam)).squeeze(-1)
        return clin_logits, dev_logits, feat
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


