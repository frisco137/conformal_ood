"""T1.7 -- a nano-PFN whose prior is known exactly by construction.

WHY THIS EXISTS
---------------
T0.1 established that the released `PriorDataset` CANNOT produce the regression
setting its own checkpoint records (`max_classes=0` raises), so rung A is our
reconstruction of the documented pipeline and is called "prior-family", never
"in-family". A model we train ourselves closes that gap: its prior is known
exactly because we wrote it.

It buys four things nothing else can (plan T1.7):
  * in-family holds BY CONSTRUCTION -- no provenance caveat;
  * two registered predictions --
        trained at FIXED sigma^2  -> projected asym should decay toward the
                                     instrument floor as training proceeds;
        trained at MIXED sigma^2  -> it should plateau near the P6 class floor;
  * a training-compute sweep, which turns the audit from a verdict into a
    PROGRESS MEASURE: does structural violation shrink as the model approaches
    the PPD?
  * closure of the one-checkpoint-per-model limitation.

THE ARCHITECTURE, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------------
A PFN is a set function: it maps a labelled context plus unlabelled queries to
predictive distributions, and it must be PERMUTATION-INVARIANT over rows. So:

  * every row becomes one token, from (x_i, y_i) with a mask flag for queries;
  * a plain transformer encoder with NO positional encoding, which makes
    permutation-equivariance exact rather than learned -- and removes the
    positional-leakage confound that E1.4 had to rule out for TabICL;
  * a Gaussian head (mu, log sigma) at query positions, trained by NLL. That is
    a genuine amortised PPD, and it gives A3 a variance channel for free.

TARGET NORMALISATION IS DELIBERATE. The model standardises y on the context rows
and un-standardises its output, exactly as TabICL does. That is what makes N1
apply and lets the SAME projection and the SAME wrapper analysis be used. A model
without it would not be comparable.

Checkpoints record the exact prior configuration in the file, so provenance can
never be lost the way it was for the released checkpoint.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

__all__ = ["NanoPFN", "PFNConfig"]


class PFNConfig:
    def __init__(self, d_model=64, nhead=4, nlayers=4, d_ff=128,
                 max_features=20, dropout=0.0):
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.d_ff = d_ff
        self.max_features = max_features
        self.dropout = dropout

    def asdict(self):
        return dict(d_model=self.d_model, nhead=self.nhead, nlayers=self.nlayers,
                    d_ff=self.d_ff, max_features=self.max_features,
                    dropout=self.dropout)


class NanoPFN(nn.Module):
    """Permutation-invariant set transformer over (x, y) row tokens.

    forward(X, y, mask) -> (mu, log_sigma) at every position.
      X    (B, T, F)  features, zero-padded to max_features
      y    (B, T)     labels; the value at masked positions is ignored
      mask (B, T)     True where the row is a QUERY (label hidden)
    """

    def __init__(self, cfg: PFNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.x_proj = nn.Linear(cfg.max_features, d)
        self.y_proj = nn.Linear(1, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.nlayers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, X, y, mask):
        # label channel: real label for context rows, a learned token for queries
        yt = self.y_proj(y.unsqueeze(-1))
        yt = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(yt), yt)
        h = self.x_proj(X) + yt
        h = self.enc(h)                      # no positional encoding: perm-equivariant
        o = self.head(self.norm(h))
        mu, log_sigma = o[..., 0], o[..., 1].clamp(-7.0, 7.0)
        return mu, log_sigma

    # ------------------------------------------------------------------ sklearn-ish
    @torch.no_grad()
    def predict_np(self, X_ctx, y_ctx, X_query, device="cuda", return_var=False):
        """The audit interface: fit on (X_ctx, y_ctx), predict at X_query.

        Standardises y on the context rows and un-standardises the output, the
        same affine wrapper TabICL wears, so N1 and the projection apply.
        """
        self.eval()
        F = self.cfg.max_features
        Xc = np.asarray(X_ctx, float); yc = np.asarray(y_ctx, float).ravel()
        Xq = np.asarray(X_query, float)
        nc, nq = len(Xc), len(Xq)

        mu_y, sd_y = float(yc.mean()), float(yc.std()) + 1e-8
        yc_s = (yc - mu_y) / sd_y

        def pad(A):
            out = np.zeros((len(A), F))
            out[:, :min(A.shape[1], F)] = A[:, :F]
            return out

        X = np.concatenate([pad(Xc), pad(Xq)], axis=0)[None]
        yv = np.concatenate([yc_s, np.zeros(nq)])[None]
        mk = np.concatenate([np.zeros(nc, bool), np.ones(nq, bool)])[None]

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(yv, dtype=torch.float32, device=device)
        mt = torch.tensor(mk, dtype=torch.bool, device=device)
        mu, ls = self(Xt, yt, mt)
        mu = mu[0, nc:].cpu().numpy() * sd_y + mu_y
        if not return_var:
            return mu
        var = (torch.exp(ls[0, nc:]).cpu().numpy() * sd_y) ** 2
        return mu, var


def gaussian_nll(y, mu, log_sigma):
    return (log_sigma + 0.5 * math.log(2 * math.pi)
            + 0.5 * ((y - mu) / torch.exp(log_sigma)) ** 2)
