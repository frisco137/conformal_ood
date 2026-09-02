"""T1.7 -- train the nano-PFN on the TabICL prior. Two arms, checkpointed for a
compute sweep.

  arm `fixed`   every training context has sigma^2 = 1.0
                REGISTERED: projected asym should DECAY toward the instrument
                floor as training proceeds. Brown applies exactly to this prior,
                so a model approaching its PPD must approach a symmetric PSD
                Jacobian.
  arm `mixed`   sigma^2 ~ log-uniform on [0.25, 4], drawn per context
                REGISTERED: projected asym should PLATEAU near the P6 class
                floor -- the exact posterior mean for a sigma^2-mixing prior is
                itself asymmetric (T3.1), so no amount of training drives it to
                zero.

Checkpoints are written at a geometric ladder of step counts so the audit can be
run as a function of training compute -- that is what turns the audit from a
verdict into a progress measure.

The prior configuration is stored IN the checkpoint. T0.1 found the released
checkpoint carries no prior provenance at all; this one cannot lose it.

Usage:  python t1_7_train_nanopfn.py <arm> [steps]
Writes results/nanopfn_<arm>_step<k>.pt and results/nanopfn_<arm>_train.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from nanopfn import NanoPFN, PFNConfig, gaussian_nll                    # noqa: E402
from prior_family import generate_prior_context                          # noqa: E402

DEVICE = "cuda"
BATCH = 8
N_ROWS = 100
N_QUERY = 30                 # rows whose label is hidden during training
MAX_FEATURES = 20
LR = 3e-4
CKPT_STEPS = [200, 500, 1000, 2000, 4000, 8000, 16000]
SIGMA2_LO, SIGMA2_HI = 0.25, 4.0


def draw_batch(rng, arm, batch=BATCH):
    """One training batch straight from the prior. sigma^2 is OURS and recorded."""
    Xs, ys, ms, s2s = [], [], [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        if arm == "fixed":
            s2 = 1.0
        else:
            s2 = float(np.exp(rng.uniform(np.log(SIGMA2_LO), np.log(SIGMA2_HI))))
        try:
            X, y, f, meta = generate_prior_context(
                seed, rung="A", n=N_ROWS, sigma=np.sqrt(s2))
        except RuntimeError:
            continue
        Xp = np.zeros((N_ROWS, MAX_FEATURES))
        k = min(X.shape[1], MAX_FEATURES)
        Xp[:, :k] = X[:, :k]
        mu, sd = y.mean(), y.std() + 1e-8
        ys.append((y - mu) / sd)
        Xs.append(Xp)
        mk = np.zeros(N_ROWS, bool)
        mk[rng.choice(N_ROWS, N_QUERY, replace=False)] = True
        ms.append(mk)
        s2s.append(s2)
    if not Xs:
        return None
    return (torch.tensor(np.stack(Xs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(ys), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(ms), dtype=torch.bool, device=DEVICE),
            s2s)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    total = int(sys.argv[2]) if len(sys.argv) > 2 else CKPT_STEPS[-1]
    assert arm in ("fixed", "mixed")

    cfg = PFNConfig(max_features=MAX_FEATURES)
    model = NanoPFN(cfg).to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total, pct_start=0.1)
    rng = np.random.default_rng(1234 if arm == "fixed" else 5678)

    prior_cfg = {"arm": arm, "rung": "A (prior-family, Gaussian terminal noise)",
                 "sigma2": 1.0 if arm == "fixed" else [SIGMA2_LO, SIGMA2_HI],
                 "sigma2_law": "fixed" if arm == "fixed" else "log-uniform",
                 "n_rows": N_ROWS, "n_query": N_QUERY,
                 "max_features": MAX_FEATURES,
                 "generator": "experiments/phase_3/src/prior_family.py",
                 "source_prior": "tabicl 2.1.1 MLPSCM/TreeSCM -> Reg2Cls(num_classes=0)"}

    print("=" * 84)
    print(f"T1.7  nano-PFN, arm '{arm}'   {nparam:,} parameters   {total} steps")
    print(f"      sigma^2: {prior_cfg['sigma2']}  ({prior_cfg['sigma2_law']})")
    print("=" * 84, flush=True)

    log, t0, run = [], time.time(), []
    for step in range(1, total + 1):
        b = draw_batch(rng, arm)
        if b is None:
            continue
        X, y, mk, s2s = b
        mu, ls = model(X, y, mk)
        loss = gaussian_nll(y[mk], mu[mk], ls[mk]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss))

        if step % 100 == 0 or step == 1:
            m = float(np.mean(run[-100:]))
            log.append({"step": step, "nll": m, "seconds": time.time() - t0})
            print(f"  step {step:>6}  NLL {m:+.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if step in CKPT_STEPS:
            p = HERE.parent / "results" / f"nanopfn_{arm}_step{step}.pt"
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.asdict(),
                        "prior_config": prior_cfg, "step": step,
                        "nll": float(np.mean(run[-100:]))}, p)
            print(f"    -> checkpoint {p.name}", flush=True)

    json.dump({"arm": arm, "prior_config": prior_cfg, "n_params": nparam,
               "total_steps": total, "ckpt_steps": CKPT_STEPS, "log": log},
              open(HERE.parent / "results" / f"nanopfn_{arm}_train.json", "w"),
              indent=2)
    print(f"\n  done in {time.time()-t0:.0f}s, final NLL {np.mean(run[-100:]):+.5f}")


if __name__ == "__main__":
    main()
