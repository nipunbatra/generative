"""The 2-D smiley: data, training of the unconditional and part-conditional denoisers, and their recordings.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/toy2d.py train      # trains both models (a few minutes on CPU), writes models/toy-*.pt
    $PY scripts/toy2d.py preview    # a quick PNG of samples from both, for inspection
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import T, EMA, ToyDenoiser, ddpm_step, schedule, training_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["outline", "eyes", "mouth"]           # label 3 = no label
WEIGHTS = [0.5, 0.2, 0.3]


def smiley(n, seed=0):
    """n points of a smiley face and their part labels: outline 50%, eyes 20%, mouth 30%."""
    g = np.random.default_rng(seed)
    part = g.choice(3, size=n, p=WEIGHTS)
    x = np.zeros((n, 2))
    k = part == 0
    a = g.uniform(0, 2 * np.pi, k.sum())
    r = 1 + 0.025 * g.standard_normal(k.sum())
    x[k] = np.c_[r * np.cos(a), r * np.sin(a)]
    k = part == 1
    side = g.choice([-1, 1], size=k.sum())
    x[k] = np.c_[0.38 * side + 0.06 * g.standard_normal(k.sum()), 0.32 + 0.09 * g.standard_normal(k.sum())]
    k = part == 2
    a = g.uniform(np.pi + 0.45, 2 * np.pi - 0.45, k.sum())
    r = 0.58 + 0.025 * g.standard_normal(k.sum())
    x[k] = np.c_[r * np.cos(a), 0.08 + r * np.sin(a)]
    return x.astype(np.float32), part


def train(name, n_parts, steps=40000, batch=2048, lr=2e-3, seed=667, snapshots=(), p_drop=0.1):
    """Train on fresh smiley draws every step (an unlimited dataset); return the EMA model and a record."""
    torch.manual_seed(seed)
    model = ToyDenoiser(n_parts=n_parts)
    sched = schedule()
    ab = sched[2]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_at = lambda s: 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * s / steps))
    ema = EMA(model, 0.999)
    data, labels = smiley(200_000, seed=seed)
    data, labels = torch.from_numpy(data), torch.from_numpy(labels).long()
    hist, snaps, run, t0 = [], {}, 0.0, time.time()
    if 0 in snapshots:
        snaps[0] = {k: v.clone() for k, v in model.state_dict().items()}
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        ix = torch.randint(len(data), (batch,))
        c = labels[ix] if n_parts else None
        loss = training_step(model, data[ix], ab, opt, c=c, null=n_parts, p_drop=p_drop)
        ema.update(model)
        run = loss if step == 1 else 0.98 * run + 0.02 * loss
        if step == 1 or step % 100 == 0:
            hist.append([step, round(loss, 5), round(run, 5)])
        if step in snapshots:
            snaps[step] = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 5000 == 0:
            print(f"{name} step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(ema.shadow)
    rec = dict(name=name, steps=steps, batch=batch, lr=lr, seed=seed, ema=0.999, p_drop=p_drop if n_parts else None, params=sum(p.numel() for p in model.parameters()),
               seconds=round(time.time() - t0, 1), torch=torch.__version__, history=hist)
    return model, rec, snaps


@torch.no_grad()
def sample(model, n, seed=0, c=None, w=0.0, keep=()):
    """DDPM ancestral sampling of n points; with c, classifier-free guidance of strength w. keep: steps to record."""
    g = torch.Generator().manual_seed(seed)
    sched = schedule()
    x = torch.randn(n, 2, generator=g)
    kept = {T: x.clone()} if T in keep else {}
    for t in range(T, 0, -1):
        tt = torch.full((n,), t)
        if c is None:
            eps = model(x, tt)
        else:
            null = torch.full((n,), model.n_parts)
            e_c, e_u = model(x, tt, torch.full((n,), c)), model(x, tt, null)
            eps = e_u + w * (e_c - e_u) if w != "uncond" else e_u
        z = torch.randn(n, 2, generator=g)
        x = ddpm_step(eps, x, t, sched, z=z)
        if t - 1 in keep:
            kept[t - 1] = x.clone()
    return x, kept


def preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = torch.load(ROOT / "models" / "toy-uncond.pt")
    m = ToyDenoiser(); m.load_state_dict(u["model"]); m.eval()
    cnd = torch.load(ROOT / "models" / "toy-cond.pt")
    mc = ToyDenoiser(n_parts=3); mc.load_state_dict(cnd["model"]); mc.eval()
    data, _ = smiley(2000, seed=1)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax[0, 0].scatter(*data.T, s=2); ax[0, 0].set_title("data")
    x, _ = sample(m, 2000, seed=3)
    ax[0, 1].scatter(*x.numpy().T, s=2); ax[0, 1].set_title("uncond samples")
    for j, (c, w) in enumerate([(2, 0.0), (2, 1.0), (2, 3.0), (1, 1.0), (1, 4.0)]):
        x, _ = sample(mc, 1500, seed=4, c=c, w=w)
        a = ax.flat[2 + j]
        a.scatter(*x.numpy().T, s=2); a.set_title(f"part {PARTS[c]} w={w}")
    x, _ = sample(mc, 2000, seed=5, c=0, w="uncond")
    ax[1, 3].scatter(*x.numpy().T, s=2); ax[1, 3].set_title("cond model, no label")
    for a in ax.flat:
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal")
    fig.savefig(ROOT / "qa" / "toy-preview.png", dpi=70)
    print("wrote qa/toy-preview.png")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "train":
        (ROOT / "models").mkdir(exist_ok=True)
        snaps_at = (0, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000)
        if "cond" not in sys.argv[2:]:
            m, rec, snaps = train("toy-uncond", 0, snapshots=snaps_at)
            torch.save(dict(model=m.state_dict(), snapshots=snaps), ROOT / "models" / "toy-uncond.pt")
            (ROOT / "models" / "toy-uncond-log.json").write_text(json.dumps(rec))
        m, rec, _ = train("toy-cond", 3, steps=60000, p_drop=0.3)
        torch.save(dict(model=m.state_dict()), ROOT / "models" / "toy-cond.pt")
        (ROOT / "models" / "toy-cond-log.json").write_text(json.dumps(rec))
    elif what == "preview":
        preview()
