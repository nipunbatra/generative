"""Record everything the slides show about the 2-D smiley, from models/toy-*.pt.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/toy_record.py       # writes data/toy.json and data/toy-weights.json (about two minutes)

The browser runs the same networks from data/toy-weights.json: widgets.js reimplements ToyDenoiser's
forward pass, and scripts/check-science.mjs compares its outputs with PyTorch's on fixed inputs.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import T, EMA, ToyDenoiser, ddim_step, ddim_times, ddpm_step, schedule, training_step, x0_from_eps  # noqa: E402
from toy2d import PARTS, smiley  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()


def load(name, n_parts=0):
    ck = torch.load(ROOT / "models" / f"{name}.pt")
    m = ToyDenoiser(n_parts=n_parts)
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def export(m):
    """Weights as nested lists, layer by layer: W is [out][in], b is [out]."""
    layers = [mod for mod in m.net if isinstance(mod, torch.nn.Linear)]
    return {"tdim": m.tdim, "n_parts": m.n_parts,
            "layers": [{"W": np.round(l.weight.detach().numpy().astype(float), 6).tolist(), "b": np.round(l.bias.detach().numpy().astype(float), 6).tolist()} for l in layers]}


@torch.no_grad()
def run_ddpm(m, xT, seed, keep_every=10, eps_fn=None):
    sched, g = schedule(), torch.Generator().manual_seed(seed)
    x, frames = xT.clone(), [xT.clone()]
    for t in range(T, 0, -1):
        eps = eps_fn(x, t) if eps_fn else m(x, torch.full((len(x),), t))
        x = ddpm_step(eps, x, t, sched, z=torch.randn(x.shape, generator=g))
        if (t - 1) % keep_every == 0:
            frames.append(x.clone())
    return x, torch.stack(frames)


@torch.no_grad()
def run_ddim(m, xT, steps):
    sched, x, frames = schedule(), xT.clone(), [xT.clone()]
    ts = ddim_times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        x = ddim_step(m(x, torch.full((len(x),), t)), x, t, tn, sched)
        frames.append(x.clone())
    return x, torch.stack(frames), ts


def nearest(x, ref):
    return torch.cdist(x, ref).min(1).values


def main():
    t0 = time.time()
    out = {}
    unc, ck = load("toy-uncond")
    cnd, _ = load("toy-cond", 3)
    weights = {"uncond": export(unc), "cond": export(cnd), "parts": PARTS}
    # fixed inputs and PyTorch's outputs, for checking the browser's forward pass
    xs = torch.tensor([[0.0, 0.0], [0.5, -0.5], [-1.2, 0.3]])
    ts = torch.tensor([1, 250, 1000])
    weights["check"] = {"x": xs.tolist(), "t": ts.tolist(), "uncond": unc(xs, ts).tolist(),
                        "cond_mouth": cnd(xs, ts, torch.tensor([2, 2, 2])).tolist(), "cond_none": cnd(xs, ts, torch.tensor([3, 3, 3])).tolist()}
    (ROOT / "data" / "toy-weights.json").write_text(json.dumps(weights, separators=(",", ":")))

    data, part = smiley(1500, seed=1)
    out["data"] = {"x": r3(data), "part": part.tolist(), "parts": PARTS}
    for name in ("toy-uncond", "toy-cond"):
        log = json.loads((ROOT / "models" / f"{name}-log.json").read_text())
        out[name] = {k: v for k, v in log.items() if k != "history"}
    log = json.loads((ROOT / "models" / "toy-uncond-log.json").read_text())
    out["history"] = log["history"][::4]

    # the training playground: the network at each snapshot, its arrows at t = 100 and its samples
    sched = schedule()
    g = np.linspace(-1.4, 1.4, 15)
    grid = torch.tensor([[a, b] for b in g[::-1] for a in g], dtype=torch.float32)
    t_show = 100
    xT = torch.randn(300, 2, generator=torch.Generator().manual_seed(7))
    snaps = []
    for step, state in sorted(ck["snapshots"].items()):
        m = ToyDenoiser()
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            eps = m(grid, torch.full((len(grid),), t_show))
        x0h = x0_from_eps(grid, eps, sched[2][t_show])
        xs_, _ = run_ddpm(m, xT, seed=8, keep_every=1000)
        snaps.append({"step": step, "x0": r3(x0h), "samples": r3(xs_)})
    out["playground"] = {"t": t_show, "alpha_bar": round(float(sched[2][t_show]), 4), "grid": r3(grid), "snaps": snaps}
    print("playground", round(time.time() - t0), "s", flush=True)

    # sampling paths: 300 particles, every 10th step; and DDIM with 50 steps from the same starts
    xT = torch.randn(300, 2, generator=torch.Generator().manual_seed(11))
    final, frames = run_ddpm(unc, xT, seed=12, keep_every=10)
    fd, framesd, tsd = run_ddim(unc, xT, 50)
    ref, _ = smiley(20000, seed=99)
    ref = torch.from_numpy(ref)
    out["paths"] = {"ddpm_t": list(range(T, -1, -10)), "ddpm": r3(frames.permute(1, 0, 2)), "ddim_t": tsd, "ddim": r3(framesd.permute(1, 0, 2)),
                    "ddpm_to_data": round(float(nearest(final, ref).mean()), 4), "ddim_to_data": round(float(nearest(fd, ref).mean()), 4),
                    "data_to_data": round(float(nearest(torch.from_numpy(smiley(300, seed=5)[0]), ref).mean()), 4)}
    # without the fresh noise: the DDPM mean only
    with torch.no_grad():
        x = xT.clone()
        for t in range(T, 0, -1):
            x = ddpm_step(unc(x, torch.full((len(x),), t)), x, t, sched, sigma=None)
    out["paths"]["mean_only"] = r3(x)
    out["paths"]["mean_only_to_data"] = round(float(nearest(x, ref).mean()), 4)
    print("paths", round(time.time() - t0), "s", flush=True)

    # classifier-free guidance towards one part of the face
    guided = {}
    xT = torch.randn(400, 2, generator=torch.Generator().manual_seed(21))
    for c in (1, 2):
        for w in (0, 1, 2, 3, 5):
            def fn(x, t, c=c, w=w):
                tt = torch.full((len(x),), t)
                eu = cnd(x, tt, torch.full((len(x),), 3))
                return eu if w == 0 else eu + w * (cnd(x, tt, torch.full((len(x),), c)) - eu)
            xs_, _ = run_ddpm(cnd, xT, seed=22, keep_every=1000, eps_fn=fn)
            spread = float(xs_.std(0).norm())
            guided[f"{PARTS[c]}-{w}"] = {"x": r3(xs_), "spread": round(spread, 4)}
    out["guided"] = guided
    print("guided", round(time.time() - t0), "s", flush=True)

    # does it copy? 40 training points: the exact ideal denoiser, and a network trained on just those 40
    pts, ppart = smiley(40, seed=3)
    P = torch.from_numpy(pts)
    ab = sched[2]

    def ideal(x, t):
        a = ab[t]
        d2 = torch.cdist(x, a.sqrt() * P) ** 2
        w = torch.softmax(-d2 / (2 * (1 - a)), 1)
        return (x - a.sqrt() * (w @ P)) / (1 - a).sqrt()

    xT = torch.randn(400, 2, generator=torch.Generator().manual_seed(31))
    xi, _ = run_ddpm(None, xT, seed=32, keep_every=1000, eps_fn=ideal)
    torch.manual_seed(667)
    small = ToyDenoiser()
    opt = torch.optim.Adam(small.parameters(), lr=2e-3)
    ema = EMA(small)
    for step in range(1, 20001):
        ix = torch.randint(40, (512,))
        training_step(small, P[ix], ab, opt)
        ema.update(small)
    small.load_state_dict(ema.shadow)
    small.eval()
    xn, _ = run_ddpm(small, xT, seed=32, keep_every=1000)
    dfull, _ = smiley(20000, seed=99)
    out["copy"] = {"train": r3(pts), "ideal": r3(xi), "net": r3(xn),
                   "ideal_to_train": round(float(nearest(xi, P).mean()), 4), "net_to_train": round(float(nearest(xn, P).mean()), 4),
                   "net_to_shape": round(float(nearest(xn, torch.from_numpy(dfull)).mean()), 4),
                   "ideal_to_shape": round(float(nearest(xi, torch.from_numpy(dfull)).mean()), 4),
                   "train_spacing": round(float(torch.cdist(P, P).add(torch.eye(40) * 9).min(1).values.mean()), 4)}
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "data" / "toy.json").write_text(json.dumps(out, separators=(",", ":")))
    print("done", out["seconds"], "s", json.dumps({k: out["copy"][k] for k in out["copy"] if k.endswith(("train", "shape", "spacing")) and not isinstance(out["copy"][k], list)}),
          json.dumps({k: v for k, v in out["paths"].items() if not isinstance(v, list)}), flush=True)


if __name__ == "__main__":
    main()
