"""Every MNIST measurement the slides show, from the saved checkpoints in models/.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/mnist.py            # writes data/mnist.json and figures/mnist-*.png (a few minutes on MPS)

Images are greyscale MNIST scaled to [-1, 1]: paper -1, ink +1. The held 3 is official test image 90,
the 3 of Lectures 2 and 3. Pictures are display copies clipped to [-1, 1] (noise images to [-2.5, 2.5]),
drawn with ink dark on light paper. MNIST is read from the local IDX files of the GM-II rebuild.
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import (T, Classifier, ConvAE, LatentDenoiser, TinyUNet, ddim_step, ddim_times, ddpm_step,  # noqa: E402
                       schedule, x0_from_eps)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MNIST_DIR", "/Users/nipun/git/dl-teaching/generative-modelling-ii/work/data"))
DEV = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
FIG = ROOT / "figures"
HELD = 90
SEED = 2026
r3 = lambda v: float(round(float(v), 4))


def load_mnist(train=True):
    name = "train" if train else "t10k"
    x = np.frombuffer(gzip.decompress((DATA / f"{name}-images-idx3-ubyte.gz").read_bytes()), np.uint8, offset=16)
    y = np.frombuffer(gzip.decompress((DATA / f"{name}-labels-idx1-ubyte.gz").read_bytes()), np.uint8, offset=8)
    return torch.from_numpy(x.copy().reshape(-1, 1, 28, 28)).float() / 127.5 - 1, torch.from_numpy(y.copy()).long()


def load(cls, name, **kw):
    m = cls(**kw)
    ck = torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu")
    m.load_state_dict(ck["model"])
    return m.to(DEV).eval(), ck


def to_u8(x, lo=-1.0, hi=1.0):
    """Ink dark on light paper: +1 -> 20, -1 -> 250."""
    v = ((x.clamp(lo, hi) - lo) / (hi - lo)).cpu().numpy()
    return (250 - v * 230).round().astype(np.uint8)


def strip(frames, path, lo=-1.0, hi=1.0, gap=0):
    """A row of 28x28 frames as one PNG (the widgets cut it into frames)."""
    frames = [to_u8(f.reshape(28, 28), lo, hi) for f in frames]
    if gap:
        pad = np.full((28, gap), 255, np.uint8)
        frames = [x for f in frames for x in (f, pad)][:-1]
    Image.fromarray(np.concatenate(frames, 1)).save(FIG / path, optimize=True)


def grid(images, rows, path, lo=-1.0, hi=1.0, gap=2):
    cols = len(images) // rows
    canvas = np.full((rows * 28 + (rows - 1) * gap, cols * 28 + (cols - 1) * gap), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * (28 + gap):r * (28 + gap) + 28, c * (28 + gap):c * (28 + gap) + 28] = to_u8(im.reshape(28, 28), lo, hi)
    Image.fromarray(canvas).save(FIG / path, optimize=True)


@torch.no_grad()
def eps_of(model, x, t, c=None):
    tt = torch.full((x.shape[0],), t, device=DEV)
    return model(x, tt) if c is None else model(x, tt, c)


@torch.no_grad()
def ddpm(model, xT, seed, keep=(), sigma="posterior", eps_fn=None):
    """Ancestral DDPM sampling from the given x_T; the fresh noise comes from its own seeded generator."""
    sched = schedule(device=DEV)
    g = torch.Generator().manual_seed(seed)
    x, rec = xT.clone(), {}
    for t in range(T, 0, -1):
        eps = eps_fn(x, t) if eps_fn else eps_of(model, x, t)
        if t in keep:
            rec[t] = (x.cpu().clone(), x0_from_eps(x, eps, sched[2][t]).cpu().clone(), eps.cpu().clone())
        z = torch.randn(x.shape, generator=g).to(DEV)
        x = ddpm_step(eps, x, t, sched, z=z, sigma=sigma)
    rec[0] = (x.cpu().clone(), x.cpu().clone(), torch.zeros_like(x).cpu())
    return x.cpu(), rec


@torch.no_grad()
def ddim(model, xT, steps, eps_fn=None):
    sched = schedule(device=DEV)
    ts, x = ddim_times(steps), xT.clone()
    for t, tn in zip(ts[:-1], ts[1:]):
        eps = eps_fn(x, t) if eps_fn else eps_of(model, x, t)
        x = ddim_step(eps, x, t, tn, sched)
    return x.cpu()


def noise(n, seed, shape=(1, 28, 28)):
    return torch.randn((n, *shape), generator=torch.Generator().manual_seed(seed)).to(DEV)


def main():
    FIG.mkdir(exist_ok=True)
    out, t0 = {"device": DEV, "torch": torch.__version__, "held_test_index": HELD}, time.time()
    xtr, ytr = load_mnist(True)
    xte, yte = load_mnist(False)
    assert int(yte[HELD]) == 3
    x0 = xte[HELD:HELD + 1].to(DEV)
    sched = schedule(device=DEV)
    ab = sched[2]
    net, ck = load(TinyUNet, "mnist-ddpm")
    out["model"] = {"params": sum(p.numel() for p in net.parameters()), "steps": ck["steps"], "batch": ck["batch"]}

    # ---- the held 3 and one fixed noise draw: the slides' forward film is computed live from these
    eps = noise(1, SEED)
    out["held"] = {"x0": [r3(v) for v in x0.flatten().tolist()], "eps": [r3(v) for v in eps.flatten().tolist()]}

    # ---- what the network believes at each noise level (same x0 and eps as the film)
    film_t = [1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100, 50, 25, 1]
    xs, e_hats, x0s, rows = [], [], [], []
    pix = (xtr > 0).float().mean(0, keepdim=True).to(DEV)          # per-pixel ink frequency, for "one pixel alone"
    alone = []
    for t in film_t:
        a = ab[t]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        e = eps_of(net, xt, t)
        xh = x0_from_eps(xt, e, a)
        # one pixel alone: the exact posterior mean under "ink (+1) with prob pi, else paper (-1)", each pixel by itself
        s2 = 1 - a
        li = torch.exp(-(xt - a.sqrt()) ** 2 / (2 * s2)) * pix
        lp = torch.exp(-(xt + a.sqrt()) ** 2 / (2 * s2)) * (1 - pix)
        alone.append(((li - lp) / (li + lp + 1e-30)).cpu())
        xs.append(xt.cpu()); e_hats.append(e.cpu()); x0s.append(xh.cpu())
        rows.append({"t": t, "alpha_bar": r3(a), "eps_mse": r3(((e - eps) ** 2).mean()), "x0_mse": r3(((xh - x0) ** 2).mean()),
                     "alone_mse": r3(((alone[-1].to(DEV) - x0) ** 2).mean())})
    strip(xs, "mnist-film-xt.png")
    strip(e_hats, "mnist-film-eps.png", -2.5, 2.5)
    strip(x0s, "mnist-film-x0.png")
    strip(alone, "mnist-film-alone.png")
    strip([eps.cpu()], "mnist-eps.png", -2.5, 2.5)
    out["film"] = {"t": film_t, "rows": rows}
    print("film", round(time.time() - t0), "s", flush=True)

    # ---- the loss by noise level, on 512 test images (it is not zero, and depends on t)
    g = torch.Generator().manual_seed(5)
    xb = xte[:512].to(DEV)
    by_t = []
    for t in [1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        e = torch.randn(xb.shape, generator=g).to(DEV)
        xt = ab[t].sqrt() * xb + (1 - ab[t]).sqrt() * e
        by_t.append([t, r3(((eps_of(net, xt, t) - e) ** 2).mean())])
    out["loss_by_t"] = by_t

    # ---- one reverse step on an image, at t = 600
    t = 600
    xT = noise(1, 11)
    _, rec = ddpm(net, xT, seed=12, keep=(t,))
    xt, xh, e = rec[t]
    xt, xh, e = xt.to(DEV), xh.to(DEV), e.to(DEV)
    beta, alpha, _, var = sched
    mean = (xt - beta[t] / (1 - ab[t]).sqrt() * e) / alpha[t].sqrt()
    z = noise(1, 13)
    nxt = mean + var[t].sqrt() * z
    strip([xt.cpu(), e.cpu(), xh.cpu(), mean.cpu(), nxt.cpu()], "mnist-onestep.png", gap=0)
    for name, im, lo, hi in [("xt", xt, -1, 1), ("eps", e, -2.5, 2.5), ("x0", xh, -1, 1), ("mean", mean, -1, 1), ("next", nxt, -1, 1)]:
        strip([im.cpu()], f"mnist-onestep-{name}.png", lo, hi)
    out["onestep"] = {"t": t, "beta": r3(beta[t]), "alpha_bar": r3(ab[t]), "sigma": r3(var[t].sqrt())}

    # ---- trajectories: 8 seeds, x_t and the clean guess at 21 levels
    keep = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 1]
    tic = time.time()
    xT = noise(8, 21)
    final, rec = ddpm(net, xT, seed=22, keep=keep)
    if DEV == "mps":
        torch.mps.synchronize()
    out["sampling_seconds_8"] = round(time.time() - tic, 2)
    for i in range(8):
        strip([rec[t][0][i] for t in keep] + [final[i]], f"mnist-traj-{i}-x.png")
        strip([rec[t][1][i] for t in keep] + [final[i]], f"mnist-traj-{i}-x0.png")
    out["trajectory_t"] = keep + [0]
    print("trajectories", round(time.time() - t0), "s", flush=True)

    # ---- 64 samples, and the same 16 noises without the fresh noise
    tic = time.time()
    s64, _ = ddpm(net, noise(64, 31), seed=32)
    out["sampling_seconds_64"] = round(time.time() - tic, 2)
    grid(list(s64), 4, "mnist-samples-64.png")
    s16, _ = ddpm(net, noise(16, 41), seed=42)
    m16, _ = ddpm(net, noise(16, 41), seed=42, sigma=None)
    grid(list(s16), 2, "mnist-with-noise.png")
    grid(list(m16), 2, "mnist-mean-only.png")
    out["mean_only"] = {"ink_fraction_with_noise": r3((s16 > 0).float().mean()), "ink_fraction_mean_only": r3((m16 > 0).float().mean()),
                        "mean_abs_with_noise": r3(s16.abs().mean()), "mean_abs_mean_only": r3(m16.abs().mean())}

    # ---- DDIM: fewer steps, the same network and the same 8 starting noises
    xT = noise(8, 51)
    sweep = []
    for steps in [1, 2, 5, 10, 20, 50, 100, 1000]:
        tic = time.time()
        xs = ddim(net, xT, steps)
        if DEV == "mps":
            torch.mps.synchronize()
        sweep.append({"steps": steps, "seconds": round(time.time() - tic, 3)})
        grid(list(xs), 1, f"mnist-ddim-{steps}.png")
    ddpm8, _ = ddpm(net, xT, seed=52)
    grid(list(ddpm8), 1, "mnist-ddim-ddpm.png")
    out["ddim"] = sweep
    print("ddim", round(time.time() - t0), "s", flush=True)

    # ---- classifier-free guidance: 3s on demand, at several guidance scales; a classifier measures them
    cfg, _ = load(TinyUNet, "mnist-cfg", n_classes=10)
    clf, _ = load(Classifier, "mnist-classifier")
    with torch.no_grad():
        acc = float((clf(xte[:2000].to(DEV)).argmax(1).cpu() == yte[:2000]).float().mean())
    out["classifier_test_accuracy_2000"] = r3(acc)
    guided = []
    xT = noise(16, 61)
    for w in [0, 1, 2, 3, 5, 7]:                 # w = 0: the label is ignored; w = 1: the plain conditional model
        def fn(x, t, w=w):
            c3, cn = torch.full((x.shape[0],), 3, device=DEV), torch.full((x.shape[0],), 10, device=DEV)
            eu = eps_of(cfg, x, t, cn)
            if w == 0:
                return eu
            ec = eps_of(cfg, x, t, c3)
            return eu + w * (ec - eu)
        xs, _ = ddpm(None, xT, seed=62, eps_fn=fn)
        with torch.no_grad():
            p = torch.softmax(clf(xs.to(DEV)), 1).cpu()
        flat = xs.flatten(1)
        div = float(torch.cdist(flat, flat).sum() / (len(xs) * (len(xs) - 1)))
        guided.append({"w": w, "is_three": r3((p.argmax(1) == 3).float().mean()), "p_three": r3(p[:, 3].mean()), "diversity": r3(div),
                       "ink": r3((xs > 0).float().mean())})
        grid(list(xs), 2, f"mnist-cfg-{w}.png")
    out["guidance"] = guided
    digits = []
    xT = noise(10 * 6, 71)
    def fn_all(x, t):
        c = torch.arange(10, device=DEV).repeat_interleave(6)
        cn = torch.full_like(c, 10)
        eu, ec = eps_of(cfg, x, t, cn), eps_of(cfg, x, t, c)
        return eu + 3 * (ec - eu)
    xs, _ = ddpm(None, xT, seed=72, eps_fn=fn_all)
    grid(list(xs), 10, "mnist-cfg-digits.png")
    with torch.no_grad():
        pred = clf(xs.to(DEV)).argmax(1).cpu()
    out["digits_on_demand_accuracy"] = r3((pred == torch.arange(10).repeat_interleave(6)).float().mean())
    print("guidance", round(time.time() - t0), "s", flush=True)

    # ---- latent diffusion: the autoencoder's 4x7x7 code, and diffusion on it
    ae, ack = load(ConvAE, "mnist-ae")
    lat, _ = load(LatentDenoiser, "mnist-latent")
    with torch.no_grad():
        z = ae.enc(x0)
        rec3 = ae.dec(z)
        zs = ddpm(lat, noise(16, 81, (4, 7, 7)), seed=82)[0].to(DEV) * ack["latent_std"]
        dec = ae.dec(zs).cpu()
    strip([x0.cpu(), rec3.cpu()], "mnist-ae-3.png", gap=4)
    zc = z[0].cpu()
    zmin, zmax = float(zc.min()), float(zc.max())
    Image.fromarray(np.concatenate([np.kron(((zc[k] - zmin) / (zmax - zmin) * 255).numpy().astype(np.uint8), np.ones((4, 4), np.uint8))
                                    for k in range(4)], 1)).save(FIG / "mnist-latent-3.png")
    grid(list(dec), 2, "mnist-latent-samples.png")
    out["latent"] = {"shape": [4, 7, 7], "numbers": 196, "pixels": 784, "recon_mse_3": r3(((rec3 - x0) ** 2).mean()),
                     "latent_std": r3(ack["latent_std"])}

    # ---- does it copy? the ideal denoiser of 1,000 training images, against our network
    N = 1000
    train_set = xtr[:N].to(DEV).flatten(1)
    def ideal(x, t):
        a = ab[t]
        d2 = torch.cdist(x.flatten(1), a.sqrt() * train_set) ** 2
        w = torch.softmax(-d2 / (2 * (1 - a)), 1)
        x0h = (w @ train_set).reshape(x.shape)
        return (x - a.sqrt() * x0h) / (1 - a).sqrt()
    xs_ideal, _ = ddpm(None, noise(8, 91), seed=92, eps_fn=ideal)
    near = torch.cdist(xs_ideal.flatten(1).to(DEV), train_set).min(1)
    out["ideal"] = {"n_train": N, "nearest_distance": [r3(v) for v in near.values.cpu()], "nearest_index": near.indices.cpu().tolist()}
    grid(list(xs_ideal), 1, "mnist-ideal-samples.png")
    grid([train_set[i].cpu() for i in near.indices.cpu().tolist()], 1, "mnist-ideal-nearest.png")
    full = xtr.flatten(1).to(DEV)
    ours = s64[:8].flatten(1).to(DEV)
    nn_ = torch.cdist(ours, full).min(1)
    out["ours_nearest"] = {"n_train": len(full), "nearest_distance": [r3(v) for v in nn_.values.cpu()],
                           "typical_test_to_train": r3(torch.cdist(xte[:64].flatten(1).to(DEV), full).min(1).values.mean())}
    grid(list(s64[:8]), 1, "mnist-ours-8.png")
    grid([xtr[i] for i in nn_.indices.cpu().tolist()], 1, "mnist-ours-nearest.png")
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "data" / "mnist.json").write_text(json.dumps(out))
    print("done", out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
