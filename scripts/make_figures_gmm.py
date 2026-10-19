"""Simple generative models of MNIST: Part 1's recipe on real images.

    python scripts/make_figures_gmm.py      # after make_figures.py; ~30 min on a laptop CPU the first time, cached after

1. PCA: squeeze each image to d numbers (z = V^T (x - mean)), and rebuild (x = mean + V z).
2. A mixture of K full-covariance Gaussians over the d numbers, fitted with the same gradient-descent loop as the
   1-D two-bump fit (loss = -log p(z), averaged), starting from k-means centres.
3. Choose K on held-out test images (and compare with AIC and BIC), sample, rebuild to pixels.
4. For comparison, straight on the 784 binary pixels: a mixture of Bernoullis (Bishop, PRML section 9.3.3).

Every (d, K) on the grid is fitted for the in-browser explorer. Writes figures/data/gmm-*.{png,svg},
figures/data/gmm-explore/*, figures/data/bern-*.png and figures/data/numbers-gmm.json.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
from matplotlib.patches import Ellipse
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, DIGIT, FIG, grid_png, plt, save  # noqa: E402
from train_models import load_mnist  # noqa: E402

D_GRID = [2, 5, 10, 20, 50]              # PCA numbers per image (at d = 100 these 2,000-step fits became unstable)
K_GRID = [1, 2, 5, 10, 20, 50, 100, 200]  # bumps
K_MORE = [400, 800]                       # extra sizes at d = 50, for the "how many bumps" plot
D_MAIN = 50
STEPS = 2000
CACHE = ROOT / "assets" / "models" / "gmm-cache"   # fitted mixtures, so a rerun only fits what is missing (gitignored)
EXPLORE = FIG / "gmm-explore"
BASE = json.loads((FIG / "numbers.json").read_text())


def r(v, k=2):
    return float(round(float(v), k))


# =============================================================== k-means, to start every bump on its own group of images
def kmeans(z, K, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = z[torch.randint(len(z), (1,), generator=g)]
    d2 = torch.cdist(z, centres).squeeze(1) ** 2
    for _ in range(1, K):                                   # k-means++: spread the starting centres out
        nxt = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        centres = torch.cat([centres, z[nxt]])
        d2 = torch.minimum(d2, torch.cdist(z, z[nxt]).squeeze(1) ** 2)
    for _ in range(iters):
        assign = torch.cat([torch.cdist(z[i:i + 20000], centres).argmin(1) for i in range(0, len(z), 20000)])
        for k in range(K):
            m = assign == k
            if m.any():
                centres[k] = z[m].mean(0)
    return centres, assign


# =============================================================== Gaussian mixture over d numbers
def mixture(logits, means, raw_diag, off):
    L = torch.tril(off, -1) + torch.diag_embed(F.softplus(raw_diag) + 1e-3)
    return MixtureSameFamily(Categorical(logits=logits), MultivariateNormal(means, scale_tril=L))


def avg_log_prob(model, z, bs=5000):
    with torch.no_grad():
        return torch.cat([model.log_prob(z[i:i + bs]) for i in range(0, len(z), bs)]).mean().item()


def fit_gmm(K, z, steps=STEPS, bs=1000, lr=0.02, seed=0):
    """The Part 1 loop: parameters, a log_prob, minus its average, Adam. Bumps start at k-means centres."""
    d = z.shape[1]
    centres, assign = kmeans(z, K, seed=seed)
    counts = torch.bincount(assign, minlength=K).float()
    spread = torch.stack([z[assign == k].std(0) if counts[k] > 2 else z.std(0) for k in range(K)]).clamp_min(1e-2)
    means = centres.clone().requires_grad_(True)
    raw_diag = torch.log(torch.expm1(spread)).requires_grad_(True)
    off = torch.zeros(K, d, d, requires_grad=True)
    logits = torch.log(counts + 1).requires_grad_(True)
    params = [logits, means, raw_diag, off]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch = z[torch.randint(len(z), (bs,), generator=g)]
        loss = -mixture(*params).log_prob(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return [p.detach() for p in params]


# =============================================================== mixture of Bernoullis over 784 binary pixels
def bern_log_prob(w_logits, p_logits, x):
    logp, log1mp = F.logsigmoid(p_logits), F.logsigmoid(-p_logits)        # log pi, log(1 - pi)
    lp = x @ logp.T + (1 - x) @ log1mp.T                                 # [N, K]: log p(x | bump k)
    return torch.logsumexp(torch.log_softmax(w_logits, 0) + lp, 1)


def fit_bern(K, x, steps=STEPS, bs=1000, lr=0.05, seed=0):
    centres, assign = kmeans(x, K, iters=10, seed=seed)
    counts = torch.bincount(assign, minlength=K).float()
    pi0 = centres.clamp(0.05, 0.95)
    w_logits = torch.log(counts + 1).requires_grad_(True)
    p_logits = torch.log(pi0 / (1 - pi0)).requires_grad_(True)
    opt = torch.optim.Adam([w_logits, p_logits], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch = x[torch.randint(len(x), (bs,), generator=g)]
        loss = -bern_log_prob(w_logits, p_logits, batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return [w_logits.detach(), p_logits.detach()]


def cached(name, fn):
    f = CACHE / f"{name}.pt"
    if f.exists():
        c = torch.load(f)
        return c["params"], c["secs"]
    t0 = time.time()
    params = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(dict(params=params, secs=time.time() - t0), f)
    return params, time.time() - t0


def gauss_params(d, K):
    return K * (d + d * (d + 1) // 2) + (K - 1)            # centres + covariances + free weights


def pick_k(rows):
    """The toy's rule, made robust: the smallest K that scores as well as any (within 0.2 nats) on new images."""
    top = max(rows, key=lambda row: row["test"])
    return min((row for row in rows if row["test"] >= top["test"] - 0.2), key=lambda row: row["k"]), top


def main():
    torch.manual_seed(0)
    data = load_mnist(binary=False)
    xtr = data["train"][0].reshape(-1, 784)
    xte, yte = data["test"][0].reshape(-1, 784), data["test"][1]
    N = len(xtr)
    EXPLORE.mkdir(parents=True, exist_ok=True)

    # ---- PCA once: all 784 directions, sorted by how much digits vary along them
    mean = xtr.mean(0)
    _, S, Vt = torch.linalg.svd(xtr - mean, full_matrices=False)
    var_kept = {d: ((S[:d] ** 2).sum() / (S ** 2).sum()).item() for d in D_GRID}

    def squeeze(x, d):
        return (x - mean) @ Vt[:d].T

    def rebuild(z, d):
        return (mean + z @ Vt[:d]).clamp(0, 1)

    i3 = BASE["digit3_index"]
    pick = [i3 if c == 3 else int((yte == c).nonzero()[0]) for c in range(10)]
    grid_png(torch.cat([xte[pick], rebuild(squeeze(xte[pick], D_MAIN), D_MAIN)]).numpy(), 2, 10, "pca-recon", scale=4)
    for d in D_GRID:
        grid_png(rebuild(squeeze(xte[pick], d), d).numpy(), 1, 10, f"gmm-explore/recon_d{d}", scale=3)
    mse = ((rebuild(squeeze(xte, D_MAIN), D_MAIN) - xte) ** 2).sum(1).mean().item()

    # one Gaussian straight over the 784 pixels, each pixel with its own width and no covariance
    torch.manual_seed(7)
    grid_png((mean + xtr.std(0) * torch.randn(12, 784)).clamp(0, 1).numpy(), 1, 12, "gauss784-diag", scale=4)

    # ---- the grid: every (d, K), for the explorer; plus K = 400, 800 at d = 50
    explore = dict(d_grid=D_GRID, k_grid=K_GRID, var_kept={}, best_k={}, test={}, params={})
    fits = {}
    for d in D_GRID:
        ztr, zte = squeeze(xtr, d), squeeze(xte, d)
        rows = []
        for K in K_GRID + (K_MORE if d == D_MAIN else []):
            params, secs = cached(f"d{d}_k{K}", lambda: fit_gmm(K, ztr))
            model = mixture(*params)
            rows.append(dict(k=K, train=r(avg_log_prob(model, ztr), 1), test=r(avg_log_prob(model, zte), 1),
                             params=gauss_params(d, K), secs=r(secs, 0),
                             used=int((torch.softmax(params[0], 0) >= 0.001).sum())))
            fits[(d, K)] = params
            if K in K_GRID:
                torch.manual_seed(7)
                grid_png(rebuild(model.sample((24,)), d).numpy(), 2, 12, f"gmm-explore/d{d}_k{K}", scale=3)
            print(d, rows[-1], flush=True)
        best, top = pick_k(rows)
        explore["var_kept"][d] = r(100 * var_kept[d], 1)
        explore["best_k"][d] = best["k"]
        explore["test"][d] = {row["k"]: row["test"] for row in rows if row["k"] in K_GRID}
        explore["params"][d] = {row["k"]: row["params"] for row in rows if row["k"] in K_GRID}
        if d == D_MAIN:
            rows_main, best_main, top_main = rows, best, top
    (EXPLORE / "index.json").write_text(json.dumps(explore))

    # ---- d = 50: the score by K, and AIC / BIC from the training log-likelihood
    rows, kb = rows_main, best_main["k"]
    for row in rows:
        loglik = N * row["train"]
        row["aic"] = r(2 * row["params"] - 2 * loglik, 0)
        row["bic"] = r(row["params"] * math.log(N) - 2 * loglik, 0)
    k_aic = min(rows, key=lambda row: row["aic"])["k"]
    k_bic = min(rows, key=lambda row: row["bic"])["k"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = [row["k"] for row in rows]
    ax.plot(ks, [row["train"] for row in rows], "o-", color=C["ink3"], lw=2, ms=7, label="the 60,000 training images")
    ax.plot(ks, [row["test"] for row in rows], "o-", color=C["loss"], lw=2.4, ms=8, label="10,000 new test images")
    ax.plot([kb], [best_main["test"]], "o", ms=16, mfc="none", mec=C["loss"], mew=2)
    ax.annotate(f"chosen: K = {kb}", (kb, best_main["test"]), xytext=(0, -34), textcoords="offset points", ha="center",
                color=C["loss"], fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel(r"number of bumps $K$")
    ax.set_ylabel("average log-likelihood per image")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.grid(color=C["grid"])
    save(fig, "gmm-score")

    # ---- d = 50, chosen K: centres, samples, nearest training images
    model = mixture(*fits[(D_MAIN, kb)])
    logits, means = fits[(D_MAIN, kb)][0], fits[(D_MAIN, kb)][1]
    order = torch.argsort(logits, descending=True)
    grid_png(rebuild(means[order[:40]], D_MAIN).numpy(), 4, 10, "gmm-means", scale=4)
    weights = torch.softmax(logits, 0)[order]
    for K in (1, 10, kb):
        torch.manual_seed(7)
        grid_png(rebuild(mixture(*fits[(D_MAIN, K)]).sample((24,)), D_MAIN).numpy(), 2, 12, f"gmm-samples-k{K}", scale=4)
    torch.manual_seed(7)
    x_new = rebuild(model.sample((24,)), D_MAIN)[:8]
    grid_png(x_new[3:4].numpy(), 1, 1, "gmm-sample-one", scale=4, pad=0)   # a 3, to echo our 3 on the way in
    near = torch.cdist(x_new, xtr).argmin(1)
    grid_png(torch.cat([x_new, xtr[near]]).numpy(), 2, 8, "gmm-nearest", scale=4)

    # ---- d = 2: everything is visible. Test digits, the 10 fitted bumps, and 24 samples
    z2 = squeeze(xte, 2)
    m2 = mixture(*fits[(2, 10)])
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    sub = torch.randperm(len(z2), generator=torch.Generator().manual_seed(0))[:4000]
    for c in range(10):
        m = yte[sub] == c
        ax.scatter(z2[sub][m, 0], z2[sub][m, 1], s=5, color=DIGIT[c], alpha=.45, lw=0, rasterized=True)
        med = z2[yte == c].median(0).values
        ax.text(med[0], med[1], str(c), fontsize=12, fontweight="bold", ha="center", va="center", color=C["ink"],
                bbox=dict(boxstyle="circle,pad=0.12", fc=(1, 1, 1, 0.85), ec=DIGIT[c], lw=1.4), zorder=9)
    comp = m2.component_distribution
    for k in range(10):
        vals, vecs = torch.linalg.eigh(comp.covariance_matrix[k])
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        for s_, lw, ls in ((1, 1.8, "-"), (2, 1.0, (0, (4, 3)))):
            ax.add_patch(Ellipse(comp.loc[k].numpy(), 2 * s_ * math.sqrt(vals[1]), 2 * s_ * math.sqrt(vals[0]),
                                 angle=ang, fill=False, ec=C["ink"], lw=lw, ls=ls, zorder=8))
    torch.manual_seed(7)
    zs = m2.sample((24,))
    ax.scatter(zs[:, 0], zs[:, 1], marker="x", s=46, color=C["loss"], lw=2, zorder=10)
    ax.set_xlabel(r"$z_1$, the first PCA number")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    ax.set_aspect("equal")
    ax.grid(color=C["grid"])
    save(fig, "gmm-2d")
    grid_png(rebuild(zs, 2).numpy(), 2, 12, "gmm-2d-samples", scale=4)
    lo, hi = z2.quantile(0.02, 0), z2.quantile(0.98, 0)
    gx = torch.linspace(float(lo[0]), float(hi[0]), 12)
    gy = torch.linspace(float(hi[1]), float(lo[1]), 12)
    plane = torch.stack(torch.meshgrid(gy, gx, indexing="ij"), -1).reshape(-1, 2)[:, [1, 0]]
    grid_png(rebuild(plane, 2).numpy(), 12, 12, "pca2-grid", scale=2, pad=1)

    # ---- straight on 784 binary pixels: a mixture of Bernoullis
    btr, bte = (xtr > 0.5).float(), (xte > 0.5).float()
    brows, bfits = [], {}
    for K in K_GRID:
        params, secs = cached(f"bern_k{K}", lambda: fit_bern(K, btr))
        with torch.no_grad():
            tr = torch.cat([bern_log_prob(*params, btr[i:i + 10000]) for i in range(0, N, 10000)]).mean().item()
            te = bern_log_prob(*params, bte).mean().item()
        brows.append(dict(k=K, train=r(tr, 1), test=r(te, 1), params=K * 784 + K - 1, secs=r(secs, 0),
                          used=int((torch.softmax(params[0], 0) >= 0.001).sum())))
        bfits[K] = params
        print("bern", brows[-1], flush=True)
    bbest, _ = pick_k(brows)
    w_logits, p_logits = bfits[bbest["k"]]
    border = torch.argsort(w_logits, descending=True)
    grid_png(torch.sigmoid(p_logits[border[:20]]).numpy(), 2, 10, "bern-means", scale=4)
    torch.manual_seed(7)
    picks = Categorical(logits=w_logits).sample((24,))
    grid_png(torch.bernoulli(torch.sigmoid(p_logits[picks])).numpy(), 2, 12, "bern-samples", scale=4)

    num = dict(d=D_MAIN, var_kept=r(100 * var_kept[D_MAIN], 1), var_kept_2=r(100 * var_kept[2], 1),
               recon_mse=r(mse, 1), ks=rows, best_k=kb, best=best_main,
               top=top_main, k_aic=k_aic, k_bic=k_bic, params_per_bump=1 + D_MAIN + D_MAIN * (D_MAIN + 1) // 2,
               pixel_cov=784 * 785 // 2, top_weight=r(100 * weights[0], 1), n_light=int((weights < 0.001).sum()),
               best_k_by_d=explore["best_k"], bern=dict(ks=brows, best_k=bbest["k"], best=bbest))
    (FIG / "numbers-gmm.json").write_text(json.dumps(num, indent=1))
    print(json.dumps({k: v for k, v in num.items() if k != "ks"}, indent=1))


if __name__ == "__main__":
    main()
