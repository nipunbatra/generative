"""Figures and numbers for fitting by gradient descent, choosing the number of bumps,
autoencoder variants, and approximate posteriors (Laplace, MCMC, variational) for our 3.

    python scripts/make_figures_extra.py      # after train_models.py and make_figures.py

Writes figures/data/*.svg|png and figures/data/numbers-extra.json (merged by build_deck.py).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, DIGIT, DEV, FIG, MODELS, decode, encode, grid_png, load_ae, plt, save, tile_png  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from train_models import AutoEncoder, SparseAE, load_mnist, salt_pepper, toy_data  # noqa: E402

NUM = {}
BASE = json.loads((FIG / "numbers.json").read_text())


def r(v, k=2):
    return float(round(float(v), k))


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


XS = np.linspace(-5, 5, 600)


def hist_axes(ax, data, ymax=0.62):
    ax.hist(data, bins=np.linspace(-4.5, 4.5, 37), density=True, color=C["x_t"], ec="white", lw=1)
    ax.plot(data, np.full_like(data, -0.03), "|", color=C["x"], ms=8, alpha=.6, mew=1.1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.07, ymax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-4, -2, 0, 2, 4])


# =============================================================== 1. fit a Gaussian by gradient descent
def fit_gaussian(train):
    x = torch.tensor(train, dtype=torch.float32)
    mu = torch.tensor(1.0, requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=0.05)
    trace = []
    for step in range(301):
        nll = -torch.distributions.Normal(mu, log_sigma.exp()).log_prob(x).mean()
        trace.append((step, nll.item(), mu.item(), log_sigma.exp().item()))
        opt.zero_grad()
        nll.backward()
        opt.step()
    return trace


# =============================================================== 2. a mixture of K bumps by gradient descent
def fit_mixture(train, K, steps=3000, seed=0):
    torch.manual_seed(seed)
    x = torch.tensor(train, dtype=torch.float32)
    logits = torch.zeros(K, requires_grad=True)
    means = torch.tensor(np.quantile(train, (np.arange(K) + 0.5) / K), dtype=torch.float32, requires_grad=True)
    log_sd = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([logits, means, log_sd], lr=0.03)
    for _ in range(steps):
        logp = mix_logpdf(x, logits, means, log_sd)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return logits.detach(), means.detach(), log_sd.detach()


def mix_logpdf(x, logits, means, log_sd):
    comp = torch.distributions.Normal(means, log_sd.exp()).log_prob(x[:, None])     # [N, K]
    return torch.logsumexp(torch.log_softmax(logits, 0) + comp, dim=1)             # [N]


def main():
    g0, g1 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    train = toy_data(200, g0).numpy()
    held = toy_data(200, g1).numpy()

    # ---- Gaussian by gradient descent
    trace = fit_gaussian(train)
    NUM["gd"] = dict(start_mu=1.0, start_sd=1.0, mu=r(trace[-1][2]), sd=r(trace[-1][3]), nll=r(trace[-1][1], 3),
                     mu_closed=r(train.mean()), sd_closed=r(train.std()), steps=len(trace) - 1)
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 2.9), sharey=True)
    for ax, k in zip(axs, (0, 10, 40, 300)):
        _, nll, m, s = trace[k]
        hist_axes(ax, train)
        ax.plot(XS, npdf(XS, m, s), color=C["dec"], lw=2.4)
        ax.set_title(rf"step {k}:  $\mu = {m:.2f},\ \sigma = {s:.2f}$", fontsize=13, color=C["ink2"], loc="left")
        ax.text(-4.8, 0.52, rf"loss $= {nll:.2f}$", fontsize=13, color=C["loss"])
    save(fig, "gd-gauss")

    # ---- how many bumps? train vs held-out log-likelihood per number
    rows = []
    fits = {}
    for K in (1, 2, 3, 4):
        lg, mu, ls = fit_mixture(train, K)
        fits[K] = (lg, mu, ls)
        tr = mix_logpdf(torch.tensor(train), lg, mu, ls).mean().item()
        ho = mix_logpdf(torch.tensor(held), lg, mu, ls).mean().item()
        rows.append(dict(K=K, train=r(tr, 3), held=r(ho, 3), weights=[r(w, 2) for w in torch.softmax(lg, 0)],
                         means=[r(m) for m in mu], sds=[r(s) for s in ls.exp()]))
    NUM["mixK"] = rows
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 2.9), sharey=True)
    for ax, K in zip(axs, (1, 2, 3, 4)):
        lg, mu, ls = fits[K]
        hist_axes(ax, train)
        w = torch.softmax(lg, 0).numpy()
        dens = sum(w[k] * npdf(XS, mu[k].item(), ls[k].exp().item()) for k in range(K))
        for k in range(K):
            ax.plot(XS, w[k] * npdf(XS, mu[k].item(), ls[k].exp().item()), color=C["dec"], lw=1.2, ls=(0, (3, 3)))
        ax.plot(XS, dens, color=C["dec"], lw=2.4)
        ax.set_title(f"{K} bump{'s' if K > 1 else ''}", fontsize=15, color=C["ink"], loc="left")
    save(fig, "mixK-fits")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    Ks = [q["K"] for q in rows]
    ax.plot(Ks, [q["train"] for q in rows], "o-", color=C["ink3"], lw=2, ms=8)
    ax.plot(Ks, [q["held"] for q in rows], "o-", color=C["loss"], lw=2.6, ms=9)
    hi, lo = max(rows[2]["train"], rows[2]["held"]), min(rows[2]["train"], rows[2]["held"])
    top_is_held = rows[2]["held"] >= rows[2]["train"]
    ax.text(2.2, hi + 0.05, "200 new numbers" if top_is_held else "the 200 training numbers",
            color=C["loss"] if top_is_held else C["ink3"], fontsize=13)
    ax.text(2.2, lo - 0.09, "the 200 training numbers" if top_is_held else "200 new numbers",
            color=C["ink3"] if top_is_held else C["loss"], fontsize=13)
    ax.set_xticks(Ks)
    ax.set_xlabel("number of bumps $K$")
    ax.set_ylabel("average log-likelihood")
    ax.grid(color=C["grid"])
    save(fig, "mixK-score")

    # ---- sample from the K = 2 fit, and its posterior at x = 0.1
    lg, mu, ls = fits[2]
    gs = torch.Generator().manual_seed(3)
    k = torch.multinomial(torch.softmax(lg, 0), 1000, replacement=True, generator=gs)
    samples = (mu[k] + ls.exp()[k] * torch.randn(1000, generator=gs)).numpy()
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.hist(train, bins=np.linspace(-4.5, 4.5, 37), density=True, color=C["x_t"], ec="white", lw=1, label="the data")
    ax.hist(samples, bins=np.linspace(-4.5, 4.5, 37), density=True, histtype="step", color=C["z"], lw=2.4, label="1,000 samples")
    ax.set_xlim(-5, 5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, fontsize=14)
    save(fig, "mix-samples")
    comp = torch.distributions.Normal(mu, ls.exp()).log_prob(torch.tensor([0.1]))
    post = torch.softmax(torch.log_softmax(lg, 0) + comp, 0)
    NUM["mix_post01"] = [r(v, 3) for v in post]

    # ---- autoencoder variants on MNIST
    data = load_mnist()
    xte, yte = data["test"]
    raw = json.loads((MODELS / "training.json").read_text())
    meta = {}  # jobs that train several models store a list; index every model by its name
    for v in raw.values():
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, dict) and "name" in item:
                meta[item["name"]] = item
    i3 = BASE["digit3_index"]
    dae = AutoEncoder(16, variational=False).to(DEV)
    dae.load_state_dict(torch.load(MODELS / "dae16.pt", map_location=DEV))
    dae.eval()
    ae16 = load_ae("ae16")
    pick = [int(torch.nonzero(yte == d)[3]) for d in range(10)]
    clean = xte[pick]
    noisy = salt_pepper(clean, 0.2, torch.Generator().manual_seed(4))
    with torch.no_grad():
        out_dae = torch.sigmoid(dae.decoder(dae.encoder(noisy.to(DEV))[0])).cpu()
        out_ae = torch.sigmoid(ae16.decoder(ae16.encoder(noisy.to(DEV))[0])).cpu()
    grid_png(np.concatenate([noisy.flatten(1).numpy(), out_ae.numpy(), out_dae.numpy(), clean.flatten(1).numpy()]),
             4, 10, "dae-rows", scale=3, pad=2)
    NUM["dae"] = dict(rate=20, recon=r(meta["dae16"]["history"][-1]["train_recon"], 1))

    def sparse_code(name):
        m = SparseAE().to(DEV)
        m.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location=DEV))
        m.eval()
        with torch.no_grad():
            return m.enc(xte[i3:i3 + 1].to(DEV)).cpu()[0].numpy()

    h_plain, h_sparse = sparse_code("ae64"), sparse_code("sparse64")
    fig, axs = plt.subplots(2, 1, figsize=(10.4, 3.8), sharex=True)
    for ax, h, lab, col in ((axs[0], h_plain, "plain autoencoder, 64 code units", C["ink3"]),
                            (axs[1], h_sparse, "sparse autoencoder, same size, L1 penalty", C["enc"])):
        ax.bar(np.arange(64), h / max(h.max(), 1e-9), color=col, width=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(lab + f":  {int((h > 1e-3).sum())} of 64 non-zero", loc="left", fontsize=14, color=C["ink2"])
    axs[1].set_xlabel("code unit")
    save(fig, "sparse-codes")
    NUM["sparse"] = dict(active_plain=int((h_plain > 1e-3).sum()), active_sparse=int((h_sparse > 1e-3).sum()),
                         avg_plain=r(meta["ae64"]["history"][-1]["active_units"], 1),
                         avg_sparse=r(meta["sparse64"]["history"][-1]["active_units"], 1),
                         recon_plain=r(meta["ae64"]["history"][-1]["train_recon"], 1),
                         recon_sparse=r(meta["sparse64"]["history"][-1]["train_recon"], 1))

    # odd digits: the highest reconstruction error under the 16-number autoencoder
    with torch.no_grad():
        mu16, _ = encode(ae16, xte)
        logits = ae16.decoder(mu16.to(DEV)).cpu()
        err = F.binary_cross_entropy_with_logits(logits, xte.flatten(1), reduction="none").sum(1)
    worst = torch.argsort(err, descending=True)[:8]
    typical = torch.argsort((err - err.median()).abs())[:8]
    grid_png(np.concatenate([xte[typical].flatten(1).numpy(), xte[worst].flatten(1).numpy()]), 2, 8, "ae-anomaly", scale=4)
    NUM["anomaly"] = dict(median=r(err.median(), 1), worst=[r(err[i], 0) for i in worst])

    # =============================================================== 3. approximate posteriors for our 3
    vae = load_ae("vae2")
    x3 = xte[i3].reshape(1, 784).to(DEV)

    def log_joint(z):  # z: [N, 2] -> log p(z) + log p(x | z)
        logits = vae.decoder(z)
        ll = -F.binary_cross_entropy_with_logits(logits, x3.expand_as(logits), reduction="none").sum(1)
        return ll - 0.5 * (z ** 2).sum(1) - math.log(2 * math.pi)

    mu_enc = torch.tensor(BASE["our3"]["mu"])
    lin = torch.linspace(-1, 1, 401)
    cx, cy = mu_enc.tolist()
    w = 0.32
    gx, gy = cx + w * lin, cy + w * lin
    Z = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1).reshape(-1, 2)
    with torch.no_grad():
        LJ = torch.cat([log_joint(Z[i:i + 40000].to(DEV)).cpu() for i in range(0, len(Z), 40000)]).reshape(401, 401).double()
    post = torch.exp(LJ - LJ.max()).numpy()
    imax = int(torch.argmax(LJ))
    mode = Z[imax].clone()

    # Laplace: Gaussian at the mode, covariance = inverse curvature of the log posterior
    zt = mode.clone().to(DEV).requires_grad_(True)
    H = torch.autograd.functional.hessian(lambda z: log_joint(z[None])[0], zt).detach().cpu().double()
    cov_lap = torch.linalg.inv(-H).numpy()

    # MCMC: random-walk Metropolis on the log posterior
    gm = torch.Generator().manual_seed(9)
    cur = mode.clone()
    with torch.no_grad():
        cur_lp = log_joint(cur[None].to(DEV)).item()
        chain, acc = [], 0
        for t in range(4000):
            prop = cur + 0.03 * torch.randn(2, generator=gm)
            lp = log_joint(prop[None].to(DEV)).item()
            if math.log(torch.rand(1, generator=gm).item()) < lp - cur_lp:
                cur, cur_lp, acc = prop, lp, acc + 1
            if t >= 1000 and t % 3 == 0:
                chain.append(cur.numpy().copy())
    chain = np.array(chain)

    # Variational: best diagonal Gaussian by gradient ascent on the ELBO (reparameterised), from N(0, I)
    def vi_run(xrow, steps=801, seed=2, record=False):
        torch.manual_seed(seed)
        m = torch.zeros(2, requires_grad=True)
        log_s = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([m, log_s], lr=0.03)
        trace = []
        for step in range(steps):
            eps = torch.randn(64, 2)
            z = (m + log_s.exp() * eps).to(DEV)
            kl = 0.5 * (m ** 2 + (2 * log_s).exp() - 1 - 2 * log_s).sum()
            rec = -F.binary_cross_entropy_with_logits(vae.decoder(z), xrow.expand(64, -1), reduction="none").sum(1).mean().cpu()
            elbo = rec - kl
            if record and step % 10 == 0:
                trace.append((step, m.detach().numpy().copy(), log_s.exp().detach().numpy().copy(), elbo.item(), rec.item(), kl.item()))
            opt.zero_grad()
            (-elbo).backward()
            opt.step()
        return m.detach().numpy().copy(), log_s.exp().detach().numpy().copy(), trace

    _, _, path = vi_run(x3, record=True)

    def elbo_of(mq, sq, n=20000, xrow=None):
        xrow = x3 if xrow is None else xrow
        with torch.no_grad():
            eps = torch.randn(n, 2, generator=torch.Generator().manual_seed(5))
            z = torch.tensor(mq) + torch.tensor(sq) * eps
            rec = torch.cat([(-F.binary_cross_entropy_with_logits(vae.decoder(z[i:i + 5000].float().to(DEV)), xrow.expand(len(z[i:i + 5000]), -1),
                                                                    reduction="none").sum(1)).cpu() for i in range(0, n, 5000)]).mean()
            kl = 0.5 * (np.square(mq) + np.square(sq) - 1 - np.log(np.square(sq))).sum()
            return rec.item() - float(kl)

    m_vi, s_vi = path[-1][1], path[-1][2]
    with torch.no_grad():
        mu_e, lv_e = vae.encoder(x3)
    mu_e, sd_e = mu_e[0].cpu().double().numpy(), (0.5 * lv_e[0]).exp().cpu().double().numpy()
    # log p(x) from the fine local grid (the posterior is ~0.05 wide, the window 0.64): an exact reference for both gaps
    cell = math.log(float(gx[1] - gx[0]) * float(gy[1] - gy[0]))
    log_px_loc = float(torch.logsumexp(LJ.flatten(), 0)) + cell
    elbo_vi, elbo_enc = elbo_of(m_vi, s_vi), elbo_of(mu_e, sd_e)
    NUM["vi"] = dict(mu=[r(v, 2) for v in m_vi], sigma=[r(v, 3) for v in s_vi], elbo=r(elbo_vi, 1),
                     elbo_enc=r(elbo_enc, 1), log_px_local=r(log_px_loc, 2),
                     gap=r(log_px_loc - elbo_vi, 2), gap_enc=r(log_px_loc - elbo_enc, 2),
                     elbo2=r(r(log_px_loc, 2) - r(log_px_loc - elbo_vi, 2), 2), elbo_enc2=r(r(log_px_loc, 2) - r(log_px_loc - elbo_enc, 2), 2),
                     table=[dict(step=p_[0], rec=r(p_[4], 1), kl=r(p_[5], 2), elbo=r(p_[3], 1)) for p_ in path if p_[0] in (0, 20, 50, 800)],
                     log_px=BASE["evidence"]["log_px"], steps=path[-1][0], start_elbo=r(path[0][3], 0),
                     mode=[r(v, 2) for v in mode], lap_sd=[r(math.sqrt(cov_lap[0, 0]), 3), r(math.sqrt(cov_lap[1, 1]), 3)],
                     mcmc_n=len(chain), mcmc_acc=r(acc / 4000, 2))

    # ---- the three-panel figure, same zoomed window
    ext = (gx[0].item(), gx[-1].item(), gy[0].item(), gy[-1].item())
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.6), sharex=True, sharey=True)
    titles = ["Laplace: a Gaussian at the peak", "MCMC: a random walk, kept samples", "Variational: best Gaussian q"]
    for ax, title in zip(axs, titles):
        ax.contourf(gx.numpy(), gy.numpy(), post, levels=np.linspace(0.03, 1, 9), cmap="Blues", alpha=.9)
        ax.set_title(title, loc="left", fontsize=15, color=C["ink"])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$z_1$")
    axs[0].set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    vals, vecs = np.linalg.eigh(cov_lap)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    for k in (1, 2):
        axs[0].add_patch(Ellipse(mode.numpy(), 2 * k * math.sqrt(vals[1]), 2 * k * math.sqrt(vals[0]), angle=ang,
                                 fill=False, ec=C["z"], lw=2.4))
    axs[0].plot(*mode.numpy(), "o", color=C["z"], ms=7)
    axs[1].scatter(chain[:, 0], chain[:, 1], s=5, color=C["z"], alpha=.55, lw=0)
    for k in (1, 2):
        axs[2].add_patch(Ellipse(m_vi, 2 * k * s_vi[0], 2 * k * s_vi[1], fill=False, ec=C["enc"], lw=2.4))
    axs[2].plot(*m_vi, "o", color=C["enc"], ms=7)
    save(fig, "posterior-methods")

    # ---- the posterior alone (Part 3), and with the variational q (Parts 3-4): same zoomed window
    for name, with_q in (("posterior-alone", False), ("posterior-q", True)):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.contourf(gx.numpy(), gy.numpy(), post, levels=np.linspace(0.03, 1, 9), cmap="Blues", alpha=.9)
        ax.text(ext[0] + 0.03, ext[2] + 0.1, r"true posterior $p_\theta(z\mid x)$" + "\n(shaded, from the grid)", color=C["x"], fontsize=15)
        if with_q:
            for k in (1, 2):
                ax.add_patch(Ellipse(m_vi, 2 * k * s_vi[0], 2 * k * s_vi[1], fill=False, ec=C["enc"], lw=2.4))
            ax.plot(*m_vi, "o", color=C["enc"], ms=7)
            ax.text(m_vi[0] + 2.3 * s_vi[0], m_vi[1] + 2.0 * s_vi[1], r"$q(z)$, a Gaussian", color=C["enc"], fontsize=15)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
        save(fig, name)

    # ---- VI by hand: q travels from N(0, I) to our 3's patch, and the ELBO climbs to log p(x)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.28))
    lin2 = torch.linspace(-3.5, 3.5, 281)
    Z2 = torch.stack(torch.meshgrid(lin2, lin2, indexing="xy"), -1).reshape(-1, 2)
    with torch.no_grad():
        L2 = torch.cat([log_joint(Z2[i:i + 40000].to(DEV)).cpu() for i in range(0, len(Z2), 40000)]).reshape(281, 281).numpy()
    ax.imshow(np.clip(L2, L2.max() - 150, None), origin="lower", extent=(-3.5, 3.5, -3.5, 3.5), cmap="Blues", interpolation="bilinear")
    pts = np.array([p_[1] for p_ in path])
    ax.plot(pts[:, 0], pts[:, 1], color=C["enc"], lw=1.4, alpha=.7)
    shown = [0, 20, 50, 800]
    for i, (st, mq, sq, el, *_) in enumerate([p_ for p_ in path if p_[0] in shown]):
        alpha = 0.45 + 0.55 * i / (len(shown) - 1)
        ax.add_patch(Ellipse(mq, 2 * sq[0], 2 * sq[1], fill=False, ec=C["enc"], lw=2.2, alpha=alpha))
        ax.annotate(f"step {st}", xy=(mq[0] + 0.7 * sq[0], mq[1] + 0.7 * sq[1]), xytext=(1.25, 2.85 - 0.8 * i), fontsize=14,
                    color=C["enc"], arrowprops=dict(arrowstyle="-", color=C["enc"], lw=1.1, alpha=alpha),
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    ax.set_title(r"$q$ moves onto the patch", loc="left", fontsize=15, color=C["ink2"])
    bx.plot([p_[0] for p_ in path], [p_[3] for p_ in path], color=C["dec"], lw=2.4)
    bx.axhline(log_px_loc, color=C["x"], lw=1.6, ls=(0, (4, 3)))
    bx.text(800, log_px_loc + 14, r"$\log p_\theta(x) = %.2f$ (from a fine grid, only to check)" % log_px_loc,
            color=C["x"], fontsize=13, ha="right")
    bx.text(800, path[-1][3] - 26, r"$\mathrm{ELBO}(q)$", color=C["dec"], fontsize=14, ha="right")
    bx.set_ylim(-420, -90)
    bx.set_xlim(0, 820)
    bx.set_xlabel("gradient step")
    bx.set_title("the ELBO for our 3 climbs to just under it", loc="left", fontsize=15, color=C["ink2"])
    bx.grid(color=C["grid"])
    save(fig, "vi-watch")

    # ---- importance sampling: estimate log p(x) with 1,000 draws, from the prior vs from q
    def is_est(prop, S=1000, reps=20):
        out = []
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for _ in range(reps):
                eps = torch.randn(S, 2, generator=g)
                if prop == "prior":
                    z, lq = eps, -0.5 * (eps ** 2).sum(1) - math.log(2 * math.pi)
                else:
                    mq, sq = torch.tensor(m_vi), torch.tensor(s_vi)
                    z = mq + sq * eps
                    lq = -0.5 * (eps ** 2).sum(1) - math.log(2 * math.pi) - torch.log(sq).sum()
                lw = log_joint(z.float().to(DEV)).cpu().double() - lq.double()
                out.append(float(torch.logsumexp(lw, 0) - math.log(S)))
        return np.array(out)
    is_p, is_q = is_est("prior"), is_est("q")
    lo, hi = min(is_p.min(), is_q.min()) - 6, max(is_p.max(), is_q.max(), log_px_loc) + 4
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    for row, (vals, col, lab) in enumerate(((is_p, C["z"], "1,000 codes from the prior"), (is_q, C["enc"], "1,000 codes from $q$"))):
        ax.scatter(vals, np.full_like(vals, 1 - row) + np.random.default_rng(row).uniform(-.12, .12, len(vals)), s=46, color=col, alpha=.8, lw=0)
        ax.text(lo + 1, 1 - row + 0.3, lab, color=col, fontsize=14)
    ax.axvline(log_px_loc, color=C["x"], lw=1.6, ls=(0, (4, 3)))
    ax.text(log_px_loc - 1.5, 1.62, r"$\log p_\theta(x) = %.2f$" % log_px_loc, color=C["x"], fontsize=13, ha="right")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlim(lo, hi)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"estimate of $\log p_\theta(x)$ for our 3, one dot per run of 1,000 draws")
    save(fig, "is-estimates")
    NUM["isamp"] = dict(prior_med=r(np.median(is_p), 1), prior_min=r(is_p.min(), 1), prior_max=r(is_p.max(), 1),
                        q_mean=r(is_q.mean(), 2), q_sd=r(is_q.std(), 3), reps=len(is_p))

    # ---- amortisation over 20 test digits: VI by hand per digit vs the encoder's one pass
    gaps = []
    with torch.no_grad():
        mus_e, lvs_e = vae.encoder(xte[:20].reshape(20, 784).to(DEV))
    for k in range(20):
        xr = xte[k].reshape(1, 784).to(DEV)
        mk, sk, _ = vi_run(xr, seed=3)
        e_vi = elbo_of(mk, sk, n=4000, xrow=xr)
        e_en = elbo_of(mus_e[k].cpu().double().numpy(), (0.5 * lvs_e[k]).exp().cpu().double().numpy(), n=4000, xrow=xr)
        gaps.append(e_vi - e_en)
    gaps = np.array(gaps)
    NUM["amort"] = dict(n=20, mean=r(gaps.mean(), 2), median=r(np.median(gaps), 2), max=r(gaps.max(), 2), min=r(gaps.min(), 2))

    # ---- why learning needs the posterior: the pull on each bump's centre is weighted by p(bump | x)
    xs = np.linspace(-4.5, 4.5, 500)
    la, lb = npdf(xs, -2, 0.5), npdf(xs, 2, 0.5)
    rb = lb / (la + lb)
    train_pts = toy_data(200, torch.Generator().manual_seed(0)).numpy()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.plot(xs, rb, color=C["z"], lw=2.6)
    ax.plot(xs, 1 - rb, color=C["x"], lw=2.6, ls=(0, (5, 3)))
    ax.plot(train_pts, np.full_like(train_pts, -0.08), "|", color=C["ink2"], ms=9, alpha=.55, mew=1.1)
    x0 = 0.1
    rb0 = float(npdf(x0, 2, .5) / (npdf(x0, 2, .5) + npdf(x0, -2, .5)))
    ax.plot([x0, x0], [0, rb0], color=C["ink3"], lw=1.2, ls=":")
    ax.plot([x0], [rb0], "o", color=C["z"], ms=8)
    ax.plot([x0], [1 - rb0], "o", color=C["x"], ms=8)
    ax.text(x0 + 0.15, rb0 + 0.04, r"$p(\mathrm{B}\mid x{=}0.1) = %.3f$" % rb0, color=C["z"], fontsize=14)
    ax.text(x0 + 0.15, 1 - rb0 - 0.1, r"$p(\mathrm{A}\mid x{=}0.1) = %.3f$" % (1 - rb0), color=C["x"], fontsize=14)
    ax.text(3.0, 0.88, "bump B", color=C["z"], fontsize=14)
    ax.text(-4.3, 0.88, "bump A", color=C["x"], fontsize=14)
    ax.set_ylim(-0.15, 1.08)
    ax.set_xlim(-4.5, 4.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$x$")
    ax.grid(axis="y", color=C["grid"])
    save(fig, "resp-curve")
    NUM["pull"] = dict(rb=r(rb0, 3), ra=r(1 - rb0, 3), gb=r(rb0 * (x0 - 2) / 0.25, 2), ga=r((1 - rb0) * (x0 + 2) / 0.25, 2))

    # ---- why the integral is out of reach: grid cost by code size
    NUM["gridcost"] = [dict(d=d, calls=f"{901.0 ** d:.1e}") for d in (1, 2, 4, 8, 16)]

    (FIG / "numbers-extra.json").write_text(json.dumps(NUM, indent=1))
    print(json.dumps({k: NUM[k] for k in ("gd", "mixK", "mix_post01", "dae", "sparse", "anomaly", "vi", "isamp", "amort", "pull")}, indent=1))


if __name__ == "__main__":
    main()
