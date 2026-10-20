"""Figures and numbers for the lecture "KL divergence and variational inference" (vi.html).

    python scripts/make_figures_vi.py      # no trained models needed; about 30 s on a laptop CPU

The lecture follows the PML course (Lectures 23-24, notebook variational-inference.ipynb) and its companion notebook
vi-lab/vi_lecture.ipynb. Every computation here calls the same vi-lab/vi_utils.py helpers with the same seeds as the
notebook cells, so a student running the notebook sees the numbers on the slides. Writes figures/data/vik-*.svg and
figures/data/numbers-vi.json, which build_deck.py merges as {{vik.*}}.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vi-lab"))
from make_figures import C, FIG, plt, save  # noqa: E402  (shared palette and Computer Modern maths)
import vi_utils as vu  # noqa: E402
from vi_utils import dist  # noqa: E402

NUM = {}
P, Q, PRIOR, LIK, KL = C["x"], C["enc"], C["z"], C["dec"], C["loss"]


def s(v, k=2):
    """A fixed-decimals string with a typographic minus, for slide text."""
    v = float(v)
    if abs(v) < 0.5 * 10 ** -k:
        v = 0.0                                            # no "−0.00"
    return f"{v:.{k}f}".replace("-", "−")


def dens(d, x):
    return vu.density(d, x).numpy()


def no_y(ax):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def shade(ax, x, y, col, alpha=0.16):
    ax.fill_between(x, 0, y, color=col, alpha=alpha, lw=0)


# ============================================================================ N1 · Bayes and the evidence
def part1():
    # Steve (PML Lecture 2): 10 librarians and 200 farmers; 4 and 20 fit the description
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    for i in range(10):
        ax.add_patch(plt.Rectangle((0, i * 1.1), 1, 1, fc=C["x"] if i < 4 else "#E4ECFF", ec="white"))
    for k in range(200):
        col, row = divmod(k, 10)
        ax.add_patch(plt.Rectangle((2.5 + col * 1.1, row * 1.1), 1, 1, fc=C["z"] if k < 20 else "#FCEFD9", ec="white"))
    ax.text(0.5, 11.4, "10 librarians", ha="center", fontsize=14, color=C["x"])
    ax.text(2.5 + 10 * 1.1, 11.4, "200 farmers", ha="center", fontsize=14, color=C["z"])
    ax.text(0.5, -1.5, "4 fit", ha="center", fontsize=13, color=C["x"])
    ax.text(2.5 + 1.0, -1.5, "20 fit", ha="center", fontsize=13, color=C["z"])
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-2.4, 12.4)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "vik-steve")
    prior, lik = {"L": 1 / 21, "F": 20 / 21}, {"L": 0.4, "F": 0.1}
    ev = sum(prior[j] * lik[j] for j in prior)
    NUM["steve"] = dict(ev=s(ev, 4), post_L=s(prior["L"] * lik["L"] / ev, 3), post_F=s(prior["F"] * lik["F"] / ev, 3))

    # the coin: prior, likelihood, posterior, as three cumulative pictures (one per click)
    coin = vu.Coin(heads=17, tails=13, a=5, b=3)
    th = vu.grid(0.001, 0.999)
    pr, po = dens(coin.prior, th), dens(coin.posterior, th)
    ll = coin.log_likelihood(th)
    lk = (ll - ll.max()).exp().numpy() * po.max()
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(th, pr, color=PRIOR, lw=2.4, label=r"prior $\mathrm{Beta}(5, 3)$")
        if k >= 2:
            ax.plot(th, lk, color=LIK, lw=2.2, ls=(0, (5, 3)), label="likelihood (rescaled)")
        if k >= 3:
            shade(ax, th, po, P)
            ax.plot(th, po, color=P, lw=2.6, label=r"posterior $\mathrm{Beta}(22, 16)$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, po.max() * 1.12)
        ax.set_xlabel(r"$\theta$, the chance of heads")
        no_y(ax)
        ax.legend(loc="upper left")
        save(fig, f"vik-coin-{k}")
    ev_grid = vu.evidence_grid(coin.log_joint, 1e-6, 1 - 1e-6)
    NUM["coin"] = dict(log_ev=s(coin.log_evidence, 3), log_ev2=s(coin.log_evidence, 2), neg_log_ev=s(-coin.log_evidence, 3),
                       grid=s(ev_grid, 3), post_mean=s(22 / 38, 3), mle=s(17 / 30, 3), prior_mean=s(5 / 8, 3))

    # linear regression evidence: closed form vs prior sampling (PML Sampling-1, mc-linreg-evidence.ipynb)
    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100)
    y = 1.5 + 0.5 * x + 0.2 * torch.randn_like(x)
    linreg = vu.LinReg(x, y, noise_sd=0.2, prior_sd=1.0)
    log_ev_mc, ll = vu.evidence_mc(linreg.prior, linreg.log_likelihood, n=15_000)
    w = torch.softmax(ll, 0)
    ll_true = linreg.log_likelihood(torch.tensor([1.5, 0.5])).item()
    NUM["linreg_ev"] = dict(exact=s(linreg.log_evidence(), 2), mc=s(log_ev_mc, 2), pml_mc="11.20", top10=f"{100 * w.topk(10).values.sum():.0f}",
                            worst=f"{ll.min().item():,.0f}".replace("-", "−"), ll_true=s(ll_true, 1),
                            median=f"{ll.median().item():,.0f}".replace("-", "−"))
    torch.manual_seed(3)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for t in linreg.prior.sample((60,)):
        ax.plot(x, t[0] + t[1] * x, color=PRIOR, alpha=0.28, lw=1.1)
    ax.scatter(x, y, s=9, color=C["ink"], zorder=3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(color=C["grid"])
    save(fig, "vik-linreg-lines")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(np.arange(1, len(ll) + 1), ll.sort(descending=True).values, color=PRIOR, lw=2.4)
    ax.axhline(ll_true, color=C["ink"], ls=(0, (4, 3)), lw=1.2)
    ax.text(1.2, ll_true + 90, r"the true $\theta$", fontsize=13, color=C["ink"])
    ax.set_xscale("log")
    ax.set_ylim(-3000, 300)
    ax.set_xlabel("prior draws, best first")
    ax.set_ylabel(r"$\log p(D\mid\theta_i)$")
    ax.grid(color=C["grid"])
    save(fig, "vik-linreg-sorted")

    # no formula at all: a Normal prior on the coin's logit
    lcoin = vu.LogitCoin(heads=17, tails=13, mu=0.0, sigma=1.0)
    NUM["logit"] = dict(grid=s(vu.evidence_grid(lcoin.log_joint, -10, 10), 3),
                        mc=s(vu.evidence_mc(lcoin.prior, lcoin.log_likelihood, n=100_000)[0], 3))
    wv = vu.grid(-3, 3.5)
    yv = lcoin.log_joint(wv).exp().numpy() * 1e9
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(wv, 0, yv, facecolor="#F3F4F6", hatch="///", edgecolor="#C7CCD4", lw=0)
    ax.plot(wv, yv, color=P, lw=2.6)
    ax.text(-2.9, yv.max() * 0.92, r"shaded area $= p(D) =$ ?", fontsize=15)
    ax.set_xlabel(r"$w$, the coin's logit ($\theta = \mathrm{sigmoid}(w)$)")
    ax.set_ylabel(r"$p(D, w)\ \ (\times 10^{-9})$")
    ax.set_xlim(-3, 3.5)
    ax.set_ylim(0, yv.max() * 1.15)
    ax.grid(color=C["grid"])
    save(fig, "vik-logit-integrand")

    # three ways on mixture A (the Laplace notebook's target)
    target = vu.mog([0.7, 0.3], [-2.0, 2.0], [1.0, 1.0])
    xx = vu.grid(-6, 6)
    lap = vu.laplace(target.log_prob, init=0.0)
    samples, acc = vu.metropolis(target.log_prob, init=0.0, steps=20_000, step=2.0)
    h = vu.fit(vu.TrainableNormal(-0.5, 1.0), lambda q: vu.kl_grid(q, target, vu.grid(-8, 8, 2001)), steps=600)
    qv = h[-1]["q"]
    NUM["three"] = dict(lap_m=s(lap.loc, 2), lap_s=s(lap.scale, 2), mcmc_mean=s(samples.mean(), 2), mcmc_acc=s(acc, 2),
                        vi_m=s(qv.loc, 2), vi_s=s(qv.scale, 2), true_mean=s(0.7 * -2 + 0.3 * 2, 2))
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 3.5), sharey=True, gridspec_kw=dict(wspace=0.1))
    titles = ["Laplace: a Gaussian at the peak", "MCMC: where a random walk went", "VI: the closest Gaussian $q$"]
    pt = dens(target, xx)
    for ax, t in zip(axs, titles):
        shade(ax, xx, pt, P)
        ax.plot(xx, pt, color=P, lw=1.6)
        ax.set_title(t, loc="left", fontsize=14)
        ax.set_xlabel(r"$\theta$")
        no_y(ax)
    axs[0].plot(xx, dens(lap, xx), color=PRIOR, lw=2.6)
    axs[1].hist(samples.numpy(), bins=np.linspace(-6, 6, 61), density=True, color=PRIOR, alpha=0.55, ec="white", lw=0.4)
    axs[2].plot(xx, dens(qv, xx), color=Q, lw=2.6)
    save(fig, "vik-three-ways")
    return coin, target, xx


# ============================================================================ N2 · KL divergence
def part2(target, xx):
    names = ["apple", "banana", "clementine"]
    q1y, p2y = np.array([0.5, 0.4, 0.1]), np.array([0.5, 0.1, 0.4])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.arange(3)
    for k, (v, lab, col) in enumerate(((q1y, r"year 1 ($q$, the reference)", P), (p2y, r"year 2 ($p$)", Q))):
        ax.bar(xs + (k - 0.5) * 0.38, v, width=0.38, color=col, label=lab)
        for x_, vv in zip(xs + (k - 0.5) * 0.38, v):
            ax.text(x_, vv + 0.012, f"{vv:.1f}", ha="center", fontsize=13)
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("share of the class")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color=C["grid"])
    save(fig, "vik-fruit")
    ratio = p2y / q1y
    terms = p2y * np.log(ratio)                     # each fruit's p log(p/q)
    NUM["fruit"] = dict(avg=s(ratio.mean(), 2), kl=s(terms.sum(), 3), log4=s(math.log(4), 3), tb=s(terms[1], 3), tc=s(terms[2], 3))

    # ritvikmath's construction: f(4) = -f(1/4), f(2) = -f(1/2), f(1) = 0  ->  the log
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    rr = np.linspace(0.12, 4.6, 400)
    ax.plot(rr, np.log(rr), color=C["ink2"], lw=1.8, ls=(0, (5, 3)))
    for v, col in ((4, Q), (0.25, Q), (2, LIK), (0.5, LIK), (1, C["ink"])):
        ax.plot([v], [math.log(v)], "o", color=col, ms=10, mec="white", mew=1.5, zorder=4)
    ax.text(4, math.log(4) + 0.22, r"$f(4) = y$", ha="center", fontsize=14, color=Q)
    ax.text(0.25 + 0.1, math.log(0.25) - 0.1, r"$f(\frac{1}{4}) = -y$", fontsize=14, color=Q, va="top")
    ax.text(2, math.log(2) + 0.22, r"$f(2) = z$", ha="center", fontsize=14, color=LIK)
    ax.text(0.5 + 0.12, math.log(0.5) - 0.02, r"$f(\frac{1}{2}) = -z$", fontsize=14, color=LIK, va="top")
    ax.text(1.12, -0.32, r"$f(1) = 0$", fontsize=14, color=C["ink"])
    ax.text(3.2, 1.0, r"$\log r$", fontsize=16, color=C["ink2"])
    ax.axhline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"ratio $r = p(x)/q(x)$")
    ax.set_ylabel(r"$f(r)$")
    ax.set_xlim(0, 4.8)
    ax.set_ylim(-2.4, 2.0)
    ax.grid(color=C["grid"])
    save(fig, "vik-logfn")

    # bits: Jia-Bin Huang's weather
    wth, wrong = np.array([0.5, 0.25, 0.25]), np.array([0.25, 0.5, 0.25])
    H = -(wth * np.log2(wth)).sum()
    CE = -(wth * np.log2(wrong)).sum()
    NUM["bits"] = dict(H=s(H, 2), CE=s(CE, 2), KL=s(CE - H, 2))

    # KL ≥ 0: log r under its tangent
    rr = np.linspace(0.08, 3.0, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(rr, np.log(rr), color=LIK, lw=2.6)
    ax.plot(rr, rr - 1, color=PRIOR, lw=2.2, ls=(0, (5, 3)))
    ax.plot([1], [0], "o", color=C["ink"], ms=8)
    ax.text(2.25, 1.62, r"$r - 1$", color=PRIOR, fontsize=17)
    ax.text(2.5, 0.62, r"$\log r$", color=LIK, fontsize=17)
    ax.set_xlim(0, 3.05)
    ax.set_ylim(-1.6, 2.1)
    ax.axhline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"$r = p(\theta)/q(\theta)$")
    ax.grid(color=C["grid"])
    save(fig, "vik-tangent")

    # a coin, both ways
    fair, biased = dist.Bernoulli(0.5), dist.Bernoulli(0.8)
    k = dist.kl_divergence(biased, fair).item()
    NUM["coinkl"] = dict(bf=s(k, 3), fb=s(dist.kl_divergence(fair, biased).item(), 3), x10=s(math.exp(10 * k), 1),
                         x30=f"{math.exp(30 * k):.0f}")
    n = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(n, np.exp(n * k), color=KL, lw=2.6)
    for n_ in (10, 30):
        ax.plot([n_], [math.exp(n_ * k)], "o", color=KL, ms=9, mec="white", mew=1.5)
        ax.text(n_ - 0.6, math.exp(n_ * k) * 1.6, f"{math.exp(n_ * k):.0f}×" if n_ == 30 else f"{math.exp(n_ * k):.1f}×",
                ha="center" if n_ == 10 else "right", fontsize=14)
    ax.set_xlabel("tosses of the 0.8 coin, $n$")
    ax.set_ylabel(r"$p(D\mid 0.8)\ /\ p(D\mid 0.5)$")
    ax.grid(color=C["grid"])
    save(fig, "vik-coin-kl")

    # G1 (PML notebook cells 5-13)
    p = dist.Normal(1., 4.)
    q1, q2 = dist.Normal(0., 2.), dist.Normal(1., 3.)
    z = vu.grid(-5, 15)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    shade(ax, z, dens(p, z), P)
    ax.plot(z, dens(p, z), color=P, lw=2.8, label=r"$p = \mathcal{N}(1, 4^2)$")
    ax.plot(z, dens(q1, z), color=Q, lw=2.4, ls=(0, (5, 3)), label=r"$q_1 = \mathcal{N}(0, 2^2)$")
    ax.plot(z, dens(q2, z), color=LIK, lw=2.4, ls=(0, (6, 2, 1, 2)), label=r"$q_2 = \mathcal{N}(1, 3^2)$")
    ax.set_xlabel(r"$\theta$")
    no_y(ax)
    ax.legend(loc="upper right")
    save(fig, "vik-g1")
    NUM["g1"] = dict(kl1=s(dist.kl_divergence(q1, p), 2), kl2=s(dist.kl_divergence(q2, p), 2),
                     kl1x=s(dist.kl_divergence(q1, p), 4), kl2x=s(dist.kl_divergence(q2, p), 4))

    # forward vs reverse, on mixture A and on the same mixture with narrower bumps (notebook N2)
    xg = vu.grid(-8, 8, 2001)
    narrow = vu.mog([0.7, 0.3], [-2.0, 2.0], [0.5, 0.5])
    fr = {}
    fig, axs = plt.subplots(2, 3, figsize=(13.4, 5.9), sharex=True, gridspec_kw=dict(hspace=0.42, wspace=0.1))
    for row, (mix, key) in zip(axs, ((target, "wide"), (narrow, "narrow"))):
        forward_kl = lambda q, mix=mix: vu.kl_grid(mix, q, xg)
        reverse_kl = lambda q, mix=mix: vu.kl_grid(q, mix, xg)
        fits = [("fwd", r"$\mathrm{KL}(p\,\Vert\,q)$", vu.fit(vu.TrainableNormal(0.5, 0.6), forward_kl, steps=800)),
                ("revl", r"$\mathrm{KL}(q\,\Vert\,p)$, start left", vu.fit(vu.TrainableNormal(-0.5, 0.6), reverse_kl, steps=800)),
                ("revr", r"$\mathrm{KL}(q\,\Vert\,p)$, start right", vu.fit(vu.TrainableNormal(0.5, 0.6), reverse_kl, steps=800))]
        pm = dens(mix, xx)
        for ax, (tag, title, hh) in zip(row, fits):
            qv = hh[-1]["q"]
            shade(ax, xx, pm, P)
            ax.plot(xx, pm, color=P, lw=1.5)
            ax.plot(xx, dens(qv, xx), color=Q, lw=2.6)
            ax.set_title(title + rf" $= {hh[-1]['loss']:.3f}$", loc="left", fontsize=15)
            no_y(ax)
            fr[f"{key}_{tag}"] = dict(m=s(qv.loc, 2), s=s(qv.scale, 2), kl=s(hh[-1]["loss"], 3))
    for ax in axs[1]:
        ax.set_xlabel(r"$\theta$")
    save(fig, "vik-fwdrev")
    NUM["fwdrev"] = fr


# ============================================================================ N3 · optimising q (G2)
def part3():
    p = dist.Normal(1., 4.)
    grid_rows = []
    for loc in [0., 1.]:
        for scale in [1., 2., 4.]:
            grid_rows.append(dict(loc=f"{loc:.0f}", scale=f"{scale:.0f}", kl=s(dist.kl_divergence(dist.Normal(loc, scale), p), 3)))
    NUM["g2grid"] = grid_rows
    # the notebook's explicit loop: Adam on KL(q || p), from N(4, 0.5)
    q = vu.TrainableNormal(loc=4.0, scale=0.5)
    opt = torch.optim.Adam(q.parameters(), lr=0.05)
    hist = []
    for step in range(400):
        loss = dist.kl_divergence(q(), p)
        hist.append((step, q.loc.item(), q().scale.item(), loss.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    first_below = next(st for st, _, _, l in hist if l < 0.01)
    NUM["g2"] = dict(m=s(q.loc, 2), s=s(q().scale, 2), kl0=s(hist[0][3], 2), below=first_below,
                     snap=[dict(step=st, m=s(m_, 2), s=s(sd, 2), kl=s(l, 3)) for st, m_, sd, l in hist if st in (0, 10, 30, 100, 399)])
    z = vu.grid(-6, 15)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.22))
    shade(a1, z, dens(p, z), P)
    a1.plot(z, dens(p, z), color=P, lw=2.6, label=r"$p = \mathcal{N}(1, 4^2)$")
    for st, alpha, ls in ((0, 0.35, (0, (1, 2))), (30, 0.55, (0, (4, 2))), (100, 0.8, (0, (6, 2))), (399, 1.0, "-")):
        _, m_, sd, _ = hist[st]
        a1.plot(z, dens(dist.Normal(m_, sd), z), color=Q, lw=2.2, alpha=alpha, ls=ls,
                label=rf"step {st}: $q = \mathcal{{N}}({m_:.2f},\ {sd:.2f}^2)$")
    a1.set_ylim(0, 0.34)
    a1.set_xlabel(r"$\theta$")
    no_y(a1)
    a1.legend(loc="upper right", fontsize=12)
    a2.plot([h[0] for h in hist], [h[3] for h in hist], color=KL, lw=2.4)
    a2.set_xlabel("step")
    a2.set_ylabel(r"$\mathrm{KL}(q\,\Vert\,p)$")
    a2.grid(color=C["grid"])
    save(fig, "vik-g2-fit")

    # 2-D: a correlated target, diagonal vs full covariance
    rho = 0.9
    p2 = dist.MultivariateNormal(torch.zeros(2), torch.tensor([[1., rho], [rho, 1.]]))
    kl2 = lambda q: dist.kl_divergence(q, p2)
    h_diag = vu.fit(vu.TrainableGaussian2D((1.5, -1.0), 0.5, full=False), kl2, steps=600)
    h_full = vu.fit(vu.TrainableGaussian2D((1.5, -1.0), 0.5, full=True), kl2, steps=600)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.5), gridspec_kw=dict(wspace=0.25))
    for ax, hh, t in ((axs[0], h_diag, r"diagonal $q$"), (axs[1], h_full, r"full-covariance $q$")):
        vu.contour2d(ax, p2)
        vu.ellipse(ax, hh[-1]["q"].loc, hh[-1]["q"].covariance_matrix, Q)
        ax.set_title(t + rf": KL $= {hh[-1]['loss']:.3f}$", loc="left", fontsize=14)
        ax.grid(False)
    save(fig, "vik-g2-2d")
    NUM["g2d"] = dict(diag_sd=s(h_diag[-1]["q"].covariance_matrix[0, 0].sqrt(), 3), diag_kl=s(h_diag[-1]["loss"], 3),
                      full_kl=s(max(h_full[-1]["loss"], 0), 3), formula=s(math.sqrt(1 - rho ** 2), 3))


# ============================================================================ N4 · Monte Carlo KL (G3, G5)
def part4():
    p_s = vu.mog([0.5, 0.5], [-0.2, 1.0], [0.4, 0.5])
    q_1, q_2 = dist.Normal(3., 1.), dist.Normal(3., 4.5)
    z = vu.grid(-5, 15)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    shade(ax, z, dens(p_s, z), P)
    ax.plot(z, dens(p_s, z), color=P, lw=2.6, label=r"mixture $p$")
    ax.plot(z, dens(q_1, z), color=Q, lw=2.4, label=r"$q_1 = \mathcal{N}(3, 1^2)$")
    ax.plot(z, dens(q_2, z), color=LIK, lw=2.4, label=r"$q_2 = \mathcal{N}(3, 4.5^2)$")
    ax.set_xlabel(r"$\theta$")
    no_y(ax)
    ax.legend(loc="upper right")
    save(fig, "vik-g3")

    torch.manual_seed(0)                                   # as in notebook N4

    def kl_via_sampling(q, p, n_samples=100_000):
        sample_set = q.sample([n_samples])
        return torch.mean(q.log_prob(sample_set) - p.log_prob(sample_set))

    exact = dist.kl_divergence(q_1, q_2).item()
    mc12 = kl_via_sampling(q_1, q_2).item()
    m1, m2 = kl_via_sampling(q_1, p_s).item(), kl_via_sampling(q_2, p_s).item()
    NUM["g3"] = dict(exact=s(exact, 4), mc=s(mc12, 4), q1=s(m1, 2), q2=s(m2, 2), pml_q1="9.46", pml_q2="45.27", pml_mc="1.0273")

    torch.manual_seed(1)                                   # as in notebook N4 (G5)
    Ns = [10, 100, 1_000, 10_000]
    reps = torch.stack([torch.stack([kl_via_sampling(q_1, q_2, n) for _ in range(200)]) for n in Ns])
    NUM["g5"] = dict(spread=[s(v, 3) for v in reps.std(1)], ns=["10", "100", "1,000", "10,000"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(Ns, reps.mean(1), yerr=reps.std(1), fmt="o-", capsize=5, color=Q, lw=2, ms=7, label="estimate ± spread (200 runs)")
    ax.axhline(exact, color=KL, ls=(0, (5, 3)), lw=1.6, label=rf"exact {exact:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"samples $N$")
    ax.set_ylabel(r"$\mathrm{KL}(q_1\,\Vert\,q_2)$")
    ax.legend(loc="upper right")
    ax.grid(color=C["grid"])
    save(fig, "vik-g5")

    # darts for pi (PML mc_sampling_intro)
    xy = np.random.default_rng(42).random((1000, 2))
    inside = (xy ** 2).sum(1) <= 1
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c=np.where(inside, LIK, "#C7CCD4"), lw=0)
    t = np.linspace(0, math.pi / 2, 200)
    ax.plot(np.cos(t), np.sin(t), color=C["ink"], lw=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    save(fig, "vik-darts")
    NUM["darts"] = dict(inside=int(inside.sum()), est=s(4 * inside.mean(), 3))
    return p_s


# ============================================================================ N5 · reparameterisation (G4)
def part5(p_s):
    torch.manual_seed(42)
    eps = dist.Normal(0., 1.).sample()
    NUM["rep"] = dict(eps=s(eps, 4))

    def kl_loss(q, n_samples=5000):
        sample_set = q.rsample([n_samples])
        return torch.mean(q.log_prob(sample_set) - p_s.log_prob(sample_set))

    q_s = vu.TrainableNormal(loc=6.0, scale=0.1)
    h = vu.fit(q_s, kl_loss, steps=1500, lr=0.05)
    NUM["g4"] = dict(m=s(q_s.loc, 2), s=s(torch.nn.functional.softplus(q_s.raw_scale), 2), kl=s(h[-1]["loss"], 3),
                     pml_m="0.43", pml_s="0.72")
    z = vu.grid(-3, 8)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.22))
    pm = dens(p_s, z)
    shade(a1, z, pm, P)
    a1.plot(z, pm, color=P, lw=2.6, label=r"mixture $p$")
    for st, alpha, ls in ((0, 0.35, (0, (1, 2))), (40, 0.55, (0, (4, 2))), (80, 0.8, (0, (6, 2))), (1500, 1.0, "-")):
        qv = h[st]["q"]
        a1.plot(z, np.minimum(dens(qv, z), 3.9), color=Q, lw=2.2, alpha=alpha, ls=ls,
                label=rf"step {st:,}: $\mathcal{{N}}({qv.loc.item():.2f},\ {qv.scale.item():.2f}^2)$")
    a1.set_ylim(0, 1.25)
    a1.set_xlim(-3, 9)
    a1.set_xlabel(r"$\theta$")
    no_y(a1)
    a1.legend(loc="upper left", fontsize=11, ncol=2)
    a2.plot([e["step"] for e in h], [e["loss"] for e in h], color=KL, lw=1.6)
    a2.set_yscale("log")
    a2.set_xlabel("step")
    a2.set_ylabel(r"Monte Carlo $\mathrm{KL}(q\,\Vert\,p)$")
    a2.grid(color=C["grid"])
    save(fig, "vik-g4-fit")


# ============================================================================ N6 · ELBO
def part6(coin):
    torch.manual_seed(2)                                   # as in notebook N6

    def elbo(log_joint, q, n=200_000):
        theta = q.sample([n]).clamp(1e-6, 1 - 1e-6)
        return (log_joint(theta) - q.log_prob(theta)).mean().item()

    cands = [("Beta(10, 10)", dist.Beta(10., 10.)), ("Beta(30, 20)", dist.Beta(30., 20.)), ("Beta(22, 16)", coin.posterior)]
    rows = []
    for lab, qc in cands:
        e = elbo(coin.log_joint, qc)
        kl = dist.kl_divergence(qc, coin.posterior).item()
        rows.append(dict(q=lab, elbo=s(e, 3), kl=s(kl, 3), sum=s(e + kl, 3)))
    NUM["elbo"] = rows
    th = vu.grid(0.001, 0.999)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    shade(ax, th, dens(coin.posterior, th), P)
    ax.plot(th, dens(coin.posterior, th), color=P, lw=2.6, label=r"posterior $\mathrm{Beta}(22, 16)$")
    for (lab, qc), ls in zip(cands[:2], ((0, (5, 3)), (0, (2, 2)))):
        ax.plot(th, dens(qc, th), color=Q, lw=2.2, ls=ls, label=rf"$q = \mathrm{{{lab}}}$")
    ax.set_xlim(0.2, 0.95)
    ax.set_xlabel(r"$\theta$")
    no_y(ax)
    ax.legend(loc="upper left", fontsize=12)
    save(fig, "vik-elbo-qs")


# ============================================================================ N7 · the coin
def part7(coin):
    torch.manual_seed(0)                                   # as in notebook N7

    def neg_elbo(log_joint_fn, q, n_mc_samples=100):
        theta = q.rsample([n_mc_samples])
        return -(log_joint_fn(theta) - q.log_prob(theta)).mean()

    bad = neg_elbo(coin.log_joint, dist.Beta(0.1, 2.)).item()
    good = neg_elbo(coin.log_joint, coin.posterior).item()
    q_beta = vu.TrainableBeta(alpha=45.0, beta=45.0)
    h = vu.fit(q_beta, lambda q: neg_elbo(coin.log_joint, q), steps=3000, lr=0.05)
    qf = h[-1]["q"]
    NUM["coinvi"] = dict(bad=s(bad, 2), good=s(good, 3), a=s(qf.concentration1, 2), b=s(qf.concentration0, 2),
                         a1=s(qf.concentration1, 1), b1=s(qf.concentration0, 1), pml_bad1="218.88", pml_bad2="188.33",
                         pyro_a="22.9", pyro_b="16.7", final=s(np.mean([e["loss"] for e in h[-200:]]), 2))
    th = vu.grid(0.001, 0.999)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.22))
    shade(a1, th, dens(coin.posterior, th), P)
    a1.plot(th, dens(coin.posterior, th), color=P, lw=2.6, label=r"true posterior $\mathrm{Beta}(22, 16)$")
    a1.plot(th, dens(h[0]["q"], th), color=Q, lw=1.6, ls=(0, (2, 2)), alpha=0.6, label=r"start: $\mathrm{Beta}(45, 45)$")
    a1.plot(th, dens(qf, th), color=Q, lw=2.4, ls=(0, (5, 3)),
            label=rf"learned: $\mathrm{{Beta}}({qf.concentration1.item():.1f},\ {qf.concentration0.item():.1f})$")
    a1.set_xlabel(r"$\theta$")
    no_y(a1)
    a1.legend(loc="upper left", fontsize=12)
    st = np.array([e["step"] for e in h])
    ls = np.array([e["loss"] for e in h])
    a2.plot(st, ls, color="#C7CCD4", lw=0.8)
    a2.plot(st[49:], np.convolve(ls, np.ones(50) / 50, "valid"), color=KL, lw=2.2)
    a2.axhline(-coin.log_evidence, color=P, ls=(0, (5, 3)), lw=1.4)
    a2.text(3000, -coin.log_evidence - 0.06, rf"$-\log p(D) = {-coin.log_evidence:.2f}$", ha="right", va="top", fontsize=13, color=P)
    a2.set_ylim(21.2, 22.4)
    a2.set_xlabel("step")
    a2.set_ylabel(r"$-$ELBO (100 draws)")
    a2.grid(color=C["grid"])
    save(fig, "vik-coin-vi")


# ============================================================================ N8 · linear regression (G6)
def part8():
    out = {}
    for n_points in (20, 200):
        fig, row = plt.subplots(1, 2, figsize=(12.0, 3.9), gridspec_kw=dict(wspace=0.25, width_ratios=[1, 1.25]))
        torch.manual_seed(0)
        x_lin = torch.linspace(-5, 5, n_points)
        y_lin = 3 * x_lin + 4 + 5 * torch.randn(n_points)
        m = vu.LinReg(x_lin, y_lin, noise_sd=5.0, prior_sd=1.0)

        def neg_elbo_lr(q, n=100, m=m):
            theta = q.rsample([n])
            return -(m.log_joint(theta) - q.log_prob(theta)).mean()

        q_d = vu.fit(vu.TrainableGaussian2D((0., 0.), 1.0, full=False), neg_elbo_lr, steps=2000, lr=0.05)[-1]["q"]
        q_f = vu.fit(vu.TrainableGaussian2D((0., 0.), 1.0, full=True), neg_elbo_lr, steps=2000, lr=0.05)[-1]["q"]
        exact = m.posterior()
        mu, sd = exact.loc, exact.covariance_matrix.diag().sqrt()
        T0, T1 = torch.meshgrid(torch.linspace(mu[0] - 4 * sd[0], mu[0] + 4 * sd[0], 200),
                                torch.linspace(mu[1] - 4 * sd[1], mu[1] + 4 * sd[1], 200), indexing="xy")
        Z = exact.log_prob(torch.stack([T0, T1], -1)).exp()
        ax = row[0]
        ax.contourf(T0, T1, Z / Z.max(), levels=np.linspace(0.05, 1, 8), cmap="Blues")
        vu.ellipse(ax, q_d.loc, q_d.covariance_matrix, Q, label=r"diagonal $q$")
        vu.ellipse(ax, q_f.loc, q_f.covariance_matrix, LIK, label=r"full $q$", lw=1.6)
        ax.set_xlabel(r"intercept $\theta_0$")
        ax.set_ylabel(r"slope $\theta_1$")
        ax.legend(loc="upper right", fontsize=11)
        ax.set_title(f"{n_points} points: the exact posterior (blue)", loc="left", fontsize=14)
        ax = row[1]
        th = q_f.sample([1000])
        preds = th[:, :1] + th[:, 1:] * x_lin
        ax.scatter(x_lin, y_lin, s=10, color=C["ink"], alpha=0.5)
        ax.plot(x_lin, 3 * x_lin + 4, color=C["ink"], ls=(0, (4, 3)), lw=1.2, label=r"true $f(x) = 3x + 4$")
        ax.fill_between(x_lin, preds.mean(0) - 2 * preds.std(0), preds.mean(0) + 2 * preds.std(0), color=Q, alpha=0.22, lw=0)
        ax.plot(x_lin, preds.mean(0), color=Q, lw=2.2, label=r"mean of 1,000 lines from $q$")
        ax.legend(loc="upper left", fontsize=11)
        ax.set_xlabel(r"$x$")
        ax.set_title(r"predictions ($\pm 2$ sd, from $\theta$ only)", loc="left", fontsize=14)
        ax.grid(color=C["grid"])
        cf = q_f.covariance_matrix
        out[f"n{n_points}"] = dict(exact_m=[s(v, 2) for v in exact.loc], exact_s=[s(v, 3) for v in sd],
                                   q_m=[s(v, 2) for v in q_f.loc], q_s=[s(v, 3) for v in cf.diag().sqrt()],
                                   rho=s(cf[0, 1] / (cf[0, 0] * cf[1, 1]).sqrt(), 2),
                                   exact_rho=s(exact.covariance_matrix[0, 1] / (sd[0] * sd[1]), 2))
        save(fig, f"vik-linreg-vi-{n_points}")
    NUM["linreg"] = out


def main():
    for old in FIG.glob("vik-*.svg"):
        old.unlink()
    coin, target, xx = part1()
    part2(target, xx)
    part3()
    p_s = part4()
    part5(p_s)
    part6(coin)
    part7(coin)
    part8()
    (FIG / "numbers-vi.json").write_text(json.dumps(NUM, indent=1, ensure_ascii=False))
    print(json.dumps(NUM, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
