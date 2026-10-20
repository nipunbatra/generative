"""Figures and numbers for the maths lecture "KL divergence and variational inference" (vi.html).

    python scripts/make_figures_vi.py      # no trained models needed; about a minute on a laptop CPU

Everything here is a toy: a coin, two bumps, a few shapes in a plane. Writes figures/data/vik-*.svg and
figures/data/numbers-vi.json, which build_deck.py merges as {{vik.*}}. The live widgets in deck/widgets.js
compute the same quantities in the browser; the numbers quoted in slide text all come from here.

The running example is a coin tossed 10 times: 8 heads, 2 tails.
  * With a Beta(2, 2) prior on its chance of heads theta, the posterior is exact: Beta(10, 4).
  * With theta = sigmoid(z) and z ~ N(0, 1) (a one-number "decoder"), the posterior over z has no formula.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, FIG, plt, save  # noqa: E402  (shared palette and Computer Modern maths)
from matplotlib.patches import Ellipse  # noqa: E402

NUM = {}
torch.set_default_dtype(torch.float64)


def r(v, k=2):
    return float(round(float(v), k))


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def logsig(a):
    return -np.logaddexp(0.0, -a)


def sig(a):
    return 1 / (1 + np.exp(-a))


def lse(a):
    m = np.max(a)
    return m + math.log(np.sum(np.exp(a - m)))


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(x, a, b):
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - lbeta(a, b))


def no_y(ax):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


# ============================================================================ the coin
H, T = 8, 2
NT = H + T
SEQ = "HHTHHHHTHH"
assert SEQ.count("H") == H and len(SEQ) == NT


def part_bayes():
    """Part 1: three candidate coins by hand, then a Beta prior and its exact posterior."""
    pis = [0.2, 0.5, 0.8]
    lik = [p ** H * (1 - p) ** T for p in pis]
    joint = [v / 3 for v in lik]
    ev = sum(joint)
    post = [j / ev for j in joint]
    pred = sum(p * q for p, q in zip(pis, post))
    NUM["three"] = dict(
        lik=[f"{v * 1e4:.3f}" if v * 1e4 < 1 else f"{v * 1e4:.2f}" for v in lik],
        joint=[f"{v * 1e4:.3f}" if v * 1e4 < 1 else f"{v * 1e4:.2f}" for v in joint],
        ev=f"{ev * 1e4:.2f}", post=[f"{post[0]:.4f}", f"{post[1]:.3f}", f"{post[2]:.3f}"], pred=r(pred, 3),
        lik08=r(lik[2], 4))
    a0, b0, a1, b1 = 2, 2, 2 + H, 2 + T
    ev_beta = math.exp(lbeta(a1, b1) - lbeta(a0, b0))
    NUM["beta"] = dict(a=a1, b=b1, mean=r(a1 / (a1 + b1), 3), mode=r((a1 - 1) / (a1 + b1 - 2), 2), mle=r(H / NT, 1),
                       ev=f"{ev_beta:.5f}", log_ev=r(math.log(ev_beta), 3), sd=r(math.sqrt(a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1))), 3))

    x = np.linspace(0.001, 0.999, 500)
    panels = [("vik-prior", beta_pdf(x, a0, b0), C["z"], C["z_t"]),
              ("vik-lik", x ** H * (1 - x) ** T, C["dec"], C["dec_t"]),
              ("vik-post", beta_pdf(x, a1, b1), C["x"], C["x_t"])]
    for name, y, col, fill in panels:
        fig, ax = plt.subplots(figsize=(3.7, 2.5))
        ax.fill_between(x, 0, y, color=fill, lw=0)
        ax.plot(x, y, color=col, lw=2.6)
        if name == "vik-lik":
            ax.axvline(0.8, color=col, lw=1.2, ls=(0, (3, 3)))
            ax.text(0.79, y.max() * 1.02, r"peak at $\theta = 0.8$", ha="right", fontsize=13, color=col)
        if name == "vik-post":
            ax.plot(x, beta_pdf(x, a0, b0), color=C["z"], lw=1.4, ls=(0, (4, 3)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y.max() * 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(r"$\theta$")
        no_y(ax)
        save(fig, name)


# ============================================================================ the one-number decoder
ZG = np.linspace(-6, 6, 12001)
DZ = ZG[1] - ZG[0]


def log_prior(z):
    return -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)


def log_lik(z):
    return H * logsig(z) + T * logsig(-z)


def log_joint(z):
    return log_prior(z) + log_lik(z)


LJ = log_joint(ZG)
LOG_PX = lse(LJ) + math.log(DZ)
POST = np.exp(LJ - LOG_PX)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / math.sqrt(2 * math.pi)                 # E_{N(0,1)}[f] = sum(w f(x))


def e_q(f, m, s):
    return float(np.sum(GH_W * f(m + s * GH_X)))


def kl_prior(m, s):
    return 0.5 * (m ** 2 + s ** 2 - 1 - math.log(s ** 2))


def elbo(m, s):
    return e_q(log_lik, m, s) - kl_prior(m, s)


def best_gaussian(loglik, init=(0.0, 0.0)):
    """The best Gaussian q under the ELBO, by L-BFGS on the quadrature ELBO (prior N(0, 1))."""
    gx, gw = torch.tensor(GH_X), torch.tensor(GH_W)
    m = torch.tensor(init[0], requires_grad=True)
    ls = torch.tensor(init[1], requires_grad=True)
    opt = torch.optim.LBFGS([m, ls], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = ls.exp()
        z = m + s * gx
        val = -((gw * loglik(z)).sum() - 0.5 * (m ** 2 + s ** 2 - 1 - 2 * ls))
        val.backward()
        return val

    for _ in range(3):
        opt.step(closure)
    return m.item(), ls.exp().item()


def tlog_lik(z, h=H, t=T):
    return h * torch.nn.functional.logsigmoid(z) + t * torch.nn.functional.logsigmoid(-z)


def part_hard():
    """Part 2: the posterior over z has no formula; grid, Laplace, MCMC."""
    pm = float(np.sum(ZG * POST) * DZ)
    psd = math.sqrt(float(np.sum((ZG - pm) ** 2 * POST) * DZ))
    # Laplace: Newton to the peak of log p(x, z), then the curvature there
    z = 0.0
    for _ in range(50):
        g = -z + H - NT * sig(z)
        hss = -1 - NT * sig(z) * (1 - sig(z))
        z -= g / hss
    mode, curv = z, 1 + NT * sig(z) * (1 - sig(z))
    lap_sd = 1 / math.sqrt(curv)
    lap_logpx = log_joint(mode) + 0.5 * math.log(2 * math.pi) + math.log(lap_sd)
    # MCMC: random-walk Metropolis
    rng = np.random.default_rng(3)
    cur, cur_lp, acc, chain = 0.0, log_joint(0.0), 0, []
    steps, burn = 6000, 1000
    for t in range(steps):
        prop = cur + 1.0 * rng.normal()
        lp = log_joint(prop)
        if math.log(rng.random()) < lp - cur_lp:
            cur, cur_lp, acc = prop, lp, acc + 1
        chain.append(cur)
    chain = np.array(chain)
    kept = chain[burn:]
    m_vi, s_vi = best_gaussian(tlog_lik)
    NUM["coin"] = dict(log_px=r(LOG_PX, 3), px=f"{math.exp(LOG_PX):.5f}", post_mean=r(pm, 2), post_sd=r(psd, 3),
                       mode=r(mode, 2), lap_sd=r(lap_sd, 3), lap_logpx=r(lap_logpx, 3),
                       mcmc_steps=steps, mcmc_burn=burn, mcmc_acc=r(acc / steps, 2), mcmc_mean=r(kept.mean(), 2),
                       mcmc_sd=r(kept.std(), 2), lj1=r(log_joint(1.0), 3), lj0=r(log_joint(0.0), 3),
                       mle_z=r(math.log(H / T), 3), grid_n=len(ZG), pi_mode=r(sig(mode), 3))
    NUM["coin"]["pred_exact"] = r(float(np.sum(sig(ZG) * POST) * DZ), 3)

    # the story: prior over z, the decoder, the prior it implies on pi
    z = np.linspace(-4, 4, 400)
    pi = np.linspace(0.002, 0.998, 400)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.2), gridspec_kw=dict(wspace=0.35))
    axs[0].fill_between(z, 0, npdf(z, 0, 1), color=C["z_t"], lw=0)
    axs[0].plot(z, npdf(z, 0, 1), color=C["z"])
    axs[0].set_xlabel(r"$z$")
    axs[0].set_title(r"prior  $p(z) = \mathcal{N}(0, 1)$", fontsize=15, color=C["z"], loc="left")
    no_y(axs[0])
    axs[1].plot(z, sig(z), color=C["dec"])
    axs[1].axhline(0.5, color=C["line"], lw=1)
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_title(r"decoder  $\theta = \mathrm{sigmoid}(z)$", fontsize=15, color=C["dec"], loc="left")
    axs[1].grid(color=C["grid"])
    implied = npdf(np.log(pi / (1 - pi)), 0, 1) / (pi * (1 - pi))
    axs[2].fill_between(pi, 0, implied, color=C["z_t"], lw=0)
    axs[2].plot(pi, implied, color=C["z"])
    axs[2].plot(pi, beta_pdf(pi, 2, 2), color=C["ink3"], lw=1.6, ls=(0, (4, 3)))
    axs[2].text(0.5, 1.62, r"Beta$(2, 2)$, dashed", ha="center", fontsize=13, color=C["ink3"])
    axs[2].set_ylim(0, 1.95)
    axs[2].set_xlabel(r"$\theta$")
    axs[2].set_title(r"the prior it implies on $\theta$", fontsize=15, color=C["z"], loc="left")
    no_y(axs[2])
    save(fig, "vik-logit-story")

    # the joint: we can evaluate it anywhere; its area is p(x)
    zz = np.linspace(-2.5, 4, 600)
    jj = np.exp(log_joint(zz)) * 1e3
    for name, grid in (("vik-joint", False), ("vik-grid", True)):
        fig, ax = plt.subplots(figsize=(7.0, 3.7))
        if grid:
            edges = np.linspace(-2.5, 4, 27)
            mids = 0.5 * (edges[1:] + edges[:-1])
            ax.bar(mids, np.exp(log_joint(mids)) * 1e3, width=edges[1] - edges[0], color=C["x_t"], ec=C["x"], lw=0.8)
        else:
            ax.fill_between(zz, 0, jj, facecolor="#F3F4F6", lw=0, hatch="///", edgecolor="#C7CCD4")
            ax.text(-2.3, jj.max() * 1.08, r"shaded area $= p(x) = $ ?", fontsize=17, color=C["ink"])
        ax.plot(zz, jj, color=C["x"], lw=2.6)
        for zv in (0.0, 1.0):
            ax.plot([zv], [math.exp(log_joint(zv)) * 1e3], "o", color=C["ink"], ms=8, mec="white", mew=1.5, zorder=5)
        ax.set_xlim(-2.5, 4)
        ax.set_ylim(0, jj.max() * 1.3)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$p(x, z)$   $(\times 10^{-3})$")
        ax.grid(color=C["grid"])
        save(fig, name)

    # one slide, three ways: Laplace, MCMC, VI on the same posterior
    zz = np.linspace(-1.6, 3.4, 500)
    pz = np.exp(log_joint(zz) - LOG_PX)
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 3.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax in axs:
        ax.fill_between(zz, 0, pz, color=C["x_t"], lw=0)
        ax.plot(zz, pz, color=C["x"], lw=1.4)
        ax.set_xlim(-1.6, 3.4)
        ax.set_xlabel(r"$z$")
        no_y(ax)
    axs[0].set_ylim(0, pz.max() * 1.45)
    axs[0].plot(zz, npdf(zz, mode, lap_sd), color=C["z"], lw=2.6)
    axs[0].axvline(mode, color=C["z"], lw=1.2, ls=(0, (3, 3)))
    axs[0].set_title("Laplace: a Gaussian at the peak", loc="left", fontsize=15, color=C["ink"])
    axs[1].hist(kept, bins=np.linspace(-1.6, 3.4, 41), density=True, color=C["z"], alpha=.55, ec="white", lw=0.6)
    axs[1].set_title("MCMC: a random walk's visits", loc="left", fontsize=15, color=C["ink"])
    ins = axs[1].inset_axes([0.6, 0.58, 0.38, 0.36])
    ins.plot(np.arange(400), chain[:400], color=C["z"], lw=0.9)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(C["line"])
    ins.set_title("first 400 steps", fontsize=11, color=C["ink3"], pad=2)
    axs[2].plot(zz, npdf(zz, m_vi, s_vi), color=C["enc"], lw=2.6)
    axs[2].set_title("VI: the best Gaussian $q$", loc="left", fontsize=15, color=C["ink"])
    save(fig, "vik-three-ways")
    return m_vi, s_vi


def grid_cost():
    rows = []
    for d in (1, 2, 4, 10, 32):
        rows.append(dict(d=d, n=f"10^{{{2 * d}}}" if d > 1 else "100"))
    NUM["grid"] = rows


# ============================================================================ averages and sampling
def part_mc():
    mu, sd = 1.0, 0.5
    exact = mu ** 2 + sd ** 2
    per = math.sqrt(4 * mu ** 2 * sd ** 2 + 2 * sd ** 4)       # sd of z^2 for one draw
    rng = np.random.default_rng(11)
    first = np.round(mu + sd * rng.normal(size=5), 2)          # shown to 2 decimals, and used exactly as shown
    spreads = {}
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(hspace=0.25))
    for ax, S in zip(axs, (10, 100, 1000)):
        est = ((mu + sd * rng.normal(size=(2000, S))) ** 2).mean(1)
        spreads[S] = est.std()
        ax.hist(est, bins=np.linspace(0.4, 2.1, 86), color=C["dec"], alpha=.75, ec="none")
        ax.axvline(exact, color=C["ink"], lw=1.6)
        ax.text(2.1, ax.get_ylim()[1] * 0.55, rf"$S = {S:,}$:  spread ${est.std():.3f}$".replace(",", "{,}"),
                ha="right", fontsize=14, color=C["ink"])
        no_y(ax)
    axs[-1].set_xlabel(r"the estimate $\frac{1}{S}\sum_s z_s^2$  (2,000 repeats each; exact answer $1.25$)")
    save(fig, "vik-mc-error")
    NUM["mc"] = dict(exact=exact, per_sd=r(per, 3), first=[r(v, 2) for v in first], first_f=[f"{v ** 2:.3f}" for v in first],
                     first_mean=r(np.mean(first ** 2), 3), s10=r(spreads[10], 3), s100=r(spreads[100], 3), s1000=r(spreads[1000], 3),
                     th10=r(per / math.sqrt(10), 3), th100=r(per / math.sqrt(100), 3), th1000=r(per / math.sqrt(1000), 3))
    big = ((mu + sd * np.random.default_rng(12).normal(size=100000)) ** 2).mean()
    NUM["mc"]["big"] = r(big, 3)
    # KL as an average: 1,000 digits drawn from P
    d = np.random.default_rng(5).random(1000) < 0.8
    terms = np.where(d, math.log(0.8 / 0.6), math.log(0.2 / 0.4))
    NUM["mc"]["kl1000"] = r(terms.mean(), 3)
    NUM["mc"]["kl1000_ones"] = int(d.sum())


def part_darts():
    """Monte Carlo in one picture (from the PML course's mc_sampling_intro notebook): darts in a square estimate pi."""
    rows = []
    for n in (10, 100, 1000, 10000, 100000):                  # six seeds per size, as in the notebook
        est = [4 * ((np.random.default_rng(seed).random((n, 2)) ** 2).sum(1) <= 1).mean() for seed in range(6)]
        rows.append(dict(n=f"{n:,}", mean=r(np.mean(est), 3), sd=r(np.std(est), 3)))
    xy = np.random.default_rng(42).random((1000, 2))
    inside = (xy ** 2).sum(1) <= 1
    NUM["darts"] = dict(rows=rows, in1000=int(inside.sum()), est1000=r(4 * inside.mean(), 3))
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    n = 1000
    ax.scatter(xy[:n, 0], xy[:n, 1], s=9, c=np.where(inside[:n], C["dec"], C["grey"] if "grey" in C else "#C7CCD4"), lw=0)
    t = np.linspace(0, math.pi / 2, 200)
    ax.plot(np.cos(t), np.sin(t), color=C["ink"], lw=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    save(fig, "vik-darts")


def part_bits():
    """Entropy, cross-entropy and KL as code lengths (from the PML course's information-theory slides)."""
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.full(4, 0.25)
    NUM["bits"] = dict(H=r(-(p * np.log2(p)).sum(), 2), CE=r(-(p * np.log2(q)).sum(), 2), KL=r((p * np.log2(p / q)).sum(), 2),
                       KLnats=r((p * np.log(p / q)).sum(), 3))


# ============================================================================ KL figures (copied from make_figures.py)
def part_kl():
    P = np.array([0.8, 0.2])
    qs = np.linspace(0.005, 0.995, 400)
    kl_pq = P[0] * np.log(P[0] / qs) + P[1] * np.log(P[1] / (1 - qs))
    kl_qp = qs * np.log(qs / P[0]) + (1 - qs) * np.log((1 - qs) / P[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(qs, kl_pq, color=C["loss"])
    ax.plot(qs, kl_qp, color=C["ink2"], ls=(0, (5, 3)))
    ax.text(0.085, 2.0, r"$\mathrm{KL}(P\,\Vert\,Q)$", color=C["loss"], fontsize=17)
    ax.text(0.02, 0.62, r"$\mathrm{KL}(Q\,\Vert\,P)$", color=C["ink2"], fontsize=17)
    a = P[0] * math.log(P[0] / 0.6) + P[1] * math.log(P[1] / 0.4)
    b = 0.6 * math.log(0.6 / P[0]) + 0.4 * math.log(0.4 / P[1])
    ax.plot([0.6, 0.6], [a, b], "o", color=C["ink"], ms=7)
    ax.annotate(rf"$q = 0.6$:  ${a:.3f}$ vs ${b:.3f}$", xy=(0.6, b), xytext=(0.35, 1.25), fontsize=14, color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink3"]))
    ax.plot([0.8], [0], "o", color=C["x"], ms=8)
    ax.annotate(r"$q = 0.8 = p$: both are 0", xy=(0.8, 0), xytext=(0.6, 0.55), fontsize=14, color=C["x"],
                arrowprops=dict(arrowstyle="-", color=C["x"]))
    ax.set_ylim(0, 2.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"model's probability of a 1, $q$")
    ax.set_ylabel("nats")
    ax.grid(color=C["grid"])
    save(fig, "vik-kl-vs-q")

    pp = np.linspace(0.01, 1, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(pp, -np.log(pp), color=C["ink2"], lw=2.4)
    for v, col, lab, dy in ((0.8, C["x"], r"a 1 under $P$", 0.18), (0.6, C["dec"], r"a 1 under $Q$", 0.22),
                            (0.4, C["dec"], r"a 7 under $Q$", 0.25), (0.2, C["x"], r"a 7 under $P$", 0.25)):
        ax.plot([v], [-math.log(v)], "o", color=col, ms=9, mec="white", mew=1.5)
        ax.text(v + 0.03, -math.log(v) + dy, lab + rf": ${-math.log(v):.2f}$", fontsize=13, color=col)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"probability the model gave the digit, $p$")
    ax.set_ylabel(r"surprise $= -\log p$")
    ax.grid(color=C["grid"])
    save(fig, "vik-surprise-curve")

    rr = np.linspace(0.08, 3.0, 400)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(rr, np.log(rr), color=C["dec"], lw=2.6)
    ax.plot(rr, rr - 1, color=C["z"], lw=2.2, ls=(0, (5, 3)))
    ax.plot([1], [0], "o", color=C["ink"], ms=8)
    ax.text(2.3, 1.62, r"$r - 1$", color=C["z"], fontsize=17)
    ax.text(2.55, 0.62, r"$\log r$", color=C["dec"], fontsize=17)
    for v, lab, ty in ((0.75, "a 1", -0.62), (2.0, "a 7", 0.38)):
        ax.plot([v, v], [math.log(v), v - 1], color=C["loss"], lw=2.4)
        ax.text(v + 0.05, ty, rf"{lab}: $r = {v}$", color=C["loss"], fontsize=14)
    ax.set_xlim(0, 3.05)
    ax.set_ylim(-1.6, 2.1)
    ax.axhline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"$r = Q_i / P_i$")
    ax.grid(color=C["grid"])
    save(fig, "vik-kl-tangent")

    nn = np.arange(0, 101)
    k = 0.8 * math.log(0.8 / 0.6) + 0.2 * math.log(0.2 / 0.4)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.semilogy(nn, np.exp(nn * k), color=C["loss"], lw=2.6)
    for n_ in (10, 50, 100):
        v = math.exp(n_ * k)
        lab = f"{v:,.0f}×" if v > 1000 else (f"{v:.1f}×" if v < 10 else f"{v:.0f}×")
        ax.plot([n_], [v], "o", color=C["loss"], ms=9, mec="white", mew=1.5)
        ax.text(n_ - 2, v * 1.9, lab, color=C["ink"], fontsize=15, ha="right" if n_ == 100 else "center")
    ax.set_xlabel("number of digits seen, $n$")
    ax.set_ylabel(r"$p(\text{data}\mid P)\ /\ p(\text{data}\mid Q)$")
    ax.set_ylim(0.8, 3e4)
    ax.grid(color=C["grid"], which="major")
    save(fig, "vik-kl-evidence")
    NUM["kl"] = dict(r10=r(math.exp(10 * k), 1), r50=int(round(math.exp(50 * k))), r100=f"{math.exp(100 * k):,.0f}")

    zg = np.linspace(-4, 4, 800)
    p_, q_ = npdf(zg, 0, 1), npdf(zg, 1, .5)
    integrand = q_ * np.log(q_ / p_)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    a1.plot(zg, p_, color=C["z"])
    a1.plot(zg, q_, color=C["enc"])
    a1.text(-3.6, 0.3, r"$p(z) = \mathcal{N}(0, 1)$", color=C["z"], fontsize=15)
    a1.text(1.7, 0.62, r"$q(z) = \mathcal{N}(1,\,0.5^2)$", color=C["enc"], fontsize=15)
    no_y(a1)
    a2.fill_between(zg, 0, integrand, where=integrand >= 0, color=C["loss_t"], lw=0)
    a2.fill_between(zg, 0, integrand, where=integrand < 0, color="#E5E7EB", lw=0)
    a2.plot(zg, integrand, color=C["loss"], lw=2)
    a2.axhline(0, color=C["ink3"], lw=0.8)
    a2.text(-3.7, 0.3, r"$q(z)\,\log\frac{q(z)}{p(z)}$", color=C["loss"], fontsize=17)
    a2.text(1.9, 0.3, f"area = {np.trapezoid(integrand, zg):.3f} nats", color=C["ink"], fontsize=15, fontweight="bold")
    no_y(a2)
    a2.set_xlabel(r"$z$")
    save(fig, "vik-kl-gauss")
    NUM["kl"]["gauss_num"] = r(np.trapezoid(integrand, zg), 4)
    NUM["kl"]["gauss_qp"] = r(0.5 * (1 + 0.25 - 1 - math.log(0.25)), 3)
    NUM["kl"]["gauss_pq"] = r(math.log(0.5) + (1 + 1) / (2 * 0.25) - 0.5, 3)
    Qf = (0.99, 0.01)                                     # a model that almost forgets 7s
    NUM["kl"]["forget_pq"] = r(0.8 * math.log(0.8 / Qf[0]) + 0.2 * math.log(0.2 / Qf[1]), 2)
    NUM["kl"]["forget_qp"] = r(Qf[0] * math.log(Qf[0] / 0.8) + Qf[1] * math.log(Qf[1] / 0.2), 2)
    NUM["kl"]["forget_cost"] = r(math.log(0.2 / 0.01), 1)

    # forward vs reverse KL, one Gaussian fitted to two bumps at -2 and +2
    xs = np.linspace(-5, 5, 600)
    mix = 0.5 * npdf(xs, -2, .5) + 0.5 * npdf(xs, 2, .5)

    def fit(direction):
        best = None
        for m in np.linspace(-3, 3, 241):
            for s in np.linspace(0.2, 3, 141):
                qd, pd = npdf(xs, m, s) + 1e-300, mix + 1e-300
                v = np.trapezoid(pd * np.log(pd / qd), xs) if direction == "fwd" else np.trapezoid(qd * np.log(qd / pd), xs)
                if best is None or v < best[0]:
                    best = (v, m, s)
        return best

    fwd, rev = fit("fwd"), fit("rev")
    # no formula for KL(Gaussian || mixture): estimate it by sampling (the PML notebook's kl_via_sampling)
    rng = np.random.default_rng(8)

    def mix_logpdf(z):
        return np.logaddexp(np.log(0.5) + np.log(npdf(z, -2, .5)), np.log(0.5) + np.log(npdf(z, 2, .5)))

    samp = {}
    for nm, (m, s) in (("wide", (0.0, 2.06)), ("one", (-2.0, 0.5))):
        z = m + s * rng.normal(size=100000)
        lq = -0.5 * ((z - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
        wide = np.linspace(-16, 16, 32001)                    # q's tails reach far past the plot
        qd = npdf(wide, m, s)
        samp[nm] = (r(np.mean(lq - mix_logpdf(z)), 3), r(np.trapezoid(qd * (np.log(qd) - mix_logpdf(wide)), wide), 3))
    NUM["klsamp"] = dict(wide=samp["wide"][0], wide_grid=samp["wide"][1], one=samp["one"][0], one_grid=samp["one"][1])
    NUM["fit"] = dict(fwd_mu=r(fwd[1], 2), fwd_sd=r(fwd[2], 2), rev_mu=r(rev[1], 2), rev_sd=r(rev[2], 2),
                      fwd_kl=r(fwd[0], 3), rev_kl=r(rev[0], 3))
    for nm, (v, m, s) in (("fwd", fwd), ("rev", rev)):
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.fill_between(xs, 0, mix, color=C["x_t"])
        ax.plot(xs, mix, color=C["x"], lw=1.6)
        ax.plot(xs, npdf(xs, m, s), color=C["enc"], lw=2.8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 1.2)
        no_y(ax)
        ax.set_xlabel(r"$z$")
        ax.text(0.5 if nm == "fwd" else 0, 1.08, rf"best $q = \mathcal{{N}}({m:.1f},\,{s:.2f}^2)$", fontsize=16, color=C["enc"], ha="center")
        save(fig, f"vik-kl-{nm}")


# ============================================================================ VI on the coin
def part_vi(m_vi, s_vi):
    e0 = elbo(0.0, 1.0)
    cand = [(0.0, 1.0), (1.39, 0.3), (m_vi, s_vi)]
    NUM["vi"] = dict(m=r(m_vi, 2), s=r(s_vi, 2), s3=r(s_vi, 3), elbo=r(elbo(m_vi, s_vi), 3), gap=r(LOG_PX - elbo(m_vi, s_vi), 4),
                     rec=r(e_q(log_lik, m_vi, s_vi), 3), klp=r(kl_prior(m_vi, s_vi), 3),
                     elbo0=r(e0, 3), gap0=r(LOG_PX - e0, 3), rec0=r(e0, 3),
                     cand=[dict(m=f"{m:.2f}", s=f"{s:.2f}", kl=f"{LOG_PX - elbo(m, s):.3f}", elbo=f"{elbo(m, s):.3f}".replace("-", "−"),
                                rec=f"{e_q(log_lik, m, s):.3f}".replace("-", "−"), klp=f"{kl_prior(m, s):.3f}") for m, s in cand],
                     pred=r(e_q(sig, m_vi, s_vi), 3))

    zz = np.linspace(-2.2, 3.6, 500)
    pz = np.exp(log_joint(zz) - LOG_PX)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.fill_between(zz, 0, pz, color=C["x_t"], lw=0)
    ax.plot(zz, pz, color=C["x"], lw=1.6)
    styles = [(0, (2, 2)), (0, (6, 3)), "-"]
    spots = [(-1.45, 0.33), (1.62, 1.22), (-0.05, 0.6)]
    for (m, s), ls, lab, (tx, ty) in zip(cand, styles, "abc", spots):
        ax.plot(zz, npdf(zz, m, s), color=C["enc"], lw=2.4, ls=ls)
        ax.text(tx, ty, rf"$q_{lab}$", fontsize=18, color=C["enc"])
    ax.set_ylim(0, 1.5)
    ax.set_xlim(-2.2, 3.6)
    ax.set_xlabel(r"$z$")
    no_y(ax)
    save(fig, "vik-candidates")

    # stochastic VI with the reparameterisation trick: Adam on (m, log s), 10 draws per step
    def run(S, steps=400, lr=0.05, seed=0):
        g = torch.Generator().manual_seed(seed)
        m = torch.tensor(0.0, requires_grad=True)
        ls = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.Adam([m, ls], lr=lr)
        path = []
        for step in range(steps + 1):
            path.append((step, m.item(), ls.exp().item(), elbo(m.item(), ls.exp().item())))
            eps = torch.randn(S, generator=g)
            z = m + ls.exp() * eps
            est = (torch.distributions.Normal(0.0, 1.0).log_prob(z) + tlog_lik(z)
                   - torch.distributions.Normal(m, ls.exp()).log_prob(z)).mean()
            opt.zero_grad()
            (-est).backward()
            opt.step()
        return path

    path = run(10)
    path1 = run(1)
    keep = (0, 10, 25, 50, 100, 400)
    NUM["run"] = dict(S=10, lr=0.05, steps=400,
                      table=[dict(step=p[0], m=r(p[1], 2), s=r(p[2], 2), elbo=r(p[3], 3), gap=r(LOG_PX - p[3], 3)) for p in path if p[0] in keep],
                      s1_m=r(np.mean([p[1] for p in path1[-100:]]), 2), s1_s=r(np.mean([p[2] for p in path1[-100:]]), 2),
                      s1_elbo=r(np.mean([p[3] for p in path1[-100:]]), 3))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.22))
    a1.fill_between(zz, 0, pz, color=C["x_t"], lw=0)
    a1.plot(zz, pz, color=C["x"], lw=1.4)
    for k, alpha, lab in ((0, .35, (-2.05, 0.0, "step 0")), (10, .55, (-1.5, 0.0, "step 10")),
                          (25, .75, (0.3, 0.03, "steps 25 and 400")), (400, 1, None)):
        _, m, s, _ = path[k]
        a1.plot(zz, npdf(zz, m, s), color=C["enc"], lw=2.4, alpha=alpha)
        if lab:
            a1.text(m + lab[0], npdf(m, m, s) + lab[1], lab[2], fontsize=13, color=C["enc"])
    a1.set_xlim(-2.2, 3.6)
    a1.set_ylim(0, 0.95)
    a1.set_xlabel(r"$z$")
    no_y(a1)
    st = [p[0] for p in path]
    a2.plot(st, [p[3] for p in path], color=C["dec"], lw=2.4)
    a2.axhline(LOG_PX, color=C["x"], lw=1.6, ls=(0, (5, 3)))
    a2.text(400, LOG_PX + 0.06, rf"$\log p(x) = {LOG_PX:.3f}$", ha="right", fontsize=14, color=C["x"])
    a2.text(30, path[0][3] - 0.02, r"ELBO", fontsize=14, color=C["dec"], va="top")
    a2.set_xlabel("step")
    a2.set_ylabel("nats")
    a2.set_ylim(path[0][3] - 0.2, LOG_PX + 0.25)
    a2.grid(color=C["grid"])
    save(fig, "vik-ascent")

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.fill_between(zz, 0, pz, color=C["x_t"], lw=0)
    ax.plot(zz, pz, color=C["x"], lw=2.0)
    ax.plot(zz, npdf(zz, m_vi, s_vi), color=C["enc"], lw=2.2, ls=(0, (5, 3)))
    ax.set_xlim(-2.2, 3.6)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel(r"$z$")
    no_y(ax)
    save(fig, "vik-result")

    # Monte Carlo ELBO at q = N(0, 1): what the code computes
    rng = np.random.default_rng(21)

    def mc_elbo(m, s, S):
        z = m + s * rng.normal(size=S)
        return float(np.mean(log_joint(z) - (-0.5 * ((z - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi))))

    NUM["mcelbo"] = dict(s10=r(mc_elbo(0, 1, 10), 3), s100=r(mc_elbo(0, 1, 100), 3), s10000=r(mc_elbo(0, 1, 10000), 3),
                         exact=r(e0, 3), spread10=r(np.std([mc_elbo(0, 1, 10) for _ in range(2000)]), 3))
    # the two lines of code at one z: z = 1.0
    NUM["mcelbo"]["z1_lp"] = r(log_prior(1.0), 3)
    NUM["mcelbo"]["z1_ll"] = r(log_lik(1.0), 3)
    NUM["mcelbo"]["z1_lq"] = r(log_prior(1.0), 3)


# ============================================================================ gradients of an average
def part_grad():
    mu, sd = 1.0, 0.5
    rng = np.random.default_rng(7)
    eps = rng.normal(size=1_000_000)
    z = mu + sd * eps
    g_rep = 2 * z
    g_sf = z ** 2 * (z - mu) / sd ** 2
    NUM["grad"] = dict(exact=2.0, rep_mean=r(g_rep.mean(), 3), sf_mean=r(g_sf.mean(), 3),
                       rep_sd=r(g_rep.std(), 2), sf_sd=r(g_sf.std(), 2), ratio=int(round(g_sf.var() / g_rep.var())))
    # one draw, eps = -0.4, both estimators
    e1 = -0.4
    z1 = mu + sd * e1
    NUM["grad"]["one_z"] = r(z1, 2)
    NUM["grad"]["one_rep"] = r(2 * z1, 2)
    NUM["grad"]["one_sf"] = r(z1 ** 2 * (z1 - mu) / sd ** 2, 3)

    fig, axs = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True, gridspec_kw=dict(hspace=0.3))
    bins = np.linspace(-6, 12, 145)
    for ax, g, col, lab in ((axs[0], g_sf, C["ink2"], "score function"), (axs[1], g_rep, C["enc"], "reparameterisation")):
        ax.hist(g[:20000], bins=bins, color=col, alpha=.7, ec="none", density=True)
        ax.axvline(2.0, color=C["loss"], lw=1.8)
        ax.text(12, ax.get_ylim()[1] * 0.6, f"{lab}\nspread {g.std():.2f}", ha="right", fontsize=14, color=col)
        no_y(ax)
    axs[0].text(2.25, axs[0].get_ylim()[1] * 0.85, r"exact: $2$", fontsize=14, color=C["loss"])
    axs[1].set_xlabel(r"one-draw estimate of $\frac{\partial}{\partial\mu}\,\mathbb{E}[z^2]$,  20,000 draws each")
    save(fig, "vik-grad-hist")

    # on the coin: d ELBO / d m at q = N(0, 1), per draw
    m, s = 0.0, 1.0
    z = m + s * rng.normal(size=400_000)
    dlj = -z + H - NT * sig(z)
    lq = -0.5 * ((z - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
    sf = (log_joint(z) - lq) * (z - m) / s ** 2
    exact = e_q(lambda u: -u + H - NT * sig(u), m, s)
    NUM["grad"]["coin_exact"] = r(exact, 3)
    NUM["grad"]["coin_rep_sd"] = r(dlj.std(), 2)
    NUM["grad"]["coin_sf_sd"] = r(sf.std(), 2)
    NUM["grad"]["coin_ratio"] = int(round(sf.var() / dlj.var()))
    Ss = np.array([1, 3, 10, 30, 100, 300, 1000])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for g, col, lab in ((sf, C["ink2"], "score function"), (dlj, C["enc"], "reparameterisation")):
        spread = g.std() / np.sqrt(Ss)
        ax.loglog(Ss, spread, "o-", color=col, ms=6)
        ax.text(Ss[-1] * 1.4, spread[-1], lab, color=col, fontsize=14, va="center")
    ax.set_xlim(0.7, 3e4)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel(r"draws per step, $S$")
    ax.set_ylabel(r"spread of the gradient estimate")
    ax.grid(color=C["grid"], which="major")
    save(fig, "vik-grad-S")

    # reparameterisation as small multiples (copied from make_figures.py)
    eps_dots = np.array([-1.7, -1.1, -0.7, -0.4, -0.1, 0.25, 0.6, 1.0, 1.6])
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    rows = [(r"$\varepsilon \sim \mathcal{N}(0, 1)$", eps_dots, C["z"], 4.0, (0, 1)),
            (r"$\sigma\,\varepsilon$", sd * eps_dots, C["enc"], 2.0, (0, sd)),
            (r"$z = \mu + \sigma\varepsilon$", mu + sd * eps_dots, C["enc"], 0.0, (mu, sd))]
    dens_x = np.linspace(-2.6, 2.9, 300)
    for i, (lab, vals, col, y, (m0, s0)) in enumerate(rows):
        ax.plot([-2.6, 2.9], [y, y], color=C["line"], lw=1.4, zorder=1)
        dens = npdf(dens_x, m0, s0)
        ax.fill_between(dens_x, y, y + 0.9 * dens / dens.max(), color=col, alpha=.10, lw=0)
        ax.plot(dens_x, y + 0.9 * dens / dens.max(), color=col, lw=1.4, alpha=.7)
        for v in vals:
            special = abs(v - vals[3]) < 1e-9
            ax.plot([v], [y], "o", color=C["loss"] if special else col, ms=13 if special else 10, mec="white", mew=1.5, zorder=4)
        ax.text(-2.75, y + 0.15, lab, ha="right", va="center", fontsize=17, color=C["ink"])
        ax.text(vals[3], y - 0.45, rf"${vals[3]:.1f}$", ha="center", fontsize=17, color=C["loss"])
        if i < 2:
            for a_, b_ in zip(vals, rows[i + 1][1]):
                ax.annotate("", xy=(b_, rows[i + 1][3] + 0.12), xytext=(a_, y - 0.12),
                            arrowprops=dict(arrowstyle="-|>", color="#B7BDC7", lw=1.1, mutation_scale=10), zorder=2)
    ax.text(3.0, 3.0, r"$\times\,\sigma$  (squeeze, $\sigma = 0.5$)", fontsize=15, color=C["enc"], va="center")
    ax.text(3.0, 1.0, r"$+\,\mu$  (shift, $\mu = 1$)", fontsize=15, color=C["enc"], va="center")
    ax.set_xlim(-2.8, 4.6)
    ax.set_ylim(-0.6, 5.1)
    ax.axis("off")
    save(fig, "vik-reparam-dots")


# ============================================================================ beyond one number
def fit_2d(logp, family, init_m=(0.3, 0.2), steps=2500, lr=0.03, seed=0, n=4096):
    """Reverse KL: the best Gaussian q (diagonal or full covariance) for a normalised 2-D log density."""
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(n, 2, generator=g)
    m = torch.tensor(init_m, requires_grad=True)
    ld = torch.tensor([-0.5, -0.5], requires_grad=True)
    off = torch.tensor(0.0, requires_grad=True)
    params = [m, ld] + ([off] if family == "full" else [])
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        L = torch.diag(ld.exp())
        if family == "full":
            L = L + torch.tensor([[0.0, 0.0], [1.0, 0.0]]) * off
        z = m + eps @ L.T
        ent = ld.sum() + math.log(2 * math.pi) + 1          # entropy of N(m, L L^T)
        kl = -(logp(z).mean() + ent)
        opt.zero_grad()
        kl.backward()
        opt.step()
    with torch.no_grad():                                     # score the fit on fresh draws, not the ones it was fitted to
        L = torch.diag(ld.exp())
        if family == "full":
            L = L + torch.tensor([[0.0, 0.0], [1.0, 0.0]]) * off
        z = m + torch.randn(200_000, 2, generator=g) @ L.T
        kl = -(logp(z).mean() + ld.sum() + math.log(2 * math.pi) + 1)
    return m.detach().numpy(), (L @ L.T).numpy(), kl.item()


def ellipses(ax, m, cov, col, lw=2.4, ls="-"):
    vals, vecs = np.linalg.eigh(cov)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    for k in (1, 2):
        ax.add_patch(Ellipse(m, 2 * k * math.sqrt(vals[1]), 2 * k * math.sqrt(vals[0]), angle=ang, fill=False, ec=col, lw=lw, ls=ls))
    ax.plot(*m, "o", color=col, ms=6)


def density_panel(ax, logp_np, lim=3.2):
    g = np.linspace(-lim, lim, 300)
    X, Y = np.meshgrid(g, g)
    D = np.exp(logp_np(np.stack([X, Y], -1)))
    ax.contourf(X, Y, D / D.max(), levels=np.linspace(0.04, 1, 9), cmap="Blues", alpha=.9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)


RHO = 0.9


def corr_logp(z):
    det = 1 - RHO ** 2
    q = (z[..., 0] ** 2 - 2 * RHO * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / det
    lib = torch if isinstance(z, torch.Tensor) else np
    return -0.5 * q - math.log(2 * math.pi) - 0.5 * math.log(det) + 0 * lib.sum(z, -1)


def banana_logp(z):
    lib = torch if isinstance(z, torch.Tensor) else np
    a, b = z[..., 0], z[..., 1]
    mu2 = 0.6 * a ** 2 - 1.2
    return (-0.5 * (a / 1.2) ** 2 - math.log(1.2) - 0.5 * ((b - mu2) / 0.45) ** 2 - math.log(0.45) - math.log(2 * math.pi)) + 0 * lib.sum(z, -1)


def bumps_logp(z):
    lib = torch if isinstance(z, torch.Tensor) else np
    s = 0.55
    out = []
    for c in (-1.5, 1.5):
        out.append(-0.5 * ((z[..., 0] - c) ** 2 + (z[..., 1] - c) ** 2) / s ** 2 - 2 * math.log(s) - math.log(2 * math.pi) + math.log(0.5))
    a, b = out
    mx = lib.maximum(a, b)
    return mx + lib.log(lib.exp(a - mx) + lib.exp(b - mx))


def part_beta_vi():
    """VI where the answer is known: a Beta q on Part 1's coin (as in the PML course's variational-inference notebook)."""
    torch.manual_seed(0)
    la = torch.tensor(math.log(45.0), requires_grad=True)
    lb = torch.tensor(math.log(45.0), requires_grad=True)
    opt = torch.optim.Adam([la, lb], lr=0.05)
    prior = torch.distributions.Beta(2.0, 2.0)
    trace = []
    for step in range(3001):
        q = torch.distributions.Beta(la.exp(), lb.exp())
        th = q.rsample((100,)).clamp(1e-6, 1 - 1e-6)
        elbo = (prior.log_prob(th) + H * torch.log(th) + T * torch.log1p(-th) - q.log_prob(th)).mean()
        if step % 500 == 0:
            trace.append((step, la.exp().item(), lb.exp().item()))
        opt.zero_grad()
        (-elbo).backward()
        opt.step()
    a, b = la.exp().item(), lb.exp().item()
    # exact ELBO of the final q, by quadrature on a fine grid
    th = np.linspace(1e-6, 1 - 1e-6, 200001)
    qd = beta_pdf(th, a, b)
    lj = np.log(beta_pdf(th, 2, 2)) + H * np.log(th) + T * np.log1p(-th)
    e = np.trapezoid(qd * (lj - np.log(qd + 1e-300)), th)
    log_ev = lbeta(2 + H, 2 + T) - lbeta(2, 2)
    NUM["betavi"] = dict(a=r(a, 2), b=r(b, 2), elbo=r(e, 3), gap=r(log_ev - e, 4), steps=3000, start=45,
                         trace=[dict(step=s_, a=r(a_, 1), b=r(b_, 1)) for s_, a_, b_ in trace])
    x = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(x, 0, beta_pdf(x, 2 + H, 2 + T), color=C["x_t"], lw=0)
    ax.plot(x, beta_pdf(x, 2 + H, 2 + T), color=C["x"], lw=2.0)
    ax.plot(x, beta_pdf(x, 45, 45), color=C["enc"], lw=1.6, ls=(0, (2, 2)), alpha=.6)
    ax.plot(x, beta_pdf(x, a, b), color=C["enc"], lw=2.4, ls=(0, (5, 3)))
    ax.text(0.44, beta_pdf(0.5, 45, 45) * 0.8, r"start: Beta$(45, 45)$", ha="right", fontsize=13, color=C["enc"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\theta$")
    no_y(ax)
    save(fig, "vik-betavi")


def part_2d():
    S = np.array([[1, RHO], [RHO, 1]])
    lam = np.linalg.inv(S)
    mf_var = 1 / lam[0, 0]
    qcov = np.diag([mf_var, mf_var])

    def kl_gauss(S0, S1):           # KL(N(0, S0) || N(0, S1))
        return 0.5 * (np.trace(np.linalg.inv(S1) @ S0) - 2 + math.log(np.linalg.det(S1) / np.linalg.det(S0)))

    NUM["corr"] = dict(rho=RHO, mf_sd=f"{math.sqrt(mf_var):.3f}", marg_sd=1.0, rev_kl=f"{kl_gauss(qcov, S):.3f}",
                       fwd_kl=f"{kl_gauss(S, np.eye(2)):.3f}", mf_var=f"{mf_var:.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw=dict(wspace=0.08))
    for ax, cov, title in ((axs[0], qcov, r"minimise $\mathrm{KL}(q\,\Vert\,p)$: too narrow"),
                           (axs[1], np.eye(2), r"minimise $\mathrm{KL}(p\,\Vert\,q)$: too wide")):
        density_panel(ax, corr_logp)
        ellipses(ax, np.zeros(2), cov, C["enc"])
        ax.set_title(title, fontsize=15, color=C["ink"], loc="left")
    save(fig, "vik-corr")

    # the same target, three choices of q, for the widget's static state and the full-covariance slide
    m_d, c_d, kl_d = fit_2d(lambda z: corr_logp(z), "diag")
    m_f, c_f, kl_f = fit_2d(lambda z: corr_logp(z), "full")
    NUM["corr"].update(fit_diag_sd=f"{math.sqrt(c_d[0, 0]):.3f}", fit_diag_kl=f"{kl_d:.3f}", fit_full_kl=f"{max(kl_f, 0):.3f}",
                       fit_full_rho=f"{c_f[0, 1] / math.sqrt(c_f[0, 0] * c_f[1, 1]):.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw=dict(wspace=0.08))
    for ax, (m, c), title in ((axs[0], (m_d, c_d), "diagonal $q$ (mean-field)"), (axs[1], (m_f, c_f), "full-covariance $q$")):
        density_panel(ax, corr_logp)
        ellipses(ax, m, c, C["enc"])
        ax.set_title(title, fontsize=15, color=C["ink"], loc="left")
    save(fig, "vik-corr-full")

    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw=dict(wspace=0.08))
    for ax, fam, title in ((axs[0], "diag", "diagonal $q$"), (axs[1], "full", "full-covariance $q$")):
        m, c, kl = fit_2d(banana_logp, fam, init_m=(0.0, 0.0))
        out[fam] = r(kl, 3)
        density_panel(ax, banana_logp)
        ellipses(ax, m, c, C["enc"])
        ax.set_title(title, fontsize=15, color=C["ink"], loc="left")
    NUM["banana"] = dict(diag_kl=f"{out['diag']:.3f}", full_kl=f"{out['full']:.3f}")
    save(fig, "vik-banana")

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6), gridspec_kw=dict(wspace=0.08))
    res = []
    for ax, init, title in ((axs[0], (0.3, 0.2), r"$\mathrm{KL}(q\,\Vert\,p)$, start right of centre"),
                            (axs[1], (-0.3, -0.2), r"$\mathrm{KL}(q\,\Vert\,p)$, start left of centre")):
        m, c, kl = fit_2d(bumps_logp, "full", init_m=init)
        res.append((m, kl))
        density_panel(ax, bumps_logp)
        ellipses(ax, m, c, C["enc"])
        ax.set_title(title, fontsize=14, color=C["ink"], loc="left")
    fc = 0.55 ** 2 * np.eye(2) + 1.5 ** 2 * np.ones((2, 2))
    density_panel(axs[2], bumps_logp)
    ellipses(axs[2], np.zeros(2), fc, C["enc"])
    axs[2].set_title(r"$\mathrm{KL}(p\,\Vert\,q)$: cover both", fontsize=14, color=C["ink"], loc="left")
    NUM["bumps2"] = dict(kl_a=r(res[0][1], 3), kl_b=r(res[1][1], 3), m_a=[r(v, 2) for v in res[0][0]], m_b=[r(v, 2) for v in res[1][0]],
                         log2=r(math.log(2), 3))
    save(fig, "vik-bumps2d")


def part_shapes():
    """1-D targets a Gaussian cannot match: a skewed one and a two-bump one, and a flow that can."""
    xs = np.linspace(-4, 6, 4001)
    dx = xs[1] - xs[0]
    erf = np.vectorize(math.erf)
    alpha = 5.0
    skew = 2 * npdf(xs, 0, 1.2) * 0.5 * (1 + erf(alpha * xs / 1.2 / math.sqrt(2)))
    skew /= skew.sum() * dx
    mix = 0.5 * npdf(xs, -2, .5) + 0.5 * npdf(xs, 2, .5)

    def best(p):
        bestv = None
        for m in np.linspace(-3, 3, 181):
            for s in np.linspace(0.2, 2.5, 116):
                q = npdf(xs, m, s)
                v = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
                if bestv is None or v < bestv[0]:
                    bestv = (v, m, s)
        return bestv

    bs, bm = best(skew), best(mix)
    NUM["shapes"] = dict(skew_kl=f"{bs[0]:.2f}", skew_m=f"{bs[1]:.2f}", skew_s=f"{bs[2]:.2f}", mix_kl=f"{bm[0]:.3f}", mix_m=f"{bm[1]:.1f}".replace("-", "−"),
                         mix_s=f"{bm[2]:.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 3.5), gridspec_kw=dict(wspace=0.12))
    for ax, p, b, title in ((axs[0], skew, bs, "a skewed target"), (axs[1], mix, bm, "a two-bump target")):
        ax.fill_between(xs, 0, p, color=C["x_t"], lw=0)
        ax.plot(xs, p, color=C["x"], lw=1.6)
        ax.plot(xs, npdf(xs, b[1], b[2]), color=C["enc"], lw=2.6)
        ax.set_xlim(-4, 4.5)
        ax.set_ylim(0, max(p.max(), npdf(0, 0, b[2])) * 1.15)
        ax.set_xlabel(r"$z$")
        ax.set_title(title + rf":  gap $= {b[0]:.2f}$ nats", loc="left", fontsize=15, color=C["ink"])
        no_y(ax)
    save(fig, "vik-shapes")

    # a flow: push N(0, 1) through a monotone curve to get exactly the skewed target
    cdf = np.cumsum(skew) * dx
    e = np.linspace(-3, 3, 400)
    phi = 0.5 * (1 + erf(e / math.sqrt(2)))
    f = np.interp(phi, cdf, xs)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1, 1]))
    axs[0].fill_between(e, 0, npdf(e, 0, 1), color=C["z_t"], lw=0)
    axs[0].plot(e, npdf(e, 0, 1), color=C["z"])
    axs[0].set_xlabel(r"$\varepsilon$")
    axs[0].set_title(r"$\varepsilon \sim \mathcal{N}(0, 1)$", loc="left", fontsize=15, color=C["z"])
    no_y(axs[0])
    axs[1].plot(e, f, color=C["enc"], lw=2.6)
    axs[1].set_xlabel(r"$\varepsilon$")
    axs[1].set_ylabel(r"$z = f(\varepsilon)$")
    axs[1].set_title(r"a monotone curve $f$", loc="left", fontsize=15, color=C["enc"])
    axs[1].grid(color=C["grid"])
    sel = (xs > -2) & (xs < 4.5)
    axs[2].fill_between(xs[sel], 0, skew[sel], color=C["x_t"], lw=0)
    axs[2].plot(xs[sel], skew[sel], color=C["enc"], lw=2.6)
    axs[2].set_xlabel(r"$z$")
    axs[2].set_title(r"$q(z)$ now matches the target", loc="left", fontsize=15, color=C["enc"])
    no_y(axs[2])
    save(fig, "vik-flow")


# ============================================================================ amortisation
def part_amortise():
    hs = list(range(NT + 1))
    opt = {}
    for h in hs:
        m, s = best_gaussian(lambda z, h=h: tlog_lik(z, h, NT - h))
        ll = h * logsig(ZG) + (NT - h) * logsig(-ZG)
        lpx = lse(log_prior(ZG) + ll) + math.log(DZ)
        e = e_q(lambda u: h * logsig(u) + (NT - h) * logsig(-u), m, s) - kl_prior(m, s)
        opt[h] = (m, s, e, lpx)
    g = torch.Generator().manual_seed(0)
    n_coins = 500
    zc = torch.randn(n_coins, generator=g)
    X = (torch.rand(n_coins, NT, generator=g) < torch.sigmoid(zc)[:, None]).double()
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(NT, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    o = torch.optim.Adam(enc.parameters(), lr=0.01)
    for step in range(3000):
        out = enc(X)
        m, ls = out[:, 0], out[:, 1]
        z = m + ls.exp() * torch.randn(n_coins, generator=g)          # one draw per coin, like a VAE
        h = X.sum(1)
        rec = h * torch.nn.functional.logsigmoid(z) + (NT - h) * torch.nn.functional.logsigmoid(-z)
        kl = 0.5 * (m ** 2 + (2 * ls).exp() - 1 - 2 * ls)
        loss = -(rec - kl).mean()
        o.zero_grad()
        loss.backward()
        o.step()
    with torch.no_grad():
        out = enc(X)
    me, se = out[:, 0].numpy(), out[:, 1].exp().numpy()
    hc = X.sum(1).numpy().astype(int)
    gaps = []
    for i in range(n_coins):
        h = hc[i]
        e_enc = e_q(lambda u: h * logsig(u) + (NT - h) * logsig(-u), me[i], se[i]) - kl_prior(me[i], se[i])
        gaps.append(opt[h][2] - e_enc)
    gaps = np.array(gaps)
    NUM["amort"] = dict(n=n_coins, steps=3000, gap_mean=r(gaps.mean(), 4), gap_max=r(gaps.max(), 3),
                        h8_m=r(opt[8][0], 2), h8_s=r(opt[8][1], 2), h8_enc_m=r(me[hc == 8].mean(), 2), h8_enc_s=r(se[hc == 8].mean(), 2),
                        counts=[int((hc == h).sum()) for h in hs])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, n_coins)
    ax.errorbar([h - 0.25 for h in hs], [opt[h][0] for h in hs], yerr=[opt[h][1] for h in hs], fmt="o", color=C["ink"], ms=7,
                capsize=4, lw=1.6, label="per-coin VI")
    ax.scatter(hc + 0.15 + 0.4 * jit, me, s=10, color=C["enc"], alpha=.5, lw=0, label="encoder, one pass")
    ax.set_xticks(hs)
    ax.set_xlabel(r"heads in 10 tosses")
    ax.set_ylabel(r"centre of $q$:  $m$")
    ax.grid(color=C["grid"])
    ax.legend(loc="upper left", markerscale=2.2)
    save(fig, "vik-amortise")


def main():
    part_bayes()
    m_vi, s_vi = part_hard()
    grid_cost()
    part_mc()
    part_darts()
    part_bits()
    part_kl()
    part_beta_vi()
    part_vi(m_vi, s_vi)
    part_grad()
    part_2d()
    part_shapes()
    part_amortise()
    NUM["data"] = dict(H=H, T=T, n=NT, seq=SEQ)
    (FIG / "numbers-vi.json").write_text(json.dumps(NUM, indent=1))
    print(json.dumps(NUM, indent=1))


if __name__ == "__main__":
    main()
