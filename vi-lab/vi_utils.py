"""Small helpers for the KL and variational inference notebook (vi_lecture.ipynb).

Only torch, numpy and matplotlib. Each helper does one visible thing; the lines that carry the
idea (the KL by sampling, the reparameterised draw, the negative ELBO) are written out in the
notebook itself, not hidden here.
"""
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Circle

dist = torch.distributions

# one colour per role, everywhere (the same as the lecture slides)
COLORS = dict(p="#245EDB", q="#8B2CDE", prior="#AA4E08", lik="#0F766E", post="#245EDB", kl="#BE123C",
              ink="#14171F", grey="#9AA1AD", light="#E4ECFF", q2="#0F766E", q3="#AA4E08")
CYCLE = ["#245EDB", "#8B2CDE", "#0F766E", "#AA4E08", "#BE123C", "#4A5160"]


def style():
    """Clean, readable defaults for every figure in the notebook."""
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.4), "figure.dpi": 110, "savefig.dpi": 150,
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ECEEF2",
        "axes.edgecolor": "#9AA1AD", "mathtext.fontset": "cm", "legend.frameon": False, "lines.linewidth": 2.2,
        "animation.embed_limit": 40,
    })


# ----------------------------------------------------------------------------- distributions
def grid(lo, hi, n=600):
    return torch.linspace(lo, hi, n)


def density(d, x):
    """p(x) for a torch distribution, safe outside the support (returns 0 there)."""
    with torch.no_grad():
        if isinstance(d, dist.Beta):
            x = x.clamp(1e-6, 1 - 1e-6)
        return d.log_prob(x).exp()


def mog(weights, locs, scales):
    """A one-dimensional mixture of Gaussians, as in the PML notebook."""
    return dist.MixtureSameFamily(dist.Categorical(probs=torch.tensor(weights, dtype=torch.float)),
                                  dist.Normal(torch.tensor(locs, dtype=torch.float), torch.tensor(scales, dtype=torch.float)))


def name(d):
    """A short label such as N(1.00, 4.00) or Beta(22.0, 16.0)."""
    if isinstance(d, dist.Normal):
        return rf"$\mathcal{{N}}({d.loc.item():.2f},\ {d.scale.item():.2f})$"
    if isinstance(d, dist.Beta):
        return rf"Beta$({d.concentration1.item():.1f},\ {d.concentration0.item():.1f})$"
    return type(d).__name__


def plot_dists(dists, x, ax=None, fill=(), colors=None, styles=None, title=None, legend=True):
    """Plot several distributions given as {label: distribution}. Labels listed in `fill` are shaded."""
    ax = ax or plt.subplots()[1]
    colors = colors or {}
    styles = styles or {}
    for i, (label, d) in enumerate(dists.items()):
        y = density(d, x)
        c = colors.get(label, CYCLE[i % len(CYCLE)])
        if label in fill:
            ax.fill_between(x, 0, y, color=c, alpha=0.15, lw=0)
        ax.plot(x, y, color=c, ls=styles.get(label, "-"), label=label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="upper right")
    return ax


def bars(probs, labels, names, ax=None, title=None):
    """Side-by-side bars for discrete distributions, e.g. the two years of favourite fruit."""
    ax = ax or plt.subplots()[1]
    k = len(probs)
    w = 0.8 / k
    for i, (p, lab) in enumerate(zip(probs, labels)):
        xs = np.arange(len(names)) + (i - (k - 1) / 2) * w
        ax.bar(xs, p, width=w, color=CYCLE[i], label=lab)
        for x_, v in zip(xs, p):
            ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("probability")
    ax.grid(axis="x")
    ax.legend()
    if title:
        ax.set_title(title, loc="left")
    return ax


# ----------------------------------------------------------------------------- KL
def kl_via_sampling(q, p, n_samples=100_000):
    """KL(q || p) by Monte Carlo: draw from q (the first argument), average log q - log p."""
    with torch.no_grad():
        z = q.sample([n_samples])
        return (q.log_prob(z) - p.log_prob(z)).mean()


def kl_grid(q, p, x):
    """KL(q || p) by numerical integration on a fine grid x (1-D only; for checking)."""
    lq, lp = q.log_prob(x), p.log_prob(x)
    return torch.trapezoid(lq.exp() * (lq - lp), x)


# ----------------------------------------------------------------------------- trainable q's
class TrainableNormal(nn.Module):
    """q = N(loc, scale) with scale = softplus(raw_scale) > 0, as in the PML notebook."""

    def __init__(self, loc=0.0, scale=1.0):
        super().__init__()
        self.loc = nn.Parameter(torch.tensor(float(loc)))
        self.raw_scale = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(scale)))))

    def forward(self):
        return dist.Normal(self.loc, F.softplus(self.raw_scale))

    def __repr__(self):
        return f"TrainableNormal(loc={self.loc.item():.2f}, scale={F.softplus(self.raw_scale).item():.2f})"


class TrainableBeta(nn.Module):
    """q = Beta(alpha, beta), both kept positive with softplus."""

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        inv = lambda v: torch.log(torch.expm1(torch.tensor(float(v))))
        self.raw_alpha, self.raw_beta = nn.Parameter(inv(alpha)), nn.Parameter(inv(beta))

    def forward(self):
        return dist.Beta(F.softplus(self.raw_alpha), F.softplus(self.raw_beta))

    def __repr__(self):
        return f"TrainableBeta(alpha={F.softplus(self.raw_alpha).item():.2f}, beta={F.softplus(self.raw_beta).item():.2f})"


class TrainableGaussian2D(nn.Module):
    """q = N(loc, L L^T) in two dimensions; diagonal (mean-field) or full covariance."""

    def __init__(self, loc=(0.0, 0.0), scale=1.0, full=False):
        super().__init__()
        self.full = full
        self.loc = nn.Parameter(torch.tensor(loc, dtype=torch.float))
        self.raw_diag = nn.Parameter(torch.full((2,), math.log(math.expm1(scale))))
        self.off = nn.Parameter(torch.zeros(1), requires_grad=full)

    def scale_tril(self):
        d = F.softplus(self.raw_diag)
        L = torch.diag(d)
        if self.full:
            L = L + torch.tensor([[0.0, 0.0], [1.0, 0.0]]) * self.off
        return L

    def forward(self):
        return dist.MultivariateNormal(self.loc, scale_tril=self.scale_tril())


# ----------------------------------------------------------------------------- training (in place of ASTRA's train_fn)
def fit(q, loss_fn, steps=400, lr=0.05, seed=0):
    """Adam on the parameters of q; loss_fn takes the current distribution q() and returns a scalar.
    Returns the history: one entry per step with the loss and a detached copy of q()."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in q.parameters() if p.requires_grad], lr=lr)
    history = []
    for step in range(steps + 1):
        qd = q()
        loss = loss_fn(qd)
        history.append(dict(step=step, loss=loss.item(), q=_detach(qd)))
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return history


def _detach(d):
    if isinstance(d, dist.Normal):
        return dist.Normal(d.loc.detach().clone(), d.scale.detach().clone())
    if isinstance(d, dist.Beta):
        return dist.Beta(d.concentration1.detach().clone(), d.concentration0.detach().clone())
    if isinstance(d, dist.MultivariateNormal):
        return dist.MultivariateNormal(d.loc.detach().clone(), scale_tril=d.scale_tril.detach().clone())
    return copy.deepcopy(d)


def _frames(n_steps, n_frames=45):
    """Frame indices dense at the start (where q moves fastest), sparse later."""
    idx = np.unique(np.round(np.geomspace(1, n_steps + 1, n_frames)).astype(int) - 1)
    return [0] + [i for i in idx if i > 0]


def animate_fit(history, target, x, loss_name="KL", n_frames=32, target_label="target $p$", fill_target=True):
    """A play / step / slider animation of q moving towards the target, with the loss curve beside it."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=90, gridspec_kw=dict(width_ratios=[1.5, 1]))
    yt = density(target, x)
    qs = [density(h["q"], x) for h in history]
    ymax = max(yt.max().item(), max(v.max().item() for v in qs[len(qs) // 3:])) * 1.15
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    def draw(i):
        a1.clear(), a2.clear()
        if fill_target:
            a1.fill_between(x, 0, yt, color=COLORS["p"], alpha=0.15, lw=0)
        a1.plot(x, yt, color=COLORS["p"], label=target_label)
        a1.plot(x, qs[i], color=COLORS["q"], label=r"$q$ = " + name(history[i]["q"]))
        a1.set_ylim(0, ymax)
        a1.set_xlabel(r"$\theta$")
        a1.legend(loc="upper right")
        a1.set_title(f"step {steps[i]}", loc="left")
        a2.plot(steps, losses, color=COLORS["grey"], lw=1.5)
        a2.plot([steps[i]], [losses[i]], "o", color=COLORS["kl"])
        a2.set_title(f"{loss_name} = {losses[i]:.3f}", loc="left")
        a2.set_xlabel("step")
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=_frames(len(history) - 1, n_frames), interval=120)
    with plt.rc_context({"savefig.dpi": 90}):               # frames at screen resolution, not print
        html = anim.to_jshtml()
    plt.close(fig)
    from IPython.display import HTML
    return HTML(html)


def snapshots(history, target, x, at=(0, 10, 50, -1), ax=None, target_label="target $p$"):
    """A static picture of q at a few steps: the print version of animate_fit."""
    ax = ax or plt.subplots()[1]
    ax.fill_between(x, 0, density(target, x), color=COLORS["p"], alpha=0.15, lw=0)
    ax.plot(x, density(target, x), color=COLORS["p"], label=target_label)
    for k, i in enumerate(at):
        h = history[i]
        ax.plot(x, density(h["q"], x), color=COLORS["q"], alpha=0.3 + 0.7 * (k + 1) / len(at),
                label=f"step {h['step']}: " + name(h["q"]))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel(r"$\theta$")
    return ax


# ----------------------------------------------------------------------------- 2-D pictures
def ellipse(ax, mean, cov, color, label=None, k=(1, 2), lw=2.2):
    """1- and 2-standard-deviation ellipses of a 2-D Gaussian."""
    mean, cov = np.asarray(mean, float), np.asarray(cov, float)
    vals, vecs = np.linalg.eigh(cov)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    for j, s in enumerate(k):
        ax.add_patch(Ellipse(mean, 2 * s * math.sqrt(vals[1]), 2 * s * math.sqrt(vals[0]), angle=ang, fill=False,
                             ec=color, lw=lw, label=label if j == 0 else None))
    ax.plot(*mean, "o", color=color, ms=5)


def contour2d(ax, d, lim=3.0, n=200):
    """Filled contours of a 2-D torch distribution."""
    g = torch.linspace(-lim, lim, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    with torch.no_grad():
        Z = d.log_prob(torch.stack([X, Y], -1)).exp()
    ax.contourf(X, Y, Z / Z.max(), levels=np.linspace(0.05, 1, 8), cmap="Blues", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")


def animate_fit_2d(history, target, lim=3.0, n_frames=32, title="q"):
    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=90)

    def draw(i):
        ax.clear()
        contour2d(ax, target, lim)
        h = history[i]
        ellipse(ax, h["q"].loc, h["q"].covariance_matrix, COLORS["q"])
        ax.set_title(f"{title} · step {h['step']} · KL {h['loss']:.3f}", loc="left", fontsize=11)

    anim = animation.FuncAnimation(fig, draw, frames=_frames(len(history) - 1, n_frames), interval=120)
    with plt.rc_context({"savefig.dpi": 90}):               # frames at screen resolution, not print
        html = anim.to_jshtml()
    plt.close(fig)
    from IPython.display import HTML
    return HTML(html)


# ----------------------------------------------------------------------------- diagrams
def graph(kind="sample", ax=None):
    """The computational graph of the Monte Carlo KL loss (as in the PML slides):
    blue squares are deterministic nodes, red circles are stochastic ones.
    kind='sample': loc, scale -> q -> samples -> loss (no gradient path).
    kind='reparam': eps -> z = loc + scale * eps -> loss (gradients reach loc and scale)."""
    ax = ax or plt.subplots(figsize=(5.2, 3.6))[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    blue, red = "#4C9BE8", "#E5483B"

    def box(x, y, t, w=1.6, h=0.8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=blue, ec="none"))
        ax.text(x, y, t, ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    def circ(x, y, t, r=0.75, size=10):
        ax.add_patch(Circle((x, y), r, fc=red, ec="none"))
        ax.text(x, y, t, ha="center", va="center", color="white", fontsize=size, fontweight="bold")

    def arrow(a, b, color="#6B7280"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))

    if kind == "sample":
        box(3, 1, "loc")
        box(6, 1, "scale")
        box(4.5, 2.8, r"$q=\mathcal{N}$(loc, scale)", w=4.2)
        circ(4.5, 4.6, "sample", r=0.9)
        box(4.5, 6.3, "loss")
        arrow((3, 1.4), (4.1, 2.4))
        arrow((6, 1.4), (4.9, 2.4))
        arrow((4.5, 3.2), (4.5, 3.8))
        arrow((4.5, 5.4), (4.5, 5.9))
        ax.text(5.6, 3.5, "no gradient\nthrough a draw", color=red, fontsize=10, va="center")
    else:
        box(2.5, 1, "loc")
        box(5.0, 1, "scale")
        circ(8.3, 1.2, r"$\varepsilon\sim\mathcal{N}(0,1)$", r=1.15, size=11)
        ax.text(6.6, 2.9, r"$\times$", fontsize=26, ha="center", va="center")
        ax.text(3.8, 4.3, "+", fontsize=26, ha="center", va="center")
        box(3.8, 5.4, r"$z$", w=1.1)
        box(3.8, 6.5, "loss")
        arrow((5.0, 1.4), (6.4, 2.6))
        arrow((7.6, 1.9), (6.9, 2.6))
        arrow((6.3, 3.1), (4.2, 4.1))
        arrow((2.5, 1.4), (3.6, 4.0))
        arrow((3.8, 4.7), (3.8, 5.0))
        arrow((3.8, 5.8), (3.8, 6.1))
        ax.text(6.1, 5.4, r"$z = \mathrm{loc} + \mathrm{scale}\cdot\varepsilon$", fontsize=12, color="#14171F")
    return ax


def elbo_gaps(log_evidence, elbos, ax=None):
    """The evidence / ELBO / KL picture for several q's at once: log p(D) is the same dashed line for all of them;
    each q's ELBO sits below it, and the gap is that q's KL(q || p(theta | D)).  elbos: {label: elbo}."""
    ax = ax or plt.subplots(figsize=(7, 3.8))[1]
    labels, vals = list(elbos), list(elbos.values())
    xs = np.arange(len(vals))
    span = log_evidence - min(vals)
    ax.axhline(log_evidence, color=COLORS["p"], ls="--", lw=1.6)
    ax.text(-0.35, log_evidence + 0.04 * span, rf"$\log p(D) = {log_evidence:.2f}$ (fixed)", color=COLORS["p"], fontsize=12)
    for x_, v in zip(xs, vals):
        gap = log_evidence - v
        ax.plot([x_, x_], [v, log_evidence], color=COLORS["kl"], lw=3, solid_capstyle="butt")
        ax.plot([x_], [v], "o", color=COLORS["lik"], ms=9, zorder=3)
        if gap > 0.08 * span:
            ax.text(x_ + 0.08, (v + log_evidence) / 2, f"KL = {gap:.3f}", color=COLORS["kl"], va="center", fontsize=11)
            ax.text(x_ + 0.08, v, f"ELBO = {v:.2f}", color=COLORS["lik"], va="top", fontsize=10)
        else:
            ax.text(x_ + 0.08, v - 0.06 * span, f"ELBO = {v:.2f}\nKL = {gap:.3f}", color=COLORS["lik"], va="top", fontsize=10)
    ax.set_ylim(min(vals) - 0.25 * span, log_evidence + 0.18 * span)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.4, len(vals) + 0.4)
    ax.set_ylabel("nats")
    ax.grid(axis="x")
    return ax


def elbo_bar(log_evidence, elbo, ax=None, title=None):
    """The evidence / ELBO / KL picture from the PML slides, with numbers:
    log p(D) is fixed; the ELBO sits below it; the gap is KL(q || p(theta | D))."""
    ax = ax or plt.subplots(figsize=(3.4, 4.2))[1]
    kl = log_evidence - elbo
    lo = min(elbo, log_evidence) - 0.25 * max(kl, 1)
    ax.plot([0, 0], [lo, log_evidence + 0.3], color="#14171F", lw=1.5)
    for y, lab, c in ((log_evidence, rf"evidence $\log p(D) = {log_evidence:.2f}$", COLORS["p"]),
                      (elbo, rf"ELBO$(q) = {elbo:.2f}$", COLORS["lik"])):
        ax.plot([-0.08, 0.08], [y, y], color=c, lw=3)
        ax.text(0.15, y, lab, va="center", fontsize=11, color=c)
    ax.annotate("", xy=(-0.25, log_evidence), xytext=(-0.25, elbo),
                arrowprops=dict(arrowstyle="<->", color=COLORS["kl"], lw=1.8))
    ax.text(-0.32, (log_evidence + elbo) / 2, rf"KL$(q\,\|\,p(\theta\mid D)) = {kl:.2f}$", rotation=90, ha="right",
            va="center", color=COLORS["kl"], fontsize=11)
    ax.set_xlim(-0.9, 2.2)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax


# ----------------------------------------------------------------------------- Bayesian models used in the lecture
class Coin:
    """Coin toss with a Beta(a, b) prior on theta = P(heads), as in the PML notebook."""

    def __init__(self, heads=17, tails=13, a=5.0, b=3.0):
        self.heads, self.tails, self.a, self.b = heads, tails, a, b
        self.data = torch.cat([torch.ones(heads), torch.zeros(tails)])
        self.prior = dist.Beta(a, b)
        self.posterior = dist.Beta(a + heads, b + tails)          # conjugate: known exactly

    def log_prior(self, theta):
        return self.prior.log_prob(theta)

    def log_likelihood(self, theta):
        """sum over tosses of log Bernoulli(toss | theta), for one theta or a batch of thetas."""
        theta = theta.clamp(1e-30, 1 - 1e-7)                   # draws can underflow to exactly 0 or 1
        return self.heads * torch.log(theta) + self.tails * torch.log1p(-theta)

    def log_joint(self, theta):
        return self.log_likelihood(theta) + self.log_prior(theta)

    @property
    def log_evidence(self):
        lb = lambda a, b: math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return lb(self.a + self.heads, self.b + self.tails) - lb(self.a, self.b)


class LogitCoin:
    """The same coin with a Normal prior on the logit: theta = sigmoid(w), w ~ N(mu, sigma).
    No formula for the evidence (PML Bayes-for-ML lecture, p. 77; Laplace notebook)."""

    def __init__(self, heads=17, tails=13, mu=0.0, sigma=1.0):
        self.heads, self.tails = heads, tails
        self.prior = dist.Normal(mu, sigma)

    def log_likelihood(self, w):
        return self.heads * F.logsigmoid(w) + self.tails * F.logsigmoid(-w)

    def log_joint(self, w):
        return self.log_likelihood(w) + self.prior.log_prob(w)


def evidence_grid(log_joint, lo, hi, n=20001):
    """log p(D) = log of the integral of p(D, theta) over a fine 1-D grid (log-sum-exp)."""
    x = torch.linspace(lo, hi, n)
    with torch.no_grad():
        return (torch.logsumexp(log_joint(x), 0) + math.log((hi - lo) / (n - 1))).item()


def evidence_mc(prior, log_likelihood, n=15_000, seed=0):
    """log p(D) estimated from prior draws: log (1/N) sum_i p(D | theta_i), with log-sum-exp.
    Also returns the log-likelihoods of the draws, to see how few of them matter."""
    torch.manual_seed(seed)
    with torch.no_grad():
        ll = log_likelihood(prior.sample((n,)))
        return (torch.logsumexp(ll, 0) - math.log(n)).item(), ll


class LinReg:
    """Bayesian linear regression y = theta0 + theta1 x + noise, theta ~ N(0, prior_sd^2 I): the exact
    posterior and evidence exist (PML lectures 7 and Sampling-1), so every approximation can be checked."""

    def __init__(self, x, y, noise_sd=1.0, prior_sd=1.0):
        self.x, self.y, self.noise_sd, self.prior_sd = x, y, noise_sd, prior_sd
        self.X = torch.stack([torch.ones_like(x), x], 1)
        self.prior = dist.MultivariateNormal(torch.zeros(2), prior_sd ** 2 * torch.eye(2))

    def log_likelihood(self, theta):
        """theta: [..., 2] -> log p(y | x, theta) summed over the data."""
        mean = theta[..., :1] + theta[..., 1:] * self.x
        return dist.Normal(mean, self.noise_sd).log_prob(self.y).sum(-1)

    def log_joint(self, theta):
        return self.log_likelihood(theta) + self.prior.log_prob(theta)

    def posterior(self):
        S0inv = torch.eye(2) / self.prior_sd ** 2
        SN = torch.linalg.inv(S0inv + self.X.T @ self.X / self.noise_sd ** 2)
        mN = SN @ self.X.T @ self.y / self.noise_sd ** 2
        return dist.MultivariateNormal(mN, SN)

    def log_evidence(self):
        cov = self.prior_sd ** 2 * self.X @ self.X.T + self.noise_sd ** 2 * torch.eye(len(self.x))
        return dist.MultivariateNormal(torch.zeros(len(self.x)), cov).log_prob(self.y).item()


# ----------------------------------------------------------------------------- Laplace and MCMC, for the one comparison slide
def laplace(log_target, init=0.0, steps=500, lr=0.05):
    """Gaussian at the mode, with variance = -1 / (second derivative of log p) there."""
    theta = torch.tensor(float(init), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (-log_target(theta)).backward()
        opt.step()
    h = torch.autograd.functional.hessian(log_target, theta.detach())
    return dist.Normal(theta.detach(), (-1 / h).sqrt())


def metropolis(log_target, init=0.0, steps=6000, step=1.0, burn=1000, seed=0):
    """Random-walk Metropolis: propose theta + step * noise, accept with prob min(1, p(new) / p(old))."""
    g = torch.Generator().manual_seed(seed)
    cur = torch.tensor(float(init))
    cur_lp = log_target(cur)
    out, acc = [], 0
    for t in range(steps):
        prop = cur + step * torch.randn((), generator=g)
        lp = log_target(prop)
        if torch.log(torch.rand((), generator=g)) < lp - cur_lp:
            cur, cur_lp, acc = prop, lp, acc + 1
        out.append(cur.item())
    return torch.tensor(out[burn:]), acc / steps
