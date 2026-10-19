"""Every data figure and every quoted number in the lecture comes from this script.

    python scripts/make_figures.py

Reads the checkpoints written by train_models.py and writes
figures/data/*.svg|png and figures/data/numbers.json.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_models import AutoEncoder, ConvG, ConvD, CondVAE, PixelAR, load_mnist, toy_data  # noqa: E402

MODELS = ROOT / "assets" / "models"
FIG = ROOT / "figures" / "data"
FIG.mkdir(parents=True, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NUM = {}  # numbers quoted on slides

C = dict(x="#245EDB", x_t="#E4ECFF", enc="#8B2CDE", enc_t="#F1E5FC", dec="#0F766E", dec_t="#D9F2EF",
         z="#AA4E08", z_t="#FCEFD9", loss="#BE123C", loss_t="#FDE2E7", disc="#334155", disc_t="#E2E8F0",
         ink="#14171F", ink2="#4A5160", ink3="#6B7280", line="#D9DDE5", grid="#ECEEF2")
DIGIT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#7c5a3a", "#5f6b7a"]

plt.rcParams.update({
    "font.family": "Avenir", "font.size": 15, "svg.fonttype": "path", "mathtext.fontset": "cm", "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA1AD",
    "axes.labelcolor": C["ink2"], "axes.labelsize": 15, "xtick.color": C["ink3"], "ytick.color": C["ink3"],
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 1.0,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "legend.frameon": False, "legend.fontsize": 14, "lines.linewidth": 2.4,
})


def save(fig, name):
    path = FIG / f"{name}.svg"
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    svg = path.read_text()
    svg = svg.replace("'Avenir'", "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    path.write_text(svg)


def r(v, k=2):
    return float(round(float(v), k))


# ----------------------------------------------------------------------------- image helpers
INK = np.array([0x14, 0x17, 0x1F]) / 255.0
PAPER = np.array([1.0, 1.0, 1.0])


def to_rgb(img):
    img = np.clip(np.asarray(img, dtype=float), 0, 1)[..., None]
    return PAPER * (1 - img) + INK * img


def grid_png(imgs, rows, cols, name, scale=4, pad=2, pad_rgb=(0.92, 0.93, 0.95)):
    imgs = np.asarray(imgs).reshape(-1, 28, 28)
    H = rows * 28 + (rows + 1) * pad
    W = cols * 28 + (cols + 1) * pad
    out = np.ones((H, W, 3)) * np.array(pad_rgb)
    for i in range(min(len(imgs), rows * cols)):
        rr, cc = divmod(i, cols)
        y0, x0 = pad + rr * (28 + pad), pad + cc * (28 + pad)
        out[y0:y0 + 28, x0:x0 + 28] = to_rgb(imgs[i])
    out = np.repeat(np.repeat(out, scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", out)


def tile_png(img, name, scale=8):
    out = np.repeat(np.repeat(to_rgb(np.asarray(img).reshape(28, 28)), scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", out)


# ----------------------------------------------------------------------------- models
def load_ae(name):
    meta = json.loads((MODELS / "training.json").read_text())[name]
    m = AutoEncoder(meta["latent"], meta["variational"]).to(DEV)
    m.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location=DEV))
    m.eval()
    return m


@torch.no_grad()
def encode(m, x, bs=2000):
    mus, lvs = [], []
    for i in range(0, len(x), bs):
        mu, lv = m.encoder(x[i:i + bs].to(DEV))
        mus.append(mu.cpu())
        lvs.append(lv.cpu() if lv is not None else torch.zeros_like(mu.cpu()))
    return torch.cat(mus), torch.cat(lvs)


@torch.no_grad()
def decode(m, z, bs=20000):
    out = []
    for i in range(0, len(z), bs):
        out.append(torch.sigmoid(m.decoder(z[i:i + bs].to(DEV))).cpu())
    return torch.cat(out)


@torch.no_grad()
def log_px_given_z(m, x, z, bs=40000):
    """log p(x|z) for one binary image x [784] and many z [N, d]."""
    xf = x.reshape(1, 784).to(DEV)
    out = []
    for i in range(0, len(z), bs):
        logits = m.decoder(z[i:i + bs].to(DEV))
        out.append(-F.binary_cross_entropy_with_logits(logits, xf.expand_as(logits), reduction="none").sum(1).cpu())
    return torch.cat(out).double()


# ----------------------------------------------------------------------------- main
def main():
    data = load_mnist()
    xte, yte = data["test"]
    xtr, ytr = data["train"]
    torch.manual_seed(0)
    ae2, ae16, vae2, vae2b8, vae16 = (load_ae(n) for n in ("ae2", "ae16", "vae2", "vae2b8", "vae16"))
    meta = json.loads((MODELS / "training.json").read_text())

    # ---------------- running digit: a typical '3' and a typical '7'
    def typical(label):
        idx = torch.nonzero(yte == label)[:, 0]
        mu, _ = encode(vae2, xte[idx])
        with torch.no_grad():
            logits = vae2.decoder(mu.to(DEV)).cpu()
            nll = F.binary_cross_entropy_with_logits(logits, xte[idx].flatten(1), reduction="none").sum(1)
        centre = mu.median(0).values
        good = nll <= nll.median()
        d = (mu - centre).norm(dim=1) + 1e9 * (~good)
        return int(idx[d.argmin()])

    i3, i7 = typical(3), typical(7)
    x3, x7 = xte[i3], xte[i7]
    NUM["digit3_index"], NUM["digit7_index"] = i3, i7
    tile_png(x3, "digit3")
    tile_png(x7, "digit7")

    # ---------------- encoder outputs for our 3, one fixed epsilon
    with torch.no_grad():
        mu3, lv3 = vae2.encoder(x3.unsqueeze(0).to(DEV))
    mu3, lv3 = mu3.cpu()[0].double(), lv3.cpu()[0].double()
    sd3 = torch.exp(0.5 * lv3)
    eps3 = torch.tensor([0.62, -1.15], dtype=torch.float64)  # one draw, written down so the slide is reproducible
    z3 = mu3 + sd3 * eps3
    with torch.no_grad():
        logits3 = vae2.decoder(z3.float().unsqueeze(0).to(DEV)).cpu()[0].double()
    pi3 = torch.sigmoid(logits3)
    recon3 = F.binary_cross_entropy_with_logits(logits3, x3.flatten().double(), reduction="sum").item()
    kl3 = 0.5 * (mu3 ** 2 + sd3 ** 2 - 1 - torch.log(sd3 ** 2))
    tile_png(pi3.numpy(), "digit3-pi")
    with torch.no_grad():
        pimu = torch.sigmoid(vae2.decoder(mu3.float().unsqueeze(0).to(DEV))).cpu()[0]
    tile_png(pimu.numpy(), "digit3-recon")
    NUM["our3"] = dict(mu=[r(v, 2) for v in mu3], logvar=[r(v, 2) for v in lv3], sigma=[r(v, 3) for v in sd3],
                       eps=[r(v, 2) for v in eps3], z=[r(v, 2) for v in z3], recon=r(recon3, 1),
                       kl=[r(v, 2) for v in kl3], kl_total=r(kl3.sum(), 2), loss=r(r(recon3, 1) + r(kl3.sum(), 2), 1),  # the sum of the numbers shown
                       ink_pixels=int(x3.sum().item()))

    # Bernoulli pixel patch: a 5x5 window on the stroke edge
    img = x3.numpy()
    best = None
    for rr in range(4, 20):
        for cc in range(4, 20):
            p = img[rr:rr + 5, cc:cc + 5]
            score = abs(p.mean() - 0.5)
            if best is None or score < best[0]:
                best = (score, rr, cc)
    _, pr, pc = best
    pim = pi3.reshape(28, 28).numpy()
    patch_x = img[pr:pr + 5, pc:pc + 5].astype(int).tolist()
    patch_pi = np.round(pim[pr:pr + 5, pc:pc + 5], 2).tolist()
    patch_loss = [[r(-math.log(max(pv, 1e-6)) if xv else -math.log(max(1 - pv, 1e-6)), 2)
                   for xv, pv in zip(xr, prow)] for xr, prow in zip(patch_x, pim[pr:pr + 5, pc:pc + 5])]
    NUM["patch"] = dict(row=pr, col=pc, x=patch_x, pi=patch_pi, loss=patch_loss,
                        total=r(sum(map(sum, patch_loss)), 2))
    # highlight box image: digit with the patch outlined
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.imshow(to_rgb(img), interpolation="nearest")
    ax.add_patch(plt.Rectangle((pc - .5, pr - .5), 5, 5, fill=False, ec=C["loss"], lw=2.5))
    ax.axis("off")
    save(fig, "digit3-patch")

    # ---------------- 1-D toy data (the lecture's running 1-D example)
    g = torch.Generator().manual_seed(0)
    toy = toy_data(200, g).numpy()
    mu_hat, sd_hat = toy.mean(), toy.std()
    NUM["toy"] = dict(n=200, mean=r(mu_hat, 2), std=r(sd_hat, 2), var=r(sd_hat ** 2, 2), first=[r(v, 2) for v in toy[:6]])
    xs = np.linspace(-5, 5, 600)

    def npdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    mix = 0.5 * npdf(xs, -2, .5) + 0.5 * npdf(xs, 2, .5)
    rng = np.random.default_rng(4)
    gauss_samples = rng.normal(mu_hat, sd_hat, 30)
    side = rng.random(30) < .5
    mix_samples = np.where(side, -2, 2) + 0.5 * rng.normal(size=30)
    NUM["toy"]["gauss_samples_in_gap"] = int(np.sum(np.abs(gauss_samples) < 1))

    def toy_axes(ax, ymax=0.62):
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.1, ymax)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(range(-4, 5, 2))
        ax.spines["bottom"].set_position(("data", 0))

    def toy_base(ax, rug=True):
        ax.hist(toy, bins=np.linspace(-4.5, 4.5, 37), density=True, color=C["x_t"], ec="white", lw=1)
        if rug:
            ax.plot(toy, np.full_like(toy, -0.035), "|", color=C["x"], ms=9, alpha=.6, mew=1.2)

    for stage in ("data", "mean", "gauss", "gauss-samples", "mixture", "mixture-samples"):
        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        toy_base(ax)
        toy_axes(ax)
        if stage == "data":
            ax.text(-2, 0.5, "bump A", ha="center", color=C["x"], fontsize=17)
            ax.text(2, 0.5, "bump B", ha="center", color=C["x"], fontsize=17)
        if stage == "mean":
            ax.axvline(mu_hat, color=C["ink"], lw=2, ls=(0, (4, 3)), ymax=.85)
            ax.text(mu_hat, 0.56, f"mean $= {mu_hat:.2f}$", ha="center", fontsize=17, color=C["ink"])
            ax.annotate("almost no data here", xy=(0, 0.04), xytext=(0.6, 0.33), fontsize=15, color=C["ink2"],
                        arrowprops=dict(arrowstyle="->", color=C["ink3"]))
        if stage.startswith("gauss"):
            ax.plot(xs, npdf(xs, mu_hat, sd_hat), color=C["dec"])
            ax.text(3.3, 0.24, "best single Gaussian\n" + rf"$\hat\mu = {mu_hat:.2f},\ \hat\sigma = {sd_hat:.2f}$", color=C["dec"], fontsize=15)
        if stage == "gauss-samples":
            ax.plot(gauss_samples, np.full(30, 0.55), "o", color=C["z"], ms=8, mec="white", mew=1)
            ax.text(-5.15, 0.55, "samples", color=C["z"], fontsize=15, va="center", ha="right")
        if stage.startswith("mixture"):
            ax.plot(xs, mix, color=C["dec"])
            ax.plot(xs, 0.5 * npdf(xs, -2, .5), color=C["dec"], lw=1.4, ls=(0, (3, 3)))
            ax.plot(xs, 0.5 * npdf(xs, 2, .5), color=C["dec"], lw=1.4, ls=(0, (3, 3)))
            ax.text(3.0, 0.44, r"$\frac{1}{2}\,\mathcal{N}(2,\,0.5^2)$", color=C["dec"], fontsize=16)
            ax.text(-4.8, 0.44, r"$\frac{1}{2}\,\mathcal{N}(-2,\,0.5^2)$", color=C["dec"], fontsize=16)
        if stage == "mixture-samples":
            ax.plot(mix_samples, np.full(30, 0.55), "o", color=C["z"], ms=8, mec="white", mew=1)
            ax.text(-5.15, 0.55, "samples", color=C["z"], fontsize=15, va="center", ha="right")
        save(fig, f"toy-{stage}")

    # ---------------- density worked example: N(1.5; 2, 0.5^2)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xb = np.linspace(-0.2, 4.2, 400)
    ax.fill_between(xb, 0, npdf(xb, 2, .5), color=C["dec_t"])
    ax.plot(xb, npdf(xb, 2, .5), color=C["dec"])
    yv = npdf(1.5, 2, .5)
    ax.plot([1.5, 1.5], [0, yv], color=C["z"], lw=2, ls=(0, (4, 3)))
    ax.plot([1.5], [yv], "o", color=C["z"], ms=9)
    ax.annotate(rf"height $= {yv:.3f}$", xy=(1.5, yv), xytext=(0.0, 0.62), fontsize=16, color=C["z"],
                arrowprops=dict(arrowstyle="-", color=C["z"]))
    ax.plot([2, 2.5], [0.06, 0.06], color=C["ink2"], lw=1.6)
    ax.text(2.25, 0.09, r"$\sigma = 0.5$", ha="center", fontsize=15, color=C["ink2"])
    ax.text(2.08, -0.13, r"$\mu = 2$", ha="left", fontsize=15, color=C["ink2"])
    ax.text(1.42, -0.13, r"$x = 1.5$", ha="right", fontsize=15, color=C["z"])
    ax.plot([2, 2], [0, 0.03], color=C["ink2"], lw=1.6)
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.18, 0.9)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    save(fig, "gauss-density")
    NUM["gauss_15"] = dict(zscore=-1.0, expo=r(math.exp(-0.5), 3), norm=r(1 / (0.5 * math.sqrt(2 * math.pi)), 3), value=r(yv, 3))

    # Bayes at x = 0.1, the slow way
    la, lb = npdf(0.1, -2, .5), npdf(0.1, 2, .5)
    def sci(v):
        e_ = int(math.floor(math.log10(abs(v))))
        return f"{v / 10 ** e_:.2f}\\times 10^{{{e_}}}"

    NUM["bayes01"] = dict(lik_a=sci(la), lik_b=sci(lb), joint_a=sci(0.5 * la), joint_b=sci(0.5 * lb),
                          evidence=sci(0.5 * (la + lb)), post_b=r(lb / (la + lb), 3), post_a=r(la / (la + lb), 3))

    # ---------------- per-sample reparameterisation gradients for f = z^2, mu = 1, sigma = 0.5
    e = np.random.default_rng(9).normal(size=10000)
    zz_ = 1.0 + 0.5 * e
    g_mu, g_sd = 2 * zz_, 2 * zz_ * e
    NUM["rep_grad"] = dict(mean_mu=r(g_mu.mean(), 2), mean_sd=r(g_sd.mean(), 2), n=10000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.0))
    for ax_, g, ours, exact, lab in ((a1, g_mu, 1.6, 2.0, r"per-sample $\partial f/\partial\mu = 2z$"),
                                     (a2, g_sd, -0.64, 1.0, r"per-sample $\partial f/\partial\sigma = 2z\varepsilon$")):
        g = g[(g > -3) & (g < 7)] if ours < 0 else g
        ax_.hist(g, bins=60, color=C["enc_t"], ec="white", lw=0.5)
        ax_.axvline(exact_mean := (g_mu.mean() if ours > 0 else g_sd.mean()), color=C["enc"], lw=2.5)
        ax_.axvline(ours, color=C["loss"], lw=2, ls=(0, (4, 3)))
        ymax = ax_.get_ylim()[1]
        ax_.text(exact_mean + 0.15, ymax * 1.02, rf"average of 10,000 $= {exact_mean:.2f}$", color=C["enc"], fontsize=15, ha="left")
        ax_.text(ours, ymax * 0.82, rf"our $\varepsilon$: ${ours}$", color=C["loss"], fontsize=15, ha="right")
        ax_.set_title(lab, loc="left", fontsize=15, color=C["ink2"], pad=22)
        ax_.set_yticks([])
        ax_.spines["left"].set_visible(False)
    save(fig, "reparam-grads")

    # ---------------- learning = pick the parameters that make the data most probable
    def loglik(m, sd):
        return float(np.sum(-0.5 * ((toy - m) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi))))

    cands = [("a", 2.0, 0.5), ("b", 0.0, 0.5), ("c", float(mu_hat), float(sd_hat))]
    NUM["cand"] = {k: dict(mu=r(m, 2), sd=r(sd, 2), ll=r(loglik(m, sd), 0)) for k, m, sd in cands}
    for k, m, sd in cands:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(toy, bins=np.linspace(-4.5, 4.5, 37), density=True, color=C["x_t"], ec="white", lw=1)
        ax.plot(toy, np.full_like(toy, -0.03), "|", color=C["x"], ms=8, alpha=.6, mew=1.1)
        ax.plot(xs, npdf(xs, m, sd), color=C["dec"], lw=2.6)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.08, 0.85)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.text(-4.8, 0.74, rf"$\mu = {m:.2f},\ \sigma = {sd:.2f}$", fontsize=16, color=C["dec"])
        save(fig, f"cand-{k}")

    # each data point coloured by p(B | x): the backwards step, applied to all 200 points
    post = 1 / (1 + np.exp(-16 * toy))
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ab", [C["x"], "#C9CED6", C["z"]])
    ax.scatter(toy, np.random.default_rng(1).uniform(-0.3, 0.3, len(toy)), c=post, cmap=cmap, s=34, lw=0.6, ec="white")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.text(-4.8, 0.42, r"blue: surely bump A", color=C["x"], fontsize=15)
    ax.text(2.4, 0.42, r"orange: surely bump B", color=C["z"], fontsize=15)
    save(fig, "toy-colored")

    # posterior over the hidden choice: p(B | x) = sigmoid(16 x)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(xs, 1 / (1 + np.exp(-16 * xs)), color=C["enc"])
    for xv in (0.1, 1.5):
        pv = 1 / (1 + math.exp(-16 * xv))
        ax.plot([xv], [pv], "o", color=C["enc"], ms=9)
        ax.annotate(rf"$x = {xv}$:  $p(\mathrm{{B}}\mid x) = {pv:.2f}$", xy=(xv, pv), xytext=(xv + 0.4, pv - 0.32 if xv < 1 else pv - 0.2),
                    fontsize=15, color=C["ink"], arrowprops=dict(arrowstyle="-", color=C["ink3"]))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("observed $x$")
    ax.set_ylabel(r"$p(\mathrm{B}\mid x)$")
    ax.grid(axis="y", color=C["grid"])
    save(fig, "toy-posterior")
    NUM["toy"]["post_01"] = r(1 / (1 + math.exp(-1.6)), 3)

    # ---------------- AE vs VAE latent spaces
    n_sc = 5000
    xs_te, ys_te = xte[:n_sc], yte[:n_sc].numpy()
    mu_ae, _ = encode(ae2, xs_te)
    mu_vae, _ = encode(vae2, xs_te)
    mu_b8, _ = encode(vae2b8, xs_te)
    prior = torch.randn(400, 2, generator=torch.Generator().manual_seed(11))

    def scatter(ax, mu, labels, s=5, alpha=.55, lab=True, fs=13):
        mu = mu.numpy()
        for d in range(10):
            m = labels == d
            ax.scatter(mu[m, 0], mu[m, 1], s=s, color=DIGIT[d], alpha=alpha, lw=0, rasterized=True)
        if lab:
            for d in range(10):
                c = np.median(mu[labels == d], 0)
                ax.text(c[0], c[1], str(d), fontsize=fs, fontweight="bold", ha="center", va="center", color=C["ink"],
                        bbox=dict(boxstyle="circle,pad=0.12", fc=(1, 1, 1, 0.85), ec=DIGIT[d], lw=1.6), zorder=7)

    def latent_fig(mu, name, with_prior=False, lim=None, size=(6.4, 6.0), circle=True, star=None):
        fig, ax = plt.subplots(figsize=size)
        scatter(ax, mu, ys_te)
        if star is not None:
            ax.plot([star[0]], [star[1]], marker="*", ms=24, color=C["x"], mec="white", mew=1.5, zorder=9)
        if circle:
            for k in (1, 2):
                ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.6, ls=(0, (4, 3)), zorder=8))
        if with_prior:
            # random codes on top of everything, as black crosses so they cannot be mistaken for a digit class
            ax.scatter(prior[:150, 0], prior[:150, 1], s=24, marker="x", color=C["ink"], zorder=6, lw=1.2)
        if with_prior and lim is not None and lim[0][1] - lim[0][0] > 12:
            # wide autoencoder map: the draws sit in a tiny patch, so add a zoom on [-3, 3]^2
            ins = ax.inset_axes([0.6, 0.03, 0.37, 0.37])
            scatter(ins, mu, ys_te, s=3, alpha=.22, lab=False)
            for k in (1, 2):
                ins.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.4, ls=(0, (4, 3)), zorder=8))
            ins.scatter(prior[:150, 0], prior[:150, 1], s=20, marker="x", color=C["ink"], zorder=10, lw=1.2)
            ins.set_xlim(-3, 3)
            ins.set_ylim(-3, 3)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_facecolor("white")
            ins.set_title("zoom: $[-3, 3]^2$", fontsize=11, color=C["ink2"], pad=3)
            for sp in ins.spines.values():
                sp.set_edgecolor(C["ink3"])
            ax.indicate_inset_zoom(ins, edgecolor=C["ink3"], lw=1)
        if lim is not None:
            ax.set_xlim(*lim[0])
            ax.set_ylim(*lim[1])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
        ax.grid(color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        save(fig, name)

    q = np.percentile(mu_ae.numpy(), [0.5, 99.5], axis=0)
    pad = 0.08 * (q[1] - q[0])
    ae_lim = ((q[0, 0] - pad[0], q[1, 0] + pad[0]), (q[0, 1] - pad[1], q[1, 1] + pad[1]))
    code3_ae, _ = encode(ae2, x3.unsqueeze(0))
    code3_ae = code3_ae[0]
    NUM["our3_ae"] = [r(v, 1) for v in code3_ae]
    tile_png(decode(ae2, code3_ae.unsqueeze(0))[0].numpy(), "digit3-ae2-recon")
    latent_fig(mu_ae, "ae-scatter", lim=ae_lim, star=code3_ae.numpy(), circle=False)

    # what N(0, I) looks like: 500 draws of two independent standard normals
    dr = torch.randn(500, 2, generator=torch.Generator().manual_seed(13)).numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(dr[:, 0], dr[:, 1], s=14, color=C["z"], alpha=.75, lw=0.4, ec="white")
    for k, frac in ((1, 1 - math.exp(-0.5)), (2, 1 - math.exp(-2))):
        ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.8, ls=(0, (4, 3))))
        inside = np.mean(np.hypot(dr[:, 0], dr[:, 1]) < k)
        ax.text(0.9, 3.3 - 0.42 * k, rf"inside radius {k}: {100 * inside:.0f}%", fontsize=14, color=C["ink2"])
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")
    ax.axhline(0, color=C["line"], lw=1)
    ax.axvline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    save(fig, "std-normal-2d")
    latent_fig(mu_ae, "ae-scatter-prior", with_prior=True, lim=ae_lim)
    latent_fig(mu_vae, "vae-scatter", lim=((-4, 4), (-4, 4)), star=mu3.numpy())
    latent_fig(mu_vae, "vae-scatter-prior", with_prior=True, lim=((-4, 4), (-4, 4)))
    NUM["ae_range"] = [[r(q[0, 0], 1), r(q[1, 0], 1)], [r(q[0, 1], 1), r(q[1, 1], 1)]]

    def nearest_code_dist(codes, pts):
        return torch.cdist(pts, codes).min(1).values

    NUM["prior_to_code"] = dict(ae=r(nearest_code_dist(mu_ae, prior).median(), 2),
                                vae=r(nearest_code_dist(mu_vae, prior).median(), 2))

    # generations from the prior
    zs = prior[:10]
    grid_png(decode(ae2, zs).numpy(), 1, 10, "ae-prior-decodes", scale=4)
    grid_png(decode(vae2, zs).numpy(), 1, 10, "vae-prior-decodes", scale=4)
    pis = decode(vae2, zs[2:3])[0]
    tile_png(pis.numpy(), "story-pi")
    tile_png(torch.bernoulli(pis, generator=torch.Generator().manual_seed(3)).numpy(), "story-sample")
    tile_png(decode(vae2, zs[0:1])[0].numpy(), "vae-prior-one")
    NUM["story_z"] = [r(v, 2) for v in zs[2]]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for k in (1, 2, 3):
        ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.4, ls=(0, (4, 3)), alpha=.7))
    ax.plot([float(zs[2][0])], [float(zs[2][1])], "o", color=C["z"], ms=13, mec="white", mew=2)
    ax.annotate(rf"$z = ({float(zs[2][0]):.2f},\ {float(zs[2][1]):.2f})$", xy=(float(zs[2][0]), float(zs[2][1])),
                xytext=(0.6, 2.3), fontsize=16, color=C["z"], arrowprops=dict(arrowstyle="-", color=C["z"]))
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.4, 3.4)
    ax.set_aspect("equal")
    ax.axhline(0, color=C["line"], lw=1)
    ax.axvline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    save(fig, "story-plane")
    z16 = torch.randn(10, 16, generator=torch.Generator().manual_seed(12))
    grid_png(decode(ae16, z16).numpy(), 1, 10, "ae16-prior-decodes", scale=4)
    grid_png(decode(ae16, z16).numpy(), 2, 5, "ae16-prior-decodes-2x5", scale=6, pad=2)
    grid_png(decode(vae16, z16).numpy(), 2, 5, "vae16-prior-decodes-2x5", scale=6, pad=2)
    grid_png(decode(vae16, z16).numpy(), 1, 10, "vae16-prior-decodes", scale=4)

    # reconstructions of the first ten distinct digits
    pick = [int(torch.nonzero(yte == d)[1]) for d in range(10)]
    xp = xte[pick]
    mu_p, _ = encode(ae2, xp)
    grid_png(np.concatenate([xp.flatten(1).numpy(), decode(ae2, mu_p).numpy()]), 2, 10, "ae-recon", scale=4)
    mu_p, _ = encode(vae2, xp)
    grid_png(np.concatenate([xp.flatten(1).numpy(), decode(vae2, mu_p).numpy()]), 2, 10, "vae-recon", scale=4)
    mu_p, _ = encode(ae16, xp)
    grid_png(np.concatenate([xp.flatten(1).numpy(), decode(ae16, mu_p).numpy()]), 2, 10, "ae16-recon", scale=4)

    # decoded grid across the AE code box (shows the holes)
    gx = torch.linspace(ae_lim[0][0], ae_lim[0][1], 12)
    gy = torch.linspace(ae_lim[1][1], ae_lim[1][0], 12)
    zz = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1).reshape(-1, 2)
    grid_png(decode(ae2, zz).numpy(), 12, 12, "ae-grid", scale=3, pad=1)

    # VAE samples: 2-D vs 16-D latent
    grid_png(decode(vae2, torch.randn(48, 2, generator=torch.Generator().manual_seed(21))).numpy(), 6, 8, "vae2-samples", scale=3)
    grid_png(decode(vae16, torch.randn(48, 16, generator=torch.Generator().manual_seed(21))).numpy(), 6, 8, "vae16-samples", scale=3)

    # Kingma-style manifold: inverse-CDF spaced grid; plus a dense sprite for the explorer
    normal = torch.distributions.Normal(0, 1)
    u = torch.linspace(0.04, 0.96, 18)
    gz = normal.icdf(u)
    zz = torch.stack(torch.meshgrid(gz.flip(0), gz, indexing="ij")[::-1], -1).reshape(-1, 2)
    grid_png(decode(vae2, zz).numpy(), 18, 18, "vae-manifold", scale=2, pad=1)
    lin = torch.linspace(-3, 3, 41)
    zz = torch.stack(torch.meshgrid(lin.flip(0), lin, indexing="ij")[::-1], -1).reshape(-1, 2)
    sprite = decode(vae2, zz).numpy().reshape(41, 41, 28, 28).transpose(0, 2, 1, 3).reshape(41 * 28, 41 * 28)
    plt.imsave(FIG / "vae-sprite.png", to_rgb(sprite))
    latent_fig(mu_vae, "vae-scatter-3", lim=((-3, 3), (-3, 3)), size=(5.6, 5.6), circle=False)

    # interpolation 3 -> 7 in AE and VAE
    def interp(m, name, steps=10):
        mu_a, _ = encode(m, x3.unsqueeze(0))
        mu_b, _ = encode(m, x7.unsqueeze(0))
        t = torch.linspace(0, 1, steps).unsqueeze(1)
        grid_png(decode(m, (1 - t) * mu_a + t * mu_b).numpy(), 1, steps, name, scale=4)

    interp(ae2, "interp-ae")
    interp(vae2, "interp-vae")
    interp(vae16, "interp-vae16", steps=12)
    interp(ae16, "interp-ae16", steps=12)

    # beta panel: AE / VAE beta=1 / VAE beta=8
    for nm, m, mu in (("ae2", ae2, mu_ae), ("vae2", vae2, mu_vae), ("vae2b8", vae2b8, mu_b8)):
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        scatter(ax, mu, ys_te, s=3, fs=10)
        for k in (1, 2):
            ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.4, ls=(0, (4, 3)), zorder=5))
        if nm != "ae2":
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        else:
            ax.set_xlim(*ae_lim[0])
            ax.set_ylim(*ae_lim[1])
        ax.set_aspect("equal")
        ax.grid(color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        save(fig, f"beta-{nm}-scatter")
        mu_p, _ = encode(m, xp[:8])
        grid_png(decode(m, mu_p).numpy(), 1, 8, f"beta-{nm}-recon", scale=3)
        grid_png(decode(m, prior[:8]).numpy(), 1, 8, f"beta-{nm}-samples", scale=3)
    NUM["beta"] = {nm: dict(recon=r(meta[nm]["history"][-1]["test_recon"], 1), kl=r(meta[nm]["history"][-1]["test_kl"], 1))
                   for nm in ("ae2", "vae2", "vae2b8")}

    # training curves (two charts: different scales, never a dual axis)
    h = meta["vae2"]["history"]
    ep = [row["epoch"] for row in h]
    for key, col, lab, nm in (("test_recon", C["x"], "reconstruction  (nats)", "train-recon"),
                              ("test_kl", C["loss"], "KL to prior  (nats)", "train-kl")):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(ep, [row[key] for row in h], color=col)
        ax.set_xlabel("epoch")
        ax.set_title(lab, fontsize=15, color=C["ink2"], loc="left")
        ax.grid(axis="y", color=C["grid"])
        save(fig, nm)
    NUM["vae2_final"] = dict(recon=r(h[-1]["test_recon"], 1), kl=r(h[-1]["test_kl"], 1),
                             neg_elbo=r(h[-1]["test_recon"] + h[-1]["test_kl"], 1), epochs=len(h))
    NUM["vae16_final"] = dict(recon=r(meta["vae16"]["history"][-1]["test_recon"], 1),
                              kl=r(meta["vae16"]["history"][-1]["test_kl"], 1))
    NUM["ae2_final"] = dict(recon=r(meta["ae2"]["history"][-1]["test_recon"], 1))
    NUM["ae16_final"] = dict(recon=r(meta["ae16"]["history"][-1]["test_recon"], 1))

    # ---------------- the evidence integral for our 3, computed on a grid (possible because z is 2-D)
    step = 0.01
    lin = torch.arange(-4.5, 4.5 + 1e-9, step, dtype=torch.float64)
    Z = torch.stack(torch.meshgrid(lin, lin, indexing="xy"), -1).reshape(-1, 2)
    llik = log_px_given_z(vae2, x3, Z.float())
    lprior = -0.5 * (Z ** 2).sum(1) - math.log(2 * math.pi)
    joint = llik + lprior
    log_px = torch.logsumexp(joint, 0).item() + 2 * math.log(step)
    n = len(lin)
    # q for our 3 and the ELBO (analytic KL + many-sample reconstruction term)
    eps = torch.randn(20000, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    zq = mu3 + sd3 * eps
    e_llik = log_px_given_z(vae2, x3, zq.float()).mean().item()
    kl_prior = kl3.sum().item()
    elbo = e_llik - kl_prior
    # KL(q || true posterior) directly, as a check: E_q[log q - log p(z|x)]
    log_q = (-0.5 * eps ** 2 - torch.log(sd3) - 0.5 * math.log(2 * math.pi)).sum(1)
    log_post = log_px_given_z(vae2, x3, zq.float()) - 0.5 * (zq ** 2).sum(1) - math.log(2 * math.pi) - log_px
    kl_post = (log_q - log_post).mean().item()
    # sampling z from the prior instead: how often do we land where this 3 is plausible?
    zp = torch.randn(100000, 2, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    lp = log_px_given_z(vae2, x3, zp.float())
    frac = (lp > (lp.max() - 10)).double().mean().item()
    in_q = (((zp - mu3) / sd3).norm(dim=1) < 2).double().mean().item()
    NUM["evidence"] = dict(log_px=r(log_px, 1), elbo=r(elbo, 1), e_llik=r(e_llik, 1), kl_prior=r(kl_prior, 2),
                           gap=r(log_px - elbo, 2), kl_post_mc=r(kl_post, 2),
                           prior_hits_2sd=r(in_q * 100, 2), prior_draws=100000,
                           llik_best=r(lp.max(), 1), llik_median_prior=r(lp.median(), 1))

    L = llik.reshape(n, n).numpy()
    J = joint.reshape(n, n).numpy()
    ext = [-4.5, 4.5, -4.5, 4.5]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(np.clip(L, L.max() - 120, None), origin="lower", extent=ext, cmap="Blues", interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(r"$\log p_\theta(x\mid z)$ for our 3", color=C["ink2"])
    cb.outline.set_visible(False)
    for k in (1, 2, 3):
        ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=C["z"], lw=1.3, ls=(0, (4, 3))))
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    save(fig, "evidence-map")

    # zoom: true posterior vs encoder's q
    post = np.exp(J - J.max())
    cx, cy = mu3.numpy()
    w = max(float(sd3.max()) * 6, 0.45)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    xx = lin.numpy()
    sel_x = (xx > cx - w) & (xx < cx + w)
    sel_y = (xx > cy - w) & (xx < cy + w)
    sub = post[np.ix_(sel_y, sel_x)]
    ax.contourf(xx[sel_x], xx[sel_y], sub, levels=np.linspace(0.02, 1, 8), cmap="Blues", alpha=.85)
    for k in (1, 2):
        ax.add_patch(Ellipse((cx, cy), 2 * k * sd3[0].item(), 2 * k * sd3[1].item(), fill=False, ec=C["enc"], lw=2.4))
    ax.plot([cx], [cy], "o", color=C["enc"], ms=7)
    ax.text(cx + 0.35 * w, cy + 0.75 * w, r"encoder's $q_\varphi(z\mid x)$", color=C["enc"], fontsize=15)
    ax.text(cx - 0.95 * w, cy - 0.85 * w, r"true posterior $p_\theta(z\mid x)$" + "\n(shaded, from the grid)", color=C["x"], fontsize=15)
    ax.set_xlim(cx - w, cx + w)
    ax.set_ylim(cy - w, cy + w)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$", rotation=0, labelpad=12)
    save(fig, "posterior-vs-q")

    # ---------------- KL figures
    P = np.array([0.8, 0.2])
    qs = np.linspace(0.005, 0.995, 400)
    kl_pq = P[0] * np.log(P[0] / qs) + P[1] * np.log(P[1] / (1 - qs))
    kl_qp = qs * np.log(qs / P[0]) + (1 - qs) * np.log((1 - qs) / P[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(qs, kl_pq, color=C["loss"])
    ax.plot(qs, kl_qp, color=C["ink2"], ls=(0, (5, 3)))
    ax.text(0.085, 2.0, r"$\mathrm{KL}(P\,\Vert\,Q)$", color=C["loss"], fontsize=17)
    ax.text(0.02, 0.62, r"$\mathrm{KL}(Q\,\Vert\,P)$", color=C["ink2"], fontsize=17)
    for v, col in ((0.6, C["loss"]),):
        a = P[0] * math.log(P[0] / v) + P[1] * math.log(P[1] / (1 - v))
        b = v * math.log(v / P[0]) + (1 - v) * math.log((1 - v) / P[1])
        ax.plot([v, v], [a, b], "o", color=C["ink"], ms=7)
        ax.annotate(rf"$q = 0.6$:  ${a:.3f}$ vs ${b:.3f}$", xy=(v, b), xytext=(0.35, 1.25), fontsize=14, color=C["ink"],
                    arrowprops=dict(arrowstyle="-", color=C["ink3"]))
    ax.plot([0.8], [0], "o", color=C["x"], ms=8)
    ax.annotate(r"$q = 0.8 = p$: both are 0", xy=(0.8, 0), xytext=(0.6, 0.55), fontsize=14, color=C["x"],
                arrowprops=dict(arrowstyle="-", color=C["x"]))
    ax.set_ylim(0, 2.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"model's probability of a 1, $q$")
    ax.set_ylabel("nats")
    ax.grid(color=C["grid"])
    save(fig, "kl-vs-q")

    # ---------------- KL, slowly: surprise curve, tangent bound, evidence piling up
    pp = np.linspace(0.01, 1, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(pp, -np.log(pp), color=C["ink2"], lw=2.4)
    for v, col, lab, dx, dy in ((0.8, C["x"], r"a 1 under $P$", 0.03, 0.18), (0.6, C["dec"], r"a 1 under $Q$", 0.03, 0.22),
                                (0.4, C["dec"], r"a 7 under $Q$", 0.03, 0.25), (0.2, C["x"], r"a 7 under $P$", 0.03, 0.25)):
        ax.plot([v], [-math.log(v)], "o", color=col, ms=9, mec="white", mew=1.5)
        ax.text(v + dx, -math.log(v) + dy, lab + rf": ${-math.log(v):.2f}$", fontsize=13, color=col)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"probability the model gave the digit, $p$")
    ax.set_ylabel(r"surprise $= -\log p$")
    ax.grid(color=C["grid"])
    save(fig, "surprise-curve")

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
    save(fig, "kl-tangent")

    nn = np.arange(0, 101)
    kl_pq = 0.8 * math.log(0.8 / 0.6) + 0.2 * math.log(0.2 / 0.4)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.semilogy(nn, np.exp(nn * kl_pq), color=C["loss"], lw=2.6)
    for n_, lab in ((10, "2.5×"), (50, "97×"), (100, "9,400×")):
        ax.plot([n_], [math.exp(n_ * kl_pq)], "o", color=C["loss"], ms=9, mec="white", mew=1.5)
        ax.text(n_ - 2, math.exp(n_ * kl_pq) * 1.9, lab, color=C["ink"], fontsize=15, ha="right" if n_ == 100 else "center")
    ax.set_xlabel("number of digits seen, $n$")
    ax.set_ylabel(r"$p(\text{data}\mid P)\ /\ p(\text{data}\mid Q)$")
    ax.set_ylim(0.8, 3e4)
    ax.grid(color=C["grid"], which="major")
    save(fig, "kl-evidence")

    # Gaussian KL: where does 0.818 come from?
    zg = np.linspace(-4, 4, 800)
    p_ = npdf(zg, 0, 1)
    q_ = npdf(zg, 1, .5)
    integrand = q_ * np.log(q_ / p_)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    a1.plot(zg, p_, color=C["z"])
    a1.plot(zg, q_, color=C["enc"])
    a1.text(-3.6, 0.3, r"prior $p(z) = \mathcal{N}(0, 1)$", color=C["z"], fontsize=15)
    a1.text(1.7, 0.62, r"$q(z) = \mathcal{N}(1,\,0.5^2)$", color=C["enc"], fontsize=15)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a2.fill_between(zg, 0, integrand, where=integrand >= 0, color=C["loss_t"], lw=0)
    a2.fill_between(zg, 0, integrand, where=integrand < 0, color="#E5E7EB", lw=0)
    a2.plot(zg, integrand, color=C["loss"], lw=2)
    a2.axhline(0, color=C["ink3"], lw=0.8)
    a2.text(-3.7, 0.3, r"$q(z)\,\log\frac{q(z)}{p(z)}$", color=C["loss"], fontsize=17)
    a2.text(1.9, 0.3, "area = 0.818 nats", color=C["ink"], fontsize=15, fontweight="bold")
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel(r"$z$")
    save(fig, "kl-gauss")
    NUM["kl_gauss_numeric"] = r(np.trapezoid(integrand, zg), 4)

    # forward vs reverse KL, fitting one Gaussian to our two bumps
    def fit(direction):
        best = None
        for m in np.linspace(-3, 3, 241):
            for s in np.linspace(0.2, 3, 141):
                qd = npdf(xs, m, s) + 1e-300
                pd = mix + 1e-300
                if direction == "fwd":
                    v = np.trapezoid(pd * np.log(pd / qd), xs)
                else:
                    v = np.trapezoid(qd * np.log(qd / pd), xs)
                if best is None or v < best[0]:
                    best = (v, m, s)
        return best

    fwd, rev = fit("fwd"), fit("rev")
    NUM["kl_fit"] = dict(fwd_mu=r(fwd[1], 2), fwd_sd=r(fwd[2], 2), rev_mu=r(rev[1], 2), rev_sd=r(rev[2], 2))
    for nm, (v, m, s), title in (("fwd", fwd, "minimise KL(p ‖ q)"), ("rev", rev, "minimise KL(q ‖ p)")):
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.fill_between(xs, 0, mix, color=C["x_t"])
        ax.plot(xs, mix, color=C["x"], lw=1.6)
        ax.plot(xs, npdf(xs, m, s), color=C["dec"], lw=2.8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(0.5, 1.08, rf"best $q = \mathcal{{N}}({m:.1f},\,{s:.2f}^2)$", fontsize=16, color=C["dec"], ha="center")
        save(fig, f"kl-{nm}")

    # Jensen's inequality picture
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = np.linspace(0.35, 5.2, 300)
    ax.plot(t, np.log(t), color=C["dec"])
    r1, r2 = 1.0, 4.0
    ax.plot([r1, r2], [math.log(r1), math.log(r2)], color=C["z"], lw=2, ls=(0, (4, 3)))
    for v in (r1, r2):
        ax.plot([v], [math.log(v)], "o", color=C["z"], ms=8)
    m_ = (r1 + r2) / 2
    ax.plot([m_], [math.log(m_)], "o", color=C["dec"], ms=10)
    ax.plot([m_], [(math.log(r1) + math.log(r2)) / 2], "o", color=C["z"], ms=10, mfc="white", mew=2.5)
    ax.annotate(rf"$\log(\mathrm{{average}}) = \log 2.5 = {math.log(m_):.3f}$", xy=(m_, math.log(m_)), xytext=(0.6, 1.55), fontsize=14,
                color=C["dec"], arrowprops=dict(arrowstyle="-", color=C["dec"]))
    ax.annotate(rf"average of logs $= {(math.log(r1) + math.log(r2)) / 2:.3f}$", xy=(m_, 0.693), xytext=(2.9, 0.15),
                fontsize=14, color=C["z"], arrowprops=dict(arrowstyle="-", color=C["z"]))
    ax.text(r1 + 0.18, -0.12, r"$R = 1$", ha="left", fontsize=14, color=C["ink2"])
    ax.text(r2, math.log(r2) + 0.12, r"$R = 4$", ha="center", fontsize=14, color=C["ink2"])
    ax.set_xlim(0, 5.4)
    ax.set_ylim(-1.1, 2.0)
    ax.axhline(0, color=C["line"], lw=1)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$\log R$")
    save(fig, "jensen")

    # ---------------- reparameterisation as small multiples: the same dots, stretched then shifted
    eps_dots = np.array([-1.7, -1.1, -0.7, -0.4, -0.1, 0.25, 0.6, 1.0, 1.6])
    mu_r, sd_r = 1.0, 0.5
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    rows = [(r"$\varepsilon \sim \mathcal{N}(0, 1)$", eps_dots, C["z"], 4.0, (0, 1)),
            (r"$\sigma\,\varepsilon$", sd_r * eps_dots, C["enc"], 2.0, (0, sd_r)),
            (r"$z = \mu + \sigma\varepsilon$", mu_r + sd_r * eps_dots, C["enc"], 0.0, (mu_r, sd_r))]
    dens_x = np.linspace(-2.6, 2.9, 300)
    for i, (lab, vals, col, y, (m0, s0)) in enumerate(rows):
        ax.plot([-2.6, 2.9], [y, y], color=C["line"], lw=1.4, zorder=1)
        dens = npdf(dens_x, m0, s0)
        ax.fill_between(dens_x, y, y + 0.9 * dens / dens.max(), color=col, alpha=.10, lw=0)
        ax.plot(dens_x, y + 0.9 * dens / dens.max(), color=col, lw=1.4, alpha=.7)
        for v in vals:
            special = abs(v - vals[3]) < 1e-9
            ax.plot([v], [y], "o", color=C["loss"] if special else col, ms=13 if special else 10,
                    mec="white", mew=1.5, zorder=4)
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
    save(fig, "reparam-dots")

    # ---------------- GAN: saturation of the two generator losses
    a = np.linspace(-7, 3, 400)
    D = 1 / (1 + np.exp(-a))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    a1.plot(a, np.log(1 - D), color=C["ink2"], ls=(0, (5, 3)))
    a1.plot(a, -np.log(D), color=C["dec"])
    a1.set_title("generator loss", loc="left", fontsize=15, color=C["ink2"])
    a1.text(-4.2, 5.9, r"$-\log D$  (used in practice)", color=C["dec"], fontsize=14)
    a1.text(-4.0, -1.3, r"$\log(1 - D)$  (minimax)", color=C["ink2"], fontsize=14)
    a1.set_ylim(-3.4, 7.6)
    a2.plot(a, -D, color=C["ink2"], ls=(0, (5, 3)))
    a2.plot(a, D - 1, color=C["dec"])
    a2.set_title(r"its slope with respect to $a$", loc="left", fontsize=15, color=C["ink2"])
    a0 = math.log(0.01 / 0.99)
    for ax_ in (a1, a2):
        ax_.axvline(a0, color=C["disc"], lw=1.4, ls=(0, (2, 2)))
        ax_.set_xlabel(r"D's logit $a$ for a fake,  $D = \mathrm{sigmoid}(a)$")
        ax_.grid(color=C["grid"])
    a2.plot([a0], [-0.01], "o", color=C["ink2"], ms=8)
    a2.plot([a0], [-0.99], "o", color=C["dec"], ms=8)
    a2.text(a0 + 0.3, -0.13, r"$-0.01$", ha="left", fontsize=16, color=C["ink2"])
    a2.text(a0 - 0.25, -0.93, r"$-0.99$", ha="right", fontsize=16, color=C["dec"])
    a1.text(a0 + 0.15, 7.0, r"$D = 0.01$", color=C["disc"], fontsize=14)
    save(fig, "saturation")

    # ---------------- GAN on the 1-D toy data
    runs = json.loads((MODELS / "toy_gan_runs.json").read_text())
    good = max(runs, key=lambda q_: min(q_["left"], q_["right"]))
    bad = json.loads((MODELS / "toy_gan_collapse.json").read_text())  # generator with 4 hidden units
    NUM["toygan"] = dict(good_seed=good["seed"], bad_seed=bad["seed"],
                         good=dict(left=r(good["left"], 2), right=r(good["right"], 2)),
                         bad=dict(left=r(bad["left"], 2), right=r(bad["right"], 2)),
                         all=[dict(seed=q_["seed"], left=r(q_["left"], 2), right=r(q_["right"], 2)) for q_ in runs])
    grid1 = np.array(good["grid"])

    def gan_panel(run, step, name, arrows=False):
        rec = run["record"][str(step)]
        s = np.array(rec["samples"])
        d = np.array(rec["d"])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
        a1.fill_between(xs, 0, mix, color=C["x_t"])
        a1.plot(xs, mix, color=C["x"], lw=1.5)
        a1.hist(np.clip(s, -5, 5), bins=np.linspace(-5, 5, 61), density=True, histtype="stepfilled",
                color=C["dec"], alpha=.55, lw=0)
        a1.set_ylim(0, 1.0)
        a1.set_yticks([])
        a1.spines["left"].set_visible(False)
        a1.text(-4.9, 0.85, f"step {step}", fontsize=16, fontweight="bold", color=C["ink"])
        if arrows:
            dD = np.gradient(d, grid1)
            pts = np.sort(s)[np.linspace(40, len(s) - 40, 14).astype(int)]
            gmax = max(abs(np.interp(pts, grid1, dD)).max(), 1e-9)
            for i_, v in enumerate(pts):
                gv = np.interp(v, grid1, dD)
                y0 = 0.55 + 0.06 * (i_ % 4)
                a1.plot([v], [y0], "o", color=C["dec"], ms=6, zorder=4)
                a1.annotate("", xy=(v + 0.9 * gv / gmax, y0), xytext=(v, y0),
                            arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=2.2, mutation_scale=16), zorder=5)
        a2.plot(grid1, d, color=C["disc"])
        a2.axhline(0.5, color=C["ink3"], lw=1, ls=(0, (3, 3)))
        a2.set_ylim(0, 1.02)
        a2.set_yticks([0, 0.5, 1])
        a2.set_ylabel(r"$D(x)$")
        a2.set_xlim(-5, 5)
        a2.set_xlabel(r"$x$")
        a2.grid(axis="y", color=C["grid"])
        save(fig, name)

    for st in (0, 25, 100, 250, 1000, 3000):
        gan_panel(good, st, f"gan1d-{st}")
    gan_panel(good, 25, "gan1d-arrows", arrows=True)
    gan_panel(bad, 3000, "gan1d-collapse")

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=True)
    for ax_, run, title in ((axs[0], good, "run that covers both bumps"), (axs[1], bad, "run that collapses to one bump")):
        Ls = np.array(run["losses"])
        k = 50
        sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")  # noqa: E731
        ax_.plot(sm(Ls[:, 0]), color=C["disc"], lw=2)
        ax_.plot(sm(Ls[:, 1]), color=C["dec"], lw=2)
        ax_.set_title(title, loc="left", fontsize=15, color=C["ink2"])
        ax_.set_xlabel("step")
        ax_.grid(color=C["grid"])
    for ax_, run in ((axs[0], good), (axs[1], bad)):
        Ls = np.array(run["losses"])
        dl, gl = Ls[-500:, 0].mean(), Ls[-500:, 1].mean()
        up, lo = (0.09, -0.13)
        ax_.text(2050, dl + (up if dl > gl else lo), "D loss", color=C["disc"], fontsize=14)
        ax_.text(2050, gl + (up if gl > dl else lo), "G loss", color=C["dec"], fontsize=14)
    axs[0].set_ylim(0.3, 1.8)
    save(fig, "gan1d-losses")

    # ---------------- MNIST GAN: watch it learn, then vote
    snaps = np.load(MODELS / "gan_mnist_snapshots.npz")
    keys = ["0", "step50", "step150", "step300", "1", "3", "10", "30"]
    film = np.concatenate([snaps[k][:8] for k in keys])
    grid_png(film, len(keys), 8, "gan-film", scale=3, pad=2)
    grid_png(snaps["30"][:48], 6, 8, "gan-samples", scale=3)
    ck = torch.load(MODELS / "gan_mnist.pt", map_location=DEV)
    Dnet = ConvD().to(DEV)
    Dnet.load_state_dict(ck["D"])
    Dnet.eval()
    real_idx = torch.randperm(10000, generator=torch.Generator().manual_seed(31))[:6]
    xgray = load_mnist(binary=False)["test"][0]
    real = xgray[real_idx].numpy()
    fake = snaps["30"][8:14]
    items = [("real", im) for im in real] + [("generated", im) for im in fake]
    order = np.random.default_rng(8).permutation(12)
    vote = []
    for k_, j in enumerate(order):
        kind, im = items[j]
        with torch.no_grad():
            pr = torch.sigmoid(Dnet(torch.tensor(im * 2 - 1, dtype=torch.float32).reshape(1, 1, 28, 28).to(DEV))).item()
        tile_png(im, f"vote-{k_ + 1:02d}", scale=6)
        vote.append(dict(n=k_ + 1, kind=kind, d_real=r(pr, 2)))
    NUM["vote"] = vote
    tile_png(snaps["30"][2], "gan-fake-one", scale=6)       # a GAN sample (an 8)
    tile_png(xgray[i3].numpy(), "gan-real-one", scale=6)     # our 3, grey levels
    tile_png(snaps["30"][47], "gan-seven", scale=6)          # a GAN sample that happens to be a 7
    NUM["gan_final"] = meta["gan"]["history"][-1]

    # ---------------- next lecture teaser: our 3 with more and more noise
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(28, 28))
    frames = []
    for abar in (1.0, 0.8, 0.5, 0.25, 0.08, 0.0):
        v = math.sqrt(abar) * (2 * img - 1) + math.sqrt(1 - abar) * noise
        frames.append(np.clip((v + 1) / 2, 0, 1))
    grid_png(np.array(frames), 1, 6, "noise-strip", scale=5, pad=3)

    # ---------------- conditional VAE: the label picks the digit, z picks the style
    cv = CondVAE(2).to(DEV)
    cv.load_state_dict(torch.load(MODELS / "cvae2.pt", map_location=DEV))
    cv.eval()
    zc = torch.randn(8, 2, generator=torch.Generator().manual_seed(41)).to(DEV)
    with torch.no_grad():
        rows = [torch.sigmoid(cv.decode(zc, torch.full((8,), d, device=DEV))).cpu() for d in range(10)]
    grid_png(torch.cat(rows).numpy(), 10, 8, "cvae-grid", scale=3, pad=1)
    grid_png(rows[7].numpy(), 1, 8, "cvae-sevens", scale=4)
    NUM["cvae"] = dict(recon=r(meta["cvae2"]["history"][-1]["train_recon"], 1), kl=r(meta["cvae2"]["history"][-1]["train_kl"], 1))

    # ---------------- route A worked position on our 3 (filled in once the pixel models exist)
    bits = x3.reshape(28, 28).int()
    NUM["digit3_bits"] = bits.flatten().tolist()
    # the first pixel of a stroke: the 3 pixels to its left are paper, but there is ink just above
    ink = [(rr, cc) for rr in range(3, 28) for cc in range(3, 24)
           if bits[rr, cc] and bits[rr, cc - 3:cc].sum() == 0 and bits[rr - 3:rr, cc - 3:cc + 4].sum() > 0]
    ar_r, ar_c = ink[len(ink) // 2]
    NUM["routeA"] = dict(r=ar_r, c=ar_c)

    # ---------------- published DiffusionSat crops: caption + metadata as the condition
    from PIL import Image
    sat = Image.open(ROOT / "assets" / "gallery" / "satellite-samples.jpg")
    sat.crop((822, 152, 1180, 327)).save(FIG / "sat-substation.jpg", quality=92)     # August | January, first row
    sat.crop((406, 152, 765, 327)).save(FIG / "sat-stadium.jpg", quality=92)         # France | USA coordinates

    # ---------------- route A: two pixel-by-pixel models
    xt = xte.unsqueeze(1)
    for ctx in ("row3", "window7"):
        m_ = PixelAR(ctx).to(DEV)
        m_.load_state_dict(torch.load(MODELS / f"pixelar_{ctx}.pt", map_location=DEV))
        m_.eval()
        smp = m_.sample(16, generator=torch.Generator().manual_seed(5)).cpu()
        grid_png(smp[:, 0].numpy(), 2, 8, f"pixelar-{ctx}-samples", scale=4)
        with torch.no_grad():
            x3t = x3.reshape(1, 1, 28, 28).to(DEV)
            lp = -F.binary_cross_entropy_with_logits(m_(x3t), x3t, reduction="sum").item()
        with torch.no_grad():
            NUM["routeA"][f"p_{ctx}"] = r(torch.sigmoid(m_(x3t))[0, 0, ar_r, ar_c].item(), 2)
        h = meta["pixelar"][0 if ctx == "row3" else 1]["history"][-1]
        NUM[f"pixelar_{ctx}"] = dict(logp3=r(lp, 1), test_nll=r(h["test_nll"], 1))

    # cover: a slow walk in the 16-D VAE
    (FIG / "numbers.json").write_text(json.dumps(NUM, indent=1))
    print(json.dumps({k: NUM[k] for k in ("our3", "evidence", "toy", "prior_to_code", "kl_fit", "toygan", "beta")}, indent=1))


if __name__ == "__main__":
    main()
