"""From PCA to autoencoders, and three ways to build one (MLP, CNN, ViT).

    python scripts/make_figures_bridge.py      # after make_figures_gmm.py; ~10 min on an M-series GPU the first time

Grey-level MNIST and squared error throughout, so every number compares directly with PCA:
1. PCA at d = 2 and 16, exactly (one SVD).
2. A linear autoencoder (one Linear each way, no activations) trained by gradient descent: it should find PCA's subspace.
3. An MLP autoencoder at the same d: the squeeze can now bend.
4. The same job with a CNN and with a small ViT at d = 16: same loss, different networks.
5. Part 1's mixture of bumps fitted to the MLP autoencoder's 16 numbers, sampled and decoded, against the same on PCA.

Models are saved in assets/models/bridge/ and reused. Writes figures/data/bridge-*, arch-*, aecodes-* and
figures/data/numbers-bridge.json.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, DEV, DIGIT, FIG, grid_png, plt, save  # noqa: E402
from make_figures_gmm import avg_log_prob, fit_gmm, mixture  # noqa: E402
from train_models import load_mnist  # noqa: E402

OUT = ROOT / "assets" / "models" / "bridge"
BASE = json.loads((FIG / "numbers.json").read_text())


def r(v, k=2):
    return float(round(float(v), k))


# =============================================================== the networks (all map [B, 1, 28, 28] -> [B, 1, 28, 28])
class LinearAE(nn.Module):
    """PCA's shape, learned: one linear map in, one out, nothing in between."""
    def __init__(self, d):
        super().__init__()
        self.enc, self.dec = nn.Linear(784, d), nn.Linear(d, 784)

    def encode(self, x):
        return self.enc(x.flatten(1))

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))


class MLPAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, d))
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))


class CNNAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),            # 28 -> 14
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),           # 14 -> 7
                                 nn.Flatten(), nn.Linear(64 * 7 * 7, d))
        self.dec = nn.Sequential(nn.Linear(d, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 7 -> 14
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())  # 14 -> 28

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class ViTAE(nn.Module):
    """16 patches of 7 x 7 pixels -> tokens -> transformer -> one code; and back."""
    def __init__(self, d, width=64, depth=3, heads=4, p=7):
        super().__init__()
        self.p, self.n = p, (28 // p) ** 2
        layer = lambda: nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True)  # noqa: E731
        self.embed = nn.Linear(p * p, width)
        self.pos_in = nn.Parameter(torch.randn(1, self.n, width) * 0.02)
        self.enc_tf = nn.TransformerEncoder(layer(), depth)
        self.to_code = nn.Linear(width, d)
        self.from_code = nn.Linear(d, self.n * width)
        self.pos_out = nn.Parameter(torch.randn(1, self.n, width) * 0.02)
        self.dec_tf = nn.TransformerEncoder(layer(), depth)
        self.to_patch = nn.Linear(width, p * p)
        self.width = width

    def patches(self, x):                          # [B,1,28,28] -> [B,16,49]
        B, p = len(x), self.p
        return x.view(B, 1, 28 // p, p, 28 // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, self.n, p * p)

    def unpatch(self, t):                          # [B,16,49] -> [B,1,28,28]
        B, p, g = len(t), self.p, 28 // self.p
        return t.view(B, g, g, 1, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, 1, 28, 28)

    def encode(self, x):
        h = self.enc_tf(self.embed(self.patches(x)) + self.pos_in)
        return self.to_code(h.mean(1))             # average the 16 tokens, then squeeze

    def decode(self, z):
        h = self.from_code(z).view(-1, self.n, self.width) + self.pos_out
        return torch.sigmoid(self.unpatch(self.to_patch(self.dec_tf(h))))

    def forward(self, x):
        return self.decode(self.encode(x))


def sq_err(xh, x):
    return ((xh - x) ** 2).flatten(1).sum(1)       # squared error per image, summed over 784 pixels


def train(name, model, xtr, epochs, lr=1e-3, bs=256):
    f = OUT / f"{name}.pt"
    model = model.to(DEV)
    if f.exists():
        saved = torch.load(f, map_location=DEV)
        model.load_state_dict(saved["state"])
        return model.eval(), saved["secs"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(xtr) / bs))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(epochs):
        for idx in torch.randperm(len(xtr), generator=g).split(bs):
            x = xtr[idx].to(DEV)
            loss = sq_err(model(x), x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    secs = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), secs=secs), f)
    return model.eval(), secs


def test_err(model, xte):
    with torch.no_grad():
        return torch.cat([sq_err(model(xte[i:i + 2000].to(DEV)), xte[i:i + 2000].to(DEV)).cpu() for i in range(0, len(xte), 2000)]).mean().item()


def codes(model, x):
    with torch.no_grad():
        return torch.cat([model.encode(x[i:i + 2000].to(DEV)).cpu() for i in range(0, len(x), 2000)])


def decode(model, z):
    with torch.no_grad():
        return model.decode(z.float().to(DEV)).cpu().reshape(-1, 784).clamp(0, 1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    data = load_mnist(binary=False)
    xtr = data["train"][0].unsqueeze(1)            # [60000, 1, 28, 28], grey levels
    xte, yte = data["test"][0].unsqueeze(1), data["test"][1]
    flat_tr, flat_te = xtr.reshape(-1, 784), xte.reshape(-1, 784)

    # ---- PCA, exactly
    mean = flat_tr.mean(0)
    _, _, Vt = torch.linalg.svd(flat_tr - mean, full_matrices=False)
    pca = {}
    for d in (2, 16):
        V = Vt[:d]
        rec = mean + (flat_te - mean) @ V.T @ V
        pca[d] = dict(err=((rec - flat_te) ** 2).sum(1).mean().item(), rec=rec)

    # ---- the networks
    m = {}
    for d in (2, 16):
        m[f"linear{d}"] = train(f"linear{d}", LinearAE(d), xtr, epochs=150, lr=3e-3)
        m[f"mlp{d}"] = train(f"mlp{d}", MLPAE(d), xtr, epochs=20)
    m["cnn16"] = train("cnn16", CNNAE(16), xtr, epochs=20)
    m["vit16"] = train("vit16", ViTAE(16), xtr, epochs=20, lr=5e-4)
    res = {}
    for k, (model, secs) in m.items():
        res[k] = dict(err=r(test_err(model, xte), 1), params=nparams(model), secs=r(secs, 0))
        print(k, res[k], flush=True)

    # the linear autoencoder's decoder spans (almost) PCA's directions: overlap 1 = same subspace
    overlap = {}
    for d in (2, 16):
        W = m[f"linear{d}"][0].dec.weight.detach().cpu()          # [784, d]
        Q, _ = torch.linalg.qr(W)
        overlap[d] = r((Vt[:d] @ Q).pow(2).sum() / d, 3)

    # ---- reconstructions: 10 test digits, our 3 among them
    i3 = BASE["digit3_index"]
    pick = [i3 if c == 3 else int((yte == c).nonzero()[0]) for c in range(10)]
    xs = xte[pick]
    rows = lambda *ims: torch.cat([im.reshape(-1, 784) for im in ims]).clamp(0, 1).numpy()  # noqa: E731
    for d in (2, 16):
        grid_png(rows(xs, pca[d]["rec"][pick], decode(m[f"linear{d}"][0], codes(m[f"linear{d}"][0], xs)),
                      decode(m[f"mlp{d}"][0], codes(m[f"mlp{d}"][0], xs))), 4, 10, f"bridge-recon-d{d}", scale=3)
    grid_png(rows(xs, *[decode(m[k][0], codes(m[k][0], xs)) for k in ("mlp16", "cnn16", "vit16")]), 4, 10, "arch-recon", scale=3)

    # ---- 2-D codes: PCA vs the MLP autoencoder
    sub = torch.randperm(len(xte), generator=torch.Generator().manual_seed(0))[:5000]
    z_pca = ((flat_te - mean) @ Vt[:2].T)[sub]
    z_ae = codes(m["mlp2"][0], xte[sub])
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.4))
    for ax, z, title in ((axs[0], z_pca, "PCA: 2 straight-line numbers"), (axs[1], z_ae, "autoencoder: 2 learned numbers")):
        for c in range(10):
            sel = yte[sub] == c
            ax.scatter(z[sel, 0], z[sel, 1], s=4, color=DIGIT[c], alpha=.5, lw=0, rasterized=True)
            med = z[sel].median(0).values
            ax.text(med[0], med[1], str(c), fontsize=12, fontweight="bold", ha="center", va="center", color=C["ink"],
                    bbox=dict(boxstyle="circle,pad=0.12", fc=(1, 1, 1, 0.85), ec=DIGIT[c], lw=1.4), zorder=9)
        ax.set_title(title, loc="left", fontsize=15, color=C["ink2"])
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$", rotation=0, labelpad=10)
        ax.grid(color=C["grid"])
    save(fig, "bridge-scatter")

    # ---- generating: Part 1's bumps fitted to 16 numbers, from PCA and from the MLP autoencoder
    z_tr_ae, z_te_ae = codes(m["mlp16"][0], xtr), codes(m["mlp16"][0], xte)
    z_tr_pca, z_te_pca = (flat_tr - mean) @ Vt[:16].T, (flat_te - mean) @ Vt[:16].T
    K = 50
    gm_ae, gm_pca = mixture(*fit_gmm(K, z_tr_ae)), mixture(*fit_gmm(K, z_tr_pca))
    torch.manual_seed(7)
    s_ae = decode(m["mlp16"][0], gm_ae.sample((24,)))
    torch.manual_seed(7)
    s_pca = (mean + gm_pca.sample((24,)) @ Vt[:16]).clamp(0, 1)
    torch.manual_seed(7)
    s_rand = decode(m["mlp16"][0], torch.randn(24, 16))
    grid_png(s_rand.numpy(), 2, 12, "aecodes-random", scale=3)
    grid_png(s_pca.numpy(), 2, 12, "aecodes-pca-gmm", scale=3)
    grid_png(s_ae.numpy(), 2, 12, "aecodes-ae-gmm", scale=3)
    spread = z_tr_ae.std(0)

    num = dict(pca2=r(pca[2]["err"], 1), pca16=r(pca[16]["err"], 1), pca16_params=784 + 16 * 784, overlap2=overlap[2], overlap16=overlap[16],
               **res, k_codes=K, code_sd_min=r(spread.min(), 1), code_sd_max=r(spread.max(), 1),
               gmm_test_ae=r(avg_log_prob(gm_ae, z_te_ae), 1), gmm_test_pca=r(avg_log_prob(gm_pca, z_te_pca), 1))
    (FIG / "numbers-bridge.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1))


if __name__ == "__main__":
    main()
