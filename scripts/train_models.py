"""Train every model whose output appears in the Generative Modelling I slides.

All figures in the lecture are drawn from these checkpoints (see make_figures.py).

    python scripts/train_models.py            # everything (~10 min on an M-series Mac)
    python scripts/train_models.py --only vae2

Data: binarised MNIST (pixel > 0.5 -> 1). Set MNIST_ROOT to a folder that
contains MNIST/raw/*-ubyte; nothing is downloaded.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "models"
MNIST_ROOT = os.environ.get("MNIST_ROOT", "/Users/nipun/git/psdv-teaching/datasets")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def load_mnist(binary=True):
    """binary=True: pixels thresholded at 0.5 (a Bernoulli observation, used by AE/VAE).
    binary=False: grey levels in [0, 1] (used by the image GAN)."""
    from torchvision.datasets import MNIST

    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = MNIST(MNIST_ROOT, train=train, download=False)
        x = ds.data.float() / 255.0
        out[split] = ((x > 0.5).float() if binary else x, ds.targets.clone())
    return out


def batches(x, y, batch_size, generator):
    order = torch.randperm(len(x), generator=generator)
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


# ----------------------------------------------------------------------------- AE / VAE
class Encoder(nn.Module):
    def __init__(self, latent, variational):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent) if variational else None

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), (self.logvar(h) if self.logvar is not None else None)


class Decoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784))

    def forward(self, z):
        return self.net(z)  # logits, one per pixel


class AutoEncoder(nn.Module):
    """variational=False: a plain autoencoder. variational=True: a VAE."""

    def __init__(self, latent=2, variational=True):
        super().__init__()
        self.latent = latent
        self.variational = variational
        self.encoder = Encoder(latent, variational)
        self.decoder = Decoder(latent)

    def forward(self, x, eps=None):
        mu, logvar = self.encoder(x)
        if not self.variational:
            return self.decoder(mu), mu, None, mu
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mu + std * eps
        return self.decoder(z), mu, logvar, z


def vae_losses(model, x, beta=1.0):
    logits, mu, logvar, _ = model(x)
    recon = F.binary_cross_entropy_with_logits(logits, x.flatten(1), reduction="none").sum(1)
    if logvar is None:
        kl = torch.zeros_like(recon)
    else:
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(1)
    return recon, kl, (recon + beta * kl).mean()


def train_autoencoder(name, data, latent, variational, beta=1.0, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = AutoEncoder(latent, variational).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, ytr = data["train"]
    xte, _ = data["test"]
    xte = xte.to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(3)
        count = 0
        for xb, _ in batches(xtr, ytr, 128, gen):
            xb = xb.to(DEVICE)
            recon, kl, loss = vae_losses(model, xb, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = len(xb)
            sums += np.array([recon.sum().item(), kl.sum().item(), loss.item() * n])
            count += n
        model.eval()
        with torch.no_grad():
            r, k, _ = vae_losses(model, xte[:5000], beta)
        row = dict(epoch=epoch, train_recon=sums[0] / count, train_kl=sums[1] / count,
                   test_recon=r.mean().item(), test_kl=k.mean().item())
        history.append(row)
        print(f"{name} epoch {epoch:2d}  recon {row['train_recon']:7.2f}  kl {row['train_kl']:6.2f}  "
              f"test recon {row['test_recon']:7.2f}  test kl {row['test_kl']:6.2f}", flush=True)
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    return dict(name=name, latent=latent, variational=variational, beta=beta, epochs=epochs,
                seed=seed, history=history)


# ----------------------------------------------------------------------------- autoencoder variants
def salt_pepper(x, rate, generator=None):
    """Flip each pixel with probability `rate` (the corruption a denoising autoencoder learns to undo)."""
    flip = (torch.rand(x.shape, generator=generator) < rate).to(x.device)
    return torch.where(flip, 1 - x, x)


def train_denoising(data, latent=16, rate=0.2, epochs=20, seed=0):
    torch.manual_seed(seed)
    model = AutoEncoder(latent, variational=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, ytr = data["train"]
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        tot, count = 0.0, 0
        for xb, _ in batches(xtr, ytr, 128, gen):
            noisy = salt_pepper(xb, rate, gen).to(DEVICE)
            logits, _, _, _ = model(noisy)
            recon = F.binary_cross_entropy_with_logits(logits, xb.flatten(1).to(DEVICE), reduction="none").sum(1)
            loss = recon.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += recon.sum().item()
            count += len(xb)
        history.append(dict(epoch=epoch, train_recon=tot / count))
        print(f"dae epoch {epoch:2d}  recon of the clean image {tot / count:7.2f}", flush=True)
    torch.save(model.state_dict(), OUT / "dae16.pt")
    return dict(name="dae16", latent=latent, variational=False, rate=rate, epochs=epochs, history=history)


class SparseAE(nn.Module):
    """64 non-negative code units; an L1 penalty pushes most of them to exactly zero for any one image."""

    def __init__(self, units=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, units), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(units, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784))

    def forward(self, x):
        h = self.enc(x)
        return self.dec(h), h


def train_sparse(data, lam, name, epochs=20, seed=0):
    torch.manual_seed(seed)
    model = SparseAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, ytr = data["train"]
    xte, _ = data["test"]
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        tot, count = 0.0, 0
        for xb, _ in batches(xtr, ytr, 128, gen):
            xb = xb.to(DEVICE)
            logits, h = model(xb)
            recon = F.binary_cross_entropy_with_logits(logits, xb.flatten(1), reduction="none").sum(1)
            loss = (recon + lam * h.abs().sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += recon.sum().item()
            count += len(xb)
        with torch.no_grad():
            _, h = model(xte[:2000].to(DEVICE))
            active = (h > 1e-3).float().sum(1).mean().item()
        history.append(dict(epoch=epoch, train_recon=tot / count, active_units=active))
        print(f"{name} epoch {epoch:2d}  recon {tot / count:7.2f}  active units per image {active:5.1f} / 64", flush=True)
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    return dict(name=name, lam=lam, epochs=epochs, history=history)


# ----------------------------------------------------------------------------- conditional VAE
class CondVAE(nn.Module):
    """A VAE told which digit to draw: the one-hot label c goes into both encoder and decoder."""

    def __init__(self, latent=2, n_classes=10):
        super().__init__()
        self.latent, self.n_classes = latent, n_classes
        self.enc = nn.Sequential(nn.Linear(784 + n_classes, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        self.dec = nn.Sequential(nn.Linear(latent + n_classes, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784))

    def decode(self, z, y):
        return self.dec(torch.cat([z, F.one_hot(y, self.n_classes).float()], 1))

    def forward(self, x, y):
        h = self.enc(torch.cat([x.flatten(1), F.one_hot(y, self.n_classes).float()], 1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, y), mu, logvar


def train_cvae(data, latent=2, epochs=25, seed=0):
    torch.manual_seed(seed)
    model = CondVAE(latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, ytr = data["train"]
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        sums, count = np.zeros(2), 0
        for xb, yb in batches(xtr, ytr, 128, gen):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, mu, logvar = model(xb, yb)
            recon = F.binary_cross_entropy_with_logits(logits, xb.flatten(1), reduction="none").sum(1)
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(1)
            loss = (recon + kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += np.array([recon.sum().item(), kl.sum().item()])
            count += len(xb)
        history.append(dict(epoch=epoch, train_recon=sums[0] / count, train_kl=sums[1] / count))
        print(f"cvae epoch {epoch:2d}  recon {sums[0] / count:7.2f}  kl {sums[1] / count:6.2f}", flush=True)
    torch.save(model.state_dict(), OUT / "cvae2.pt")
    return dict(name="cvae2", latent=latent, epochs=epochs, seed=seed, history=history)


# ----------------------------------------------------------------------------- route A: pixel by pixel
class MaskedConv(nn.Conv2d):
    """Each output pixel sees only pixels already generated (above it, or to its left in its row)."""

    def __init__(self, cin, cout, k, context):
        super().__init__(cin, cout, k, padding=k // 2)
        m = torch.zeros(k, k)
        c = k // 2
        if context == "row3":          # the previous 3 pixels in the same row, like the name model's 3 characters
            m[c, c - 3:c] = 1
        else:                          # every earlier pixel in a k x k window: rows above, plus the left of this row
            m[:c, :] = 1
            m[c, :c] = 1
        self.register_buffer("mask", m[None, None].expand_as(self.weight).clone())

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)


class PixelAR(nn.Module):
    """p(x) = prod_i p(x_i | earlier pixels). Row and column are given as two extra (unmasked) channels."""

    def __init__(self, context="window7", k=7, hidden=64):
        super().__init__()
        self.context = context
        self.pix = MaskedConv(1, hidden, k, context)
        self.pos = nn.Conv2d(2, hidden, 1)
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        lin = torch.linspace(-1, 1, 28)
        self.register_buffer("coords", torch.stack(torch.meshgrid(lin, lin, indexing="ij"))[None])

    def forward(self, x):  # x: [B, 1, 28, 28] in {0, 1}; returns one logit per pixel
        return self.head(self.pix(x) + self.pos(self.coords.expand(len(x), -1, -1, -1)))

    @torch.no_grad()
    def sample(self, n, generator=None):
        x = torch.zeros(n, 1, 28, 28, device=self.coords.device)
        for r in range(28):
            for c in range(28):
                p = torch.sigmoid(self(x)[:, 0, r, c])
                u = torch.rand(n, generator=generator).to(x.device)
                x[:, 0, r, c] = (u < p.cpu().to(x.device)).float()
        return x


def train_pixel_ar(data, context, epochs=8, seed=0):
    torch.manual_seed(seed)
    model = PixelAR(context).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xtr, ytr = data["train"]
    xte, _ = data["test"]
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        tot, count = 0.0, 0
        for xb, _ in batches(xtr, ytr, 128, gen):
            xb = xb.unsqueeze(1).to(DEVICE)
            nll = F.binary_cross_entropy_with_logits(model(xb), xb, reduction="none").sum((1, 2, 3))
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += nll.sum().item()
            count += len(xb)
        with torch.no_grad():
            xt = xte[:5000].unsqueeze(1).to(DEVICE)
            test = F.binary_cross_entropy_with_logits(model(xt), xt, reduction="none").sum((1, 2, 3)).mean().item()
        history.append(dict(epoch=epoch, train_nll=tot / count, test_nll=test))
        print(f"pixelAR {context} epoch {epoch}  train nll {tot / count:7.2f}  test nll {test:7.2f}", flush=True)
    torch.save(model.state_dict(), OUT / f"pixelar_{context}.pt")
    return dict(name=f"pixelar_{context}", context=context, epochs=epochs, history=history)


# ----------------------------------------------------------------------------- MNIST GAN
class ConvG(nn.Module):
    """z [B, 100] -> image [B, 1, 28, 28] with pixels in [-1, 1]."""

    def __init__(self, zdim=100):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(zdim, 256 * 7 * 7), nn.BatchNorm1d(256 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),    # 14 -> 28
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 7, 7))


class ConvD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),     # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),   # 14 -> 7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),                           # one raw logit
        )

    def forward(self, x):
        return self.net(x)


def gan_step(G, D, opt_G, opt_D, real, zdim):
    """The exact two-phase update taught on the slides."""
    b = real.shape[0]
    # D phase: real -> 1, fake -> 0, fake detached so G receives nothing.
    z = torch.randn(b, zdim, device=real.device)
    fake = G(z)
    real_logits = D(real)
    fake_logits = D(fake.detach())
    loss_D = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
              + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()
    # G phase: fresh fakes, D's parameters frozen, gradient flows through D's input.
    D.requires_grad_(False)
    z = torch.randn(b, zdim, device=real.device)
    fake_logits = D(G(z))
    loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()
    D.requires_grad_(True)
    return loss_D.item(), loss_G.item(), torch.sigmoid(real_logits).mean().item(), torch.sigmoid(fake_logits).mean().item()


def train_mnist_gan(data, epochs=30, seed=1, zdim=100):
    """Grey-level MNIST scaled to [-1, 1]; same gan_step as the 1-D toy and the slides."""
    torch.manual_seed(seed)
    G, D = ConvG(zdim).to(DEVICE), ConvD().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    xtr, ytr = data["train"]
    fixed_z = torch.randn(64, zdim, generator=torch.Generator().manual_seed(123)).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    snapshots = {}

    def snap(epoch):
        G.eval()
        with torch.no_grad():
            snapshots[epoch] = (G(fixed_z).cpu().numpy()[:, 0] + 1) / 2
        G.train()

    snap(0)
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        rows = []
        for xb, _ in batches(xtr, ytr, 128, gen):
            real = (xb.unsqueeze(1) * 2 - 1).to(DEVICE)
            rows.append(gan_step(G, D, opt_G, opt_D, real, zdim))
            step += 1
            if epoch == 1 and step in (50, 150, 300):
                snap(f"step{step}")
        m = np.mean(rows, axis=0)
        history.append(dict(epoch=epoch, loss_D=m[0], loss_G=m[1], D_real=m[2], D_fake=m[3]))
        print(f"GAN epoch {epoch:2d}  D {m[0]:.3f}  G {m[1]:.3f}  D(real) {m[2]:.2f}  D(fake) {m[3]:.2f}", flush=True)
        if epoch in (1, 2, 3, 5, 10, 20, epochs):
            snap(epoch)
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, OUT / "gan_mnist.pt")
    np.savez_compressed(OUT / "gan_mnist_snapshots.npz", **{str(k): v for k, v in snapshots.items()})
    return dict(name="gan_mnist", epochs=epochs, seed=seed, zdim=zdim, history=history)


# ----------------------------------------------------------------------------- 1-D toy GAN
def toy_data(n, generator):
    """The lecture's running 1-D data: two equal bumps at -2 and +2, std 0.5."""
    side = torch.where(torch.rand(n, generator=generator) < 0.5, -2.0, 2.0)
    return side + 0.5 * torch.randn(n, generator=generator)


def train_toy_gan(seed, steps=3000, lr=1e-3, lr_g=None, hidden=64):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    D = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g or lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(1000 + seed)
    fixed_z = torch.randn(2000, 1, generator=torch.Generator().manual_seed(7))
    grid = torch.linspace(-5, 5, 401).unsqueeze(1)
    keep = {0, 25, 100, 250, 500, 1000, 2000, steps}
    record = {}

    def snap(step):
        with torch.no_grad():
            record[step] = dict(samples=G(fixed_z)[:, 0].numpy().tolist(),
                                d=torch.sigmoid(D(grid))[:, 0].numpy().tolist())

    snap(0)
    losses = []
    for step in range(1, steps + 1):
        real = toy_data(128, gen).unsqueeze(1)
        losses.append(gan_step(G, D, opt_G, opt_D, real, 1)[:2])
        if step in keep:
            snap(step)
    final = np.array(record[steps]["samples"])
    left, right = np.mean(np.abs(final + 2) < 1.0), np.mean(np.abs(final - 2) < 1.0)
    return dict(seed=seed, steps=steps, lr=lr, left=float(left), right=float(right),
                grid=grid[:, 0].numpy().tolist(), record=record, losses=np.array(losses).tolist())


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "training.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t0 = time.time()
    need_data = a.only in ("all", "ae2", "ae16", "vae2", "vae2b8", "vae16", "gan", "cvae2", "pixelar", "variants")
    data = load_mnist() if need_data else None
    jobs = {
        "ae2": lambda: train_autoencoder("ae2", data, 2, False, epochs=30),
        "ae16": lambda: train_autoencoder("ae16", data, 16, False, epochs=30),
        "vae2": lambda: train_autoencoder("vae2", data, 2, True, beta=1.0, epochs=40),
        "vae2b8": lambda: train_autoencoder("vae2b8", data, 2, True, beta=8.0, epochs=40),
        "vae16": lambda: train_autoencoder("vae16", data, 16, True, beta=1.0, epochs=40),
        "gan": lambda: train_mnist_gan(load_mnist(binary=False), epochs=30),
        "cvae2": lambda: train_cvae(data),
        "variants": lambda: [train_denoising(data), train_sparse(data, 0.0, "ae64"), train_sparse(data, 0.5, "sparse64")],
        "pixelar": lambda: [train_pixel_ar(data, "row3"), train_pixel_ar(data, "window7")],
    }
    for name, job in jobs.items():
        if a.only in ("all", name):
            meta[name] = job()
            meta_path.write_text(json.dumps(meta, indent=1))
    if a.only in ("all", "toygan"):
        runs = [train_toy_gan(seed) for seed in range(8)]
        # Same data and training loop, but a generator with only 4 hidden units per layer.
        # In our runs this small generator settles on a single bump (mode collapse).
        collapse = train_toy_gan(0, hidden=4)
        collapse["variant"] = "generator hidden width 4"
        for r in runs + [collapse]:
            print(f"toy GAN seed {r['seed']} {r.get('variant', '')}: fraction near -2 = {r['left']:.2f}, "
                  f"near +2 = {r['right']:.2f}", flush=True)
        (OUT / "toy_gan_runs.json").write_text(json.dumps(runs))
        (OUT / "toy_gan_collapse.json").write_text(json.dumps(collapse))
        meta["toygan"] = [dict(seed=r["seed"], left=r["left"], right=r["right"], steps=r["steps"], lr=r["lr"]) for r in runs]
        meta["toygan_collapse"] = dict(seed=0, hidden=4, left=collapse["left"], right=collapse["right"])
        meta_path.write_text(json.dumps(meta, indent=1))
    meta["device"] = DEVICE
    meta["torch"] = torch.__version__
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
