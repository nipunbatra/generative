"""Train every network whose output appears in Lecture 2.

    python lecture2/scripts/train.py                 # everything, about 20 minutes on an M-series Mac
    python lecture2/scripts/train.py ae2 vae2        # just these
    python lecture2/scripts/train.py --list

Data: grey-level MNIST in [0, 1], read from $MNIST_ROOT (a folder containing MNIST/raw/*-ubyte);
nothing is downloaded. Checkpoints and per-epoch histories go to lecture2/.cache/ (git-ignored);
scripts/science.py turns them into the lecture's figures and numbers, and copies the few
checkpoints the notebook loads into lecture2/models/.

Every network has one hidden layer of 400 ReLU units on each side, as in Kingma and Welling (2014):
    encoder 784 -> 400 -> z   (a VAE's encoder outputs 2z numbers: a mean and a log variance per code)
    decoder z -> 400 -> 784, then a sigmoid, so every output pixel is in (0, 1)
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('CACHE', ROOT / '.cache'))
MNIST_ROOT = os.environ.get('MNIST_ROOT', '/Users/nipun/git/psdv-teaching/datasets')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
HIDDEN = 400
SIGMA = float(os.environ.get('SIGMA', 0.2))  # the decoder's pixel noise: p(x | z) = N(x; decoder(z), SIGMA^2 I)
WATCH = [3, 2, 1, 18, 4, 8, 11, 0]   # test digits whose reconstructions are recorded while training


def load_mnist():
    from torchvision.datasets import MNIST
    out = {}
    for split, train in (('train', True), ('test', False)):
        ds = MNIST(MNIST_ROOT, train=train, download=False)
        out[split] = (ds.data.reshape(-1, 784).float() / 255.0, ds.targets.clone())
    return out


class AutoEncoder(nn.Module):
    """variational=False: a plain autoencoder. True: a VAE (the encoder outputs mu and log sigma^2).
    linear=True: no hidden layer and no nonlinearity anywhere (the autoencoder that rediscovers PCA)."""

    def __init__(self, z, variational=False, linear=False):
        super().__init__()
        self.z, self.variational, self.linear = z, variational, linear
        out = 2 * z if variational else z
        if linear:
            self.encoder = nn.Linear(784, z)
            self.decoder = nn.Linear(z, 784)
        else:
            self.encoder = nn.Sequential(nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, out))
            self.decoder = nn.Sequential(nn.Linear(z, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        if self.variational:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar
        return h, None

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation
        else:
            z = mu
        return self.decoder(z), mu, logvar


def losses(model, x, target, beta):
    """Per-batch means of: squared error summed over the 784 pixels, and (VAE) KL(q(z|x) || N(0, I))."""
    x_hat, mu, logvar = model(x)
    sse = ((x_hat - target) ** 2).sum(1)
    if not model.variational:
        return sse.mean(), sse.mean(), torch.zeros(())
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)
    recon = sse / (2 * SIGMA ** 2)
    return (recon + beta * kl).mean(), sse.mean(), kl.mean()


CONFIGS = {}
for z in (2, 4, 8, 16, 32, 64, 128):
    CONFIGS[f'ae{z}'] = dict(z=z, epochs=30)
CONFIGS['lin2'] = dict(z=2, linear=True, epochs=30)
CONFIGS['dae16'] = dict(z=16, epochs=30, mask=0.5)
for z in (2, 16):
    CONFIGS[f'vae{z}'] = dict(z=z, variational=True, beta=1.0, epochs=50)
for beta in (0.0, 0.1, 4.0, 16.0, 64.0, 256.0):
    CONFIGS[f'vae2-b{beta:g}'] = dict(z=2, variational=True, beta=beta, epochs=50)


def train(name, data):
    cfg = CONFIGS[name]
    seed = 1000 + sorted(CONFIGS).index(name)
    torch.manual_seed(seed)
    model = AutoEncoder(cfg['z'], cfg.get('variational', False), cfg.get('linear', False)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, _ = data['train']
    xte, _ = data['test']
    xtr, xte = xtr.to(DEVICE), xte.to(DEVICE)
    watch = xte[WATCH]
    beta, mask = cfg.get('beta', 1.0), cfg.get('mask', 0.0)
    gen = torch.Generator().manual_seed(seed)
    hist = dict(name=name, config=cfg, seed=seed, sigma=SIGMA, hidden=HIDDEN, epoch=[], train_loss=[], train_sse=[],
                train_kl=[], test_sse=[], test_kl=[], snapshots={}, step_snapshots={})

    def snapshot():
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(watch)
            out = model.decoder(mu)
        model.train()
        return [[round(float(v), 3) for v in row] for row in out.cpu()]

    def evaluate():
        model.eval()
        with torch.no_grad():
            sse, kl = [], []
            for i in range(0, len(xte), 2000):
                xb = xte[i:i + 2000]
                mu, logvar = model.encode(xb)
                sse.append(((model.decoder(mu) - xb) ** 2).sum(1))
                if model.variational:
                    kl.append(0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1))
        model.train()
        return float(torch.cat(sse).mean()), (float(torch.cat(kl).mean()) if kl else 0.0)

    hist['snapshots']['0'] = snapshot()
    hist['test_sse_init'] = evaluate()[0]
    t0, step = time.time(), 0
    epochs = int(os.environ.get('EPOCHS', cfg['epochs']))
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(xtr), generator=gen).to(DEVICE)
        tot = [0.0, 0.0, 0.0]
        nb = 0
        for i in range(0, len(xtr), 128):
            xb = xtr[order[i:i + 128]]
            inp = xb * (torch.rand_like(xb) > mask).float() if mask else xb
            loss, sse, kl = losses(model, inp, xb, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            tot[0] += float(loss); tot[1] += float(sse); tot[2] += float(kl); nb += 1
            if epoch == 1 and step in (10, 30, 100, 250):
                hist['step_snapshots'][str(step)] = snapshot()
        te_sse, te_kl = evaluate()
        hist['epoch'].append(epoch)
        hist['train_loss'].append(tot[0] / nb); hist['train_sse'].append(tot[1] / nb); hist['train_kl'].append(tot[2] / nb)
        hist['test_sse'].append(te_sse); hist['test_kl'].append(te_kl)
        if epoch in (1, 2, 3, 5, 10, 20, 30, 50):
            hist['snapshots'][str(epoch)] = snapshot()
        print(f'{name} epoch {epoch:3d} loss {tot[0] / nb:9.3f} sse {tot[1] / nb:8.3f} kl {tot[2] / nb:6.3f} '
              f'test sse {te_sse:8.3f} ({time.time() - t0:5.1f}s)', flush=True)
    hist['seconds'] = time.time() - t0
    hist['steps_per_epoch'] = nb
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({'config': cfg, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'seed': seed,
                'sigma': SIGMA, 'hidden': HIDDEN}, CACHE / f'{name}.pt')
    json.dump(hist, open(CACHE / f'{name}.json', 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        print('\n'.join(f'{k}: {v}' for k, v in CONFIGS.items()))
        raise SystemExit
    data = load_mnist()
    for name in a.names or list(CONFIGS):
        train(name, data)
