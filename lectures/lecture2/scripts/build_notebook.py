"""Build and execute the Lecture 2 companion notebook.

    MNIST_ROOT=/path/to/datasets python lecture2/scripts/build_notebook.py

The notebook follows the deck's section numbers (1.1 ... 6.8); each section's first cell has the id
sec-X-Y, which the slides link to (data/notebook-anchors.json lists them). It embeds the Old Faithful
data, loads the trained checkpoints from lecture2/models (or downloads them in Colab), and retrains the
small extra models it needs, so it runs top to bottom in a few minutes.
"""
import csv
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
NAME = 'lecture2-generative-models.ipynb'
DECK = 'https://nipunbatra.github.io/generative/lectures/lecture2/'
COLAB = f'https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture2/{NAME}'
RAW = 'https://raw.githubusercontent.com/nipunbatra/generative/main/lectures/lecture2'
cells, anchors = [], []


def add(cell, anchor):
    cell_id = anchor or f'c{len(cells):03d}'
    cell['id'] = cell_id
    cell['metadata']['id'] = cell_id
    if anchor:
        anchors.append(anchor)
    cells.append(cell)


def md(s, anchor=None):
    add(nbf.v4.new_markdown_cell(s.strip().replace('{DECK}', DECK).replace('{COLAB}', COLAB)), anchor)


def code(s, anchor=None):
    add(nbf.v4.new_code_cell(s.strip().replace('{RAW}', RAW)), anchor)


def section(k, title, text=''):
    slides = f'[slides]({DECK})'
    md(f'## {k} {title}\n\n{text}\n\n*Slides: section {k} of the {slides}.*', anchor=f'sec-{k.replace(".", "-")}')


rows = list(csv.DictReader(open(ROOT / 'data' / 'faithful.csv')))
WAIT = [int(float(r['waiting'])) for r in rows]
DUR = [float(r['eruptions']) for r in rows]

# ------------------------------------------------------------------------------------------------ title
md(r"""
# Lecture 2 · From fitting a distribution to the VAE

[▶ Open in Colab]({COLAB}) · [Slides]({DECK}) · ES 667 Deep Learning, IIT Gandhinagar

This notebook runs the experiments behind the Lecture 2 slides, **in the same order and with the same section numbers** (1.1 … 6.8). Every slide's **▶ Notebook** link jumps to its section here.

Run everything once from the top (*Runtime → Run all*). It needs PyTorch, torchvision, NumPy and matplotlib. In Colab it downloads MNIST (about 12 MB) and the three trained networks from the course repository; from a clone of the repository it uses the local copies. The slides' networks were trained by `lecture2/scripts/train.py` (30 epochs for autoencoders, 50 for VAEs); the extra networks trained here use fewer epochs so the notebook stays quick, and their numbers differ slightly from the slides.
""", anchor='top')
code(r"""
import math, os, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
F64 = torch.float64
plt.rcParams.update({'figure.dpi': 90, 'axes.spines.top': False, 'axes.spines.right': False})
print('device:', DEVICE)

def show_digits(x, ncol=8, size=0.9, title=None):
    # Show a batch of 784-pixel images as ink on paper.
    x = torch.as_tensor(x).detach().float().cpu().reshape(-1, 28, 28).clamp(0, 1)
    n = len(x); nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n: ax.imshow(1 - x[i], cmap='gray', vmin=0, vmax=1)
    if title: fig.suptitle(title, fontsize=10)
    plt.show()
""")

# ------------------------------------------------------------------------------------------------ Part 1
md('# Part 1 · Why generate?')
section('1.1', 'Things that do not exist', 'The faces and the real-or-generated digits are on the slides. The fake digits come from the 16-number VAE trained in Part 6.')
section('1.2', 'Text, pictures and sound', 'A language model writes by sampling one word at a time. These are the next-word probabilities on the slide, from SmolLM2-360M; sampling from them gives a different word each time.')
code(r"""
words = ['wet', 'sweat', 'damp', 'earth', 'the', 'salt', 'ozone', 'mud', '(any other word)']
probs = torch.tensor([0.1735, 0.0560, 0.0481, 0.0402, 0.0338, 0.0337, 0.0328, 0.0305, 0.5513])
g = torch.Generator().manual_seed(1)
picks = torch.multinomial(probs, 10, replacement=True, generator=g)
print('ten samples of the next word:', [words[i] for i in picks])
""")
section('1.3', 'Beyond pictures', 'Weather ensembles, protein design, materials and filling in missing sensor readings: see the slide for references.')
section('1.4', 'What they all share', 'Why not output the average? The average of many 3s is a blur nobody writes. (MNIST is loaded in section 3.1; this cell runs after it if you jump here first.)')

# ------------------------------------------------------------------------------------------------ Part 2
md('# Part 2 · Fit a distribution, then sample')
section('2.1', 'The data: Old Faithful', '272 eruptions of the Old Faithful geyser: how long each eruption lasted, and the wait until the next one (minutes). Azzalini and Bowman (1990), as distributed with R.')
code(f"""
wait = torch.tensor({WAIT}, dtype=F64)
duration = torch.tensor({DUR}, dtype=F64)
print(len(wait), 'eruptions; first five waits:', wait[:5].tolist())
print('waits between 66 and 76 minutes:', int(((wait >= 66) & (wait <= 76)).sum()))
plt.figure(figsize=(6, 2.6)); plt.hist(wait.numpy(), bins=range(40, 101, 2), color='#245EDB', alpha=.6)
plt.xlabel('minutes until the next eruption'); plt.show()
""")
section('2.2', 'One Gaussian: fit, score, sample', 'Maximum likelihood for one Gaussian has a formula: the average and the average squared deviation.')
code(r"""
mu_hat = wait.mean()
sd_hat = ((wait - mu_hat) ** 2).mean().sqrt()          # divide by n: the maximum-likelihood estimate
one_bell = Normal(mu_hat, sd_hat)
print(f'mu = {mu_hat:.2f}, sigma = {sd_hat:.2f}')
for x in (54., 71., 80.):
    print(f'density at {x:.0f} minutes: {one_bell.log_prob(torch.tensor(x, dtype=F64)).exp():.4f}')
bell_ll = float(one_bell.log_prob(wait).mean())
print(f'average log-likelihood: {bell_ll:.3f}')
""")
code(r"""
torch.manual_seed(7)
draws = one_bell.sample((272,))
print('draws between 62 and 72 minutes:', int(((draws >= 62) & (draws <= 72)).sum()), ' data:', int(((wait >= 62) & (wait <= 72)).sum()))
plt.figure(figsize=(6, 2.6)); bins = range(40, 101, 2)
plt.hist(wait.numpy(), bins=bins, color='#9AA5B1', alpha=.6, label='real waits')
plt.hist(draws.numpy(), bins=bins, color='#8B2CDE', alpha=.45, label='draws from the bell'); plt.legend(); plt.show()
""")
section('2.3', 'Two bumps: a hidden choice', 'Plot eruption length against the next wait: two kinds of eruption.')
code(r"""
plt.figure(figsize=(4.5, 3.4)); plt.scatter(duration, wait, s=10, color='#245EDB', alpha=.6)
plt.xlabel('eruption length (minutes)'); plt.ylabel('next wait (minutes)'); plt.show()
""")
section('2.4', 'Fit the mixture, then sample it', 'The same gradient-descent loop as in Lecture 1, in minutes, one line per term of the loss. Centres are learned directly; widths through sigma = exp(log_s), so they stay positive; shares through softmax, so they are positive and add to 1.')
code(r"""
def bell(x, m, s):                    # N(x; m, s^2), the formula from 2.2
    return torch.exp(-((x - m) / s)**2 / 2) / (s * math.sqrt(2 * math.pi))

x = wait                                                  # all 272 waits, in minutes
mu = torch.tensor([65., 75.], dtype=F64, requires_grad=True)   # centres
log_s = torch.tensor([2.3, 2.3], dtype=F64, requires_grad=True) # widths: sigma = exp(log_s), about 10
w = torch.zeros(2, dtype=F64, requires_grad=True)               # shares: pi = softmax(w) = (1/2, 1/2)
opt = torch.optim.Adam([mu, log_s, w], lr=0.1)
for step in range(300):
    pi, sigma = torch.softmax(w, 0), torch.exp(log_s)
    p_short = pi[0] * bell(x, mu[0], sigma[0])    # pi_1 N(x; mu_1, sigma_1^2), one number per wait
    p_long  = pi[1] * bell(x, mu[1], sigma[1])    # pi_2 N(x; mu_2, sigma_2^2)
    loss = -torch.log(p_short + p_long).mean()    # minus the average log p(x)
    opt.zero_grad(); loss.backward(); opt.step()
pi, mu_min, sd_min = torch.softmax(w, 0).detach(), mu.detach(), torch.exp(log_s).detach()
print('shares', pi.numpy().round(3), ' centres', mu_min.numpy().round(2), ' widths', sd_min.numpy().round(2))
print(f'average log-likelihood: {-loss.item():.3f}  (one Gaussian: {bell_ll:.3f})')
""")
code(r"""
# Which bump does each wait belong to? Compare with the eruption lengths the fit never saw.
logp = pi.log() + Normal(mu_min, sd_min).log_prob(wait[:, None])
long_bump = (logp[:, 1] > logp[:, 0])
print(f'mean eruption length, short bump: {duration[~long_bump].mean():.2f} min;  long bump: {duration[long_bump].mean():.2f} min')
# Sample: pick a bump, then draw a time from it.
torch.manual_seed(41)
k = torch.multinomial(pi, 272, replacement=True)
new = Normal(mu_min[k], sd_min[k]).sample()
plt.figure(figsize=(6, 2.6)); bins = range(40, 101, 2)
plt.hist(wait.numpy(), bins=bins, color='#9AA5B1', alpha=.6, label='real waits')
plt.hist(new.numpy(), bins=bins, color='#0F766E', alpha=.45, label='draws from the mixture'); plt.legend(); plt.show()
""")
section('2.5', 'How many bumps?', 'Expectation maximisation (EM) fits a mixture faster than gradient descent; we use it here and in Part 3. Fit on 200 waits, score on the other 72.')
code(r"""
def em(Z, K, iters=200, seed=0, reg=1e-4):
    # Maximum-likelihood mixture of K full-covariance Gaussians by EM. Z: N x d tensor (float64).
    N, d = Z.shape
    g = torch.Generator().manual_seed(seed)
    C0 = torch.cov(Z.T).reshape(d, d)
    eye = torch.eye(d, dtype=F64) * reg * float(torch.trace(C0)) / d
    means = [Z[torch.randint(N, (1,), generator=g)].squeeze(0)]
    for _ in range(1, K):                                   # spread the starting means out
        dist = torch.stack([((Z - m) ** 2).sum(1) for m in means]).min(0).values
        means.append(Z[torch.multinomial(dist / dist.sum(), 1, generator=g)].squeeze(0))
    means = torch.stack(means); covs = C0.expand(K, d, d).clone() + eye
    logw = torch.full((K,), -math.log(K), dtype=F64)
    for it in range(iters):
        logp = torch.stack([logw[k] + torch.distributions.MultivariateNormal(means[k], covs[k]).log_prob(Z) for k in range(K)])
        tot = torch.logsumexp(logp, 0); R = torch.exp(logp - tot)         # E step: responsibilities
        Nk = R.sum(1) + 1e-10; logw = torch.log(Nk / N); means = (R @ Z) / Nk[:, None]   # M step
        for k in range(K):
            D = Z - means[k]; covs[k] = (R[k][:, None] * D).T @ D / Nk[k] + eye
    return logw.exp(), means, covs, float(tot.mean())

def mog_logpdf(Z, w, means, covs):
    return torch.logsumexp(torch.stack([w[k].log() + torch.distributions.MultivariateNormal(means[k], covs[k]).log_prob(Z) for k in range(len(w))]), 0)

perm = torch.randperm(272, generator=torch.Generator().manual_seed(11))
tr, te = wait[perm[:200]][:, None], wait[perm[200:]][:, None]
for K in range(1, 7):
    best = max((em(tr, K, iters=300, seed=s, reg=1e-3) for s in range(8)), key=lambda r: r[3])
    print(f'K = {K}: train {best[3]:.3f}   held out {mog_logpdf(te, *best[:3]).mean():.3f}')
""")
# ------------------------------------------------------------------------------------------------ Part 3
md('# Part 3 · Images: PCA and Gaussians')
section('3.1', 'An image is 784 numbers', 'MNIST, grey levels scaled to [0, 1]. Our running example is test digit #90, a 3.')
code(r"""
from torchvision.datasets import MNIST
MNIST_ROOT = os.environ.get('MNIST_ROOT', 'data')
have = (Path(MNIST_ROOT) / 'MNIST' / 'raw' / 'train-images-idx3-ubyte').exists()
train_set = MNIST(MNIST_ROOT, train=True, download=not have)
test_set = MNIST(MNIST_ROOT, train=False, download=not have)
Xtr = train_set.data.reshape(-1, 784).double() / 255; ytr = train_set.targets
Xte = test_set.data.reshape(-1, 784).double() / 255; yte = test_set.targets
x3 = Xte[90]
print(Xtr.shape, Xte.shape, 'test digit #90 is a', int(yte[90]))
print('a patch of its pixels (rows 6-15, columns 8-18):'); print(x3.reshape(28, 28)[6:16, 8:19].numpy().round(1))
show_digits(Xtr[:16], title='the first 16 training digits')
""")
code(r"""
# (section 1.4) the average of 100 test 3s
threes = Xte[yte == 3][:100]
show_digits(torch.cat([threes[:5], threes.mean(0, keepdim=True)]), ncol=6, title='five 3s, and the average of 100')
show_digits(torch.rand(8, 784, generator=torch.Generator().manual_seed(1)), title='random points in [0, 1]^784')
""")
section('3.2', 'A bell for every pixel', 'One distribution for all 60,000 training digits (all ten digits, labels ignored): first one Gaussian per pixel, sampled independently, then (appendix on the slides) one 784-dimensional Gaussian with a full covariance.')
code(r"""
m = Xtr.mean(0); s = Xtr.std(0, correction=0)
g = torch.Generator().manual_seed(1)
show_digits((m + s * torch.randn(8, 784, generator=g, dtype=F64)).clamp(0, 1), title='independent pixel Gaussians: 1,568 numbers')
C = (Xtr - m).T @ (Xtr - m) / len(Xtr)
corr = C / torch.sqrt(torch.outer(C.diagonal(), C.diagonal()) + 1e-12)
c = 14 * 28 + 14
print('correlation of pixel (14,14) with its right neighbour:', round(float(corr[c, c + 1]), 3))
L = torch.linalg.cholesky(C + 1e-4 * torch.eye(784, dtype=F64))
show_digits((m + torch.randn(8, 784, generator=g, dtype=F64) @ L.T).clamp(0, 1), title='one full-covariance Gaussian: 308,504 numbers')
""")
section('3.3', 'A few numbers per digit', 'PCA is the simplest encoder and decoder: encode z = Wᵀ(x − m), decode x = m + W z. Its directions are the top eigenvectors of the covariance matrix.')
code(r"""
evals, evecs = torch.linalg.eigh(C)
order = torch.argsort(evals, descending=True)
evals, W = evals[order].clamp(min=0), evecs[:, order]
W = W * torch.sign((x3 - m) @ W)                        # cosmetic: the 3's coefficients come out positive
kept = torch.cumsum(evals, 0) / evals.sum()
print('share of the spread kept by d = 2, 10, 50, 100:', [round(float(kept[d - 1]), 3) for d in (2, 10, 50, 100)])
z3 = (x3 - m) @ W                                       # encode: all 784 PCA numbers of our 3
for d in (1, 2, 5, 10, 20, 50, 100):
    xh = m + z3[:d] @ W[:, :d].T                        # decode from the first d
    print(f'd = {d:3d}: squared error {((xh - x3) ** 2).sum():.2f}')
show_digits(torch.stack([(m + z3[:d] @ W[:, :d].T) for d in (1, 2, 5, 10, 20, 50, 100, 784)]), title='our 3 from its first 1, 2, 5, 10, 20, 50, 100, 784 PCA numbers')
""")
code(r"""
Z2 = (Xte[:2000] - m) @ W[:, :2]
plt.figure(figsize=(4.5, 4)); plt.scatter(*Z2.T, c=yte[:2000], cmap='tab10', s=5); plt.colorbar(label='digit'); plt.title('first two PCA numbers'); plt.show()
""")
section('3.4', 'Generate: sample a code, decode it', 'Four steps: encode the training digits, fit a distribution to their codes, draw a new code, decode it with x = m + W z.')
code(r"""
def pca_sample(d, K, n=16, seed=7, sub=30000):
    Z = (Xtr[:sub] - m) @ W[:, :d]
    w, means, covs, ll = em(Z, K, iters=80, seed=d * 100 + K)
    g = torch.Generator().manual_seed(seed)
    k = torch.multinomial(w, n, replacement=True, generator=g)
    z = means[k] + torch.einsum('nij,nj->ni', torch.linalg.cholesky(covs)[k], torch.randn(n, d, generator=g, dtype=F64))
    return (m + z @ W[:, :d].T).clamp(0, 1), ll

for d in (2, 10, 50):
    show_digits(pca_sample(d, 1)[0], ncol=16, size=.6, title=f'one Gaussian over {d} PCA numbers')
xs, ll = pca_sample(20, 10)
show_digits(xs, ncol=16, size=.6, title=f'10 Gaussians over 20 PCA numbers (average log-likelihood {ll:.1f})')
""")

# ------------------------------------------------------------------------------------------------ Part 4
md('# Part 4 · Autoencoders')
section('4.1', 'From PCA to an autoencoder', 'One hidden layer of 400 ReLU units on each side, as in Kingma and Welling (2014). The same class becomes a VAE in Part 6.')
code(r"""
class AutoEncoder(nn.Module):
    # variational=True: the encoder outputs a mean and a log variance per code number (Part 6).
    def __init__(self, z=2, variational=False):
        super().__init__()
        self.z, self.variational = z, variational
        self.encoder = nn.Sequential(nn.Linear(784, 400), nn.ReLU(), nn.Linear(400, 2 * z if variational else z))
        self.decoder = nn.Sequential(nn.Linear(z, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())
    def encode(self, x):
        h = self.encoder(x)
        return h.chunk(2, dim=1) if self.variational else (h, None)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.variational else mu
        return self.decoder(z), mu, logvar

SIGMA = 0.2                                              # the VAE's pixel noise (Part 6)
def loss_fn(model, x, target, beta=1.0):
    x_hat, mu, logvar = model(x)
    sse = ((x_hat - target) ** 2).sum(1)                 # squared error summed over the 784 pixels
    if not model.variational:
        return sse.mean()
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)
    return (sse / (2 * SIGMA ** 2) + beta * kl).mean()

def train(model, epochs, beta=1.0, mask=0.0, seed=0, lr=1e-3):
    torch.manual_seed(seed); model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = Xtr.float().to(DEVICE)
    for epoch in range(epochs):
        for idx in torch.randperm(len(X)).split(128):
            xb = X[idx]; inp = xb * (torch.rand_like(xb) > mask) if mask else xb
            loss = loss_fn(model, inp, xb, beta)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval().cpu()

@torch.no_grad()
def test_sse(model, X=None):
    X = (Xte if X is None else X).float()
    mu, _ = model.encode(X)
    return float(((model.decoder(mu) - X) ** 2).sum(1).mean())

def load_trained(name, z, variational):
    # The slides' networks: from lecture2/models in a clone, else downloaded from the repository.
    path = Path('models') / f'{name}.pt'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f'{RAW}/models/{name}.pt', path)
    net = AutoEncoder(z, variational)
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    if variational:
        sd = {k: v for k, v in sd.items()}
    net.load_state_dict(sd); return net.eval()
""")
section('4.2', 'Train one', 'Always look at the untrained network first. Then load the trained 2-number autoencoder (or set `RETRAIN = True` to train it here: 30 epochs, under a minute on a GPU).')
code(r"""
torch.manual_seed(0)
fresh = AutoEncoder(z=2).eval()
with torch.no_grad(): out = fresh(Xte[:8].float())[0]
print(f'untrained outputs lie between {out.min():.2f} and {out.max():.2f}; test squared error {test_sse(fresh):.1f} per image')
RETRAIN = False
ae2 = train(AutoEncoder(z=2), epochs=30) if RETRAIN else load_trained('ae2', 2, False)
print(f'trained: test squared error {test_sse(ae2):.2f}, training {test_sse(ae2, Xtr[:10000]):.2f} (first 10,000 training digits)')
with torch.no_grad(): rec = ae2.decoder(ae2.encode(Xte[[3, 2, 1, 18, 4, 8, 11, 0]].float())[0])
show_digits(torch.cat([Xte[[3, 2, 1, 18, 4, 8, 11, 0]], rec.double()]), title='8 test digits and their reconstructions from 2 numbers')
""")
section('4.3', 'How many numbers in the code?', 'Train autoencoders with wider codes (10 epochs each here; 30 on the slides) and compare with PCA at the same size.')
code(r"""
aes = {2: ae2}
for z in (8, 16, 128):
    aes[z] = train(AutoEncoder(z=z), epochs=10, seed=z)
for z, net in aes.items():
    Zt = (Xte - m) @ W[:, :z]
    pca = float((((m + Zt @ W[:, :z].T) - Xte) ** 2).sum(1).mean())
    print(f'code of {z:3d} numbers: autoencoder {test_sse(net):6.2f}   PCA {pca:6.2f}')
""")
section('4.4', 'Where the codes land', 'This autoencoder has exactly 2 numbers in its code, so we can plot every code.')
code(r"""
with torch.no_grad(): C2 = ae2.encode(Xte[:2000].float())[0]
plt.figure(figsize=(5, 4)); plt.scatter(*C2.T, c=yte[:2000], cmap='tab10', s=5); plt.colorbar(label='digit'); plt.title('2-number autoencoder codes of 2,000 test digits'); plt.show()
""")

# ------------------------------------------------------------------------------------------------ Part 5
md('# Part 5 · Can an autoencoder generate?')
section('5.1', 'Decode a random code', 'Feed draws from N(0, I) to the decoders.')
code(r"""
for z in (2, 8, 128):
    with torch.no_grad(): xs = aes[z].decoder(torch.randn(16, z, generator=torch.Generator().manual_seed(5)))
    show_digits(xs, ncol=16, size=.6, title=f'N(0, I) codes through the {z}-number autoencoder')
""")
section('5.2', 'Where the codes are', 'Overlay N(0, I) draws on the codes of real test digits; decode a grid; measure how far apart codes of the same digit are.')
code(r"""
with torch.no_grad(): C2 = ae2.encode(Xte[:2000].float())[0]
prior = torch.randn(500, 2, generator=torch.Generator().manual_seed(6))
plt.figure(figsize=(5, 4)); plt.scatter(*C2.T, c=yte[:2000], cmap='tab10', s=5); plt.scatter(*prior.T, s=4, c='red', label='N(0, I) draws')
plt.legend(); plt.title('autoencoder codes of 2,000 test digits'); plt.show()
lo, hi = C2.quantile(.01, 0), C2.quantile(.99, 0)
gx, gy = torch.linspace(lo[0], hi[0], 12), torch.linspace(hi[1], lo[1], 12)
grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1).reshape(-1, 2)
with torch.no_grad(): show_digits(ae2.decoder(grid), ncol=12, size=.5, title='decoder output over a 12 x 12 grid of codes')
def spread(net):
    with torch.no_grad(): z0, z1 = net.encode(Xte[yte == 0][:100].float())[0], net.encode(Xte[yte == 1][:100].float())[0]
    same = torch.pdist(z0).mean(); cross = torch.cdist(z0, z1).mean()
    return float(same), float(cross)
for z, net in aes.items():
    a, b = spread(net); print(f'{z:3d} numbers: two 0s {a:6.2f} apart, a 0 and a 1 {b:6.2f} apart, ratio {a / b:.2f}')
""")
section('5.3', 'What we want instead', 'Codes that fill N(0, I); each image owning a small cloud of codes; one probability model that can score and sample. The VAE does all three.')

# ------------------------------------------------------------------------------------------------ Part 6
md('# Part 6 · The variational autoencoder')
section('6.1', 'Two roads to the VAE', 'Road 1: an autoencoder whose encoder outputs a cloud N(μ, σ²) instead of a point, plus a KL penalty. Road 2: a generative model z ~ N(0, I), x | z ~ N(decoder(z), 0.2² I), trained with variational inference. Both give the same network and loss. Load the trained 2-number and 16-number VAEs (50 epochs).')
code(r"""
vae2 = load_trained('vae2', 2, True); vae16 = load_trained('vae16', 16, True)
const = 784 * math.log(SIGMA * math.sqrt(2 * math.pi))
@torch.no_grad()
def log_lik(net, x, Z):                     # log p(x | z) for a batch of codes Z
    return -((net.decoder(Z.float()).double() - x) ** 2).sum(1) / (2 * SIGMA ** 2) - const
print(f'the constant 784 log(sigma sqrt(2 pi)) = {const:.2f}')
""")
section('6.2', 'Road 1: from the autoencoder', 'The encoder ends in 2 x 2 = 4 numbers: a mean and a log variance per code number. The code is drawn by reparameterisation, and the loss adds a KL to N(0, I).')
code(r"""
with torch.no_grad():
    h = vae2.encoder(x3[None].float())
    mu_, logvar_ = h.chunk(2, dim=1)
print('encoder output for our 3 (4 numbers):', h.numpy().round(3))
print('mean', mu_.numpy().round(3), ' sigma', torch.exp(0.5 * logvar_).numpy().round(4))
eps = torch.randn(1, 2, generator=torch.Generator().manual_seed(1))
z = mu_ + torch.exp(0.5 * logvar_) * eps                        # reparameterisation
kl = 0.5 * (mu_ ** 2 + logvar_.exp() - logvar_ - 1).sum()
print('one draw z =', z.numpy().round(4), f'; KL to N(0, I) = {kl:.2f}')
""")
section('6.3', 'Road 2: from a generative model', 'The posterior of our 3 on a grid (possible with two numbers), naive Monte Carlo for p(x), variational inference for this one image, and the encoder against both.')
md('### The posterior and the evidence on a grid')
code(r"""
with torch.no_grad(): mu3, lv3 = vae2.encode(x3[None].float())
mu3, sd3 = mu3[0].double(), (0.5 * lv3[0].double()).exp()
half, n = 0.15, 241                                  # a fine window around the posterior
ax_x = torch.linspace(mu3[0] - half, mu3[0] + half, n, dtype=F64); ax_y = torch.linspace(mu3[1] - half, mu3[1] + half, n, dtype=F64)
Zg = torch.stack(torch.meshgrid(ax_x, ax_y, indexing='xy'), -1).reshape(-1, 2)
lj = log_lik(vae2, x3, Zg) - 0.5 * (Zg ** 2).sum(1) - math.log(2 * math.pi)    # log p(x | z) + log p(z)
cell = float((ax_x[1] - ax_x[0]) * (ax_y[1] - ax_y[0]))
log_px = float(torch.logsumexp(lj, 0)) + math.log(cell)
print(f'log p(x) by adding rectangles: {log_px:.2f}')
plt.figure(figsize=(4, 3.6)); plt.contourf(ax_x, ax_y, torch.softmax(lj, 0).reshape(n, n), levels=20, cmap='Greens'); plt.title('p(z | our 3), zoomed'); plt.show()
# Naive Monte Carlo: average p(x | z) over prior draws.
g = torch.Generator().manual_seed(22)
for S in (10, 100, 1000, 10000, 100000):
    z = torch.randn(S, 2, generator=g, dtype=F64)
    print(f'S = {S:6d}: estimate {float(torch.logsumexp(log_lik(vae2, x3, z), 0)) - math.log(S):8.2f}')
""")
md("### Variational inference for one image: Lecture 1's tool, fitting q(z) = N(m, diag s²) by maximising the ELBO")
code(r"""
def elbo(net, x, m_, s_, draws=20000, seed=21):
    eps = torch.randn(draws, 2, generator=torch.Generator().manual_seed(seed), dtype=F64)
    kl = 0.5 * float((m_ ** 2 + s_ ** 2 - 2 * s_.log() - 1).sum())
    return float(log_lik(net, x, m_ + s_ * eps).mean()) - kl

torch.manual_seed(23)
mq = torch.zeros(2, requires_grad=True); ls = torch.zeros(2, requires_grad=True)
opt = torch.optim.Adam([mq, ls], lr=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0 if t < 800 else 0.2)
x3f = x3.float()
for step in range(1200):
    zq = mq + ls.exp() * torch.randn(64, 2)                                 # reparameterised draws
    recon = -((vae2.decoder(zq) - x3f) ** 2).sum(1) / (2 * SIGMA ** 2) - const
    kl = 0.5 * (mq ** 2 + (2 * ls).exp() - 2 * ls - 1).sum()
    loss = -(recon.mean() - kl)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
m_vi, s_vi = mq.detach().double(), ls.detach().double().exp()
print('per-image q: mean', m_vi.numpy().round(4), ' sd', s_vi.numpy().round(4), f' ELBO {elbo(vae2, x3, m_vi, s_vi):.2f}')
""")
md('### The encoder does it in one pass: compare its ELBO with the per-image fit and with log p(x)')
code(r"""
print('encoder q: mean', mu3.numpy().round(4), ' sd', sd3.numpy().round(4))
print(f'log p(x) {log_px:.2f}  |  per-image q {elbo(vae2, x3, m_vi, s_vi):.2f}  |  encoder q {elbo(vae2, x3, mu3, sd3):.2f}')
""")
section('6.4', 'The loss, term by term', 'One reparameterised draw for our 3, with numbers.')
code(r"""
eps = torch.tensor([0.5, -1.0], dtype=F64)
z = mu3 + sd3 * eps
with torch.no_grad(): xh = vae2.decoder(z[None].float())[0].double()
sse = float(((xh - x3) ** 2).sum()); kl = 0.5 * float((mu3 ** 2 + sd3 ** 2 - 2 * sd3.log() - 1).sum())
print(f'z = {z.numpy().round(4)}; squared error {sse:.2f}; reconstruction term {sse / (2 * SIGMA ** 2):.1f}; KL {kl:.2f}; loss {sse / (2 * SIGMA ** 2) + kl:.1f}')
""")
section('6.5', 'What the VAE learned', 'Sample z ~ N(0, I) and decode; look at where the codes are.')
code(r"""
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    show_digits(vae2.decoder(torch.randn(16, 2, generator=g)), ncol=16, size=.6, title='2-number VAE: z ~ N(0, I), decoded')
    show_digits(vae16.decoder(torch.randn(16, 16, generator=torch.Generator().manual_seed(5))), ncol=16, size=.6, title='16-number VAE')
    V2 = vae2.encode(Xte.float())[0]
print('codes of all test digits: mean', V2.mean(0).numpy().round(2), ' sd', V2.std(0).numpy().round(2))
plt.figure(figsize=(5, 4)); plt.scatter(*V2[:2000].T, c=yte[:2000], cmap='tab10', s=5); plt.scatter(*prior.T, s=4, c='red'); plt.title('VAE codes and N(0, I) draws'); plt.show()
""")
section('6.6', 'The KL weight β', 'Train 2-number VAEs with β = 0 and β = 256 (10 epochs each) and compare with β = 1.')
code(r"""
for beta, net in ((0.0, train(AutoEncoder(2, True), epochs=10, beta=0.0, seed=40)), (1.0, vae2), (256.0, train(AutoEncoder(2, True), epochs=10, beta=256.0, seed=41))):
    with torch.no_grad():
        mu_, lv_ = net.encode(Xte.float())
        kl = 0.5 * (mu_ ** 2 + lv_.exp() - lv_ - 1).sum(1).mean()
    print(f'beta = {beta:5.0f}: KL per image {kl:6.2f}, average sigma of q {torch.exp(0.5 * lv_).mean():.4f}, spread of codes {mu_.std(0).mean():.2f}')
""")
section('6.7', 'Wrap-up', 'From one number per example to 784: a hidden code first, then the data. Next lectures: GANs, then diffusion.')

# ------------------------------------------------------------------------------------------------ Appendix
md('# Appendix · slides outside the main route')
md('### Search by code, and a denoising autoencoder')
code(r"""
with torch.no_grad():
    codes = aes[16].encode(Xtr.float())[0]; q = aes[16].encode(x3[None].float())[0]
nearest = torch.cdist(q, codes)[0].argsort()[:5]
show_digits(torch.cat([x3[None], Xtr[nearest]]), ncol=6, title='our 3 and its 5 nearest training digits by 16-number code')
dae = train(AutoEncoder(z=16), epochs=10, mask=0.5, seed=12)
g = torch.Generator().manual_seed(12); xw = Xte[[3, 2, 1, 18, 4, 8, 11, 0]]
damaged = xw * (torch.rand(xw.shape, generator=g, dtype=F64) > 0.5)
with torch.no_grad():
    plain, clean = [n.decoder(n.encode(damaged.float())[0]).double() for n in (aes[16], dae)]
print(f'squared error against the clean digits: plain {((plain - xw) ** 2).sum(1).mean():.1f}, denoising {((clean - xw) ** 2).sum(1).mean():.1f}')
show_digits(torch.cat([xw, damaged, plain, clean]), title='clean · half the pixels removed · plain autoencoder · denoising autoencoder')
""")

md("### Fit a mixture to the codes, then sample (Part 3's recipe)")
code(r"""
with torch.no_grad(): Ctr = ae2.encode(Xtr.float())[0].double()
w, means, covs, ll = em(Ctr, 10, iters=200, seed=31)
k = torch.multinomial(w, 16, replacement=True, generator=torch.Generator().manual_seed(8))
zs = torch.stack([torch.distributions.MultivariateNormal(means[j], covs[j]).sample() for j in k])
with torch.no_grad(): show_digits(ae2.decoder(zs.float()), ncol=16, size=.6, title='codes from a 10-bump mixture fitted to the codes, decoded')
""")



nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                                                'language_info': {'name': 'python'}, 'colab': {'provenance': []}})
if __name__ == '__main__':
    out = ROOT / NAME
    os.chdir(ROOT)
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']   # this interpreter
    client = NotebookClient(nb, timeout=1800, kernel_name='python3', km=km, resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    nbf.write(nb, out)
    json.dump(anchors, open(ROOT / 'data' / 'notebook-anchors.json', 'w'), indent=1)
    print('wrote', out, len(cells), 'cells,', len(anchors), 'anchors')
