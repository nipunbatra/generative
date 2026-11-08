"""Build and execute the Lecture 5 companion notebook.

    MNIST_ROOT=/path/to/datasets python lecture5/scripts/build_notebook.py

The notebook follows the deck's section numbers (1.1 ... 4.8); each section's heading cell has the id
sec-X-Y, each part's heading part-k, which the slides link to (data/notebook-anchors.json lists them).
It loads Lecture 2's trained checkpoints from lecture2/models (or downloads them in Colab) and trains only
a few small extra models, so it runs top to bottom in a few minutes.
"""
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
NAME = 'lecture5-autoencoders-vae.ipynb'
DECK = 'https://nipunbatra.github.io/generative/lectures/lecture5/'
COLAB = f'https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture5/{NAME}'
RAW = 'https://raw.githubusercontent.com/nipunbatra/generative/main/lectures/lecture2'
L2NB = 'https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture2/lecture2-generative-models.ipynb'
cells, anchors = [], []


def add(cell, anchor):
    cell_id = anchor or f'c{len(cells):03d}'
    cell['id'] = cell_id
    cell['metadata']['id'] = cell_id
    if anchor:
        anchors.append(anchor)
    cells.append(cell)


def md(s, anchor=None):
    add(nbf.v4.new_markdown_cell(s.strip().replace('{DECK}', DECK).replace('{COLAB}', COLAB).replace('{L2NB}', L2NB)), anchor)


def code(s, anchor=None):
    add(nbf.v4.new_code_cell(s.strip().replace('{RAW}', RAW)), anchor)


def part(k, title, text=''):
    md(f'# Part {k} · {title}\n\n{text}', anchor=f'part-{k}')


def section(k, title, first, text=''):
    md(f'## {k} {title}\n\nSlides: [{k} {title}]({DECK}#{first}/1)\n\n{text}', anchor=f'sec-{k.replace(".", "-")}')


# ------------------------------------------------------------------------------------------------ title
md(r"""
# Lecture 5 · Autoencoders and the VAE in one hour

[▶ Open in Colab]({COLAB}) · [Slides]({DECK}) · [Lecture 5 at a glance]({DECK}#contents/1) · ES 667 Deep Learning, IIT Gandhinagar

This notebook runs the experiments behind the Lecture 5 slides, **in the same order and with the same section numbers** (1.1 … 4.8). Every slide's **▶ Notebook** link jumps to its section here.

The networks are Lecture 2's: a 2-number autoencoder, and 2-number and 16-number VAEs, trained there for 30 and 50 epochs. This notebook loads them and trains only a few small extra models for a few epochs each.

Run everything once from the top (*Runtime → Run all*). It needs PyTorch, torchvision, NumPy and matplotlib. In Colab it downloads MNIST (about 12 MB) and the three trained networks from the course repository; in a clone of the repository it uses the files in `lectures/lecture2/models`.
""", anchor='top')
code(r"""
import math, os, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
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
part('1', 'Where we start: the ELBO', 'Lecture 1 ended with one objective we can compute, the ELBO. Two of its tools come back today.')
section('1.1', 'What Lecture 1 gave us', 'recap-elbo',
        'Tool ①, the reparameterisation trick: z = μ + σ ⊙ ε with ε ~ N(0, I), so gradients reach μ and σ. '
        'Tool ②, the KL from a Gaussian to N(0, 1) by formula: ½(μ² + σ² − log σ² − 1).')
code(r"""
# Tool 1: a random draw that gradients pass through.
mu = torch.tensor([0.5, -1.0], requires_grad=True)
log_sigma = torch.tensor([0.0, -0.5], requires_grad=True)
eps = torch.randn(2, generator=torch.Generator().manual_seed(0))
z = mu + log_sigma.exp() * eps          # the randomness sits in eps
z.sum().backward()
print('z =', z.detach().numpy().round(3))
print('dz/dmu =', mu.grad.numpy(), '  dz/dlog_sigma =', log_sigma.grad.numpy().round(3), '(= sigma * eps)')

# Tool 2: the KL from N(mu, sigma^2) to N(0, 1), three ways.
def kl_to_std_normal(m, s):
    return 0.5 * (m ** 2 + s ** 2 - math.log(s ** 2) - 1)
m_, s_ = 2.0, 1.0
draws = Normal(m_, s_).sample((200_000,), )
mc = float((Normal(m_, s_).log_prob(draws) - Normal(0., 1.).log_prob(draws)).mean())
print(f'formula {kl_to_std_normal(m_, s_):.4f}   torch {float(kl_divergence(Normal(m_, s_), Normal(0., 1.))):.4f}   Monte Carlo {mc:.4f}')
print(f'at mu = 0, sigma = 1 the KL is {kl_to_std_normal(0.0, 1.0):.1f}')
""")
section('1.2', 'Today’s question', 'realfake',
        'Six of the twelve digits on the slide come from a network: the first six draws z ~ N(0, I) through Lecture 2’s 16-number VAE. '
        'Section 4.6 makes more of them.')

# ------------------------------------------------------------------------------------------------ Part 2
part('2', 'Autoencoders: squeeze, then rebuild', 'Squeeze each digit into 2 numbers, then rebuild it from them.')
section('2.1', 'An image is 784 numbers', 'mnist', 'MNIST, grey levels scaled to [0, 1]. Our running example is test digit #90, a 3.')
code(r"""
from torchvision.datasets import MNIST
MNIST_ROOT = os.environ.get('MNIST_ROOT', 'data')
have = (Path(MNIST_ROOT) / 'MNIST' / 'raw' / 'train-images-idx3-ubyte').exists()
train_set = MNIST(MNIST_ROOT, train=True, download=not have)
test_set = MNIST(MNIST_ROOT, train=False, download=not have)
Xtr = train_set.data.reshape(-1, 784).double() / 255; ytr = train_set.targets
Xte = test_set.data.reshape(-1, 784).double() / 255; yte = test_set.targets
x3 = Xte[90]
print(f'{len(Xtr):,} training and {len(Xte):,} test digits, {Xtr.shape[1]} numbers each; test digit #90 is a {int(yte[90])}')
show_digits(Xtr[:16], title='the first 16 training digits')
show_digits(x3[None], ncol=1, size=2, title='our 3')
""")
section('2.2', 'Encoder, code, decoder', 'ae-arch',
        'One hidden layer of 400 ReLU units on each side, as in Kingma and Welling (2014). The same class becomes a VAE in Part 4.')
code(r"""
class AutoEncoder(nn.Module):
    # variational=True: the encoder outputs a mean and a log variance per code number (Part 4).
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

SIGMA = 0.2                                              # the VAE's pixel noise (Part 4)
def loss_fn(model, x, target, beta=1.0):
    x_hat, mu, logvar = model(x)
    sse = ((x_hat - target) ** 2).sum(1)                 # squared error summed over the 784 pixels
    if not model.variational:
        return sse.mean()
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)
    return (sse / (2 * SIGMA ** 2) + beta * kl).mean()

def train(model, epochs, beta=1.0, seed=0, lr=1e-3, log_every=None):
    torch.manual_seed(seed); model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = Xtr.float().to(DEVICE); step = 0
    for epoch in range(epochs):
        for idx in torch.randperm(len(X)).split(128):
            xb = X[idx]
            loss = loss_fn(model, xb, xb, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            if log_every and step % log_every == 0:
                print(f'step {step:4d}: loss {loss.item():7.2f}')
            step += 1
    return model.eval().cpu()

@torch.no_grad()
def test_sse(model, X=None):
    X = (Xte if X is None else X).float()
    mu, _ = model.encode(X)
    return float(((model.decoder(mu) - X) ** 2).sum(1).mean())

def load_trained(name, z, variational):
    # Lecture 2's networks: from lecture2/models in a clone, else downloaded from the repository.
    for path in (Path('..') / 'lecture2' / 'models' / f'{name}.pt', Path('models') / f'{name}.pt'):
        if path.exists(): break
    else:
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f'{RAW}/models/{name}.pt', path)
    net = AutoEncoder(z, variational)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])
    return net.eval()

print(AutoEncoder(z=2))
print(f'encoder weights: {sum(p.numel() for p in AutoEncoder(z=2).encoder.parameters()):,}')
""")
section('2.3', 'The loss, and training', 'ae-loss',
        'Look at the untrained network first. Then load Lecture 2’s trained 2-number autoencoder (30 epochs), or set `RETRAIN = True` to train it here.')
code(r"""
torch.manual_seed(0)
fresh = AutoEncoder(z=2).eval()
with torch.no_grad(): out = fresh(Xte[:8].float())[0]
print(f'untrained outputs lie between {out.min():.2f} and {out.max():.2f}; test squared error {test_sse(fresh):.1f} per image')
RETRAIN = False
ae2 = train(AutoEncoder(z=2), epochs=30) if RETRAIN else load_trained('ae2', 2, False)
print(f'trained: test squared error {test_sse(ae2):.2f} per image')
""")
md('### Our 3, pixel by pixel: target x, prediction x̂, and each pixel’s squared error')
code(r"""
with torch.no_grad(): x3_hat = ae2.decoder(ae2.encode(x3[None].float())[0])[0].double()
err = (x3 - x3_hat) ** 2
print(f'squared error of our 3, summed over 784 pixels: {float(err.sum()):.2f}')
for (r, c), what in (((24, 10), 'ink the network missed'), ((5, 11), 'ink where the 3 has none'), ((3, 24), 'background')):
    j = r * 28 + c
    print(f'row {r:2d}, col {c:2d} ({what}): x = {x3[j]:.2f}, x_hat = {x3_hat[j]:.2f}, (x - x_hat)^2 = {err[j]:.2f}')
fig, ax = plt.subplots(1, 3, figsize=(6, 2.2))
for a, img, t, cmap in zip(ax, (1 - x3, 1 - x3_hat, err), ('target x', 'prediction x_hat', '(x - x_hat)^2'), ('gray', 'gray', 'Reds')):
    a.imshow(img.reshape(28, 28), cmap=cmap); a.set_title(t, fontsize=9); a.axis('off')
plt.show()
""")
md('### Training in a few lines: one pass over the data, from scratch')
code(r"""
demo = train(AutoEncoder(z=2), epochs=1, seed=1, log_every=100)
print(f'after one pass: test squared error {test_sse(demo):.1f} (Lecture 2’s 30 passes: {test_sse(ae2):.1f})')
""")
section('2.4', 'What the code holds', 'code-size',
        'Train autoencoders with wider codes (3 epochs each here; 30 on the slides), then plot where the 2-number codes land.')
code(r"""
aes = {2: ae2}
for z in (8, 128):
    aes[z] = train(AutoEncoder(z=z), epochs=3, seed=z)
for z, net in aes.items():
    print(f'code of {z:3d} numbers: test squared error {test_sse(net):6.2f}')
idx = [3, 2, 1, 18, 4, 8, 11, 0]
with torch.no_grad():
    show_digits(torch.cat([Xte[idx].float()] + [net.decoder(net.encode(Xte[idx].float())[0]) for net in aes.values()]),
                title='8 test digits, then rebuilt from 2, 8 and 128 numbers')
with torch.no_grad(): C2 = ae2.encode(Xte[:2000].float())[0]
plt.figure(figsize=(5, 4)); plt.scatter(*C2.T, c=yte[:2000], cmap='tab10', s=5); plt.colorbar(label='digit')
plt.title('2-number autoencoder codes of 2,000 test digits'); plt.show()
""")

# ------------------------------------------------------------------------------------------------ Part 3
part('3', 'Can an autoencoder generate?', 'Keep the decoder, throw the encoder away, and invent the code.')
section('3.1', 'Decode random codes', 'gen-question', 'Feed draws from N(0, I) to the decoders.')
code(r"""
for z, net in aes.items():
    with torch.no_grad(): xs = net.decoder(torch.randn(16, z, generator=torch.Generator().manual_seed(5)))
    show_digits(xs, ncol=16, size=.6, title=f'N(0, I) codes through the {z}-number autoencoder')
""")
section('3.2', 'Where the codes are', 'overlay', 'Overlay N(0, I) draws on the codes of real test digits, then decode a grid of codes.')
code(r"""
prior = torch.randn(500, 2, generator=torch.Generator().manual_seed(6))
plt.figure(figsize=(5, 4)); plt.scatter(*C2.T, c=yte[:2000], cmap='tab10', s=5); plt.scatter(*prior.T, s=4, c='red', label='N(0, I) draws')
plt.legend(); plt.title('autoencoder codes of 2,000 test digits'); plt.show()
print(f'real codes run from {C2.min(0).values.numpy().round(1)} to {C2.max(0).values.numpy().round(1)}')
lo, hi = C2.quantile(.01, 0), C2.quantile(.99, 0)
gx, gy = torch.linspace(lo[0], hi[0], 12), torch.linspace(hi[1], lo[1], 12)
grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1).reshape(-1, 2)
with torch.no_grad(): show_digits(ae2.decoder(grid), ncol=12, size=.5, title='decoder output over a 12 x 12 grid of codes')
""")
section('3.3', 'What we want instead', 'wishes',
        'Codes that fill N(0, I); each image owning a small cloud of codes; one probability model that can score and sample. The VAE does all three.')

# ------------------------------------------------------------------------------------------------ Part 4
part('4', 'The VAE: Lecture 1’s ELBO, one image at a time',
     'A model with a hidden code: z ~ N(0, I), x | z ~ N(decoder(z), 0.2² I). Lecture 2’s trained VAEs (50 epochs).')
section('4.1', 'A model with a hidden code', 'vae-pgm', 'To draw an image, follow the arrows: first z from the prior, then the decoder, then pixel noise.')
code(r"""
vae2 = load_trained('vae2', 2, True); vae16 = load_trained('vae16', 16, True)
z = torch.randn(1, 2, generator=torch.Generator().manual_seed(3))
with torch.no_grad(): mean = vae2.decoder(z)
x_sample = mean + SIGMA * torch.randn(mean.shape, generator=torch.Generator().manual_seed(4))
print('z =', z.numpy().round(2))
show_digits(torch.cat([mean, x_sample.clamp(0, 1)]), ncol=2, size=1.6, title='the mean image, and the mean plus 0.2 x noise')
const = 784 * math.log(SIGMA * math.sqrt(2 * math.pi))
@torch.no_grad()
def log_lik(net, x, Z):                     # log p(x | z) for a batch of codes Z
    return -((net.decoder(Z.float()).double() - x) ** 2).sum(1) / (2 * SIGMA ** 2) - const
print(f'the constant 784 log(sigma sqrt(2 pi)) = {const:.2f}')
""")
section('4.2', 'The same wall as Lecture 1', 'vae-wall',
        'p(x) is an integral over every code. With only 2 numbers we can still add up rectangles on a fine grid; with 16 we could not.')
code(r"""
with torch.no_grad(): mu3, lv3 = vae2.encode(x3[None].float())
mu3, sd3 = mu3[0].double(), (0.5 * lv3[0].double()).exp()
half, n = 0.15, 241                                  # a fine window around the posterior
ax_x = torch.linspace(mu3[0] - half, mu3[0] + half, n, dtype=F64); ax_y = torch.linspace(mu3[1] - half, mu3[1] + half, n, dtype=F64)
Zg = torch.stack(torch.meshgrid(ax_x, ax_y, indexing='xy'), -1).reshape(-1, 2)
lj = log_lik(vae2, x3, Zg) - 0.5 * (Zg ** 2).sum(1) - math.log(2 * math.pi)    # log p(x | z) + log p(z)
cell = float((ax_x[1] - ax_x[0]) * (ax_y[1] - ax_y[0]))
log_px = float(torch.logsumexp(lj, 0)) + math.log(cell)
post = torch.softmax(lj, 0)
pm = (post[:, None] * Zg).sum(0); ps = ((post[:, None] * (Zg - pm) ** 2).sum(0)).sqrt()
print(f'log p(x) by adding rectangles: {log_px:.2f}')
print('posterior of our 3: mean', pm.numpy().round(4), ' sd', ps.numpy().round(4))
plt.figure(figsize=(4, 3.6)); plt.contourf(ax_x, ax_y, post.reshape(n, n), levels=20, cmap='Greens'); plt.title('p(z | our 3), zoomed'); plt.show()
""")
section('4.3', 'The ELBO for one image', 'vae-elbo',
        'ELBO(x) = E_q[log p(x | z)] − KL(q(z | x) ‖ p(z)). The encoder gives q for our 3 in one pass; its ELBO sits just below log p(x).')
code(r"""
def elbo(net, x, m_, s_, draws=20000, seed=21):
    eps = torch.randn(draws, 2, generator=torch.Generator().manual_seed(seed), dtype=F64)
    kl = 0.5 * float((m_ ** 2 + s_ ** 2 - 2 * s_.log() - 1).sum())
    return float(log_lik(net, x, m_ + s_ * eps).mean()) - kl

print('encoder q for our 3: mean', mu3.numpy().round(4), ' sd', sd3.numpy().round(4))
e3 = elbo(vae2, x3, mu3, sd3)
print(f'log p(x) {log_px:.2f}  |  ELBO with the encoder’s q {e3:.2f}  |  gap = KL to the posterior {log_px - e3:.2f}')
""")
section('4.4', 'The loss, term by term', 'vae-loss', 'One reparameterised draw for our 3, with ε = (0.5, −1.0).')
code(r"""
eps = torch.tensor([0.5, -1.0], dtype=F64)
z = mu3 + sd3 * eps
with torch.no_grad(): xh = vae2.decoder(z[None].float())[0].double()
sse = float(((xh - x3) ** 2).sum()); kl = 0.5 * float((mu3 ** 2 + sd3 ** 2 - 2 * sd3.log() - 1).sum())
print(f'z = {z.numpy().round(4)}')
print(f'squared error {sse:.2f}; reconstruction term {sse / (2 * SIGMA ** 2):.1f}; KL {kl:.2f}; loss {sse / (2 * SIGMA ** 2) + kl:.1f}')
""")
section('4.5', 'Two terms pull against each other', 'vae-tug',
        'The reconstruction term wants tiny, separated clouds; the KL wants every cloud to be N(0, I). Averages over the 10,000 test digits:')
code(r"""
with torch.no_grad():
    mu_, lv_ = vae2.encode(Xte.float())
    kl_te = 0.5 * (mu_ ** 2 + lv_.exp() - lv_ - 1).sum(1).mean()
print(f'2-number VAE: squared error {test_sse(vae2):.2f}, KL {kl_te:.2f} per image, average sigma of q {torch.exp(0.5 * lv_).mean():.4f}')
print(f'2-number autoencoder: squared error {test_sse(ae2):.2f}')
""")
section('4.6', 'What the VAE learned', 'vae-samples', 'Sample z ~ N(0, I) and decode; look at where the codes are.')
code(r"""
with torch.no_grad():
    show_digits(ae2.decoder(torch.randn(16, 2, generator=torch.Generator().manual_seed(5))), ncol=16, size=.6, title='2-number autoencoder: z ~ N(0, I), decoded')
    show_digits(vae2.decoder(torch.randn(16, 2, generator=torch.Generator().manual_seed(5))), ncol=16, size=.6, title='2-number VAE')
    show_digits(vae16.decoder(torch.randn(16, 16, generator=torch.Generator().manual_seed(5))), ncol=16, size=.6, title='16-number VAE')
    V2 = vae2.encode(Xte.float())[0]
print('VAE codes of all test digits: mean', V2.mean(0).numpy().round(2), ' sd', V2.std(0).numpy().round(2))
plt.figure(figsize=(5, 4)); plt.scatter(*V2[:2000].T, c=yte[:2000], cmap='tab10', s=5); plt.scatter(*prior.T, s=4, c='red')
plt.title('VAE codes and N(0, I) draws (red)'); plt.show()
""")
section('4.7', 'The KL weight β', 'beta-what',
        'Train 2-number VAEs with β = 0 and β = 256 (5 epochs each here; 50 in Lecture 2) and compare with β = 1.')
code(r"""
for beta, net in ((0.0, train(AutoEncoder(2, True), epochs=5, beta=0.0, seed=40)), (1.0, vae2), (256.0, train(AutoEncoder(2, True), epochs=5, beta=256.0, seed=41))):
    with torch.no_grad():
        mu_, lv_ = net.encode(Xte.float())
        kl = 0.5 * (mu_ ** 2 + lv_.exp() - lv_ - 1).sum(1).mean()
    print(f'beta = {beta:5.0f}: KL per image {kl:6.2f}, average sigma of q {torch.exp(0.5 * lv_).mean():.4f}, spread of codes {mu_.std(0).mean():.2f}')
""")
section('4.8', 'Wrap-up', 'summary',
        'A VAE is an autoencoder trained with Lecture 1’s objective: the encoder outputs a cloud, the code is a reparameterised draw, and the loss adds a KL to N(0, I). '
        'Next lectures replace the pixel-wise Gaussian likelihood: GANs, then diffusion.')

# ------------------------------------------------------------------------------------------------ Appendix
md(r"""
# Appendix · optional slides

The optional slides reuse Lecture 2’s experiments, and Lecture 2’s [notebook]({L2NB}) has the code for each: PCA against the autoencoder, random codes and straight-line walks between digits, per-image variational inference for our 3 against the encoder’s q, and the β sweep.
""", anchor='optional')

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
    (ROOT / 'data').mkdir(exist_ok=True)
    json.dump(anchors, open(ROOT / 'data' / 'notebook-anchors.json', 'w'), indent=1)
    print('wrote', out, len(cells), 'cells,', len(anchors), 'anchors')
