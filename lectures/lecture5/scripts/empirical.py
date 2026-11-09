"""Why not sample from the real codes instead of N(0, I)? The data for the slide ae-empirical.

    MNIST_ROOT=... python lecture5/scripts/empirical.py      # from lectures/, about a minute on the CPU

Uses Lecture 2's trained 2-number autoencoder (lecture2/models/ae2.pt) and the codes of the 60,000 training digits.
Three ways to sample a code from the real ones, and N(0, I) for comparison:
  copy    pick a training digit's code: the decoder rebuilds that training digit
  jitter  a training digit's code plus N(0, I) noise (a Gaussian kernel density with width 1)
  mog     Lecture 2's mixture of 10 Gaussians fitted to the training codes (science.json: gen.ae2_mog)
  prior   N(0, I)
A sampled code is "in a gap" if its nearest training code is further away than the 99th percentile of that distance
for the 10,000 test digits' codes: further than real new digits ever land.
Writes lecture5/data/empirical.json: for each method 300 codes for the scatter, 8 decoded images (uint8, base64),
and the share of 10,000 draws in a gap; for copy also the 8 training digits themselves.
"""
import base64
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'lecture2' / 'scripts'))
import train as T  # noqa: E402  Lecture 2's network and data loader

S = json.load(open(HERE.parent / 'lecture2' / 'data' / 'science.json'))
MOG = S['gen']['ae2_mog']
N_SHOW, N_SCATTER, N_GAP, SEED = 8, 300, 10000, 7


def b64(x):
    return base64.b64encode((x.clamp(0, 1) * 255).round().to(torch.uint8).numpy().tobytes()).decode()


def nearest(c, ref):
    return torch.cat([torch.cdist(c[i:i + 2000], ref).min(1).values for i in range(0, len(c), 2000)])


if __name__ == '__main__':
    data = T.load_mnist()
    xtr, xte = data['train'][0], data['test'][0]
    net = T.AutoEncoder(2)
    net.load_state_dict(torch.load(HERE.parent / 'lecture2' / 'models' / 'ae2.pt', map_location='cpu', weights_only=False)['state_dict'])
    net.eval()
    with torch.no_grad():
        ctr, cte = net.encode(xtr)[0], net.encode(xte)[0]
    gap = float(nearest(cte, ctr).quantile(0.99))
    g = torch.Generator().manual_seed(SEED)
    w, mu, cov = torch.tensor(MOG['w']), torch.tensor(MOG['means']), torch.tensor(MOG['covs'])
    chol = torch.linalg.cholesky(cov)

    def draw(method, n):
        if method == 'copy':
            i = torch.randint(len(ctr), (n,), generator=g)
            return ctr[i], i
        if method == 'jitter':
            i = torch.randint(len(ctr), (n,), generator=g)
            return ctr[i] + torch.randn(n, 2, generator=g), i
        if method == 'mog':
            k = torch.multinomial(w / w.sum(), n, replacement=True, generator=g)
            return mu[k] + (chol[k] @ torch.randn(n, 2, 1, generator=g)).squeeze(-1), None
        return torch.randn(n, 2, generator=g), None

    out = dict(gap_distance=round(gap, 3), seed=SEED, methods={})
    for method in ('copy', 'jitter', 'mog', 'prior'):
        c, i = draw(method, N_GAP)
        share = float((nearest(c, ctr) > gap).float().mean())
        with torch.no_grad():
            img = net.decoder(c[:N_SHOW])
        row = dict(codes=[[round(float(a), 2), round(float(b), 2)] for a, b in c[:N_SCATTER]], images=b64(img), in_gap=round(share, 4))
        if method == 'copy':
            row['train_images'] = b64(xtr[i[:N_SHOW]])
            row['train_index'] = i[:N_SHOW].tolist()
        out['methods'][method] = row
        print(f'{method:6s}: {share:6.1%} of {N_GAP:,} draws in a gap (nearest training code further than {gap:.2f})')
    (HERE / 'data' / 'empirical.json').write_text(json.dumps(out, separators=(',', ':')))
    print('wrote', HERE / 'data' / 'empirical.json')
