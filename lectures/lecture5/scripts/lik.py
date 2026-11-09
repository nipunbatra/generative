"""How well does each code draw our 3? The data for the likelihood map on the slides sigma and vae-wall.

    MNIST_ROOT=... python lecture5/scripts/lik.py      # from lectures/, a few seconds on the CPU

Uses Lecture 2's trained 2-number VAE (lecture2/models/vae2.pt). For our 3 (test digit #90) it scores every code on an
81 x 81 grid over [-4, 4]^2 with the log-likelihood
    log p(x | z) = -||x - decoder(z)||^2 / (2 * 0.2^2) - 784 log(0.2 sqrt(2 pi)),
the same score as Lecture 2's science.py (vae.grid) and lecture5/widgets.js (logLikFor).
Writes lecture5/data/lik.json: the grid's log-likelihoods (row by row from y = -4, rounded to 0.1), the best code on the grid
and its score. Section 5.2 of the notebook draws the same map.
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'lecture2' / 'scripts'))
import train as T  # noqa: E402  Lecture 2's network and data loader

S = json.load(open(HERE.parent / 'lecture2' / 'data' / 'science.json'))
EXAMPLE, SIGMA = S['meta']['example_index'], S['meta']['sigma']
N, LIM = 81, 4.0

if __name__ == '__main__':
    x3 = T.load_mnist()['test'][0][EXAMPLE].double()
    ck = torch.load(HERE.parent / 'lecture2' / 'models' / 'vae2.pt', map_location='cpu', weights_only=False)
    net = T.AutoEncoder(2, True)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    ax = torch.linspace(-LIM, LIM, N, dtype=torch.float64)
    Z = torch.stack(torch.meshgrid(ax, ax, indexing='xy'), -1).reshape(-1, 2)
    with torch.no_grad():
        sse = ((net.decoder(Z.float()).double() - x3) ** 2).sum(1)
    const = 784 * math.log(SIGMA * math.sqrt(2 * math.pi))
    ll = -sse / (2 * SIGMA ** 2) - const
    k = int(ll.argmax())
    out = dict(n=N, lim=LIM, loglik=[round(float(v), 1) for v in ll], best=[round(float(v), 2) for v in Z[k]], best_loglik=round(float(ll[k]), 1),
               min_loglik=round(float(ll.min()), 1))
    print(f'log p(x | z) on the grid: best {out["best_loglik"]} at {out["best"]}, worst {out["min_loglik"]}')
    print(f'share of grid codes within 100 nats of the best: {float((ll > ll.max() - 100).double().mean()):.4f}')
    (HERE / 'data' / 'lik.json').write_text(json.dumps(out, separators=(',', ':')))
    print('wrote', HERE / 'data' / 'lik.json')
