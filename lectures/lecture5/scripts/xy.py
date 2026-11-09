"""x, y and the probabilities about them: the data for the slides xy and gen-disc.

    MNIST_ROOT=... python lecture5/scripts/xy.py      # from lectures/, under a minute on the CPU

Writes lecture5/data/xy.json:
  p_y         the share of each digit among the 60,000 training digits
  classifier  a softmax regression (784 -> 10 numbers, then softmax) trained on the 60,000 training digits:
              its test accuracy, and p(y | x) for our 3 (test digit #90) and for the display digit it is least sure about
  ink         one number per image, its total ink (the sum of its 784 pixel values), for the training 0s and 1s:
              p(ink | y) for each class (a Gaussian kernel density), p(y) among the 0s and 1s, p(ink, y), p(ink),
              and p(y = 1 | ink) by Bayes' rule
Section 1.3 of the notebook repeats all of it.
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'lecture2' / 'scripts'))
import train as T  # noqa: E402  Lecture 2's data loader

S = json.load(open(HERE.parent / 'lecture2' / 'data' / 'science.json'))
DISP = torch.tensor(S['mnist']['display_index'])
EXAMPLE = S['meta']['example_index']            # test digit #90, our 3
GRID = torch.linspace(0, 260, 261, dtype=torch.float64)


def classifier(xtr, ytr, xte, yte):
    torch.manual_seed(0)
    net = torch.nn.Linear(784, 10)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for epoch in range(5):
        for idx in torch.randperm(len(xtr), generator=gen).split(128):
            loss = torch.nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((net(xte).argmax(1) == yte).float().mean())
        p3 = net(xte[EXAMPLE][None]).softmax(1)[0]
        pd = net(xte[DISP]).softmax(1)
    hard = int(pd.max(1).values.argmin())               # the display digit whose top label is least probable
    return dict(test_accuracy=round(acc, 4), our3=[round(float(v), 4) for v in p3],
                hard=dict(display_i=hard, label=int(yte[DISP[hard]]), p=[round(float(v), 4) for v in pd[hard]]))


def kde(v):
    v = v.double()
    h = 1.06 * float(v.std()) * len(v) ** -0.2           # Silverman's rule
    dens = torch.exp(-0.5 * ((GRID[:, None] - v[None]) / h) ** 2).sum(1) / (len(v) * h * math.sqrt(2 * math.pi))
    return dens, h


def ink(xtr, ytr):
    keep = (ytr == 0) | (ytr == 1)
    s, y = xtr[keep].sum(1), ytr[keep]
    p1 = float((y == 1).double().mean())
    d0, h0 = kde(s[y == 0])
    d1, h1 = kde(s[y == 1])
    j0, j1 = (1 - p1) * d0, p1 * d1                     # p(ink, y) = p(y) p(ink | y)
    px = j0 + j1                                         # p(ink) = sum over y of p(ink, y)
    post1 = j1 / px                                      # p(y = 1 | ink) = p(ink, y = 1) / p(ink)
    cross = float(GRID[int((post1 < 0.5).nonzero()[0])])
    r = lambda t, n=6: [round(float(v), n) for v in t]
    return dict(grid=r(GRID, 1), n=[int((y == 0).sum()), int((y == 1).sum())], p_y=[round(1 - p1, 4), round(p1, 4)],
                bandwidth=[round(h0, 2), round(h1, 2)], mean=[round(float(s[y == c].mean()), 1) for c in (0, 1)],
                p_x_given_y=[r(d0), r(d1)], p_x=r(px), p_y1_given_x=r(post1, 4), boundary=cross)


if __name__ == '__main__':
    data = T.load_mnist()
    (xtr, ytr), (xte, yte) = data['train'], data['test']
    out = dict(example_index=EXAMPLE,
               p_y=[round(float((ytr == c).double().mean()), 4) for c in range(10)],
               classifier=classifier(xtr, ytr, xte, yte), ink=ink(xtr, ytr))
    c, k = out['classifier'], out['ink']
    print('p(y):', out['p_y'])
    print(f"classifier test accuracy {c['test_accuracy']:.4f}; p(y = 3 | our 3) = {c['our3'][3]:.3f}")
    print(f"least sure display digit: #{c['hard']['display_i']} (a {c['hard']['label']}), p(y | x) = {c['hard']['p']}")
    print(f"ink: {k['n']} training 0s and 1s, p(y) = {k['p_y']}, mean ink {k['mean']}, p(y = 1 | ink) = 0.5 at ink {k['boundary']}")
    (HERE / 'data' / 'xy.json').write_text(json.dumps(out, separators=(',', ':')))
    print('wrote', HERE / 'data' / 'xy.json')
