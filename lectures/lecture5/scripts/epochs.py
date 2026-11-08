"""Where the 2-number codes go while an autoencoder trains: the data for the slides ae-epochs and ae-overfit.

    MNIST_ROOT=... python lecture5/scripts/epochs.py      # from lectures/, about 5 minutes on an M-series Mac

Two runs, both with Lecture 2's 2-number autoencoder (784 -> 400 -> 2 -> 400 -> 784):
  all  Lecture 2's own training of ae2, replayed step for step (same seed, device and batches), so its epoch-30 codes
       are the ones on the slide ae-scatter; then it keeps going to epoch 300. It needs the device Lecture 2 used (mps):
       on another device the run is similar but not identical.
  few  the same network trained on only 200 training digits (20 of each), all 200 in every step, for 10,000 steps,
       on the CPU. It overfits: the training error keeps falling while the test error rises.
Writes lecture5/data/epochs.json: the codes of Lecture 2's 1,000 display digits (test set) at chosen epochs,
training and test error per epoch, and our 3 (test digit #90) rebuilt at each chosen epoch.
"""
import base64
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'lecture2' / 'scripts'))
import train as T  # noqa: E402  Lecture 2's network, data loader and loss

S = json.load(open(HERE.parent / 'lecture2' / 'data' / 'science.json'))
DISP = torch.tensor(S['mnist']['display_index'])
EXAMPLE = S['meta']['example_index']            # test digit #90, our 3
SAVE_ALL = [0, 1, 3, 10, 30, 100, 300]
SAVE_FEW = [0, 10, 30, 100, 300, 1000, 3000, 10000]
N_FEW, STEPS_FEW, SEED_FEW = 20, 10000, 5


def png_bytes(x):
    return base64.b64encode((x.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()).decode()


def rnd(t, n):
    return [[round(float(v), n) for v in row] for row in t]


def errors(model, X):
    model.eval()
    with torch.no_grad():
        out = torch.cat([((model.decoder(model.encode(X[i:i + 5000])[0]) - X[i:i + 5000]) ** 2).sum(1) for i in range(0, len(X), 5000)])
    model.train()
    return float(out.mean())


def record(model, D, x3, store, ep, n):
    model.eval()
    with torch.no_grad():
        store['codes'][str(ep)] = rnd(model.encode(D)[0].cpu(), n)
        store['recon3'][str(ep)] = png_bytes(model.decoder(model.encode(x3[None])[0])[0])
    model.train()


def run_all(data):
    seed = 1000 + sorted(T.CONFIGS).index('ae2')
    torch.manual_seed(seed)
    model = T.AutoEncoder(2).to(T.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr, xte = data['train'][0].to(T.DEVICE), data['test'][0].to(T.DEVICE)
    D, x3 = xte[DISP.to(T.DEVICE)], xte[EXAMPLE]
    gen = torch.Generator().manual_seed(seed)
    out = dict(n_train=len(xtr), device=T.DEVICE, save=SAVE_ALL, codes={}, recon3={}, epoch=[0],
               train=[errors(model, xtr)], test=[errors(model, xte)])
    record(model, D, x3, out, 0, 3)
    for epoch in range(1, SAVE_ALL[-1] + 1):
        order = torch.randperm(len(xtr), generator=gen).to(T.DEVICE)
        for i in range(0, len(xtr), 128):
            xb = xtr[order[i:i + 128]]
            loss, _, _ = T.losses(model, xb, xb, 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        out['epoch'].append(epoch); out['train'].append(errors(model, xtr)); out['test'].append(errors(model, xte))
        if epoch in SAVE_ALL:
            record(model, D, x3, out, epoch, 3)
            print(f'all  epoch {epoch:5d}  train {out["train"][-1]:6.2f}  test {out["test"][-1]:6.2f}', flush=True)
    ref = torch.tensor(S['ae']['ae2_codes'])
    out['max_diff_from_lecture2'] = round(float((torch.tensor(out['codes']['30']) - ref).abs().max()), 4)
    print('largest difference from Lecture 2’s epoch-30 codes:', out['max_diff_from_lecture2'])
    out['train'] = [round(v, 3) for v in out['train']]; out['test'] = [round(v, 3) for v in out['test']]
    return out


def run_few(data):
    xtr, ytr = data['train']
    xte = data['test'][0]
    g = torch.Generator().manual_seed(SEED_FEW)
    pick = torch.cat([torch.nonzero(ytr == c).squeeze(1)[torch.randperm(int((ytr == c).sum()), generator=g)[:N_FEW]] for c in range(10)])
    X = xtr[pick]
    torch.manual_seed(SEED_FEW)
    model = T.AutoEncoder(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    D, x3 = xte[DISP], xte[EXAMPLE]
    log_steps = sorted({int(round(10 ** (i / 10))) for i in range(0, 41)} | set(SAVE_FEW))
    out = dict(n_train=len(X), train_index=pick.tolist(), save=SAVE_FEW, codes={}, recon3={}, epoch=[], train=[], test=[])

    def log(step):
        out['epoch'].append(step); out['train'].append(round(errors(model, X), 3)); out['test'].append(round(errors(model, xte), 3))

    log(0); record(model, D, x3, out, 0, 3)
    for step in range(1, STEPS_FEW + 1):
        loss = ((model.decoder(model.encoder(X)) - X) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in log_steps:
            log(step)
        if step in SAVE_FEW:
            record(model, D, x3, out, step, 2)
            print(f'few  epoch {step:5d}  train {out["train"][-1]:6.2f}  test {out["test"][-1]:6.2f}', flush=True)
    model.eval()
    with torch.no_grad():
        out['train_codes'] = rnd(model.encoder(X), 2)
        # our 3 rebuilt at the end, and the training digit nearest to that rebuild: has it memorised a training digit?
        r3 = model.decoder(model.encoder(x3[None]))[0]
        j = int(((X - r3) ** 2).sum(1).argmin())
        out['nearest_train_to_recon3'] = dict(i=j, label=int(ytr[pick[j]]), image=png_bytes(X[j]),
                                              sse_to_recon=round(float(((X[j] - r3) ** 2).sum()), 2),
                                              sse_to_x3=round(float(((x3 - r3) ** 2).sum()), 2))
    best = min(range(len(out['test'])), key=lambda i: out['test'][i])
    out['best'] = dict(epoch=out['epoch'][best], test=out['test'][best], train=out['train'][best])
    return out


def summary(run, eps):
    rows = {}
    for e in eps:
        c = torch.tensor(run['codes'][str(e)])
        lab = torch.tensor(S['mnist']['display_labels'])
        d = torch.cdist(c, c); d.fill_diagonal_(float('inf'))
        nn5 = torch.mode(lab[d.topk(5, largest=False).indices], 1).values
        i = run['epoch'].index(e)
        rows[str(e)] = dict(train=run['train'][i], test=run['test'][i],
                            radius=round(float(c.norm(dim=1).median()), 2),
                            box=rnd(torch.stack([c.quantile(.01, 0), c.quantile(.99, 0)]), 1),
                            knn5=round(float((nn5 == lab).float().mean()), 3))
    return rows


if __name__ == '__main__':
    data = T.load_mnist()
    out = dict(example_index=EXAMPLE, all=run_all(data), few=run_few(data))
    for k in ('all', 'few'):
        out[k]['summary'] = summary(out[k], out[k]['save'])
        for e, r in out[k]['summary'].items():
            print(k, e, r)
    (HERE / 'data' / 'epochs.json').write_text(json.dumps(out, separators=(',', ':')))
    print('wrote', HERE / 'data' / 'epochs.json', (HERE / 'data' / 'epochs.json').stat().st_size // 1024, 'kB')
