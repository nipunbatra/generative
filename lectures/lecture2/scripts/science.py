"""Compute every number, curve and image set in Lecture 2 from the data and the trained networks.

    python lecture2/scripts/train.py      # first: the networks (lecture2/.cache/)
    python lecture2/scripts/science.py    # then this, about 5 minutes on a laptop CPU

Writes
    data/science.json   numbers, curves and small image sets (images are base64 uint8, 784 bytes per digit)
    data/digits.json    the 1,000 display digits (100 per class, test set), for the hover scatters
    data/nets.json      decoder and encoder weights (float16) for the in-browser explorers
    models/*.pt         the three checkpoints the notebook loads
Everything is seeded; running it twice gives the same file.
"""
import base64
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import AutoEncoder, CACHE, ROOT, SIGMA, WATCH, load_mnist  # noqa: E402

torch.set_num_threads(8)
F64 = torch.float64
OUT = ROOT / 'data'
EXAMPLE = 90            # MNIST test digit #90, a handwritten 3: the lecture's running example
S = {'meta': {'sigma': SIGMA, 'example_index': EXAMPLE, 'watch': WATCH}}
IMG = {}


# ----------------------------------------------------------------------------- helpers
def rnd(v, n=4):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [rnd(x, n) for x in v]
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def put(key, imgs, signed=False):
    """Store a set of 28 x 28 images. Unsigned: values in [0, 1]. Signed: symmetric about 0, scaled by max |v|."""
    a = np.asarray(imgs.detach().cpu() if torch.is_tensor(imgs) else imgs, dtype=np.float64).reshape(-1, 784)
    rec = {'n': int(len(a))}
    if signed:
        scale = float(np.abs(a).max()) or 1.0
        a = 0.5 + 0.5 * a / scale
        rec.update(signed=True, scale=rnd(scale, 5))
    rec['data'] = base64.b64encode(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes()).decode()
    IMG[key] = rec


def f16(t):
    a = np.asarray(t.detach().cpu() if torch.is_tensor(t) else t, dtype=np.float16)
    return {'shape': list(a.shape), 'data': base64.b64encode(a.tobytes()).decode()}


def f32(t):
    a = np.asarray(t.detach().cpu() if torch.is_tensor(t) else t, dtype=np.float32)
    return {'shape': list(a.shape), 'data': base64.b64encode(a.tobytes()).decode()}


def normal_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi))


def em(Z, K, iters=100, seed=0, reg=1e-4):
    """Maximum-likelihood mixture of K full-covariance Gaussians by EM. Z: N x d (float64 tensor)."""
    N, d = Z.shape
    g = torch.Generator().manual_seed(seed)
    C0 = torch.cov(Z.T).reshape(d, d)
    eye = torch.eye(d, dtype=F64) * reg * float(torch.trace(C0)) / d
    # k-means++ style seeding: spread the starting means out
    means = [Z[torch.randint(N, (1,), generator=g)].squeeze(0)]
    for _ in range(1, K):
        dist = torch.stack([((Z - m) ** 2).sum(1) for m in means]).min(0).values
        means.append(Z[torch.multinomial(dist / dist.sum(), 1, generator=g)].squeeze(0))
    means = torch.stack(means)
    covs = C0.expand(K, d, d).clone() + eye
    logw = torch.full((K,), -math.log(K), dtype=F64)
    ll = None
    for it in range(iters):
        logp = torch.empty(K, N, dtype=F64)
        for k in range(K):
            L = torch.linalg.cholesky(covs[k])
            sol = torch.linalg.solve_triangular(L, (Z - means[k]).T, upper=False)
            logp[k] = logw[k] - 0.5 * ((sol ** 2).sum(0) + 2 * torch.log(torch.diagonal(L)).sum() + d * math.log(2 * math.pi))
        tot = torch.logsumexp(logp, 0)
        new = float(tot.mean())
        R = torch.exp(logp - tot)
        Nk = R.sum(1) + 1e-10
        logw = torch.log(Nk / N)
        means = (R @ Z) / Nk[:, None]
        for k in range(K):
            D = Z - means[k]
            covs[k] = (R[k][:, None] * D).T @ D / Nk[k] + eye
        if ll is not None and abs(new - ll) < 1e-7 * abs(ll):
            break
        ll = new
    return torch.exp(logw), means, covs, new, it + 1


def mog_logpdf(Z, w, means, covs):
    K, d = means.shape
    logp = torch.empty(K, len(Z), dtype=F64)
    for k in range(K):
        L = torch.linalg.cholesky(covs[k])
        sol = torch.linalg.solve_triangular(L, (Z - means[k]).T, upper=False)
        logp[k] = math.log(float(w[k])) - 0.5 * ((sol ** 2).sum(0) + 2 * torch.log(torch.diagonal(L)).sum() + d * math.log(2 * math.pi))
    return torch.logsumexp(logp, 0)


def mog_sample(n, w, means, covs, seed):
    g = torch.Generator().manual_seed(seed)
    ks = torch.multinomial(w, n, replacement=True, generator=g)
    eps = torch.randn(n, means.shape[1], generator=g, dtype=F64)
    Ls = torch.linalg.cholesky(covs)
    return means[ks] + torch.einsum('nij,nj->ni', Ls[ks], eps), ks


# ----------------------------------------------------------------------------- Part 2: Old Faithful
def faithful():
    rows = list(csv.DictReader(open(OUT / 'faithful.csv')))
    dur = np.array([float(r['eruptions']) for r in rows])
    wait = np.array([float(r['waiting']) for r in rows])
    n = len(wait)
    F = {'n': n, 'waiting': wait.tolist(), 'duration': dur.tolist()}
    mu, sd = wait.mean(), wait.std()
    F['gauss'] = {'mu': rnd(mu), 'sd': rnd(sd), 'avg_ll': rnd(normal_logpdf(wait, mu, sd).mean()),
                  'var': rnd(sd ** 2), 'gap_density': rnd(math.exp(normal_logpdf(mu, mu, sd)))}
    F['first5'] = wait[:5].tolist()
    # one Gaussian's density at a few waiting times, for the "score one point" slide
    F['gauss']['at'] = {str(v): rnd(math.exp(normal_logpdf(v, mu, sd)), 5) for v in (54, 66, 71, 80)}

    # a two-bump mixture, fitted by gradient descent on minus the average log-likelihood (as in Lecture 1).
    # The same code as the slide: waits in minutes, an explicit bell-curve formula, one line per term.
    x = torch.tensor(wait)
    bell = lambda x_, m_, s_: torch.exp(-((x_ - m_) / s_) ** 2 / 2) / (s_ * math.sqrt(2 * math.pi))
    mu_ = torch.tensor([65., 75.], dtype=F64, requires_grad=True)       # centres (minutes)
    log_s = torch.tensor([2.3, 2.3], dtype=F64, requires_grad=True)     # widths: sigma = exp(log_s), starts at 10
    w_ = torch.zeros(2, dtype=F64, requires_grad=True)                  # shares: pi = softmax(w), starts at (1/2, 1/2)
    opt = torch.optim.Adam([mu_, log_s, w_], lr=0.1)
    frames, keep = [], set(list(range(0, 60)) + list(range(60, 300, 4)) + [299])

    def state():
        return torch.softmax(w_.detach(), 0).numpy(), mu_.detach().numpy(), log_s.detach().exp().numpy()

    for step in range(300):
        pi_, sigma_ = torch.softmax(w_, 0), torch.exp(log_s)
        p_short = pi_[0] * bell(x, mu_[0], sigma_[0])
        p_long = pi_[1] * bell(x, mu_[1], sigma_[1])
        loss = -torch.log(p_short + p_long).mean()
        if step in keep:
            pi, mm, ss = state()
            frames.append([step, rnd(pi[1], 4), rnd(mm[0], 3), rnd(mm[1], 3), rnd(ss[0], 3), rnd(ss[1], 3), rnd(-float(loss.detach()), 4)])
        opt.zero_grad()
        loss.backward()
        opt.step()
    pi, mm, ss = state()
    order = np.argsort(mm)
    pi, mm, ss = pi[order], mm[order], ss[order]
    comp = np.stack([np.log(pi[k]) + normal_logpdf(wait, mm[k], ss[k]) for k in range(2)])
    tot = np.logaddexp(comp[0], comp[1])
    resp_long = np.exp(comp[1] - tot)
    F['mix'] = {'pi': rnd(pi), 'mu': rnd(mm), 'sd': rnd(ss), 'avg_ll': rnd(tot.mean()), 'frames': frames,
                'frame_cols': ['step', 'pi_long', 'mu_short', 'mu_long', 'sd_short', 'sd_long', 'avg_ll'],
                'resp_long': rnd(resp_long, 3)}
    F['mix']['at'] = {str(v): rnd(sum(pi[k] * math.exp(normal_logpdf(v, mm[k], ss[k])) for k in range(2)), 5) for v in (54, 66, 71, 80)}
    F['mix']['terms'] = {str(v): [float(f'{pi[k] * math.exp(normal_logpdf(v, mm[k], ss[k])):.4g}') for k in range(2)] for v in (54, 66, 80)}
    F['mix']['comp_pdf'] = {str(v): [float(f'{math.exp(normal_logpdf(v, mm[k], ss[k])):.4g}') for k in range(2)] for v in (54, 66, 80)}
    long = resp_long > 0.5
    F['mix']['dur_short'] = rnd(dur[~long].mean(), 2)
    F['mix']['dur_long'] = rnd(dur[long].mean(), 2)
    F['mix']['n_long'] = int(long.sum())
    F['ratio_likelihood'] = rnd(math.exp(n * (tot.mean() - F['gauss']['avg_ll'])), 0) if n * (tot.mean() - F['gauss']['avg_ll']) < 700 else None
    F['ll_gain_total'] = rnd(n * (tot.mean() - F['gauss']['avg_ll']), 2)

    # samples: 272 from each model
    rng = np.random.default_rng(7)
    gs = rng.normal(mu, sd, n)
    ks = (rng.random(n) < pi[1]).astype(int)
    ms = rng.normal(mm[ks], ss[ks])
    F['samples'] = {'gauss': rnd(gs, 2), 'mix': rnd(ms, 2), 'mix_k': ks.tolist()}
    F['first_draws'] = {'k': ks[:6].tolist(), 'x': rnd(ms[:6], 1)}

    # held-out: more bumps always fit the training data better
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(11)).numpy()   # the notebook's split
    tr, te = wait[idx[:200]], wait[idx[200:]]
    held = []
    for K in range(1, 7):
        best = None
        for seed in range(8):
            w_, m_, c_, ll_, _ = em(torch.tensor(tr[:, None]), K, iters=300, seed=seed, reg=1e-3)
            if best is None or ll_ > best[3]:
                best = (w_, m_, c_, ll_)
        w_, m_, c_, ll_ = best
        held.append({'K': K, 'train': rnd(ll_), 'test': rnd(float(mog_logpdf(torch.tensor(te[:, None]), w_, m_, c_).mean()))})
    F['heldout'] = held

    # two dimensions: (duration, waiting)
    X2 = torch.tensor(np.stack([dur, wait], 1))
    m2 = X2.mean(0)
    C2 = torch.cov(X2.T, correction=0)
    F['g2'] = {'mean': rnd(m2), 'cov': rnd(C2), 'corr': rnd(C2[0, 1] / torch.sqrt(C2[0, 0] * C2[1, 1]))}
    best = None
    for seed in range(5):
        r_ = em(X2, 2, iters=500, seed=seed, reg=1e-6)
        if best is None or r_[3] > best[3]:
            best = r_
    w2, mu2, cv2, ll2, _ = best
    order = torch.argsort(mu2[:, 1])
    w2, mu2, cv2 = w2[order], mu2[order], cv2[order]
    F['m2'] = {'w': rnd(w2), 'means': rnd(mu2), 'covs': rnd(cv2), 'avg_ll': rnd(ll2)}
    F['g2']['avg_ll'] = rnd(float(torch.distributions.MultivariateNormal(m2, C2).log_prob(X2).mean()))
    g = torch.Generator().manual_seed(3)
    F['samples2'] = {'gauss': rnd(m2 + torch.randn(n, 2, generator=g, dtype=F64) @ torch.linalg.cholesky(C2).T, 2),
                     'mix': rnd(mog_sample(n, w2, mu2, cv2, 4)[0], 2)}
    S['faithful'] = F


# ----------------------------------------------------------------------------- Part 3: MNIST, PCA, Gaussians
def mnist_part(data):
    Xtr = data['train'][0].double()
    ytr = data['train'][1]
    Xte = data['test'][0].double()
    yte = data['test'][1]
    x3 = Xte[EXAMPLE]
    M = {}
    # display set: 100 test digits per class, the running example first
    g = torch.Generator().manual_seed(2024)
    disp = [EXAMPLE]
    for c in range(10):
        idx = torch.nonzero(yte == c).squeeze(1)
        idx = idx[idx != EXAMPLE]
        disp += idx[torch.randperm(len(idx), generator=g)[:100 - (1 if c == 3 else 0)]].tolist()
    disp = disp[:1000]
    M['display_index'] = disp
    D = Xte[disp]
    M['display_labels'] = yte[disp].tolist()
    json.dump({'n': len(disp), 'index': disp, 'labels': yte[disp].tolist(),
               'data': base64.b64encode(np.round(D.numpy() * 255).astype(np.uint8).tobytes()).decode()},
              open(OUT / 'digits.json', 'w'))
    M['pixels3'] = rnd(x3.reshape(28, 28), 2)
    M['n_train'], M['n_test'] = len(Xtr), len(Xte)
    put('example', x3[None])
    put('first_train', Xtr[:16])
    gr = torch.Generator().manual_seed(1)
    put('uniform', torch.rand(8, 784, generator=gr, dtype=F64))

    # independent pixels: one Gaussian per pixel
    m = Xtr.mean(0)
    s = Xtr.std(0, correction=0)
    # one pixel, worked: position (14, 14) in all 60,000 training digits
    c = 14 * 28 + 14
    vals = Xtr[:, c]
    M['pixel_c'] = {'row': 14, 'col': 14, 'mean': rnd(m[c], 4), 'sd': rnd(s[c], 4),
                    'hist': torch.histc(vals.float(), bins=20, min=0, max=1).int().tolist(),
                    'frac_zero': rnd((vals == 0).double().mean(), 3), 'frac_inked': rnd((vals > 0.5).double().mean(), 3),
                    'first6': rnd(vals[:6], 2), 'labels6': ytr[:6].tolist()}
    M['pixel_corner'] = {'mean': rnd(m[0], 4), 'sd': rnd(s[0], 4)}
    M['pixel_spread_zero'] = int((s == 0).sum())
    put('pixel_mean', m[None])
    put('pixel_std', (s / s.max())[None])
    put('indep_samples', (m + s * torch.randn(8, 784, generator=gr, dtype=F64)).clamp(0, 1))
    # one 784-dimensional Gaussian with a full covariance
    C = (Xtr - m).T @ (Xtr - m) / len(Xtr)
    L = torch.linalg.cholesky(C + 1e-4 * torch.eye(784, dtype=F64))
    put('full_samples', (m + torch.randn(8, 784, generator=gr, dtype=F64) @ L.T).clamp(0, 1))
    M['params'] = {'indep': 2 * 784, 'full': 784 + 784 * 785 // 2}
    # neighbouring pixels move together: the correlation of pixel (14, 14) with its neighbours
    corr = C / torch.sqrt(torch.outer(torch.diagonal(C), torch.diagonal(C)) + 1e-12)
    centre = 14 * 28 + 14
    M['corr_centre'] = rnd(corr[centre].reshape(28, 28), 3)
    M['corr_pairs'] = {'right': rnd(corr[centre, centre + 1], 3), 'far': rnd(corr[centre, 3 * 28 + 3], 3),
                       'two_right': rnd(corr[centre, centre + 2], 3), 'five_right': rnd(corr[centre, centre + 5], 3)}

    # PCA
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    evals, W = evals[order].clamp(min=0), evecs[:, order]
    # fix the sign of each direction so the running example's coefficient is positive (cosmetic, deterministic)
    sgn = torch.sign((x3 - m) @ W)
    sgn[sgn == 0] = 1
    W = W * sgn
    cum = torch.cumsum(evals, 0) / evals.sum()
    M['pca'] = {'var_explained': rnd(cum[:200], 4), 'at': {str(d): rnd(cum[d - 1], 4) for d in (1, 2, 5, 10, 20, 50, 100, 200)},
                'eigen': rnd(evals[:10], 4)}
    put('pca_mean', m[None])
    put('pca_components', W[:, :8].T, signed=True)
    z3 = (x3 - m) @ W
    M['pca']['z3'] = rnd(z3[:10], 3)
    recon = {}
    for d in (1, 2, 5, 10, 20, 50, 100, 200, 784):
        xh = m + z3[:d] @ W[:, :d].T
        recon[str(d)] = rnd(((xh - x3) ** 2).sum(), 3)
    M['pca']['sse3'] = recon
    put('pca_recon3', torch.stack([(m + z3[:d] @ W[:, :d].T) for d in (1, 2, 5, 10, 20, 50, 100, 784)]).clamp(0, 1))
    M['pca']['recon_ds'] = [1, 2, 5, 10, 20, 50, 100, 784]
    test_sse = {}
    for d in (2, 4, 8, 16, 32, 64, 128):
        Zt = (Xte - m) @ W[:, :d]
        test_sse[str(d)] = rnd((((m + Zt @ W[:, :d].T) - Xte) ** 2).sum(1).mean(), 3)
    M['pca']['test_sse'] = test_sse
    put('pca_recon_watch_2', (m + ((Xte[WATCH] - m) @ W[:, :2]) @ W[:, :2].T).clamp(0, 1))
    put('pca_recon_watch_16', (m + ((Xte[WATCH] - m) @ W[:, :16]) @ W[:, :16].T).clamp(0, 1))
    M['pca']['display_codes2'] = rnd((D - m) @ W[:, :2], 3)
    t3 = Xtr[ytr == 3]
    m3 = t3.mean(0)
    e3, v3 = torch.linalg.eigh((t3 - m3).T @ (t3 - m3) / len(t3))
    V3 = v3[:, torch.argsort(e3, descending=True)[:2]]
    te3 = torch.nonzero(yte == 3).squeeze(1)
    z3s = (Xte[te3] - m3) @ V3
    lo3, hi3 = torch.quantile(z3s, 0.04, dim=0), torch.quantile(z3s, 0.96, dim=0)
    pick, pos = [], []
    for r in range(5):
        for c in range(7):
            target = torch.stack([lo3[0] + (hi3[0] - lo3[0]) * c / 6, hi3[1] - (hi3[1] - lo3[1]) * r / 4])
            j = int(((z3s - target) ** 2).sum(1).argmin())
            if j not in pick:
                pick.append(j); pos.append([c, r])
    put('threes_map', Xte[te3[pick]])
    M['threes_map'] = {'pos': pos, 'n_threes_train': int(len(t3)),
                       'var2': rnd((e3.sort(descending=True).values[:2].sum() / e3.sum()), 3)}
    M['pca']['basis'] = f32(W[:, :100].T.contiguous())   # 100 x 784, float32 so the live slider matches these numbers
    M['pca']['mean'] = f32(m)
    M['pca']['z3_100'] = rnd(z3[:100], 4)

    # Gaussian, then mixtures, on the first d PCA numbers; sample, then decode with inverse PCA
    gens = {}
    for d in (2, 10, 20, 50):
        Ztr = (Xtr - m) @ W[:, :d]
        for K in (1, 5, 10, 20):
            sub = Ztr if d <= 10 else Ztr[:30000]
            w, mus, covs, ll, its = em(sub, K, iters=150 if d <= 10 else 80, seed=d * 100 + K)
            zs, ks = mog_sample(16, w, mus, covs, seed=7)
            xs = (m + zs @ W[:, :d].T).clamp(0, 1)
            key = f'd{d}k{K}'
            put('gen_' + key, xs)
            if (d, K) == (2, 1):
                # one sample drawn by hand: z = mean + L eps, then x = m + z1 w1 + z2 w2
                eps = torch.tensor([0.8, -0.5], dtype=F64)
                L = torch.linalg.cholesky(covs[0])
                z = mus[0] + L @ eps
                M['worked_gen'] = {'eps': rnd(eps, 2), 'z': rnd(z, 3), 'mean_z': rnd(mus[0], 3), 'sd_z': rnd(torch.sqrt(torch.diagonal(covs[0])), 3),
                                   'corr_z': rnd(covs[0][0, 1] / torch.sqrt(covs[0][0, 0] * covs[0][1, 1]), 3)}
                put('worked_gen', (m + z @ W[:, :2].T).clamp(0, 1)[None])
                put('worked_terms', torch.stack([z[0] * W[:, 0], z[1] * W[:, 1]]), signed=True)
                M['codes2_train'] = rnd(((Xtr[:3000] - m) @ W[:, :2]), 2)
                M['gauss2'] = {'mean': rnd(mus[0], 3), 'cov': rnd(covs[0], 3)}
            if (d, K) == (2, 10):
                M['mog2'] = {'w': rnd(w, 3), 'means': rnd(mus, 3), 'covs': rnd(covs, 3)}
            held = float(mog_logpdf((Xte - m) @ W[:, :d], w, mus, covs).mean())
            gens[key] = {'d': d, 'K': K, 'train_ll': rnd(ll), 'test_ll': rnd(held), 'iters': its,
                         'params': int(K - 1 + K * d + K * d * (d + 1) // 2),
                         'w': f32(w), 'means': f32(mus), 'chol': f32(torch.linalg.cholesky(covs))}
            print('mog', key, round(ll, 3), round(held, 3), its, flush=True)
    M['gens'] = gens
    S['mnist'] = M
    return m, W


# ----------------------------------------------------------------------------- networks
def load(name):
    ck = torch.load(CACHE / f'{name}.pt', weights_only=False)
    cfg = ck['config']
    net = AutoEncoder(cfg['z'], cfg.get('variational', False), cfg.get('linear', False))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    hist = json.load(open(CACHE / f'{name}.json'))
    return net, hist


@torch.no_grad()
def enc(net, X):
    mu, logvar = net.encode(X.float())
    return mu.double(), (logvar.double() if logvar is not None else None)


@torch.no_grad()
def dec(net, Z):
    return net.decoder(Z.float()).double()


def pair_distances(codes, labels):
    codes = torch.as_tensor(codes, dtype=F64)
    labels = torch.as_tensor(labels)
    z0, z1 = codes[labels == 0], codes[labels == 1]
    d00 = torch.cdist(z0, z0)
    iu = torch.triu_indices(len(z0), len(z0), 1)
    same = float(d00[iu[0], iu[1]].mean())
    diff = float(torch.cdist(z0, z1).mean())
    return {'same0': rnd(same, 3), 'zero_one': rnd(diff, 3), 'ratio': rnd(same / diff, 3) if diff > 1e-9 else None}


def networks(data, m, W):
    Xtr = data['train'][0].double()
    Xte = data['test'][0].double()
    yte = data['test'][1]
    disp = S['mnist']['display_index']
    D, Dy = Xte[disp], yte[disp]
    x3 = Xte[EXAMPLE]
    A = {}
    # ---- training an autoencoder with a 2-number code
    net2, h2 = load('ae2')
    A['init_test_sse'] = rnd(h2['test_sse_init'], 2)
    A['init_mean_pixel'] = None
    frames = [('0', 0)] + [(k, int(k)) for k in sorted(h2['step_snapshots'], key=int)] + \
             [(k, int(k) * h2['steps_per_epoch']) for k in sorted(h2['snapshots'], key=int) if k != '0']
    imgs = []
    labels = []
    for k, step in frames:
        snap = h2['snapshots']['0'] if k == '0' else (h2['step_snapshots'][k] if step < h2['steps_per_epoch'] and k in h2['step_snapshots'] else h2['snapshots'][k])
        imgs += snap
        labels.append(step)
    put('ae2_train_frames', imgs)
    A['ae2_frames_steps'] = labels
    A['ae2_frames_sse'] = [rnd(h2['step_test_sse'][str(st)], 2) for st in labels]
    A['ae2_frames_digit_sse'] = [h2['step_digit_sse'][str(st)] for st in labels]
    dense = sorted(h2['step_test_sse'], key=int)
    A['ae2_step_curve'] = {'step': [int(k) for k in dense], 'test_sse': [rnd(h2['step_test_sse'][k], 2) for k in dense]}
    A['ae2_init_pixel_mse'] = rnd(h2['test_sse_init'] / 784, 4)
    first = np.array(h2['snapshots']['0'])
    A['init_pixel_range'] = [rnd(first.min(), 3), rnd(first.max(), 3)]
    A['ae2_curve'] = {'epoch': h2['epoch'], 'train_sse': rnd(h2['train_sse'], 3), 'test_sse': rnd(h2['test_sse'], 3)}
    A['steps_per_epoch'] = h2['steps_per_epoch']
    A['seconds_ae2'] = rnd(h2['seconds'], 1)
    put('watch', Xte[WATCH])
    with torch.no_grad():
        xh3 = dec(net2, enc(net2, x3[None])[0])[0]
    A['ae2_sse3'] = rnd(((xh3 - x3) ** 2).sum(), 2)
    A['ae2_z3'] = rnd(enc(net2, x3[None])[0][0], 3)
    put('ae2_recon3', xh3[None])
    # the squared error of the 3 under the untrained network, pixel by pixel
    A['sse3_pca2'] = S['mnist']['pca']['sse3']['2']

    # ---- bottleneck sizes
    sweep = {}
    for z in (2, 4, 8, 16, 32, 64, 128):
        net, h = load(f'ae{z}')
        with torch.no_grad():
            tr_sse = float(torch.cat([((dec(net, enc(net, Xtr[i:i + 10000])[0]) - Xtr[i:i + 10000]) ** 2).sum(1) for i in range(0, len(Xtr), 10000)]).mean())
        sweep[str(z)] = {'train_sse': rnd(tr_sse, 3), 'test_sse': rnd(h['test_sse'][-1], 3),
                         'curve': rnd(h['test_sse'], 3), 'train_curve': rnd(h['train_sse'], 3),
                         'params': int(sum(p.numel() for p in net.parameters())),
                         'pca_test_sse': S['mnist']['pca']['test_sse'][str(z)]}
        put(f'ae{z}_recon_watch', dec(net, enc(net, Xte[WATCH])[0]))
        codes = enc(net, D)[0]
        sweep[str(z)]['dist'] = pair_distances(codes, Dy)
        sweep[str(z)]['code_sd'] = rnd(float(enc(net, Xte[:5000])[0].std(0).mean()), 3)
        g = torch.Generator().manual_seed(5)
        put(f'ae{z}_prior_samples', dec(net, torch.randn(16, z, generator=g, dtype=F64)))
    A['sweep'] = sweep
    # ---- the linear autoencoder and PCA
    lin, hl = load('lin2')
    A['lin2_test_sse'] = rnd(hl['test_sse'][-1], 3)
    with torch.no_grad():
        put('lin2_recon_watch', lin.decoder(lin.encoder(Xte[WATCH].float())).double().clamp(0, 1))
    # angle between the linear autoencoder's decoder directions and the first two PCA directions
    Wd = lin.decoder.weight.detach().double()           # 784 x 2
    Q, _ = torch.linalg.qr(Wd)
    sv = torch.linalg.svdvals(Q.T @ W[:, :2])
    A['lin2_cos_angles'] = rnd(sv, 4)
    # ---- 2-number codes of the display set
    c2 = enc(net2, D)[0]
    A['ae2_codes'] = rnd(c2, 3)
    Dl = torch.as_tensor(S['mnist']['display_labels'])
    zeros, ones = torch.nonzero(Dl == 0).squeeze(1), torch.nonzero(Dl == 1).squeeze(1)
    A['spread_example'] = {'zero_a': int(zeros[0]), 'zero_b': int(zeros[1]), 'one': int(ones[0])}
    lo, hi = torch.quantile(c2, 0.01, dim=0), torch.quantile(c2, 0.99, dim=0)
    A['ae2_box'] = rnd(torch.stack([lo, hi]), 3)
    # ---- denoising
    dae, hd = load('dae16')
    ae16, _ = load('ae16')
    g = torch.Generator().manual_seed(12)
    xw = Xte[WATCH]
    damaged = xw * (torch.rand(xw.shape, generator=g, dtype=F64) > 0.5)
    put('damaged_watch', damaged)
    put('dae16_clean', dec(dae, enc(dae, damaged)[0]))
    put('ae16_on_damaged', dec(ae16, enc(ae16, damaged)[0]))
    A['dae16_test_sse_damaged'] = rnd(float(((dec(dae, enc(dae, damaged)[0]) - xw) ** 2).sum(1).mean()), 2)
    A['ae16_test_sse_damaged'] = rnd(float(((dec(ae16, enc(ae16, damaged)[0]) - xw) ** 2).sum(1).mean()), 2)
    # ---- search by code: nearest training digits to a query, by 16-number code
    q = x3
    ctr = enc(ae16, Xtr)[0]
    cq = enc(ae16, q[None])[0]
    nn_code = torch.cdist(cq, ctr)[0].argsort()[:5]
    put('search_query', q[None])
    put('search_code', Xtr[nn_code])
    S['ae'] = A

    # ---- Part 5: can it generate?
    G = {}
    g = torch.Generator().manual_seed(6)
    G['prior_points'] = rnd(torch.randn(500, 2, generator=g, dtype=F64), 3)
    G['frac_prior_in_box'] = None
    # hallucination grid over the codes' box
    def grid(net, lo, hi, n=12):
        xs = torch.linspace(float(lo[0]), float(hi[0]), n, dtype=F64)
        ys = torch.linspace(float(hi[1]), float(lo[1]), n, dtype=F64)
        Z = torch.stack(torch.meshgrid(ys, xs, indexing='ij')[::-1], -1).reshape(-1, 2)
        return dec(net, Z)
    pad = (hi - lo) * 0.25
    glo, ghi = lo - pad, hi + pad
    put('ae2_grid', grid(net2, glo, ghi))
    G['ae2_grid_box'] = rnd(torch.stack([glo, ghi]), 3)
    # interpolation between a 4 and a 9
    i4 = int(torch.nonzero(yte == 1)[0])    # a 1 ...
    i9 = int(torch.nonzero(yte == 0)[0])    # ... and a 0
    G['interp_index'] = [i4, i9]
    put('interp_ends', Xte[[i4, i9]])
    for name in ('ae2', 'ae16'):
        net, _ = load(name)
        za, zb = enc(net, Xte[[i4]])[0], enc(net, Xte[[i9]])[0]
        t = torch.linspace(0, 1, 9, dtype=F64)[:, None]
        put(f'{name}_interp', dec(net, za + t * (zb - za)))
    # fit mixtures to where the codes actually are, then sample (a density fitted after training)
    post = {}
    for name, K in (('ae2', 10), ('ae16', 10)):
        net, _ = load(name)
        ctr = enc(net, Xtr)[0]
        w, mus, covs, ll, its = em(ctr, K, iters=200, seed=31)
        zs, _ = mog_sample(16, w, mus, covs, seed=8)
        put(f'{name}_mog_samples', dec(net, zs))
        post[name] = {'K': K, 'train_ll': rnd(ll), 'iters': its}
        if name == 'ae2':
            G['ae2_mog'] = {'w': rnd(w), 'means': rnd(mus), 'covs': rnd(covs)}
    G['posthoc'] = post
    S['gen'] = G
    return net2


def vae_part(data, m, W):
    Xtr = data['train'][0].double()
    Xte = data['test'][0].double()
    yte = data['test'][1]
    disp = S['mnist']['display_index']
    D, Dy = Xte[disp], yte[disp]
    x3 = Xte[EXAMPLE]
    V = {}
    net, h = load('vae2')
    V['curve'] = {'epoch': h['epoch'], 'recon': rnd([s / (2 * SIGMA ** 2) for s in h['train_sse']], 3),
                  'kl': rnd(h['train_kl'], 3), 'loss': rnd(h['train_loss'], 3), 'test_sse': rnd(h['test_sse'], 3)}
    V['test_sse'] = rnd(h['test_sse'][-1], 3)
    V['test_kl'] = rnd(h['test_kl'][-1], 3)
    const = 784 * math.log(SIGMA * math.sqrt(2 * math.pi))
    V['loglik_const'] = rnd(const, 3)

    def log_lik(Z):   # log p(x3 | z) for a batch of codes
        return -((dec(net, Z) - x3) ** 2).sum(1) / (2 * SIGMA ** 2) - const

    # the encoder's q for our 3
    mu3, lv3 = enc(net, x3[None])
    mu3, sd3 = mu3[0], torch.exp(0.5 * lv3[0])
    V['q3'] = {'mu': rnd(mu3, 4), 'sd': rnd(sd3, 4)}
    kl3 = 0.5 * float((mu3 ** 2 + sd3 ** 2 - 2 * torch.log(sd3) - 1).sum())
    g = torch.Generator().manual_seed(21)
    eps = torch.randn(20000, 2, generator=g, dtype=F64)
    ll_q = log_lik(mu3 + sd3 * eps)
    V['q3']['kl_prior'] = rnd(kl3, 4)
    V['q3']['exp_loglik'] = rnd(float(ll_q.mean()), 3)
    V['q3']['exp_sse'] = rnd(float(((dec(net, (mu3 + sd3 * eps[:2000])) - x3) ** 2).sum(1).mean()), 3)
    V['q3']['elbo'] = rnd(float(ll_q.mean()) - kl3, 3)
    # one draw of eps, written out (the worked reparameterisation example)
    e1 = torch.tensor([0.5, -1.0], dtype=F64)
    z1 = mu3 + sd3 * e1
    xh1 = dec(net, z1[None])[0]
    V['worked'] = {'eps': rnd(e1, 2), 'z': rnd(z1, 4), 'sse': rnd(((xh1 - x3) ** 2).sum(), 3),
                   'recon': rnd(((xh1 - x3) ** 2).sum() / (2 * SIGMA ** 2), 3)}
    V['worked']['loss'] = rnd(float(V['worked']['recon']) + kl3, 3)
    put('vae2_worked_recon', xh1[None])
    put('vae2_mean_recon3', dec(net, mu3[None]))
    # the exact posterior on a grid (two numbers: we can afford it), and the evidence by adding rectangles.
    # Wide view: the whole prior, [-4, 4]^2. Zoomed view: a window around the posterior, fine enough to resolve it.
    def grid_post(cx, cy, half, n):
        ax_x = torch.linspace(cx - half, cx + half, n, dtype=F64)
        ax_y = torch.linspace(cy - half, cy + half, n, dtype=F64)
        Zg = torch.stack(torch.meshgrid(ax_x, ax_y, indexing='xy'), -1).reshape(-1, 2)
        lj = torch.cat([log_lik(Zg[i:i + 20000]) for i in range(0, len(Zg), 20000)]) - 0.5 * (Zg ** 2).sum(1) - math.log(2 * math.pi)
        cell = float((ax_x[1] - ax_x[0]) * (ax_y[1] - ax_y[0]))
        return Zg, lj, float(torch.logsumexp(lj, 0)) + math.log(cell), cell
    n = 161
    Zw, ljw, logpx_wide, cellw = grid_post(0.0, 0.0, 4.0, n)
    post_w = torch.exp(ljw - ljw.max()).reshape(n, n)
    V['grid'] = {'n': n, 'lim': 4, 'log_px_wide': rnd(logpx_wide, 3), 'post81': rnd(post_w[::2, ::2].reshape(-1), 3)}
    half = 0.15
    cx, cy = float(mu3[0]), float(mu3[1])
    Zz, ljz, logpx, cellz = grid_post(cx, cy, half, 241)
    pw = torch.softmax(ljz, 0)       # probability of each cell
    pm = (pw[:, None] * Zz).sum(0)
    pc = (pw[:, None] * (Zz - pm)).T @ (Zz - pm)
    zz = torch.exp(ljz - ljz.max()).reshape(241, 241)
    V['grid'].update({'log_px': rnd(logpx, 3), 'zoom': {'cx': rnd(cx, 4), 'cy': rnd(cy, 4), 'half': half, 'n': 81,
                      'post': rnd(zz[::3, ::3].reshape(-1), 3)},
                      'post_mean': rnd(pm, 4), 'post_sd': rnd(torch.sqrt(torch.diagonal(pc)), 4),
                      'post_corr': rnd(pc[0, 1] / torch.sqrt(pc[0, 0] * pc[1, 1]), 3)})
    # check the evidence a second way: importance sampling with a widened q as the proposal
    g = torch.Generator().manual_seed(24)
    prop_sd = 3 * sd3
    zi = mu3 + prop_sd * torch.randn(100000, 2, generator=g, dtype=F64)
    logq = torch.distributions.Normal(mu3, prop_sd).log_prob(zi).sum(1)
    lw = torch.cat([log_lik(zi[i:i + 20000]) for i in range(0, len(zi), 20000)]) - 0.5 * (zi ** 2).sum(1) - math.log(2 * math.pi) - logq
    V['grid']['log_px_is'] = rnd(float(torch.logsumexp(lw, 0)) - math.log(len(zi)), 3)
    V['q3']['gap'] = rnd(logpx - float(V['q3']['elbo']), 3)
    # how much of the prior gives this 3 a reasonable likelihood: naive Monte Carlo for p(x)
    g = torch.Generator().manual_seed(22)
    mc = []
    for S_ in (1, 10, 100, 1000, 10000, 100000):
        ests = []
        for rep in range(30):
            zz = torch.randn(S_, 2, generator=g, dtype=F64)
            ests.append(float(torch.logsumexp(log_lik(zz), 0)) - math.log(S_))
        e = np.array(ests)
        mc.append({'S': S_, 'median': rnd(np.median(e), 3), 'lo': rnd(np.percentile(e, 10), 3), 'hi': rnd(np.percentile(e, 90), 3),
                   'min': rnd(e.min(), 3), 'max': rnd(e.max(), 3)})
    V['naive_mc'] = mc
    # share of the evidence carried by the best 1% of 100,000 prior draws
    zz = torch.randn(100000, 2, generator=g, dtype=F64)
    llz = log_lik(zz)
    wgt = torch.softmax(llz, 0)
    V['top1pct_share'] = rnd(float(wgt.sort(descending=True).values[:1000].sum()), 4)
    V['top10_share'] = rnd(float(wgt.sort(descending=True).values[:10].sum()), 4)
    # variational inference for this one image: fit q = N(m, diag s^2) by maximising the ELBO
    torch.manual_seed(23)
    mq = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    ls = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([mq, ls], lr=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0 if t < 800 else 0.2)
    x3f = x3.float()
    frames = []
    for step in range(1201):
        e = torch.randn(64, 2)
        zq = mq + ls.exp() * e
        recon = -((net.decoder(zq) - x3f) ** 2).sum(1) / (2 * SIGMA ** 2) - const
        kl = 0.5 * (mq ** 2 + (2 * ls).exp() - 2 * ls - 1).sum()
        elbo = recon.mean() - kl
        if step % 16 == 0 or step < 12 or (step < 100 and step % 4 == 0):
            frames.append([step, rnd(mq[0], 4), rnd(mq[1], 4), rnd(ls[0].exp(), 4), rnd(ls[1].exp(), 4), rnd(elbo, 2)])
        opt.zero_grad()
        (-elbo).backward()
        opt.step()
        sched.step()
    mqd, sqd = mq.detach().double(), ls.detach().double().exp()
    llf = log_lik(mqd + sqd * eps)
    klf = 0.5 * float((mqd ** 2 + sqd ** 2 - 2 * torch.log(sqd) - 1).sum())
    V['vi3'] = {'frames': frames, 'cols': ['step', 'm1', 'm2', 's1', 's2', 'elbo'], 'final_elbo': rnd(float(llf.mean()) - klf, 3),
                'final_mu': rnd(mqd, 4), 'final_sd': rnd(sqd, 4)}
    # samples from the prior, through the decoder
    g = torch.Generator().manual_seed(5)
    zp = torch.randn(16, 2, generator=g, dtype=F64)
    V['prior_z'] = rnd(zp[:4], 3)
    put('vae2_prior_samples', dec(net, zp))
    gn = torch.Generator().manual_seed(55)
    put('vae2_noisy_sample', (dec(net, zp[:1]) + SIGMA * torch.randn(1, 784, generator=gn, dtype=F64)).clamp(0, 1))
    net16, h16 = load('vae16')
    V['vae16'] = {'test_sse': rnd(h16['test_sse'][-1], 3), 'test_kl': rnd(h16['test_kl'][-1], 3)}
    g = torch.Generator().manual_seed(5)
    put('vae16_prior_samples', dec(net16, torch.randn(16, 16, generator=g, dtype=F64)))
    put('vae2_recon_watch', dec(net, enc(net, Xte[WATCH])[0]))
    put('vae16_recon_watch', dec(net16, enc(net16, Xte[WATCH])[0]))
    c2, lv = enc(net, D)
    V['codes'] = rnd(c2, 3)
    V['code_sd_mean'] = rnd(float(torch.exp(0.5 * lv).mean()), 4)
    # where do the codes of all 10,000 test digits sit? (the aggregate posterior should look like the prior)
    ca = enc(net, Xte)[0]
    V['agg'] = {'mean': rnd(ca.mean(0), 3), 'sd': rnd(ca.std(0), 3)}
    V['grid_box'] = [[-3, -3], [3, 3]]
    xs = torch.linspace(-3, 3, 12, dtype=F64)
    ys = torch.linspace(3, -3, 12, dtype=F64)
    Zg = torch.stack(torch.meshgrid(ys, xs, indexing='ij')[::-1], -1).reshape(-1, 2)
    put('vae2_grid', dec(net, Zg))
    i4, i9 = S['gen']['interp_index']
    for name, nt in (('vae2', net), ('vae16', net16)):
        za, zb = enc(nt, Xte[[i4]])[0], enc(nt, Xte[[i9]])[0]
        t = torch.linspace(0, 1, 9, dtype=F64)[:, None]
        put(f'{name}_interp', dec(nt, za + t * (zb - za)))
    V['dist'] = pair_distances(c2, Dy)
    # ---- the KL weight beta
    sweep = {}
    for beta, name in ((0.0, 'vae2-b0'), (0.1, 'vae2-b0.1'), (1.0, 'vae2'), (4.0, 'vae2-b4'), (16.0, 'vae2-b16'), (64.0, 'vae2-b64'), (256.0, 'vae2-b256')):
        nt, hb = load(name)
        cb, lvb = enc(nt, D)
        key = f'{beta:g}'
        sweep[key] = {'beta': beta, 'test_sse': rnd(hb['test_sse'][-1], 3), 'test_kl': rnd(hb['test_kl'][-1], 3),
                      'sd_mean': rnd(float(torch.exp(0.5 * lvb).mean()), 4), 'codes': rnd(cb, 3), 'dist': pair_distances(cb, Dy),
                      'code_spread': rnd(float(cb.std(0).mean()), 3)}
        put(f'beta{key}_recon_watch', dec(nt, enc(nt, Xte[WATCH])[0]))
        g = torch.Generator().manual_seed(5)
        put(f'beta{key}_prior_samples', dec(nt, torch.randn(16, 2, generator=g, dtype=F64)))
    V['beta'] = sweep
    # ---- real or generated? six real test digits and six samples from the 16-number VAE
    g = torch.Generator().manual_seed(77)
    pool = torch.tensor([i for i in range(len(Xte)) if i not in set(disp)])
    real = pool[torch.randperm(len(pool), generator=g)[:6]]
    g2 = torch.Generator().manual_seed(78)
    fake = dec(net16, torch.randn(6, 16, generator=g2, dtype=F64))
    order = torch.randperm(12, generator=g).tolist()
    allx = torch.cat([Xte[real], fake])[order]
    put('realfake', allx)
    V['realfake_is_fake'] = [i >= 6 for i in order]
    S['vae'] = V

    # ---- weights for the in-browser explorers
    nets = {}
    ae2, _ = load('ae2')
    for name, nt in (('ae2', ae2), ('vae2', net)):
        nets[name] = {'dec': [f16(nt.decoder[0].weight), f16(nt.decoder[0].bias), f16(nt.decoder[2].weight), f16(nt.decoder[2].bias)],
                      'enc': [f16(nt.encoder[0].weight), f16(nt.encoder[0].bias), f16(nt.encoder[2].weight[:2]), f16(nt.encoder[2].bias[:2])]}
    json.dump(nets, open(OUT / 'nets.json', 'w'))
    for name in ('ae2', 'vae2', 'vae16'):
        shutil.copy(CACHE / f'{name}.pt', ROOT / 'models' / f'{name}.pt')


if __name__ == '__main__':
    data = load_mnist()
    faithful()
    print('faithful done', flush=True)
    m, W = mnist_part(data)
    print('mnist done', flush=True)
    networks(data, m, W)
    print('networks done', flush=True)
    vae_part(data, m, W)
    S['img'] = IMG
    json.dump(S, open(OUT / 'science.json', 'w'), separators=(',', ':'))
    print('wrote', OUT / 'science.json', round((OUT / 'science.json').stat().st_size / 1e6, 2), 'MB')
