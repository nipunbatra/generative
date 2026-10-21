"""The one-pixel example: a pixel is paper (-1) or ink (+1), so every diffusion quantity is exact.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/pixel.py            # writes data/pixel.json

The pixel is the one of our 3 (MNIST test image 90) chosen below; pi is how often it is ink across the
60,000 training images ("ink" = grey level above the middle). Everything else follows from Bayes' rule
and Gaussian densities; math.mjs implements the same formulas for the live figures, and
scripts/check-science.mjs compares the two.
"""
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MNIST_DIR", "/Users/nipun/git/dl-teaching/generative-modelling-ii/work/data"))
T = 1000
beta = np.r_[0.0, np.linspace(1e-4, 0.02, T)]
alpha = 1 - beta
ab = np.cumprod(alpha)
btil = np.r_[0.0, beta[1:] * (1 - ab[:-1]) / (1 - ab[1:])]
r4 = lambda v: float(round(float(v), 4))
npdf = lambda x, m, v: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def p_ink(xt, t, pi):
    """Bayes: P(x0 = +1 | x_t) from the two likelihoods N(x_t; +-sqrt(ab), 1 - ab) and the prior pi.

    The log of the likelihood ratio is 2 sqrt(ab) x_t / (1 - ab), so the posterior is a sigmoid of x_t."""
    logit = math.log(pi / (1 - pi)) + 2 * math.sqrt(ab[t]) * np.asarray(xt) / (1 - ab[t])
    return 1 / (1 + np.exp(-logit))


def mu_tilde(xt, x0, t):
    """Mean of q(x_{t-1} | x_t, x0)."""
    return (math.sqrt(ab[t - 1]) * beta[t] * x0 + math.sqrt(alpha[t]) * (1 - ab[t - 1]) * xt) / (1 - ab[t])


def main():
    x = np.frombuffer(gzip.decompress((DATA / "train-images-idx3-ubyte.gz").read_bytes()), np.uint8, offset=16).reshape(-1, 28, 28)
    te = np.frombuffer(gzip.decompress((DATA / "t10k-images-idx3-ubyte.gz").read_bytes()), np.uint8, offset=16).reshape(-1, 28, 28)
    held = te[90]
    ink = (x > 127).mean(0)
    # the pixel: ink in our 3, and ink in roughly a third of all training images
    cands = [(abs(ink[r, c] - 0.35), r, c) for r in range(28) for c in range(28) if held[r, c] > 230]
    _, R, C = min(cands)
    pi = float(ink[R, C])
    vals = x[:, R, C] / 127.5 - 1
    out = {"pixel": [R, C], "pi": r4(pi), "share_paper": r4((x[:, R, C] < 26).mean()), "share_ink": r4((x[:, R, C] > 229).mean()),
           "share_between": r4(((x[:, R, C] >= 26) & (x[:, R, C] <= 229)).mean()), "held_value": r4(held[R, C] / 127.5 - 1),
           "hist": np.histogram(vals, bins=20, range=(-1, 1))[0].tolist()}

    # schedule landmarks
    out["schedule"] = {t: {"beta": r4(beta[t]), "alpha_bar": r4(ab[t]), "signal": r4(math.sqrt(ab[t])), "noise": r4(math.sqrt(1 - ab[t])),
                           "snr": float(f"{ab[t] / (1 - ab[t]):.4g}")} for t in [1, 50, 100, 250, 500, 600, 750, 1000]}
    out["sum_beta"] = r4(beta.sum())

    # 1.2: noising without the shrink lets the spread grow; with it, the variance stays 1
    out["explode"] = {"var_naive_T": r4(1 + beta.sum()), "sd_naive_T": r4(math.sqrt(1 + beta.sum())), "var_shrunk_T": 1.0}

    # 1.5: two big illustrative steps, beta1 = 0.1 and beta2 = 0.2, from x0 = 1 with e1 = 0.3, e2 = -0.5
    b1, b2, e1, e2, x0 = 0.1, 0.2, 0.3, -0.5, 1.0
    x1 = math.sqrt(1 - b1) * x0 + math.sqrt(b1) * e1
    x2 = math.sqrt(1 - b2) * x1 + math.sqrt(b2) * e2
    a2 = (1 - b1) * (1 - b2)
    comb = (math.sqrt(1 - b2) * math.sqrt(b1) * e1 + math.sqrt(b2) * e2) / math.sqrt(1 - a2)
    out["two_steps"] = {"beta1": b1, "beta2": b2, "e1": e1, "e2": e2, "x1": r4(x1), "x2": r4(x2), "alpha_bar2": r4(a2),
                        "signal": r4(math.sqrt(a2)), "noise": r4(math.sqrt(1 - a2)), "eps": r4(comb),
                        "check": r4(math.sqrt(a2) * x0 + math.sqrt(1 - a2) * comb),
                        "noise_var_parts": [r4((1 - b2) * b1), r4(b2)], "noise_var": r4((1 - b2) * b1 + b2)}

    # the running example: x0 = +1 (ink), t = 250, eps = -0.5
    t, eps = 250, -0.5
    s, v = math.sqrt(ab[t]), 1 - ab[t]
    xt = s * x0 + math.sqrt(v) * eps
    li, lp = npdf(xt, s, v), npdf(xt, -s, v)
    P = p_ink(xt, t, pi)
    m0 = P - (1 - P)                                   # E[x0 | x_t]
    e_star = (xt - s * m0) / math.sqrt(v)              # the best possible noise prediction E[eps | x_t]
    mt_true = mu_tilde(xt, x0, t)                      # the step back if we knew x0
    mt_hat = mu_tilde(xt, m0, t)                       # ... using the best guess of x0 instead
    mean_eps = (xt - beta[t] / math.sqrt(v) * e_star) / math.sqrt(alpha[t])
    z = 0.8
    x_prev = mt_hat + math.sqrt(btil[t]) * z
    kl = (mt_true - mt_hat) ** 2 / (2 * btil[t])
    w_t = beta[t] ** 2 / (2 * btil[t] * alpha[t] * v)  # KL = w_t (eps - eps_hat)^2
    score = -e_star / math.sqrt(v)
    out["run"] = {"x0": x0, "t": t, "eps": eps, "alpha_bar": r4(ab[t]), "signal": r4(s), "noise": r4(math.sqrt(v)), "var": r4(v),
                  "xt": r4(xt), "lik_ink": r4(li), "lik_paper": r4(lp), "prior_ink": r4(pi), "post_ink": r4(P), "post_paper": r4(1 - P),
                  "x0_hat": r4(m0), "eps_star": r4(e_star), "beta": r4(beta[t]), "alpha": r4(alpha[t]), "alpha_bar_prev": r4(ab[t - 1]),
                  "beta_tilde": float(f"{btil[t]:.6g}"), "sigma": r4(math.sqrt(btil[t])),
                  "coef_x0": r4(math.sqrt(ab[t - 1]) * beta[t] / v), "coef_xt": r4(math.sqrt(alpha[t]) * (1 - ab[t - 1]) / v),
                  "mu_true": r4(mt_true), "mu_hat": r4(mt_hat), "mu_from_eps": r4(mean_eps), "z": z, "x_prev": r4(x_prev),
                  "kl": float(f"{kl:.6g}"), "weight": r4(w_t), "kl_check": float(f"{w_t * (eps - e_star) ** 2:.6g}"),
                  "score": r4(score), "score_conditional": r4(-eps / math.sqrt(v))}

    # 2.4: the exact reverse density q(x_s | x_t) for jumps of several sizes, as a mixture of two Gaussians
    def reverse(xt, t, s):
        a_ts = ab[t] / ab[s]
        var = (1 - ab[s]) * (1 - a_ts) / (1 - ab[t]) if s > 0 else 0.0
        means = [(math.sqrt(ab[s]) * (1 - a_ts) * k + math.sqrt(a_ts) * (1 - ab[s]) * xt) / (1 - ab[t]) for k in (1, -1)]
        P = p_ink(xt, t, pi)
        mix_mean = P * means[0] + (1 - P) * means[1]
        mix_var = var + P * (1 - P) * (means[0] - means[1]) ** 2
        return {"s": s, "weights": [r4(P), r4(1 - P)], "means": [r4(m) for m in means], "sd": r4(math.sqrt(var)),
                "fit_mean": r4(mix_mean), "fit_sd": r4(math.sqrt(mix_var)), "separation": r4(abs(means[0] - means[1]) / max(math.sqrt(var), 1e-12))}
    out["reverse"] = [reverse(0.0, 500, s) for s in (0, 100, 300, 450, 490, 499)]

    # 3.5: the loss cannot reach zero: E[(eps - E[eps | x_t])^2] for each t, by numerical integration over x_t
    grid = np.linspace(-6, 6, 4001)
    irr = []
    for t in [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        s, v = math.sqrt(ab[t]), 1 - ab[t]
        P = p_ink(grid, t, pi)
        dens = pi * npdf(grid, s, v) + (1 - pi) * npdf(grid, -s, v)
        # Var(eps | x_t) = ab / (1 - ab) * Var(x0 | x_t), and Var(x0 | x_t) = 1 - E[x0 | x_t]^2 for x0 = +-1
        var_eps = ab[t] / v * (1 - (2 * P - 1) ** 2)
        irr.append([t, r4(np.trapezoid(var_eps * dens, grid))])
    out["irreducible"] = irr
    (ROOT / "data" / "pixel.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pixel", "pi", "share_paper", "share_ink", "held_value")}), json.dumps(out["run"], indent=0))


if __name__ == "__main__":
    main()
