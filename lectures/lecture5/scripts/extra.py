"""Dataset statistics the slides use: the grey-level histogram of every MNIST training pixel, and average images.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/extra.py            # writes data/extra.json, figures/mnist-mean-3.png, figures/mnist-mean-all.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist import grid, load_mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    x, y = load_mnist(True)
    levels = np.round((x.numpy().ravel() + 1) * 127.5).astype(int)
    hist = np.bincount(levels, minlength=256) / levels.size
    threes = x[y == 3]
    grid([threes.mean(0)], 1, "mnist-mean-3.png")
    grid([x.mean(0)], 1, "mnist-mean-all.png")
    out = {"pixel_levels": np.round(hist, 7).tolist(), "n_pixels": int(levels.size), "n_threes": int(len(threes)),
           "share_exact_paper": round(float(hist[0]), 4), "share_paper_like": round(float(hist[:26].sum()), 4),
           "share_ink_like": round(float(hist[230:].sum()), 4)}
    (ROOT / "data" / "extra.json").write_text(json.dumps(out))
    print({k: v for k, v in out.items() if k != "pixel_levels"})


if __name__ == "__main__":
    main()
