"""Export what the in-browser latent-space app needs (deck/widgets.js, data-widget="latent").

    python scripts/export_explorer.py

For each 2-D model (the autoencoder ae2 and the VAE vae2):
  figures/data/explorer-<model>.bin   decoder weights, float32, in the order the browser multiplies them
  figures/data/explorer-<model>.json  codes of 1,500 test digits, their labels, and the axis range
Shared: figures/data/explorer-sprite.png, the same 1,500 test digits as one image (50 per row).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_models import AutoEncoder, load_mnist  # noqa: E402

MODELS = ROOT / "assets" / "models"
FIG = ROOT / "figures" / "data"
N, COLS = 1500, 50


def main():
    import matplotlib.pyplot as plt

    xte, yte = load_mnist()["test"]
    idx = torch.randperm(len(xte), generator=torch.Generator().manual_seed(77))[:N]
    x, y = xte[idx], yte[idx]
    rows = N // COLS
    sprite = x.reshape(rows, COLS, 28, 28).permute(0, 2, 1, 3).reshape(rows * 28, COLS * 28).numpy()
    plt.imsave(FIG / "explorer-sprite.png", 1.0 - sprite, cmap="gray", vmin=0, vmax=1)

    meta_all = json.loads((MODELS / "training.json").read_text())
    for name in ("ae2", "vae2"):
        meta = meta_all[name]
        m = AutoEncoder(meta["latent"], meta["variational"])
        m.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            mu, _ = m.encoder(x)
        sd = m.decoder.net.state_dict()
        parts = [sd["0.weight"], sd["0.bias"], sd["2.weight"], sd["2.bias"], sd["4.weight"], sd["4.bias"]]
        np.concatenate([t.numpy().astype("<f4").ravel() for t in parts]).tofile(FIG / f"explorer-{name}.bin")
        z = mu.numpy()
        if name == "ae2":
            lo, hi = np.percentile(z, [0.5, 99.5], axis=0)
            pad = 0.08 * (hi - lo)
            rng = [[float(lo[0] - pad[0]), float(hi[0] + pad[0])], [float(lo[1] - pad[1]), float(hi[1] + pad[1])]]
        else:
            rng = [[-4.0, 4.0], [-4.0, 4.0]]
        (FIG / f"explorer-{name}.json").write_text(json.dumps({
            "range": rng, "sizes": [256, 512, 784], "cols": COLS,
            "points": [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(z, y.tolist())],
        }))
        print(name, "range", [[round(v, 1) for v in r] for r in rng])


if __name__ == "__main__":
    main()
