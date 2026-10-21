"""Copy and prepare the images Lecture 4 borrows: published samples (credited on the slide) and callbacks.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/assets.py           # writes figures/pub/*, figures/vae-samples.png, gan-samples.png, mnist-samples-16.png

Published images were acquired for the Generative Modelling II rebuild from each paper's own figure;
figures/provenance.json lists source and credit for every file.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
GM2 = Path("/Users/nipun/git/dl-teaching/generative-modelling-ii/figures/data")
PUB = {
    "pub-cherries-1.png": "LDM, Rombach et al. (2022), Fig. 1: “a basket of cherries”",
    "pub-repaint-1.png": "RePaint, Lugmayr et al. (2022), inpainting sample",
    "pub-sr3-face-3.png": "SR3, Saharia et al. (2022), 16×16 → 128×128 face",
    "pub-sdedit-out1.png": "SDEdit, Meng et al. (2022), stroke painting → image",
    "pub-control-2.png": "ControlNet, Zhang et al. (2023), Canny edges + “bird”",
    "pub-dreambooth-acropolis.png": "DreamBooth, Ruiz et al. (2023), subject “at the Acropolis”",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tiles(img, rows, cols, cell, gap=0):
    a = np.asarray(img)
    return [a[r * (cell + gap):r * (cell + gap) + cell, c * (cell + gap):c * (cell + gap) + cell] for r in range(rows) for c in range(cols)]


def grid(cells, rows, cols, gap=2, bg=255):
    h, w = cells[0].shape[:2]
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), bg, np.uint8)
    for i, c in enumerate(cells[:rows * cols]):
        r, k = divmod(i, cols)
        out[r * (h + gap):r * (h + gap) + h, k * (w + gap):k * (w + gap) + w] = c
    return Image.fromarray(out)


def main():
    prov = {}
    (FIG / "pub").mkdir(parents=True, exist_ok=True)
    for name, credit in PUB.items():
        shutil.copyfile(GM2 / name, FIG / "pub" / name)
        prov[f"pub/{name}"] = {"credit": credit, "source": f"dl-teaching/generative-modelling-ii/figures/data/{name}", "sha256": sha(FIG / "pub" / name)}
    # VAE (Lecture 2): the first ten prior samples of its 16-number VAE (mean images), ink on paper; 2 x 5
    lec = ROOT.parent
    sci = json.loads((lec / "lecture2" / "data" / "science.json").read_text())
    import base64
    px = np.frombuffer(base64.b64decode(sci["img"]["vae16_prior_samples"]["data"]), dtype=np.uint8).reshape(-1, 28, 28)
    cells = [255 - px[k] for k in range(10)]
    grid([np.asarray(Image.fromarray(c).resize((56, 56), Image.NEAREST)) for c in cells], 2, 5).save(FIG / "vae-samples.png")
    prov["vae-samples.png"] = {"credit": "Lecture 2 VAE (16-number code, β = 1), prior samples (mean images)", "source": "lectures/lecture2/data/science.json: img.vae16_prior_samples"}
    # GAN (Lecture 3): the first 16 samples after 30 epochs, re-coloured from white-on-black to ink-on-paper; 4 x 4
    g = Image.open(lec / "lecture3" / "figures" / "mnist-epoch-30.png").convert("L")
    cells = [255 - c for c in tiles(g, 4, 8, 96)][:16]
    grid([np.asarray(Image.fromarray(c).resize((56, 56), Image.LANCZOS)) for c in cells], 4, 4).save(FIG / "gan-samples.png")
    prov["gan-samples.png"] = {"credit": "Lecture 3 GAN, samples after 30 epochs (colours inverted to ink on paper)", "source": "lectures/lecture3/figures/mnist-epoch-30.png"}
    # ours: the first 16 of the 64 samples; 4 x 4
    s = Image.open(FIG / "mnist-samples-64.png").convert("L")
    cells = tiles(s, 4, 16, 28, gap=2)[:16]
    grid([np.asarray(Image.fromarray(c).resize((56, 56), Image.NEAREST)) for c in cells], 4, 4).save(FIG / "mnist-samples-16.png")
    prov["mnist-samples-16.png"] = {"credit": "our DDPM, Part 4", "source": "figures/mnist-samples-64.png"}
    prov["illustrations/ink-drop.jpg"] = {"credit": "Illustration generated with Gemini (gemini-3-pro-image-preview), 21 Oct 2026, cropped",
                                          "prompt": "five identical glasses of water, a drop of ink spreading until uniform; no text"}
    (FIG / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(len(prov), "assets")


if __name__ == "__main__":
    main()
