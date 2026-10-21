# Lecture 3: generative adversarial networks

A 61-minute slide-led lecture: **60 core scenes + 3 optional scenes** (a contents slide, a break slide for each of the six parts, and 53 teaching scenes), with 120 authored presentation states. The route makes one discriminator update and one generator update visible before introducing the alternating game. Speaker notes and PDFs are generated from `content.js` by the series build.

**Sections.** The deck opens with a contents slide, and each of the six parts starts with a break slide that lists its sections. Every other slide sits in a numbered section (1.1, 1.2, …) shown in its header, with a ▶ Notebook link to the cell that runs it. The table is `data/sections.json`; `python ../scripts/notebook_sections.py 3` gives the notebook the matching anchors and slide links, and `scripts/make_notebook.py` runs it after every rebuild.

Open `index.html` through the local series server. Use the same reading, notes, keyboard and print controls as the other lectures. No external scripts or services are needed.

## Teaching route

| Minutes | Episode | Notebook action |
|---|---|---|
| 0–6 | A generator needs feedback | Predict the missing distributional coverage |
| 6–16 | D loss, gradient, update, recomputation | Inspect scalar D parameter changes |
| 16–28 | G loss and gradient through frozen D | Inspect G update and freeze/detach checks |
| 28–39 | Alternation and non-saturating loss | Run the short neural ownership test |
| 39–50 | Collapse, coverage and changing opponents | Compare saved runs; fresh quick toy run |
| 50–60 | MNIST transfer and retrieval | Replay saved image generator |

Timings are planned, not classroom rehearsed. The optional ideal-discriminator, mutable-state and experimental-controls scenes sit outside the main route. A 45-minute version protects the two explicit updates and one collapse example, while shortening the image and objective discussions.

## Run the notebook

`lecture3.ipynb` is saved with outputs from a complete CPU execution. It includes a fresh 400-update toy run, actual scalar and neural updates, and checkpoint replay. It does not claim that the image GAN was freshly retrained.

The tested environment uses Python 3.12 with PyTorch 2.13.0, NumPy, Matplotlib, Pillow, nbformat, nbclient and ipykernel already installed. No dependency was added.

```sh
# From the lectures/ directory, with the Python environment activated:
cd lecture3
python scripts/execute_notebook.py
```

Execute from the lecture directory in Jupyter, or use the series repository root supported by the initial cell. To regenerate a clean notebook from its authored cells, run `scripts/make_notebook.py`, then execute it; regeneration clears saved outputs deliberately.

## Evidence and reproducibility

- `data/fixtures.json`: full-precision scalar update, gradient checks, measured coverage and replay facts. All displayed arithmetic reads this file.
- `scripts/prepare.py`: regenerates computed figures and checks from bundled local data, recorded runs and checkpoint.
- `models/gan_mnist.pt`: unchanged original 30-epoch checkpoint; SHA-256 and source in `data/reuse-manifest.json`.
- `data/gan_mnist_snapshots.npz`: original fixed-noise training film, epochs 0–30.
- `data/mnist-subset.npz`: official training indices 0–511 and test indices 0–127, with labels. The held 3 is test index 90. Images remain uint8 until explicitly transformed.
- `qa/science-checks.json` and `qa/notebook-execution.json`: measured execution checks. `qa/notebook-figures/` contains rendered notebook outputs for inspection.

The image convention is greyscale `uint8/127.5−1`, unlike the binary Bernoulli VAE. The source-normalised D loss is the sum of real and fake batch means; G uses the non-saturating mean BCE. Raw GAN/VAE loss values are not comparable.

The stable and collapsed illustrative runs differ in generator width **and seed**. Their contrast demonstrates the observed failure, not a causal isolation of width. Coverage counts the fraction of 2,000 generated samples within one unit of each true mode; that is a limited diagnostic, not a full quality metric. KDE smoothing is labelled.

## Full training

Optional full image-model training additionally needs `torchvision`, compatible with the installed PyTorch version (`python -m pip install torchvision`). It is not required for the default notebook demonstrations or checkpoint replay.

The final notebook cell has `RUN_FULL=False`. To run it, set `MNIST_ROOT` to a local directory containing `MNIST/raw` and enable the cell. The script never downloads data. It uses the full training split, 30 epochs, batch size 128, Adam learning rate 0.0002, seed 1 and latent dimension 100. CPU works; optional `gan.DEVICE="mps"` uses an available Apple GPU. The new result goes to `models/full-retrain/`, preserving the supplied checkpoint.

BatchNorm running statistics and dropout modes are distinct from parameter freezing. The explicit ownership checks use MLPs without mutable buffers. The image D has dropout but no BatchNorm; the supplied training implementation retains its documented training-mode behaviour.

Original conceptual reference: [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661). Figures here are new computed plots or local measured outputs; no published third-party figure is reproduced.
