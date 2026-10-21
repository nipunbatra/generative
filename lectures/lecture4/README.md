# Lecture 4: denoising diffusion probabilistic models

A 70-minute slide-led lecture: **60 core scenes + 3 optional scenes**, with 126 presentation states. It progresses from one noisy pixel to one supervised tuple, an inspected update, one reverse mean/variance calculation, a full small DDPM sampler and a brief text-conditioning connection.

Speaker notes and PDFs are generated from `content.js` by the shared series build. Open `index.html` through the local series server; everything needed for the slides is local.

## Teaching route

| Minutes | Episode | Notebook action |
|---|---|---|
| 0–6 | Noise-to-image problem and one held 7 | Inspect the recorded reverse trajectory |
| 6–17 | Fixed forward process, schedule and one pixel | Verify ᾱ, forward mean/variance and scalar arithmetic |
| 17–29 | Input/target tuple, time and ambiguity | Construct a real MNIST noisy input |
| 29–40 | Noise loss, scalar update and small U-Net | Run one update on training images |
| 40–54 | Clean estimate, reverse mean, variance and final boundary | Independently check one reverse step |
| 54–62 | VI connection and small DDPM | Replay the local checkpoint sampler |
| 62–70 | Text conditioning, latent space, guidance and retrieval | Retrieve the training/sampling distinctions |

Budgets are planned, not rehearsed. Optional posterior algebra and display semantics are outside the main route. The 45-minute route uses prepared schedule/VI results while protecting the tuple, literal update and reverse-step calculation.

## Run the notebook

`lecture4.ipynb` contains actual outputs from a complete CPU run. Its quick path generates two samples while retaining all 1,000 reverse steps. The default does not run full training.

```sh
# From the lectures/ directory, with the Python environment activated:
cd lecture4
python scripts/execute_notebook.py
```

Tested environment: Python 3.12, PyTorch 2.13.0, NumPy, Matplotlib, Pillow, nbformat, nbclient and ipykernel. No dependencies were installed. The notebook records actual sampling runtime; hardware/software differences affect it.

`make_notebook.py` regenerates the authored cells and clears outputs; execute afterwards. The companion model code is deliberately small and inspectable: `scripts/ddpm.py` contains schedule, U-Net, loss and ancestral sampler; `scripts/train.py` provides local-data quick and full training.

## The numerical contract

- One-based `t = 1,…,1000`, with `alpha_bar[0] = 1`.
- Linear β from 0.0001 to 0.02. The endpoint has ᾱ ≈ 0.00004036.
- Epsilon prediction; mean squared error averaged over all batch/channel/pixel elements.
- Reverse variance β̃ₜ = βₜ(1−ᾱₜ₋₁)/(1−ᾱₜ). No added random noise at t = 1.
- No clipping of training or sampler states. Display copies use declared finite ranges.
- Greyscale MNIST becomes `uint8/127.5−1`. This differs from the earlier binary Bernoulli VAE convention. No unlike raw loss comparison is made.
- The held image is official **test index 0, a 7**, explicitly introduced as a change from the earlier held 3. Actual updates use the training split.

The scalar denoiser ε̂ = θxₜ is a chosen arithmetic example, not the image network. Noise prediction, clean estimate x̂₀, reverse mean μθ and sampled previous state have separate scenes and values.

## Measured evidence

`models/ddpm.pt` is the unchanged 814,145-parameter checkpoint from the validated local diffusion lecture: 2,000 training updates, batch 64, seed 667, original MPS run. The current work executes fresh mathematical/gradient checks, a fresh training update on a copy, and a full 8-sample CPU replay. It does not claim a new full training run.

- `data/fixtures.json`: full-precision scalar, reverse and neural-update numbers.
- `data/replayed-trajectory.npz`: current full 1,000-step replay, including intermediate states.
- `data/original-trajectory.npz`, `data/original-training.json`: retained original measurements.
- `data/reuse-manifest.json`: source paths and hashes.
- `data/mnist-subset.npz`: official train indices 0–511 and test indices 0–127, with original labels.
- `qa/science-checks.json`, `qa/notebook-execution.json`: exact check scope and measured runtime; notebook figures are saved for visual inspection.

No image-quality metric or complete coverage claim is made for this small teaching model. Differences between CPU and original-device arithmetic are measured rather than hidden.

## Optional fresh full training

Set `MNIST_ROOT` to the local folder containing `train-images-idx3-ubyte.gz`. No download is attempted. Full mode uses all training examples and preserves the supplied checkpoint:

```sh
python scripts/train.py --mode full --steps 2000 --device cpu --mnist-root /path/to/local/MNIST/raw
```

The result is `models/full-training.pt` plus a measured log. Use `--device mps` only on a supported machine. Quick training uses the bundled subset and 20 updates, and is explicitly an execution check rather than an image-quality demonstration. Reducing the schedule to 20 steps is intentionally unsupported.

References: [Ho, Jain & Abbeel, 2020](https://arxiv.org/abs/2006.11239), [Rombach et al., 2022](https://arxiv.org/abs/2112.10752), [Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598). All diagrams are original teaching constructions; no paper figure is reproduced.
