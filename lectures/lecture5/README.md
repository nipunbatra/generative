# Lecture 5: Diffusion models · create by learning to undo noise

A main route of 70 slides (about 130 minutes, two sessions) in eight parts with numbered sections, plus 24 slides of optional depth. It follows Paras Chopra's explainer [Diffusion Models from First Principles](https://paraschopra.github.io/explainers/diffusion/index.html) and expands it with the best moments of the videos listed below. The aim is the **simple version first**: every number visible, the forward and reverse processes followed step by step on small examples, live calculators, and clean diagrams. The heavier mathematics (the posterior by completing the square, the full ELBO, Tweedie, Langevin, SDEs) is in optional depth.

| Part | Sections |
|---|---|
| 0 · Opening | 0.1 Noise to a digit (our own model's film; published applications) · 0.2 Three generators (VAE, GAN, diffusion) · 0.3 The idea (ink spreading in water; the plan) |
| 1 · Destroying is easy | 1.1 One noising step (why shrink: the spread explodes without it) · 1.2 A toy schedule, by hand (β → α → ᾱ, five steps, every number) · 1.3 The shortcut (two steps collapse into one; simulation check) · 1.4 The real schedule (1,000 steps; the forward calculator) · 1.5 Watch our 3 dissolve (every image ends at N(0, 1)) |
| 2 · Undoing it: one step back | 2.1 What we want · 2.2 Big jumps back are hard, small ones easy (exact one-pixel reverse densities) · 2.3 Guess the noise · 2.4 One step back (the update, as a picture with numbers) |
| 3 · Training: labels for free | 3.1 The training example · 3.2 Why the network needs t · 3.3 The loss (it learns an average) · 3.4 The network (MLP, U-Net) · 3.5 Training, live (the smiley playground) · 3.6 What the network believes (x̂₀ at every noise level) |
| 4 · Generating: walk back from noise | 4.1 One step back, by hand · 4.2 The toy, all the way back (every number; the reverse calculator) · 4.3 The loop · 4.4 Watch it (live 2-D particles; MNIST runs) · 4.5 Why the fresh noise? · 4.6 New digits, and the napkin |
| 5 · Steering | 5.1 A 3 on demand · 5.2 Classifier-free guidance (and as arrows) · 5.3 The guidance dial · 5.4 Latent diffusion · 5.5 Text in, through cross-attention |
| 6 · Faster sampling | 6.1 DDIM: fewer steps · 6.2 The schedule matters (linear vs cosine) |
| 7 · What the network learns | 7.1 Arrows to the data (the score) · 7.2 It is a VAE (its ELBO is a weighted noise error) · 7.3 Does it just copy? (the ideal denoiser memorises; networks generalise) · 7.4 Wrap-up |

## Running examples

1. **One pixel** of our 3 (MNIST test image 90), pixel (22, 16): paper (−1) or ink (+1), ink in 35% of training images. Everything about it is exact, so the posterior, the best possible noise guess, the reverse densities and the score are computed in the browser (`math.mjs`).
2. **A toy schedule** of five big steps, β = 0.1, 0.2, 0.3, 0.5, 0.8, for arithmetic by hand: forward from x₀ = 1 with draws 0.3, −0.5, 0.6, −1.2, 0.5; back from x₅ with z = −0.3, 0.5, 0.2, −0.4.
3. **Our 3** through the real schedule (linear β from 10⁻⁴ to 0.02, T = 1,000) and the trained MNIST U-Net.
4. **A 2-D smiley** (outline, eyes, mouth): a small MLP that also runs live in the browser, for training, sampling, scores, guidance and memorisation.

## Live figures

`widgets.js`: the 3 dissolving, the spread with and without the shrink, the pixel histogram, the step-by-step vs one-jump check, the exact reverse-step explorer, the forward and reverse calculators (toy or real schedule, every number), the training playground, the network's x̂₀ film, 300 sampling particles with click-to-drop live particles, eight MNIST runs, DDPM vs DDIM paths, the DDIM step slider, linear vs cosine, guidance as arrows (live network) and the guidance dial, scores and Langevin, and the memorisation test. Print and PDF show fixed states.

## Use

Serve `lectures/` over HTTP (`python3 -m http.server 8767 --bind 127.0.0.1` from `lectures/`) and open `lecture5/`. Keys: arrows for builds, N for speaker notes, O for the outline; `?all` includes optional depth, `?reading` gives a scrolling version.

The notebook `lecture5-diffusion.ipynb` is self-contained (Colab: downloads MNIST and the checkpoints) and runs on CPU in a few minutes. Every slide links to its section.

## Rebuild

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python
$PY scripts/pixel.py             # data/pixel.json: the exact one-pixel numbers
$PY scripts/toy2d.py train       # models/toy-*.pt: the smiley denoisers (a few minutes on CPU)
$PY scripts/toy_record.py        # data/toy.json, data/toy-weights.json
$PY scripts/mnist.py             # data/mnist.json, figures/mnist-*.png (about 90 s on MPS)
$PY scripts/extra.py             # data/extra.json, average images
$PY scripts/assets.py            # published images, callbacks, figures/provenance.json
MNIST_DIR=... $PY scripts/build_notebook.py   # executes the notebook, writes data/notebook-anchors.json
```

## Evidence and provenance

- **MNIST models** (`models/mnist-*.pt`): the pixel DDPM (814,145 parameters, 30,000 updates, EMA), its class-conditional twin with label dropout, the 4 × 7 × 7 autoencoder, the latent DDPM and the classifier, all from the Generative Modelling II rebuild (`dl-teaching/generative-modelling-ii/models/v2`), saved here as EMA weights with source hashes in `models/provenance.json`. Network code is copied unchanged into `scripts/diffusion.py`.
- **Smiley models** (`models/toy-*.pt`): trained for this lecture (`scripts/toy2d.py`, 40,000 and 60,000 Adam steps; the conditional one with 30% label dropout).
- **Measurements** (`data/mnist.json`): x̂₀ errors by noise level, loss by t, trajectories, the σ = 0 collapse, DDIM timings, guidance accuracy and variety (classifier 97.8% on test digits), latent samples, the ideal-denoiser copies and our samples' nearest training neighbours.
- **Illustration**: ink in water, generated with Gemini (`figures/illustrations/`); labelled on the slide. **Published images**: credited on the slide and in `figures/provenance.json`.

## Sources

Paras Chopra's explainer, and the videos Gemini studied for this lecture: Ari Seff, *What are Diffusion Models?*; Deepia, *Diffusion Models: DDPM*; Outlier, *Diffusion Models, Math Explained*; Vizuara, *Principles of Diffusion Models*, Lecture 1; MIT 6.S183 *A Practical Introduction to Diffusion Models*, Lectures 1–6. Gemini's studies and the plan are in `plan/lecture5-diffusion/`.
