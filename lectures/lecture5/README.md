# Lecture 5: Diffusion models · create by learning to undo noise

A main route of 68 slides (about 138 minutes, two sessions) in nine parts with numbered sections, plus 34 slides of optional depth. It follows Paras Chopra's explainer [Diffusion Models from First Principles](https://paraschopra.github.io/explainers/diffusion/index.html) section for section, and expands it with the best moments of the videos listed below and real image-model outputs from the companion film: the original DDPM cat model and Stable Diffusion v1.5. The aim is the **simple version first**: every number visible, the forward and reverse processes followed step by step on small examples, live calculators, and clean diagrams. The heavier mathematics (the one-pixel Bayes calculation, the posterior by completing the square, the full ELBO, Tweedie, SDEs) is in optional depth.

| Part | Sections |
|---|---|
| 0 · Introduction: destruction as creation | 0.1 Static in, a cat out (the original DDPM cat model; published applications) · 0.2 Destruction as creation (ink spreading in water; the plan) |
| 1 · The forward process | 1.1 One noising step (why shrink: the spread explodes without it) · 1.2 A toy schedule, by hand (β → α → ᾱ, five steps, every number) · 1.3 A beautiful shortcut (two steps collapse into one; simulation check) · 1.4 The real schedule (1,000 steps; the forward calculator) · 1.5 Watch it dissolve (our 3; every image ends at N(0, 1); a real photograph) |
| 2 · The reverse process | 2.1 Run the film backwards · 2.2 Small steps back are simple (exact one-pixel reverse densities) · 2.3 Guess the noise · 2.4 One step back (the update, as a picture with numbers) |
| 3 · What the network learns, and training | 3.1 What does the network learn? (noise, clean image, score) · 3.2 The network (U-Net; why it needs t) · 3.3 The training loop (one example, the loss, eight lines of code) · 3.4 Training, live (the smiley playground) · 3.5 What the network believes (x̂₀ for our 3, and for a photo with the real DDPM network) |
| 4 · Sampling: birth of an image | 4.1 One step back, by hand · 4.2 The toy, all the way back (every number; the reverse calculator) · 4.3 The sampling loop · 4.4 Watch it (live 2-D particles; MNIST runs; eight real cat runs) · 4.5 Why the fresh noise? · 4.6 New digits, and the napkin |
| 5 · The score: arrows to the data | 5.1 Arrows to the data · 5.2 Follow the arrows (Langevin) |
| 6 · Noise schedules and faster sampling | 6.1 Noise schedules (linear vs cosine) · 6.2 DDIM: fewer steps · 6.3 Faster still (Stable Diffusion from 1 to 50 DDIM steps, and a consistency model in one step; the frontier) |
| 7 · Steering: labels, guidance and text | 7.1 A 3 on demand · 7.2 Classifier-free guidance (and as arrows) · 7.3 The guidance dial · 7.4 From pixels to latents (ours, and Stable Diffusion’s autoencoder) · 7.5 Text in, through cross-attention (with Stable Diffusion’s real attention maps) · 7.6 A sentence in, a picture out (five prompts sampled; the guidance dial on a sentence) |
| 8 · Wrap-up | 8.1 Three generators side by side, exit questions, references |

## Running examples

1. **One pixel** of our 3 (MNIST test image 90), pixel (22, 16): paper (−1) or ink (+1), ink in 35% of training images. Everything about it is exact, so the posterior, the best possible noise guess, the reverse densities and the score are computed in the browser (`math.mjs`).
2. **A toy schedule** of five big steps, β = 0.1, 0.2, 0.3, 0.5, 0.8, for arithmetic by hand: forward from x₀ = 1 with draws 0.3, −0.5, 0.6, −1.2, 0.5; back from x₅ with z = −0.3, 0.5, 0.2, −0.4.
3. **Our 3** through the real schedule (linear β from 10⁻⁴ to 0.02, T = 1,000) and the trained MNIST U-Net.
4. **A 2-D smiley** (outline, eyes, mouth): a small MLP that also runs live in the browser, for training, sampling, scores, guidance and memorisation.

## Live figures

`widgets.js`: real DDPM cat and Stable Diffusion sampling films and a photo jumping to any noise level, the 3 dissolving, the spread with and without the shrink, the pixel histogram, the step-by-step vs one-jump check, the exact reverse-step explorer, the forward and reverse calculators (toy or real schedule, every number), the training playground, the network's x̂₀ film, 300 sampling particles with click-to-drop live particles, eight MNIST runs, DDPM vs DDIM paths, the DDIM step slider, linear vs cosine, guidance as arrows (live network) and the guidance dial, scores and Langevin, and the memorisation test. Print and PDF show fixed states.

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
$PY scripts/film_assets.py       # figures/film/: real DDPM cat and Stable Diffusion outputs from the companion film
MNIST_DIR=... $PY scripts/build_notebook.py   # executes the notebook, writes data/notebook-anchors.json
```

## Evidence and provenance

- **MNIST models** (`models/mnist-*.pt`): the pixel DDPM (814,145 parameters, 30,000 updates, EMA), its class-conditional twin with label dropout, the 4 × 7 × 7 autoencoder, the latent DDPM and the classifier, all from the Generative Modelling II rebuild (`dl-teaching/generative-modelling-ii/models/v2`), saved here as EMA weights with source hashes in `models/provenance.json`. Network code is copied unchanged into `scripts/diffusion.py`.
- **Smiley models** (`models/toy-*.pt`): trained for this lecture (`scripts/toy2d.py`, 40,000 and 60,000 Adam steps; the conditional one with 30% label dropout).
- **Measurements** (`data/mnist.json`): x̂₀ errors by noise level, loss by t, trajectories, the σ = 0 collapse, DDIM timings, guidance accuracy and variety (classifier 97.8% on test digits), latent samples, the ideal-denoiser copies and our samples' nearest training neighbours.
- **Real photos** (`figures/film/`, `figures/film/provenance.json`): outputs of the original DDPM LSUN-cat checkpoint (google/ddpm-cat-256, 114M parameters) and the CC0 photo "Chelsea" (scikit-image), computed by the companion film "Diffusion models, from noise to images, step by step" (`dl-summaries/diffusion/math/ddpm_images.py`). Stable Diffusion v1.5 (`figures/film/sd/`): guidance and DDIM-step sweeps, the LCM-LoRA adapter in 1–8 steps, the autoencoder round trip of Chelsea, 16 × 16 cross-attention maps for "a red fox sitting in the snow", and five sampling films (`math/sd_runs.py`, settings in `figures/film/provenance.json`).
- **Illustration**: ink in water, generated with Gemini (`figures/illustrations/`); labelled on the slide. **Published images**: credited on the slide and in `figures/provenance.json`.

## Sources

Paras Chopra's explainer, and the videos Gemini studied for this lecture: Ari Seff, *What are Diffusion Models?*; Deepia, *Diffusion Models: DDPM*; Outlier, *Diffusion Models, Math Explained*; Vizuara, *Principles of Diffusion Models*, Lecture 1; MIT 6.S183 *A Practical Introduction to Diffusion Models*, Lectures 1–6. Gemini's studies and the plan are in `plan/lecture5-diffusion/`.
