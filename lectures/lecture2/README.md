# Lecture 2: From fitting a distribution to the VAE

A main route of 100 slides (about 140 minutes, two sessions) in six parts and twenty-seven numbered sections, with thirteen appendix slides after it. It follows the sequence of the PML 2023 recordings (Lectures 23–25: KL and VI, the ELBO and the coin, autoencoders and the VAE), rebuilt in the Lecture 1 style: one running example per part, one move per click, every number computed, and a takeaway box wherever a slide makes a point. The bottom bar carries the story line; questions, expected answers and **when to switch to the notebook** live in the speaker notes (press N). Slides that rely on Lecture 1 link back to it (↗ Lecture 1, section).

**Running examples.** Part 2: the 272 eruptions of Old Faithful. Parts 3–6: grey-level MNIST, with test digit #90 (a 3) followed through PCA, the autoencoder and the VAE.

| Part | Sections |
|---|---|
| 1 · Why generate? | 1.1 Things that do not exist (StyleGAN faces; six real and six generated digits, answer revealed) · 1.2 Text, pictures and sound (one prompt sampled three times from an LLM; real next-word probabilities from SmolLM2; one prompt sampled four times from a text-to-image model; pix2pix, SPADE, CycleGAN; text to speech) · 1.3 Beyond pictures · 1.4 What they all share (random numbers in, network, sample out; conditions; score and sample; why not the average) |
| 2 · Fit a distribution, then sample | 2.1 Old Faithful; the average is a poor sample · 2.2 One Gaussian: fit by formula, score, sample live · 2.3 Two kinds of eruption; a hidden step; scoring adds over both kinds · 2.4 What we minimise, the numbers to learn and how each stays valid, the loop line by line, the fit replayed; it recovers the eruption kinds; sample live · 2.5 Held-out data picks two bumps |
| 3 · Images: PCA and Gaussians | 3.1 784 numbers per digit · 3.2 One model for all digits; one pixel worked through (its 60,000 values and its bell); all 784 bells and their speckled samples; why it fails · 3.3 Digits vary in only a few ways (a map of 3s); encoder and decoder; PCA in 2-D, then on MNIST with a live reconstruction slider and a hover map · 3.4 Generate in four drawn steps (encode, fit a bell, draw, decode), one sample by hand, a live mixture sampler, what limits PCA |
| 4 · Autoencoders | 4.1 PCA as an encoder–decoder; the autoencoder with g_φ and f_θ; code; loss · 4.2 The untrained output (784 × 0.235 ≈ 185), one step, training replayed with per-digit errors, train vs test, a linear autoencoder rediscovers PCA · 4.3 Code sizes 2–128, against PCA · 4.4 Where the codes land |
| 5 · Can an autoencoder generate? | 5.1 N(0, I) explained; random codes through three autoencoders · 5.2 Random codes over the real ones (hover to decode, N(0, I) circles), the decoder explorer, a decoded grid, how close codes of similar digits are and how wider codes drift apart, interpolation · 5.3 What we saw and what we want instead |
| 6 · The VAE | 6.1 Two roads to the VAE · 6.2 Road 1: a cloud instead of a point, the network, the KL penalty · 6.3 Road 2: the VAE as a graphical model ("this is our prior"), drawing a sample by following the arrows, the likelihood as squared error, the intractable p(x), the posterior of our 3, the variational part (q_φ, the encoder), the encoder against the exact posterior, the ELBO equals Road 1's loss · 6.4 The loss term by term, a worked step, PyTorch, training curves · 6.5 Samples, codes filling N(0, I), the explorer, grid and interpolation · 6.6 β from 0 to 256 · 6.7 Wrap-up |

**Appendix:** a full-covariance Gaussian on pixels, PCA's variance curve, one Gaussian over d PCA numbers, search by code, denoising, a mixture fitted to autoencoder codes, the evidence by rectangles, naive Monte Carlo for p(x), per-image VI replayed, amortisation, a sparse autoencoder, held-out PCA-mixture scores, and EM.

Live figures (`widgets.js`): eruption sampling from one Gaussian and from the mixture, the mixture fit, the PCA reconstruction slider, the PCA mixture sampler, hover maps of codes (with decoding and N(0, I) circles), autoencoder training, the decoder explorer (the networks run in the browser from `data/nets.json`), per-image variational inference and the β comparison. Animations replay recorded runs: frame 0 until you advance, then they play. Print and PDF show static states.

## Use

Open `index.html` through the series HTTP server (`python3 -m http.server` in `lectures/`). The browser caches the slide modules; after editing, reload with the cache bypassed.

The notebook `lecture2-generative-models.ipynb` has saved outputs and the same section numbers as the slides; every slide's ▶ Notebook link opens its section in Colab. It needs PyTorch, torchvision, NumPy and matplotlib, downloads MNIST in Colab, and loads the three trained networks from `models/` (or the repository).

## Rebuild

From `lectures/`, with a Python that has torch, torchvision and matplotlib, and MNIST in `$MNIST_ROOT/MNIST/raw/`:

```sh
python lecture2/scripts/train.py            # 17 networks into lecture2/.cache/, about 15 minutes on an M-series Mac
python lecture2/scripts/science.py          # data/science.json, digits.json, nets.json, models/*.pt (about 1 minute)
python lecture2/scripts/build_notebook.py   # executes the notebook and writes data/notebook-anchors.json
node scripts/verify.mjs 2                    # every state, layout checks, speaker-notes.md, manifest.json
node scripts/export.mjs 2                    # exports/completed.pdf and exports/all-builds.pdf
```

## Files and evidence

- `content.js`: all slides, builds, story lines, notes and static figures. `widgets.js`: live figures, the chart kit and the digit renderer.
- `scripts/train.py`: every network (MLP, one hidden layer of 400 ReLU units each side; Adam, batch 128; autoencoders 30 epochs, VAEs 50; fixed seeds). σ = 0.2 for the VAE's Gaussian pixel noise.
- `scripts/science.py`: every number and image set on the slides (PCA by eigendecomposition, mixtures by EM, the posterior grid, importance sampling, per-image VI, the β sweep).
- `data/faithful.csv`: Old Faithful (Azzalini and Bowman, 1990), from R's `datasets` package via Rdatasets.
- `assets/examples/`: text, image and speech samples generated for this lecture on 21 October 2026 (Gemini 3.5 Flash, Gemini 3 Pro Image, Gemini 3.8 Flash TTS) and next-word probabilities from SmolLM2-360M; prompts and settings in `provenance.json`.
- `assets/gallery/`: published examples (StyleGAN, pix2pix, SPADE, CycleGAN, Context Encoders), attributed on their slides; see the repository's `assets/manifest.json`.
- `qa/browser-checks.json`: the automated layout checks of every rendered state.

Adapted from Nipun Batra's PML 2023 Lectures 23–25 and the [VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb). Differences: grey MNIST with an explicit Gaussian likelihood (σ = 0.2) instead of mean squared error plus an unscaled β; sums over pixels and means over images stated explicitly; the encoder's ELBO compared with per-image VI and with the exact evidence on a grid.
