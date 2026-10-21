# Lecture 2: From fitting a distribution to the VAE

A main route of 98 slides (about 150 minutes, two sessions) in six parts and twenty-nine numbered sections, with three optional slides after it. It follows the sequence of the PML 2023 recordings (Lectures 23–25: KL and VI, the ELBO and the coin, autoencoders and the VAE), rebuilt in the Lecture 1 style: one running example per part, one move per click, every number computed. The bottom bar carries the story line; questions, expected answers and **when to switch to the notebook** live in the speaker notes (press N).

**Running examples.** Part 2: the 272 eruptions of Old Faithful (wait and eruption length). Parts 3–6: grey-level MNIST, with test digit #90 (a 3) followed through PCA, the autoencoder and the VAE.

| Part | Sections |
|---|---|
| 1 · Why generate? | 1.1 Things that do not exist (StyleGAN faces; six real and six generated digits, answer revealed) · 1.2 Text, pictures and sound (one prompt sampled three times from an LLM; real next-word probabilities from SmolLM2; one prompt sampled four times from a text-to-image model; pix2pix, SPADE, CycleGAN; text to speech) · 1.3 Beyond pictures (weather ensembles, proteins, materials, missing sensor data; what p(x) is good for) · 1.4 What they all share (random numbers in, network, sample out; conditions; score and sample; why not the average) |
| 2 · Fit a distribution, then sample | 2.1 Old Faithful; the average is a poor sample · 2.2 One Gaussian: fit by formula, score, sample live · 2.3 Two kinds of eruption; a hidden step; scoring adds over both kinds · 2.4 Fit the mixture by gradient descent (replayed run); it recovers the eruption kinds; sample live · 2.5 Held-out data picks two bumps · 2.6 Two numbers per eruption: covariances, a 2-D mixture and its samples |
| 3 · Images: PCA and Gaussians | 3.1 784 numbers per digit; random points are static · 3.2 Independent pixel Gaussians (1,568 numbers), neighbour correlations, a full Gaussian (308,504 numbers) · 3.3 PCA in 2-D on the eruptions, then on MNIST: eigen-digits, a live reconstruction slider, the variance curve, a hover map · 3.4 Sample a code, decode: one Gaussian, then a live mixture sampler over d PCA numbers; what limits the recipe |
| 4 · Autoencoders | 4.1 PCA as an encoder–decoder; the autoencoder, its code and loss (our 3: 26.3 against PCA's 41.3) · 4.2 The untrained output, one training step, training replayed, train vs test, a linear autoencoder rediscovers PCA · 4.3 Code sizes 2–128, against PCA at every size · 4.4 Codes on a hover map, search by code, denoising |
| 5 · Can an autoencoder generate? | 5.1 N(0, I) explained; random codes through 2-, 8- and 128-number autoencoders · 5.2 The overlay (PML L25 at 29–31 min), a live decoder explorer (random code, encoded digit, click anywhere), a decoded grid, codes spreading with size, interpolation · 5.3 A mixture fitted to the codes; what we want instead |
| 6 · The VAE | 6.1 The generative story; σ = 0.2 and squared error as a likelihood; maximum likelihood needs an integral · 6.2 The exact posterior of our 3 on a grid, log p(x) = 244.07 (checked by importance sampling), naive Monte Carlo · 6.3 Lecture 1's ELBO with new roles; per-image VI replayed (ELBO 244.05) · 6.4 Amortisation: the encoder (one pass, 243.37); the VAE diagram · 6.5 The loss term by term, a worked step for our 3, PyTorch, training curves · 6.6 Samples, codes filling N(0, I), the explorer again, grid and interpolation · 6.7 β from 0 to 256: a live comparison, β = 0 and posterior collapse · 6.8 Wrap-up |

Nine slides carry live figures (`widgets.js`): sampling eruptions from one Gaussian and from the mixture, the mixture fit, the PCA reconstruction slider, the PCA mixture sampler, hover maps of codes, autoencoder training, the decoder explorer (the networks run in the browser from `data/nets.json`), per-image variational inference and the β comparison. Animations replay recorded runs: frame 0 until you advance, then they play. Print and PDF show static states.

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
