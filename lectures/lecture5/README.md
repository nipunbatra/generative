# Lecture 5: Autoencoders and the VAE in one hour

A main route of 43 slides (about 60 minutes) in four parts and seventeen numbered sections, with eleven optional slides after it. It is the one-hour version of Lecture 2's Parts 4–6 for a class that has already seen Lecture 1's ELBO. Nothing from Lecture 2 was removed: Lecture 2 remains the full two-session route.

The idea of the hour: an autoencoder squeezes a digit into 2 numbers and rebuilds it; its decoder cannot turn random numbers into digits; the VAE fixes that by training the same two networks with Lecture 1's objective, the ELBO, written for one image at a time. Lecture 1's colours carry over (q purple, likelihood orange, prior green, evidence blue, posterior teal, KL to the posterior red), and loss slides show the target x in teal, the prediction x̂ in purple and the loss in red. Slides that rely on Lecture 1 link back to it (↗ Lecture 1, section).

**Running example.** Test digit #90 of MNIST, a 3, followed through the autoencoder (its loss pixel by pixel), the posterior over codes, and one VAE training step.

| Part | Sections |
|---|---|
| 1 · Where we start: the ELBO | 1.1 What Lecture 1 gave us (the objective; reparameterisation and the Gaussian KL as tools ① and ②) · 1.2 Today's question (six real and six generated digits; the route, ending at the VAE's loss) |
| 2 · Autoencoders: squeeze, then rebuild | 2.1 An image is 784 numbers · 2.2 Encoder, code, decoder (the diagram; PyTorch) · 2.3 The loss, and training (target, prediction and per-pixel error for our 3; the untrained output; one step; training replayed) · 2.4 What the code holds (codes of 2 to 128 numbers; where the 2-number codes land) |
| 3 · Can an autoencoder generate? | 3.1 Decode random codes (three autoencoders) · 3.2 Where the codes are (N(0, I) draws over the real codes; a decoded grid) · 3.3 What we want instead |
| 4 · The VAE: Lecture 1's ELBO, one image at a time | 4.1 A model with a hidden code (graphical model; drawing a sample; the likelihood as squared error) · 4.2 The same wall as Lecture 1 (Bayes' rule and the intractable p(x); the posterior of our 3; Lecture 1 and today side by side) · 4.3 The ELBO for one image (Lecture 1's ELBO with new names; the encoder as q; the network with tool ①) · 4.4 The loss, term by term (minus the ELBO; one step for our 3 with numbers; PyTorch) · 4.5 Two terms pull against each other · 4.6 What the VAE learned (samples, codes, grid and interpolation) · 4.7 The KL weight β (seven trained VAEs) · 4.8 Wrap-up |

**Optional:** training against test error, the linear autoencoder and PCA, N(0, I) in two dimensions, the autoencoder code explorer, straight-line walks between autoencoder codes, points against clouds (Lecture 2's "Road 1"), the encoder's q against the exact posterior, the VAE's training curves, the VAE code explorer, β = 0 and β = 256.

## Use

Open `index.html` through the series HTTP server (`python3 -m http.server` in `lectures/`). Press F to present full screen. Questions, expected answers and transitions are in the speaker notes (press N).

The deck reuses Lecture 2's data and live figures: `content.js` imports `../lecture2/widgets.js` and loads `../lecture2/data/science.json`, `digits.json` and `nets.json`. Every number on the slides was computed by Lecture 2's pipeline; nothing here is retrained.

The notebook `lecture5-autoencoders-vae.ipynb` has saved outputs and the same section numbers as the slides; every slide's ▶ Notebook link opens its section in Colab. It loads Lecture 2's trained networks from `../lecture2/models/` (in Colab it downloads them and MNIST) and trains only a few small models for a few epochs, so it runs in about a minute.

## Rebuild

From `lectures/`, with a Python that has torch, torchvision, nbclient and matplotlib, and MNIST in `$MNIST_ROOT/MNIST/raw/`:

```sh
python lecture5/scripts/build_notebook.py   # executes the notebook and writes data/notebook-anchors.json
node scripts/verify.mjs 5                    # every state, layout checks, speaker-notes.md, manifest.json
node scripts/export.mjs 5                    # exports/completed.pdf and exports/all-builds.pdf
```

If Lecture 2's data or networks change (`lecture2/scripts/train.py`, `science.py`), rebuild this notebook and re-run the two commands above.
