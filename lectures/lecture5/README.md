# Lecture 5: Autoencoders and the VAE, from the ELBO

A main route of 61 slides (about 80 minutes) in five parts and twenty-one numbered sections, with nine optional slides after it. It is a condensed version of Lecture 2 (its generation story, autoencoders and the VAE) for a class that has already seen Lecture 1's ELBO. Nothing from Lecture 2 was removed: Lecture 2 remains the full two-session route. A 60-minute cut is listed in the speaker notes of the contents slide (and below).

The story: generative models are everywhere, and they share one recipe (random numbers in, a learned network, a sample out). An autoencoder squeezes a digit into 2 numbers and rebuilds it, but its decoder cannot turn random numbers into digits. The VAE fixes that by training the same two networks with Lecture 1's objective, the ELBO, written for one image at a time. The last part puts all the maths on one page and traces every term of the loss back to KL and the ELBO. Lecture 1's colours carry over (q purple, likelihood orange, prior green, evidence blue, posterior teal, KL to the posterior red), and loss slides show the target x in teal, the prediction x̂ in purple and the loss in red. Slides that rely on Lecture 1 link back to it (↗ Lecture 1, section).

**Running example.** Test digit #90 of MNIST, a 3, followed through the autoencoder (its loss pixel by pixel), the posterior over codes, per-image variational inference, the encoder's q, and one VAE training step.

**Live figures.** Autoencoder training replayed; hover maps of the codes; two code explorers (the autoencoder, then the VAE); Lecture 1's VI fitting q for our 3 (press play); an ELBO explorer, new here, where you move q for our 3 and watch the reconstruction term, the KL to the prior and the ELBO against log p(x) (Lecture 2's decoder runs in the browser, with 8 × 8 Gauss–Hermite averages that match the notebook to about 0.1); and seven trained VAEs with β from 0 to 256.

| Part | Sections |
|---|---|
| 1 · Why generate? | 1.1 Things that do not exist (StyleGAN faces; six real and six generated digits) · 1.2 Text, pictures and sound (a chatbot, next-word probabilities, text to image, image to image, speech) · 1.3 What generative models are for (science; create, fill in, simulate, anomalies, compression, more data) · 1.4 What they all share (the recipe; score and sample; why not the average; today's route) |
| 2 · Autoencoders: squeeze, then rebuild | 2.1 An image is 784 numbers · 2.2 Encoder, code, decoder · 2.3 The loss, and training (target, prediction and per-pixel error for our 3; the untrained output; one step; training replayed) · 2.4 What the code holds |
| 3 · Can an autoencoder generate? | 3.1 Decode random codes · 3.2 Where the codes are (N(0, I) draws over the real codes; a code explorer; a decoded grid) · 3.3 What we want instead |
| 4 · The VAE: Lecture 1's ELBO, one image at a time | 4.1 What Lecture 1 gave us (the objective; reparameterisation and the Gaussian KL as tools ① and ②) · 4.2 A model with a hidden code · 4.3 The same wall as Lecture 1 (Bayes' rule; the posterior of our 3; Lecture 1 and today side by side) · 4.4 The ELBO for one image (Lecture 1's ELBO with new names; Lecture 1's VI on our 3; the encoder as q; the network) · 4.5 The loss, term by term · 4.6 Two terms pull against each other (the sketch, then the ELBO explorer) · 4.7 What the VAE learned (samples, codes, grid, a code explorer) · 4.8 The KL weight β |
| 5 · Putting it together | 5.1 The whole lecture in maths (five lines, each with where we met it; every term of the loss traced to its source) · 5.2 Wrap-up |

**Optional:** training against test error, the linear autoencoder and PCA, N(0, I) in two dimensions, straight-line walks between autoencoder codes, points against clouds (Lecture 2's "Road 1"), the encoder's q against the exact posterior, the VAE's training curves, β = 0 and β = 256.

**A 60-minute cut.** Skip next-word, img2img, speech and science in Part 1; ae-untrained, ae-step and code-size in Part 2; ae-grid in Part 3; vae-code, explore-vae and beta-what in Part 4.

## Use

Open `index.html` through the series HTTP server (`python3 -m http.server` in `lectures/`). Press F to present full screen. Questions, expected answers and transitions are in the speaker notes (press N).

The deck reuses Lecture 2's data, examples and live figures: `content.js` imports `../lecture2/widgets.js` and loads `../lecture2/data/science.json`, `digits.json`, `nets.json` and the generated examples in `../lecture2/assets/`. `widgets.js` holds this lecture's own ELBO explorer. Every number on the slides was computed by Lecture 2's pipeline or, for the explorer, by Lecture 2's decoder in the browser; nothing here is retrained.

The notebook `lecture5-autoencoders-vae.ipynb` has saved outputs and the same section numbers as the slides; every slide's ▶ Notebook link opens its section in Colab. It loads Lecture 2's trained networks from `../lecture2/models/` (in Colab it downloads them and MNIST) and trains only a few small models for a few epochs, so it runs in about a minute.

## Rebuild

From `lectures/`, with a Python that has torch, torchvision, nbclient and matplotlib, and MNIST in `$MNIST_ROOT/MNIST/raw/`:

```sh
python lecture5/scripts/build_notebook.py   # executes the notebook and writes data/notebook-anchors.json
node scripts/verify.mjs 5                    # every state, layout checks, speaker-notes.md, manifest.json
node scripts/export.mjs 5                    # exports/completed.pdf and exports/all-builds.pdf
```

If Lecture 2's data or networks change (`lecture2/scripts/train.py`, `science.py`), rebuild this notebook and re-run the two commands above.
