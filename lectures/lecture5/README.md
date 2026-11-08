# Lecture 5: Autoencoders and the VAE, from the ELBO

A main route of 69 slides (about 90 minutes) in six parts and twenty-five numbered sections, with seven optional slides after it. It is a condensed version of Lecture 2 (its generation story, autoencoders and the VAE) for a class that has already seen Lecture 1's ELBO. Nothing from Lecture 2 was removed: Lecture 2 remains the full two-session route. A 60-minute cut is listed in the speaker notes of the contents slide (and below).

The story: generative models are everywhere, and they share one recipe (random numbers in, a learned network, a sample out). An autoencoder squeezes a digit into 2 numbers and rebuilds it; watching its codes during training shows clusters forming at once and then drifting outward, and an autoencoder trained on only 200 digits overfits, its codes flying apart. But its decoder cannot turn random numbers into digits. Three fixes, one at a time, turn it into a VAE: choose where codes live (N(0, I)), send each image to a small cloud of codes, and pull every cloud towards N(0, I) with a KL. The fixes give a loss; the next part shows why it is the right one. It tells the model's story from z: a code from the prior, an image from the likelihood; then runs it backwards to the posterior, shows why the evidence is out of reach, and picks the closest Gaussian, the variational approximation, by its ELBO. The loss is minus Lecture 1's ELBO, written for one image at a time, and its terms are the three fixes. The last part puts all the maths on one page and traces every term of the loss back to KL and the ELBO. Lecture 1's colours carry over (q purple, likelihood orange, prior green, evidence blue, posterior teal, KL to the posterior red), and loss slides show the target x in teal, the prediction x̂ in purple and the loss in red. Slides that rely on Lecture 1 link back to it (↗ Lecture 1, section).

**Running example.** Test digit #90 of MNIST, a 3, followed through the autoencoder (its loss pixel by pixel), the posterior over codes, per-image variational inference, the encoder's q, and one VAE training step.

**Live figures.** Autoencoder training replayed; the codes during training, with all 60,000 digits and with 200 (overfitting); hover maps of the codes; two code explorers (the autoencoder, then the VAE); Lecture 1's VI fitting q for our 3 (press play); an ELBO explorer, new here, where you move q for our 3 and watch the reconstruction term, the KL to the prior and the ELBO against log p(x) (Lecture 2's decoder runs in the browser, with 8 × 8 Gauss–Hermite averages that match the notebook to about 0.1); and seven trained VAEs with β from 0 to 256.

| Part | Sections |
|---|---|
| 1 · Why generate? | 1.1 Things that do not exist (StyleGAN faces; six real and six generated digits) · 1.2 Text, pictures and sound (a chatbot, next-word probabilities, text to image, image to image, speech) · 1.3 What generative models are for (science; create, fill in, simulate, anomalies, compression, more data) · 1.4 What they all share (the recipe; score and sample; why not the average; today's route) |
| 2 · Autoencoders: squeeze, then rebuild | 2.1 An image is 784 numbers · 2.2 Encoder, code, decoder · 2.3 The loss, and training (target, prediction and per-pixel error for our 3; the untrained output; one step; training replayed) · 2.4 What the code holds (code size; where the codes land; the codes during training; overfitting on 200 digits) |
| 3 · Can an autoencoder generate? | 3.1 Decode random codes · 3.2 Where the codes are (N(0, I) draws over the real codes; a code explorer; a decoded grid) · 3.3 What we want instead |
| 4 · From an autoencoder to a VAE: three fixes | 4.1 Fix 1: choose where codes live (z ~ N(0, I)) · 4.2 Fix 2: a cloud for each image (points against clouds; the encoder outputs μ and log σ², and the code is drawn with Lecture 1's reparameterisation trick) · 4.3 Fix 3: pull the clouds to N(0, I) (a KL penalty, Lecture 1's Gaussian KL, with numbers) · 4.4 The loss the fixes give (two open questions; the two terms pull against each other) |
| 5 · Why this loss: Lecture 1's ELBO, one image at a time | 5.1 What Lecture 1 gave us · 5.2 A model with a hidden code (the story: z from the prior, then x from the likelihood; the graphical model; the likelihood as a score for a code) · 5.3 The same wall as Lecture 1 (the story backwards: the posterior of our 3 and Bayes' rule; the evidence by brute force; Lecture 1 and today side by side) · 5.4 The variational approximation (the closest Gaussian, by its ELBO; Lecture 1's ELBO with new names; read term by term, it is the three fixes; Lecture 1's VI on our 3; the encoder's cloud is q) · 5.5 The loss, term by term · 5.6 Both terms, live (the ELBO explorer) · 5.7 What the VAE learned · 5.8 The KL weight β |
| 6 · Putting it together | 6.1 The whole lecture in maths (five lines, each with where we met it; every term of the loss traced to its source) · 6.2 Wrap-up |

**Optional:** training against test error, the linear autoencoder and PCA, straight-line walks between autoencoder codes, the encoder's q against the exact posterior, the VAE's training curves, β = 0 and β = 256.

**A 60-minute cut.** Skip next-word, img2img, speech and science in Part 1; ae-untrained, ae-step and code-size in Part 2; ae-grid in Part 3; keep all of Part 4; skip vae-pgm, vae-code, explore-vae and beta-what in Part 5.

## Use

Open `index.html` through the series HTTP server (`python3 -m http.server` in `lectures/`). Press F to present full screen. Questions, expected answers and transitions are in the speaker notes (press N).

The deck reuses Lecture 2's data, examples and live figures: `content.js` imports `../lecture2/widgets.js` and loads `../lecture2/data/science.json`, `digits.json`, `nets.json` and the generated examples in `../lecture2/assets/`. `widgets.js` holds this lecture's own ELBO explorer. Every number on the slides was computed by Lecture 2's pipeline or, for the explorer, by Lecture 2's decoder in the browser; nothing here is retrained.

The notebook `lecture5-autoencoders-vae.ipynb` has saved outputs and the same section numbers as the slides; every slide's ▶ Notebook link opens its section in Colab. It loads Lecture 2's trained networks from `../lecture2/models/` (in Colab it downloads them and MNIST) and trains only a few small models, so it runs in about three minutes. Section 2.4 repeats the overfitting run on 200 digits exactly, Section 5.2 scores three codes against our 3, Section 5.3 estimates the evidence by brute force, and Sections 4.1–4.3 run the three fixes (N(0, I) draws, the reparameterisation trick, the KL penalties), Section 5.4 repeats Lecture 1's variational inference on our 3 and compares four Gaussians q, Section 5.6 computes the ELBO explorer's three presets, and Section 6.1 computes every term of the loss for our 3 from scratch.

## Rebuild

From `lectures/`, with a Python that has torch, torchvision, nbclient and matplotlib, and MNIST in `$MNIST_ROOT/MNIST/raw/`:

```sh
python lecture5/scripts/epochs.py           # the codes during training: data/epochs.json (about 5 minutes)
python lecture5/scripts/build_notebook.py   # executes the notebook and writes data/notebook-anchors.json
node scripts/verify.mjs 5                    # every state, layout checks, speaker-notes.md, manifest.json
node scripts/export.mjs 5                    # exports/completed.pdf and exports/all-builds.pdf
```

`epochs.py` replays Lecture 2's training of its 2-number autoencoder step for step, so its epoch-30 codes are the ones on the slide before; that needs the device Lecture 2 trained on (mps). If Lecture 2's data or networks change (`lecture2/scripts/train.py`, `science.py`), re-run all four commands.
