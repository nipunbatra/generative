# Learning to generate · ES 667 Generative Modelling I (VAEs)

A rebuilt lecture: distributions and fitting, autoencoders, the probabilistic view, KL divergence, variational inference
and the ELBO, reparameterisation, and the VAE, taught slowly around **one running example** (MNIST test digit #90, a handwritten 3)
and one 1-D toy dataset (two bumps at ±2). Every figure and every number on the slides comes from models trained by the scripts below.
GANs are a separate lecture (`deck/slides/10-gan.html` is kept here as its source; it is not in these decks).

Two 75-minute sessions, seven parts:

| Deck | Slides | Content |
|---|---|---|
| `part1.html` · [exports/session1.pdf](exports/session1.pdf) | 95 | why generate · **1** distributions and fitting (PyTorch fitting, how many bumps, sampling, Bayes, why the gradient needs the posterior, and a second half on MNIST: PCA and inverse PCA, Gaussian mixtures over d PCA numbers with a live d × K explorer, held-out vs AIC vs BIC, a mixture of Bernoullis on raw pixels, and why the recipe does not scale) · **2** autoencoders (the bridge from PCA: a linear autoencoder rediscovers it; MLP, CNN and ViT encoders compared; denoising, sparse, contractive; why they cannot generate, and Part 1's bumps fitted to the codes) · **3** the probabilistic view (prior, decoder as likelihood, the intractable evidence and posterior, Laplace / MCMC / VI) |
| `part2.html` · [exports/session2.pdf](exports/session2.pdf) | 79 | recap · **4** KL divergence, the slow way (a betting game, surprise, entropy, cross-entropy) · **5** variational inference and the ELBO (derived from KL(q ‖ posterior), VI by hand for our 3, importance sampling, amortisation = the VAE) · **6** reparameterisation · **7** the VAE on MNIST · wrap-up · appendix (the ELBO through Jensen) |
| `index.html` · [exports/lecture.pdf](exports/lecture.pdf) | 174 | both sessions in one deck |

`*-builds.pdf` versions show every click.

## The maths lecture before the VAE: `vi.html`

**KL divergence and variational inference**, a separate 60–75 minute lecture that teaches the maths on toys, so the VAE lecture can
apply it instead of introducing it. It follows the PML course's Lectures 23–24 (variational inference) frame by frame, with the
course's own examples: the Beta(5, 3) coin with 17 heads in 30, p = N(1, 4²), and two mixtures of Gaussians. The plan it was built
from, with the source of every slide, is [plan/vi-lecture-plan.md](plan/vi-lecture-plan.md).
70 slides (a cover, then 9 parts), 168 clicks: `vi.html` · [exports/vi.pdf](exports/vi.pdf) · [exports/vi-builds.pdf](exports/vi-builds.pdf).
It does not touch the VAE decks above (their Parts 4–6 are to be trimmed only after this deck is reviewed).

| Part | Slides | Content |
|---|---|---|
| 1 · Recap: Bayes and the evidence | 11 | Steve, librarian or farmer (posterior 1/6); Bayes for ML; the coin, prior Beta(5, 3) → posterior Beta(22, 16), with a live prior × likelihood widget; its evidence by formula (log p(D) = −21.397); linear-regression evidence in closed form (12.35) vs 15,000 prior draws (12.22; the best 10 draws carry 95%); a Normal prior on the logit has no formula; **one** slide on Laplace, MCMC and VI on a two-bump posterior; the VI objective and the notebook's goals G1–G6 |
| 2 · KL divergence (G1) | 13 | built from a fruit survey (ratios, the log, weighting by p: 0.416 nats); the same in bits (weather codes, 0.25 extra bits); KL for densities, ≥ 0 by the tangent line, asymmetric; a 0.8 coin vs a fair one (325× after 30 tosses); which q is closer to N(1, 4²) (0.35 vs 0.07), live; ✎ derive the Gaussian KL, then torch's `_kl_normal_normal`; forward vs reverse KL on the two-bump mixture, static and live |
| 3 · Fitting q (G2) | 6 | a grid of candidates; `TrainableNormal` with softplus and an Adam loop; press play from N(4, 0.5²) to N(1.00, 4.00²); the snapshots; diagonal vs full covariance on a correlated 2-D target (width 0.436 = √(1 − ρ²)) |
| 4 · Monte Carlo (G3, G5) | 6 | the notebook's mixture and two candidates; torch's `NotImplementedError`; Monte Carlo and darts for π; `kl_via_sampling` (1.0276 vs exact 1.0288; mixture 9.45 vs 45.02); the spread vs N |
| 5 · Reparameterisation (G4) | 8 | the computational graph with a blocked draw; `sample().backward()` fails; the reparameterised graph and ✎ its gradients; checked in code (scale.grad = ε = 0.3367); `rsample` inside torch; fitting the mixture from N(6, 0.1²) to N(0.43, 0.73²), static and live |
| 6 · The ELBO | 8 | the catch (the KL contains the posterior); ✎ substitute Bayes; the evidence comes out; the lower bound; ELBO + KL = −21.397 for three Beta q's; a live Beta(α, β) number line; the two forms of the ELBO |
| 7 · The coin, worked | 8 | the model as functions; why a Beta q; `neg_elbo`; two guesses (239.38 vs exactly 21.397); training from Beta(45, 45) to Beta(22.6, 16.4); press play; the same in Pyro |
| 8 · Linear regression (G6) | 4 | the model and its `log_joint`; diagonal and full q against the exact posterior with 20 and with 200 points (homework if time is short) |
| 9 · Neural networks and the VAE | 5 | mean-field for millions of weights; the VAE in one slide; recap; five exit questions |

Slides marked ▶ N1 … N8 have a matching section in the **companion notebook** `vi-lab/vi_lecture.ipynb` (helpers in
`vi-lab/vi_utils.py`; torch, numpy and matplotlib only, no ASTRA; Pyro optional for one guarded cell; 63 cells, about 30 s).
Slides marked ✎ are in-class exercises. The small grey line at the foot of a slide cites its PML source (lecture and timestamp,
notebook cells, slide frame).

Every number is computed: `scripts/make_figures_vi.py` replicates the notebook with the same seeds and writes the figures
`figures/data/vik-*.svg` and `figures/data/numbers-vi.json` (merged by `build_deck.py` as `{{vik.*}}`); `scripts/diagrams.py` draws the
two computational graphs (`figures/diagrams/vik-graph-*.svg`). Slides live in `deck/slides/v0-open.html` … `v9-vae.html`; the deck has
its own 9-part route (`<!--viroute:N-->`, `VI_PARTS` in `build_deck.py`).

Live widgets (in `deck/widgets.js`, each with a print state): `betacoin` (prior × likelihood → posterior), `klgauss` (KL between two
Gaussians as shaded areas, both directions), `fwdrev` (forward vs reverse KL on the mixture, by gradient descent on the notebook's grid),
`chase` (q chasing a target: exact KL to N(1, 4²), or a Monte Carlo KL with reparameterised draws to the mixture), `coinelbo` (the
coin's exact ELBO and KL for any Beta(α, β) on a number line, and a play button that fits it). The VAE decks' `kl`, `elbo`, `explorer`,
`latent` and `gmmx` widgets are unchanged.

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python
$PY scripts/make_vi_notebook.py                  # rebuild and run the companion notebook
$PY scripts/make_figures_vi.py && $PY scripts/diagrams.py && $PY scripts/build_deck.py
python3 scripts/review_server.py --port 8774      # then http://127.0.0.1:8774/vi.html (8771: the VAE decks, 8773: the GAN deck)
PORT=8774 node scripts/_overflow.mjs vi.html && PORT=8774 node scripts/_overlap.mjs vi.html
node scripts/export_pdf.mjs vi                     # vi.pdf and vi-builds.pdf, with the maths check
```

## Open it

```sh
python3 scripts/review_server.py      # or: python3 -m http.server 8771
```

Then open http://127.0.0.1:8771/part1.html (or `part2.html`, `index.html`).

- → / Space next click · ← back · Shift+→ next slide · **O** overview · **S** speaker notes · **F** fullscreen · `?reading` for all slides with notes

## Review notes (right-click annotations)

Start the review server (it serves the deck and receives notes):

```sh
python3 scripts/review_server.py      # then open http://127.0.0.1:8771/part1.html
```

- **Right-click** anything on a slide, type what should change, press **Send** (or ⌘/Ctrl+Enter).
- The note records the deck, slide, click number and what you clicked (an equation's exact LaTeX).
  It is written to `review/notes.json` and appended to `review/inbox.jsonl`, which a running Claude session watches,
  so the request arrives immediately.
- **Red pin** = open request. **Green ✓** = Claude made the change; hover or click it to read the reply.
- When the deck is rebuilt, open pages reload themselves onto the same slide and click.
- Press **A** to list all notes (Copy all / Clear all). Shift+right-click opens the normal browser menu.
- Without the server (file:// or a static host) notes stay in the browser; use **A → Copy all** and paste them into the chat.

## Maths

All maths is typeset by KaTeX (slides, diagram labels) or matplotlib's Computer Modern (charts), so every symbol matches.
Shared macros (`\N`, `\KL`, `\E`, `\eps`, `\ELBO`, role colours `\cx \cenc \cdec \cz \closs \cdisc`, and `\hl{}` for "this part just changed")
are defined once in `scripts/build_deck.py`. The PDF export fails if any maths character appears outside KaTeX.

## Rebuild everything

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python   # any Python with torch + torchvision + matplotlib
$PY scripts/train_models.py     # ~10 min on an M-series GPU; MNIST read from $MNIST_ROOT, nothing downloaded
$PY scripts/make_figures.py     # figures/data/*.svg|png and figures/data/numbers.json (~4 min)
$PY scripts/make_figures_extra.py   # fitting, how-many-bumps, AE variants, Laplace/MCMC/VI, VI by hand; numbers-extra.json
$PY scripts/make_figures_gmm.py     # MNIST mixtures: every (d, K) for the explorer, AIC/BIC, Bernoulli mixture (~30 min CPU first time, cached after)
$PY scripts/make_figures_bridge.py  # PCA vs linear / MLP / CNN / ViT autoencoders on grey MNIST, bumps on their codes (~3 min GPU)
$PY scripts/export_explorer.py  # decoder weights for the in-browser latent explorers
$PY scripts/diagrams.py         # figures/diagrams/*.svg (hand-built SVG, numbers read from numbers.json)
$PY scripts/build_deck.py       # inlines diagrams, fills {{numbers}}, writes part1.html, part2.html, index.html
$PY scripts/make_figures_vi.py  # the maths lecture (vi.html): toy figures and numbers-vi.json, no models needed
node scripts/export_pdf.mjs     # session1/2, lecture and vi PDFs (+ -builds), and the maths check; `node scripts/export_pdf.mjs vi` for one deck
```

## Where things live

| Edit this | To change |
|---|---|
| `deck/slides/01-hook.html` … `09-wrap.html` | slide text, builds (`data-b="2"`), speaker notes (`<aside class="notes">`). 01–04 = session 1, 05–09 = session 2 |
| `scripts/diagrams.py` | the SVG diagrams (encoder/decoder, VAE, reparameterisation, GAN steps …) |
| `scripts/make_figures.py`, `scripts/make_figures_extra.py` | every chart and digit grid, `numbers.json` and `numbers-extra.json` |
| `scripts/train_models.py` | all models: AE (2, 16, 64 and sparse-64 numbers), denoising AE, VAE (2, 16, β = 8), conditional VAE, PixelAR, and the GAN-lecture models |
| `deck/slides/v0-open.html` … `v9-vae.html` | the maths lecture `vi.html` (KL and VI, after the PML course) |
| `scripts/make_figures_vi.py` | every figure and number of `vi.html` (`figures/data/vik-*`, `numbers-vi.json`) |
| `vi-lab/vi_utils.py`, `scripts/make_vi_notebook.py` | the VI lecture's companion notebook `vi-lab/vi_lecture.ipynb` (helpers, and the script that writes and runs it) |
| `deck/deck.css`, `deck/deck.js`, `deck/widgets.js`, `deck/annotate.js` | look, navigation, the live widgets (the VI lecture's at the end of `widgets.js`), and right-click review notes |

Build syntax: `data-b="3"` appears from click 3 on, `data-b="2-4"` only during clicks 2–4, `data-b="=1"` only on click 1, `data-hl="2"` highlights from click 2.

## What is real

- AE and VAE (2- and 16-number codes, plus a β = 8 variant): binarised MNIST, MLP 784-512-256-z, 30–40 epochs.
- Autoencoder variants: a denoising AE (16 numbers, 20% of pixels flipped) and a 64-unit AE with and without an L1 penalty on the code.
- Part 1 fits: one Gaussian by gradient descent, and mixtures of 1–4 bumps scored on 200 held-out numbers.
- Part 3 and 5 inference on our 3, all on the trained 2-number decoder: Laplace (autograd Hessian at the peak), random-walk Metropolis,
  and variational inference by hand (a diagonal Gaussian, 800 Adam steps on the ELBO: −128.83, against −129.30 for the encoder's one pass;
  log p(x) = −128.75 on a fine grid around the posterior, so the gaps are 0.08 and 0.55 nats).
- Layout checks: `node scripts/_overflow.mjs index.html` and `node scripts/_overlap.mjs index.html` (with the review server running) list any slide whose content leaves the frame or runs into its title or take-away line.
- Opening images are published model outputs (StyleGAN, pix2pix, CycleGAN, Context Encoders, SPADE), with sources on each slide.

## The previous (Codex) version

The earlier Codex build is still in this folder, untouched and **not committed**: `codex-v1.html` (its entry page), `src/`, `notes.html`,
`exports/completed-scenes.pdf`, `exports/all-builds.pdf`, `notebooks/`, `qa/`. Its README follows.

---

# Learning to Generate

A completely new Generative Modelling I lecture for **Nipun Batra's ES 667 Deep Learning course, IIT Gandhinagar**. Applications, distributions, KL, latent variables, the ELBO, reparameterisation, VAEs and a substantial GAN block.

The older lecture files and the old diffusion folder were removed from this working directory as requested. Git history and the local KaTeX dependency were retained. No site was deployed.

## Open the lecture

From this folder:

```sh
python3 -m http.server 8766 --bind 127.0.0.1
```

- Presentation: http://127.0.0.1:8766/?present
- Reading mode with complete derivations and instructor notes: http://127.0.0.1:8766/?reading
- Saved reading edition: [notes.html](notes.html)
- [Completed-scene PDF](exports/completed-scenes.pdf): **64 pages**, comprising 60 core scenes, three appendix scenes and an ELBO continuation page.
- [All-builds PDF](exports/all-builds.pdf): **151 pages**, comprising 145 core reveal states and six appendix states.
- [Two-page recap](exports/recap.pdf)
- [Executed notebook](notebooks/lecture-lab.ipynb), [student lab](notebooks/LAB.md), [answer notes](notebooks/ANSWERS.md)

**Every scene has a visible Say cue; its final reveal has a labelled Takeaway.** Press **N** for a fuller spoken explanation and **Q** for a discussion question with its answer. These cues also appear in both PDFs and the saved notes.

Use the static server for the interactive ES-module version. The saved notes and PDFs are immediately readable files. All classroom assets, mathematics, fonts and recorded model outputs are local; there is no API, sign-in, GPU or external media dependency during class.

## Routes and controls

Both routes contain **60 scenes and 145 core reveals**. The full route budgets **120 minutes**. The compressed route budgets **90 minutes**, using shorter gallery narration, recorded playback and bounded discussion. It does not remove the KL, ELBO, gradient or GAN teaching. Appendix material is separate. The practical lab adds **45–60 minutes**.

| Section | Scenes | Reveals | Full | Compressed |
|---|---|---:|---:|---:|
| Applications | 01–10 | 20 | 12 min | 8 min |
| Distributions | 11–18 | 20 | 14 min | 10 min |
| KL | 19–23 | 13 | 10 min | 8 min |
| ELBO | 24–31 | 20 | 18 min | 14 min |
| Gradients | 32–36 | 12 | 10 min | 8 min |
| VAE | 37–44 | 21 | 18 min | 13 min |
| GAN | 45–58 | 35 | 32 min | 25 min |
| Comparison | 59–60 | 4 | 6 min | 4 min |

Arrow keys, Space and Page Up/Down move through reveals. Shift+arrow changes scene. Home/End go to the core lecture ends. O opens the overview, N presenter notes, Q the discussion prompt. Escape closes overlays; from the stage it opens reading mode. Focused inputs keep their native keys. Scene/build deep links use `?present#s27/4`. Demo controls provide deterministic reset, step, play and pause. Resampling is distinct from changing parameters. Playback stops when the scene changes.

## What is editable

- `src/content.mjs`: classroom text and LaTeX for every reveal.
- `src/teaching-cues.mjs`: short speaking cues, fuller explanations and discussion answers.
- `src/figures.mjs`: mechanism diagrams, including the VAE encoder branches, prior sampling, conditional GAN and gradient paths. `figures/` contains their exported SVGs.
- `src/scene-manifest.json`: stable scene IDs, planned sequence, goals, notes, sources and route membership.
- `src/visuals.mjs`: original SVG diagrams and live numerical panels.
- `src/math.mjs`: the shared numerical source used by the interface, lab and tests.
- `src/app.mjs`, `src/style.css`: navigation, reading view, export adapter and course styling.
- `notebooks/models.py`: complete VAE and image-GAN definitions and training steps.
- `notebooks/train.py`: procedural glyph data, fresh VAE/image-GAN/toy-GAN training and artifact export.
- `assets/manifest.json`, [SOURCES.md](SOURCES.md): figure provenance, transformations and source terms.
- [REQUIREMENTS.md](REQUIREMENTS.md): all 20 acceptance requirements mapped to scenes and evidence.

The HTML uses an independent plain JavaScript runtime. Its `window.lecture` export adapter exposes `listScenes`, `getBuildCount`, `goTo`, `setBuild`, `finish` and `ready`; a `window.lectureReveals` compatibility adapter is also provided. The 1280×720 stage and semantic colour mapping follow the course visual system. The figures were revised against the local Beyond Generation deck: branch where the computation branches, keep one example in view, and reveal the mechanism in steps. 

## Build, verify and export

The runtime has no build-time frontend framework. Python builds the entry page; Puppeteer drives the browser checks and native PDF export.

```sh
npm install
python3 scripts/build.py
npm test
# Start the static server in another terminal, then:
node scripts/verify.mjs
node scripts/check-reading.mjs
node scripts/check-decoder.mjs
node scripts/export.mjs
python3 scripts/refresh-manifest.py
```

The browser helper uses `PUPPETEER_EXECUTABLE_PATH` when supplied, an installed Chrome/Chromium if found, or Puppeteer's browser. With Python and `pypdf` available, `python3 scripts/check-pdf.py` also verifies page counts, text, speaking cues and PDF link annotations. PDF export preserves selectable text, vector diagrams, source-link annotations and internal section links. Application figures and canvas-generated pixel outputs remain raster images within the PDF. Both export commands include the whole lecture and appendix regardless of the selected route.

Verified on this Mac with Chrome, Node, and `/Users/nipun/git/clip-lab/.venv/bin/python` running PyTorch 2.13.0. The Python training environment was reused without modifying it. A new checkout can create its own environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python notebooks/validate.py
.venv/bin/python notebooks/train.py --epochs 65 --steps 1800
```

The notebook contains **eight executed code cells**. Open it with the new environment's Python kernel. It obtains the exact numerical fixtures by invoking the same local Node module as the lecture.

## Recorded runs and interpretation

All neural demonstration outputs were freshly generated for this lecture on 19 October 2026. `assets/recorded/provenance.json` records seeds, hyperparameters, losses, versions and elapsed time. The VAE has a 2D latent and width-128 encoder/decoder. Training uses 2,048 binary procedural seven-segment glyphs, not MNIST, and 65 epochs. Held-out glyphs use a different seed. The browser runs the actual exported decoder weights; its outputs were compared against PyTorch.

The toy GAN uses an equal mixture of two Gaussians centred at (−2,0) and (2,0), standard deviation 0.25, and 1,800 alternating updates. Every checkpoint displays the same 256 latent draws. Readout losses are recomputed on exactly the displayed real/fake points; the separately labelled history curves use training batches. Toy realism means distance under 0.65 from either centre. Coverage counts centres with at least five generated samples within that radius.

A short image-GAN run uses the specified 32→128→256→784 generator and 784→256→128→1 discriminator. It demonstrates the code and its limitations; it is not a claim of convergence or competitive image quality. Both models deliberately omit BatchNorm and Dropout, keeping the parameter-freezing demonstration unambiguous.

The opening MNIST figures are published AEVB results, independently attributed. The local seven-segment examples are distinct. The sleep signals are a labelled proposed application. Satellite generations are possible images, not measurements. No synthetic demonstration claims a validated downstream benefit.

## Verification

See [QA report](qa/QA_REPORT.md) for mathematical, model, browser, reading and PDF checks. Every reveal was captured; every completed logical scene was inspected in contact sheets, with dense mathematical pages also checked individually. The gallery crops preserve source/input/output correspondence. Numerical infinity is represented explicitly, not hidden behind a finite substitute.

The brief is retained in `SPECIFICATION.md`. One-off acquisition material and debugging logs stay in ignored `work/`; final outputs are in `exports/`, with compact verification records in `qa/`.
