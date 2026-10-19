# Learning to generate · ES 667 Generative Modelling I

A rebuilt lecture: autoencoders, KL divergence, the ELBO, reparameterisation, the VAE and GANs,
taught slowly around **one running example** (MNIST test digit #90, a handwritten 3) and one 1-D toy dataset (two bumps at ±2).
Every figure and every number on the slides comes from models trained by the scripts below.

Two 75-minute sessions:

| Deck | Slides | Content |
|---|---|---|
| `part1.html` · [exports/session1.pdf](exports/session1.pdf) | 63 | why generate, distributions, autoencoders, KL, the ELBO (derived one move per slide), the Bernoulli loss |
| `part2.html` · [exports/session2.pdf](exports/session2.pdf) | 51 | recap, reparameterisation, the VAE on MNIST, GANs, wrap-up |
| `index.html` · [exports/lecture.pdf](exports/lecture.pdf) | 114 | both sessions in one deck |

`*-builds.pdf` versions show every click.

## Open it

```sh
python3 -m http.server 8771
```

Then open http://127.0.0.1:8771/part1.html (or `part2.html`, `index.html`).

- → / Space next click · ← back · Shift+→ next slide · **O** overview · **S** speaker notes · **F** fullscreen · `?reading` for all slides with notes

## Review notes (right-click annotations)

- **Right-click** anything on a slide → type a note → **Save** (or ⌘/Ctrl+Enter). A numbered red pin marks the spot; click it to edit or delete.
- The note records the slide, the click (build) number and what you clicked: for an equation, its exact LaTeX.
- Press **A** to list all notes, then **Copy all** and paste them into the chat. If you review in the Claude app's browser pane, just say "read my notes": Claude reads them with `window.deckNotes()`.
- Shift+right-click still opens the normal browser menu. Notes live in that browser's local storage, one list per deck file.

## Maths

All maths is typeset by KaTeX (slides, diagram labels) or matplotlib's Computer Modern (charts), so every symbol matches.
Shared macros (`\N`, `\KL`, `\E`, `\eps`, `\ELBO`, role colours `\cx \cenc \cdec \cz \closs \cdisc`, and `\hl{}` for "this part just changed")
are defined once in `scripts/build_deck.py`. The PDF export fails if any maths character appears outside KaTeX.

## Rebuild everything

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python   # any Python with torch + torchvision + matplotlib
$PY scripts/train_models.py     # ~10 min on an M-series GPU; MNIST read from $MNIST_ROOT, nothing downloaded
$PY scripts/make_figures.py     # figures/data/*.svg|png and figures/data/numbers.json
$PY scripts/diagrams.py         # figures/diagrams/*.svg (hand-built SVG, numbers read from numbers.json)
$PY scripts/build_deck.py       # inlines diagrams, fills {{numbers}}, writes part1.html, part2.html, index.html
node scripts/export_pdf.mjs     # session1/2 and lecture PDFs (+ -builds), and the maths check
```

## Where things live

| Edit this | To change |
|---|---|
| `deck/slides/01-hook.html` … `09-wrap.html` | slide text, builds (`data-b="2"`), speaker notes (`<aside class="notes">`). 01–05 = session 1, 06–09 = session 2 |
| `scripts/diagrams.py` | the SVG diagrams (encoder/decoder, VAE, reparameterisation, GAN steps …) |
| `scripts/make_figures.py` | every chart and digit grid, and `numbers.json` |
| `scripts/train_models.py` | the AE / VAE / GAN models and the exact `gan_step` shown on the slides |
| `deck/deck.css`, `deck/deck.js`, `deck/widgets.js`, `deck/annotate.js` | look, navigation, the three live widgets, and right-click review notes |

Build syntax: `data-b="3"` appears from click 3 on, `data-b="2-4"` only during clicks 2–4, `data-b="=1"` only on click 1, `data-hl="2"` highlights from click 2.

## What is real

- AE and VAE (2- and 16-number codes, plus a β = 8 variant): binarised MNIST, MLP 784-512-256-z, 30–40 epochs.
- MNIST GAN: grey-level MNIST, small DCGAN, 30 epochs, trained with the same `gan_step` as on the slides.
- 1-D GANs: 8 seeds on the two-bump data (all cover both bumps by step 3,000), plus one run with a 4-unit generator that collapses to one bump.
- ELBO gap for our 3: log p(x) by grid integration over the 2-D code space (−128.7), ELBO by 20,000 samples (−129.3), and the gap computed independently as KL(q ‖ true posterior) = 0.61.
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
