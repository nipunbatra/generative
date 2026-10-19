# Generative Modelling I — From Randomness to Data

ES667 · Nipun Batra · 95-minute full route / 80-minute core. Distributions, priors, latent variables, VAEs and GANs.

Open **[index.html](index.html)**. The lecture works offline, including mathematics, interactive figures, the saved VAE decoder and recorded fallback. Keep the `vendor/` directory beside the HTML. No paid API or model service is used.

Online: **[Open the lecture](https://nipunbatra.github.io/generative/)** · [Resources](https://nipunbatra.github.io/generative/resources.html) · [Source repository](https://github.com/nipunbatra/generative).

## Publication

GitHub Pages serves the committed files from the root of `main`. `.nojekyll` keeps the prebuilt lecture and local assets unchanged. Rebuild and verify locally before committing generated changes; pushing `main` publishes the update. PDFs, the executed notebook, model checkpoints and compact QA reports are included in Git. Raw QA screenshots and the complete ZIP remain local and can be regenerated with the scripts below.

## Deliverables

- [Interactive lecture](index.html): 87 conceptual screens, 208 main builds, 5 optional backup screens.
- [Compact PDF](output/pdf/generative-i.pdf): 87 final-state conceptual pages.
- [Every-build PDF](output/pdf/generative-i-all-builds.pdf): 208 pages preserving the teaching sequence. Use this PDF for teaching without a browser; the compact version shows each concept once; the interpolation uses its midpoint.
- [Backup PDF](output/pdf/generative-i-backup.pdf): derivation, objective distinctions, measured protocol and references.
- [Executed companion notebook](notebooks/generative_i_lab.ipynb): saved-model inspection, checked arithmetic and optional retraining.
- [Presenter guide](TEACHING_GUIDE.md): 95-minute timing, discussion questions and expected answers.
- [Edition 4 checklist](GENERATIVE_I_V4_CHECKLIST.md), [storyboard](GENERATIVE_I_V4_STORYBOARD.md), [changelog](GENERATIVE_I_CHANGELOG.md), [provenance](SOURCES.md), [QA](output/rebuild-qa/QA_REPORT.md).

This is a separate local project in `~/git/generative`. The attention-series reference repositories were only read. Any `diffusion/` directory is separate work and is outside this lecture's build and package.

## Probability detours and linked reading

The full route adds five minutes on ELBO construction and ten on probabilistic GANs. The [storyboard](GENERATIVE_I_V4_STORYBOARD.md) lists the nine optional detour screens; skipping them restores the existing 80-minute route without reducing the measured-demo time. The full lecture has 87 screens and 208 reveals.

The ELBO detour reweights the same finite evidence calculation, draws Jensen’s inequality on a log curve, then shows the computable training objective. GANs start with a source-classification experiment, derive Bernoulli log loss and the Bayes discriminator, and connect the ideal minimax game to Jensen–Shannon divergence. Interactive density and logit examples explain the distribution comparison and non-saturating loss. These analytical examples are explicitly separate from the saved neural-training run.

[Resources and further reading](resources.html) contains all 17 linked references, including both Lilian Weng articles, Jaan’s tutorial, Nipun’s two lectures and PML notebook, MIT 6.S191, primary papers and implementation references. Links also appear in contextual notes, the teaching guide, notebook and PDF reference page.

## Motivation, inference and measured models

The opening now tests image replay and an independent-pixel generator before introducing latent variables. A generation/inference diagram leads to marginalization, a shared approximate encoder, a three-screen KL detour, and a visible ELBO gap. All new diagrams are authored SVG; no blog diagrams are embedded. Sources and teaching references are credited in `SOURCES.md`.

`experiments/pedagogy_evidence.py` reproduces the new baselines, exact teaching fixtures, autograd checks and beta sweep. The sweep uses **three separately trained 2D VAEs**, beta 0/1/4, the first 12,000 binary training images, 12 epochs, batch 256, Adam 0.001 and seed 667. Initialization, data order and noise protocols are shared. Both BCE and KL sum their respective dimensions before averaging examples. Test metrics use one posterior sample on each of all 10,000 held-out images. This single-seed teaching comparison is separate from the original full-data run below.

| Beta | Test reconstruction, nats/example | Test KL, nats/example |
|---|---:|---:|
| 0 | 152.883 | 42.707 |
| 1 | 156.218 | 5.162 |
| 4 | 162.816 | 3.042 |

Saved hashes, histories and the exact protocol are in `metrics/pedagogy.json`; browser recordings are in `results/pedagogy.json`. The beta controls change the saved checkpoint, not an untrained model parameter. The posterior plot uses the same axes across models.

Browser export and QA use Puppeteer's bundled browser when available, then fall back to installed Chrome/Brave on this Mac. `PUPPETEER_EXECUTABLE_PATH` can select another compatible installation. The known working Python environment here is `/Users/nipun/git/clip-lab/.venv/bin/python`; a fresh checkout can install `requirements.txt` in its own environment.

## Teaching controls

Arrow keys / Space: reveal the next step. Left arrow: go back through builds. N: next concept. S: presenter notes. Q: discussion and optional answer. O: overview. P: present / reading mode. The Controls menu reveals backups and file links. On a phone, reading mode is the default.

Teaching interactions are included for: categorical sampling, Gaussian mean/spread/count, interval probability, mixture components, MLE, prior sampling, AE/VAE comparisons, posterior ellipses, reparameterization graph steps, gradient steps, KL, the latent decoder plane, interpolation, GAN alternation, measured GAN snapshots, invertible deformation, and the noise teaser. The new KL detour adds editable discrete probabilities, a two-mode comparison in both KL directions, and an exact evidence-bound explorer. Three beta buttons select separately trained checkpoints. Sliders have keyboard support. The decoder plane supports clicking and arrow keys.

## Run locally

Double-click `index.html`, or serve the folder:

```sh
cd ~/git/generative
python3 -m http.server 8766 --bind 127.0.0.1
```

Open <http://127.0.0.1:8766>. The source is plain Python, HTML, CSS and JavaScript, following the existing attention lecture's slide-mode pattern.

## Reproduce training and figures

Python 3.12 was used. The exact observed versions are pinned in `requirements.txt`; `metrics/mnist.json` records versions, seed, data and checkpoint SHA-256 hashes. No package installation is needed just to present the lecture.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python experiments/train_mnist.py --epochs 15 --seed 667
.venv/bin/python experiments/distributions.py
.venv/bin/python experiments/teaching_numbers.py
.venv/bin/python experiments/rebuild_evidence.py
.venv/bin/python experiments/pedagogy_evidence.py
.venv/bin/python experiments/verify_math.py
.venv/bin/python experiments/verify_rebuild.py
.venv/bin/python experiments/verify_pedagogy.py
.venv/bin/python experiments/verify_probability.py
```

The first training run downloads MNIST from the public CVDF mirror into the ignored `experiments/data/` directory. Later runs use that cache. The included checkpoints, images and notebook inspection work without downloading anything.

The saved run used the existing environment `/Users/nipun/git/clip-lab/.venv/bin/python`; its environment was not modified. The complete three-model CPU training took approximately 42 seconds on this machine. Timing and bitwise reproducibility can vary across hardware and library versions.

### Measured protocol

- Fixed binary preprocessing: original uint8 pixel > 127. Bernoulli output likelihood.
- 60,000 train and 10,000 held-out test examples; labels unused by training.
- MLP encoder 784 → 256 → 128, ReLU hidden layers; mean/logvar heads for the VAEs.
- Decoder d → 128 → 256 → 784 logits, ReLU hidden layers. d=2 and d=16.
- Adam 0.001, batch 256, 15 epochs, seed 667, CPU deterministic operations, four threads.
- Standard negative ELBO, β=1: sum pixel BCE and latent-coordinate KL per example, then average the batch.
- Ordinary AE-2 uses the same reconstruction objective, a deterministic code and no KL.
- Test VAE metrics use one posterior draw per example at seed 668. These are instructional results, not a tuned benchmark.

| Model | Test reconstruction (nats/example) | Test KL (nats/example) |
|---|---:|---:|
| AE-2 | 133.197 | 0 |
| VAE-2 | 135.392 | 6.339 |
| VAE-16 | 69.963 | 22.023 |

Reconstruction panels use the decoder mean at the posterior mean. Grid and interpolation panels also show decoder means. Files ending `prior-draws.png` additionally sample binary observations. The panels use fixed first examples or fixed random seeds; no outputs were retouched. Small, blurred and imperfect digits remain visible.

The 2D cloud uses 35 Gaussian draws per mode for a balanced teaching visualization. A separate full-covariance Gaussian mixture is actually fitted to these 280 observations by EM; its learned parameters and convergence history are in `metrics/mixture-fit.json`. The single-Gaussian/known-mixture comparison is analytical. The mode-collapse sampler and flow shear are explicitly illustrative. A separate small GAN is trained on iid draws from the same mixture. Its five saved snapshots, losses and coverage diagnostics are in `results/toy-gan.json`; its weights are in `models/toy-gan.pt`. It uses two 64-unit hidden layers per network, 2D noise, Adam 0.0005 with betas (0.5, 0.999), batch 256, 2,000 alternating D/G steps and seed 667. No run or snapshot was chosen for visual success. At step 2,000 it covers 8/8 modes under the stated threshold, but only 77.6% of its fixed 1,000 samples lie within three component standard deviations of a centre. The gaps still receive too much mass.

### Browser decoder and fallback

`models/vae2-browser.json` stores exported float weights and full recorded outputs. The build embeds the weights and an 8-bit display recording of the 25×25 grid. Live clicks run the saved MLP locally. The fallback button uses the nearest recorded grid location and reports that location. It never invents a decoded image. Browser predictions are compared against three saved PyTorch vectors in the browser verification.

## Build and verify

Node dependencies are declared in `package.json`. On this machine the ignored `node_modules` symlink reuses the existing `dl-teaching` installation. For an independent copy:

```sh
npm install
python3 src/build.py
node src/verify.cjs
node src/audit_geometry.cjs
node src/verify_pedagogy.cjs
node src/verify_probability.cjs
.venv/bin/python experiments/verify_pedagogy.py
.venv/bin/python experiments/verify_probability.py
node src/export.cjs
.venv/bin/python src/verify_pdf.py
python3 src/verify_links.py
python3 src/package_release.py
```

`src/build.py` writes the offline HTML, manifests and teaching guide, and prerenders every equation with local KaTeX. `src/verify.cjs` renders every main build and backup, exercises the browser interactions, checks dimensions and mathematical rendering, validates decoder parity and saves source SVG snapshots. `src/audit_geometry.cjs` checks label-label, label-stroke and node-border collisions through every reveal and slider endpoints. `src/export.cjs` writes three vector PDFs with recorded static canvas states. `pdfinfo` and Poppler renderings were used for the final PDF check.

Regenerate the unexecuted notebook source with `python3 src/build_notebook.py`. Execute it with Jupyter in the installed environment; leave `RUN_TRAINING=False` for offline checkpoint inspection, or enable it to repeat training. After training, rerun the HTML build and QA to synchronize all outputs.

## Source map

`src/probability_story.py` adds the probabilistic detours, `src/probability.js` draws their figures, and `src/resource_catalog.py` supplies the linked bibliography. `src/pedagogy_story.py` defines the core route and individual timings; `src/pedagogy.js` draws the new original diagrams and `src/pedagogy.css` styles them. `src/story.py` preserves the original route; `src/rebuild_story.py` defines the revised diagrams, reveals and notes. `src/figures.js` supplies shared primitives; `src/rebuild.js` owns the large teaching graphs and measured interactions. `src/lecture.js` owns navigation and interactions. `src/lecture.css` and `src/rebuild.css` adapt the Beyond Attention/CLIP shell to the supplied palette. `experiments/` contains training and analytical reproduction. `models/` and `metrics/` preserve evidence. `figures/` contains measured panels and reusable SVGs. `images/` contains the credited recurring photograph. Scratch files and cached data are excluded from the release bundle.

## Rebuild evidence

The shared numerical fixture is `results/generative_i_teaching_numbers.json`, generated and asserted by `experiments/teaching_numbers.py`. Independent PyTorch and checkpoint checks are in `experiments/verify_rebuild.py`. Browser checks verify both exported decoders, interaction state, reveal arithmetic, topic preservation, the 95-minute full route and the 80-minute core.

The large latent explorer decodes the selected coordinate, displays its 3×3 neighborhood at spacing 0.25, and marks the same point under the prior. Its fallback uses measured grid coordinates. VAE-16 interpolation uses 41 checkpoint evaluations, not image blending. The AE/VAE prior comparison uses the same 2D vectors and held-out inputs.

The final six dog frames are byte-identical to the opening sequence in `diffusion/`. Their hashes and forward-process settings are saved in `results/bridge-provenance.json`; the local files are included so the lecture remains independent of the other folder. The reversal poses a question and is not a learned denoising result.

The revised route allocates 7 minutes to motivation, 20 to inference/KL/ELBO, 13 to reparameterization, 17 to losses and measured VAEs, and 20 to GANs. The decoder explorer has 4 minutes and the beta comparison 3. Treat both route allocations as a teaching plan, not a measured classroom rehearsal.

The baseline source, original release ZIP and unchecked initial checklist are preserved in `work/rebuild-before/`. The revision audit and storyboard are in `GENERATIVE_I_REBUILD_AUDIT.md` and `GENERATIVE_I_REBUILD_STORYBOARD.md`.
