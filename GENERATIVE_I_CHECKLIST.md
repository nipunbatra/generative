> Historical checklist for the first edition. The current audit is [GENERATIVE_I_REBUILD_CHECKLIST.md](GENERATIVE_I_REBUILD_CHECKLIST.md).

# Generative Modelling I — verification checklist

Created before implementation. Checkboxes require recorded evidence.



## Narrative
- [x] One central question throughout
- [x] Motivation comes before model names
- [x] Distribution intuition precedes VAE
- [x] Prior is defined visually and mathematically
- [x] Autoencoder sampling problem motivates VAE
- [x] GAN introduced as alternative training signal
- [x] Flow introduced as invertible alternative
- [x] Diffusion teaser ends the lecture

## Mathematical accuracy
- [x] Density vs probability distinguished
- [x] Data distribution vs model distribution distinguished
- [x] Prior vs posterior vs decoder likelihood distinguished
- [x] VAE parameter dimensions checked
- [x] Reparameterization distribution identity correct
- [x] Reparameterization numerical example checked
- [x] Reparameterization gradient example checked
- [x] KL formula correct
- [x] KL numerical example checked
- [x] ELBO signs correct
- [x] Reconstruction likelihood caveat included
- [x] GAN minimax vs practical generator loss distinguished
- [x] Flow Jacobian sign/direction correct

## Visual quality
- [x] Meaningful visual on ≥70% of main slides
- [x] Same examples and colors reused consistently
- [x] One concept per reveal
- [x] No future arithmetic visible prematurely
- [x] All equation typography correct
- [x] All diagrams readable from projector distance

## Experiments
- [x] 2D distribution scripts reproduce figures
- [x] VAE train script runs
- [x] VAE prior samples measured
- [x] VAE latent grid measured
- [x] Interpolations reproducible
- [x] GAN run measured if presented
- [x] Recorded fallback works
- [x] Figure provenance captured

## Final QA
- [x] Every main build rendered
- [x] Every main slide visually inspected
- [x] No overflow/clipping
- [x] No malformed mathematics
- [x] No tiny text
- [x] Appendix separated
- [x] PDF export checked
- [x] Checklist and changelog completed


## Verification evidence — 12 October 2026

- Narrative and mathematical scope: `src/story.py`, `TEACHING_GUIDE.md`, `SOURCES.md`; reviewed in order. Main route is 50 screens / 80 builds / 4,800 seconds. Reparameterization occupies 15 progressive builds (screens 23–29).
- Exact numerical and autograd checks: `metrics/analytical-checks.json`; z=1.6, reconstruction derivatives 3 and −2.4, logvar derivative −.6, KL=.818147 nats, interval probability=.682689, inverse-Jacobian convention, frozen-D gradient and EM convergence all checked.
- Dimensions: actual checkpoint architecture and saved decoder parity checks; mean/logvar have shape B×d; output logits have shape B×784.
- Visuals: 42 of 50 main screens have a diagram, image or interactive (84%; mathematical equations are not needed to meet the threshold). Every main build was rendered. All 50 final-state pages were visually inspected in PDF contact sheets; crowded/arithmetic/graph pages were also inspected at full size. Revised pixel-order, code and gradient layouts were rechecked. The small provenance footer is intentionally secondary; teaching labels use larger type.
- Browser: `output/qa/browser-report.json`, 31 checks, zero failures; all 80 main builds and 5 backups rendered; controls, mobile, offline, absent-weight fallback, default values, endpoints, hidden answers and decoder parity tested.
- Experiments: three completed 15-epoch CPU runs; data/checkpoint hashes and per-epoch curves in `metrics/mnist.json`; learned 2D density fit in `metrics/mixture-fit.json`. Both interpolation montages recomputed from saved weights with zero pixel differences.
- “GAN run measured if presented”: conditional requirement satisfied because no GAN training run is presented. The mode-collapse example is explicitly constructed and labelled ILLUSTRATIVE. The optional measured GAN run was intentionally omitted.
- Recorded fallback: manually toggled and automatically selected with weights removed in a browser test; it uses real saved 25×25 decoder outputs and reports the snapped coordinate.
- Notebook: 8 code cells executed without error, using included checkpoints offline. The notebook does not claim to repeat training unless RUN_TRAINING is enabled.
- PDF: 50/80/5 pages, 16:9, no blank pages, math text present; an early arithmetic build excludes the final answer while its final build includes it. `output/qa/pdf-report.json` records checks. Every interactive has a static export state; learned/analytical density comparison snapshots are tailored for print.
- Checklist/change log: this file and `GENERATIVE_I_CHANGELOG.md` completed after verification.

Projector-distance readability is a visual review judgment at the authored 1280×720 size, not a test in a physical classroom. No claims of classroom testing, deployment, GAN convergence or state-of-the-art MNIST performance are made.
