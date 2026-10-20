# Lecture 1: KL and variational inference

A 70-minute main route of 39 slides in seven acts plus an exit, with 14 optional slides after it. The lecture follows `plan/2026-10-20-review/STORY-REVIEW.md`: each act opens with an obstacle the previous act exposed and ends on a picture. The bottom bar carries the story line; questions and answers live in the speaker notes, apart from ten marked pauses (PML moments where students predict before the reveal).

| Act | Opens with | Ends with |
|---|---|---|
| 0 · The wall | Is this coin fair? With a Beta prior Bayes has a formula; with a Normal prior on the log-odds it does not. A grid needs 100^d points. | Laplace, MCMC and VI on one two-bump target; the plan and its five obstacles |
| 1 · Closer how? | Two fruit surveys: ratios, then logs, then weighting give KL (0.42 nats) | KL as message length, as cross-entropy minus entropy, as the rate evidence piles up (325× after 30 tosses), as an area (live), and why VI uses KL(q‖p): the reverse fit picks one bump (animation) |
| 2 · Make q move | A score picks a winner. Can it improve a loser? | q sliding onto p (animation); a diagonal q shrinking to widths 0.44 inside a correlated target |
| 3 · No formula | `kl_divergence(q, mixture)` raises NotImplementedError | Monte Carlo, checked on a solvable pair and tightening as S grows |
| 4 · Gradients through a draw | The `sample()` loop runs and q does not move (animation) | Reparameterisation with a pause for ∂u/∂μ and ∂u/∂σ, then `rsample()` fitting the mixture (animation) |
| 5 · The real target | KL to a posterior needs the evidence | Students substitute Bayes' rule; the ELBO's two forces; log p(𝒟) = ELBO + KL as one fixed total |
| 6 · Back to the wall | VI on the coin with no formula (animation), from the prior to within 0.0006 nats of the grid evidence | Is the coin fair? P(r > 0.5) ≈ 0.75; the Beta coin as an answer key; a live ELBO + KL slider |
| Exit | Five obstacles, five tools, and where each reappears in the VAE | A transfer question and the bridge to Lecture 2 |

Six slides carry live figures (`widgets.js`): KL as a net area, forward vs reverse fits, q sliding onto p, the `sample()` and `rsample()` mixture fits, VI on the logit coin, and the Beta ELBO slider. Each animation replays a recorded optimisation from `data/science.json`: frame 0 until you advance, then it plays. Print and PDF show the start and end frames as separate builds.

## Use

Open `index.html` through the series HTTP server. Root integration generates the presentation page, speaker notes and completed/all-build PDF exports from `content.js`.

The self-contained `lecture1-variational-inference.ipynb` includes saved outputs and runs top-to-bottom with PyTorch, NumPy, matplotlib and IPython. It uses CPU float64, no downloads and no external helper imports.

From the `lectures/` directory, using the Python environment described in the series README:

```sh
python lecture1/scripts/build_notebook.py
node lecture1/scripts/check-science.mjs
```

The builder uses the invoking Python interpreter for the notebook kernel. Regenerating the notebook also creates original SVG/PNG figures, full-precision fixtures and a verification record. `scripts/run_science.py` is a generated code-only copy; run it from this lecture directory if a notebook kernel is unnecessary.

## Files and evidence

- `content.js`: all slide content, builds, story lines, notes (questions, answers, misconception cues, transitions) and static figures.
- `widgets.js`: the live figures and the small SVG chart kit the static figures share.
- `lecture1-variational-inference.ipynb`: executed self-contained notebook.
- `data/science.json`: full-precision measurements, trajectories, plotted curves, seeds and configurations.
- `math.mjs`: independent JavaScript analytic formulas, adapted unchanged from the checked prototype.
- `figures/`: original deterministic plots from the notebook execution.
- `qa/notebook-checks.json`: in-notebook execution assertions.
- `qa/science-checks.json`: independent checks of recorded values and plotted curves.
- `qa/SCIENCE.md`: verification scope, tolerances and limitations.
- `qa/svg-text-bounds.json`: all-state inline SVG text bounds, checked by `scripts/check-svg-bounds.mjs` with the series server running.
- `figures/provenance.json`: source and generation provenance for each figure.

The original PML sequence is adapted from Nipun Batra's [2023 VI notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/variational-inference.ipynb). This implementation corrects stale computation graphs, positive-parameter constraints and reduction ambiguity. It does not reuse the historical notebook's hidden training helpers or stored training measurements.
