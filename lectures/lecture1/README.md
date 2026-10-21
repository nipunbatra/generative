# Lecture 1: KL and variational inference

An 80-minute main route of 53 slides in five parts, with 15 optional slides after it. The running example is a tiny Bayesian neural network: one input, two tanh hidden units and one output, so θ is seven numbers, each with a N(0, 1) prior, fitted to 17 points with a gap. The bottom bar carries the story line; questions and answers live in the speaker notes, apart from the marked pauses where students predict before the reveal.

| Part | What it covers |
|---|---|
| 1 · Bayes’ rule for a neural network | Bayes’ rule, then for parameters and data; the network and all seven parameters with their priors; one draw of θ as one function, then twenty draws with each θ shown on hover, and the prior predictive; θ_A (the MAP) walked through: the likelihood one data point at a time, the prior one parameter at a time; θ_A against a random θ; the evidence as a 7-dimensional integral (a grid needs 10¹⁴ points, prior sampling needs millions of draws for about 100 useful ones); the posterior predictive band from an MCMC reference; Laplace, MCMC and VI; the VI idea, one Gaussian per parameter |
| 2 · How far apart are two distributions? | Same mean and variance, different shapes; the KL ladder (fruit surveys, ratios, logs, weighting, message length, cross-entropy, how fast evidence piles up, KL ≥ 0); closed forms (Bernoulli, categorical, Gaussian, derived); the PML vote; KL as an area (live); Monte Carlo with code, and a live estimator |
| 3 · Find the closest q | Six candidates; a grid search over (μ, σ); the same search as gradient descent, in code and animated; a diagonal q in 2-D; the mixture with no formula, the `sample()` loop that does not move, the reparameterisation trick over five slides, and the `rsample()` fit; forward vs reverse KL |
| 4 · Back to Bayes: the ELBO | Why KL(q ‖ posterior) needs p(𝒟); substituting Bayes’ rule step by step; the ELBO as fit-the-data minus KL to the prior; the identity checked live on a coin with a known p(𝒟) |
| 5 · Implement it | The network’s ELBO in seven lines; VI animated from random start to fit; what q learned (one hidden unit switched off); VI against the MCMC reference: the right curve with too-narrow bands, and a 5.8-nat gap to log p(𝒟) |

Eight slides carry live figures (`widgets.js`): KL as a net area, a Monte Carlo estimator, forward vs reverse fits, q sliding onto p, the `sample()` and `rsample()` mixture fits, the coin’s ELBO + KL slider and VI on the network. Each animation replays a recorded run from `data/science.json`: frame 0 until you advance, then it plays. Print and PDF show the start and end frames as separate builds.

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
