# Lecture 1: KL and variational inference

A main route of 117 slides (about 140 minutes) in five parts and twenty-four numbered sections, with an appendix and optional slides after it (26). The running example is a tiny Bayesian neural network: one input, two tanh hidden units and one output, so θ is seven numbers, each with a N(0, 1) prior, fitted to 17 points with a gap. The bottom bar carries the story line; questions, answers and **when to switch to the notebook** live in the speaker notes (press N).

**Sections and the notebook.** The first slide is a clickable map of the twenty-four sections. Every slide’s header names its section (for example *3.5 The reparameterisation trick*), and the **▶ Notebook** pill at its top right opens the matching notebook cell in Colab. The notebook uses the same numbers and headings, and every notebook section links back to its slides. The anchors are stable cell ids (`sec-3-5`, `cell-fit-code`, …) listed in `data/notebook-anchors.json`; `content.js` refuses to load if a slide points to a missing one.

| Part | Sections |
|---|---|
| 1 · Bayes’ rule for a neural network | 1.1 Bayes’ rule · 1.2 A tiny network and its prior (one draw, twenty draws with θ on hover, the prior predictive) · 1.3 Likelihood and training (MAP training animation, θ_A, likelihood and prior as formulas then values) · 1.4 The evidence is intractable (a 1-D Riemann sum, then 10¹⁴ grid points) · 1.5 The posterior, and how to approximate it (posterior predictive, the core problem, Laplace/MCMC/VI, the VI idea) |
| 2 · How far apart are two distributions? | Built from information theory, after Jia-Bin Huang’s video “Fantastic KL Divergence and How to (Actually) Compute It”: 2.1 Surprise (why log: five days multiply, surprises add) · 2.2 Entropy as average surprise (why an expectation: simulated days; 0 to 1 bit; yes/no questions) · 2.3 Codes (prefix codes, Huffman, codeword length = surprise) · 2.4 Cross-entropy and KL (a wrong belief, 1,000 days through both codes, the extra bits, asymmetry) · 2.5 Classification loss (the loss is the surprise at the true label; one-hot: cross-entropy = KL; soft labels) · 2.6 Maximum likelihood on twenty recorded days (average surprise = H(p̂, q) = H(p̂) + KL) · 2.7 KL at work (knowledge distillation: a big frozen teacher, a small student, the loss term by term, soft labels and temperature, and a 27-parameter student learning a 4,547-parameter teacher; RLHF: policy, reward model and reference, reward hacking, the exact best policy π* ∝ π_ref e^{r/β} on a β slider, and the per-token KL estimate) · 2.8 Closed forms and PyTorch · 2.9 KL by sampling. The grades route (after ritvikmath) is now an appendix |
| 3 · Find the closest q | 3.1 Search a grid (hover any cell to see that q against p) · 3.2 Gradient descent (the code, and the loss and curve animated) · 3.3 Covariance in 2-D (isotropic, diagonal, full; all three fitted from one random start, with Σ shown as it changes) · 3.4 Mixtures: no formula (pick a bump, then draw; live two-step sampling; the `sample()` loop that does not move: is it a bug?) · 3.5 The reparameterisation trick (computational graphs with stochastic and deterministic nodes, after Lilian Weng; θ = μ + σε; `rsample()`) · 3.6 Which direction of KL? |
| 4 · Back to Bayes: the ELBO | 4.1 Deriving the ELBO one rule per slide (Bayes in logs; substitute; a constant leaves an average; rearrange and name the ELBO; what each piece is; why KL ≥ 0; what “lower bound” does and does not say; raising the ELBO lowers the KL; fit the data, stay near the prior), after Gundersen · 4.2 Checking it on a coin |
| 5 · Implement it | 5.1 VI for the tiny network (the ELBO in code, VI animated, what q learned, mean-field against full-covariance VI, both against the MCMC reference) · 5.2 Wrap-up |

Fourteen slides carry live figures (`widgets.js`): the prior functions with θ on hover, MAP training, grade sampling, KL as a net area, a Monte Carlo estimator, the KL grid with hover, q sliding onto p with its loss, three covariance families fitting one target, two-step mixture sampling, the `sample()` and `rsample()` mixture fits, forward vs reverse fits, the coin’s ELBO + KL slider and VI on the network. Each animation replays a recorded run from `data/science.json`: frame 0 until you advance, then it plays, with a slider to scrub. Print and PDF show the start and end frames as separate builds.

The main route is long for one class. A natural split is after Part 2 (Bayes and information theory to KL, about 95 minutes) and Parts 3–5 (fitting q, the ELBO and VI, about 45 minutes).

## Use

Open `index.html` through the series HTTP server. Root integration generates the presentation page, speaker notes and completed/all-build PDF exports from `content.js`.

The self-contained `lecture1-variational-inference.ipynb` includes saved outputs and runs top-to-bottom with PyTorch, NumPy, matplotlib and IPython (plus `ipywidgets` for the optional step slider, present in Colab). It uses CPU float64, no downloads and no external helper imports, and runs in about three minutes. Open it in Colab: <https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture1/lecture1-variational-inference.ipynb>.

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
