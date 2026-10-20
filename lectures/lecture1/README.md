# Lecture 1: KL and variational inference

A 70-minute core route with 66 logical scenes, plus six optional derivations. Slides lead; three short notebook demonstrations cover Gaussian optimisation, Monte Carlo/gradient inspection and joint-only coin fitting. Pacing is planned, not classroom-rehearsed.

The lecture builds from available target densities to an unknown posterior:

1. Compare two Gaussian approximations using directional KL.
2. Inspect one literal gradient step and a measured full fit.
3. Replace an unavailable library formula with a Monte Carlo estimate.
4. Diagnose missing sampling gradients and derive reparameterisation.
5. Rearrange Bayes' rule into the ELBO identity.
6. Fit a Beta coin approximation using only its joint density.
7. Retrieve the method and transfer it to another inference problem.

The selected Beta candidates used to verify the identity are distinct from the measured optimisation trajectory. Negative Monte Carlo estimates, imperfect final fitting, fixed seeds and loss reductions are explicitly labelled.

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

- `content.js`: all slide content, builds, questions, answers, misconception cues and transitions.
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
