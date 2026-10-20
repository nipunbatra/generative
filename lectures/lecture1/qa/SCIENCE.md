# Lecture 1 science verification

The saved notebook executed all **18 code cells** in the existing clip-lab environment, using CPU and PyTorch float64. No dataset or package download was required. Its two toy optimisation runs were executed for this package; the traces are measured outputs.

The independent JavaScript check passed **2,780 assertions**, including **2,005 numerical comparisons**. The largest absolute discrepancy was approximately **6.00 × 10⁻¹²**. The detailed comparison records are in `science-checks.json`.

## What was checked

- The Normal KL formula agrees with PyTorch, numerical integration and reversed-argument checks.
- The literal mean update agrees with the analytic derivative and a central finite difference.
- Every recorded Normal trajectory state has a positive scale and the correctly recomputed analytic KL.
- Each selected Beta candidate's ELBO was calculated directly from expected log likelihood + expected log prior − expected log q.
- Independent Beta KL and evidence calculations verify the identity, including every recorded coin trajectory state.
- Numerical integration independently checks the three selected Beta ELBO values.
- Every recorded coin state has positive shape parameters. The training loss uses joint-minus-q values, not the posterior reference.
- The five-draw negative KL example is the recorded mean of its displayed log ratios.
- The repeated Monte Carlo experiments' empirical statistics match their saved samples.
- The missing `sample()` gradient path and the differing naive detached/pathwise derivatives are recorded.
- Both pathwise derivatives agree with fixed-noise finite differences.
- Every displayed Normal/Beta density curve checked against the independent formula implementation agrees within tolerance.
- Invalid Normal scales and Beta shape parameters are rejected.

## Tolerances and meaning

Normal formula comparisons use `2e-12`, Beta analytic comparisons up to `7e-11`, Beta plotted-density comparisons `2e-10`, and finite-difference/integration checks `1e-8`. These are floating-point checks, not claims that printed rounded decimals are exact identities.

For the mixture there is no asserted analytic KL formula. A wide-domain numerical integral is checked against a 100,000-draw Monte Carlo estimate within five estimated standard errors. The selected results differ by about 2.1 and 0.6 standard errors respectively.

The final measured Gaussian KL is about `4.36e-9`. The final coin approximation has `a≈22.3936`, `b≈16.1893` and posterior KL about `0.00022154`. The coin fit is close to its reference, not exactly equal.

The three selected Beta candidates `(10,10)`, `(30,20)` and `(22,16)` are independent identity-checking points, not purported training checkpoints.

## Conventions and scope

- `Normal(loc, scale)` uses standard deviation. Mathematical `N(μ,σ²)` uses variance.
- The Gaussian optimiser stores `ρ=log σ`; it does not use `logvar`.
- Beta shapes are `a,b=exp(ψ)`. They remain positive.
- The coin data are one ordered sequence with 17 heads and 13 tails. No binomial coefficient appears in the main joint or evidence.
- Scalar coin samples and per-sample log densities have shape `[S]`. The log likelihood already sums the observations; the ELBO averages the Monte Carlo dimension once.
- A fixed random array is reused only for the numerical gradient check. Training uses fresh draws.
- A finite-sample KL estimate may be negative, and a finite-sample ELBO estimate may exceed evidence. Those observations do not invalidate the identities for exact expectations.
- This report verifies mathematical values and execution. Root integration performs slide, build, mobile and PDF inspection separately.

## Reproduce

From `/Users/nipun/git/dl-teaching`:

```sh
/Users/nipun/git/clip-lab/.venv/bin/python generative-series/lecture1/scripts/build_notebook.py
node generative-series/lecture1/scripts/check-science.mjs
```

The builder embeds all executable code in the notebook, then executes it with the invoking interpreter. `math.mjs` supplies a separate implementation of Normal/Beta formulae for the independent check; it is not imported by the notebook.
