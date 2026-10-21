# Lecture 1 science verification

The saved notebook executed all **28 code cells** in the existing clip-lab environment, using CPU and PyTorch float64. No dataset or package download was required. Its toy optimisation runs (two from the original package, plus the forward/reverse, Gaussian, 2-D and mixture fits, the network MAP, its MCMC reference and its VI fit added for the rebuilt lecture) were executed for this package; the traces are measured outputs.

The independent JavaScript check passed **3,628 assertions**, including **2,636 numerical comparisons**. Among floating-point identity checks the largest absolute discrepancy is about **6.00 × 10⁻¹²**. The largest discrepancy overall, **1.5 × 10⁻⁴**, compares a fitted reverse-KL value with its idealised limit, −log 0.3, which ignores the slight overlap between the two bumps (tolerance 10⁻³). The detailed comparison records are in `science-checks.json`.

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

## Story fixtures for the rebuilt lecture

`data/science.json` now also holds `story`, the numbers and recorded trajectories behind the rebuilt lecture's acts. `check-science.mjs` checks each independently:

- **The tiny Bayesian network.** For θ_A (the MAP) and θ_B (a prior draw), each of the 17 likelihood terms, both log likelihoods, both log priors and both network curves are recomputed by an independent forward pass. The four 10-million-draw prior Monte Carlo estimates of log p(𝒟) agree within 0.3 (mean −1.43); fewer than 1,000 of the 10⁷ draws carry each estimate. The MCMC reference has acceptance between 0.6 and 0.99, and its bands widen beyond the data. For VI, the closed-form KL to the prior is recomputed from the fitted means and widths, the ELBO equals the expected log likelihood minus that KL, the ELBO lies below the evidence estimate, the bands are narrower than the reference, and the first hidden unit is switched off.
- **The KL ladder.** The fruit KL, all three ratios, the weather entropy and cross-entropy in bits, both codes' average lengths, both Bernoulli KLs and the 10- and 30-toss evidence ratios are recomputed. The 20,000-sequence simulation's mean log ratio after 30 tosses agrees with 30 × KL within 0.05.
- **Fits.** Every recorded state of the N(4, 0.5²) → N(1, 4²) animation has a positive scale and the analytic KL. The 2-D diagonal fit matches √(1 − ρ²) and −½ log(1 − ρ²). Narrow-bump reverse KL settles at −log 0.7 and −log 0.3; forward KL matches the mixture's mean and spread.
- **The mixture fit.** The final quadrature KL of both runs is recomputed. The `rsample()` run reaches the best single Gaussian, KL 0.0562, which an independent L-BFGS optimisation also finds at N(0.429, 0.719²). The `sample()` run stays above KL 40. Plain SGD is used so that the stuck run visibly stays put; with Adam it wanders aimlessly instead.
- **Three ways.** Laplace sits on the big bump at N(−2.00, 1.00²); the Metropolis chain's mean is within 0.1 of the true −0.80.

The `vi.html` deck quotes the mixture optimum as N(0.43, 0.73²) with KL ≈ 0.049. Both the measured fit and the direct optimisation give N(0.43, 0.72²) with KL 0.056; the older figure was a noisy estimate.
