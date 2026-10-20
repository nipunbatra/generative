# KL divergence and variational inference: rebuild plan, version 2

Version 2, after your comments: closer to the PML course in order and content, a simple companion notebook with no ASTRA,
good diagrams, and your mixture-of-Gaussians examples throughout. The deck itself is unchanged until you approve.

## The backbone: your PML Lecture 23 slides, frame by frame

Your `slides/Variational-Inference.tex` already gives the order. This plan keeps it, and fills each frame:

| PML frame | This lecture |
|---|---|
| Bayesian ML: Recap | **Part 1** Bayes, the evidence, and why it is hard |
| Bayesian ML: Methods (Laplace, MCMC, VI) | Part 1, one slide, all three on the same mixture of Gaussians |
| KL Divergence | **Part 2** KL, built slowly (the two videos), then your G1 example |
| Exercise: KL between two Gaussians (4 frames) | Part 2, an in-class exercise, checked against torch's source |
| Notebook demo · Optimizing · Notebook demo | **Part 3** G2: q chases p by gradient descent, in 1-D and 2-D |
| Monte Carlo Sampling · Notebook demo | **Part 4** G3 and G5: KL to a mixture, by sampling |
| Reparameterization Trick (2 frames) · Notebook demo | **Part 5** G4: gradients through samples |
| Evidence Lower Bound (3 frames, the mbernste picture) | **Part 6** the ELBO |
| Worked out example: Coin Toss | **Part 7** Beta q on the coin, `neg_elbo`, Pyro |
| Worked out example: Linear Regression | **Part 8** G6: diagonal vs full covariance |
| Worked out example: Neural Networks | **Part 9** mean-field at scale, and the VAE in one slide |

The lecture's map is your notebook's own list of goals (G1–G6, cell 0 of `variational-inference.ipynb`).

## Sources

| Short name | Source |
|---|---|
| **L23**, **L24**, **L25** | your PML lectures: [L23 VI part 1](https://www.youtube.com/watch?v=x5JPd-bgfhc), [L24 VI part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU), [L25 VAEs](https://www.youtube.com/watch?v=bo8wceOS6C0) (timestamps mm:ss) |
| **NB n** | cell n of `pml-teaching/notebooks/variational-inference.ipynb` |
| **VI-slides** | `pml-teaching/slides/Variational-Inference.tex` |
| **L2** | `pml2023/lectures/Bayes-ML.pdf` (page numbers): Steve, Bayes for ML, the hostel, "Main challenge" (p. 77) |
| **L8** | `slides/laplace-approximation.tex`, `notebooks/laplace-approx.ipynb`: Beta and Normal-prior coins, Laplace on a mixture |
| **S1** | `slides/Sampling-1.tex`, `notebooks/mc-linreg-evidence.ipynb`, `notebooks/mc_sampling_intro.ipynb` |
| **IT** | `slides/Information-Theory.tex` |
| **MCMC** | `figures/sampling/mcmc/hamiltorch-samples-mog.gif` (PML L13–14) |
| **RM**, **JBH** | ritvikmath, [The KL Divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY); Jia-Bin Huang, [Fantastic KL Divergence](https://www.youtube.com/watch?v=tXE23653JrU) |

## The examples (all from your course)

| Example | Where it is used | From |
|---|---|---|
| Steve, librarian or farmer | Bayes' rule | L2 |
| Coin: Beta(5, 3) prior, 17 heads in 30 → Beta(22, 16), log P(D) = −21.40 | Bayes, ELBO, the worked example | NB 61–77 |
| Linear regression evidence: closed form 12.35; from 15,000 prior draws 11.20 (PML run) or 12.22 (companion run); the best 10 draws carry 95% of the estimate | why the denominator is hard | S1 (mc-linreg-evidence) |
| Coin with a Normal prior on the logit | "no formula" | L2 p. 77; L8 |
| **Mixture of Gaussians A**: 0.7 N(−2, 1) + 0.3 N(2, 1), and the same with bumps of width 0.5 | Laplace vs MCMC vs VI; forward vs reverse KL | L8 (Laplace on a mixture); MCMC gif |
| p = N(1, 4); q1 = N(0, 2), q2 = N(1, 3): KL 0.35 vs 0.07 | G1 | NB 5–13 |
| q from N(4, 0.5) to N(1, 4) | G2 | NB 14–21 |
| **Mixture of Gaussians B**: 0.5 N(−0.2, 0.4) + 0.5 N(1, 0.5); q1 = N(3, 1), q2 = N(3, 4.5): KL 9.46 vs 45.27 | G3, G4 | NB 23–35, 50–58 |
| y = 3x + 4 + noise, diagonal vs full covariance | G6 | NB 78–89 |

## Slide-by-slide

Legend: ▶ N§ = run that section of the companion notebook · ◆ W = live widget in the deck · ✎ = students work in their notebooks.

### Part 1 · The recap: Bayes, the evidence, and why it is hard (10 slides, 11 min)

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 1.1 | Title | KL divergence and variational inference | | |
| 1.2 | Bayes' rule | Steve: prior 1 librarian : 20 farmers; the description fits 40% vs 10%; posterior 4/(4 + 20) = 1/6. The four words: prior, likelihood, evidence, posterior | L2 pp. 6–18 | |
| 1.3 | Bayes' rule for ML | p(θ ∣ D) = p(D ∣ θ) p(θ) / p(D), p(D) = ∫ p(D ∣ θ) p(θ) dθ | L2 pp. 19–20; VI-slides "Recap"; L23 0:00 | |
| 1.4 | The coin | θ = P(heads), Beta(5, 3), 17 heads in 30 → Beta(22, 16); prior, likelihood, posterior on one plot | NB 61–65 | ▶ N1 · ◆ W1 |
| 1.5 | The evidence for the coin | B(22, 16)/B(5, 3): log p(D) = −21.40. A formula only because the prior is conjugate | L8 Beta-Bernoulli frames | ▶ N1 |
| 1.6 | The evidence for linear regression | y = 1.5 + 0.5x + noise, θ ∼ N(0, I): closed form N(y ∣ X m₀, X S₀ Xᵀ + σ²I) gives log p(D) = 12.35 | S1 "marginal likelihood" frames; mc-linreg-evidence 19–22 | ▶ N1 |
| 1.7 | Estimate it by sampling the prior? | 15,000 prior draws with log-sum-exp: 11.20 in the PML run, 12.22 in ours, against 12.35. The best 10 of the 15,000 draws carry 95% of the estimate; almost every draw explains the data terribly (log-likelihoods down to −36,000). Sampling converges slowly, and in more dimensions it is hopeless | S1 "Log-sum-exp", "Sampling converges slowly"; mc-linreg-evidence 14–18 | ▶ N1 |
| 1.8 | When there is no formula at all | Bernoulli likelihood + Normal prior (on the logit): ∫ has no closed form; a neural network has millions of θ. "Intractable" | L2 p. 77; L8 "Normal prior for coin toss"; L23 3:06 | |
| 1.9 | Three ways (the one Laplace/MCMC slide) | the same mixture A as a "posterior": **Laplace** fits the higher mode only, N(−2, 1); **MCMC** visits both (mean −0.88, true −0.8); **VI** (today): the closest Gaussian, here a broad N(−0.78, 1.90) because the bumps overlap | VI-slides "Methods"; L8 mixture figure; MCMC gif; L23 0:47–3:06 | ▶ N1 |
| 1.10 | The VI idea, and today's map | ψ* = argmin KL(q_ψ(θ) ‖ p(θ ∣ D)), q any family we can sample and evaluate. The map: your notebook's goals G1–G6 | VI-slides "Methods"; L23 3:06–4:38; NB 0 | |

### Part 2 · KL divergence (10 slides, 15 min) · G1

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 2.1 | Start from a problem | favourite fruit in two years (50/40/10 → 50/10/40). One number for "how much changed, relative to year 1". Must be asymmetric | RM 2:20–7:02; JBH 4:39 (two brothers) | |
| 2.2 | Ratios, then why the log | p/q = 1, ¼, 4; plain average 1.75 is pulled by the 4; need f(4) = −f(¼), f(1) = 0: the log | RM 7:02–10:57 | |
| 2.3 | Weight by what is common | Σ p log(p/q) = KL(p ‖ q) = 0.416 nats for the fruit | RM 11:44–14:52 | ▶ N2 |
| 2.4 | The same number in bits | surprise (snow in Kashmir vs Gandhinagar); entropy; weather 50/25/25 with codes 0, 10, 11 = 1.5 bits; the wrong code 1.75 bits; KL = the 0.25 extra bits | JBH 0:46–4:39; IT "Self information", "Cross Entropy" | ▶ N2 |
| 2.5 | KL for densities | KL(q ‖ p) = ∫ q log(q/p) = E_q[log q − log p]; ≥ 0 (log r under its tangent), = 0 only if q = p; not symmetric | VI-slides "KL Divergence"; L23 5:27–6:16; JBH 10:13 | |
| 2.6 | A coin, both ways | KL(Bern(0.8) ‖ Bern(0.5)) = 0.193, the other way 0.223; after 30 tosses the right coin is 325× more likely | (your "do the coin toss") | ▶ N2 |
| 2.7 | G1: which q is closer? | p = N(1, 4); q1 = N(0, 2), q2 = N(1, 3): vote, then `dist.kl_divergence`: **0.35 vs 0.07** | NB 0, 5–13 (dkl.png); L23 7:30–9:52 | ▶ N2 · ◆ W2 |
| 2.8 | ✎ Exercise: KL of two Gaussians | write it as an expectation, expand, 7 terms; E[θ] = μ_q, E[θ²] = σ_q² + μ_q² | VI-slides "Exercise" frames 1–3; L23 15:19–22:59 | ✎ |
| 2.9 | The answer, and torch's version | ½[log(σ_p²/σ_q²) − 1 + ((μ_q − μ_p)² + σ_q²)/σ_p²]; the same lines in `torch/distributions/kl.py` (`_kl_normal_normal`) | VI-slides "Exercise" frame 4; L23 22:59–28:48 | ▶ N2 |
| 2.10 | Forward vs reverse, on mixture A | with bumps of width 1 both directions give a broad Gaussian (they overlap); narrow the bumps to width 0.5 and they split: KL(p ‖ q) covers both (0.724), KL(q ‖ p) sits on whichever bump it starts near (0.357 on the big one, 1.204 on the small one: a local optimum). VI minimises the second | JBH 4:39–5:25 | ▶ N2 · ◆ W3 |

### Part 3 · Optimising q (4 slides, 7 min) · G2

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 3.1 | Search a grid first | a few (loc, scale) pairs, the KL of each | NB 14–15; L23 10:40–12:13 | |
| 3.2 | Then gradient descent | loc and scale as parameters, scale = softplus(raw) > 0, loss = `kl_divergence(q, p)`, Adam | NB 16–17; L23 12:13–14:33; L24 4:00 | ▶ N3 |
| 3.3 | Watch q become p | from N(4, 0.5) to N(1.00, 4.00) in about 200 steps | NB 18–21 (kl_qp.gif); L24 4:46–5:32 | ▶ N3 · ◆ W4 |
| 3.4 | The same in 2-D | q fits a correlated 2-D Gaussian: a full-covariance q matches it; a diagonal q comes out too narrow (Bishop Fig. 10.2) | NB 0 ("kl_qp_2.gif"); L23 1:00:19 | ▶ N3 |

### Part 4 · A target from another family: Monte Carlo (5 slides, 7 min) · G3, G5

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 4.1 | Mixture B, and two candidates | q1 = N(3, 1), q2 = N(3, 4.5): which is closer? (dkl-different.png) | NB 23–28; L23 28:48–29:36 | ▶ N4 |
| 4.2 | torch cannot do it | `kl_divergence(q1, p_s)` → NotImplementedError: no formula between these families | NB 29; L23 29:36–30:24 | ▶ N4 |
| 4.3 | Monte Carlo, the general form | E_p[f(x)] ≈ (1/N) Σ f(x_i), x_i ∼ p; darts for π | S1 "General Form", "Estimating Pi" | |
| 4.4 | `kl_via_sampling` | draw from q (the first argument), average log q − log p. Check where we know the answer: **1.0273 vs 1.0288**. Mixture B: **9.46 vs 45.27** | NB 30–35; L23 30:24–34:39; L24 6:18–8:40 | ▶ N4 |
| 4.5 | G5: how many samples? | the estimate's spread vs N (unbiased, slow: 1/√N); a lower-variance estimator, (r − 1) − log r | S1 "Unbiased Estimator?"; JBH 7:02–11:00 | ▶ N4 |

### Part 5 · Gradients through samples: reparameterisation (6 slides, 10 min) · G4

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 5.1 | The original formulation | the computational graph: loc, scale → q → **samples** → loss; deterministic (square) and stochastic (circle) nodes; no path for the gradient | NB 48 (reparam.001); L23 35:26–37:46 | ▶ N5 |
| 5.2 | In code | `s = q.sample(); s.backward()` → *element 0 of tensors does not require grad* | NB 39; L24 9:27–10:15 | ▶ N5 |
| 5.3 | The reparameterised formulation | ε ∼ N(0, 1), z = loc + scale·ε; the new graph | NB 43–45, 49 (reparam.002); L23 38:53–41:33 | ▶ N5 |
| 5.4 | ✎ Exercise | after `z.backward()`: loc.grad = ? scale.grad = ? → 1 and ε (ε = 0.3367 with seed 42) | NB 44; L24 11:02–13:26 | ✎ |
| 5.5 | `rsample` | the same three lines inside torch (`Normal.rsample`); Beta.rsample goes through Dirichlet; Poisson has none | NB 46–47; L24 14:15–18:18 | ▶ N5 |
| 5.6 | G4: fit q to mixture B | Monte Carlo KL + reparameterisation: from N(6, 0.1) to N(0.43, 0.72) | NB 50–58 (kl_qp_mg.gif); L24 20:03–21:51 | ▶ N5 · ◆ W5 |

### Part 6 · The ELBO (6 slides, 8 min)

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 6.1 | The catch | our target is p(θ ∣ D): "if we had it, we would not need q" | L23 47:15–48:55; L24 23:44 | |
| 6.2 | ✎ Substitute Bayes | E_q[log q(θ) − log p(θ ∣ D)], with p(θ ∣ D) = p(D ∣ θ) p(θ) / p(D) | VI-slides ELBO 1–2; L23 48:55–50:44 | ✎ |
| 6.3 | The evidence comes out | log of a product = sum of logs; E_q[log p(D)] = log p(D) | VI-slides ELBO 2; L23 50:44–52:18 | |
| 6.4 | A lower bound | KL ≥ 0 ⇒ log p(D) ≥ E_q[log p(D, θ) − log q(θ)] = ELBO(q) | VI-slides ELBO 3; L23 52:18–53:55 | |
| 6.5 | The picture | your evidence / ELBO / KL bracket (redrawn), with the coin's numbers: log p(D) = −21.40 fixed, the ELBO and KL trade off as q moves | VI-slides ELBO 3 (kl-elbo.png); L23 53:55–56:27 | ▶ N6 · ◆ W6 |
| 6.6 | Two ways to write it | E_q[log p(D, θ) − log q(θ)] (what the code computes) = E_q[log p(D ∣ θ)] − KL(q(θ) ‖ p(θ)) (the VAE's form) | (sets up L25) | |

### Part 7 · Worked example: the coin (6 slides, 8 min)

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 7.1 | The model in code | `log_prior`, `log_likelihood`, `log_joint`, and `true_log_posterior` (only for checking) | NB 62–65; L24 30:14–37:42 | ▶ N7 |
| 7.2 | Which q? | Beta(α, β): support [0, 1] and an `rsample`. A Normal would need a transform (ADVI) | L24 31:50–32:37, 47:15–48:04 | |
| 7.3 | `neg_elbo` | rsample from q; log q; log joint; minus the mean | NB 66–67; L24 38:29–40:57 | ▶ N7 |
| 7.4 | Score two guesses | Beta(0.1, 2): about 190–240 (218.9 and 188.3 in the two PML runs, 239.4 in ours: it is a Monte Carlo estimate); the true posterior: 21.397 = −log p(D) exactly, since its KL is 0 | NB 68–69 (both notebook versions); L24 40:57–41:46 | ▶ N7 |
| 7.5 | Train it | Beta(45, 45) → Beta(22.6, 16.4) after 3,000 steps (true posterior Beta(22, 16)) | NB 70–71; L24 41:46–42:32 | ▶ N7 · ◆ W6 |
| 7.6 | The same in Pyro | model, guide, `Trace_ELBO`, `SVI`: α ≈ 22.9, β ≈ 16.7 (PML run; Pyro is not installed here, so the companion cell is guarded) | NB 72–77; L24 42:32–45:41 | ▶ N7 (optional cell) |

### Part 8 · Worked example: linear regression (3 slides, 7 min) · G6 — if time; otherwise homework

| # | Slide | Content | Source | Live |
|---|---|---|---|---|
| 8.1 | The model | y = 3x + 4 + noise; θ ∼ N(0, I); `log_joint(θ, x, y)` | NB 78–82; L24 56:55–57:47 | ▶ N8 |
| 8.2 | A diagonal q | learn two means and two widths; the predictive band from 1,000 draws of θ; compare with the exact posterior (Bayesian linear regression has one) | NB 83–86; L24 57:47–1:01:57 | ▶ N8 |
| 8.3 | A full-covariance q | Σ = U Uᵀ keeps it valid; the correlation it finds; 20 vs 200 points: epistemic uncertainty shrinks | NB 87–89; L24 1:01:57–1:05:09 | ▶ N8 |

### Part 9 · Neural networks, and the VAE (2 slides, 4 min)

| # | Slide | Content | Source |
|---|---|---|---|
| 9.1 | Millions of θ | a full covariance is impossible: mean-field (one Normal per weight); you lose some uncertainty; MCMC remains the gold standard | VI-slides "Neural Networks"; L24 49:02–50:39, 52:50–56:04 |
| 9.2 | The VAE in one slide | θ → z (a code), D → x (an image); one posterior per image, so an encoder predicts q(z ∣ x) = N(μ(x), σ(x)²); loss = −ELBO = reconstruction − KL(q ‖ N(0, I)), the Gaussian KL of 2.9, with z = μ + σε of 5.3 | L25 44:00–51:13, 1:06:27–1:08:05 |

### Part 10 · Recap (2 slides, 2 min)

| 10.1 | One page | Bayes → the evidence is intractable → propose q → KL → optimise q → Monte Carlo when there is no formula → reparameterise → ELBO avoids p(D) → the coin: VI recovers Beta(22, 16) |
|---|---|---|
| 10.2 | Exit questions | five, answers on click |

**54 slides. Core (Parts 1–7, 9, 10): about 72 minutes; Part 8 adds 7.**

## The companion notebook: `vi-lab/vi_lecture.ipynb` + `vi-lab/vi_utils.py` (built and run)

Built by `scripts/make_vi_notebook.py`, executed end to end (63 cells, about 30 s on a laptop CPU, 6.4 MB with outputs and
four play/step animations). Every number above marked "ours" comes from that run.

No ASTRA; only torch, numpy and matplotlib (Pyro optional for one cell). The cells mirror your notebook and use the same
variable names. High-level helpers keep each cell to a few lines; the lines that carry the idea (the KL by sampling, the
reparameterised draw, `neg_elbo`) stay written out in the notebook, not hidden.

`vi_utils.py`: `plot_dists` (several distributions, consistent colours), `mog`, `kl_via_sampling`, `kl_grid`,
`TrainableNormal` / `TrainableBeta` / `TrainableGaussian2D` (softplus widths, diagonal or full), `fit` (a 10-line Adam loop
in place of `train_fn`, recording the history), `animate_fit` (a play/step/slider animation, like your `interact` cells),
`snapshots`, `graph` (the deterministic/stochastic computational graph, both versions), `elbo_bar` (the evidence/ELBO/KL
bracket), `coin`, `logit_coin`, `blr` (exact Bayesian linear regression), `evidence_grid`, `evidence_mc`, `laplace`, `metropolis`.

| § | Slides | What runs |
|---|---|---|
| N1 | 1.4–1.9 | coin posterior and evidence; linear-regression evidence, closed form vs prior sampling; Laplace, MCMC and VI on mixture A |
| N2 | 2.3–2.10 | fruit KL; bits; coin KL; G1 (0.35 vs 0.07); the Gaussian-KL formula vs torch; forward vs reverse on mixture A |
| N3 | 3.1–3.4 | G2: grid, gradient descent, animation; 2-D diagonal vs full |
| N4 | 4.1–4.5 | G3/G5: mixture B, NotImplementedError, `kl_via_sampling`, its spread vs N |
| N5 | 5.1–5.6 | G4: `sample` error, reparameterising by hand, `rsample`, the two graphs, fitting q to mixture B |
| N6 | 6.5 | log p(D) = ELBO + KL checked numerically on the coin, for several q |
| N7 | 7.1–7.6 | coin VI: `neg_elbo`, two guesses, training a Beta q, Pyro (optional) |
| N8 | 8.1–8.3 | G6: linear regression, diagonal vs full, against the exact posterior |

## Live widgets in the deck (each mirrors a notebook cell; each has a print state)

| Widget | Slide | Mirrors |
|---|---|---|
| W1 Beta prior × likelihood → posterior (α, β, heads, tails) | 1.4 | MAP.ipynb `plot_beta_all` |
| W2 KL between two Gaussians, p = N(1, 4), both directions | 2.7 | NB 11–13 |
| W3 forward vs reverse on mixture A, press play | 2.10 | N2 |
| W4 q chasing N(1, 4), press play | 3.3 | NB 21 |
| W5 Monte Carlo KL + reparameterisation on mixture B, press play | 5.6 | NB 57 |
| W6 the coin: Beta(α, β) sliders with the ELBO + KL bars (total fixed at −21.40), plus play to train | 6.5, 7.5 | NB 68–71 |

## What changed from version 1

- Order now follows your VI slides frame by frame; the map is your notebook's G1–G6.
- Two mixtures of Gaussians from your notebooks carry Laplace/MCMC/VI, forward vs reverse KL, Monte Carlo KL and the reparameterised fit.
- The denominator is shown with your Sampling-1 linear-regression evidence (closed form vs prior sampling), not an invented example.
- 2-D appears where your notebook has it (G2's 2-D GIF), as diagonal vs full covariance.
- The companion notebook is a clean rewrite with helpers, no ASTRA.
