# Lecture 5 · Diffusion models: create by learning to undo noise

21 October 2026. A from-scratch plan for the diffusion lecture of the generative-modelling block (ES 667). It expands Paras Chopra's explainer [Diffusion Models from First Principles](https://paraschopra.github.io/explainers/diffusion/index.html) into a two-session lecture on the Lecture 1 runtime, and folds in the best moments of the videos Nipun recommended. Gemini (Pro) watched every video and wrote a study of each; those studies, Gemini's own draft plan and the prompts are in [`sources/`](sources/). This plan is the merge of Gemini's draft with Claude's, checked against the feedback log for Lectures 1–4.

## Revision, same day: the simple version first

Nipun's feedback on this plan: *not too complicated; follow Paras Chopra's article in detail, with text conditioning; students must see every number (how ᾱₜ comes from αₜ and βₜ), follow the forward and the reverse steps on simple examples, and play with them; simple, clean diagrams.* The built lecture therefore differs from the parts below:

- **Main route** (70 slides): Opening · 1 Destroying is easy · 2 Undoing it: one step back · 3 Training · 4 Generating · 5 Steering: labels, guidance and text · 6 Faster sampling · 7 What the network learns, and wrap-up.
- **New "every number" material**: a toy schedule of five big steps (β = 0.1, 0.2, 0.3, 0.5, 0.8) with the table β → α → ᾱ; the ink pixel pushed forward step by step and walked back step by step; forward and reverse calculators (toy or real schedule) showing every substitution; one-step diagrams; guidance drawn as arrows from the live network.
- **Moved to optional depth**: the one-pixel Bayes slides, the posterior by completing the square, the mean-in-noise substitution, the full ELBO derivation and KL-to-MSE, Tweedie, Langevin, the 2×2 map of samplers, classifier guidance from Bayes, SDEs, other targets, score distillation.

## The one-sentence version

> **If you learn to undo each tiny step of a destruction process, you can create from scratch.** (Paras Chopra)

**Central question.** Can many small, learned denoising steps turn pure noise into a new image?

**After the lecture a student can**

1. noise an image to any step in one line, and explain why the image is shrunk as well as noised;
2. explain, with Bayes' rule on one pixel, why one big step back is ambiguous and a tiny step back is almost Gaussian;
3. write the training tuple (x₀, t, ε) → xₜ, say what the network sees and what it predicts, and why it predicts an *average*;
4. carry out one reverse step by hand (ε̂ → x̂₀ → mean → add noise) and write the sampling loop;
5. derive the noise MSE from the ELBO: diffusion is a VAE with a fixed encoder, and each KL between two Gaussians becomes a squared difference of noises;
6. read the noise predictor as a score (arrows towards data), correctly;
7. explain DDIM (fewer steps, same network) and classifier-free guidance (the dial between variety and fidelity);
8. place Stable Diffusion: an autoencoder (Lecture 3), a denoiser (today), text through cross-attention (the attention lectures).

## What makes this lecture better than every source

| Source | Its best moment, which we keep | Its weakness, which we fix |
|---|---|---|
| Paras Chopra, explainer | The core sentence; an image dissolving under a slider; the pixel histogram turning Gaussian; step-by-step vs one jump; ε / x₀ / score table; the training loop "shockingly simple"; the score field with a particle; schedules; speed vs quality; "derive it on a napkin" | Says the endpoint is Gaussian *because of the Central Limit Theorem* (it is because the signal is shrunk to zero and the added noise is Gaussian); writes ∇log p(xₜ) = −ε/√(1−ᾱₜ) (that is the score of q(xₜ \| x₀); the marginal score needs E[ε \| xₜ]); his "reverse the film" uses the known clean image; DDIM is not "DDPM without the noise term" |
| Ari Seff, *What are Diffusion Models?* (very good) | 2:56 the 1-D picture: a big jump back is bimodal, a small step back is Gaussian; 6:28 VAE diagram stacked over the diffusion chain | ELBO refactoring, the Bayes step and the ε substitution flash past |
| Deepia, *DDPM, Generative AI Animated* (very good) | 3:29 naive noising makes the variance explode, *then* √(1−β) fixes it; 7:38 a 1-D bimodal density morphing into N(0,1); 17:23 fixed variance → KL = sliding one mean onto the other | Skips the ELBO and posterior-mean derivations; library code |
| Outlier, *Math Explained* (very good) | 4:48 three things the network could predict; 6:26 linear vs cosine on one image; 18:23 "why condition on x₀": the noisy image could have come from anywhere, the clean one pins the step | 16:00–24:00 walls of LaTeX; KL→MSE hand-waved |
| MIT 6.S183 L1 (Yuan) | 2-D toy MLP; the learned denoiser as a vector field onto a spiral | EDM notation; no sampler |
| MIT 6.S183 L2 (Becker) | 18:01 "100 cats": one step averages them into a blur; 16:15 ε is the *total* noise, not the last step's; the DDPM/DDIM/SDE/ODE 2×2 map | Skips the ELBO |
| MIT 6.S183 L3 (Chen) | Bayes' rule on scores → classifier guidance → CFG; three blobs sharpening as the guidance grows | Three names for the guidance scale |
| MIT 6.S183 L4 (Scarvelis) | The ideal denoiser of a finite training set only *copies* it: a perfect loss is a lookup table; networks generalise through their biases | A rapid literature survey |
| MIT 6.S183 L6 (Yuan) | Noise drowns fine detail first: generation is coarse to fine | Rushed reweighting |
| Vizuara, *Principles of Diffusion Models* L1 | The five families of generative models as a prologue; the energy landscape flipped into a density | Prologue only; slow |

Our additions, which no source has: **one pixel where everything is exact** (paper or ink, so the posterior, the ideal denoiser, the true reverse step and the score are all closed-form and live under sliders); **a 2-D model running live in the browser** (drop particles and watch the trained network carry them onto the data); the network's **guess of the clean 3 at every noise level**; the **ELBO derived with the KL formula students derived themselves in Lecture 1**; and the **memorisation experiment computed exactly** (the ideal denoiser on our own training points).

## Running examples (three, introduced in this order)

1. **One pixel** (1-D, exact). A pixel of MNIST is either paper (−1) or ink (+1). Pixel (r, c) of our 3 is ink, and across the training set it is ink in a measured fraction π of images (chosen so π ≈ 0.3–0.5; set in the science script). Numbers carried through Parts 1, 2, 4 and 5: x₀ = +1, t = 250 (ᾱ₂₅₀ ≈ 0.524, √ᾱ ≈ 0.724, √(1−ᾱ) ≈ 0.690), one noise draw ε = −0.5, so xₜ ≈ 0.37. Two illustrative big steps (β₁ = 0.1, β₂ = 0.2) are used *only* to show the two-step algebra of the shortcut, and are labelled as such. Values printed with two or three decimals.
2. **Our 3** (MNIST test image 90, the held 3 of Lectures 2–3, now in grey levels scaled to [−1, 1]). The forward film, the training tuple, the network's guesses, reverse trajectories, guidance ("draw a 3"), latent diffusion.
3. **A 2-D smiley** (point cloud: face outline, two eyes, mouth). The training playground, vector fields, live sampling, DDPM vs DDIM paths, guidance by part of the face, memorisation. A nod to Paras's smiley, but now every dot is a data point x₀ ∈ ℝ².

Rule: every equation is followed by the pixel's numbers; every mechanism is then shown on the 3 or the smiley; every claim about a trained model is backed by our own model.

## The story (two sessions, seven parts)

Each part opens with its question and how we will answer it, and closes with a recap that names the next obstacle. A route strip on part openers shows where we are.

### Opening (5 min)

- **Our own model, noise to a 3.** A film from the trained DDPM: static → a 3. *What turned this static into a digit?*
- **What these systems do** (credited published results): text-to-image, inpainting, super-resolution, video, proteins (MIT L1's showcase, condensed to one slide). *All of them run the loop we will write today.*
- **Three generators so far.** VAE (one decoder step: blurry), GAN (one step: sharp but unstable, collapses). *What if generation were a thousand easy steps instead of one hard leap?*
- **The core insight** in one sentence, and the **ink film** (illustration generated with Gemini): a drop of ink spreads through water and nothing pulls it back together, but if you had a film of it, each pair of frames differs only a little; learn to undo each small change and run the film backwards. This is also where the name comes from (Sohl-Dickstein et al. 2015). Nipun is vegetarian: Paras's scrambled-egg analogy is deliberately not used.

### Session A: build a working DDPM (≈ 75 min)

**Part 1 · Destroying is easy: the forward process** (≈ 15 min)

| § | Slides (one move per build) | Figure / widget | Source |
|---|---|---|---|
| 1.1 One noising step | Add a little noise to one pixel: x₁ = x₀ + √β ε. Numbers. | — | Paras III |
| 1.2 Why shrink? | Predict: add noise 1,000 times; what happens to the spread? Variance grows without bound. Fix: x_t = √(1−β) x_{t−1} + √β ε, and Var stays 1: (1−β)·1 + β = 1. | **W1 exploding vs shrunk** (1-D particles, live) | Deepia 3:29 |
| 1.3 Watch our 3 dissolve | Slider t: the 3, the signal bar √ᾱₜ, the noise bar √(1−ᾱₜ). At t = 500 an ink pixel can read as paper. | **W2 destruction film** (live from x₀ and a fixed ε) | Paras II, Ari Seff 0:00 |
| 1.4 Every dataset ends in the same place | The pixel histogram of many digits turns into N(0, 1) — *not* the CLT: the signal is shrunk to zero and only Gaussian noise is left. The pixel's density river: two thin lines at ±1 widen and merge. | **W3 pixel histogram**, **W4 density river** | Paras III (corrected), Deepia 7:38 |
| 1.5 The shortcut | Do we need a 1,000-step loop to make one training example? Two big illustrative steps: substitute, group, variances add (Lecture 1: independent Gaussians). The nested formula grows off the slide, then collapses: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε. Pixel numbers. Same distribution either way (not the same draw). | **W5 shortcut check** (10,000 runs each way, histograms) | Ari Seff 8:34, Paras IV (corrected) |
| 1.6 The schedule | β from 10⁻⁴ to 0.02 over 1,000 steps; ᾱₜ and the signal-to-noise ratio as plots, not tables. | static plot | DDPM |
| 1.7 Code + recap | `q_sample` in three lines. Recap: destruction needs no learning, and hands us (x₀, t, ε, xₜ) for free. **Next:** can we go back? | code | — |

**Part 2 · One step back is a Bayes problem** (≈ 20 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 2.1 What we want | q(x_{t−1} \| xₜ) ∝ q(xₜ \| x_{t−1}) q(x_{t−1}). The likelihood is our noising rule; the prior is *the distribution of all noisy images*: unknown. | — | Lecture 1 Bayes |
| 2.2 One pixel: ink or paper? | With a pixel the prior is known: π ink, 1−π paper. Observe xₜ = 0.37 at t = 250. Two likelihood bumps, Bayes, P(ink \| xₜ). The best guess under squared error is the average: E[x₀ \| xₜ] = P(ink)·(+1) + P(paper)·(−1): a grey. | **W6 ink or paper?** (sliders t and xₜ, exact) | new |
| 2.3 A big jump back is ambiguous | Straight from x_T to x₀: two spikes, at ink and at paper. A single Gaussian cannot be two spikes; its best fit is the grey average. On images: the average of many 3s is a blur ("100 cats"). | W7 at maximum jump; mean of training 3s | MIT L2 18:01 |
| 2.4 A tiny step back is almost Gaussian | Shrink the jump: the exact reverse density (a mixture of two Gaussians) turns into one bell. Why: a narrow likelihood times a smooth prior is close to Gaussian. So: learn N(μ_θ(xₜ, t), σₜ²) for each tiny step, 1,000 times. | **W7 reverse-step explorer** (slider: jump size; exact density vs best Gaussian) | Ari Seff 2:56 (made exact) |
| 2.5 If we knew x₀ | The step back *given x₀* is exactly Gaussian. Bayes with three Gaussians, complete the square one line per build; μ̃ₜ(xₜ, x₀) and β̃ₜ; pixel numbers. The clean image pins the step. | step-by-step algebra; picture: noisy image with question marks, then x₀ revealed | Outlier 18:23 |
| 2.6 So: guess x₀ | We do not know x₀, so estimate it from xₜ: that is the network's job. Rewrite with ε (x₀ = (xₜ − √(1−ᾱ) ε)/√ᾱ): μ = (xₜ − β/√(1−ᾱ) · ε)/√α. Three equivalent targets: ε ("what is the noise?"), x₀ ("what is under it?"), score (Part 5). Recap. **Next:** how do we learn to guess? | table with pixel numbers | Paras VI, Outlier 4:48, MIT L2 20:23 |

**Part 3 · Training: labels for free** (≈ 20 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 3.1 The training tuple | Pick x₀, t, ε; make xₜ. Input (xₜ, t); target ε. We made the noise, so we know the answer. | tuple diagram on our 3 | Paras VII |
| 3.2 ε is the total noise | Not the last step's noise: everything added since x₀. | — | MIT L2 16:15 |
| 3.3 Why t is an input | One set of weights serves all 1,000 noise levels; the same xₜ means different things at t = 50 and t = 900. Sinusoidal features, as for positions in a transformer. | time-feature stripes | Outlier 7:01, MIT L1 22:21 |
| 3.4 The loss, once by hand | ‖ε − ε̂‖² on the pixel, then on our 3 (residual heatmap). | — | DDPM Alg. 1 |
| 3.5 It cannot be exact | The same xₜ can come from many (x₀, ε): the best possible prediction is the average E[ε \| xₜ], and the loss cannot reach zero. Pixel: the irreducible loss against t. | plot | MIT L2 hint |
| 3.6 The network | 2-D: an MLP on (x, y, time features). Images: a U-Net (encoder, decoder, skip connections) with the time embedding added in every block. | mechanism diagram | Paras VI, Outlier 7:01 |
| 3.7 Code, and training live | The training step in eight annotated lines. Play the recorded 2-D training: the loss falls and the arrows at t = 100 sharpen onto the smiley. | **W8 training playground** | MIT L1 23:00 |
| 3.8 What the network believes | Our 3 at t = 900 … 50: xₜ, ε̂, x̂₀. At high noise x̂₀ is a blurry average digit; at low noise it is sharp. One pixel alone gives a grey; all 784 together give a 3: neighbours coordinate (Lecture 2: strokes are coordinated choices). Recap. **Next:** use the guesses to walk back. | **W9 x̂₀ film** | Lecture 2 callback |

**Part 4 · Generating: walk back from noise** (≈ 20 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 4.1 Start from noise | x_T ~ N(0, I): Part 1 showed every image ends there. | — | — |
| 4.2 One reverse step by hand | Pixel: ε̂ → x̂₀ → μ → + σₜ z → x_{t−1}, one build each. | — | MIT L2 |
| 4.3 One step on an image | Our model at t = 600: four panels. | static panels | — |
| 4.4 The loop | Algorithm 2 beside eight lines of PyTorch; no noise at the last step. | code | Outlier 26:55 |
| 4.5 Watch it | 2-D: click to drop particles; the trained network (running in the browser) carries them onto the smiley; trails show each path. MNIST: several seeds, xₜ and x̂₀ side by side, scrubbable. Coarse to fine. | **W10 live 2-D sampler**, **W11 MNIST trajectories** | Ari Seff 5:17, Paras VIII |
| 4.6 Why the fresh noise? | Run the same model with σₜ = 0: smooth, washed-out digits. The noise lets each path commit (pixel: to ink or to paper). | side by side | — |
| 4.7 64 new digits | The payoff, next to VAE and GAN samples from Lectures 3–4. | grid | — |
| 4.8 The napkin | Training and sampling side by side on one slide. **Session A ends.** **Next time:** why this is maximum likelihood, and how to make it fast and steerable. | — | Paras XIII |

### Session B: why it works, and how far it goes (≈ 70 min)

**Part 5 · Why it works: a VAE, and a score** (≈ 25 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 5.1 Diffusion is a VAE | Lecture 3's VAE diagram above the diffusion chain: latents x₁…x_T, encoder = fixed noising (no parameters), decoder = learned denoiser. | **diagram build** (VAE slides onto the chain) | Ari Seff 6:28 |
| 5.2 The ELBO of the chain | Same identity as Lecture 1: log p(x₀) ≥ E_q[log p(x₀:T) − log q(x₁:T \| x₀)]. | step-by-step | Lecture 1 §4 |
| 5.3 Condition on x₀, and telescope | Bayes rewrites each forward step backwards *given x₀*; the ratio telescopes; L_T + Σ L_{t−1} + L₀. One rule per build. | — | Outlier 16:23 (slowed) |
| 5.4 Each term is a KL of two Gaussians | Lecture 1's formula; equal variances → (μ̃ − μ_θ)²/(2σ²): slide one mean onto the other. Pixel numbers. | sliding-Gaussians figure | Deepia 17:23 |
| 5.5 From means to noises | Substitute both means in ε: weight × (ε − ε̂)². Drop the weight: the simple loss. A re-weighted ELBO, which Ho et al. found gives better samples. Weight against t plot. | — | DDPM §3.4 |
| 5.6 The score: arrows to data | ∇ log p(x): an arrow everywhere, pointing uphill in density. Pixel mixture, then the smiley. Langevin: follow the arrows plus a little noise and you sample. | **W12 score field**, **W13 Langevin** | Paras IX, Vizuara 39:27 |
| 5.7 Why noise levels | Far from data the arrows are unknown (no examples there). Noise blurs the density, so arrows exist everywhere; lower the noise step by step = DDPM's schedule. | field at three noise levels | Song & Ermon 2019 |
| 5.8 The noise predictor is a score | Tweedie: E[x₀ \| xₜ] = (xₜ + (1−ᾱ) ∇log qₜ(xₜ))/√ᾱ, so ε* = −√(1−ᾱ) ∇log qₜ. The corrected version of Paras's formula, checked on the pixel. The 2×2 map: DDPM, DDIM, SDE, ODE. Recap. | 2×2 map | MIT L2 39:04 |

**Part 6 · Faster and steerable** (≈ 25 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 6.1 A thousand network calls | Measured sampling time of our model; can we skip? | — | — |
| 6.2 DDIM | Jump to the guess, re-noise to the next level: x_{t'} = √ᾱ_{t'} x̂₀ + √(1−ᾱ_{t'}) ε̂, from the shortcut of 1.5. No fresh noise: deterministic, any subsequence of steps. | 2-D: DDPM paths jitter, DDIM paths are smooth (live) | MIT L2 24:35 |
| 6.3 Same network, fewer steps | MNIST with 1, 2, 5, 10, 20, 50, 100, 1,000 steps, same seeds. | **W14 steps slider** | Paras XI |
| 6.4 The schedule matters | Linear destroys our 3 by about t = 600; cosine keeps it longer (SNR curves). | W2 with linear/cosine toggle | Outlier 6:26 |
| 6.5 Draw a 3 on demand | Add a label embedding to the time embedding (one line). | — | — |
| 6.6 Classifier guidance | Bayes on scores: ∇log p(x \| c) = ∇log p(x) + ∇log p(c \| x). Scale the classifier term by w. Needs a classifier of *noisy* images. | — | MIT L3 6:19 |
| 6.7 Classifier-free guidance | Rearrange Bayes: the classifier term is the difference of two denoisers. One network, trained with the label dropped 10% of the time. ε̃ = ε̂(∅) + w (ε̂(c) − ε̂(∅)). How many networks? One. | code | MIT L3 17:28 |
| 6.8 The guidance dial | MNIST 3s at w = 0, 1, 2, 3, 5, 7: more w, more 3-like, less variety (measured with our classifier). 2-D: guide towards "mouth" with a live slider. Recap. | **W15 guidance dial** | MIT L3 13:20 |

**Part 7 · From digits to text-to-image** (≈ 20 min)

| § | Slides | Figure / widget | Source |
|---|---|---|---|
| 7.1 Latent diffusion | 512×512×3 = 786,432 numbers per step is expensive. Compress with an autoencoder (Lectures 2–3), diffuse the code, decode. Our 4×7×7 MNIST latent model. | latent samples | Rombach et al. |
| 7.2 Text in, through cross-attention | Text encoder; queries from image features, keys and values from words (the attention lectures). The Stable Diffusion pipeline, each box mapped to a lecture. | pipeline diagram | Paras XII |
| 7.3 Does it just copy? | The best possible denoiser for a finite training set is a weighted average of training images, and sampling with it returns training images exactly. Our networks cannot represent that lookup table, and generalise. 2-D: exact ideal denoiser on 40 points vs the MLP; MNIST: our samples and their nearest training images. | **W16 memorisation** | MIT L4 13:04 |
| 7.4 Where the field went | Transformers as denoisers (DiT), straight paths (flow matching, rectified flow), few-step models (distillation, consistency), video, audio, proteins, robot policies, text by masked diffusion. One slide, pointers only. | — | MIT L2, L6 |
| 7.5 Three generators | VAE, GAN, diffusion: training signal, sampling cost, likelihood, failure mode. | table | — |
| 7.6 Exit questions | What is available in training but not at sampling? Why is t an input? What happens in one reverse step? Why add noise when sampling? Why does a perfect loss copy? | — | — |
| 7.7 References | Every source with the slide it supports. | — | — |

### Optional depth (appendix)

The continuous view (SDE and probability-flow ODE, built from the discrete update); generation in frequency (power spectra: noise drowns fine detail first); v-prediction and flow matching; learned variances (improved DDPM); score distillation (DreamFusion) as a one-slide pointer; the posterior algebra with every term.

## Key derivations, one rule per build, with what students reuse

| Derivation | Reuses | Where |
|---|---|---|
| Variance stays 1 with the shrink | Var(aX + bε) = a²Var X + b² | 1.2 |
| Two steps → the shortcut | Sum of independent Gaussians (Lecture 1) | 1.5 |
| P(ink \| xₜ), E[x₀ \| xₜ] | Bayes' rule (Lecture 1 §1.1) | 2.2 |
| q(x_{t−1} \| xₜ, x₀) by completing the square | Gaussian densities, Bayes | 2.5 |
| μ̃ in terms of ε | substitution | 2.6 |
| One reverse step | 2.5–2.6 | 4.2 |
| ELBO of the chain, telescoping | Lecture 1 §4.1 ELBO steps | 5.2–5.3 |
| KL of two equal-variance Gaussians → squared means → noise MSE | Lecture 1 §2.7 Gaussian KL | 5.4–5.5 |
| Tweedie, ε as a score | Gaussian density, Bayes | 5.8 |
| DDIM update | the shortcut (1.5) with x̂₀ and ε̂ | 6.2 |
| Classifier-free guidance | Bayes on log-densities, gradients | 6.6–6.7 |

## Live figures (ranked by teaching value)

| | Widget | Inputs | Data |
|---|---|---|---|
| W7 | Reverse-step explorer: exact q(x_{t−Δ} \| xₜ) vs best single Gaussian | jump size, xₜ | exact (pixel) |
| W10 | Live 2-D sampler: drop particles, watch paths, DDPM/DDIM, noise on/off | clicks, buttons | trained MLP in the browser |
| W9 | x̂₀ film of our 3: xₜ, ε̂, x̂₀ | t | recorded network outputs |
| W6 | Ink or paper? posterior and E[x₀ \| xₜ] | t, xₜ | exact |
| W2 | Destruction film with signal/noise bars (linear/cosine) | t | live from x₀, fixed ε |
| W15 | Guidance dial (MNIST grid, 2-D live) | w | recorded + live MLP |
| W12/13 | Score field and Langevin particles | noise level | live MLP + exact mixture |
| W16 | Memorisation: ideal denoiser vs network | — | exact + recorded |
| W1 | Exploding vs shrunk noising | t | live |
| W4 | Pixel density river with paths | t | exact |
| W3 | Pixel histogram → N(0, 1) | t | live |
| W5 | Shortcut check | t | live Monte Carlo |
| W8 | Training playground (loss + arrows sharpening) | training step | recorded |
| W11 | MNIST trajectories, several seeds | t, seed | recorded |
| W14 | Steps slider (DDIM) | steps | recorded |

Every live figure has a fixed state for the PDF (start and end builds), as in Lecture 1.

## Models and measurements

- **Pixel**: no training; exact formulas in Python (checks) and JavaScript (widgets).
- **2-D smiley**: MLP on (x, y, 16 time features) → 3 × 128 SiLU → 2; unconditional and part-conditional (outline / eyes / mouth / none) versions; same linear schedule, T = 1,000; trained in minutes. Recorded: loss curve, arrow fields at checkpoints, sample paths, weights exported for the browser.
- **MNIST**: the validated 30k-step pixel DDPM (814k parameters, EMA), the class-conditional twin with label dropout, the 4×7×7 autoencoder with its latent DDPM, and the MNIST classifier from the GM-II rebuild, with their code ported verbatim and hashes recorded. Recorded: x̂₀ film of the 3, trajectories for 8 seeds, 64 samples, σ = 0 comparison, DDIM sweep, guidance grid with classifier accuracy and a diversity measure, latent samples, nearest training neighbours, measured sampling time.
- **Memorisation**: exact ideal denoiser on 40 smiley points and on 1,000 MNIST images; sampling returns training points (measured distance to the nearest one).

## Code on slides

`q_sample` (3 lines), the training step (8), the sampler (8), DDIM step (3), guidance (2 lines added to the sampler), the label embedding (1). Each with the active line highlighted per build and annotations beside it.

## Companion notebook

`lecture5-diffusion.ipynb`, sections numbered as the slides, self-contained, Colab: the pixel (exact, instant), the forward process on MNIST (downloads MNIST), the 2-D smiley trained from scratch on CPU (about a minute), sampling it with DDPM and DDIM, guidance on 2-D, then the MNIST U-Net: one real training step, sampling from the saved checkpoint (loaded from the repository), x̂₀ films, guidance. Optional: full MNIST training on a GPU.

## Gemini illustrations (conceptual only, labelled as illustrations)

1. A drop of ink spreading in water, as a five-frame film (opening): the physics the name comes from, and the film we learn to run backwards.

No food analogies: Paras's scrambled egg is replaced by the ink film throughout.

Never for data, mechanisms or results: those come from our models and formulas.

## Misconceptions corrected on the slide that needs it

| Misconception | Correction | § |
|---|---|---|
| The endpoint is Gaussian by the Central Limit Theorem | The signal is shrunk to zero; what is left is Gaussian noise by construction | 1.4 |
| Step-by-step and one jump give the same noisy image | The same *distribution*; different draws | 1.5 |
| ε is the noise added in the last step | It is all the noise added since x₀ | 3.2 |
| The network outputs the clean image exactly | It outputs an average guess; the loss cannot reach zero | 3.5 |
| The network does not need t | One set of weights for all noise levels | 3.3 |
| The fresh noise in sampling is a nuisance | Without it DDPM samples are washed out | 4.6 |
| −ε/√(1−ᾱ) is the score of the noisy data | It is the score of q(xₜ \| x₀); the network's ε̂ estimates the marginal score | 5.8 |
| DDIM = DDPM with the noise removed | A different update that re-noises the clean guess; same marginals | 6.2 |
| Guidance samples p(x \| c) | It samples a sharpened distribution: w trades variety for fidelity | 6.8 |
| A lower loss is always better | The global minimum memorises | 7.3 |

## References (each cited on the slide it supports)

Paras Chopra, *Diffusion Models from First Principles* (2025) · Ari Seff, *What are Diffusion Models?* (2022) · Deepia, *Diffusion Models: DDPM, Generative AI Animated* (2025) · Outlier, *Diffusion Models, Paper Explanation, Math Explained* (2022) · Vizuara, *Principles of Diffusion Models, Lecture 1* (2025) · MIT 6.S183 *A Practical Introduction to Diffusion Models*, Lectures 1–6 (IAP 2025) · Sohl-Dickstein et al. 2015 · Ho, Jain, Abbeel 2020 (DDPM) · Song & Ermon 2019 · Song et al. 2021 (score SDEs) · Nichol & Dhariwal 2021 · Song, Meng, Ermon 2021 (DDIM) · Dhariwal & Nichol 2021 (classifier guidance) · Ho & Salimans 2022 (CFG) · Rombach et al. 2022 (latent diffusion) · Karras et al. 2022 (EDM) · Lipman et al. 2023 (flow matching) · Liu et al. 2023 (rectified flow) · Peebles & Xie 2023 (DiT) · Kadkhodaie et al. 2024 (generalisation) · Lilian Weng, *What are Diffusion Models?* · Calvin Luo, *Understanding Diffusion Models: A Unified Perspective* · Sander Dieleman's blog.

## Build order

1. Science: pixel formulas and checks; train the 2-D models; run the MNIST experiments from the ported checkpoints; write `data/science.json` and figures.
2. Slides: `content.js` on the Lecture 1 runtime (numbered sections, notebook links, notes with ask / expected / transition), `widgets.js` for the live figures.
3. Notebook: built and executed by a script, anchors matching the slides.
4. Verify every build in the browser, KaTeX errors, overflow, reading view, PDFs; then commit on `rebuild/lecture5-diffusion` (publishing is a separate step).
