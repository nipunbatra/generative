# Teaching the four lectures

The audience knows probability and Bayes' rule and is new to variational inference. Each episode follows **predict → inspect → compute → encounter a limitation → introduce the next idea → test it**. The extra slides supply intermediate reasoning steps rather than additional survey topics.

## Recommended routes

| Lecture | Minutes | Protect these episodes |
|---|---:|---|
| 1. KL, VI, ELBO and optimisation | 140 | Compare Gaussian approximations; take a literal update; estimate an expectation; inspect the gradient path; derive the ELBO from Bayes; fit and check the coin posterior. |
| 2. From fitting a distribution to the VAE | 145 | Fit then sample the geyser waits; one model for all digits and why pixels need a code; encode, fit, draw, decode; random codes against the actual code cloud; both roads to the VAE and the ELBO derived for one image; one worked loss step. |
| 3. GANs | 60 | Compute the two players' updates separately; preserve the correct gradient paths; alternate; inspect coverage and collapse; transfer to image sampling. |
| 4. Diffusion | 70 | Make corruption exact; build one supervised example; inspect the loss and update; calculate one reverse step; assemble a DDPM; connect briefly to text conditioning. |

Section budgets are recorded in each `manifest.json` and displayed in the lecture outline. Notes provide the question, expected answer, visual focus, misconception and transition. Pause on the initial state for a prediction. Use the answer reveal to discuss the reason, not merely to read the answer aloud.

## Live demonstrations

Use the saved outputs as the fallback. Open notebooks before class; the local demonstrations do not require data or checkpoint downloads.

| Lecture | Short live path | What to ask before executing |
|---|---|---|
| 1 | Gaussian update; `sample` versus `rsample`; coin objective and prepared fit | Which parameter can move? Which distribution supplies the draws? Does a successful backward call prove the estimator is correct? |
| 2 | Geyser mixture sampling; PCA slider and mixture sampler; one AE update; random-latent decoding and code overlay; encoder mean/log-variance, one loss step, prior sampling | Does a lower reconstruction loss prescribe where new codes should come from? Where is the random draw? Does sampling a new image use the encoder? |
| 3 | Scalar D then G updates; parameter ownership checks | Which parameter differences must be zero? When should fake samples be detached? |
| 4 | Construct a noisy batch; one denoiser update; prepared reverse trajectory | What can the network see? What is the target? Where is new randomness added in a reverse step? |

Do not spend class time waiting for full image-model training. The notebooks distinguish measured saved training, reproduced checkpoint samples, fresh short updates and optional longer training.

## Forty-five-minute alternatives

These are reduced learning contracts, not a recommendation to rush every slide.

- **Lecture 1:** give the Gaussian KL formula; shorten the fitting demo; use prepared Monte Carlo comparisons. Retain scalar reparameterisation intuition, the complete ELBO derivation and the coin check. Assign the full gradient-estimator investigation as a companion activity.
- **Lecture 2:** teach Part 2 with the one-Gaussian and mixture samplers only, and Part 3 through the four-step generation diagram. Protect the reconstruction versus prior-sampling contrast, the code overlay, both roads to the VAE and one worked loss step; move the β sweep to reading.
- **Lecture 3:** retain both explicit updates and one failure case. Shorten the objective derivation and image architecture segment.
- **Lecture 4:** retain the forward example, noise-prediction task and one reverse step. Use the prepared architecture/checkpoint; move the variational bridge to reading and limit the text connection to two minutes.

Optional extensions are excluded from normal presentation mode. In reading mode and both PDFs they are clearly labelled and placed after the core route.

## Continuity and evidence

Lectures 2 and 3 keep the same official MNIST test image, digit 3 at test index 90, as an image anchor. Diffusion explicitly introduces the held digit 7 at test index 0 for its preserved, validated corruption experiment. Lecture 2 uses grey levels in [0, 1] with a Gaussian pixel likelihood (σ = 0.2), so its reconstruction loss is a scaled squared error. GAN/DDPM use greyscale images scaled to `[-1, 1]`; the change is labelled. Loss numbers across these conventions are not directly comparable.

Lecture 1 uses `Normal(loc=1, scale=4)` as the known target and a Beta(5,3) coin prior with an **ordered** sequence containing 17 heads and 13 tails. The exact posterior is a checking reference; the fitting objective uses the joint. No binomial coefficient belongs to that ordered-sequence convention.

The PML recording sequence is documented in the audit: Lecture 23 (`x5JPd-bgfhc`), Lecture 24 (`fQ8D0-tTCBU`) and Lecture 25 (`bo8wceOS6C0`). The 2023 schedule repeats the Lecture 24 link in its Lecture 25 row.

This package's figures are original constructions, computations or attributed local model evidence. Lilian Weng's articles inform the explanatory clarity of the diagrams; they are not copied as decorative slide images.
