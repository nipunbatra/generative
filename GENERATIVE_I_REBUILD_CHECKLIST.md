# Generative I rebuild checklist

Created before implementation, 12 October 2026.

Final status: **149/149 checked**. Evidence, verification scope, editorial section scores and limitations: [QA report](output/rebuild-qa/QA_REPORT.md). The original unchecked version is preserved in `work/rebuild-before/checklist-initial.md`. Rendering claims refer to the tested layouts and visual review, not every possible environment.

## A. Narrative and structure

- [x] Strong title and subtitle
- [x] Clear chapter dividers
- [x] Central question maintained throughout
- [x] Original topics preserved
- [x] Prior introduced before VAE
- [x] Ordinary autoencoder problem motivates VAE
- [x] Reparameterization motivated before formula
- [x] GAN introduced as a different training mechanism
- [x] Normalizing flow introduced as an invertible alternative
- [x] Diffusion cliffhanger preserved
- [x] Smooth transition to Generative II

## B. Visual quality

- [x] Existing semantic palette reused
- [x] Meaningful visuals on at least 70% of main slides
- [x] Major plots enlarged
- [x] Gaussian charts readable
- [x] Latent plots readable
- [x] VAE diagram consistent
- [x] GAN diagram consistent
- [x] One new major operation per reveal
- [x] No unnecessary empty space around tiny figures
- [x] No decorative imagery replacing actual mathematical diagrams

## C. Distributions and priors

- [x] Discrete sampling example preserved
- [x] Observed versus theoretical frequencies clear
- [x] Gaussian mean control works
- [x] Gaussian variance/standard-deviation control works
- [x] Continuous density distinguished from point probability
- [x] Interval probability visualized
- [x] Single Gaussian limitation visible
- [x] Gaussian mixture components visible
- [x] Mixture sampling procedure shown
- [x] Observed data versus learned density distinguished
- [x] MLE numerical inconsistency fixed
- [x] Sampling versus density evaluation clear

## D. Latent-variable motivation

- [x] Autoregressive generation recalled
- [x] Pixel-generation sequential cost visualized
- [x] Latent prior explained
- [x] One sampled z followed through generator
- [x] Ordinary AE computation shown
- [x] AE reconstructions preserved
- [x] AE code distribution preserved
- [x] AE prior-sampling experiment added
- [x] AE/VAE comparison uses consistent axes/conditions
- [x] No claim that arbitrary AE samples must fail

## E. VAE distributions and architecture

- [x] Prior \(p(z)\) visually distinct
- [x] Approximate posterior \(q_\phi(z\mid x)\) visually distinct
- [x] Decoder likelihood \(p_\theta(x\mid z)\) visually distinct
- [x] Latent-space versus data-space difference clear
- [x] Encoder predicts \(\mu\) and log-variance
- [x] Decoder output distribution explained
- [x] VAE mother diagram created and reused
- [x] Learned versus computed quantities distinguished

## F. Reparameterization

- [x] Sampling-gradient problem explained carefully
- [x] Independent ε introduced
- [x] Scale-and-shift identity shown
- [x] Same noise draws reused for before/after comparison
- [x] Complete computation graph enlarged
- [x] \(\mu=2,\sigma=0.5,\epsilon=-0.8\) preserved
- [x] Multiplication shown separately
- [x] Addition shown separately
- [x] \(z=1.6\) verified
- [x] Reconstruction gradient through μ calculated
- [x] Reconstruction gradient through σ calculated
- [x] Red gradient path displayed
- [x] Log-variance parameterization shown
- [x] Log-variance derivative verified
- [x] Complete VAE loss-gradient distinction stated
- [x] PDF page 40 title collision fixed

## G. VAE objective

- [x] Reconstruction-only limitation motivated
- [x] KL/prior-compatibility motivation shown
- [x] Complete negative ELBO shown only after both terms
- [x] Gaussian versus Bernoulli likelihoods distinguished
- [x] Bernoulli reconstruction worked example added
- [x] Source of each reconstruction-loss number visible
- [x] KL prior/posterior curves visible
- [x] KL example calculated progressively
- [x] KL ≈0.818 verified
- [x] No claim that posterior must exactly equal prior
- [x] Training versus sampling distinction clear

## H. Measured VAE experiments

- [x] VAE-2 results preserved
- [x] VAE-16 results preserved
- [x] Same inputs used for reconstruction comparison
- [x] Prior draws clearly distinguished from reconstructions
- [x] Decoder probabilities distinguished from Bernoulli samples
- [x] Latent-space explorer enlarged
- [x] Chosen latent coordinate visible
- [x] Decoded image enlarged
- [x] Interpolation preserved
- [x] Latent path shown
- [x] Reconstruction curves preserved
- [x] KL curves preserved
- [x] Training-curve title corrected
- [x] All measured outputs traceable to saved checkpoints/results

## I. GANs

- [x] Generator architecture enlarged
- [x] Discriminator architecture enlarged
- [x] Real/fake inputs visually distinguished
- [x] Discriminator training step shown
- [x] Generator training step shown
- [x] Frozen/trainable parameters explicit
- [x] Generator gradients pass through discriminator computation
- [x] Non-saturating objective correct
- [x] One GAN loss calculation shown
- [x] Loss arithmetic tested
- [x] Alternating update animation works
- [x] Existing illustrative mode-collapse figure retained
- [x] Mode-collapse illustration labelled illustrative
- [x] Measured 2D GAN experiment added if feasible
- [x] Sample quality versus coverage distinguished

## J. Normalizing flows

- [x] Base distribution shown
- [x] Invertible deformation shown
- [x] Forward transformation shown
- [x] Inverse transformation shown
- [x] Local area/volume change explained
- [x] 1D stretch example shown
- [x] Probability mass conservation explained
- [x] Jacobian formula correct
- [x] Normalizing-flow versus VAE distinction clear
- [x] No confusion with flow matching

## K. Diffusion bridge

- [x] Original dog photograph retained
- [x] Gaussian corruption stages enlarged
- [x] Clean-to-noise sequence visible
- [x] Same image used in Generative II opening
- [x] Reverse-process question preserved
- [x] No premature DDPM derivation

## L. Technical/scientific accuracy

- [x] Continuous density terminology correct
- [x] All plotted values match underlying math
- [x] MLE numeric correction verified
- [x] Model likelihood versus sampling distinction maintained
- [x] VAE reconstruction probability model correct
- [x] ELBO signs correct
- [x] Reparameterization gradients correct
- [x] KL example correct
- [x] GAN objectives correct
- [x] Discriminator score not presented as calibrated realism
- [x] Flow Jacobian formula correct
- [x] Illustrative and measured outputs visibly distinguished
- [x] Model failures not hidden

## M. Rendering and final verification

- [x] Every main slide rendered
- [x] Every progressive build inspected
- [x] No clipping or overlap
- [x] No broken mathematical glyphs
- [x] No unreadably small labels
- [x] No arrows crossing content
- [x] Navigation works
- [x] Interactive controls work
- [x] PDF static states are meaningful
- [x] Numerical unit tests pass
- [x] Provenance manifest completed
- [x] Reproducibility README completed
- [x] Change log completed

---

# 29. Section-level audit scoring

For every major section, rate:

1. **Motivation:** Is the need for this mechanism clear?
2. **Visual clarity:** Does the diagram explain more than the text?
3. **Mathematical progression:** Can students reproduce calculations?
4. **Continuity:** Does the next mechanism follow naturally?
5. **Teachability:** Can a student understand each reveal without being overwhelmed?

Score each 0–2.

If the total is below 8/10, improve that section before declaring it finished.

Do not treat the score as a scientifically validated teaching-quality metric. It is an editorial self-audit.

---

# 30. Definition of done

Do not report the lecture complete until:

- All original concepts remain.
- The MLE inconsistency is corrected.
- The title collision is corrected.
- Major diagrams have been enlarged or rebuilt.
- The VAE architecture is visually persistent.
- The reparameterization derivation is substantially stronger.
- The KL and reconstruction calculations are traceable.
- The measured VAE outputs are clearer and honestly labelled.
- The GAN training diagram is substantially improved.
- The normalizing-flow geometry is more intuitive.
- The final cliffhanger matches Generative II.
- All numerical assertions pass.
- Every main slide and progressive build has been visually inspected.
- Checklist status and outstanding limitations are reported accurately.

Report:

- Original and revised conceptual-slide counts
- Original and revised physical-build counts
- Figures created/revised
- Numerical examples improved
- Experiments rerun or reused
- Interactive components tested
- Checklist completion count
- Known limitations

**Final teaching outcome:**

Students should be able to explain:

"We observe examples from an unknown data distribution.

We choose a simple latent prior.

A generator maps those latent samples toward plausible data.

A VAE learns that map using an encoder, reparameterization and a likelihood-plus-KL objective.

A GAN learns a generator using adversarial feedback.

A normalizing flow uses invertibility to make density evaluation tractable.

And perhaps we could generate data in a completely different way — by reversing a gradual corruption process."

**Final question:**

*Destroying structure by adding noise is easy. Can we learn to run that process backwards?*

That must lead directly into Generative Modelling II.
