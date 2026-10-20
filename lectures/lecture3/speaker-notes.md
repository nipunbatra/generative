# Variational autoencoders: speaker notes

Planned main route: 70 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Reopen the sampling question | 6 |
| Specify the generative story | 10 |
| Infer this image’s code | 10 |
| Reuse the ELBO | 12 |
| Make the training path differentiable | 11 |
| One step, then model evidence | 13 |
| What did we gain? | 8 |

## same-failure: This is exactly where the autoencoder left us

**Build 1, ask:** What did the amber crosses test?

**Expected:** Where standard-normal draws land relative to learned AE codes.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Remember what happened when we decoded them.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## same-samples: The same decoder, an unsupported choice of codes

**Build 1, ask:** Could fitting a density to AE codes be a valid alternative?

**Expected:** Yes. Here we explore learning a probabilistic generative model directly.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** State that model before designing an encoder.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## what-we-want: Start with a sampling procedure we can name

**Build 1, ask:** Which part produces a new latent value?

**Expected:** The prior p(z), without an input image.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Bring back the same observed 3.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## same-three: And explain this observed 3

**Build 1, ask:** What are we uncertain about once x is observed?

**Expected:** Which latent values could have produced this x.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Separate model weights from this image’s unknown.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## owners: Weights and latent values have different jobs

**Build 1, ask:** Does every image have its own decoder θ?

**Expected:** No. θ is shared across images.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Build the generative direction.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## route: The model tells us what inference must solve

**Build 1, ask:** Why not begin with “add noise to an AE”?

**Expected:** That would hide the probability model and the reason for the objective.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Choose the simplest useful latent prior.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## prior: Choose a fixed prior for latent vectors

**Build 1, ask:** What does I mean here?

**Expected:** Unit variance and zero covariance across latent coordinates.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Map a drawn z to likelihood parameters.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## decoder-logits: The decoder outputs 784 logits

**Build 1, ask:** Are the logits already a sampled binary image?

**Expected:** No. They parameterise probabilities.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Turn those parameters into a likelihood.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## likelihood: Conditioned on z, specify a Bernoulli likelihood

**Build 1, ask:** Can pixels be dependent after we average over z?

**Expected:** Yes. The shared latent can coordinate their probabilities.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Visualise the two kinds of arrows separately.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## prob-model: A probabilistic graph says who depends on whom

**Build 1, ask:** What does the solid arrow represent in this view?

**Expected:** A conditional dependency pθ(x|z), not a layer-by-layer network.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Return to computation and inspect one sampled observation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## probability-versus-draw: A probability image and a sampled observation differ

**Build 1, ask:** Why can the same z produce different binary images?

**Expected:** Bernoulli pixel sampling adds observation randomness.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Write the complete sampling code.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## sampling-code: Two random draws belong to different levels

**Build 1, ask:** Which line changes the handwriting-level latent choice?

**Expected:** The z draw; the final Bernoulli draw samples pixels given that choice.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Ask how this model assigns probability to a real image.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## image-probability: Every possible z can contribute to pθ(x)

**Build 1, ask:** Why is evaluating one decoder output insufficient?

**Expected:** The marginal likelihood sums or integrates all latent explanations.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** The inverse question will involve this same integral.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## sampling-without-encoder: Generation does not require the encoder

**Build 1, ask:** Then why will we train an encoder at all?

**Expected:** To make inference tractable when learning from observed images.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Invert the question for our 3.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## posterior-three: Which latent values explain this image?

**Build 1, ask:** What is hard even if likelihood and prior are easy to evaluate?

**Expected:** Normalising over all z.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Reuse the variational strategy.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## free-q: Fit a tractable distribution for this one image

**Build 1, ask:** Which variables would a per-image optimiser update?

**Expected:** Its μ and log variance, not the fixed decoder weights.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Run that actual small inference problem.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## per-image-fit: Per-image VI is a real optimisation loop

**Build 1, ask:** Which curve would be constant if we changed only the check random seed?

**Expected:** The true objective; its Monte Carlo estimate can vary.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Notebook demonstration:** Notebook §3: fit_one_q on held3 with decoder weights frozen. Uses 64 fresh draws per update and 256 fixed check draws.

**Transition:** Inspect where the mean travelled.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## per-image-path: One fit moves its own variational parameters

**Build 1, ask:** What must we repeat for the next image if we keep this approach?

**Expected:** A new optimisation of that image’s q parameters.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Make that repeated work concrete.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## many-images: Do we run a new optimiser for every image?

**Build 1, ask:** What repeated structure could a network learn?

**Expected:** A mapping from an image to useful variational parameters.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Replace repeated fitting with a shared prediction.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## amortisation: Learn to predict the variational parameters

**Build 1, ask:** What is shared across all images now?

**Expected:** The network weights φ; its output parameters depend on x.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Name the computational tradeoff.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## amortisation-tradeoff: Amortisation trades repeated solving for a shared map

**Build 1, ask:** Does a fast encoder guarantee exact inference?

**Expected:** No. It is a learned approximation.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Show its direction alongside the model.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## two-directions: The model and inference answer opposite questions

**Build 1, ask:** Is the dashed arrow another generative edge?

**Expected:** No. It labels the approximate-inference relationship.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Map Lecture 1’s identity onto these roles.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## recall-identity: Reuse the identity, with x fixed

**Build 1, ask:** What changes when we later update θ too?

**Expected:** The model and its evidence change; the total is no longer fixed across updates.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Write the computable ELBO before simplifying it.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## joint-elbo: The joint-density form remains the starting point

**Build 1, ask:** Which equality comes from the generative story?

**Expected:** The factorisation into likelihood and prior.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Substitute that factorisation into the same expectation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## expand-elbo: Split likelihood from the prior-to-q comparison

**Build 1, ask:** Why does the second expectation become minus KL?

**Expected:** Its log ratio is the negative of log q minus log p.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Distinguish the two KL quantities explicitly.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## two-kls: These two KL terms compare different distributions

**Build 1, ask:** If the prior KL is zero, is inference necessarily exact?

**Expected:** No. q then equals the prior, which need not be this image’s posterior.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Interpret the likelihood term on one image.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## reconstruction-term: A sampled z should explain the observed pixels

**Build 1, ask:** Where does the target image enter this loss?

**Expected:** It selects the observed Bernoulli outcomes for all pixels.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Compute the other contribution.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## gaussian-prior-kl: The diagonal-Gaussian prior KL is available exactly

**Build 1, ask:** Which q makes this KL exactly zero?

**Expected:** μ=0 and σ²=1 in every coordinate.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Interpret what mean and scale each contribute.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## kl-geometry: The formula penalises shifted or mismatched uncertainty

**Build 1, ask:** Would shrinking σ towards zero minimise the prior KL?

**Expected:** No. The −log σ² term grows without bound.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Use the actual encoder outputs for our 3.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## held-loss: For this 3, the two contributions are measured

**Build 1, ask:** Which number would change if we changed only ε?

**Expected:** The reconstruction term and total estimate; the analytic KL stays fixed.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Set the optimisation sign clearly.

**Source / evidence:** CPU replay of vae2, official test90, stated chosen ε; fixture full precision, displayed rounding.

## objective-sign: Maximise the bound, or minimise its negative

**Build 1, ask:** Does minimising this loss require knowing pθ(x)?

**Expected:** No. It uses the likelihood, q and the prior.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Extend from one image to a batch.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## batch-objective: The reduction is part of the model implementation

**Build 1, ask:** Would averaging pixels but summing KL preserve the same weighting?

**Expected:** No. It would change their relative scale by 784.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Now differentiate through the sampled latent.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## two-heads: The encoder predicts two vectors

**Build 1, ask:** Which objects are network outputs here?

**Expected:** Both μ and log variance.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Convert log variance to standard deviation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## std-conversion: Half the log variance before exponentiating

**Build 1, ask:** What would exp(v) give instead?

**Expected:** The variance, not the standard deviation.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Substitute the measured values.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## encoder-numbers: Our 3 produces a narrow Gaussian

**Build 1, ask:** Is this q centred at the origin with unit spread?

**Expected:** No. It is the approximation for this particular image.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Expose the separate source of randomness.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## noise-source: Draw noise that does not depend on φ

**Build 1, ask:** Where do the encoder parameters enter this expression?

**Expected:** Through μ and σ, while ε is an independent base draw.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Carry out the vector calculation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## latent-calculation: Calculate this particular latent draw

**Build 1, ask:** Which coordinate moves further in absolute distance from μ?

**Expected:** Compare σjεj; the second displacement has larger magnitude here.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Draw the differentiable computation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## pathwise-graph: Randomness enters through one explicit input

**Build 1, ask:** Which arrow would be missing with a detached sample?

**Expected:** The dependence of sampled z on the encoder outputs.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Recall the scalar derivatives.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## scalar-path: Follow one coordinate by hand

**Build 1, ask:** Why is there a factor one half in the log-variance derivative?

**Expected:** Differentiating exp(v/2) supplies it.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Test the gradient rather than trusting its existence.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## gradient-check: A gradient check keeps the same noise

**Build 1, ask:** What would fresh noise at each side contaminate?

**Expected:** The finite difference would mix parameter change with sampling noise.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Notebook demonstration:** Notebook §4: compare the scalar autograd derivatives with analytic derivatives and a centred finite difference.

**Transition:** Distinguish this method from a universal claim about randomness.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## gradient-scope: Reparameterisation supplies the pathwise gradient

**Build 1, ask:** Is a nonempty gradient tensor proof of the right objective gradient?

**Expected:** No. Inspect which paths the computation graph includes.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Assemble the loss from the pieces already derived.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## forward-code: The forward pass has one source of latent noise

**Build 1, ask:** Which tensors have shape B×d?

**Expected:** μ, logvar, std, ε and z. Logits have shape B×784.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Compute the per-image reconstruction contribution.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## recon-code: Keep the event sum visible

**Build 1, ask:** What is the shape after summing pixels?

**Expected:** B values, one reconstruction NLL per image.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Compute one prior KL per image.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## kl-code: The analytic KL has the same outer shape

**Build 1, ask:** Why is the second sum over latent dimensions?

**Expected:** The diagonal-Gaussian KL adds coordinate contributions.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Average the combined per-image loss.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## update-code: One complete optimisation step

**Build 1, ask:** Which parameters should change?

**Expected:** Both decoder θ and encoder φ, including both encoder heads.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Inspect those changes directly.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## actual-step: Check a real update, not only a loss formula

**Build 1, ask:** Why compare with the same ε after the step?

**Expected:** It isolates the parameter update from sampling variation.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Notebook demonstration:** Notebook §5: inspected_step copies the existing checkpoint, performs one float64 update on train images and reports each parameter-group delta.

**Transition:** Load the recorded trained model for a broader inspection.

**Source / evidence:** Actual copied-checkpoint SGD execution; fixed-noise diagnostic, not a claim every stochastic training step must decrease loss.

## trained-recon: Inspect the existing trained model on held-out images

**Build 1, ask:** Why is decoding μ different from averaging over q draws?

**Expected:** A nonlinear decoder generally maps the mean differently from the mean of its outputs.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Now use the generation path that needs no image.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## prior-probs: Draw z from the prior and decode

**Build 1, ask:** Which trained weights are needed for this figure?

**Expected:** Only the decoder θ; the prior is fixed.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Turn the same probability images into observations.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## prior-binary: Sample observations from those same likelihoods

**Build 1, ask:** What randomness changed relative to the previous figure?

**Expected:** Only the Bernoulli observation draws; z stayed fixed.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Inspect the latent geometry on the old axes.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## vae-geometry: Reuse the AE axes for the VAE comparison

**Build 1, ask:** Does a normal-looking mean cloud prove q(z)=p(z)?

**Expected:** No. Each q also has a covariance; this plot alone cannot establish equality.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Check whether a wider code improves the reconstruction picture.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## dimension-16: A larger latent changes capacity, not the objective’s meaning

**Build 1, ask:** Can you compare their raw loss to a different likelihood or reduction?

**Expected:** No. Comparisons need the same observations and reduction convention.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Summarise what this construction actually bought us.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## what-gained: We now have both a generative model and fast inference

**Build 1, ask:** What is the cleanest difference from an ordinary AE?

**Expected:** A specified probabilistic generative model and a variational training objective.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Keep the prior KL’s meaning bounded.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## prior-no-guarantee: A chosen prior is not a guarantee of perfect samples

**Build 1, ask:** Does every z from the prior have to produce a convincing digit?

**Expected:** No such guarantee follows from the objective.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Consider the limits of the inference approximation.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## posterior-gap: A stable ELBO does not certify exact inference

**Build 1, ask:** Which reference would we need to directly compute this gap?

**Expected:** The exact posterior KL or exact log evidence, generally unavailable here.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Do not confuse zero KL weighting with a deterministic encoder.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## beta-zero: Setting the KL weight to zero does not remove sampling

**Build 1, ask:** Is βKL=0 automatically the ordinary deterministic AE?

**Expected:** No. The stochastic encoder and likelihood computation can remain.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Retrieve the training and sampling paths.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## exit-paths: Explain what each path needs

**Build 1, ask:** Which path uses the image as a target in a loss?

**Expected:** Training; prior generation does not have a reconstruction target.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Bridge to a different learning signal.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## next-gan: Could a learned critic supply the training signal?

**Build 1, ask:** Which checks should carry into the next model family?

**Expected:** Explicit sampling path, parameter owners, gradient path and honest sample-quality claims.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Lecture 4: GANs.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## optional-beta: Optional: a controlled change in KL weighting (optional)

**Build 1, ask:** What tradeoff changed in these particular runs?

**Expected:** Compare the measured reconstruction and KL values; do not claim a universal monotone sample-quality trend.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Read training metadata before generalising.

**Source / evidence:** Existing d2 β1/β8 checkpoints: same architecture, seed and 40-epoch schedule; held-out one-draw replay seed673. Different objectives.

## optional-aggregate: Optional: mean clouds omit posterior spread (optional)

**Build 1, ask:** What extra information is missing from the centre plot?

**Expected:** The covariance and probability mass around each centre.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Distinguish aggregate geometry from per-image KL.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.

## optional-constant: Optional: when is log evidence a fixed total? (optional)

**Build 1, ask:** Why can the same identity support two different visual explanations?

**Expected:** The quantities held fixed differ across the two optimisation problems.

**Point here:** Track the same image, latent coordinate and parameter owner across the reveal.

**Transition:** Return to the owner of each parameter.

**Source / evidence:** Original teaching construction; exact formula or measured checkpoint replay. See data/provenance.json and fixtures.json.
