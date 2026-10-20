# From generative models to autoencoders: speaker notes

Planned main route: 65 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| What counts as a new example? | 7 |
| Fit, then sample | 10 |
| What changes for images? | 8 |
| Learn a compact representation | 10 |
| One autoencoder update | 12 |
| Does this give us a sampler? | 11 |
| Name what is missing | 7 |

## new-example: What would count as a new plausible example?

**Build 1, ask:** What would you ask a digit generator to produce?

**Expected:** A fresh plausible digit, with variation.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Separate three tasks that can look similar.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## copy-reconstruct-sample: Three outputs can look similar

**Build 1, ask:** What information does each procedure need?

**Expected:** Sampling does not require an input image.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Try the distinction on one concrete 3.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## held-three: Keep this handwritten 3 in view

**Build 1, ask:** Is the right image evidence that we can sample a new digit?

**Expected:** It shows reconstruction; a sampling procedure is still missing.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** State the distributional objective.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## data-distribution: We see examples, not the entire distribution

**Build 1, ask:** What must the fitted model support?

**Expected:** A rule for producing new examples.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Use a one-dimensional example where every step is visible.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## quality-coverage: One convincing example is not enough

**Build 1, ask:** Would you call that a successful digit generator?

**Expected:** It may produce a plausible image but misses almost all variation.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Make both fitting and sampling explicit.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## today-route: Fit a distribution. Then challenge its samples.

**Build 1, ask:** Which part of this route is already familiar from Lecture 1?

**Expected:** Choosing a distribution family and fitting its parameters.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Start with twelve observed values.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## two-bumps: Twelve observations suggest two groups

**Build 1, ask:** Where should a new sample usually land?

**Expected:** Near either observed group, rather than mainly between them.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Fit the simplest Gaussian baseline.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## normal-fit: Fit the mean and the spread

**Build 1, ask:** Is the fitted centre near either group?

**Expected:** It lies between the groups; a single Gaussian pools both.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Inspect the fitted curve.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## normal-curve: The fitted centre is in the gap

**Build 1, ask:** What structure did this family discard?

**Expected:** The two distinct groups.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Turn this fitted density into actual samples.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## normal-sample: Sampling is a separate computation

**Build 1, ask:** Which values stay fixed while we generate many new x?

**Expected:** The fitted mean and scale; only epsilon changes.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Give the two groups separate components.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## mixture-story: First choose a group, then a value

**Build 1, ask:** What extra random choice creates two groups?

**Expected:** The component choice c.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Connect the story to the marginal density.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## mixture-density: The hidden choice becomes a sum

**Build 1, ask:** Why do we add the two possibilities?

**Expected:** The component is unobserved; marginalisation sums its alternatives.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Check the fitted shape and its draws.

**Source / evidence:** Computed 1D two-component EM fit, fixed 12 observations, 40 updates; scripts/build_evidence.py.

## mixture-fit: A richer family can express both groups

**Build 1, ask:** Where has the density changed most?

**Expected:** The mixture concentrates around the two groups.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Now judge generated values, not only the fitted curve.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## compare-draws: Look at the samples the models produce

**Build 1, ask:** Which row better reflects the two groups?

**Expected:** The mixture row follows the two-mode structure.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Notebook demonstration:** Notebook §2: rerun only the seeded sampling cell. Change sample count, retaining fitted parameters.

**Transition:** Images make the same task much larger.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## pixels: An image is 784 observed values

**Build 1, ask:** What information was removed by thresholding?

**Expected:** Grey intensity levels; the spatial arrangement remains.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Try a simple probabilistic model for these bits.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## binary-pixel: One pixel has a simple distribution

**Build 1, ask:** If πj is 0.8, what does the model say?

**Expected:** The pixel is one with probability 0.8, not always grey 0.8.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Estimate each pixel’s probability from data.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## average-image: Fit one probability per pixel

**Build 1, ask:** Is this mean image itself a typical observed digit?

**Expected:** No; averaging can combine incompatible strokes.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Sample the independent-pixel model.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## independent-pixels: Correct pixel frequencies can make poor images

**Build 1, ask:** What relation between pixels is missing?

**Expected:** Whether neighbouring strokes and distant parts belong to the same digit.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Express that missing dependence explicitly.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## dependence: Strokes are coordinated choices

**Build 1, ask:** What was the analogous hidden choice in the mixture?

**Expected:** The component index c controlled where x was likely to fall.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** First learn a compact representation without requiring a probabilistic model.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## linear-compression: A low-dimensional code can reconstruct a vector

**Build 1, ask:** What restriction does a linear decoder impose?

**Expected:** Its reconstructions lie in a low-dimensional linear subspace.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Connect this idea to PCA with the right qualification.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## pca-bridge: PCA is a useful linear reference

**Build 1, ask:** Would every neural autoencoder automatically equal PCA?

**Expected:** No. Linearity, squared error and optimisation assumptions matter.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Common misconception:** A nonlinear network or a BCE objective does not inherit the PCA equivalence.

**Transition:** Build the nonlinear encoder and decoder.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## encoder: Let a network compute the code

**Build 1, ask:** Is z a stored parameter for this particular image?

**Expected:** It is a computed activation; φ are the shared stored weights.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Decode that computed code.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## decoder: A second network maps the code back

**Build 1, ask:** Why are there 784 outputs again?

**Expected:** Each output corresponds to a pixel in the reconstruction.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Name what each node means before training.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## roles: Two kinds of quantities change

**Build 1, ask:** Which quantities does the optimiser store and update?

**Expected:** The encoder and decoder weights.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Trace our held 3 through the trained encoder.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## code-three: Our 3 becomes two numbers

**Build 1, ask:** Do the coordinates have to mean digit identity and slant?

**Expected:** No. Coordinate meanings are learned and need not align with named factors.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Decode these exact coordinates.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## decode-three: Decode the code that the encoder produced

**Build 1, ask:** What should the training target be?

**Expected:** The same original image x.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Write the training pair without an external label.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## self-target: The input supplies its own target

**Build 1, ask:** Could this training rule work if digit labels were unavailable?

**Expected:** Yes. The observed image itself supplies the target.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Inspect the capacity restriction.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## bottleneck: Two values must carry enough information

**Build 1, ask:** What do you expect to lose first with a small code?

**Expected:** Fine details may be lost while broad shape remains.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Prepare a held-out comparison rather than assuming.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## predict-bottleneck: Predict before looking at reconstructions

**Build 1, ask:** Which should have more capacity to preserve handwriting detail?

**Expected:** The 16-dimensional code has more capacity, though training still matters.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Define the scalar training objective first.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## one-pixel-loss: Start with one observed pixel

**Build 1, ask:** What probability should increase to reduce this loss?

**Expected:** The probability of the observed one.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Now make the observed pixel zero.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## zero-pixel-loss: A zero uses the other probability

**Build 1, ask:** Why is the loss not −log(0.1)?

**Expected:** The observation is zero, whose probability is 1−0.1.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Combine both cases in one expression.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## bce: One expression covers both outcomes

**Build 1, ask:** Which term survives for an observed one?

**Expected:** Only −log x̂j.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Add contributions for one tiny image.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## four-pixel: Compute a four-pixel reconstruction loss

**Build 1, ask:** Do we add or average these four event contributions?

**Expected:** Here we add them; they belong to one image.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Keep the same convention for a real image.

**Source / evidence:** Chosen four-pixel example. Exact BCE independently checked against the Bernoulli log-likelihood in notebooks/lab.py.

## loss-reductions: Sum within an image; average over images

**Build 1, ask:** What unit does the final number have?

**Expected:** Mean reconstruction negative log-likelihood in nats per image.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Make those two reductions visible in code.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## ae-loss-code: The code has the same two reductions

**Build 1, ask:** Which axis is summed and which is averaged?

**Expected:** Sum 784 pixels for each image; average the batch.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Notebook demonstration:** Notebook §4: compare the manual four-pixel loss to BCEWithLogits; inspect shapes.

**Transition:** Trace the gradient to both networks.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## ae-gradient: Both networks receive the reconstruction signal

**Build 1, ask:** Does the encoder need a separate target code?

**Expected:** No. The reconstruction loss supplies a gradient through the decoder.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Perform a real update on a copied checkpoint.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## ae-update: One actual SGD step changes both networks

**Build 1, ask:** What evidence confirms that the update happened?

**Expected:** The parameter-delta checks and recomputed batch loss.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Notebook demonstration:** Notebook §4: inspected_step copies the checkpoint, updates on eight training images and checks both parameter groups.

**Transition:** Now inspect the previously trained models.

**Source / evidence:** Actual float64 SGD replay on copied ae2 checkpoint. data/fixtures.json; qa/science-checks.json.

## heldout-recon: Test the prediction on unseen images

**Build 1, ask:** Where does the larger code preserve visible detail?

**Expected:** Identify concrete stroke differences in these examples.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Use measured losses without claiming a universal ordering.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## bottleneck-measure: Measure reconstruction on the same test subset

**Build 1, ask:** Does lower reconstruction loss prove better prior samples?

**Expected:** No. This evaluates reconstruction at encoded inputs.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Try the missing sampling experiment.

**Source / evidence:** Fresh CPU replay of existing 30-epoch AE checkpoints; test0:2000, no test updates.

## decoder-sampler: The decoder can accept any two numbers

**Build 1, ask:** Where should a new code come from?

**Expected:** The reconstruction objective has not specified a distribution for it.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Try a common choice and inspect it honestly.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## prior-experiment: Try standard-normal codes anyway

**Build 1, ask:** Should these look as good as reconstructions?

**Expected:** Not necessarily: these codes were not guaranteed to resemble training codes.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Plot where the encoder actually put the data.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## code-cloud: Where did the training objective place the codes?

**Build 1, ask:** Is there any reason this cloud must be centred at zero?

**Expected:** No. Reconstruction alone leaves its location and scale unconstrained.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Put the attempted prior draws on this very plot.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## ae-failure-frame: Our proposed prior misses the learned code geometry

**Build 1, ask:** What changed from the previous figure?

**Expected:** Only the prior draws were added; the encoded data and axes stayed fixed.

**Point here:** Compare the amber draw locations with the purple cloud before discussing generated digits.

**Common misconception:** The displayed model has a mismatch; this is not a theorem that all AEs must fail for every possible prior.

**Transition:** Relate this mismatch to the observed decoded outputs.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## recon-vs-new: Two inputs to the same decoder

**Build 1, ask:** Why can one experiment work while the other fails?

**Expected:** The decoder is trained around encoded data, not necessarily every proposed random code.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Try a path between known codes.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## interpolate-rule: Interpolation supplies a different rule for z

**Build 1, ask:** Is a point on this path a standard-normal draw?

**Expected:** No. It is constructed from two encoded examples and a chosen t.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Decode the path.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## interpolation-result: A smooth path can produce a smooth visual change

**Build 1, ask:** What conclusion does this sequence support?

**Expected:** The decoder changes along this path; it does not establish global sampling quality.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Consider fitting a code distribution after training.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## fit-code-density: A density fitted to codes is a legitimate extra model

**Build 1, ask:** What additional fitting task have we introduced?

**Expected:** Learning a distribution over encoded images.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Inspect one simple fitted-code experiment.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## fitted-code-output: A fitted Gaussian changes where codes are drawn

**Build 1, ask:** Does fitting one Gaussian guarantee that it captures every code cluster?

**Expected:** No. One Gaussian may miss multiple groups, even after its mean and covariance are fitted.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Common misconception:** This exploratory display is not a quantitative generative evaluation.

**Transition:** State the precise limitation of the original objective.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## objective-says: Read what the objective actually asks for

**Build 1, ask:** Which part of this expression defines a latent prior?

**Expected:** None. The code distribution is an outcome of training, not a prescribed prior.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Check three claims about the learned representation.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## three-claims: Which claim have we earned?

**Build 1, ask:** What would you plot before decoding a proposed prior?

**Expected:** The encoded data and prior draws on exactly the same axes.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Separate a latent variable from a deterministic code.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## deterministic-code: An AE code is a computed value

**Build 1, ask:** What would a distribution-valued encoder need to output?

**Expected:** Parameters describing uncertainty, such as a mean and scale.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** But first specify the probabilistic model we actually want.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## next-model: Choose a sampling story before learning inference

**Build 1, ask:** Which earlier lecture prepared us for that inference problem?

**Expected:** Lecture 1: fit a tractable approximation to a posterior.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Retrieve the differences among our three experiments.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## exit-three-tests: Name the experiment from its input

**Build 1, ask:** Which one was the unresolved challenge for our ordinary AE?

**Expected:** Sampling from a justified code distribution.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Carry the precise failure frame into the next lecture.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## bridge-vae: Next: keep the sampling question visible

**Build 1, ask:** What should our next model specify explicitly?

**Expected:** A prior, a likelihood, and a way to infer latent variables from images.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Lecture 3 begins with this same figure and this same 3.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## optional-gaussian-likelihood: Optional: why squared error appears elsewhere (optional)

**Build 1, ask:** What has to be fixed for MSE to match likelihood maximisation?

**Expected:** The Gaussian variance, with a consistent reduction.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Compare observation models without comparing unlike loss numbers.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## optional-em: Optional: what the mixture fitting step did (optional)

**Build 1, ask:** Is c observed in the fitting data?

**Expected:** No; responsibilities are model-based soft assignments.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Return to sampling by first drawing c.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.

## optional-code-change: Optional: codes are not uniquely identified (optional)

**Build 1, ask:** Why does reconstruction alone leave code scale ambiguous?

**Expected:** The decoder can undo a corresponding invertible change in the encoder code.

**Point here:** Point to the quantity that changed, then the quantities held fixed.

**Transition:** Use a probabilistic objective to state the desired distributional roles.

**Source / evidence:** Original teaching construction; measured figures replay local binary-MNIST checkpoints. See data/provenance.json.
