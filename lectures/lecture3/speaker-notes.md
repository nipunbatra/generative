# Generative adversarial networks: speaker notes

Planned main route: 60 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| A learning signal | 6 |
| One discriminator update | 10 |
| One generator update | 12 |
| Alternate the players | 11 |
| Samples, coverage and failure | 11 |
| MNIST and retrieval | 10 |

## start: Where could a generator get feedback?

**Build 1, ask:** What training signal could produce these samples?

**Expected:** A learned discriminator can supply differentiable feedback.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** First separate producing an image from scoring it.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## generator-is-sampler: A generator turns a draw into a sample

**Build 1, ask:** Which part introduces randomness?

**Expected:** The sampled z; G is deterministic once z and θ are fixed.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Sampling does not automatically give us a convenient density.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## likelihood-question: Sampling is easier than evaluating a density

**Build 1, ask:** Can this network readily report the probability of an image?

**Expected:** The usual GAN supplies an implicit sampler; its training signal comes from a classifier.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Make that classifier’s job concrete.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## toy-vote: Which distribution is missing something?

**Build 1, ask:** Where should generated samples move?

**Expected:** They need to cover both observed regions, not merely one plausible point.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** A discriminator can learn where the two sources differ.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## two-players: One network proposes; one distinguishes

**Build 1, ask:** Must we keep D to generate after training?

**Expected:** No. Draw z and run G.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Now hold G still and train the classifier.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## data-labels: Fix the samples and label their source

**Build 1, ask:** Does label 1 mean “a perfect digit”?

**Expected:** It means the example came from the observed-data source in this classification task.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Use a scalar classifier so every number is visible.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## chosen-networks: Choose networks small enough to inspect

**Build 1, ask:** What are the trainable quantities?

**Expected:** a,b belong to G; w,c belong to D. x and z are fixed for this calculation.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Evaluate the generator first.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## fake-arithmetic: First compute the fake

**Build 1, ask:** What sample does this z produce?

**Expected:** Zero. The observed example is at x = 2.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Feed each example into the same discriminator.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-logits: A score comes before a probability

**Build 1, ask:** Why keep raw logits in the implementation?

**Expected:** Stable binary cross-entropy can operate directly on logits.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Translate the labels into a loss.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-real-term: Reward a high score on real data

**Build 1, ask:** Would D(real) closer to one increase this loss?

**Expected:** No, it reduces the negative log probability of the correct label.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** The fake example requires the opposite label.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-fake-term: Reward a low score on generated data

**Build 1, ask:** Which direction should D(fake) move?

**Expected:** Towards zero during the discriminator phase.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Combine the two contributions with a declared reduction.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-loss: Add the two classification losses

**Build 1, ask:** Is this an average over the combined two-example batch?

**Expected:** Our convention sums the two source means; a combined mean differs by a factor of two.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Now take a derivative while holding G fixed.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-gradient: Which derivative changes the classifier?

**Build 1, ask:** Which term contributes to the weight derivative when fake x is zero?

**Expected:** Only the real term contributes to ∂L/∂w; both contribute to the bias derivative.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Make one literal parameter update.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-update: One D update, with G unchanged

**Build 1, ask:** Which row must remain identical?

**Expected:** The generator parameters. This phase updates only ω.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Re-evaluate rather than assuming improvement.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-recompute: Check the same examples after the update

**Build 1, ask:** Did the discriminator improve on these two examples?

**Expected:** Yes: real score rises, fake score falls, and the same-batch loss falls.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** The code must prevent this update from reaching G.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## detach: Stop this phase at the fake sample

**Build 1, ask:** Why detach fake rather than remove D from the graph?

**Expected:** D needs gradients; G should not receive this phase’s gradients.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** A test should check actual parameter changes.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## detach-code: Detach only the generated sample

**Build 1, ask:** Which part of this computation must still be differentiable?

**Expected:** The discriminator forward pass; only the path back into G is cut.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Test that only the intended parameters actually change.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-check: Check ownership of the update

**Build 1, ask:** Does a nonzero stale .grad prove this phase updated a parameter?

**Expected:** No. Clear gradients and compare parameter values before and after the actual optimiser step.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** See what repeated D-only updates do.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## d-only: What happens if only D learns?

**Build 1, ask:** Predict the score near the real modes and the fake region.

**Expected:** D learns to distinguish these fixed sources; the generator itself has not improved.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** We need a signal that can move G’s samples.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-goal: Now let the generator ask for label 1

**Build 1, ask:** Are we relabelling generated data for the next D update?

**Expected:** No. Label 1 defines G’s objective; D still trains fake examples with label 0.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Keep D fixed and trace the derivative through it.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## freeze-not-detach: Frozen weights can still transmit a gradient

**Build 1, ask:** What does “freeze D” mean here?

**Expected:** Do not update ω. Preserve the differentiable path from loss through D to G.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Separate the derivative chain into its local pieces.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-chain: Follow the chain rule into the generator

**Build 1, ask:** Can a fixed w appear in a derivative?

**Expected:** Yes. It multiplies the input derivative even though we do not optimise w in this phase.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Insert the values from the updated discriminator.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-gradient: The generator receives a nonzero derivative

**Build 1, ask:** Will gradient descent move a up or down?

**Expected:** Up: subtracting a negative derivative increases a.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Perform that update on both generator parameters.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-update: One G update, with D unchanged

**Build 1, ask:** Which object moved even though the random draw stayed fixed?

**Expected:** The generated sample moved because G’s parameters changed.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Recompute G’s objective against the same fixed D.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-recompute: Check the updated sample and loss

**Build 1, ask:** Does this one improvement establish coverage of both modes?

**Expected:** No. It checks the local learning signal for one fixed draw.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Implement the same ownership in a neural network.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-code: Keep the path; freeze the owner

**Build 1, ask:** What would G(z).detach() do in this code?

**Expected:** It would sever the very path needed to update G.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Compare the two operations side by side.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## freeze-vs-detach: Two operations with different effects

**Build 1, ask:** Which operation is correct during the G phase?

**Expected:** Freeze D parameters, keep ordinary autograd through its forward computation.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Inspect the assertions rather than trust the diagram.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-check: Verify the actual parameter changes

**Build 1, ask:** Why compare parameters as well as gradients?

**Expected:** The optimiser decides what actually changes; a gradient alone is insufficient evidence.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** What if D remains frozen for hundreds of updates?

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## g-only: A fixed critic can be exploited

**Build 1, ask:** Would optimising against one fixed D guarantee the data distribution?

**Expected:** No. G can exploit that particular critic; alternating training changes the task.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Join the two verified phases.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## alternating: The training loop alternates two objectives

**Build 1, ask:** Are both players minimising one fixed function of their own weights?

**Expected:** Each player sees a changing opponent. The practical update uses two stated objectives.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Write the batch losses with their sampling distributions.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## batch-d: The discriminator has two expectations

**Build 1, ask:** Where does each expectation get its samples?

**Expected:** Real examples come from the dataset; latent draws come from the chosen noise distribution.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** The generator needs only latent draws and D’s response.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## batch-g: The practical generator loss

**Build 1, ask:** Why is there no paired target image for each z?

**Expected:** G is judged by the discriminator, not by reconstruction of a paired observation.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Distinguish this from the original minimax generator term.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## minimax: The original game uses a different G term

**Build 1, ask:** Is −log D(fake) identical to log(1−D(fake))?

**Expected:** No. They encourage the same desired direction but have different gradients.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Compare their logit derivatives.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## saturation-derivative: A confident discriminator can weaken one signal

**Build 1, ask:** Which derivative is stronger when D rejects fake samples confidently?

**Expected:** The non-saturating objective’s logit derivative is near −1.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Plot the derivatives over the same logit range.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## saturation-plot: The difference is visible before any training

**Build 1, ask:** Does a stronger logit derivative guarantee stable GAN training?

**Expected:** No. The rest of the network Jacobian and the moving opponent still matter.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Assemble the two stable BCE losses in code.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## alternating-code-d: The first half of one neural update

**Build 1, ask:** Why detach only in this half?

**Expected:** This loss trains the discriminator while keeping G fixed.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Now rebuild fresh fakes for the second half.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## alternating-code-g: The second half of one neural update

**Build 1, ask:** What changed between the two forward passes?

**Expected:** D’s parameters changed. G needs feedback from the updated discriminator.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Test the full loop on the simple distribution.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## toy-progress: Can alternating updates cover both regions?

**Build 1, ask:** What would count as success in this simple test?

**Expected:** Samples near both modes, reasonable spread, and limited mass between them.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Quantify coverage, with the metric’s limits visible.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## coverage: Count mass near each known mode

**Build 1, ask:** What does a near-zero entry tell us?

**Expected:** The generator misses one observed region under this diagnostic.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Look directly at the failing sample distribution.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## collapse: Plausible samples can omit a whole mode

**Build 1, ask:** Why is looking at a single generated sample insufficient?

**Expected:** It cannot reveal missing modes or the frequency of different outputs.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Compare loss curves for the stable and collapsed runs.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## losses-not-quality: A loss curve is not a coverage certificate

**Build 1, ask:** Can the lower displayed G loss alone identify the better distribution?

**Expected:** No. D changes during training, so the numerical loss is not a fixed external quality measure.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Explain why the opponent changes the landscape.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## moving-objective: The target of optimisation keeps moving

**Build 1, ask:** Must both plotted losses fall on every iteration?

**Expected:** No. The opponent changes the function being evaluated.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Choose diagnostics that answer concrete questions.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## diagnostic-grid: Ask separate questions about the samples

**Build 1, ask:** What does fixed z control in a training film?

**Expected:** It separates changes in G from changes in the sampled latent inputs.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Transfer the same two-phase algorithm to images.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## image-representation: Keep the data convention visible

**Build 1, ask:** Can we compare GAN loss numbers to the VAE reconstruction loss?

**Expected:** No. Their objectives, observation representations and reductions differ.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Expand the generator’s tensor shapes.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## image-g-shapes: The generator expands a noise vector

**Build 1, ask:** Why use tanh at the output here?

**Expected:** It matches the chosen [−1,1] image representation.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** The discriminator performs the opposite spatial reduction.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## image-d-shapes: The discriminator returns one logit per image

**Build 1, ask:** Does D predict one target per pixel?

**Expected:** No. This discriminator produces one source-classification logit for each whole image.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Inspect the same fixed draws over training.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## image-training-early: The same z before and after early learning

**Build 1, ask:** What is changing between these grids?

**Expected:** Generator weights; the latent vectors are held fixed.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Compare later checkpoints without changing the selection.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## image-training-later: Structure emerges, but quality varies

**Build 1, ask:** Which limitations remain visible?

**Expected:** Some samples are ambiguous; a fixed grid alone does not establish distributional fidelity.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Separate reproduced evidence from newly executed updates.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## replay-evidence: What did we reproduce today?

**Build 1, ask:** Does successful replay mean we retrained the image GAN?

**Expected:** No. It verifies the checkpoint and sample-generation path.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Run the notebook’s short inspected training step.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## demo: Live: inspect who changes

**Build 1, ask:** Which two parameter-change checks must be exactly zero?

**Expected:** G during the D phase and D during the G phase.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Notebook demonstration:** lecture3.ipynb: “Two neural updates with parameter ownership checks”.

**Transition:** Retrieve the central gradient-path distinction.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## exit-detach: Exit check: where should detach go?

**Build 1, ask:** How does the answer change between the two phases?

**Expected:** The same forward topology serves different parameter owners and targets.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Connect the training signal to the other generative models.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## compare-vae: Two ways to train a sampler

**Build 1, ask:** What did the discriminator replace in the training story?

**Expected:** It supplies feedback without evaluating a decoder likelihood or an ELBO.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Diffusion will build a supervised target by corrupting real data.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## next-question: Can we create the training target ourselves?

**Build 1, ask:** What information becomes available when we create the corruption?

**Expected:** The clean example, timestep and injected noise are all known during training.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Begin diffusion with a concrete corrupted image.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## optional-optimal-d: Optional: the ideal classifier at fixed G (optional)

**Build 1, ask:** Under what assumptions is this ideal discriminator statement meaningful?

**Expected:** Unlimited classifier capacity and exact optimisation at fixed distributions; finite neural training need not reach it.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Return to measured diagnostics rather than treating the ideal as a guarantee.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## optional-batchnorm: Optional: frozen parameters are not all model state (optional)

**Build 1, ask:** Does requires_grad_(False) switch a network to evaluation mode?

**Expected:** No. Parameter gradients and train/eval mode are separate controls.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Make the intended training-state convention explicit in implementations.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661

## optional-controls: Optional: changes to the game are experiments (optional)

**Build 1, ask:** Does one stabilised run prove a remedy always works?

**Expected:** No. It supports the measured configuration only.

**Point here:** Follow the labelled computation or data on this frame.

**Common misconception:** A plausible sample alone does not establish the learned distribution.

**Transition:** Use the companion notebook for controlled extensions.

**Source / evidence:** Original teaching construction · Goodfellow et al. (2014), arxiv.org/abs/1406.2661
