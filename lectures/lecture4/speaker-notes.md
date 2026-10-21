# Denoising diffusion probabilistic models: speaker notes

Planned main route: 70 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| From noise to images | 6 |
| The known forward process | 11 |
| One supervised example | 12 |
| One loss and one update | 11 |
| One reverse step | 14 |
| Run the small DDPM | 8 |
| Text conditioning and retrieval | 8 |

## noise-to-image: How could this become an image?

**Build 1, ask:** What computation could connect these two states?

**Expected:** A sequence of learned reverse transitions, starting from noise.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Before learning to reverse anything, define a process we can run exactly.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## show-trajectory: Many small decisions make one sample

**Build 1, ask:** Which clean training image is this sampler trying to reconstruct?

**Expected:** None. It starts from new noise and generates a new sample.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Use a real image to construct the training task.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## held-example: For this lecture, follow one handwritten 7

**Build 1, ask:** Why keep one image visible through the calculations?

**Expected:** It lets us connect each symbol to a concrete state rather than swap examples silently.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Corrupt this same image at different noise levels.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## forward-film: The easy direction destroys information

**Build 1, ask:** Is this particular display a sequential Markov-chain path?

**Expected:** No. It uses one fixed noise draw to compare direct marginals at different t.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Now define the actual one-step forward transition.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## two-directions: Known corruption; learned reverse transitions

**Build 1, ask:** Which direction requires training?

**Expected:** The reverse distribution. The forward corruption schedule is chosen and fixed.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Start with one pixel so the coefficients have meaning.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## one-step-pixel: One pixel, one small corruption

**Build 1, ask:** Why not multiply noise by β₁ directly?

**Expected:** A standard-normal noise term multiplied by √β₁ has variance β₁.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Write the same operation as a conditional distribution.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## forward-kernel: The forward step is a Gaussian conditional

**Build 1, ask:** Does q contain a learned encoder in this lecture?

**Expected:** No. The symbol q names a fixed, declared noising distribution.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Separate one-step retention from cumulative retention.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## alpha-vs-alphabar: One step and many steps need different symbols

**Build 1, ask:** Is ᾱₜ the same as αₜ?

**Expected:** Only at the first step. The bar denotes a product over the preceding steps.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Expand two steps to see why the product appears.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## two-step-expansion: Expand two corruption steps

**Build 1, ask:** Why can we combine the two noise terms?

**Expected:** Independent Gaussian terms sum to a Gaussian; their variances add after scaling.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** The same argument gives a direct marginal at any time.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## direct-marginal: Jump directly to any chosen timestep

**Build 1, ask:** Must training simulate all earlier corruption steps?

**Expected:** No. This direct construction draws the correct marginal at the selected t.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Give the symbolic coefficients a concrete schedule.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## schedule: Choose a schedule whose endpoint is near noise

**Build 1, ask:** Why is setting T = 20 with the same endpoints a different model?

**Expected:** It changes the accumulated product and may leave substantial signal at the endpoint.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Read a few schedule values rather than hide them in code.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## schedule-values: The bar carries the cumulative effect

**Build 1, ask:** Which coefficient belongs in the direct xₜ formula?

**Expected:** ᾱₜ, not αₜ.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Calculate one scalar instance using that exact schedule.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-inputs: Choose one scalar training example

**Build 1, ask:** Which of these three quantities is learned?

**Expected:** None. They define one training example; model weights are learned later.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Combine the signal and noise contributions.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-forward: Compute the noisy pixel

**Build 1, ask:** Is the result required to stay in [−1,1]?

**Expected:** No. Gaussian corruption has unbounded support. Do not clip training states.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Move from one scalar to a full image without changing the rule.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## image-forward: Apply the same construction to every pixel

**Build 1, ask:** Are noise values shared across pixels?

**Expected:** No. Each pixel receives a Gaussian draw; the same scalar coefficients apply to the full tensor.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** A supervised learning tuple is now available.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## tuple: What does one training example contain?

**Build 1, ask:** Does the denoiser see x₀ during its forward call?

**Expected:** No. It receives xₜ and t; ε is supplied to the loss as the target.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Make the availability of each variable explicit.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## known-training: Training knows how the input was made

**Build 1, ask:** Why can this be supervised without manual noise labels?

**Expected:** We sampled ε and used it to construct xₜ, so the label is already known.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Time must be part of the prediction problem.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## time-needed: The same value can mean different things at different t

**Build 1, ask:** Could a denoiser ignore the noise level?

**Expected:** It would mix prediction tasks with different signal/noise proportions. t identifies the task.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Inspect one image across clearly different timesteps.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## noise-levels: The prediction problem changes with noise level

**Build 1, ask:** At which stage is the original image least identifiable?

**Expected:** At high t, where the remaining signal coefficient is very small.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Known training targets do not make the inverse unique.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## ambiguity: One noisy input can have several explanations

**Build 1, ask:** Can the network always recover the exact injected ε from xₜ alone?

**Expected:** No. The inverse problem is ambiguous; training minimises an expected prediction error.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** What does squared error prefer in an ambiguous problem?

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## conditional-mean: Squared error learns a conditional average

**Build 1, ask:** Is an imperfect match to a particular ε necessarily a bug?

**Expected:** No. Some uncertainty is irreducible from the available input.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Keep this prediction separate from a cleaned image.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## noise-not-image: The network predicts a noise tensor

**Build 1, ask:** Why does the network output look like noise?

**Expected:** That is the declared target. A clean estimate is computed from it in a later step.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Specify the shape before the loss.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## shapes: The output shape matches the noise target

**Build 1, ask:** What shape mistake could silently corrupt the construction?

**Expected:** Mixing batch coefficients with channel or spatial positions. Assert tensor shapes.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Write the data construction in a few lines.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## tuple-code: Construct inputs and labels in one short block

**Build 1, ask:** Where is the label tensor created?

**Expected:** eps is the label as well as the known random input to corruption.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Compute a loss before showing a full network.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-predictor: Begin with a predictor we can differentiate by hand

**Build 1, ask:** What is the single trainable quantity?

**Expected:** θ. The selected x₀, t and ε remain fixed for this inspected update.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Subtract the known noise target.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-residual: Compare prediction with the injected noise

**Build 1, ask:** Which tensor is the target in this subtraction?

**Expected:** The sampled ε, not x₀ or xₜ.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Square the residual to obtain the objective.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-loss: A squared residual gives one scalar loss

**Build 1, ask:** Is this already the batch mean?

**Expected:** It is one scalar example. Image and batch reductions come next.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Take the derivative with the random choices held fixed.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-gradient: Differentiate through the prediction

**Build 1, ask:** Why hold ε fixed during a finite-difference check?

**Expected:** Otherwise changes in the random draw contaminate the comparison between perturbed parameters.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Perform one literal SGD step.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## scalar-update: One update, then recompute the same loss

**Build 1, ask:** Does this same-example decrease prove a good image model?

**Expected:** No. It verifies the direction and arithmetic of a local update.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Extend the loss to pixels and a batch.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## loss-reduction: State exactly what “mean squared error” averages

**Build 1, ask:** How does an event sum followed by batch mean differ?

**Expected:** For fixed image size it differs by the factor CHW, changing the scale and effective learning rate.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Use the same reduction in the code and the reported values.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## update-code: A complete training update is short

**Build 1, ask:** Which variables does the optimiser update?

**Expected:** Only the denoiser parameters θ. The forward schedule stays fixed.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Replace the scalar predictor with a small image network.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## unet-shape: A small U-Net keeps context and detail

**Build 1, ask:** Why retain high-resolution features?

**Expected:** They help localise details while the coarse path supplies wider spatial context.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Time conditions those feature computations.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## time-conditioning: The timestep enters the network explicitly

**Build 1, ask:** Do we train a separate network at each timestep?

**Expected:** No. One set of weights is conditioned on t.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Check one real update on training images.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## neural-update: Inspect a real image-model update

**Build 1, ask:** Did we train on the held image shown earlier?

**Expected:** No. This update uses the official training split.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Training learned a predictor. Sampling now needs a transition.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## training-vs-sampling: Training and sampling start with different information

**Build 1, ask:** Can the sampler subtract the true injected noise?

**Expected:** No. A newly generated state has no known training corruption label.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** First see what a predicted noise value implies about x₀.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## clean-estimate: Solve the forward equation for a clean estimate

**Build 1, ask:** Is this already the next noisy state xₜ₋₁?

**Expected:** No. It estimates x₀, while the sampler needs a distribution for the previous noisy state.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Compare the estimate to the actual reverse mean.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## estimate-image: A clean estimate is useful, but is not a reverse step

**Build 1, ask:** Would jumping straight to this estimate reproduce the DDPM sampler?

**Expected:** No. The declared sampler takes successive reverse transitions.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Define that learned conditional distribution.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## reverse-kernel: Model the previous state with a Gaussian

**Build 1, ask:** Which part of this distribution is learned here?

**Expected:** The mean through εθ. The variance is an explicit, fixed choice.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Express the mean in the noise-prediction parameterisation.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## reverse-mean: The predicted noise determines the reverse mean

**Build 1, ask:** Why are both αₜ and ᾱₜ present?

**Expected:** One describes the local transition; the other describes cumulative corruption relative to x₀.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Verify a numerical reverse mean independently.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## reverse-numbers: One scalar reverse step has explicit inputs

**Build 1, ask:** Do we know the clean image during this calculation?

**Expected:** The reverse formula uses xₜ, t and the predicted noise, not a known x₀.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Compute both derived quantities and keep them separate.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## mean-vs-estimate: The clean estimate and reverse mean are different

**Build 1, ask:** Why is the previous-state mean closer to xₜ than x̂₀ is?

**Expected:** It is one small reverse transition, not the endpoint of the full denoising chain.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** State the variance before drawing the previous state.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## posterior-variance: Choose the reverse variance explicitly

**Build 1, ask:** Why does the t = 1 boundary need special attention?

**Expected:** ᾱ₀ = 1 makes this added variance zero at the last step.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Add a sampled stochastic term at non-final steps.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## sample-previous: Draw around the reverse mean

**Build 1, ask:** If ξ were positive instead, would this be the same sample?

**Expected:** No. It would be another draw from the specified reverse Gaussian.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Handle the final boundary without adding fresh noise.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## final-boundary: At the final step, do not add noise

**Build 1, ask:** What happens if we add ordinary noise again at t = 1?

**Expected:** We corrupt the final sample instead of following the declared sampler.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Now repeat the verified transition in a loop.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## reverse-code: The complete sampling loop uses the model repeatedly

**Build 1, ask:** Where does the sampler obtain its first state?

**Expected:** A new standard-normal tensor, justified by the near-noise endpoint of the forward schedule.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Check the resulting trajectory rather than only the final frame.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## reverse-replay: Inspect successive states on one trajectory

**Build 1, ask:** Are all intermediate states expected to look like clean images?

**Expected:** No. They represent progressively less corrupted states, not independent final samples.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Connect this construction to the latent-variable framework.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## latent-chain: The noisy states form a latent-variable model

**Build 1, ask:** Which states are observed in the training data?

**Expected:** Only x₀. The forward process constructs auxiliary noisy states for training.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** The known forward chain supports a variational bound.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## vi-connection: The forward process supplies a tractable training distribution

**Build 1, ask:** Is this q trained as an amortised encoder like the VAE?

**Expected:** Not in this DDPM: it is a fixed corruption chain.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** For Gaussian transitions, the comparison connects to a squared error.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## weighted-vs-simple: The common noise loss simplifies the variational objective

**Build 1, ask:** Does any noise MSE value directly equal a log likelihood?

**Expected:** No. The weighting and boundary terms matter for the variational objective.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Inspect exactly what our small model was trained to minimise.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## training-record: What was actually trained?

**Build 1, ask:** Does this package claim state-of-the-art image quality?

**Expected:** No. It is a small teaching model with explicit training and sampling evidence.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Look at training loss and samples with their proper scope.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## history: A falling training loss is useful but incomplete evidence

**Build 1, ask:** Why should successive batch values fluctuate?

**Expected:** They reflect different sampled training problems as well as changing parameters.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Check the generated outputs too.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## sample-gallery: Inspect several fresh-noise samples

**Build 1, ask:** What can and cannot this grid establish?

**Expected:** It reveals visible sample behaviour; it does not prove generalisation or distributional equality.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Run one real update and a small sampler in the notebook.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## notebook-demo: Live: one update, then use the saved sampler

**Build 1, ask:** Which work is reduced in quick mode?

**Expected:** Training iterations and number of generated samples, not the validated schedule.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Notebook demonstration:** lecture4.ipynb: inspected update and quick checkpoint sampling. Full-training cell defaults to disabled.

**Transition:** Connect the same denoiser idea to text-conditioned images.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## text-conditioning: Text adds a condition to the denoiser

**Build 1, ask:** Where should the text information enter the computation?

**Expected:** Into the denoiser’s prediction, rather than as a replacement for the random starting state.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Relate this to the course’s cross-attention machinery.

**Source / evidence:** Original schematic · Rombach et al. (2022), arxiv.org/abs/2112.10752

## cross-attention: Cross-attention lets image features consult text

**Build 1, ask:** Which side asks the questions in this diagram?

**Expected:** The current image/latent features supply queries; text embeddings supply keys and values.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Compression can make the denoising computation cheaper.

**Source / evidence:** Original schematic · Rombach et al. (2022), arxiv.org/abs/2112.10752

## latent-diffusion: The same idea can operate in a compressed space

**Build 1, ask:** Is the final image decoder the denoiser itself?

**Expected:** No. Denoising occurs in latent space; a separate decoder maps the final latent to pixels.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Guidance changes how conditional predictions are combined.

**Source / evidence:** Original teaching redraw · Rombach et al. (2022), arxiv.org/abs/2112.10752

## guidance: Guidance combines two predictions

**Build 1, ask:** Does larger guidance simply make every aspect of a sample better?

**Expected:** No. It changes the sampling trade-off and can reduce diversity or create artefacts.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Return to the basic algorithm students should now reconstruct.

**Source / evidence:** Original notation · Ho & Salimans (2022), arxiv.org/abs/2207.12598

## exit-training: Exit check: what is known during training?

**Build 1, ask:** Which quantities disappear when generating from new noise?

**Expected:** There is no clean target image or true injected ε for the current generated state.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Retrieve one reverse step.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## exit-reverse: Exit check: reconstruct one reverse step

**Build 1, ask:** What special action happens at the final boundary?

**Expected:** No extra noise is added at t = 1.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Connect the five lecture questions.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## five-model-questions: Four lectures, four connected questions

**Build 1, ask:** What common habit prevents confusion across all these models?

**Expected:** Name what is observed, sampled, predicted, fixed and updated.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Use the notebooks to reproduce the calculations.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## optional-posterior: Optional: the tractable training posterior (optional)

**Build 1, ask:** Why is this posterior available during training but not ordinary generation?

**Expected:** Training starts from a known x₀; generation does not.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Replace x₀ with the network-derived estimate when interpreting the reverse mean.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## optional-mean-check: Optional: two forms of the same reverse mean (optional)

**Build 1, ask:** What substitution links this to the ε-prediction formula?

**Expected:** Insert x̂₀ = (xₜ−√(1−ᾱₜ)εθ)/√ᾱₜ and simplify.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** The notebook independently checks the two implementations.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239

## optional-display: Optional: image display is not model arithmetic (optional)

**Build 1, ask:** Why can identical-looking grey pixels represent different model values?

**Expected:** The display clips extreme values; the numerical tensor retains them.

**Point here:** Follow the labelled state, known target or computation on the frame.

**Common misconception:** A noisy state, a noise prediction, a clean estimate and a reverse mean are different quantities.

**Transition:** Inspect raw tensor ranges alongside pictures.

**Source / evidence:** Original teaching construction · Ho, Jain & Abbeel (2020), arxiv.org/abs/2006.11239
