# Diffusion models: speaker notes

Planned main route: 130 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Lecture map | 1 |
| Opening · Noise to a digit | 6 |
| Part 1 · Destroying is easy | 20 |
| Part 2 · Undoing it: one step back | 12 |
| Part 3 · Training: labels for free | 18 |
| Part 4 · Generating: walk back from noise | 18 |
| Part 5 · Steering: labels, guidance and text | 20 |
| Part 6 · Faster sampling | 12 |
| Part 7 · What the network learns, and wrap-up | 16 |

## contents: Lecture 5 at a glance: eight parts, numbered sections

**Build 1, ask:** Which part do you expect to be hardest?

**Expected:** Usually Part 2, the step back; we follow every number.

**Point here:** Click a section to jump to its first slide; ▶ opens the matching notebook section. Run the notebook top to bottom once before class.

**Transition:** Opening: noise to a digit.

## hook: Static in, a handwritten digit out

**Build 1, ask:** What could turn random numbers into a digit?

**Expected:** Collect guesses: a decoder (VAE), a generator (GAN)… Today: many small denoising steps.

**Build 2, ask:** Is this digit copied from the training set?

**Expected:** No. Part 7 measures it: our samples are as far from the nearest training image as real test digits are.

**Point here:** Our model: a U-Net with 814,145 parameters, 30,000 training updates of batch 128. The film is sampling run 0, shown every 50 steps.

**Transition:** Before the how: what this same loop does at scale.

## apps: The same loop runs today’s image generators

**Build 1, ask:** What is the same across all six?

**Expected:** Each starts from noise and removes it step by step, steered by a different input.

**Build 2, ask:** Which of them do you think is hardest?

**Expected:** Open question; return to it at the end.

**Transition:** We have built two generators already. What was wrong with them?

**Source / evidence:** Published samples: LDM (Rombach et al. 2022) · RePaint (Lugmayr et al. 2022) · SR3 (Saharia et al. 2022) · SDEdit (Meng et al. 2022) · ControlNet (Zhang et al. 2023) · DreamBooth (Ruiz et al. 2023). Showcase after <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 1</a>

## three-gens: Two generators so far, and a third idea

**Build 1, ask:** Why are VAE samples blurry?

**Expected:** The decoder predicts an average over the plausible images (a Gaussian or Bernoulli mean).

**Build 2, ask:** What went wrong with GANs?

**Expected:** Mode collapse and unstable alternating training; no likelihood to monitor.

**Build 3, ask:** What might we pay for a thousand steps?

**Expected:** Time: a thousand network calls per image. Part 6 cuts that.

**Point here:** VAE: the β = 1 model of Lecture 3, prior samples (probabilities shown). GAN: Lecture 4’s MNIST GAN after 30 epochs. Diffusion: our model, Part 4.

**Transition:** The idea in one sentence.

## core-idea: The idea in one sentence

**Build 1, ask:** Can you un-mix ink from water?

**Expected:** Not physically: diffusion only runs forwards.

**Build 2, ask:** Why would small steps be easier to undo than one big one?

**Expected:** Between neighbouring frames almost nothing changes, so the previous frame is almost determined by the next one. Part 2 makes this exact.

**Point here:** This physical process gave the method its name: Sohl-Dickstein et al. (2015) borrowed it from non-equilibrium thermodynamics. Paras Chopra’s explainer uses a different everyday example; we use ink in water throughout.

**Transition:** The plan for today.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> · <a href="https://arxiv.org/abs/1503.03585">Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” (2015)</a>

## route: The plan: destroy, step back, train, generate

**Build 1, ask:** Which direction is easy?

**Expected:** Adding noise: one line of code.

**Build 2, ask:** What does the network have to learn?

**Expected:** How to undo one small step of noise, at every noise level.

**Transition:** Part 1: destroying is easy.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a>

## one-step: One noising step: shrink a little, add a little noise

**Build 1, ask:** Why √β and not β?

**Expected:** β is a variance; its square root is the size (standard deviation) of the kick.

**Build 2, ask:** Check: √0.9 × 1 + √0.1 × 0.3 = ?

**Expected:** 0.949 + 0.095 = 1.044.

**Build 3, ask:** What would happen without the shrink?

**Expected:** Next slide: the values spread out more and more.

**Transition:** Without the shrink.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> · <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a>

## explode: Keep adding noise: what happens to the spread?

**Build 1, ask:** Where does 3.3 come from?

**Expected:** The variances add up: 0.8 (the image) + Σβ = 10.1. Every extra step adds more.

**Build 2, ask:** Why does the shrink keep the spread at 1?

**Expected:** The next slide: (1 − β) of the old variance plus β of new noise.

**Point here:** Naive noising: Var(x_T) = Var(x₀) + Σβₜ ≈ 10.9. With more steps or bigger β it grows without bound, and where the noise ends depends on the image.

**Transition:** Why the shrink works, in one line.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=EhndHhIvWWw">Deepia, “Diffusion Models: DDPM, Generative AI Animated” (2025)</a>

## shrink-math: Shrink a little, then add: the spread stays at 1

**Build 1, ask:** Why √(1 − β) and not (1 − β)?

**Expected:** So that the variance, which goes with the square, is multiplied by exactly 1 − β.

**Build 2, ask:** Which rule did we use?

**Expected:** Variances of independent terms add; a constant factor comes out squared.

**Build 3, ask:** What does this guarantee at the end?

**Expected:** Whatever the image, after many steps the values have mean 0 and variance 1.

**Transition:** Watch it on the whole 3.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=EhndHhIvWWw">Deepia, “Diffusion Models: DDPM, Generative AI Animated” (2025)</a> · <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a>

## toy-schedule: A toy schedule you can do by hand: five steps

**Build 1, ask:** Why does α fall as β grows?

**Expected:** α = 1 − β: more noise means keeping less of the old value.

**Build 2, ask:** Why multiply the α’s?

**Expected:** Every step shrinks the signal again: after t steps, by √(α₁ α₂ ⋯ αₜ) = √ᾱₜ.

**Build 3, ask:** Do the squares of the two coefficients add to 1?

**Expected:** Yes, ᾱ + (1 − ᾱ) = 1: signal and noise share a fixed budget, so the spread stays 1.

**Point here:** A toy for the arithmetic only. Real schedules take 1,000 small steps, so that each step back is easy (Part 2).

**Transition:** Push our pixel through the five steps.

## toy-forward: Forward, step by step: every number

**Build 1, ask:** Compute x₂ yourself.

**Expected:** √0.8 × 1.044 + √0.2 × (−0.5) = 0.710.

**Build 2, ask:** How can an ink pixel become negative?

**Expected:** The noise kick is as large as the shrunk signal.

**Build 3, ask:** Could you recover x₀ from x₅?

**Expected:** Not reliably: ink and paper pixels end up in overlapping places. That is Part 2’s problem.

**Transition:** Do we have to run every step?

## shortcut-q: Do we need a 1,000-step loop for one noisy example?

**Build 1, ask:** What kind of random variable is x₂?

**Expected:** A constant times x₀ plus a sum of independent Gaussians.

**Build 2, ask:** How many noise draws does x₁₀₀₀ involve?

**Expected:** A thousand.

**Build 3, ask:** Any way to collapse this?

**Expected:** A sum of independent Gaussians is one Gaussian: next slide.

**Transition:** Collapse two steps by hand.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a>

## two-steps: Two steps by hand: two noises become one

**Build 1, ask:** Why may we add the variances?

**Expected:** ε₁ and ε₂ are independent.

**Build 2, ask:** Check: α₂β₁ + β₂ = 1 − α₁α₂?

**Expected:** α₂(1 − α₁) + 1 − α₂ = 1 − α₁α₂. Yes.

**Build 3, ask:** What is the signal coefficient after two steps?

**Expected:** √(α₁α₂): the product of the shrink factors.

**Build 4, ask:** Is the combined ε a new random draw?

**Expected:** It is a fixed combination of ε₁ and ε₂, with the right N(0, 1) distribution. To make a training example we can draw it directly.

**Point here:** Illustrative numbers only: real steps are tiny (β₁ = 0.0001). Step by step: x₁ = √0.9 + √0.1 × 0.3 = 1.0436; x₂ = √0.8 × 1.0436 + √0.2 × (−0.5) = 0.7098. Combined ε = (√0.8 √0.1 × 0.3 + √0.2 × (−0.5)) / √0.28 = −0.2622.

**Transition:** The same argument for any t.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a>

## shortcut: The shortcut: jump straight to any step

**Build 1, ask:** Which columns of the toy table are the two coefficients?

**Expected:** √ᾱₜ (signal) and √(1 − ᾱₜ) (noise).

**Build 2, ask:** What is x₂ with ε = 0?

**Expected:** √0.72 = 0.849: just the shrunk signal.

**Build 3, ask:** How many lines of code to make a noisy example?

**Expected:** One.

**Common misconception:** The shortcut gives the same distribution as stepping, not the same draw: a stepwise run uses t noise draws, the jump uses one. The next slide checks it.

**Transition:** Check the shortcut by simulation.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (IV)

## shortcut-check: Step by step or one jump: the same distribution

**Build 1, ask:** Are the two runs giving the same noisy pixel?

**Expected:** No, different draws; the same distribution. For training that is all we need.

**Transition:** How fast should we destroy?

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a>

## schedule: The real schedule: 1,000 small steps

**Build 1, ask:** Why 1,000 small steps rather than 5 big ones?

**Expected:** Small steps are easy to undo (Part 2).

**Build 2, ask:** At t = 250, which is bigger, signal or noise?

**Expected:** About equal: 0.724 and 0.690.

**Point here:** ᾱ₂₅₀ is a product of 250 α’s, each a little below 1. The calculator on the next slide shows the product for any t.

**Transition:** Try any step yourself.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · <a href="https://arxiv.org/abs/2102.09672">Nichol &amp; Dhariwal, “Improved Denoising Diffusion Probabilistic Models” (2021)</a>

## calc-forward: Try it: every number of a forward jump

**Build 1, ask:** Real schedule, t = 250, ink, ε = −0.5: what is xₜ?

**Expected:** 0.379, our running example.

**Transition:** Now on a whole image.

## film: Slide t: watch our 3 dissolve

**Build 1, ask:** At what t can you no longer see the 3?

**Expected:** Around t = 500–600.

**Build 2, ask:** Our pixel is ink. Why can it read 0 or below?

**Expected:** The noise part √(1 − ᾱ)ε (here ε = −0.5) outweighs the shrunk signal.

**Point here:** The bars are the two coefficients of 1.2, √ᾱₜ and √(1 − ᾱₜ). The film is computed live from the clean 3 and one fixed noise draw.

**Transition:** Every image ends up in the same place.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a>

## pixel-hist: Every image ends in the same place: N(0, 1)

**Build 1, ask:** Would a dataset of faces end in the same place?

**Expected:** Yes: the end point does not depend on the data.

**Build 2, ask:** Why does it matter that every dataset ends at N(0, 1)?

**Expected:** Because then we know how to start the reverse process: draw from N(0, I).

**Common misconception:** Explainers often credit the Central Limit Theorem. The forward noise is Gaussian by construction and the signal is shrunk away; no averaging of many non-Gaussian pieces is involved.

**Transition:** Follow one pixel through all thousand steps.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> · (endpoint argument corrected)

## fwd-recap: Destroying is easy, and it hands us labels for free

**Build 1, ask:** Which of these will the network see?

**Expected:** xₜ and t. Part 3 makes ε its target.

**Build 2, ask:** What would we need to go back?

**Expected:** The distribution of x_{t−1} given xₜ.

**Transition:** Part 2.

## reverse-q: Going back: where was xₜ₋₁?

**Build 1, ask:** Which term is the likelihood here?

**Expected:** q(xₜ | xₜ₋₁): how the noisy value arises from the previous one.

**Build 2, ask:** Why is q(x_{t−1}) hard?

**Expected:** It averages the noising over every image in the world.

**Build 3, ask:** What do we know exactly about one pixel?

**Expected:** Its distribution: ink with probability π, paper otherwise, so its noisy prior is a mixture of two Gaussians.

**Transition:** What if we jump all the way back at once?

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a>

## big-jump: From pure noise straight back to the pixel: two spikes

**Build 1, ask:** What does x_T tell us about the pixel?

**Expected:** Nothing: at t = 1,000 the signal coefficient is 0.006.

**Build 2, ask:** Why is the average a bad sample?

**Expected:** It is not a value the data ever takes: no image is grey there.

**Build 3, ask:** What did the VAE decoder output, and why was it blurry?

**Expected:** A mean over the plausible images given z: the same averaging.

**Transition:** So take smaller steps back.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> (the “100 cats” question)

## reverse-explorer: Shrink the jump: the step back becomes one Gaussian

**Build 1, ask:** Where are the spikes, and why those heights?

**Expected:** At ±1, with the posterior probabilities of paper and ink given xₜ.

**Build 2, ask:** What makes the two bumps merge?

**Expected:** Each bump is narrow when the jump is short, but the bumps also move together: one step changes x by very little.

**Build 3, ask:** So what should one learned step look like?

**Expected:** A single Gaussian, N(μ, σ²), with a mean that depends on xₜ and t.

**Point here:** Everything here is exact: q(x_s | xₜ) = Σ over ink/paper of P(x₀ | xₜ) · q(x_s | xₜ, x₀), each term a Gaussian (optional depth derives it for one step).

**Transition:** Why a single step back is Gaussian, in general.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a> · (the 1-D picture, here computed exactly)

## guess-noise: If we knew the noise, we could undo the jump exactly

**Build 1, ask:** Why can we solve for x₀?

**Expected:** The shortcut is a straight line in x₀ and ε: given one, the other follows.

**Build 2, ask:** What does the network see?

**Expected:** xₜ and t only.

**Build 3, ask:** Can its guess be exact?

**Expected:** Not always: the same xₜ can come from different (x₀, ε) pairs, so it guesses an average (3.3).

**Transition:** Three names for one guess.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (VI)

## three-targets: Three ways to say the same guess

**Build 1, ask:** Convert: given x̂₀ = 0.26, what is ε̂?

**Expected:** ε̂ = (xₜ − √ᾱ x̂₀)/√(1 − ᾱ) = (0.379 − 0.724 × 0.26)/0.690 = 0.28.

**Build 2, ask:** What is the network’s only job?

**Expected:** Guess the noise in xₜ (equivalently the clean image), at every t.

**Transition:** One step back.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> · <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (4:48)

## reverse-step: One step back, in one line

**Build 1, ask:** Which part needs the network?

**Expected:** Only ε̂.

**Build 2, ask:** Why divide by √αₜ?

**Expected:** The forward step shrank by √αₜ; going back we grow by the same factor.

**Build 3, ask:** Why add noise when we want to remove it?

**Expected:** Each step back is a draw from a narrow Gaussian (2.2), not a single point. 4.5 shows what goes wrong without it.

**Point here:** Where the mean comes from: it is the exact step back given x₀, with x₀ replaced by the guess from ε̂ (derived in optional depth by completing the square). Here σₜ = √βₜ; our MNIST runs use the slightly smaller √β̃ₜ, β̃ₜ = βₜ(1 − ᾱₜ₋₁)/(1 − ᾱₜ). Both work (Ho et al. 2020).

**Transition:** Part 3: how does the network learn to guess the noise?

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> (Algorithm 2) · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (V)

## tuple: A training example costs one line

**Build 1, ask:** Which of these does the network see?

**Expected:** Only xₜ and t.

**Build 2, ask:** Is this supervised learning?

**Expected:** Yes: input (xₜ, t), target ε. We manufactured the labels.

**Build 3, ask:** How many training examples can we make from one image?

**Expected:** As many as we like: every (t, ε) pair is a new example.

**Transition:** Which noise exactly is the target?

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> (Algorithm 1)

## total-noise: ε is all the noise since x₀, not the last step’s

**Build 1, ask:** Why not predict the last step’s noise?

**Expected:** Training never runs the steps: it jumps with the shortcut, so there is no “last step” noise to target.

**Build 2, ask:** Then how do we take one step?

**Expected:** From the total-noise guess we get the mean μ (2.4): Part 4.

**Common misconception:** A common confusion: ε is not the small noise between xₜ₋₁ and xₜ.

**Transition:** The network also needs t.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> (16:15)

## why-t: One network for every noise level: tell it t

**Build 1, ask:** Could one network without t work?

**Expected:** Only by guessing the noise level from the image, which is hard at both ends. Feeding t is free.

**Build 2, ask:** Why not feed t as a single number?

**Expected:** A plain number is a poor input for a network to make many different behaviours from; sinusoids give it a rich, smooth code.

**Build 3, ask:** Where have you seen these features before?

**Expected:** Positional encodings in “Attention is all you need”.

**Transition:** The loss, once by hand.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (7:01) · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 1</a> (22:21)

## loss-hand: The loss: squared error between the noise and its guess

**Build 1, ask:** Why squared error?

**Expected:** Its minimiser is the conditional mean (next slide); and Part 7 shows it comes from the ELBO.

**Build 2, ask:** Why is the best guess wrong for our pixel?

**Expected:** The same xₜ could have come from paper with a different noise. Next slide.

**Build 3, ask:** Why can the network do so much better on the image than we did on one pixel?

**Expected:** It sees all 784 pixels: neighbours reveal whether this one is a stroke (3.6).

**Transition:** Why even the best guess cannot be exact.

## average: The same noisy value can come from different noises

**Build 1, ask:** Why the average, again?

**Expected:** Squared error is minimised by the mean (Lecture 1).

**Build 2, ask:** At which t is the pixel’s unavoidable loss zero?

**Expected:** At both ends: at small t the pixel is obvious; at large t the noise ≈ xₜ itself.

**Build 3, ask:** Why is the network’s loss largest at small t?

**Expected:** Tiny noise on grey stroke edges is hard to tell apart from the stroke itself: real pixels are not just ink or paper.

**Point here:** Irreducible loss = E[Var(ε | xₜ)] = ᾱ/(1 − ᾱ) · E[1 − x̂₀²] for a ±1 pixel. Computed by numerical integration over xₜ.

**Transition:** What network makes the guess?

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> (homework hint: the conditional expectation minimises squared error)

## mlp: For points in 2-D: a small MLP

**Build 1, ask:** What are the inputs and outputs?

**Expected:** Inputs: x, y and 16 features of t. Outputs: the two components of ε̂.

**Transition:** For images, a network that can look at the whole picture.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 1</a> (2-D toy model) · smiley after <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a>

## unet: For images: a U-Net, so every pixel can see the whole digit

**Build 1, ask:** Why shrink the image at all?

**Expected:** To coordinate pixels far apart: whether a pixel is a stroke depends on the whole digit.

**Build 2, ask:** What would be lost without the skip connections?

**Expected:** Exact positions of edges: the decoder would have to reinvent them from 7×7.

**Build 3, ask:** Where does t enter?

**Expected:** In every block, as a learned per-channel shift.

**Point here:** Channels 32 → 64 → 128 → 64 → 32, GroupNorm and SiLU, time features → MLP → a per-block projection added to the activations. The code is in scripts/diffusion.py and the notebook.

**Transition:** The training step.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (7:01)

## train-code: The training step in eight lines

**Build 1, ask:** Which line is the shortcut?

**Expected:** xt = a.sqrt() * x0 + (1 - a).sqrt() * eps.

**Build 2, ask:** What does a batch contain?

**Expected:** Different images, each at its own random noise level.

**Transition:** Watch it learn.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> (Algorithm 1)

## playground: Press play: the network learns the smiley

**Build 1, ask:** What should the arrows point at?

**Expected:** The nearest part of the face: where a clean point near this noisy one most likely was.

**Build 2, ask:** Why does the loss stay well above zero?

**Expected:** 3.3: an unavoidable part remains.

**Point here:** 40,000 Adam steps on fresh smiley points, batch 2048; frames at steps 0, 200, 500, 1k, 2k, 5k, 10k, 20k, 40k.

**Transition:** And on images: what does the trained network believe?

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (“Training Playground”, here with a real network)

## x0film: What the network believes our 3 is, at each noise level

**Build 1, ask:** Why is the guess blurry at high noise?

**Expected:** It is an average over every digit that could hide under this much noise (2.2).

**Build 2, ask:** Why is the network so much better than one pixel at a time?

**Expected:** It sees all 784 pixels together: a stroke here makes a stroke there likely (next slide).

**Transition:** The difference, at one noise level.

## coordinate: One pixel alone gives a grey; all 784 together give a 3

**Build 1, ask:** Which is the per-pixel model of Lecture 2?

**Expected:** Each pixel alone: one probability per pixel, independent.

**Build 2, ask:** Why can the network see the stroke?

**Expected:** Its 7×7 layer sees the whole digit; neighbouring noisy pixels together are much less noisy than one.

**Transition:** Recap of training.

## train-recap: Training: a regression problem we made ourselves

**Build 1, ask:** What is the target, in one word?

**Expected:** The noise.

**Build 2, ask:** What do we still need to generate an image?

**Expected:** A starting point (pure noise) and the reverse step of 2.4.

**Transition:** Part 4.

## step-hand: One step back, by hand, on the real schedule

**Build 1, ask:** Where does ε̂ come from?

**Expected:** From the trained network, given xₜ and t.

**Build 2, ask:** Why does one step move so little?

**Expected:** βₜ is tiny: each real step removes only a little noise.

**Build 3, ask:** What changes in the next step?

**Expected:** t becomes 249: new βₜ, ᾱₜ, and a new guess from the network.

**Build 4, ask:** What will x̂₀ do as t falls?

**Expected:** Sharpen towards ink or paper as the noise disappears.

**Transition:** The same step on a whole image.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> (Algorithm 2)

## step-image: The same step on an image, at t = 600

**Build 1, ask:** Can you see the 7 in x₆₀₀?

**Expected:** No.

**Build 2, ask:** Why do μ and x₅₉₉ look the same as x₆₀₀?

**Expected:** One step changes each pixel by about 1%.

**Build 3, ask:** So where is the progress?

**Expected:** In x̂₀: the guess sharpens step by step (next slides).

**Transition:** The loop.

## toy-reverse: The toy, all the way back: every number

**Build 1, ask:** Check row 5: (0.360 − c₅ × 0.420)/√0.2?

**Expected:** c₅ = 0.8/0.974 = 0.821; (0.360 − 0.345)/0.447 = 0.033.

**Build 2, ask:** Why is the kick at step 4 smaller than at step 5?

**Expected:** √β₄ = 0.707 < √β₅ = 0.894: the toy’s β shrinks going back.

**Build 3, ask:** Why can it hover near 0?

**Expected:** While the noise is large, both paper and ink are plausible.

**Build 4, ask:** Which way will step 1 go?

**Expected:** Towards paper: x₁ is clearly negative.

**Build 5, ask:** We started from an ink run. Why paper?

**Expected:** x₅ is shared by ink and paper runs; going back is sampling, not recall.

**Transition:** Try it yourself.

## calc-reverse: Try it: walk back from noise, one step at a time

**Build 1, ask:** After 20 runs, what fraction ends at ink?

**Expected:** About 35%, the share of ink at this pixel in the data: the sampler reproduces the data’s distribution.

**Point here:** The “network” here is the best possible denoiser for the one-pixel dataset (ink with probability 0.35, else paper), computed exactly with Bayes’ rule; see optional depth.

**Transition:** The same loop, in code, for images.

## sample-code: The sampling loop: start from noise, repeat a thousand times

**Build 1, ask:** Why may we start from N(0, I)?

**Expected:** x_T ≈ N(0, I) for every image (1.5).

**Build 2, ask:** Why is there no noise at t = 1?

**Expected:** We want the final, clean sample, not a noisy draw around it.

**Point here:** σₜ = √βₜ as in 2.4. Our MNIST film and samples use the slightly smaller √β̃ₜ; both work (Ho et al. 2020).

**Transition:** Watch it, in 2-D and on MNIST.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> (Algorithm 2)

## sampler2d: Watch 300 particles find the smiley, and drop your own

**Build 1, ask:** When do the particles start to look like a face?

**Expected:** Late: most of the structure appears in the last few hundred steps.

**Build 2, ask:** Do two particles that start close end close?

**Expected:** Not necessarily: fresh noise at every step can send them to different parts of the face.

**Transition:** The same on MNIST.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a> (5:17) · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 1</a>

## mnist-traj: Eight runs from noise: x_t, and the network’s guess of the end

**Build 1, ask:** Which appears first, the outline or the stroke details?

**Expected:** The outline: high noise drowns fine detail first, so it is decided last.

**Build 2, ask:** Why can the guess change its mind?

**Expected:** Early on, many digits are consistent with xₜ; each step’s fresh noise and new evidence can tip the balance.

**Transition:** Why the fresh noise at every step?

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 6</a> (coarse to fine) · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (“The Generator”)

## why-noise: Take out the fresh noise, and every sample collapses

**Build 1, ask:** Predict: what happens without the noise?

**Expected:** Students guess: blurry? the same digit?

**Build 2, ask:** Why a 1?

**Expected:** A thin 1 is the least ink: the closest digit to the average, which every mean step pulls towards.

**Build 3, ask:** So is the noise a nuisance?

**Expected:** No: it is the sampling. It makes each step a draw from the Gaussian of 2.2, not its centre.

**Common misconception:** Removing the noise is not DDIM: DDIM (Part 6) changes the update so that it stays correct without noise.

**Transition:** The payoff.

## samples64: 64 new digits

**Build 1, ask:** Find a bad one.

**Expected:** Usually a few odd strokes: the model is small and trained for 50 minutes.

**Build 2, ask:** What did it cost?

**Expected:** A thousand network calls per image instead of one.

**Transition:** The whole method on one slide.

## napkin: The whole method on a napkin

**Build 1, ask:** Which line needs the network?

**Expected:** Train step 3 and sample step 2.

**Build 2, ask:** Which line is Part 1?

**Expected:** Train step 2: the shortcut.

**Transition:** Part 5: steering with labels and text.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (“derive it on a napkin”)

## label: A 3 on demand: give the network the label

**Build 1, ask:** Where does the label enter?

**Expected:** Added to the time embedding, so it reaches every block.

**Build 2, ask:** How do we check what the samples are?

**Expected:** With a separate classifier trained on clean MNIST.

**Build 3, ask:** How would you push towards “more 3-like”?

**Expected:** Add a push towards images a classifier is confident are 3s: next slide.

**Transition:** Push harder: guidance.

## cfg: Guidance: ask twice, then exaggerate the difference

**Build 1, ask:** What is the guided guess when w = 1?

**Expected:** The conditional guess ε̂(xₜ, t, c).

**Build 2, ask:** What might go wrong at w = 20?

**Expected:** Over-exaggerated, unnatural samples, all very alike.

**Build 3, ask:** How many networks do we train?

**Expected:** One, with label dropout.

**Point here:** Why the difference of two guesses acts like a classifier: Bayes’ rule (optional depth).

**Transition:** See the two guesses as arrows.

**Source / evidence:** <a href="https://arxiv.org/abs/2207.12598">Ho &amp; Salimans, “Classifier-Free Diffusion Guidance” (2022)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (XII)

## cfgvec: Guidance as arrows: two guesses, and a push past them

**Build 1, ask:** Where is the red arrow at w = 0? At w = 1?

**Expected:** On the black arrow; on the purple arrow.

**Transition:** Turn the dial on whole samples.

## dial: The guidance dial: more w, more 3, less variety

**Build 1, ask:** Which w would you choose?

**Expected:** Somewhere around 2–3 here; image generators use roughly 3–8.

**Build 2, ask:** Is w = 7 sampling from p(x | c)?

**Expected:** No: it samples a sharpened distribution, more typical than real 3s. That is why images look “generated” at high w.

**Point here:** Variety is the mean distance between pairs of the 16 samples: w = 0: 19.4, w = 1: 17.1, w = 2: 16.1, w = 3: 15.8, w = 5: 15.9, w = 7: 15.8. Classifier: a separate CNN, 98% on test digits.

**Transition:** Every digit on demand.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 3</a> (13:20) · <a href="https://arxiv.org/abs/2207.12598">Ho &amp; Salimans, “Classifier-Free Diffusion Guidance” (2022)</a>

## digits-demand: Every digit on demand

**Build 1, ask:** What would change for a text prompt?

**Expected:** The label embedding becomes a text encoder, and conditioning uses cross-attention.

**Build 2, ask:** What stays the same?

**Expected:** The training loop, the sampler and the guidance formula.

**Transition:** Bigger images: latent diffusion.

## latent: Pixels are expensive: diffuse a compressed code

**Build 1, ask:** What does the autoencoder have to get right?

**Expected:** Reconstruction: the code must keep everything visible.

**Build 2, ask:** What does the diffusion model now learn?

**Expected:** The distribution of codes: smaller, and with the pixel-level detail handled by the decoder.

**Build 3, ask:** Why do we need the decoder at the end?

**Expected:** The samples are codes; the decoder turns them into images.

**Point here:** Our autoencoder: convolutions down to 4 × 7 × 7, codes scaled by their standard deviation (5.32) before diffusion. Latent DDPM: 4 time-conditioned blocks, 20,000 updates.

**Transition:** From a label to a sentence.

**Source / evidence:** <a href="https://arxiv.org/abs/2112.10752">Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models” (2022)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (XII)

## text: Text in, through cross-attention

**Build 1, ask:** Where do queries and keys come from?

**Expected:** Queries from image features, keys and values from words.

**Build 2, ask:** Which box is today’s lecture?

**Expected:** The U-Net denoiser and its sampler.

**Build 3, ask:** What does guidance do to a text prompt?

**Expected:** Pushes images to match the caption more strongly, at the cost of variety (5.3).

**Transition:** Part 6: faster sampling.

**Source / evidence:** <a href="https://arxiv.org/abs/2112.10752">Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models” (2022)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (XII)

## slow: A thousand network calls per image

**Build 1, ask:** Where does the time go?

**Expected:** Into the network calls: everything else is arithmetic.

**Build 2, ask:** Why not just use fewer DDPM steps?

**Expected:** The step back is only Gaussian for small steps (2.2); big DDPM steps get the distribution wrong.

**Transition:** Jump to the guess, then re-noise.

## ddim-idea: DDIM: jump to the guess, then re-noise it to an earlier level

**Build 1, ask:** Which formula is the second line?

**Expected:** The shortcut xₜ = √ᾱ x₀ + √(1 − ᾱ) ε with the two guesses.

**Build 2, ask:** Why reuse ε̂ rather than draw new noise?

**Expected:** It keeps the noisy image consistent with the guess; DDIM shows these jumps keep the same marginals as the forward process.

**Build 3, ask:** What do we lose?

**Expected:** The randomness of each step: the same starting noise always gives the same sample.

**Common misconception:** DDIM is not “DDPM with the noise term removed” (4.5 showed that collapses). It is a different update, with a non-Markov forward process that has the same marginals q(xₜ | x₀).

**Transition:** See the paths.

**Source / evidence:** <a href="https://arxiv.org/abs/2010.02502">Song, Meng &amp; Ermon, “Denoising Diffusion Implicit Models” (2021)</a> · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> (24:35)

## ddim-2d: Same network: DDPM paths jitter, DDIM paths glide

**Build 1, ask:** Why are DDIM paths so straight?

**Expected:** Each jump aims at the current guess of the clean point, and the guess changes slowly.

**Build 2, ask:** Do the two samplers give the same final points?

**Expected:** No: different particles end in different places, but both sets follow the data distribution.

**Point here:** Mean distance to the face: DDPM 0.0034, DDIM 0.0083; fresh data points 0.0040.

**Transition:** And on MNIST?

## ddim-steps: Fewer steps, same network: from 1 to 1,000

**Build 1, ask:** Why does one step give blobs, and how few steps can you get away with?

**Expected:** One jump returns x̂₀ at t = 1,000: the average digit. Here about 20 steps suffice; distilled and consistency models get to 1–4 (7.4).

**Transition:** One more design choice: how fast the noise grows.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (XI, “Speed vs Quality”)

## cosine: The schedule matters: linear destroys too early

**Build 1, ask:** At which t does the 3 vanish under each?

**Expected:** Linear about 600, cosine about 800–850.

**Build 2, ask:** Why does that matter for training?

**Expected:** Training samples t uniformly: with linear, about a third of the examples are nearly pure noise.

**Point here:** Our models use the linear schedule (simpler, and enough for MNIST). The cosine schedule is ᾱₜ = cos²(((t/T) + s)/(1 + s) · π/2), s = 0.008.

**Transition:** Part 7: what the network learns.

**Source / evidence:** <a href="https://arxiv.org/abs/2102.09672">Nichol &amp; Dhariwal, “Improved Denoising Diffusion Probabilistic Models” (2021)</a> · <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (6:26)

## score-def: The score: an arrow at every point, pointing uphill

**Build 1, ask:** Where is the score zero?

**Expected:** At the peaks, and at the dip between them.

**Build 2, ask:** Why does the normaliser not matter?

**Expected:** log(p/Z) = log p − log Z, and log Z has no x: its gradient is zero.

**Build 3, ask:** How could you sample using only arrows?

**Expected:** Walk uphill with some noise: Langevin dynamics, next.

**Point here:** Energy view (Vizuara): p ∝ exp(−E). The score is −∇E: downhill in energy, uphill in probability.

**Transition:** The smiley’s arrows, from our network.

**Source / evidence:** <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (IX) · <a href="https://www.youtube.com/watch?v=bVrweqE-r38">Vizuara, “Principles of Diffusion Models”, Lecture 1 (2025)</a> (39:27) · <a href="https://arxiv.org/abs/1907.05600">Song &amp; Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution” (2019)</a>

## scorefield: The smiley’s arrows, at different noise levels

**Build 1, ask:** Why does the arrow length grow far from the face?

**Expected:** The arrow is √ᾱ x̂₀ − x: the further from the face, the further to go.

**Build 2, ask:** Why use many noise levels?

**Expected:** Low noise: precise but only near the data. High noise: defined everywhere but blurry. Start high and lower it: the schedule.

**Transition:** Diffusion is also a VAE.

**Source / evidence:** <a href="https://arxiv.org/abs/1907.05600">Song &amp; Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution” (2019)</a> · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 1</a> (23:00)

## vae-map: Diffusion is a VAE with a fixed encoder

**Build 1, ask:** What plays z?

**Expected:** All of x₁ … x_T.

**Build 2, ask:** What is unusual about this encoder?

**Expected:** It has no parameters: q is chosen, not learned. And the latents are as big as the image.

**Build 3, ask:** What should the training objective be, then?

**Expected:** The ELBO, as in Lectures 1 and 3.

**Transition:** Its ELBO, in one line.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a> (6:28) · <a href="https://arxiv.org/abs/2208.11970">Calvin Luo, “Understanding Diffusion Models: A Unified Perspective” (2022)</a>

## elbo-simple: Its ELBO is a weighted noise error

**Build 1, ask:** What are the latents and the encoder here?

**Expected:** x₁ … x_T; the fixed noising.

**Build 2, ask:** Why drop the weights?

**Expected:** Empirically better samples: the true weights concentrate on the almost-clean steps.

**Transition:** Does it just copy its training set?

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · §3.4

## ideal: The best possible denoiser only copies its training set

**Build 1, ask:** Why is this the best possible guess?

**Expected:** Bayes with N equally likely clean images: the ink-or-paper calculation of optional depth, with N choices instead of two.

**Build 2, ask:** So is a lower training loss always better?

**Expected:** No: the global optimum memorises.

**Build 3, ask:** What stops our network from copying?

**Expected:** Its architecture and training: a smooth function cannot pick out 60,000 exact points (but very small datasets or huge models can memorise).

**Transition:** See it in 2-D.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 4</a> (13:04)

## copy-2d: In 2-D: the perfect denoiser copies, a network fills in

**Build 1, ask:** Is the ideal denoiser wrong?

**Expected:** No: it is optimal for these 40 points. It is a bad generator.

**Build 2, ask:** What did the network assume?

**Expected:** That nearby points behave alike: smoothness. That bias is what makes new samples.

**Transition:** Where the field went.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 4</a> (20:47)

## frontier: Where the field went: same idea, better parts

**Build 1, ask:** Which of these would you try first for faster sampling?

**Expected:** Distillation or consistency models for speed; flow matching for simpler training and straighter paths.

**Transition:** Wrap-up.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 6</a> · Peebles &amp; Xie (2023) · Lipman et al. (2023) · Liu et al. (2023) · Song et al. (2023)

## compare: Three generators, side by side

**Build 1, ask:** Which would you choose for a phone app that must generate in 10 ms?

**Expected:** A GAN or a distilled diffusion model: one or a few network calls.

**Transition:** Exit questions.

## exit: Exit questions

**Build 1, ask:** Answer Q1 before the reveal.

**Expected:** xₜ and t in; ε out.

**Build 2, ask:** Q2.

**Expected:** Different noise levels need different behaviour from shared weights.

**Build 3, ask:** Q3.

**Expected:** ε̂ → x̂₀ → μ → μ + σz.

**Build 4, ask:** Q4.

**Expected:** Without it every sample is the same thin 1.

**Build 5, ask:** Q5.

**Expected:** Bayes with N training images: the guess is their weighted average.

**Transition:** References.

## references: References and credits

**Build 1, ask:** Which video would you watch next?

**Expected:** Ari Seff for the overview in 15 minutes; Outlier for the full derivation; MIT 6.S183 for depth.

**Transition:** End of the main route.

## pixel-prior: One pixel: paper or ink (optional)

**Build 1, ask:** Why is this pixel ink so often?

**Expected:** It sits where many digits have a stroke (the bottom curve of 3, 5, 8, 0…).

**Build 2, ask:** Is a coin a good model of one pixel?

**Expected:** For one pixel, yes, up to the greys. Lecture 2: one probability per pixel gets each pixel right but makes poor images, because strokes are coordinated. Remember that for 3.7.

**Transition:** We observe the noisy pixel. Was it ink?

## pixel-bayes: We see x₂₅₀ = 0.38. Was the pixel ink? (optional)

**Build 1, ask:** Which bump is taller at 0.38?

**Expected:** The ink bump: 0.51 against 0.16.

**Build 2, ask:** Why is the answer below the likelihood ratio’s 0.76?

**Expected:** The prior: paper is 0.65 likely before we look.

**Build 3, ask:** Why the average, and not the more likely value +1?

**Expected:** Squared error is minimised by the mean (Lecture 1: the mean minimises the expected squared distance).

**Build 4, ask:** Where is the sigmoid centred?

**Expected:** Where the two posterior probabilities are equal; the prior shifts it from 0 towards ink.

**Point here:** ᾱ₂₅₀ = 0.5241, so the bumps sit at ±√ᾱ = ±0.724 with variance 1 − ᾱ = 0.476. Log-likelihood ratio: −[(x − s)² − (x + s)²]/(2v) = 2sx/v.

**Transition:** Move t and xₜ yourself.

## inkpaper: Move t and x_t: the posterior and the best guess (optional)

**Build 1, ask:** At t = 50, what is the best guess for xₜ = 0.38? At t = 900?

**Expected:** At t = 50: ink, almost surely (x̂₀ ≈ 1.00). At t = 900: almost the prior, x̂₀ ≈ −0.29.

**Transition:** What if we jump all the way back in one step?

## river: One pixel, a thousand steps: the density flows (optional)

**Build 1, ask:** Where is the density at t = 0?

**Expected:** Two spikes: at −1 with probability 0.65 and at +1 with 0.35.

**Build 2, ask:** Pick one path at t = 1,000. Did it start at ink or paper?

**Expected:** You cannot tell. That is the problem Part 2 faces.

**Transition:** Do we need all thousand steps to make one noisy example?

**Source / evidence:** <a href="https://www.youtube.com/watch?v=EhndHhIvWWw">Deepia, “Diffusion Models: DDPM, Generative AI Animated” (2025)</a>

## why-gaussian: Why: a narrow likelihood times a smooth prior (optional)

**Build 1, ask:** Why does a small β make the likelihood narrow?

**Expected:** β is the variance of one step’s noise.

**Build 2, ask:** What if β were large?

**Expected:** The likelihood would be wide and the product would inherit the prior’s two bumps: 2.4 with a big jump.

**Build 3, ask:** Which part needs learning: the mean or the width?

**Expected:** The mean. DDPM fixes σₜ² to βₜ or to β̃ₜ of the next slide.

**Point here:** The general result is old (Feller 1949): for small steps, the reverse of a Gaussian diffusion has the same Gaussian form. Sohl-Dickstein et al. (2015) built on it.

**Transition:** One case where the step back is exactly Gaussian.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a> · <a href="https://arxiv.org/abs/1503.03585">Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” (2015)</a>

## know-x0: If we knew the clean image, the step back would be exact (optional)

**Build 1, ask:** Why does knowing x₀ help?

**Expected:** It removes the uncertainty about the image: what is left is only the noise of one step.

**Build 2, ask:** When do we know x₀?

**Expected:** During training, never during generation.

**Transition:** Derive it: Bayes with three Gaussians.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (18:23)

## posterior-1: Bayes with three Gaussians (optional)

**Build 1, ask:** Why may we drop x₀ from q(xₜ | x_{t−1}, x₀)?

**Expected:** Each step only looks at the previous value.

**Build 2, ask:** Which of the three is the shortcut?

**Expected:** The second and the third: x_{t−1} and xₜ given x₀.

**Build 3, ask:** Why does a quadratic in the exponent mean Gaussian?

**Expected:** N(x; μ, s²) ∝ exp(−(x − μ)²/(2s²)): a quadratic in x.

**Transition:** Read off the mean and the width.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> · <a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">Lilian Weng, “What are Diffusion Models?” (2021)</a>

## posterior-2: Complete the square: the mean and the width (optional)

**Build 1, ask:** Check: α/β + 1/(1 − ᾱₜ₋₁) = (1 − ᾱₜ)/(β(1 − ᾱₜ₋₁))?

**Expected:** Common denominator: α(1 − ᾱₜ₋₁) + β = α − ᾱₜ + 1 − α = 1 − ᾱₜ. Yes.

**Build 2, ask:** Is μ̃ closer to xₜ or to x₀?

**Expected:** To xₜ: 99% of the weight.

**Build 3, ask:** How far does one step move the pixel?

**Expected:** From 0.379 to a mean of 0.384, plus noise of size 0.071.

**Point here:** Linear term: the x_{t−1} coefficient in the bracket is −2[√αₜ xₜ/βₜ + √ᾱₜ₋₁ x₀/(1 − ᾱₜ₋₁)]; times β̃ₜ gives μ̃. Same algebra as Lecture 1’s Gaussian KL, one completion of the square.

**Transition:** But at generation time we do not have x₀.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · eq. 6–7

## guess-x0: We do not have x₀ when generating, so guess it (optional)

**Build 1, ask:** What should the network output, ideally?

**Expected:** E[x₀ | xₜ]: the posterior mean, as in 2.2.

**Build 2, ask:** Is a small error per step harmless?

**Expected:** It accumulates over a thousand steps; the noise term in each step lets later steps correct it.

**Build 3, ask:** Why might guessing ε be easier than guessing x₀?

**Expected:** ε always has mean 0 and variance 1 at every t: a well-scaled target. Part 5 shows the choice reweights the loss.

**Transition:** Write the mean directly in terms of ε.

## mean-eps: Substitute: the mean in terms of the noise (optional)

**Build 1, ask:** Why is the x₀-weight a written with √ᾱₜ₋₁?

**Expected:** x_{t−1} given x₀ is the shortcut at t − 1.

**Build 2, ask:** Expand a/√ᾱₜ + b: what do you get?

**Expected:** [β + α − ᾱₜ]/((1 − ᾱₜ)√αₜ) = 1/√αₜ.

**Build 3, ask:** What does the step do, in words?

**Expected:** Remove a small multiple of the noise from xₜ, then undo the shrink.

**Transition:** Three names for one guess.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · eq. 11

## elbo-chain: The same bound as Lecture 1 (optional)

**Build 1, ask:** What plays θ in Lecture 1’s version?

**Expected:** The latents x₁ … x_T.

**Build 2, ask:** Which distribution do we average under?

**Expected:** q(x₁:T | x₀): run the noising from the training image.

**Build 3, ask:** Why does each chain factorise?

**Expected:** Each is Markov: every step depends only on the previous state.

**Transition:** Turn the forward steps around.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · Lecture 1 §4.1

## elbo-flip: Turn each forward step around, given x₀ (optional)

**Build 1, ask:** Why compare in the same direction?

**Expected:** The network’s step goes backwards; to compare like with like we need the backward step of q.

**Build 2, ask:** Why may we add x₀ to q(xₜ | xₜ₋₁)?

**Expected:** Markov: x₀ adds nothing once xₜ₋₁ is known.

**Build 3, ask:** What is left after the cancellation?

**Expected:** q(x_T | x₀) / q(x₁ | x₀).

**Transition:** Collect the terms.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a> (16:23) · <a href="https://arxiv.org/abs/2208.11970">Calvin Luo, “Understanding Diffusion Models: A Unified Perspective” (2022)</a>

## elbo-terms: Three kinds of term; the middle ones are the training loss (optional)

**Build 1, ask:** Why is L_T ≈ 0?

**Expected:** q(x_T | x₀) = N(√ᾱ_T x₀, 1 − ᾱ_T) ≈ N(0, 1): ᾱ_T ≈ 0.00004.

**Build 2, ask:** Which terms does training change?

**Expected:** L₀ and the L_{t−1}.

**Build 3, ask:** What are the two Gaussians in L_{t−1}?

**Expected:** N(μ̃ₜ(xₜ, x₀), β̃ₜ) from 2.5, and N(μ_θ(xₜ, t), σₜ²) with σₜ² = β̃ₜ.

**Point here:** The KL form appears by grouping log p_θ(xₜ₋₁ | xₜ) − log q(xₜ₋₁ | xₜ, x₀) under the expectation; the two leftover ends make L_T and L₀.

**Transition:** A KL of two Gaussians: Lecture 1 has the formula.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · eq. 5

## kl-gauss: Lecture 1’s formula: two Gaussians of the same width (optional)

**Build 1, ask:** Which terms of Lecture 1’s formula vanish?

**Expected:** log(σ₂/σ₁) = 0, and σ₁²/(2σ₂²) − ½ = 0.

**Build 2, ask:** So what does minimising each L_{t−1} do?

**Expected:** Moves the network’s mean onto the exact mean μ̃ₜ(xₜ, x₀).

**Build 3, ask:** Why are the two means so close?

**Expected:** Both are 99% xₜ; they differ only in the 1% that depends on x₀ versus its guess.

**Transition:** Both means are written in noise.

**Source / evidence:** <a href="https://www.youtube.com/watch?v=EhndHhIvWWw">Deepia, “Diffusion Models: DDPM, Generative AI Animated” (2025)</a> (17:23) · Lecture 1 §2.7

## kl-mse: Write both means with noise: the KL is a squared noise error (optional)

**Build 1, ask:** Why does xₜ cancel?

**Expected:** It appears identically in both means.

**Build 2, ask:** What depends on θ?

**Expected:** Only ε̂: the weight is fixed by the schedule.

**Build 3, ask:** Check the number against 5.3.

**Expected:** 0.0032 = 0.0032: the same.

**Transition:** And the weights?

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · eq. 12

## lsimple: Drop the weights: the simple loss we trained in Part 3 (optional)

**Build 1, ask:** Which steps does the true ELBO emphasise?

**Expected:** The small-t ones: almost clean images, tiny noise.

**Build 2, ask:** Is L_simple still a bound on log p(x₀)?

**Expected:** Not exactly: it is a reweighted version. It trains the same network to the same optimum E[ε | xₜ] at every t; only the effort per t changes.

**Transition:** A second way to see the same network: as arrows.

**Source / evidence:** <a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a> · §3.4

## tweedie: The noise predictor is a score estimator (optional)

**Build 1, ask:** Where does the minus sign come from?

**Expected:** Gaussian log density: −(x − mean)²/(2 var), whose gradient is −(x − mean)/var.

**Build 2, ask:** Why may the gradient move inside the average?

**Expected:** ∇q_t = ∫ p(x₀) ∇q(x | x₀) dx₀; divide by q_t and Bayes turns p(x₀) q(x | x₀)/q_t(x) into p(x₀ | x).

**Build 3, ask:** So what does the network estimate?

**Expected:** The score of the noisy data distribution at every noise level, scaled by −√(1 − ᾱ).

**Build 4, ask:** Why do the two numbers differ?

**Expected:** One uses the true noise, unknown at generation time; the score averages over both stories, ink and paper.

**Common misconception:** Paras Chopra’s explainer writes ∇ log p(xₜ) = −ε/√(1 − ᾱₜ). That is the score of q(xₜ | x₀), for a known clean image. The score of the noisy data needs the posterior average E[ε | xₜ], which is what the network learns.

**Transition:** Four ways to use the same trained network.

**Source / evidence:** <a href="https://arxiv.org/abs/1907.05600">Song &amp; Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution” (2019)</a> · <a href="https://arxiv.org/abs/2208.11970">Calvin Luo, “Understanding Diffusion Models: A Unified Perspective” (2022)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (IX, corrected)

## langevin: Follow the arrows, with a little noise: Langevin sampling (optional)

**Build 1, ask:** Why the random kick?

**Expected:** Without it every particle climbs to a peak and stops: no spread, just like 4.4.

**Build 2, ask:** How does this compare with Part 4’s loop?

**Expected:** Same structure: a nudge along the network’s guess plus fresh noise, at decreasing noise levels.

**Point here:** Step size here: η = 0.1 (1 − ᾱₜ), one step per level, every fourth level. DDPM’s step is a particular choice of this nudge and kick.

**Transition:** Connect the arrows to the noise predictor.

**Source / evidence:** <a href="https://arxiv.org/abs/1907.05600">Song &amp; Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution” (2019)</a> · <a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a> (IX)

## map-2x2: Four ways to run the same trained network (optional)

**Build 1, ask:** Which of these is the DDPM loop?

**Expected:** Discrete and random.

**Build 2, ask:** Does changing the sampler need retraining?

**Expected:** No: all four use the same ε̂-network.

**Transition:** Part 6.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 2</a> (39:04) · <a href="https://arxiv.org/abs/2011.13456">Song et al., “Score-Based Generative Modeling through SDEs” (2021)</a>

## cg: Classifier guidance: Bayes’ rule on scores (optional)

**Build 1, ask:** Which term is the classifier?

**Expected:** ∇ log p(c | x): which way makes the image more like class c.

**Build 2, ask:** Why does p(c) drop out?

**Expected:** No x in it: its gradient is zero.

**Build 3, ask:** What is hard about this classifier?

**Expected:** It must classify xₜ at all noise levels; ordinary classifiers fail on noise.

**Transition:** Get rid of the classifier.

**Source / evidence:** <a href="https://arxiv.org/abs/2105.05233">Dhariwal &amp; Nichol, “Diffusion Models Beat GANs on Image Synthesis” (2021)</a> · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 3</a> (6:19)

## cfg-bayes: Why guidance works: the classifier cancels out (optional)

**Build 1, ask:** Where did the classifier go?

**Expected:** It is the difference of two denoisers: an implicit classifier.

**Build 2, ask:** What is w = 1?

**Expected:** The plain conditional model, ε̂(xₜ, t, c).

**Build 3, ask:** How many networks do we train?

**Expected:** One, trained with label dropout.

**Build 4, ask:** What does guidance cost at sampling time?

**Expected:** Two network calls per step instead of one.

**Transition:** Turn the dial.

**Source / evidence:** <a href="https://arxiv.org/abs/2207.12598">Ho &amp; Salimans, “Classifier-Free Diffusion Guidance” (2022)</a> · <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 3</a> (17:28)

## opt-sde: The continuous view: a stochastic differential equation (optional)

**Build 1, ask:** Where does the drift −½βx come from?

**Expected:** From √(1 − β) x ≈ (1 − β/2) x for small β: the shrink.

**Build 2, ask:** Why does the reverse need the score?

**Expected:** Anderson (1982): running a diffusion backwards adds a drift along the score of the current density. Part 5’s network supplies it.

**Point here:** Dropping the noise and halving the score term gives the probability-flow ODE, with the same marginals; DDIM is a discretisation of it.

**Source / evidence:** <a href="https://arxiv.org/abs/2011.13456">Song et al., “Score-Based Generative Modeling through SDEs” (2021)</a>

## opt-targets: Other targets: x₀, v and flow matching (optional)

**Build 1, ask:** Do these targets carry different information?

**Expected:** No: given xₜ and t each converts into the others. They weight the loss differently across t.

**Build 2, ask:** Why are flow-matching paths straighter?

**Expected:** The noising path itself is a straight line, so the learned velocity field is close to straight.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 6</a> · Lipman et al. (2023) · Liu et al. (2023)

## opt-sds: A frozen diffusion model as a critic: score distillation (optional)

**Build 1, ask:** What is being trained here?

**Expected:** Only the 3-D parameters ψ; the diffusion model is frozen and acts as a critic.

**Source / evidence:** <a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture 5</a>
