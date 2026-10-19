# Codex build prompt — Generative Modelling I

## Learning to Generate: Applications, VAEs and GANs

**Usage:** Open the intended course repository in Codex and give it this entire file with the instruction: “Implement this lecture completely, including its demonstrations, notes, notebooks and both PDF exports.” If no course checkout is available, the execution contract defines a standalone fallback. Read the complete file before choosing an implementation approach.

This is a self-contained implementation brief for **Nipun Batra's ES 667 Deep Learning course, IIT Gandhinagar**. Every required scene, mathematical detour, numerical fixture, application source, demonstration and delivery requirement is included here. No earlier chat or companion prompt is needed to build this lecture.

| Item | Specification |
|---|---|
| Suggested output folder | `generative-modelling-i/` |
| Main content | 60 logical scenes; 145 planned reveal states |
| Teaching routes | Approximately 120 minutes in full; 90 minutes compressed |
| Classroom experience | Applications first; substantial images; progressively revealed diagrams, equations and code |
| Primary implementation | Interactive static HTML following the existing course runtime and visual conventions |
| Required companion outputs | Instructor notes, completed-scene PDF, all-builds PDF, runnable PyTorch material, recap, asset/source manifest and QA record |

The numbered storyboard controls the teaching sequence; its mathematical specification controls equations and code. The execution contract controls integration, navigation, sources and exports. Improve layout through documented additional builds when necessary. Preserve the mathematical steps, concrete examples and mandatory topics. Keep optional appendix material separately identified and outside the core route timings.

---

## Lecture I implementation body — Learning to generate: applications, latent variables, VAEs and GANs

Build the lecture specified below. This is a complete implementation specification, not a request for another outline. Make routine decisions autonomously. Preserve every required mathematical explanation and every core scene. Use the surrounding prompt's course references and design requirements as the visual system for this specification.

### 1. Learning contract and realistic scope

The audience already knows distributions, MLE/MAP, cross-entropy and MSE, backpropagation, next-token name generation, causal transformers, encoder models, ViT, CLIP and the main VLM conditioning architectures. Reuse those concepts. Explain unfamiliar probability carefully even though the audience can follow algebra.

The lecture must start with **ten visual applications before a formal definition or a model taxonomy**. It must then explain how to learn and sample a distribution; establish latent-variable modelling; teach KL divergence through an exact worked example; derive the ELBO; explain reparameterization as a gradient estimator; build a VAE with fully specified shapes; and teach GANs as a substantial second method with complete alternating training.

At the end, a student should be able to:

1. Distinguish an observed sample, a distribution, a likelihood, a prior and a posterior.
2. Explain why the earlier name generator is already a generative model.
3. Compute a small discrete KL divergence and interpret its direction.
4. Derive the ELBO and identify the posterior approximation gap.
5. Show where gradients flow through a reparameterized Gaussian sample.
6. Implement one VAE training step and generate an image without the encoder.
7. Implement one GAN discriminator update and one generator update with correct gradient isolation.
8. Compare sample quality and distribution coverage without relying on a single loss curve.

#### Route budget

There are **60 logical scenes and 145 reveal states**, counting the initial state of a scene as its first build. The full teaching route is **120 minutes**. A compressed route is **90 minutes**; achieve it through shorter gallery narration, precomputed demo playback and tightly bounded discussion. Do not remove the application gallery, KL calculation, ELBO derivation, reparameterization, or GAN training from the compressed route. Do not market either route as an 80-minute slow walkthrough. The separate lab is additional time.

| Section | Scenes | Builds | Full route | Compressed route |
|---|---:|---:|---:|---:|
| A. What can we generate? | 01–10 | 20 | 12 min | 8 min |
| B. From samples to distributions | 11–18 | 20 | 14 min | 10 min |
| C. KL divergence detour | 19–23 | 13 | 10 min | 8 min |
| D. Latent variables and the ELBO | 24–31 | 20 | 18 min | 14 min |
| E. Reparameterization detour | 32–36 | 12 | 10 min | 8 min |
| F. Build and use a VAE | 37–44 | 21 | 18 min | 13 min |
| G. Generative adversarial networks | 45–58 | 35 | 32 min | 25 min |
| H. Comparison and exit questions | 59–60 | 4 | 6 min | 4 min |
| **Total** | **60** | **145** | **120 min** | **90 min** |

Use a persistent, unobtrusive section map and visible distinction between logical scene number and reveal number. Put discussion prompts in presenter notes first, with an optional on-screen reveal. Do not crowd a scene with all notes at once.

### 2. Nonnegotiable pedagogical and interaction requirements

- One question should drive each scene. Introduce an equation only after its quantities appear in a diagram, a table or a concrete example.
- Preserve a stable color mapping for observed data, random noise/latent variables, learned generative components, inference components, and objectives. Use the surrounding course style; the semantic roles matter more than any particular color.
- Distinguish a distribution node from one sampled value. For example, a Gaussian curve is `q(z|x)`; a dot is one `z`; an encoder box outputs distribution parameters.
- Distinguish model parameters `theta, phi, psi`, a per-example latent `z`, and an auxiliary noise draw `epsilon`. Use a small glossary drawer and hover/focus definitions.
- Every probability example must use a shared computation module, not duplicated hard-coded labels. Every displayed quantity must update when a control changes.
- Use the same fixed sample seeds when comparing settings. A `Resample` control is separate from a `Change parameter` control; otherwise the effect of changing a parameter becomes ambiguous.
- Every key architecture needs a training and a generation view. Freeze or remove unused modules visibly. Do not use the word “inference” ambiguously: label either `posterior inference / encoding` or `generation / sampling`.
- Required interactive panels: Gaussian/mixture sampler, discrete KL calculator, exact finite-latent ELBO-gap calculator, Gaussian reparameterization gradient panel, VAE sample/reconstruction explorer, and GAN alternating-update explorer.
- Interaction should expose mathematics, not hide it behind a decorative animation. Provide explicit numerical readouts, pause/step/reset controls, and keyboard operation.
- A live neural-network optimization demo is optional if it compromises lecture reliability. A deterministic playback of actual saved training checkpoints is acceptable and must be labelled `recorded training run`. A hand-authored illustration must be labelled `illustration`; never present it as measured learning.
- Keep the core route self-contained after initial load. Assets, fonts, model checkpoints and dependencies must follow the surrounding prompt's deployment requirements. Do not rely on a third-party video, paid API or account login during class.
- Every external research example needs a compact source link in notes and an appropriate caption. Hypothetical lab applications must be marked as proposals. Do not present generated medical or satellite examples as actual measurements.

### 3. The 60-scene storyboard

The `Builds` list below is the implementation order. Each numbered build is a distinct reveal state. Treat the specified counts as the baseline and preserve the ordered mathematical steps. Split crowded material into additional meaningful builds or uniquely identified continuation scenes when this improves classroom readability. Record and explain revised counts in the manifest; never silently combine the KL/ELBO/reparameterization/GAN derivation steps or pad the deck to hit a number.

#### A. What can we generate? — 12 minutes

##### Scene 01 — Where did these faces come from?

- **Goal:** Open with compelling published GAN samples and a question, before introducing technical names.
- **Visual:** A legible selection from the original StyleGAN face-synthesis teaser. The lecture title is a small overlay, not a preceding text-only slide. Preserve source-image aspect ratios and use different faces to make variation visible.
- **Builds (2):** 1. Show the portraits and ask where they came from. 2. Reveal that these are published generated samples; identify a random seed as the input and highlight variation across samples.
- **Math:** Only the informal relation `random input → learned generator → image`; no formal probability definition yet.
- **Discuss:** “What would a model need to learn to produce many plausible, varied faces?” Do not infer or identify real people from the images.
- **Reference and asset:** Official repository https://github.com/NVlabs/stylegan ; image https://raw.githubusercontent.com/NVlabs/stylegan/master/stylegan-teaser.png . Follow the original repository's image-material terms and attribution. These are generated samples, not input photographs passed through an editing model.

##### Scene 02 — Generate handwriting and move through a latent grid

- **Goal:** Put an accessible VAE example beside the initial GAN example and promise a mechanism the lecture will explain.
- **Visual:** First show a small, readable grid of published VAE MNIST samples; then show the published decoded two-dimensional latent grid with its source caption.
- **Builds (2):** 1. Reveal generated digit samples and ask how new samples could be drawn after training. 2. Reveal the latent grid and ask how moving through a small hidden space could change the output.
- **Math:** No VAE equation yet. Clearly label `generated samples` versus `decoded grid`. The grid is a visualization rather than random draws from the prior.
- **Discuss:** “What could a point in a two-dimensional space have to do with a 28×28 image?” Do not claim every generated image has been proved distinct from every training image.
- **Reference and assets:** Kingma–Welling https://arxiv.org/pdf/1312.6114 , Figure 5 (samples for several latent dimensions) and Figure 4(b) (MNIST manifold), PDF page 10, zero-based page 9. The published manifold axes use Gaussian inverse-CDF coordinates; do not silently describe them as an equally spaced grid in raw z. A live linear interpolation must be a separate, honestly labelled trained demo.

##### Scene 03 — Mix styles while keeping some properties

- **Goal:** Show controlled variation using a concrete GAN result without making an interpolation claim.
- **Visual:** A carefully cropped, legible example from original StyleGAN Figure 3: two source generated faces and one or more style-mixed results. Preserve row/column correspondence.
- **Builds (2):** 1. Show the source samples and ask which properties might be combined. 2. Reveal the published mixed result and label which source supplies styles at the selected layer groups.
- **Math:** Label `style mixing`, explicitly distinct from `linear latent interpolation`. Do not introduce StyleGAN's full mapping network or claim that semantic properties can always be independently controlled.
- **Discuss:** “Can one model change some visual properties while keeping others recognisable?”
- **Reference and asset:** https://github.com/NVlabs/stylegan ; author-linked Figure 3 download https://drive.google.com/uc?id=1U-nlMDtpnf1RcYkaFQtbh5oxnhA97hy6 ; reproduction code https://github.com/NVlabs/stylegan/blob/master/generate_figures.py . If the figure download is unavailable, use the verified figure in the repository-linked paper.

##### Scene 04 — A shoe outline becomes a photograph

- **Goal:** Introduce conditional generation with a clear observed input and generated output.
- **Visual:** One complete pix2pix shoe example, preserving the author's `Input`, `Output` and `Ground Truth` labels. Keep the model output visually distinct from the reference photo.
- **Builds (2):** 1. Show the outline and ask which appearance details it does not specify. 2. Reveal the generated output and optionally the ground-truth reference in a separately labelled column.
- **Math:** The outline is a condition; the image is an output. Do not claim that a deterministic published input/output pair demonstrates diversity across random seeds.
- **Discuss:** “Which parts of the result come from the observed edges, and which details must the model supply?”
- **Reference and assets:** https://phillipi.github.io/pix2pix/ ; shoe gallery https://phillipi.github.io/pix2pix/images/sketch2photo_shoes/latest_net_G_val/index.html . Select and download the input/output/reference images from the same complete row, such as the row based on `100_AB.jpg`; never relabel its ground-truth photo as a generated result.

##### Scene 05 — Turn a semantic layout into a landscape

- **Goal:** Connect generation directly to the segmentation labels students already know.
- **Visual:** A clear semantic layout and its corresponding published SPADE/GauGAN output. Retain the layout's class legend. Use the pix2pix facade comparison as an optional later replay after the GAN loss is explained.
- **Builds (2):** 1. Reveal the layout and named regions. 2. Reveal the generated image and highlight the difference between spatial constraints and appearance details.
- **Math:** A segmentation/layout map is a condition. Do not infer a same-layout diversity experiment from a figure that only shows one output; show multiple outputs only when their provenance confirms the shared condition.
- **Discuss:** “Could more than one plausible photograph agree with this layout?”
- **Reference and assets:** https://nvlabs.github.io/SPADE/ ; https://nvlabs.github.io/SPADE/images/teaser_high_res_uncompressed.png ; optional authored animations https://nvlabs.github.io/SPADE/images/treepond.gif and https://nvlabs.github.io/SPADE/images/ocean.gif . For a later L1 versus conditional-adversarial comparison, use https://phillipi.github.io/pix2pix/images/index_facades2_loss_variations.html and preserve the original objective labels.

##### Scene 06 — Translate a horse photograph into the zebra domain

- **Goal:** Motivate image translation with unpaired training data, without adding a full new architecture lesson.
- **Visual:** A clearly matched CycleGAN input/output pair. The original horse photo remains labelled as an input; the zebra-domain result remains labelled as generated. Summer/winter translation can be a small optional comparison drawer.
- **Builds (2):** 1. Show the horse image and desired target domain. 2. Reveal the published translation and the question of learning without paired horse/zebra photographs.
- **Math:** No cycle-consistency derivation in the core gallery. Label the task as `unpaired domain translation`; do not imply that the vanilla GAN introduced later alone implements every detail of CycleGAN.
- **Discuss:** “What should remain consistent when appearance moves from one domain to another?”
- **Reference and assets:** https://junyanz.github.io/CycleGAN/ ; https://junyanz.github.io/CycleGAN/images/objects.jpg ; alternate horse gallery https://taesung.me/cyclegan/2017/03/25/horse-to-zebra-supplemental-best.html . Optional seasonal example: https://junyanz.github.io/CycleGAN/images/season.jpg . A seasonal translation is not a verified prediction of an actual future observation.

##### Scene 07 — Complete a missing image region

- **Goal:** Show semantic inpainting as another image-generation application and make uncertainty concrete.
- **Visual:** One Context Encoders held-out example with its missing region clearly outlined, the generated completion and any reference image kept under separate labels. Align the unchanged context at identical coordinates.
- **Builds (2):** 1. Show the masked observation and ask what the surrounding pixels constrain. 2. Reveal the published model completion and distinguish plausible filling from uniquely recovered hidden detail.
- **Math:** The visible pixels and mask form a condition. The word `encoder` in this model's name does not by itself mean a variational posterior. Do not infer a multiple-sample experiment from a single completion.
- **Discuss:** “Which aspects of a missing region are strongly constrained by context, and which can remain ambiguous?”
- **Reference and assets:** https://www.cs.cmu.edu/~dpathak/context_encoder/ ; model-result figure https://www.cs.cmu.edu/~dpathak/context_encoder/resources/result_fig.jpg ; full result gallery https://www.cs.cmu.edu/~dpathak/context_encoder/1M_imagenet_full.html . Do **not** use `resources/teaser.jpg` as a model output: that image shows human-artwork inpainting.

##### Scene 08 — A lab idea: synthesize a sleep-signal segment

- **Goal:** Connect generative modelling to the lab's sensing work without fabricating a result.
- **Visual:** Three clearly labelled synthetic EEG-like traces conditioned on an illustrative event such as a spindle. Use procedurally drawn signals with a fixed scale and an event window; add the caption “Proposed lab application; illustrative signals.”
- **Builds (2):** 1. Show the condition and one trace. 2. Reveal variation in amplitude/timing while preserving the described event.
- **Math:** A signal is a multichannel time series, e.g. `channels × samples`; the example event is a condition.
- **Discuss:** “Which statistics and event labels must be preserved before synthetic data could help?”
- **Reference:** This is a proposed lab application, not a claim about a published model. Do not assert that synthetic data guarantees privacy or improves a classifier.

##### Scene 09 — Generate satellite-image possibilities

- **Goal:** Connect conditioning and spatial structure to geospatial research.
- **Visual:** Attributed satellite-generation examples or clearly labelled illustrative tiles, with a metadata card and multiple possible outputs. Separate observed input imagery from generated candidates.
- **Builds (2):** 1. Show a metadata-conditioned sample. 2. Reveal alternative possibilities and mark structures that vary.
- **Math:** An optional input tuple `(location/context, acquisition metadata)`; do not promise supported conditions beyond the source.
- **Discuss:** “How would we assess realism while checking whether rare structures are covered?”
- **Reference:** DiffusionSat, https://arxiv.org/abs/2312.03606 . This is not an actual forecast or a recovered measurement.

##### Scene 10 — You have already generated something

- **Goal:** End the gallery by reconnecting to the students' own next-token name generator.
- **Visual:** Reuse the course's name-generation interface or recreate its recognisable token-by-token build. Show two generated names from the same start token with different seeds; a small code-completion card may broaden the application.
- **Builds (2):** 1. Animate next-token sampling for one name. 2. Reveal another sample and highlight that the training recipe is already familiar.
- **Math:** Show a categorical next-token distribution for one step, but save the product factorization for Scene 15.
- **Discuss:** “What connects generated names, images and signal segments?”
- **Reference:** Prefer the existing course lecture and code. Do not add an unrelated model survey.

Optional additional application material belongs in a references/gallery drawer outside the 60-scene route, not as extra required opening scenes: text-to-audio (AudioLDM, https://arxiv.org/abs/2301.12503), video (https://arxiv.org/abs/2204.03458), text-to-3D (DreamFusion, https://arxiv.org/abs/2209.14988), and structured molecules (https://arxiv.org/abs/2203.17003). Use verified published examples with correct captions, or explicitly labelled illustrations. These can provide further breadth on demand; Lecture II has its own modern diffusion-application gallery.

#### B. From samples to distributions — 14 minutes

##### Scene 11 — What did the gallery have in common?

- **Goal:** Define the common task only after its applications are concrete.
- **Visual:** Collapse the ten examples into three slots: condition `c`, source of variation, output `x`; keep a few examples visible as small anchors.
- **Builds (3):** 1. Label observed outputs as samples. 2. Introduce a model distribution `p_theta(x)`. 3. Add an optional condition to obtain `p_theta(x | c)`.
- **Math:** `x ~ p_theta(x)` and `x ~ p_theta(x | c)`. Explain that the notation denotes sampling, not setting `x` equal to a probability.
- **Discuss:** “Is image generation unconditional, conditional, or either?”

##### Scene 12 — A good average can be a bad sample

- **Goal:** Separate representing a distribution from predicting one central value.
- **Visual:** A two-cluster point cloud, its mean in the gap, and generated points. Offer a switch between a faithful mixture, a collapsed cluster and a central point.
- **Builds (2):** 1. Show the mean as a summary statistic. 2. Compare realism and coverage of the three generated sets.
- **Math:** `mean = E[X]` does not imply `X = E[X]`. For a bimodal distribution, the mean may have low density.
- **Discuss:** “Would low distance to the mean prove that samples are good?”

##### Scene 13 — The simplest generative model: fit a Gaussian, then sample

- **Goal:** Make likelihood fitting and sampling tangible using familiar probability.
- **Visual:** One-dimensional observations at `[-2.2, -2.0, -1.8, 1.8, 2.0, 2.2]`; a fitted Gaussian curve; a sample tray. Parameter and resampling controls are independent.
- **Builds (3):** 1. Show observations and fit the mean. 2. Fit the MLE variance and overlay density. 3. Draw fresh samples and reveal the poor fit near the gap.
- **Math:** `mu_hat = mean(x_i) = 0`; `sigma_hat^2 = (1/N) sum_i (x_i-mu_hat)^2 = 4.0266667`. Label this as the Gaussian MLE variance, not the unbiased `1/(N-1)` estimator. Generate `x = mu_hat + sigma_hat * epsilon` with `epsilon ~ N(0,1)`.
- **Discuss:** “Is the model trained? Does that mean it is expressive enough?”

##### Scene 14 — A hidden choice gives a mixture

- **Goal:** Introduce a latent variable first as an understandable sampling procedure.
- **Visual:** A fair two-way selector chooses a left or right Gaussian; then a point appears. Display selector outcomes over repeated samples.
- **Builds (2):** 1. Sample `z` from two categories. 2. Sample `x` from the selected Gaussian and combine both into a mixture curve.
- **Math:** `P(z=0)=P(z=1)=1/2`; `p(x)=sum_z p(z)p(x|z)`. Illustrative component means are `-2,+2` and standard deviation `0.25`; do not imply these are the exact MLE fit to Scene 13.
- **Discuss:** “If we observe only x, do we know which component produced it?”

##### Scene 15 — Our name generator learned a joint distribution

- **Goal:** Reuse autoregression as the first neural generative model.
- **Visual:** The previous course's token sequence and progressively extended context; place one conditional distribution beneath each generated token.
- **Builds (2):** 1. Display conditional next-token probabilities. 2. Multiply them to obtain the probability of a complete sequence, including the end token.
- **Math:** `p_theta(x_1,...,x_T)=prod_t p_theta(x_t|x_<t)`; `-log p_theta(x)=sum_t -log p_theta(x_t|x_<t)`.
- **Discuss:** “Where did maximum likelihood appear in the code you already wrote?”

##### Scene 16 — The same idea could generate pixels

- **Goal:** Establish an alternative to latent generation without reteaching a transformer.
- **Visual:** A small `4×4` binary image revealed in raster order; side-by-side training and generation views.
- **Builds (3):** 1. Linearize pixels into a sequence. 2. During training, show known preceding pixels and next-pixel likelihoods. 3. During generation, sample a pixel and feed it into the next step.
- **Math:** `p_theta(x)=prod_i p_theta(x_i|x_<i)`; shape labels identify one token/value versus the image.
- **Discuss:** “Which part is parallelizable under teacher forcing, and which dependency remains during ancestral generation?”

##### Scene 17 — Could one random vector generate an entire image?

- **Goal:** Motivate a latent generator as a modelling choice, not an impossibility result about autoregression.
- **Visual:** A two-dimensional noise plane feeds a learned network and a digit canvas. Selecting different points changes output.
- **Builds (2):** 1. Show `z` and a learned mapping. 2. Reveal multiple generated outputs for multiple `z` values.
- **Math:** `z ~ p(z)`; for an implicit generator, `x = G_theta(z)`. A VAE's probabilistic decoder will be introduced separately.
- **Discuss:** “What training signal tells this network which images to make?”

##### Scene 18 — An autoencoder solves reconstruction; what about sampling?

- **Goal:** Explain the missing ingredient that motivates a probabilistic latent model.
- **Visual:** `x → encoder → z → decoder → reconstruction`; then plot encoded training points and sample from a standard Gaussian in the same latent plane.
- **Builds (3):** 1. Reveal the reconstruction pathway. 2. Show a possible irregular arrangement of learned codes. 3. Ask what distribution should be used for new latent samples.
- **Math:** A reconstruction objective alone does not specify a convenient prior or ensure that its samples land in well-trained regions. Do not claim that autoencoders can never be used generatively; modelling the learned latent distribution is another possible method.
- **Discuss:** “What is missing if we only optimize reconstruction?”

#### C. KL divergence detour — 10 minutes

##### Scene 19 — Compare two probability distributions

- **Goal:** Introduce KL through two probability bars and a question about assigning probability to observations.
- **Visual:** Two-category bars for `P=(0.8,0.2)` and `Q=(0.6,0.4)`; sample labels drawn from `P`.
- **Builds (3):** 1. Reveal `P` and `Q`. 2. Introduce surprisal `-log probability` for an observed category. 3. Ask about the average extra surprisal from using `Q` for data distributed as `P`.
- **Math:** Natural logarithms throughout; units are nats. `D_KL(P||Q)=sum_i P_i log(P_i/Q_i)`.
- **Discuss:** “Which distribution weights the average?” Expected: the first argument, `P`.

##### Scene 20 — Compute KL completely, with real numbers

- **Goal:** Make KL computable rather than a vague ‘distance’ label.
- **Visual:** A synchronized table with columns `i`, `P_i`, `Q_i`, `log(P_i/Q_i)`, `P_i log(P_i/Q_i)`; the corresponding bar is highlighted for each row.
- **Builds (3):** 1. Compute the first contribution, `0.8 log(0.8/0.6)=0.230145658`. 2. Compute the second contribution, `0.2 log(0.2/0.4)=-0.138629436`. 3. Sum to `D_KL(P||Q)=0.091516222 nats`.
- **Math:** Show the full substitution, not only the result. A negative individual contribution is valid; the complete divergence is nonnegative.
- **Discuss:** “How can one term be negative when KL cannot be negative?”

##### Scene 21 — Direction and support matter

- **Goal:** Prevent the two most common KL misconceptions.
- **Visual:** Swap the roles of the two bars without recoloring the definitions; then let `Q_2` approach zero using a slider.
- **Builds (3):** 1. Compute the reverse divergence `D_KL(Q||P)=0.104649629`. 2. Show `KL(P||P)=0`. 3. Set `Q_i=0` where `P_i>0` and display infinity with a short explanation.
- **Math:** KL is not symmetric and is not a metric. Use `0 log(0/q)=0`; a positive-probability event assigned zero probability contributes `+infinity`. Handle these cases explicitly in code rather than showing `NaN`.
- **Discuss:** “Why can underassigning probability to an event that actually occurs be costly?”

##### Scene 22 — Your cross-entropy loss already contains a KL

- **Goal:** Connect the detour to familiar supervised learning.
- **Visual:** A decomposition bar with entropy, extra mismatch, and cross-entropy; values update from the same `P,Q` controls.
- **Builds (2):** 1. Compute `H(P)=0.500402424` and `H(P,Q)=0.591918645`. 2. Show `H(P,Q)=H(P)+D_KL(P||Q)`.
- **Math:** When `P` is fixed, minimizing cross-entropy over `Q` is equivalent to minimizing `D_KL(P||Q)`. Explain the expectation over labels; do not conflate the empirical one-hot label for a single example with an unknown population distribution.
- **Discuss:** “Which term changes when model predictions change?”

##### Scene 23 — From category bars to latent densities

- **Goal:** Prepare the continuous KL used in a VAE, with arguments kept explicit.
- **Visual:** A standard Gaussian prior curve and one example-specific Gaussian curve. Their overlap is visually illustrative; numerical KL comes from the correct formula later.
- **Builds (2):** 1. Replace the discrete sum by an integral. 2. Label the future comparison as `q_phi(z|x)` versus `p(z)`.
- **Math:** `D_KL(q||p)=integral q(z) log(q(z)/p(z)) dz`. A density value is not a probability mass; the ratio inside the logarithm is the relevant quantity.
- **Discuss:** “Why does this comparison change when the observed x changes?”

#### D. Latent variables and the ELBO — 18 minutes

##### Scene 24 — Specify a generative story before its training algorithm

- **Goal:** Define the VAE's generative model cleanly.
- **Visual:** A prior distribution produces `z`; a learned decoder produces likelihood parameters; an observation is sampled from that likelihood. Keep the encoder absent initially.
- **Builds (2):** 1. Sample `z ~ N(0,I)`. 2. Sample `x ~ p_theta(x|z)` and show the joint model.
- **Math:** `p_theta(x,z)=p(z)p_theta(x|z)`. The prior here is fixed and has no learned parameters. The decoder parameterizes a distribution; its mean/probabilities are not automatically a sample from that distribution.
- **Discuss:** “What randomness remains after we have fixed z?”

##### Scene 25 — The quantity MLE asks us to compute

- **Goal:** Explain ‘evidence’ and the computational difficulty that motivates a bound.
- **Visual:** Many latent values each contribute to the likelihood of the same observed digit; progressively overlay their weighted contributions.
- **Builds (2):** 1. Write the marginal likelihood as an integral. 2. Show why evaluating a nonlinear decoder over all continuous latent possibilities is generally intractable.
- **Math:** `p_theta(x)=integral p(z)p_theta(x|z) dz`; MLE maximizes `sum_n log p_theta(x_n)`. “Evidence” here means this marginal probability/density of the observation.
- **Discuss:** “Why is one random latent draw not the exact marginal likelihood?”

##### Scene 26 — Introduce an approximate posterior

- **Goal:** Separate the model's true posterior from the trainable inference network.
- **Visual:** A solid forward generative arrow `z → x`; a dashed inference arrow `x → q_phi(z|x)`; a ghost label for the true posterior `p_theta(z|x)`.
- **Builds (2):** 1. Use Bayes' rule to identify the desired posterior. 2. Introduce `q_phi(z|x)` as a tractable normalized approximation parameterized by an encoder.
- **Math:** `p_theta(z|x)=p_theta(x,z)/p_theta(x)`. The intractable evidence appears in the denominator. The approximate posterior is not the prior and is not the exact posterior by definition.
- **Discuss:** “Which network parameters control q, and which control the generative model?”

##### Scene 27 — Derive the ELBO, part 1: move the logarithm

- **Goal:** Give the first four steps of the actual derivation, with no skipped expectation identity.
- **Visual:** An equation canvas with one active line and up to two earlier lines dimmed above it; a small concave `log` plot makes Jensen's direction visible. Each introduced `q` term is highlighted with the same color.
- **Builds (4):** 1. Start with `log p_theta(x)=log integral p_theta(x,z) dz`. 2. Multiply and divide by `q_phi(z|x)` inside the integral. 3. Recognize `log E_q[p_theta(x,z)/q_phi(z|x)]`. 4. Apply concavity: `log E_q[R] >= E_q[log R]`.
- **Math:** Define `R=p_theta(x,z)/q_phi(z|x)`. State in notes that the normalized `q` must have the support needed for the displayed expectation identity; the Gaussian choices used here have positive density everywhere. Do not reverse Jensen's inequality.
- **Discuss:** “Where did an equality become an inequality?”

##### Scene 28 — Derive the ELBO, part 2: recognize its two terms

- **Goal:** Finish the seven-build derivation and connect it to an optimizable objective.
- **Visual:** Continue the same equation style; group reconstruction likelihood and prior/posterior terms with braces. Do not show the final answer before the factorization is revealed.
- **Builds (3):** 1. Substitute `p_theta(x,z)=p_theta(x|z)p(z)` and split the logarithm. 2. Recognize `E_q[log p_theta(x|z)] - D_KL(q_phi(z|x)||p(z))`. 3. Name this quantity `ELBO(theta,phi;x)` and show the loss as its negative.
- **Math:** `log p_theta(x) >= ELBO = E_q log p_theta(x|z) - KL(q||p)`. Maximizing the ELBO is equivalent to minimizing expected reconstruction NLL plus KL. The vanilla objective has coefficient one on KL.
- **Discuss:** “Which part rewards explaining the observation, and which compares distributions?”

##### Scene 29 — What is the gap between the bound and the target?

- **Goal:** Explain what the lower bound actually means rather than treating it as a new arbitrary loss.
- **Visual:** A vertical number line with `ELBO` below `log p_theta(x)` and a bracket labelled by posterior KL; the likelihood can itself change as `theta` changes.
- **Builds (3):** 1. Expand `KL(q_phi(z|x)||p_theta(z|x))`. 2. Rearrange to `log p_theta(x)=ELBO+KL(q_phi(z|x)||p_theta(z|x))`. 3. Show equality when `q` equals the true posterior, within the appropriate almost-everywhere meaning.
- **Math:** Contrast the two KLs explicitly: the objective uses `KL(q||prior)`; the bound gap uses `KL(q||true posterior)`. The latter is generally not directly computable in the full nonlinear model.
- **Discuss:** “Does pushing q toward the prior alone make the posterior approximation exact?” Expected: no.

##### Scene 30 — Verify the ELBO with a tiny exact model

- **Goal:** Turn the bound into numbers the students can independently verify.
- **Visual:** A two-state latent tree and a synchronized calculator. For the observed `x=1`, set `p(z)=(0.5,0.5)`, `p(x=1|z)=(0.8,0.2)`, and `q(z|x)=(0.6,0.4)`. An adjustable `q_0` slider changes the bound but not the fixed model evidence.
- **Builds (2):** 1. Compute `p(x=1)=0.5` and true posterior `(0.8,0.2)`. 2. Reveal the exact decomposition below and let q move toward the posterior.
- **Math:** Rounded to six decimals: `E_q log p(x|z)=-0.777661`; `KL(q||prior)=0.020136`; `ELBO=-0.797797`; posterior-gap KL `=0.104650`; `log p(x)=-0.693147`. Verify `ELBO + posterior_gap = log_evidence` using full-precision computed quantities; the displayed sum is approximately `-0.797797+0.104650=-0.693147`.
- **Discuss:** “Why can we compute the exact gap here but usually not for a neural decoder?”

##### Scene 31 — Check the meaning before introducing more machinery

- **Goal:** Consolidate prior, posterior, bound and objective.
- **Visual:** Four draggable or selectable labels: prior, true posterior, approximate posterior and likelihood; reveal the correct assignments in the existing diagram. Keep this short, not a new visual motif.
- **Builds (2):** 1. Ask students to assign each distribution and say what is conditioned on. 2. Reveal the assignments and the question that remains: differentiating through a sample from q.
- **Math:** The exact ELBO is an expectation. A one-sample Monte Carlo objective estimates it; a single noisy estimate need not itself lie below the exact log evidence on every draw.
- **Discuss:** “When someone says ‘the VAE loss is reconstruction plus KL,’ what derivation and averaging are hidden in that sentence?”

#### E. Reparameterization detour — 10 minutes

##### Scene 32 — A parameter-dependent sampling operation hides a gradient path

- **Goal:** Explain the specific problem without the false claim that random variables cannot be differentiated through by any method.
- **Visual:** Encoder parameters produce `mu,sigma`; a black-box sampler yields `z`; a decoder produces a loss. Highlight the dependency on the distribution parameters.
- **Builds (2):** 1. Show the naive sampling box and the desired gradient. 2. Explain that an opaque sampling API does not expose the pathwise derivative we need; alternate estimators exist but are not this detour's topic.
- **Math:** We want `grad_phi E_{z~q_phi(z|x)}[f(z)]`, not merely a derivative with respect to a sampled constant `z`.
- **Discuss:** “If the mean moves slightly, how should comparable samples move?”

##### Scene 33 — Move the randomness into a fixed distribution

- **Goal:** Derive Gaussian reparameterization visually.
- **Visual:** Fixed standard-normal samples are stretched by sigma and shifted by mu; arrows connect the same sampled dots across panels.
- **Builds (3):** 1. Draw `epsilon ~ N(0,1)`. 2. Scale by positive sigma. 3. Shift by mu and label `z=mu+sigma epsilon ~ N(mu,sigma^2)`.
- **Math:** For vectors, `z=mu+sigma odot epsilon`, `epsilon~N(0,I)`. The auxiliary distribution no longer depends on phi.
- **Discuss:** “Did the distribution of z change when we rewrote how it is sampled?” Expected: no.

##### Scene 34 — Follow the pathwise gradient

- **Goal:** Make the chain rule explicit, including the role of a fixed noise draw.
- **Visual:** An animated computation graph with independently highlighted paths from `mu` and `sigma` through `z` to `f(z)`. Sliders move parameters while keeping epsilon fixed until `Resample` is pressed.
- **Builds (3):** 1. Display `E_q f(z)=E_epsilon f(mu+sigma epsilon)`. 2. Show `partial z/partial mu=1` and `partial z/partial sigma=epsilon`. 3. Apply the chain rule and mark that averaging sample gradients estimates the expectation's gradient under the usual regularity conditions.
- **Math:** `partial f/partial mu=f'(z)` and `partial f/partial sigma=f'(z) epsilon` for the scalar example. Do not claim that a single sample yields the exact expected gradient.
- **Discuss:** “Which quantity is held fixed while taking this sample's derivative?”

##### Scene 35 — A numerical gradient you can check by hand

- **Goal:** Demonstrate that the pathwise explanation predicts actual derivatives.
- **Visual:** Number cards feed the same graph: `mu=1`, `sigma=0.5`, `epsilon=-0.4`, and target `a=0`; show loss `f(z)=(z-a)^2`.
- **Builds (2):** 1. Compute `z=0.8`, `f(z)=0.64`. 2. Reveal `df/dmu=1.6` and `df/dsigma=-0.64`; optionally compare with finite differences at the same epsilon.
- **Math:** For `v=log sigma^2`, `sigma=exp(v/2)` and `df/dv=f'(z)*(sigma epsilon/2)=-0.16` here.
- **Discuss:** “Why would resampling epsilon between the two finite-difference evaluations make the comparison noisy?”

##### Scene 36 — Implement the encoder's Gaussian safely

- **Goal:** Translate probability parameters into the values a network actually outputs.
- **Visual:** One hidden representation branches into `mu` and `logvar` heads; an explicit exponentiation box converts log variance to standard deviation; noise arrives through a separate input.
- **Builds (2):** 1. Label `logvar=log(sigma^2)` and `std=exp(0.5*logvar)`. 2. Highlight the three implementation lines `std = exp(0.5*logvar)`, `eps = randn_like(std)`, `z = mu + std*eps`.
- **Math:** Shapes of `mu,logvar,std,eps,z` all match. A normal distribution's reparameterized sampling method, where supported, corresponds to this construction; an ordinary detached sample does not.
- **Discuss:** “What goes wrong if you use exp(logvar) where standard deviation is required?”

#### F. Build and use a VAE — 18 minutes

##### Scene 37 — The complete VAE, with tensor shapes

- **Goal:** Assemble everything already introduced into one architecture.
- **Visual:** A clean left-to-right graph, split across reveals to avoid a wall of boxes. Distinguish encoder and decoder parameters by color.
- **Builds (3):** 1. Show image `x: [B,1,28,28] → flatten [B,784] → Linear(784,128) → h [B,128]`. 2. Branch to `mu,logvar: [B,2]`, reparameterize to `z: [B,2]`. 3. Decode `Linear(2,128) → Linear(128,784)` to logits `[B,784]`, probabilities and image-shaped display.
- **Math:** `q_phi(z|x)=N(mu_phi(x),diag(exp(logvar_phi(x))))`. A 2D latent is chosen for visualization, not claimed to be optimal for image quality.
- **Discuss:** “Why are there two encoder heads but only one sampled latent tensor?”

##### Scene 38 — Reconstruction is a likelihood, not an unexplained penalty

- **Goal:** Tie decoder output and familiar losses to an explicit observation model.
- **Visual:** Magnify a tiny binary image; each decoder logit produces a Bernoulli probability. Show one correct and one poor predicted pixel probability.
- **Builds (3):** 1. Define a factorized Bernoulli likelihood for binary pixels. 2. Expand its negative log likelihood into binary cross-entropy summed over pixels. 3. Compare briefly with a fixed-variance Gaussian observation model, whose NLL is proportional to squared error plus constants.
- **Math:** `-log p_theta(x|z)=-sum_i[x_i log pi_i+(1-x_i)log(1-pi_i)]`. Use logits with `BCEWithLogitsLoss`/`binary_cross_entropy_with_logits`; sigmoid is for displaying pi. If using MNIST, explicitly binarize it for this Bernoulli interpretation. Soft grayscale targets must not be silently described as observed Bernoulli outcomes.
- **Discuss:** “What distributional assumption makes MSE an appropriate NLL?”

##### Scene 39 — The Gaussian KL has a closed form

- **Goal:** Explain every term in the exact KL used by the implementation.
- **Visual:** Prior and posterior curves for one coordinate with separate controls for mean and variance; then a sum over latent coordinates.
- **Builds (3):** 1. Show the one-dimensional formula. 2. Generalize to diagonal covariance. 3. Translate it into log-variance code and verify zero at `mu=0,logvar=0`.
- **Math:** `KL(N(mu,diag(sigma^2))||N(0,I))=0.5 sum_j[mu_j^2+sigma_j^2-1-log sigma_j^2]`; equivalently `-0.5 sum_j[1+logvar_j-mu_j^2-exp(logvar_j)]`.
- **Discuss:** “Why is sigma=1 the prior match here, whereas sigma approaching zero is not?”

##### Scene 40 — A complete two-pixel VAE loss, computed by hand

- **Goal:** Connect a latent sample, reconstruction likelihood and KL numerically.
- **Visual:** Use a compact two-pixel observation only for this worked example; clearly mark the temporary change from the `28×28` architecture. Show all number cards as outputs of the corresponding conceptual network steps.
- **Builds (3):** 1. Set `x=(1,0)`, `mu=(1,-1)`, `sigma=(0.5,1)`, `epsilon=(0.2,-0.4)`, so `z=(1.1,-1.4)`. 2. Given decoder probabilities `pi=(0.75,0.25)` at this z, compute reconstruction NLL `-log(0.75)-log(0.75)=0.575364145`. 3. Compute KL `1.318147181` and the one-sample negative-ELBO estimate `1.893511326`.
- **Math:** `logvar=(log 0.25,0)=(-1.386294361,0)`. Show both latent-coordinate KL contributions: `0.818147181` and `0.5`. The decoder probabilities are specified outputs for this numerical example, not numbers inferred from mu or sigma.
- **Discuss:** “Which term is estimated using this one epsilon, and which has been computed exactly?” Expected: reconstruction expectation is sampled; Gaussian KL is analytic.

##### Scene 41 — One VAE training step, with the correct reductions

- **Goal:** Give a complete executable training recipe and map every line to the architecture.
- **Visual:** Left: the architecture. Right: a short code panel, highlighting a few lines at a time; below: per-example reconstruction, KL and total values.
- **Builds (3):** 1. Encode, reparameterize and decode. 2. Sum reconstruction over pixels and KL over latent dimensions for each example. 3. Add the terms, average over the batch, backpropagate and update both encoder and decoder.
- **Math:** `loss=(recon_per_example + kl_per_example).mean()`. Do not average reconstruction over all pixels while leaving KL summed unless an explicit scaling change is intended and explained.
- **Discuss:** “Why would changing reduction conventions silently change the relative strength of the KL term?”

##### Scene 42 — Generation requires the prior and decoder

- **Goal:** Make the deployment computation unmistakable.
- **Visual:** Toggle between reconstruction and generation; in generation, remove the observed image and encoder and sample z from the prior. Show a toggle between decoder probability image and a Bernoulli sample.
- **Builds (2):** 1. Reconstruction: encode observed x, use a posterior sample or explicitly labelled mean, decode. 2. Generation: sample `z~N(0,I)` and decode; choose mean/probability display or sample the observation model.
- **Math:** `z: [N,2] → decoder logits [N,784] → sigmoid → probabilities`; actual binary samples use a Bernoulli draw from those probabilities.
- **Discuss:** “Is the encoder needed to generate a new image from scratch?”

##### Scene 43 — Explore the latent plane without inventing semantic axes

- **Goal:** Show continuity, diversity and the distinction between posterior encoding and prior sampling.
- **Visual:** A 2D plane with a draggable latent point, a prior-density backdrop, an optional encoded-data overlay and a linked output. A small grid shows decoded probabilities at fixed coordinates.
- **Builds (2):** 1. Move z while holding the decoder fixed. 2. Compare uniformly selected grid points, prior-weighted samples and encoded observations.
- **Math:** A uniformly spaced latent grid is a visualization, not a sample from the Gaussian prior. Do not label coordinates as “rotation” or “thickness” unless the trained example supports that interpretation.
- **Discuss:** “Why might far-away grid points produce worse images even if the central region is useful?”

##### Scene 44 — What the VAE objective encourages—and what it does not guarantee

- **Goal:** Provide an honest model interpretation before introducing GANs.
- **Visual:** Three compact, evidence-labelled examples: good reconstruction with poor prior samples; strong regularization with lost detail; a functioning compromise. Use actual recorded model outputs where possible.
- **Builds (2):** 1. Explain the reconstruction/regularization tradeoff. 2. Introduce posterior collapse and restrictive likelihoods as possible failure modes, while avoiding blanket “all VAEs are blurry” claims.
- **Math:** If a beta slider is provided, label `beta=1` as the vanilla negative ELBO and `beta!=1` as a modified objective. A changed slider only displays real retrained checkpoints if such checkpoints exist; it must not pretend an untrained model instantly adapts.
- **Discuss:** “Could we train a generator using a learned comparison with real data instead?”

#### G. Generative adversarial networks — 32 minutes

##### Scene 45 — Learn a training signal from examples

- **Goal:** Introduce the adversarial idea as a concrete supervised subproblem.
- **Visual:** Real samples and generated samples enter a small binary classifier. Use a tiny point distribution first, with image thumbnails as a secondary connection.
- **Builds (2):** 1. Show the generator's outputs and the real data. 2. Introduce a discriminator trained to distinguish their sources.
- **Math:** `z~N(0,I)`, `x_fake=G_theta(z)`; `D_psi(x)=sigmoid(a_psi(x))`. The discriminator's logit `a` is distinct from its probability display `D`.
- **Discuss:** “Do we need a hand-coded similarity function between every generated image and a real image?”

##### Scene 46 — Two networks, two parameter sets, one generated sample

- **Goal:** Establish the GAN's architecture before writing its losses.
- **Visual:** A generator branch and a real-data branch meet at the same discriminator; shared discriminator weights are indicated clearly rather than duplicating independent networks.
- **Builds (3):** 1. `z:[B,32] → G → fake:[B,784]`. 2. `real:[B,784] → D → real_logits:[B,1]`. 3. `fake:[B,784] → the same D → fake_logits:[B,1]`.
- **Math:** Suggested image demo: `G:32→128→256→784→Sigmoid`; `D:784→256→128→1` with LeakyReLU hidden activations and no terminal sigmoid in the implemented D. Use real and fake pixel values on the same `[0,1]` scale.
- **Discuss:** “Why must both branches use the same discriminator parameters?”

##### Scene 47 — The discriminator has ordinary binary labels

- **Goal:** Connect adversarial training to familiar BCE.
- **Visual:** A real example receives target 1; a generated example receives target 0; show their logits, displayed probabilities and individual losses.
- **Builds (3):** 1. Reveal real target and term. 2. Reveal fake target and term. 3. Sum the two batch means.
- **Math:** `L_D=-E_real log D(x)-E_z log(1-D(G(z)))`. With `D(real)=0.8` and `D(fake)=0.3`, the corresponding single-pair loss is `-log 0.8-log 0.7=0.579818495`.
- **Discuss:** “Which network should change while we train this classifier?”

##### Scene 48 — Update D: detach the generated sample

- **Goal:** Make discriminator-step gradient isolation visible and executable.
- **Visual:** The generator is visually fixed; the fake tensor crosses a `detach` boundary; backpropagation arrows end inside D and do not reach G.
- **Builds (3):** 1. Clear D gradients and generate fake samples. 2. Feed `fake.detach()` into D and compute `L_D` from raw logits. 3. Backpropagate and update only D; show unchanged G parameter values for a tiny numeric readout.
- **Math:** `loss_D = BCEWithLogits(real_logits,ones)+BCEWithLogits(fake_logits,zeros)`. Detaching changes the gradient graph, not the fake tensor's numerical values.
- **Discuss:** “Would merely calling opt_D.step() prevent unnecessary or stale gradients from accumulating in G?”

##### Scene 49 — Update G: the target changes, the gradient path remains

- **Goal:** Explain the practical non-saturating generator loss and correct freezing semantics.
- **Visual:** D's parameters are locked, but the path from its output through its input to G remains lit. Use different visual conventions for a locked parameter and a severed computational edge.
- **Builds (3):** 1. Generate a fresh fake batch and pass it through the updated D. 2. Set the generator objective's target to 1. 3. Backpropagate through D into G and update G while D parameters remain fixed.
- **Math:** `L_G=-E_z log D(G(z))`; implemented as `BCEWithLogits(D(G(z)),ones)`. Do not detach fake, use `no_grad()` around D, or detach D's output during the G step.
- **Discuss:** “Why is freezing D's parameters different from stopping gradients through D?”

##### Scene 50 — The original minimax objective and the practical objective

- **Goal:** Connect the commonly quoted GAN formula to the code students will write.
- **Visual:** Place the theoretical two-player objective above two clearly labelled generator-loss alternatives. Mark the practical code path.
- **Builds (2):** 1. Show `min_G max_D [E_real log D(x)+E_z log(1-D(G(z)))]`. 2. Contrast the minimax generator term `E_z log(1-D(G(z)))` with the non-saturating choice `-E_z log D(G(z))`.
- **Math:** These are different generator objectives with a shared desirable distribution-matching solution under the idealized analysis. Do not call them algebraically identical or claim finite neural-network training is guaranteed to converge.
- **Discuss:** “Why might we use a different generator objective from the original minimax term?”

##### Scene 51 — See the gradient saturation in one scalar

- **Goal:** Explain the non-saturating choice with an actual derivative, not a slogan.
- **Visual:** A logit `a` slider controls `D=sigmoid(a)`. Plot both generator losses and their derivatives with respect to a. Keep loss and gradient axes separate.
- **Builds (3):** 1. Differentiate the minimax generator term: `d log(1-sigmoid(a))/da = -sigmoid(a)`. 2. Differentiate the non-saturating loss: `d[-log sigmoid(a)]/da = sigmoid(a)-1`. 3. Set `D=0.01`: derivatives are `-0.01` and `-0.99`, respectively.
- **Math:** Both derivatives are with respect to D's input logit. The complete generator gradient also includes `da/dx_fake` and `dx_fake/dtheta`; the scalar comparison does not guarantee that all parameter gradients are large.
- **Discuss:** “What happens early when D confidently identifies a generated sample as fake?”

##### Scene 52 — One complete alternating iteration

- **Goal:** Consolidate the two update phases and their sequence.
- **Visual:** A two-phase timeline with four tensor/parameter readouts: G parameters, D parameters, fake tensor and discriminator logits. A `Next operation` button advances individual operations.
- **Builds (3):** 1. Sample real/noise, compute detached fake and update D. 2. Recompute a fresh forward pass using updated D, freeze D parameter gradients and update G. 3. Restore D parameter gradients for the next iteration and show which quantities changed in each phase.
- **Math:** One D step and one G step are a pedagogical default, not a universal optimal update ratio. No `retain_graph=True` is needed with these fresh forward passes.
- **Discuss:** “Why is it unsafe to reuse a stale graph across an in-place optimizer update?”

##### Scene 53 — Follow an image through every GAN tensor

- **Goal:** Make the image implementation as concrete as the VAE implementation.
- **Visual:** Highlight one row from a batch and show shape badges after every layer; synchronize with a few code lines.
- **Builds (3):** 1. Trace `z [64,32] → [64,128] → [64,256] → [64,784]`. 2. Reshape one fake to `[1,28,28]`; trace D through `[64,256] → [64,128] → [64,1]`. 3. Show matching target shapes `[64,1]` and scalar losses after reduction.
- **Math:** Label latent dimension 32, batch size 64 and pixel dimension 784 as distinct roles. Never let broadcasting hide a mismatched `[B]` target versus `[B,1]` logit.
- **Discuss:** “Which dimensions change if the batch size doubles, and which learned matrix dimensions remain fixed?”

##### Scene 54 — Generation uses G alone

- **Goal:** Separate GAN training cost from its sampling pathway.
- **Visual:** Switch to generation: remove real data, the discriminator, labels, losses and optimizers. A prior feeds G and a sample grid.
- **Builds (2):** 1. Display the whole training system. 2. Fade unused components and generate several samples from fixed random seeds.
- **Math:** `z~p(z)`, `x=G_theta(z)`. The basic GAN discussed here supplies samples without a generally tractable explicit image likelihood.
- **Discuss:** “Does a discriminator score need to be computed for every generated image after training?”

##### Scene 55 — High-quality samples can still miss a mode

- **Goal:** Make mode collapse concrete and distinguish it from poor realism.
- **Visual:** Reuse the two-cluster point cloud. Compare a model covering both clusters with a generator covering one cluster very well. Show separate realism and coverage counters for this fully specified toy geometry.
- **Builds (2):** 1. Reveal a sample set concentrated in one real cluster. 2. Reveal the missing mode and show many distinct latent draws mapped to similar outputs.
- **Math:** The counters are toy diagnostics based on distances to known cluster centers, not general-purpose image-generation metrics. State their definitions and thresholds.
- **Discuss:** “Would selecting only the most attractive generated samples reveal this failure?”

##### Scene 56 — Read a training run without overinterpreting its losses

- **Goal:** Demonstrate useful monitoring and the limits of adversarial loss curves.
- **Visual:** Fixed-seed generated samples across checkpoints; D and G loss histories; cluster coverage for the toy dataset; a small held-out/nearest-neighbour comparison when image samples are available. Keep plot axes and sample counts fixed across checkpoints. If a one-dimensional density view is offered, label a density estimated from generated samples as a KDE/histogram rather than an analytically known GAN density.
- **Builds (2):** 1. Inspect sample changes and coverage across actual recorded checkpoints. 2. Overlay losses and reveal a case where a superficially reassuring loss does not establish a good generator.
- **Math:** In an ideal matched-distribution setting an optimal D returns 1/2; observing approximately 1/2 in a finite model is not proof of distribution matching. Avoid asserting a universal best GAN loss value.
- **Discuss:** “What else would you inspect before claiming the model works?”

##### Scene 57 — Condition both players

- **Goal:** Reconnect GANs to the application gallery and earlier conditioning architectures.
- **Visual:** A class label `c` enters G alongside z and enters D alongside the real or generated sample. Swap the condition while keeping z fixed.
- **Builds (2):** 1. Add c to G to control the sample. 2. Add c to D so it can assess whether the sample matches the requested condition.
- **Math:** `x_fake=G(z,c)` and `D(x,c)`. A simple classroom implementation concatenates a one-hot class vector; more elaborate attention conditioning is outside the core route.
- **Discuss:** “What can go wrong if the discriminator never sees the requested condition?”

##### Scene 58 — Read the complete code and set up the lab

- **Goal:** Make the GAN block immediately actionable without turning the lecture into a terminal session.
- **Visual:** A compact, syntax-highlighted two-update function, with toggles for D-step and G-step emphasis. Link to the separate runnable lab files.
- **Builds (2):** 1. Highlight stable logits-based BCE, `detach()` in the D step and parameter freezing in the G step. 2. Show the lab's three experiments: correct training, one-mode coverage failure, and a deliberately broken gradient-path diagnostic.
- **Math:** Map code symbols to the existing shapes and objectives. Explain that BatchNorm/Dropout state handling would require care in richer models; the minimal supplied MLP deliberately has neither.
- **Discuss:** “Which single incorrect line would make the generator receive no learning signal?”

#### H. Comparison and exit questions — 6 minutes

##### Scene 59 — Compare the recipes on the same questions

- **Goal:** Leave a coherent model map rather than a list of unrelated acronyms.
- **Visual:** A small comparison table: autoregressive model, VAE and GAN; columns are training signal, generation steps, auxiliary network needed for generation and likelihood access. Pair it with three miniature sampling diagrams.
- **Builds (2):** 1. Populate AR and VAE using student answers. 2. Add GAN and highlight the common role of randomness and learned structure.
- **Math:** AR: exact factorized likelihood for the specified model; VAE: explicit latent model with generally intractable marginal likelihood and an optimized lower bound; GAN: implicit sample distribution without a generally tractable explicit density. Avoid ranking methods as universally better/worse.
- **Discuss:** “Which pieces of the sampling process are learned, and which are specified?”

##### Scene 60 — Three exit questions and the next lecture's problem

- **Goal:** Check understanding and create a precise diffusion transition.
- **Visual:** Three concise questions revealed together after the first one has been considered; then a small noisy-image filmstrip leading into the next lecture.
- **Builds (2):** 1. Ask: “Why is the VAE encoder unnecessary for prior sampling?”, “What are the two different KL divergences in the ELBO discussion?”, and “How can D remain frozen while gradients reach G?” 2. Reveal answer fragments and ask whether generation could be learned as a sequence of small denoising steps.
- **Math:** No diffusion equation yet. Do not imply that a one-step generator is mathematically impossible or that diffusion is simply repeated subtraction of known noise.
- **Discuss:** The handoff question is “Can a deliberately corrupted training example give us its own supervised learning target?”

### 4. Mathematical source of truth

Implement one shared module for all numbers and use it in the lecture, lab and validation. The following values are correctness targets, with natural logarithms and a tolerance of `1e-7` for double-precision UI calculations where appropriate.

```text
P = [0.8, 0.2]
Q = [0.6, 0.4]
KL(P||Q) = 0.09151622184943578
KL(Q||P) = 0.10464962875290959
H(P) = 0.5004024235381879
H(P,Q) = 0.5919186453876236

Finite latent example, observed x=1:
prior = [0.5, 0.5]
likelihood_at_x = [0.8, 0.2]
q = [0.6, 0.4]
evidence = 0.5
posterior = [0.8, 0.2]
expected_log_likelihood = -0.7776612957621659
KL(q||prior) = 0.020135513550688863
ELBO = -0.7977968093128548
posterior_gap = 0.10464962875290959
log_evidence = -0.6931471805599453

Reparameterization example:
mu = 1.0; sigma = 0.5; epsilon = -0.4; a = 0.0
z = 0.8
(z-a)^2 = 0.64
d_loss/d_mu = 1.6
d_loss/d_sigma = -0.64
d_loss/d_logvar = -0.16

Two-pixel VAE example:
x = [1, 0]
mu = [1, -1]
sigma = [0.5, 1]
epsilon = [0.2, -0.4]
z = [1.1, -1.4]
decoder_probability = [0.75, 0.25]
reconstruction_NLL = 0.5753641449035618
Gaussian_KL = 1.3181471805599454
one_sample_negative_ELBO = 1.8935113254635072

GAN pair:
D(real)=0.8; D(fake)=0.3
discriminator_loss = -log(0.8)-log(0.7) = 0.5798184952529422
non_saturating_generator_loss = -log(0.3) = 1.2039728043259361
```

Implement a stable `klDiscrete(p,q)` that validates nonnegative entries and normalization, treats a zero-weight term as zero, and returns positive infinity when `p_i>0,q_i=0`. Do not silently renormalize invalid data supplied through code; the UI controls can enforce normalization explicitly. Do not replace true infinity with an arbitrary large finite result. For display only, round to a sensible number of digits; calculations retain full precision.

Keep exact quantities distinct from Monte Carlo estimates in notation and labels. In particular, the analytical ELBO is a lower bound; the estimate obtained from one random latent draw is not guaranteed to respect that inequality for every realized draw.

#### Canonical LaTeX for the mathematical scenes

Use these expressions as the rendering source of truth. They make the storyboard's compact code-style notation explicit; they do not add a second mathematical lecture or new core scenes. Introduce their symbols through the specified pictures and reveals.

**KL and the familiar cross-entropy loss.** For discrete normalized distributions with the required support,

\[
D_{\mathrm{KL}}(P\|Q)=\sum_i p_i\log\frac{p_i}{q_i},\qquad
H(P)=-\sum_i p_i\log p_i,\qquad
H(P,Q)=-\sum_i p_i\log q_i
=H(P)+D_{\mathrm{KL}}(P\|Q).
\]

At fixed target distribution \(P\), minimizing cross-entropy over \(Q\) minimizes this KL because \(H(P)\) is constant. Explain zero-probability terms using the conventions already specified. For continuous densities, replace the sum with the appropriate integral; density values are not individual point probabilities.

**ELBO derivation, with the seven prescribed reveal steps.** Keep the observed \(x\) fixed. Use full arguments until students know which distribution each expectation uses:

\[
\begin{aligned}
\log p_\theta(x)
&=\log\int p_\theta(x,z)\,dz \\
&=\log\int q_\phi(z\mid x)\,
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz \\
&=\log\mathbb E_{z\sim q_\phi(z\mid x)}
\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \\
&\geq\mathbb E_{z\sim q_\phi(z\mid x)}
\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \\
&=\mathbb E_{q_\phi(z\mid x)}
\left[\log p_\theta(x\mid z)+\log p(z)-\log q_\phi(z\mid x)\right] \\
&=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right) \\
&=: \mathcal E(\theta,\phi;x).
\end{aligned}
\]

Render at most the active line and the few preceding lines needed for context on a projector frame. The entire aligned block belongs in reading notes and a continuation export. The equality involving a ratio requires the support condition described in Scene 27. Concavity of the logarithm establishes the inequality direction.

**The gap has a different KL.** Expand Bayes' rule to obtain

\[
\log p_\theta(x)
=\mathcal E(\theta,\phi;x)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
\]

The loss compares the approximate posterior with the prior; the bound gap compares it with the true posterior. Use the finite-latent calculator to show both values independently.

**Reparameterization and its gradient.** For the diagonal Gaussian encoder,

\[
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\qquad \sigma_\phi(x)=\exp\!\left(\tfrac12\operatorname{logvar}_\phi(x)\right).
\]

When \(f\) depends on \(\phi\) through this sample and differentiation may pass through the expectation,

\[
\nabla_\phi\mathbb E_{q_\phi(z\mid x)}[f(z)]
=\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[\nabla_\phi f\!\left(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\right)\right].
\]

For the scalar case with one fixed noise draw,

\[
\frac{\partial f}{\partial\mu}=f'(z),\qquad
\frac{\partial f}{\partial\sigma}=f'(z)\epsilon,\qquad
\frac{\partial f}{\partial v}=f'(z)\frac{\sigma\epsilon}{2},
\quad v=\log\sigma^2.
\]

An individual sample gradient estimates the expectation's gradient. The expectation identity does not imply an individual draw equals its population average.

**VAE reconstruction and analytic Gaussian KL.** With binary observations and decoder probabilities \(\pi_{\theta,i}(z)\),

\[
-\log p_\theta(x\mid z)
=-\sum_i\left[x_i\log\pi_{\theta,i}(z)
+(1-x_i)\log(1-\pi_{\theta,i}(z))\right],
\]
\[
D_{\mathrm{KL}}\!\left(\mathcal N(\mu,\operatorname{diag}(\sigma^2))\|
\mathcal N(0,I)\right)
=\frac12\sum_j\left[\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right].
\]

Sum reconstruction terms over pixels and KL terms over latent coordinates, add them per example, then average across the batch. The code uses raw decoder logits for numerical stability.

**GAN objectives with one label convention.** Let \(D_\psi(x)\) denote the displayed probability that an example is real, and let \(G_\theta(z)\) generate an example. The original game is

\[
\min_\theta\max_\psi\;
\mathbb E_{x\sim p_{\mathrm{data}}}\log D_\psi(x)
+\mathbb E_{z\sim p(z)}\log\left(1-D_\psi(G_\theta(z))\right).
\]

The implemented discriminator and non-saturating generator minimization losses are

\[
\mathcal L_D=-\mathbb E_{x\sim p_{\mathrm{data}}}\log D_\psi(x)
-\mathbb E_{z\sim p(z)}\log\left(1-D_\psi(G_\theta(z))\right),
\qquad
\mathcal L_G=-\mathbb E_{z\sim p(z)}\log D_\psi(G_\theta(z)).
\]

Display these losses separately during their respective update phases. Executable code passes raw logits to stable BCE. With \(D=\operatorname{sigmoid}(a)\), the minimax generator term has derivative \(-D\) with respect to \(a\), while the non-saturating loss has derivative \(D-1\); continue the chain rule through the discriminator input into generator parameters.

### 5. Minimal runnable VAE code contract

The downloadable lab must include a genuinely runnable version of this example, with imports, a small data source, a seeded configuration and a generation command. The core below is executable PyTorch code. A procedural binary glyph dataset can make the smallest path work offline; optionally provide binarized MNIST as an additional dataset. The binary data interpretation is part of the lesson.

```python
import torch
from torch import nn
from torch.nn import functional as F

class TinyVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Linear(784, 128)
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, 128)
        self.decoder_logits = nn.Linear(128, 784)

    def encode(self, x):
        h = F.relu(self.encoder(x.flatten(1)))
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder_logits(F.relu(self.decoder_hidden(z)))

    def forward(self, x, eps=None):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mu + std * eps
        return self.decode(z), mu, logvar

def vae_step(model, optimizer, x):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, mu, logvar = model(x)
    target = x.flatten(1)
    recon = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).sum(dim=1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1)
    loss = (recon + kl).mean()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "recon": float(recon.detach().mean()),
        "kl": float(kl.detach().mean()),
    }

@torch.no_grad()
def sample_vae(model, count=16, sample_pixels=False):
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.mu_head.out_features
    z = torch.randn(count, latent_dim, device=device)
    probabilities = model.decode(z).sigmoid()
    output = torch.bernoulli(probabilities) if sample_pixels else probabilities
    return output.reshape(count, 1, 28, 28)

if __name__ == "__main__":
    # Offline smoke example, not a claim of meaningful image learning.
    torch.manual_seed(7)
    model = TinyVAE(latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.bernoulli(torch.full((8, 1, 28, 28), 0.2))
    print(vae_step(model, optimizer, x))
    print(sample_vae(model, count=4).shape)
```

In the actual lab, replace the smoke example's independent random pixels with binary glyphs or binarized MNIST before presenting any learned image results. Separate `smoke test`, `training run`, and `recorded checkpoint` clearly. Explain that independent Bernoulli pixels are a decoder likelihood conditional on z; marginal dependencies can still arise through the shared latent variable.

### 6. Minimal runnable GAN code contract

Implement this basic MLP without BatchNorm or Dropout so that freezing parameter gradients has a clean interpretation. Retain fresh forward passes between optimizer updates. The module D returns logits; apply sigmoid only for probability displays.

```python
import torch
from torch import nn
from torch.nn import functional as F

def make_image_gan(latent_dim=32):
    generator = nn.Sequential(
        nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 784), nn.Sigmoid(),
    )
    discriminator = nn.Sequential(
        nn.Linear(784, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 1),  # raw logits: no final sigmoid
    )
    return generator, discriminator

def gan_step(G, D, opt_G, opt_D, real, latent_dim=32):
    G.train()
    D.train()
    real = real.flatten(1)
    batch_size = real.shape[0]
    device = real.device

    # D phase: real=1, generated=0; fake.detach() isolates G.
    D.requires_grad_(True)
    opt_D.zero_grad(set_to_none=True)
    opt_G.zero_grad(set_to_none=True)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)
    real_logits = D(real)
    fake_logits = D(fake.detach())
    loss_D = (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    )
    loss_D.backward()
    opt_D.step()

    # G phase: freeze D parameters, retain differentiation through D's input.
    opt_D.zero_grad(set_to_none=True)
    opt_G.zero_grad(set_to_none=True)
    D.requires_grad_(False)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)                  # fresh graph
    fake_logits = D(fake)        # no detach and no torch.no_grad()
    loss_G = F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )
    loss_G.backward()
    opt_G.step()
    D.requires_grad_(True)       # ready for the next D phase

    return {"D": float(loss_D.detach()), "G": float(loss_G.detach())}

@torch.no_grad()
def sample_gan(G, count=16, latent_dim=32):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(count, latent_dim, device=device)
    return G(z).reshape(count, 1, 28, 28)

if __name__ == "__main__":
    # Smoke example only; meaningful sample figures require a real training run.
    torch.manual_seed(11)
    G, D = make_image_gan()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real = torch.rand(8, 1, 28, 28)  # shared [0,1] scale for real and generated data
    print(gan_step(G, D, opt_G, opt_D, real))
    print(sample_gan(G, count=4).shape)
```

Do not merely paste this code as a static image. Supply readable code, a copy action, highlighted execution phases, downloadable runnable files, and notes mapping tensors to scenes. The offline lab should also include a smaller 2D GAN using the same update semantics so that students can see distribution coverage without needing an image model to converge during class.

For the gradient demonstration, inspect gradients or parameter differences at the phase boundaries: during D's phase, G has no newly computed gradients; during G's phase, G has a nonzero gradient path while D parameters remain unchanged. A numeric sample can have small gradients by chance, so validate the graph behavior on a deliberately nondegenerate seeded test, not an arbitrary single output. Do not use `retain_graph=True` to conceal a stale-forward-pass design.

### 7. The separate practical lab — 45–60 minutes

Supply a short notebook or equivalent script-based walkthrough, with small CPU-capable defaults and an optional larger run. It must work without a paid API. Do not promise a fixed training time or guaranteed high-quality GAN convergence on every machine.

1. **KL and evidence calculator, 8–10 minutes.** Reproduce both KL directions. Verify cross-entropy minus entropy. Reproduce the finite latent ELBO and move q toward the exact posterior. Students answer why the evidence stays fixed while q changes.
2. **Reparameterization gradient, 7–10 minutes.** Implement both the manual expression and autograd for the numeric example. Check finite differences with epsilon held fixed. Compare one draw with an average over many draws, keeping the distinction between sample gradients and expected gradients visible.
3. **VAE, 15–20 minutes.** Train on a small binary glyph dataset or load an explicitly labelled saved checkpoint. Inspect reconstructions, prior samples, probability images, binary samples and the latent grid. Record the exact reconstruction/KL reduction conventions. An optional beta comparison uses separately trained checkpoints and is labelled as a modified objective.
4. **GAN, 15–20 minutes.** Train the 2D version or inspect recorded checkpoints. Step through D and G updates. Show sample coverage as well as losses. Run one clearly isolated broken-code exercise that detaches the fake input during the G step and diagnose the absent generator gradient; restore correct code afterwards.

Provide answer notes separately from student instructions. The lab should reuse the lecture's computation modules and numerical targets. The same seed should reproduce a given recorded example. Do not fabricate successful convergence if a run fails; retain the actual run and explain what it demonstrates.

### 8. Required source and notation notes

Use primary references to verify mathematics and model-specific claims. The prose, diagrams, numeric examples and code should be newly authored; avoid copying a source's exposition or figures without the required permission/attribution. The surrounding prompt supplies course-style and pedagogy references separately.

- VAE / ELBO / reparameterization: Kingma and Welling, *Auto-Encoding Variational Bayes*, https://arxiv.org/abs/1312.6114 . In particular, verify the posterior-gap identity, the reconstruction-minus-KL bound, the Gaussian reparameterization and the diagonal Gaussian KL.
- GAN: Goodfellow et al., *Generative Adversarial Nets*, https://arxiv.org/abs/1406.2661 . Verify the minimax objective, the non-saturating alternative and the distinction between idealized theory and finite-network optimization.
- PyTorch logits-based BCE and gradient recording: use the official documentation for the installed PyTorch version. Check that freeze/detach/no-grad are explained as distinct operations.
- Application references are attached to Scenes 01–10. The proposed sleep-sensing example is explicitly hypothetical.

Put full source titles and URLs in a references drawer and speaker notes; use compact source labels on the relevant application or mathematical scene. Do not drown the primary teaching canvas in citations, but do not omit provenance.

### 9. Acceptance checks — complete all 20

1. **Scene inventory:** The implemented main route contains all 60 logical scenes in order and the planned 145 reveal states, or a documented justified expansion for legibility. Section counts agree with the final manifest and route table. A machine-readable scene manifest records identifiers, titles, section, build count and presenter notes.
2. **Scope honesty:** The UI/notes identify the 120-minute full route, 90-minute compressed route and additional lab time. No claim that the slow route fits in 80 minutes. Compression retains every mandatory topic and all ten application vignettes.
3. **Visual opening:** The lecture begins directly with applications; at least ten concrete application scenes appear before the formal definition, including images, signals and names, with the proposed sleep example visibly labelled.
4. **Equation staging:** The ELBO derivation uses all seven prescribed builds, with the equality-to-inequality transition and Jensen direction visible. A reviewer can navigate backward and forward without losing mathematical context.
5. **KL arithmetic:** Both discrete KL directions, entropy and cross-entropy match the source-of-truth values. Sliders update bars, rows and totals together. Equal distributions produce zero within tolerance.
6. **KL support behavior:** Zero-weight terms produce zero, positive weight against zero model probability produces infinity, and the UI never emits `NaN` or misleading finite substitutes.
7. **Exact ELBO calculator:** The finite latent example computes evidence, posterior, reconstruction expectation, prior KL, ELBO and posterior gap from the generative parameters. The gap identity holds numerically as q changes.
8. **Two KLs distinguished:** The objective's prior KL and the posterior-gap KL have explicit, correctly conditioned arguments. The exact ELBO is distinguished from its one-sample Monte Carlo estimate.
9. **Reparameterization math:** The scalar sample and derivatives, including the log-variance derivative, match the required numbers. The finite-difference comparison uses the same epsilon on both sides.
10. **VAE numerical loss:** The two-pixel example reproduces z, reconstruction NLL, per-coordinate KL, total KL and one-sample negative-ELBO estimate. The example is visibly separate from the `28×28` architecture.
11. **Tensor consistency:** All VAE and GAN shape badges agree with their executable models for at least two batch sizes. Discriminator logits and targets have identical shapes. No accidental broadcasting is relied on.
12. **VAE likelihood and reductions:** Binary likelihood examples use actual binary targets; logits-based BCE sums pixels per example and Gaussian KL sums latent coordinates per example before batch averaging. Decoder probabilities and sampled pixels are labelled distinctly.
13. **VAE generation:** The generation diagram and executable sample function use the prior and decoder without an observed x or encoder forward pass. Reconstruction remains available as a separate operation.
14. **GAN loss correctness:** D uses target 1 for real and 0 for detached fake; G uses the non-saturating target 1; raw logits enter stable BCE. The minimax and practical generator losses are not presented as algebraically identical.
15. **GAN gradient isolation:** A focused validation confirms that D updates leave G parameters unchanged, G updates leave D parameters unchanged, and a valid gradient passes through frozen D into G. Fresh forward passes follow optimizer updates.
16. **No misleading metrics:** Mode-collapse and coverage examples use clearly defined toy diagnostics. Recorded histories come from actual runs. An observed discriminator output near 1/2 is never presented as proof of success.
17. **Interaction reliability:** Required panels support keyboard/focus interaction, reset, deterministic seeds and separate resampling. Navigation keys do not hijack text/number inputs. Restoring a scene does not unexpectedly resample its numerical example.
18. **Visual and export QA:** Check representative desktop and projector-sized layouts, reduced motion and the all-builds export. Equations, code and labels do not clip; figures are vector-native where precision matters; every mathematical symbol remains legible.
19. **Offline and provenance:** The normal teaching route works without live external APIs or account login. Every borrowed visual has provenance; illustrations and hypothetical lab examples are labelled. Source links and local downloadable lab files resolve.
20. **Delivery completeness:** Deliver the lecture implementation, scene/reveal manifest, speaker notes, runnable lab and answer notes, numerical validation results, representative screenshots and exports requested by the surrounding prompt. Report any actual remaining limitation precisely rather than silently dropping a requirement.

Do not finish with a proposal or a request for clarification. Build the specified lecture, verify the acceptance checks, and provide concrete reviewable outputs in the format requested by the surrounding prompt.

---

## Execution contract for the coding agent

Implement the complete lecture described below. This is a build request, not a request for another outline. Work through source inspection, mathematical validation, assets, implementation, notebooks, slide exports, and visual review. Resolve routine implementation choices yourself. Use the supplied scene sequence as the teaching specification. Split a crowded scene into additional builds or slides while preserving its content and parent scene ID. Give continuation slides unique IDs such as `scene-id-a` and `scene-id-b`, with a recorded `parent_scene_id`; never duplicate an HTML ID. Do not replace worked examples with generic explanatory prose.

The deliverable is a classroom-ready interactive lecture for Prof. Nipun Batra's ES 667 Deep Learning course at IIT Gandhinagar. Students know basic probability, MLE/MAP, familiar deep-learning losses, gradients and autograd, constrained outputs, next-token name generation, self-attention, encoder/decoder Transformers, cross-attention, ViT, CLIP, VLM connections, concept bottlenecks, detection, and segmentation. They have not yet learned the generative-model mathematics introduced here. Explain each new probability distribution and expectation before using it.

The teacher explicitly requires a substantial application opening with many actual images, slow mathematical detours, and a generous GAN treatment in Lecture I. The first lecture must teach KL divergence, reparameterization, and ELBO in its main teaching path. Lecture II must reconstruct diffusion from its training examples and sampling algorithm before assembling text-to-image systems. The full specification contains more material than an unhurried 80-minute session. Preserve the approximately 120-minute full route and mark the supplied approximately 90-minute route. A route is a navigation choice over a complete lecture, never deletion of the required content.

### 1. Inspect and extend the existing lecture system

Read applicable repository instructions and inspect the actual checkout with `rg` and `rg --files`. Identify the lecture repository, source layout, build commands, and export tools before editing. The course schedule and lecture content may live in different repositories. Do not assume a particular working directory or create a second competing lecture at a guessed location.

Required style references:

- Course sequence: https://nipunbatra.github.io/dl-2026/schedule.html
- Name-generation lecture: https://nipunbatra.github.io/attention/part1.html
- From Attention to Applications: https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/
- CLIP lecture: https://nipunbatra.github.io/attention/clip/
- VLM lecture: https://nipunbatra.github.io/attention/from-clip-to-vlm/
- VLM PDF: https://nipunbatra.github.io/attention/from-clip-to-vlm/from-clip-to-vlm.pdf
- Concept-bottleneck notes: https://nipunbatra.github.io/cbm/notes.html

Inspect at least an application sequence, a mathematical sequence, a training sequence, and the presentation/export controls. Large single-file HTML lectures may exceed a web reader's limit; use their public HTML/source repository or PDF instead. An oversized response does not establish that the material is unavailable.

Copy the teaching grammar: concrete task, question before explanation, progressive diagram, exact arithmetic, training, generation, then a return to the opening application. In the attention lecture, a fixed sentence anchors multiple tasks. In CLIP, individual dot products and normalization precede the complete loss. In VLM, the same image remains while the query and information flow change. Use the specified running examples in that spirit. Write original slide text and original mathematical figures.

Current VLM styling provides these useful reference tokens. Recheck the source before integrating; preserve established conventions if they differ:

```css
--paper: #F7F8FA;
--ink: #202633;
--muted: #566171;
--line: #D9DEE6;
--vision: #3478E5;
--language: #8B5BB5;
--bridge: #178F82;
--connector: #B98224;
--loss: #D45555;
```

Use blue for images/data, purple for text where present, teal for latent/probabilistic transformations, amber for sampled randomness/time, and red for loss/gradient highlights. Put a visible label next to every color role. Treat these as a coordinated palette, not a rule that every object must use a distinct color. Prefer Avenir Next, Segoe UI, and system sans-serif, with SF Mono/Menlo or an available equivalent for code. Use only redistributable fonts in bundled exports.

Use the current 1280 by 720 classroom stage, scaled uniformly to the viewport. Keep the reading view responsive. Preserve generous margins and room for a bottom source line. Target 24–30 px body text, 36–44 px headings, and readable equation labels. A source footer can be smaller. Do not reduce technical text below classroom readability to force a fit. Move derivation details into successive builds and narration.

The existing VLM reveal API has used `window.lectureReveals` with plans/apply/reset/move/finish/toggle/status; CLIP has used `window.clipLesson`. A generic attention exporter has expected `window.AT.present`. These are different interfaces. Inspect and adapt the selected exporter instead of assuming that one global exists everywhere. A thin adapter is preferable to replacing the working lecture runtime.

If no relevant checkout exists, create a self-contained folder using plain HTML/CSS/JavaScript, SVG/canvas, and a locally bundled math renderer. Use the same reference style and provide an ordinary static-server launch command. The lecture must work without a backend, authentication, a paid API, or a classroom GPU. Model-training scripts run separately. Do not publish or deploy as a consequence of building these artifacts.

### 2. Learning design and visual requirements

1. Begin with the specified 8–10 application scenes. Each must contain a substantial visual example, not an icon or a title card. Use one main image, a before/after pair, or an aligned condition/output strip. Reserve roughly two-thirds of each application frame for evidence.
2. For each application use three teaching beats where appropriate: what is supplied, what could be generated, and what property makes the problem difficult. Reveal the output after the student prediction question. A source caption may identify the model; do not require students to know the family name yet.
3. Make image selection purposeful. Show variability for generation, a missing region for completion, correspondence for paired translation, and structure preservation for conditioning. Include model failures or competing plausible answers when they teach uncertainty.
4. Reuse several opening images when the mechanism is finally explained. Supply an explicit map from application scene IDs to their later explanatory scenes.
5. Use original SVG for distributions, probability bars, computation graphs, gradients, network blocks, matrix arithmetic, and tensor-shape traces. Never use an image generator for exact equations, charts, or architecture diagrams.
6. Keep no more than three meaningful visual panels on a frame unless a sample grid itself is the teaching object. Avoid small decorative cards, stock robot imagery, busy gradients, unexplained latent-space arrows, and paragraphs pasted over photographs.
7. Present equations in three passes: a picture and verbal question, the equation with symbols introduced, then numbers or code. Highlight only the term under discussion. Define the measure/conditioning of every new expectation.
8. A reveal adds one idea or arithmetic operation. It should not expose an entire dense section. Preserve the previous visual positions while adding a new branch, tensor, factor, or gradient.
9. For each trained component explicitly show its input, output, learnable parameters, target, and loss. For generation identify the supplied condition, sampled randomness, repeated operation, and output. Frozen parameters and changing activations must remain distinct.
10. Make discussion answers hidden until requested. Instructor narration should explain misconceptions and likely student questions, not merely restate the visible slide. Give each mathematical scene a meaningful teaching note.
11. Keep mathematical notation consistent across picture, equation, code, and narration. Distinguish an observed sample from a random variable when it matters. Use natural logarithms and nats throughout unless an explicitly labelled extension changes the base.
12. Avoid claims that an attractive sample validates a model, that a latent dimension is necessarily semantic, that reconstruction proves useful generation, or that visual realism makes a generated scientific quantity a measurement.

### 3. Runtime and navigation specification

Use a structured scene registry or its equivalent in the existing runtime. Every scene needs a stable ID, section, title, learning goal, ordered build states, visual description, narration, question/answer when applicable, source/asset IDs, and route membership. Keep the source of displayed numbers separate from styling.

Required controls:

- Reading mode with completed diagrams, expandable calculations, and notes.
- Presentation mode compatible with the existing `?present` convention, stable scene/build deep links, next/back, section overview, and visible progress.
- Keyboard navigation that respects focused inputs. Escape exits overlays/presentation. An overview lets the teacher jump directly to KL, ELBO, reparameterization, GAN training, or the appropriate Lecture II section.
- Full and shortened teaching routes. Show scene count and build progress separately; do not confuse 60 logical scenes with 150 exported reveal states.
- Reset/step/play/pause controls for each simulation. Reset must restore the exact seed and inputs. Pause simulations when leaving their scene.
- Accessible names, adequate contrast, and reduced-motion support. Motion must never be the only way to understand a mathematical operation.
- Every interactive has a meaningful static fallback for reading and PDF. A PDF must show a defined example state, not an empty canvas or an instruction to click.

Prefer deterministic seeded JavaScript for analytic demos. If a demo uses a trained network, export real weights/checkpoints or recorded outputs with provenance. A parameter slider may recompute a mathematical function; it must not pretend to retrain an image model instantly. If showing saved checkpoints, label the control as checkpoint selection. Avoid running expensive training or a remote image-generation call during classroom presentation.

An implementation may expose a small export adapter with `listScenes`, `getBuildCount`, `goTo`, `setBuild`, `finish`, and `ready`. These names are a proposed internal interface, not a claim about an existing runtime. Map it to whichever tested interface the repository provides. Wait for `document.fonts.ready`, decoded images, completed math rendering, and a settled scene before capture.

### 4. Source and asset handling

Use the reference/source ledger below as a starting point. Resolve and inspect actual assets before embedding. Download permitted assets into the lecture folder so the classroom presentation does not depend on third-party hotlinks. Keep the source image at a useful resolution and use an appropriately sized derivative in HTML. Retain animation/video posters and playback controls.

Create a machine-readable asset manifest with: `id`, `local_path`, `source_page`, `asset_url` if available, `authors_or_owner`, `paper_or_dataset`, `figure_or_caption`, `license_or_usage_note`, `retrieved_date`, `transformations`, `evidence_type`, and `used_in_scene_ids`. Use a real acquisition date. Evidence types should distinguish published model output, local trained-model output, deterministic simulation, and illustrative teaching image.

Cite each reused image/plot next to the visual, and each technical claim in its notes/source list. Source links need to be usable in the exported PDF and reading mode. If adapting MIT material, follow the attribution requested on the MIT course page, including the named course authors and course title. A code repository's license does not automatically license every third-party photograph shown by that project. Use original author assets with appropriate attribution, openly licensed images, or independently produced teaching examples when reuse is unclear. Do not erase watermarks or authorship credits.

Do not copy entire MIT slides or Lilian Weng article passages. Use their pedagogical ideas and links, and create this course's own explanations and diagrams. Validate equations against the original papers and independent calculations. Teaching resources may simplify probability, gradients, and disentanglement; retain the precise qualifications in this specification.

If an exact published asset is inaccessible, choose another inspected primary-source asset for the same teaching goal. If none can be used, create a clearly labelled conceptual example, record that substitution, and keep the source claim narrow. Never label an AI-created example as an output from a named published model or a real lab experiment. A missing asset should not become a broken image or an unexplained placeholder.

### 5. Required output package

Use the lecture-specific folder name and title. A comparable existing repo structure is acceptable if documented. Provide:

1. A complete interactive HTML lecture and its editable source/assets.
2. A reading/notes view containing all instructor explanations and worked derivations, with stable links to scenes.
3. A completed-scene PDF with one completed view per logical scene, plus an all-builds PDF with every meaningful reveal state. Both default exports cover the full lecture and appendix regardless of the route currently selected in the browser. Clearly identify appendix pages and show logical scene/build identifiers unobtrusively. Completed views may need an extra continuation page when the final state intentionally omits earlier derivation lines. If the exporter rasterizes the stage, explicitly add PDF link annotations for citations/navigation or use a print route that preserves working links. Verify an external source link and an internal section link. Describe a raster PDF as raster; keep accessible selectable text in HTML/notes.
4. The specified runnable PyTorch notebook(s) or scripts, including deterministic toy examples, small-model training, and artifact export. Separate essential CPU demonstrations from optional GPU experiments. Include cached illustrative outputs so the lecture is usable immediately.
5. A concise two-page recap with the core notation, algorithms, and misconceptions. Build this after the lecture is complete.
6. An asset/source manifest and bibliography, plus a requirements-to-scene map covering every required topic.
7. A README with verified setup/build/export commands, full/short route timings, controls, data/checkpoint provenance, and exact output paths.
8. A brief QA record containing mathematical checks, executed notebook checks, browser checks, PDF page counts, and unresolved limitations if any. Do not claim that an unexecuted check passed.

Use descriptive relative paths such as `generative-modelling-i/` or `generative-modelling-ii/`, with `assets/`, `src/`, `notebooks/`, `exports/`, and `scripts/` as appropriate. Keep each lecture independently buildable. If the counterpart lecture is absent, include the needed recap and local utilities. If it is present, reuse its established notation and assets deliberately without introducing a fragile dependency on another working directory.

### 6. Verification and completion

Compute every displayed numerical example from code and compare it with the equations. Use meaningful tests for mathematical semantics and interactions: loss reductions, gradient paths, schedule endpoints, deterministic resets, and sampler indexing. Do not write tests that merely duplicate a hard-coded label.

Render and inspect every logical scene in its final state. Capture all reveal states automatically and inspect the dense derivations, diagrams, images, and code frames individually. Check at the intended 1280 by 720 stage and at a standard browser viewport. Reject missing fonts/images, broken math, clipped legends, intersecting arrows/text, labels outside their objects, empty canvases, unreadably small expressions, and frames whose content overflows. Repair the layout by splitting content, not by hiding the overflow.

Check keyboard navigation, every demo's reset/play/step behavior, section jumps, deep links, full/short routes, reading mode, and both PDF exporters. The loss displayed beside a visual must come from exactly that visual's values. Replaying a saved checkpoint must identify it accurately. Report expensive training runs separately from lightweight checks.

Finish by providing the working local lecture, both PDFs, notebooks, and a concise explanation of what was built and verified. Continue to fix concrete remaining defects. Do not end after an implementation plan, a scaffold, an image gallery, or a deck full of placeholder diagrams.

---

## Pedagogical references and how to use them

Read the following selected resources while implementing. The links are durable references; use current primary papers and source files to resolve any ambiguity. The storyboard in this export is the controlling teaching sequence. The sources support the content and visual approach, but their complete slide text or article structure should not be reproduced.

| Reference | Intended use in this lecture pair |
|---|---|
| [Nipun Batra's ES 667 schedule](https://nipunbatra.github.io/dl-2026/schedule.html) and the linked attention/CLIP/VLM lectures | Establish prerequisites and the actual classroom visual style. Reuse the course's reveal grammar, stable examples, shape labels, question/answer controls, and training/generation explanations. |
| [MIT 6.S191 Deep Generative Modeling slides](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf) and [course page](https://introtodeeplearning.com/) | The inspected 2026 deck has 84 image-based pages. Study the application opening, incremental autoencoder/VAE computation graph, latent-space visuals, and repeated discriminator/generator updates on fixed distribution axes. Follow the course's attribution instructions for any reused/adapted material. The page's publication date and the slide's teaching date may differ. |
| [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | A reading companion for the transition from reconstruction to latent-variable inference. Use it to anticipate where readers lose the connection between the encoder distribution, prior, ELBO, and sampling. Validate the mathematical presentation against Kingma and Welling. |
| [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | A reading companion for distribution comparison and GAN training difficulties. Keep KL, JS, and Wasserstein conceptually distinct. The mandatory lecture uses one unambiguous real/fake convention and the practical non-saturating generator loss. The optimal-discriminator/JS and WGAN material is an appendix extension. |
| [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | A map connecting forward corruption, reverse transitions, training, conditioning, and efficient generation. The main lecture carefully reconstructs DDPM; sampler families, score-based formulations, and flow matching are identified separately. |
| [Stanford CS236: VAE notes](https://deepgenerativemodels.github.io/notes/vae/) | A second educational reference for the marginal-likelihood difficulty and variational lower bound. Derive this export's small examples independently and show the missing algebraic steps. |
| [Stanford CS236: GAN notes](https://deepgenerativemodels.github.io/notes/gan/) | Verify the relationship among the generator's induced distribution, discriminator, optimization objectives, and conditional extensions. |
| [UvA Deep Learning: Deep Autoencoders](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html) | An example of connecting reconstruction pictures, latent dimensions, code, checkpoints, and evaluation. Build the specified simple PyTorch teaching code rather than importing a large training framework merely because this notebook uses one. |

These resources inspire the exposition. The formulas, numerical examples, implementations, and figures in the new lecture must agree with the primary papers listed in its mathematical reference section. In particular, retain these distinctions:

- A reparameterization supplies a pathwise gradient estimator through a deterministic transformation of parameter-independent noise; it is not a universal claim that every stochastic computation is undifferentiable.
- The exact ELBO is a lower bound on log evidence. One Monte Carlo realization is an estimator and need not itself respect that bound.
- The Gaussian prior and diagonal approximate posterior do not guarantee interpretable or disentangled latent coordinates.
- Discriminator label conventions vary across resources. In this lecture, `D(x)` consistently means probability of a real sample, while executable code uses logits.
- The unweighted DDPM noise-prediction objective is related to a variational objective with timestep weights; it is not obtained by dropping all weights as if only an additive constant changed.
- Image-generation applications may use related but different architectures and training methods. Do not present the gallery as evidence that all examples use the exact toy VAE, vanilla GAN, or DDPM implementation taught here.

Each direct source figure needs a local caption and source link; each original diagram should be labelled as an authored teaching diagram in the notes. Keep a record of figure substitutions. Revisit the application images after the corresponding mechanism is explained so the opening gallery becomes part of the conceptual argument.

---

## Image acquisition and application-to-explanation map

Use the following concrete sources to acquire the gallery images. Inspect the linked figure before cropping it and preserve its input/output correspondence. These are candidate assets with precise provenance, not a claim that every source grants identical reuse rights. Follow the asset-handling contract and the actual source terms. Record local filenames and `used_in_scene_ids` after acquisition. The asset key is independent of a scene's HTML ID.

| Asset key | Source page and exact selection | Evidence and teaching use | Provenance detail |
|---|---|---|---|
| `stylegan-samples` | [Official StyleGAN repository](https://github.com/NVlabs/stylegan); [teaser image](https://raw.githubusercontent.com/NVlabs/stylegan/master/stylegan-teaser.png) | Published generated faces for Scene 01; return during GAN sampling and monitoring | Identify the original StyleGAN version. Preserve the authors' credit and check image-material terms separately from any other StyleGAN release. |
| `vae-samples`, `vae-manifold` | [AEVB paper PDF](https://arxiv.org/pdf/1312.6114), Figure 5 and Figure 4(b), PDF page 10 | Published digit samples and decoded latent grid for Scene 02; return during prior sampling and latent exploration | The manifold grid uses Gaussian inverse-CDF coordinates. A decoded grid is a different object from a random sample batch or reconstruction. |
| `stylegan-mixing` | Figure 3 linked by [official figure generation source](https://github.com/NVlabs/stylegan/blob/master/generate_figures.py); [author-linked figure download](https://drive.google.com/uc?id=1U-nlMDtpnf1RcYkaFQtbh5oxnhA97hy6) | Published style mixing for Scene 03 | Preserve the row/column source identities and selected layer-group labels. Describe style mixing precisely; use a separate trained demo for linear interpolation. |
| `pix2pix-shoes` | [pix2pix project](https://phillipi.github.io/pix2pix/); [shoe output gallery](https://phillipi.github.io/pix2pix/images/sketch2photo_shoes/latest_net_G_val/index.html) | A complete input/output/reference row for Scene 04 | Download all images from the same row. Ground truth remains explicitly labelled as a reference photograph. |
| `spade-layout` | [SPADE project](https://nvlabs.github.io/SPADE/); [high-resolution layout/image teaser](https://nvlabs.github.io/SPADE/images/teaser_high_res_uncompressed.png) | Semantic layout to generated appearance for Scene 05 | Retain the semantic class legend. Only claim shared-condition diversity where the figure actually documents it. |
| `pix2pix-facades` | [Facade objective comparison](https://phillipi.github.io/pix2pix/images/index_facades2_loss_variations.html) | Optional replay around Scenes 50 and 57: reconstruction and conditional adversarial objectives | Preserve the source's exact loss names and comparable input/reference images. Do not substitute an unrelated image and call it an objective ablation. |
| `cyclegan-objects`, `cyclegan-seasons` | [CycleGAN project](https://junyanz.github.io/CycleGAN/); [object examples](https://junyanz.github.io/CycleGAN/images/objects.jpg); [season examples](https://junyanz.github.io/CycleGAN/images/season.jpg) | Unpaired translation for Scene 06; optional seasonal example | Distinguish original photos from generated translations. The later vanilla GAN code does not implement CycleGAN's full objective. |
| `context-inpainting` | [Context Encoders project](https://www.cs.cmu.edu/~dpathak/context_encoder/); [published model result figure](https://www.cs.cmu.edu/~dpathak/context_encoder/resources/result_fig.jpg) | Missing-region completion for Scene 07 | Use `result_fig.jpg`. The project's `teaser.jpg` depicts human-artwork inpainting and must not be presented as model output. |
| `sleep-proposal` | Author a deterministic signal schematic locally | Explicitly illustrative proposed research task for Scene 08 | Use a fixed amplitude/time scale and the visible caption “Proposed application; illustrative signals.” Record the generating code. |
| `satellite-reference` | [DiffusionSat project](https://www.samarkhanna.com/DiffusionSat/); [published sample figure](https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_samples.png) | Published metadata-conditioned imagery for Scene 09; comparison with Lecture II | Identify the actual conditions supported by the source. A proposed kiln/rare-structure task should be drawn and labelled separately. |
| `course-name-generator` | [Existing course name-generation lecture](https://nipunbatra.github.io/attention/part1.html), its source and actual generated outputs | Familiar autoregressive generation for Scene 10 | Reuse verified local course assets or execute a compatible example. Record the tokenizer, sampling rule and seed when recreating it. |

### Return to the opening examples

Make these revisits explicit in the scene manifest and presenter notes:

| Opening | Later explanatory scene(s) | Question resolved |
|---|---|---|
| Faces, Scene 01 | 45–54, 56 | How can a learned generator turn random inputs into samples, and how is it trained? |
| Digits and latent grid, Scene 02 | 24–31, 37–43 | What distribution does the decoder represent, and how do prior sampling and encoding differ? |
| Style mixing, Scene 03 | 43, 54 and an optional note | Which manipulation is interpolation in this teaching VAE, and which is the published StyleGAN mixing operation? |
| Shoes and layouts, Scenes 04–05 | 50, 57 | How can an adversarial training signal depend on an observed condition? |
| Translation and completion, Scenes 06–07 | 57, 59 | What additional objectives and conditions distinguish the application systems from the basic GAN? |
| Proposed signals and satellite possibilities, Scenes 08–09 | 55–56, 59 | What do plausibility, coverage and consistency mean for the task? |
| Names, Scene 10 | 15–16, 59 | How does the familiar next-token factorization compare with latent and adversarial generation? |

### Optional depth after the main ending

Provide short reference drawers or clearly marked appendix scenes for the diagonal Gaussian KL derivation, the exact posterior-gap algebra, and the optimal-discriminator result in the original GAN paper. For the latter, define the common support/density assumptions before writing `D*(x)=p_data(x)/(p_data(x)+p_g(x))`; explain the idealized JS-divergence connection with its constants. A brief WGAN reading pointer can link to [Arjovsky, Chintala and Bottou](https://arxiv.org/abs/1701.07875). These extensions should support questions without consuming the main GAN training walkthrough. Keep their IDs and timings outside the 60-scene core inventory.
