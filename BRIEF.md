
MASTER CODEX BRIEF
GENERATIVE MODELLING I — FROM RANDOMNESS TO DATA

Subtitle:
Distributions, Priors, Latent Variables, VAEs and GANs

Duration: 80 minutes.

This is a new lecture in the ES667 Deep Learning course.

It follows the student's experience with:
- next-token prediction / autoregressive language models
- encoder / decoder architectures
- ViT and CLIP
- deep neural networks, backprop and optimization
- elementary probability, distributions and expectation.

QUALITY BAR:
"Beyond Attention" is the primary reference.
Use the improved ViT, CLIP and VLM lectures for the
semantic color system, progressive SVG builds and
question-first explanations.

The lecture must NOT become a historical tour of
generative-model families.

The story is:

    We have learned to predict labels from data.

    Could we instead learn what valid data looks like?

    Where does randomness enter?

    How do we train a neural generator?

    How do different families solve this problem?


==================================================
MANDATORY WORKFLOW
==================================================

Before implementation:

1. Inspect the project/repository.
2. Find the existing Beyond Attention slide-mode source.
3. Find reusable typography, layout, reveal logic,
   color tokens and SVG patterns.
4. Create GENERATIVE_I_CHECKLIST.md.
5. Copy the checklist below into it with all [ ].
6. Write a brief slide-by-slide storyboard before coding.
7. Implement sections and mathematical figures.
8. Run small illustrative / real experiments.
9. Render every main slide/build.
10. Visually inspect and fix the deck.

Mark [x] ONLY for requirements you have checked.
Never fabricate results or mark an untested feature complete.

Produce:
- slide-mode HTML
- PDF export
- reusable source SVG diagrams
- runnable training notebooks/scripts
- metrics / figure manifests
- README for reproducing experiments
- GENERATIVE_I_CHECKLIST.md
- GENERATIVE_I_CHANGELOG.md


==================================================
PEDAGOGICAL CONSTRAINTS
==================================================

Every important idea follows:

QUESTION
→ INTUITION
→ VISUAL CONSTRUCTION
→ SMALL MATHEMATICAL EXAMPLE
→ NEW QUESTION

One major new idea per reveal.

Students must never see the finished VAE or GAN
architecture before understanding why it is needed.

Avoid slides with:
- 5 equations already visible
- 8 explanatory paragraphs
- multiple independent diagrams
- all arithmetic answers revealed upfront.

Use as many physical builds as necessary.

Aim for 45–50 conceptual screens and 65–80
physical builds, while remaining teachable in 80 minutes.

Move optional derivations into backup, not main lecture.


==================================================
SHARED VISUAL SYSTEM
==================================================

Observed data x:
BLUE #3478E5

Randomness and latent variable z:
PURPLE #8B5BB5

Learned generator / encoder / decoder:
TEAL #178F82

Distribution parameters / μ / σ / prior:
AMBER #B98224

Loss / error / gradient:
CORAL #D45555

Structure:
CHARCOAL #30343B

Background:
WARM WHITE #FAFAF8

Arrow semantics:
- blue: observed-data forward flow
- purple: latent/random sampling
- teal: learned transformation
- red dashed: backward gradient
- gray: bookkeeping / reshape.

Mathematics must use proper LaTeX via KaTeX/MathJax.


==================================================
RECURRING EXAMPLES
==================================================

Use only a few primary examples, consistently.

EXAMPLE A:
A simple 1D distribution.

Used to explain:
probability, density, sampling, likelihood.

EXAMPLE B:
One multimodal 2D distribution,
preferably a mixture of Gaussians / two moons.

Used to explain:
learning a distribution, samples, latent mappings,
mode coverage, GANs and flows.

EXAMPLE C:
MNIST handwritten digits.

Used for:
autoencoder, VAE, prior samples, interpolation.

EXAMPLE D:
One recurring image from the vision lecture series.

Used at the very END for the diffusion teaser.

Reuse the same random samples and axes whenever
comparing distribution transformations.


==================================================
SECTION 00 — WHY GENERATE?                 0–6 MIN
==================================================

Opening question:

    What have our models been doing so far?

Show:

IMAGE → ViT → class
TEXT → model → intent
IMAGE + TEXT → CLIP → match score

Then:

    What if I want ANOTHER valid image?

Show real digits and newly generated digits,
with source/provenance clearly labelled.

Contrast:

discriminative:
    x → y

generative:
    draw x_new from learned distribution.

IMPORTANT:
Generation is not simply "put noise into a network."
It is about learning a distribution over possible data.

End section with:

    What does it mean to LEARN a distribution?


==================================================
SECTION 01 — DISTRIBUTIONS AND SAMPLING     6–18 MIN
==================================================

This is foundational.
DO NOT SKIP OR RUSH.

A. DISCRETE DISTRIBUTION

Start with colored categories:

Red    P=.2
Blue   P=.5
Green  P=.3

Ask:
If we sample 100 times, what might happen?

Reveal a frequency histogram progressively.

Explain:
sample frequencies approximate probabilities.

B. CONTINUOUS DISTRIBUTION

Show a Gaussian density.

Explain:
- x-axis: possible numerical values
- curve height: density
- probability of interval: area
- density at one point is NOT the probability
  of obtaining that exact real number.

Introduce:

x ~ N(μ, σ²)

Use μ and σ controls.

Show:
- μ moves the centre
- σ changes spread
- fresh random draws follow the distribution.

C. MULTIMODAL DATA

Show two Gaussian clusters or two moons.

Try fitting:
one Gaussian.

Ask:
Can this represent both modes adequately?

Then:
mixture of two Gaussians.

Show a SIMPLE generative recipe:

1. Sample component k.
2. Sample x from that component's Gaussian.
3. Return x.

This is a REAL simple generative model,
even though it is not neural.

D. LEARNING A DISTRIBUTION

Observed samples:
x1, x2, ..., xN

Fit:
p_θ(x)

Goal:
new samples resemble the underlying data distribution.

Define log-likelihood intuition:

maximize Σ_i log p_θ(x_i).

Do not conflate "can sample" with
"can evaluate normalized density."

Make a slide with two abilities:

DENSITY EVALUATION
    evaluate p_θ(x), if tractable

SAMPLING
    x_new ~ p_θ.

Different families offer different access to these.


==================================================
SECTION 02 — WE ALREADY KNOW ONE           18–24 MIN
==================================================

Title:
Our decoder-only language model was generative.

Recall:

p(x1,...,xT)
=
∏_t p(xt | x_<t).

Use:

"The"
→ "The dog"
→ "The dog ran"
→ ...

Then show the same idea for image pixels/tokens,
schematically.

Ask:

    Must generation always be sequential?

Do not reteach the Transformer.

Finish with a bottleneck:

Autoregressive models generate one element at a time.

Can randomness generate a whole object through
a learned transformation instead?


==================================================
SECTION 03 — LATENT VARIABLES               24–32 MIN
==================================================

Start from a digit image.

Ask:
What factors make two handwritten 7s different?

Possible intuitive factors:
slant, thickness, height, shape.

These are MOTIVATING examples, not guaranteed
independent learned latent dimensions.

Introduce:

z ~ p(z)

x ~ p_θ(x | z).

Explain the role of the PRIOR:

p(z) is chosen/assumed before observing
the specific example x.

Often:

p(z) = N(0,I).

Show a 2D Gaussian cloud.

Sample a purple z from it.
Pass through a teal generator.
Produce x.

Now ask:
During training we observe x, but do not know z.
Which z would reconstruct this example?

Introduce a simple AUTOENCODER:

x → encoder → z → decoder → x_hat.

Show:
original
latent
reconstruction.

Then ask:

    If I now draw a random z from N(0,I),
    will the ordinary autoencoder decode it well?

Not necessarily.

Why?
The ordinary autoencoder has not been trained
to organize its latent codes according to that prior.

Show latent scatter with empty regions / gaps.

This motivates VAE.


==================================================
SECTION 04 — VARIATIONAL AUTOENCODER        32–49 MIN
==================================================

Title:
What if the encoder predicts a DISTRIBUTION?

The standard encoder produces a single vector.

The VAE encoder produces PARAMETERS:

μ_φ(x)

log σ²_φ(x).

Then defines:

q_φ(z|x)
=
N(μ_φ(x), diag(σ²_φ(x))).

Keep terminology explicit:

p(z):
PRIOR distribution in latent space.

q_φ(z|x):
APPROXIMATE POSTERIOR given this x.

p_θ(x|z):
DECODER / LIKELIHOOD MODEL.

These are THREE DIFFERENT OBJECTS.

Show them side-by-side before moving on.


==================================================
REPARAMETERIZATION — THE CENTERPIECE
==================================================

Devote at least 7–9 progressive builds here.

STEP 1:
The encoder predicts μ and σ.

STEP 2:
A latent sample z is needed to reconstruct x.

STEP 3:
Naively sampling z ~ N(μ,σ²) leaves the
gradient estimator unclear if treated
as an opaque stochastic node.

Do NOT say:
"Sampling is mathematically impossible to differentiate."

STEP 4:
Sample standard noise independently:

ε ~ N(0,I).

STEP 5:
Transform it deterministically:

z = μ + σ ⊙ ε.

STEP 6:
Show why this has the same distribution:

if ε ~ N(0,1), then
μ + σ ε ~ N(μ,σ²).

STEP 7:
Draw the actual computation graph:

            ε ~ N(0,I)
                  |
                  v
σ(x) ----------> multiply
                   |
μ(x) ------------> add
                   |
                   v
                   z
                   |
                 decoder
                   |
                  x_hat
                   |
                  loss

STEP 8:
Show gradients backward using dashed red arrows.

For a 1D example:

∂z/∂μ = 1

∂z/∂σ = ε.

STEP 9:
NUMERICAL EXAMPLE.

μ = 2
σ = 0.5
ε = -0.8

z = 2 + 0.5(-0.8)
  = 1.6.

If upstream reconstruction gradient:

∂L_recon/∂z = 3,

then:

∂L_recon/∂μ = 3

∂L_recon/∂σ = -2.4.

Label these as reconstruction-path gradients;
the complete VAE loss also includes the KL term.

STEP 10:
Connect to practical code:

logvar = encoder_logvar(x)
std = exp(0.5*logvar)
eps = randn_like(std)
z = mu + std*eps.

Also show:
∂z/∂logvar = 0.5 σ ε
for the parameterization used in code.

One reusable SVG must highlight μ, σ, ε, z,
decoder and the gradient route.

Students should be able to draw the graph
themselves after this section.


==================================================
SECTION 05 — WHY THE VAE LOSS HAS TWO PARTS 49–59 MIN
==================================================

Ask:
Why not optimize reconstruction error alone?

Show:
the decoder reconstructs well,
but the encoder's latent distributions may
be incompatible with the simple prior.

We need:

1. explain/reconstruct data

2. keep approximate posteriors compatible
   with the sampleable prior.

Introduce the ELBO idea.

Main negative-ELBO form:

L_VAE
=
-E_{q_φ(z|x)}[log p_θ(x|z)]
+
KL(q_φ(z|x) || p(z)).

Do NOT present arbitrary MSE as the only
possible reconstruction likelihood.

Explain:
- Gaussian likelihood can yield squared-error-like loss.
- Bernoulli likelihood yields binary cross entropy
  for appropriate binary data.
- Practical implementations vary.

SHOW KL VISUALLY:

q_φ(z|x)
versus
p(z)=N(0,I).

Move μ and σ.
Show divergence changing.

Then one explicit 1D KL example:

KL =
0.5(μ²+σ²-log σ²-1).

μ=1
σ=.5

KL≈0.818 nats.

If μ=0 and σ=1:
KL=0.

Then complete VAE diagram:

x
→ encoder
→ μ, logvar
→ ε / reparameterize
→ z
→ decoder
→ x_hat

with:
reconstruction loss
+
KL-to-prior.

Important nuance:
the KL term compares individual approximate
posteriors to the prior and is part of the ELBO;
do not assert that every encoded class must form
one perfectly Gaussian cluster.

Finally:

TRAINING:
x → encoder → latent sample → decoder.

GENERATION:
z ~ p(z) → decoder → new x.

The encoder is not needed just to draw prior samples.


==================================================
MNIST VAE DEMONSTRATION
==================================================

Build a small reproducible PyTorch example.

Variant A:
latent_dim = 2

Purpose:
visualize latent geometry and clicked samples.

Variant B:
latent_dim = 16

Purpose:
show sample/reconstruction quality with greater capacity.

Start with small MLPs.

Produce measured plots:

- reconstructions
- prior samples
- 2D latent scatter
- latent grid decoded to digits
- interpolations
- reconstruction vs KL curves.

Do not fabricate or silently retouch outputs.

Interactive:
click a point z in the latent plane.
Display decoded image.

When model weights are unavailable:
use a clearly marked static recording
from a real prior run, or show an explanatory
synthetic diagram—not fake measured digits.


==================================================
SECTION 06 — GANs                          59–71 MIN
==================================================

Transition:

    VAEs train through a probability model
    and a reconstruction/regularization objective.

    Must we use that particular objective?

What if another model could learn to judge realism?

Start with generator:

z ~ N(0,I)
→ G_θ
→ fake x.

Now show real data and generated data mixed.

Ask:
How do we know generated examples are realistic?

Introduce discriminator:

real x ---------+
                |
                v
                D → real/fake score
                ^
                |
z → G → fake x -+.

Do NOT show complete GAN game immediately.

BUILD:

1. generator alone
2. need for feedback
3. discriminator judges real/fake
4. discriminator learns
5. generator learns through discriminator
6. alternate training.

Provide separate forward/backward diagrams.

Classical minimax objective can be shown
as a reference:

min_G max_D
E_data log D(x)
+
E_z log(1-D(G(z))).

For PRACTICAL toy training introduce
the common non-saturating generator objective:

L_D =
-E_data log D(x)
-E_z log(1-D(G(z))).

L_G =
-E_z log D(G(z)).

Clearly label that the generator objective
differs from the saturating minimax formulation.

IMPORTANT:
When training G, a gradient can pass through D
even when D parameters are held fixed.

This is analogous to the frozen-weight
gradient concept already taught elsewhere.

MODE COLLAPSE:

Use the same 2D multimodal dataset.

True distribution:
8 Gaussian modes.

Generator:
only 2–3 modes.

Show:
samples can look plausible yet fail
to cover the distribution.

Distinguish:
sample quality
from
sample diversity / coverage.

Use an optional tiny measured 2D GAN run.
Failure can be educational; never fabricate
a clean convergence story.


==================================================
SECTION 07 — NORMALIZING FLOWS             71–75 MIN
==================================================

Keep concise and visual.

Question:

    What if our generator were invertible?

Start from:
z ~ N(0,I).

Learn:
x = f_θ(z).

For observed x:
z = f_θ^{-1}(x).

Show continuous deformation of a 2D grid
from a simple distribution to a complex one.

Then, optionally, one change-of-variables equation:

log p_X(x)
=
log p_Z(f^{-1}(x))
+
log |det J_{f^{-1}}(x)|.

Takeaway:
invertibility permits exact density calculation
when inverse/Jacobian are tractable.

Do NOT confuse these normalizing flows with
the flow-matching objective in Lecture II.


==================================================
SECTION 08 — THE CLIFFHANGER                75–80 MIN
==================================================

Compare:

AUTOREGRESSIVE:
sample one part at a time.

VAE:
sample from latent prior, decode.

GAN:
generate and learn through adversarial feedback.

FLOW:
invertibly transform a known density.

Now ask:

    Could generation instead be a sequence
    of many small, easy transformations?

Take one real photograph.

Show progressive addition of Gaussian noise.

FIRST:
clean.

THEN:
slightly noisy.

THEN:
very noisy.

FINALLY:
almost pure Gaussian noise.

Say:

    This forward direction is easy.

Then reverse the arrow visually.

Ask:

    If a neural network could learn to undo
    small amounts of corruption,
    could it generate data from noise?

STOP.

Do not teach DDPM equations here.

FINAL SLIDE:

    NEXT:
    GENERATIVE MODELLING II —
    LEARNING TO REVERSE NOISE.


==================================================
REQUIRED INTERACTIVE FIGURES
==================================================

1. Discrete sampling histogram.
2. Gaussian μ and σ controls.
3. Single Gaussian vs mixture distribution.
4. Sample cloud vs learned density visualization.
5. Latent Gaussian sampler.
6. Ordinary AE vs VAE.
7. VAE μ/σ ellipse around one input.
8. Reparameterization graph with gradient path.
9. KL vs prior interactive.
10. VAE 2D decoder grid.
11. GAN alternating-training diagram.
12. GAN mode collapse comparison.
13. Flow deformation.
14. Forward-noise image slider (teaser only).

Every interactive must also have a useful
static PDF-export state.


==================================================
SCIENTIFIC HYGIENE
==================================================

Clearly distinguish:

ILLUSTRATIVE
MEASURED
ANALYTICAL / EXACT.

Never fabricate model results.

Do not imply:
- all generative models have tractable densities
- AE latent codes automatically form a Gaussian
- VAE latent components are automatically semantic
- VAE reconstruction loss is universally MSE
- GAN samples are necessarily sharper than all VAEs
- GAN discriminator output is a calibrated truth probability
- normalizing flow equals flow matching.


==================================================
AUDIT CHECKLIST — CREATE AS MARKDOWN
==================================================

## Narrative
[ ] One central question throughout
[ ] Motivation comes before model names
[ ] Distribution intuition precedes VAE
[ ] Prior is defined visually and mathematically
[ ] Autoencoder sampling problem motivates VAE
[ ] GAN introduced as alternative training signal
[ ] Flow introduced as invertible alternative
[ ] Diffusion teaser ends the lecture

## Mathematical accuracy
[ ] Density vs probability distinguished
[ ] Data distribution vs model distribution distinguished
[ ] Prior vs posterior vs decoder likelihood distinguished
[ ] VAE parameter dimensions checked
[ ] Reparameterization distribution identity correct
[ ] Reparameterization numerical example checked
[ ] Reparameterization gradient example checked
[ ] KL formula correct
[ ] KL numerical example checked
[ ] ELBO signs correct
[ ] Reconstruction likelihood caveat included
[ ] GAN minimax vs practical generator loss distinguished
[ ] Flow Jacobian sign/direction correct

## Visual quality
[ ] Meaningful visual on ≥70% of main slides
[ ] Same examples and colors reused consistently
[ ] One concept per reveal
[ ] No future arithmetic visible prematurely
[ ] All equation typography correct
[ ] All diagrams readable from projector distance

## Experiments
[ ] 2D distribution scripts reproduce figures
[ ] VAE train script runs
[ ] VAE prior samples measured
[ ] VAE latent grid measured
[ ] Interpolations reproducible
[ ] GAN run measured if presented
[ ] Recorded fallback works
[ ] Figure provenance captured

## Final QA
[ ] Every main build rendered
[ ] Every main slide visually inspected
[ ] No overflow/clipping
[ ] No malformed mathematics
[ ] No tiny text
[ ] Appendix separated
[ ] PDF export checked
[ ] Checklist and changelog completed

FINAL REQUIREMENT:

Students should leave able to explain:

"I draw a simple random z.
A neural generator transforms it into data.
The main question is how to learn that generator."

And they should be curious about:

"Can generation instead happen through many tiny
learned steps?"
