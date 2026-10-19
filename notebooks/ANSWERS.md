# Practical answer notes

The numerical source of truth is `src/math.mjs`; `assets/fixtures.json` is its exported snapshot.

- KL(P∥Q) = 0.09151622184943578 nats; KL(Q∥P) = 0.10464962875290959 nats. The first argument weights the log ratios. A negative category contribution is allowed; the full sum is nonnegative. Positive P against zero Q gives infinity, while a zero P contribution is zero.
- H(P) = 0.5004024235381879; H(P,Q) = 0.5919186453876236. Their difference is KL(P∥Q). At fixed P, entropy does not depend on the model.
- Evidence is 0.5 for all q in the finite model. The model's prior and likelihood determine it. At q=(0.6,0.4), ELBO = −0.7977968093 and posterior gap = 0.1046496288. At q=(0.8,0.2), the gap is zero. The prior KL and posterior-gap KL have different second arguments.
- The reparameterized draw gives z=0.8, loss=0.64, dμ=1.6, dσ=−0.64 and dlogvar=−0.16. Resampling between finite-difference evaluations introduces sampling noise. For f=z², the exact expected loss is μ²+σ². Its μ derivative is 2μ and its logvar derivative is σ². A single draw generally gives different derivatives.
- The two-pixel example gives z=(1.1,−1.4), reconstruction NLL=0.5753641449, KL contributions 0.8181471806 and 0.5, total KL=1.3181471806, and one-sample negative ELBO=1.8935113255. Only the reconstruction expectation is estimated; KL is analytic.
- BCE is summed over 784 pixels and KL over two coordinates. Averaging pixel BCE while summing latent KL changes their relative weights. The vanilla negative ELBO has β=1. Other β values define modified objectives and need separate training runs.
- A prior generation graph is noise → latent z → decoder logits → probabilities → optional Bernoulli draw. Encoding is only needed when an observed x is used to approximate its posterior. A probability image is a mean, not a Bernoulli draw.
- D update: real=1, detached fake=0, update D. G update: a fresh fake forward, D parameters frozen, target=1, update G. Freeze preserves derivatives with respect to D's input. Detach severs that path. An ordinary `no_grad()` block around D would also remove it. Restore D parameter gradients before the next discriminator update.
- A broken G-step detach makes the loss have no G graph. With frozen D as well, the loss has no gradient function; a backward call raises an error. With D unfrozen, backward can produce D gradients but G still receives none. Neither is a generator update.
- The toy realism diagnostic is the fraction within Euclidean distance 0.65 of either known centre. Coverage counts centres with at least five generated samples within that radius. A one-mode generator can score well on realism but only 1/2 on coverage. These are not general image metrics.
- GAN losses are batch measurements, not distances to a known density. D≈1/2 can reflect an ideal match, an undertrained discriminator or other failures. Inspect fixed-seed outputs, mode coverage and held-out/nearest-neighbour evidence as appropriate.
