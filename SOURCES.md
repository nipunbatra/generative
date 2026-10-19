# Sources, evidence and visual continuity

Primary sources checked 12 October 2026. The lecture follows the supplied brief; it does not claim to replicate paper benchmark results.

## Scientific references

- Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): approximate posterior, reparameterization and variational objective. The Gaussian worked numbers are independently checked in `experiments/verify_math.py` and `metrics/analytical-checks.json`.
- Goodfellow et al., [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661): discriminator/generator game and the non-saturating generator alternative. The diagram distinguishes parameter freezing from stopping input gradients. A local autograd check verifies this distinction. The added 2D GAN experiment is one locally measured run; it does not establish general convergence or reproduce a paper benchmark.
- Rezende & Mohamed, [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770): invertible transformations and density calculation. The main equation uses the inverse Jacobian with a plus sign. The smooth shear is an authored exact invertible example, not a learned flow. Normalizing flows are not identified with flow matching.
- [MNIST](https://yann.lecun.com/exdb/mnist/), LeCun, Cortes & Burges. Public files downloaded from the [CVDF mirror](https://storage.googleapis.com/cvdf-datasets/mnist/). SHA-256 hashes for every downloaded file are in `metrics/mnist.json`. Fixed thresholding is explicit, so the Bernoulli likelihood corresponds to binary observations.

## Reused image

`images/newfoundland.png` copies the exact preprocessed Vision I input image from `/Users/nipun/git/attention-vision1/figures/vision1/model-input.png`. It is the same recurring Newfoundland used in the CLIP recap. SHA-256: `69fb20ff4108c52a57391669dbe1f28d30b086f3c22afa52917bd5586a0be4f5`.

Original image: `newfoundland_31.jpg`, 500×334, Oxford-IIIT Pet. Original SHA-256: `617dc6f6dcd26c69303b54985eadd5c75bc07b8ba22cdb1ea511307096098d0a`. Dataset attribution: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012. [Dataset and image provenance](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original image owners retain copyright. The cropped display and corruption derivatives retain this attribution and share-alike license.

The teaser reuses the exact saved Generative II frames at t = 0, 100, 300, 600, 900 and 1000. That source uses a fresh-noise Markov chain, seed 667, 1,000 linearly spaced beta values from 0.0001 to 0.02, and grayscale input scaled to [−1,1]. Only display values are clipped, then tinted from blue toward grey-purple. Frame hashes and source pointers are in `results/bridge-provenance.json`. The reverse order is an illustrative question, not a learned denoiser or a recovered image.

## Visual reference mapping

Read-only references:

- `/Users/nipun/git/dl-teaching/attention-followups/transformer-outputs/`: the **Beyond Attention / From Attention to Applications** slide-mode source. Its Avenir hierarchy, 1280×720 stage, restrained controls, question bands and presenter support guided this lecture. The reference opening was inspected as a rendered image.
- `/Users/nipun/git/dl-teaching/attention-followups/from-clip-to-vlm/src/clip-theme.css`: later warm-white shell and simple teal/blue semantic diagrams.
- `/Users/nipun/git/dl-teaching/attention-followups/beyond-token-generation/src/`: self-contained HTML, progressive reveals, notes, provenance and explicit recorded fallback approach.
- Existing Vision I / CLIP provenance for the Newfoundland photograph, including `attention-followups/clip/output/vit-recap-evidence.json` and `figures/ATTRIBUTION.md`.

The requested colors take precedence: observed data #3478E5; randomness #8B5BB5; learned transforms #178F82; distribution parameters #B98224; loss/gradient #D45555; structure #30343B; paper #FAFAF8. Forward arrows use data/randomness/learned-transform colors; reconstruction-gradient routes use dashed coral.

No reference lecture or its navigation was changed. The new diagrams are reproducible authored SVGs. The decoder-only recall reuses `transformer-outputs/figures/decoder-use.svg` as `figures/attention-decoder.svg`. It has causal self-attention, an MLP and a vocabulary head. The references are not screenshots or image-generated equations. KaTeX is bundled locally under its MIT license in `vendor/katex/LICENSE`.

## Evidence categories

- **OBSERVED DATA**: real MNIST test digits after the declared binary threshold; real recurring pet photograph.
- **MEASURED**: newly trained AE/VAE outputs, model weights, loss curves and held-out metrics.
- **MEASURED FIT on ILLUSTRATIVE DATA**: EM mixture fitted to the fixed balanced toy cloud.
- **ILLUSTRATIVE**: authored model diagrams, stratified toy cloud, deliberate mode restriction and invertible shear. No fake GAN result is hidden in these illustrations.
- **ANALYTICAL / EXACT**: probability formulas and checked arithmetic. Rounded displays are labelled as approximate.

The prior, approximate posterior and decoder likelihood remain separate objects. Decoder means are distinguished from sampled binary observations. Latent factors are motivations, not promised semantic axes. The VAE's marginal density is not presented as tractable just because its conditional likelihood is. The discriminator score is not asserted to be calibrated truth.

## Rebuild measurements and references

`results/rebuild-mnist.json` links all new AE/VAE panels and 41-point interpolations to unchanged checkpoint hashes. `experiments/rebuild_evidence.py` creates these panels and records the 2D GAN experiment. `results/toy-gan.json` records the architecture, seed, optimizer, snapshots, losses and diagnostic convention. The display keeps fixed axes and clips points outside that view; coverage is calculated on all 1,000 diagnostic samples.

The [TensorFlow CVAE tutorial](https://www.tensorflow.org/tutorials/generative/cvae) was inspected for the requested latent-grid visual reference. The lecture's displayed digits, decoder weights and training results come from this repository's PyTorch runs.

The reparameterization values are an exact scalar teaching example, not activations from the pictured MNIST encoder. Its observed 7 is reused to identify the training route. All reported reconstruction gradients assume upstream derivative 3; the full objective also adds KL gradients.

The Bernoulli calculation uses natural logs and sums four pixel terms. The GAN calculation sums one real and one fake BCE term. The normalizing-flow equation uses the inverse Jacobian with a plus sign. The scale-and-shift sequence multiplies epsilon by sigma before adding mu; reversing those operations would change the mean unless the shift were adjusted.


## Edition 3 teaching references — inspected 19 October 2026

- [Lilian Weng, From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/): inspected the KL-direction, reparameterization and Gaussian-VAE diagrams. Our two-mode comparison, inference graph and ELBO figures are newly authored SVG with different layouts and explicit computed examples. The blog credits Eric Jang for its KL comparison and Kingma's workshop talk for its reparameterization figure; those raster figures are not included here.
- [Jaan Altosaar, What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial): generation versus inference, the neural-network/probabilistic perspectives, and amortized inference.
- [Nipun Batra, Lecture 24: Variational Inference: Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU): sample/rsample autograd discussion around 9–12 minutes and ELBO gap around 27–30 minutes.
- [Nipun Batra, Lecture 25: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0): reconstruction versus random-code sampling and latent geometry around 21–32 minutes; beta discussion later in the lecture.
- [Nipun's PML notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb): experimental teaching sequence. Public and local sources were compared and all 80 cell sources matched. The new experiment uses explicitly summed BCE/KL reductions and the existing local Model implementation; it does not copy the notebook's dimension-dependent loss normalization.
- [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI): audience prediction and application motivation before architecture.

Video content was inspected through selected automatic-caption sections; timestamps are approximate. No video frames are embedded. The review is recorded in `GENERATIVE_I_PEDAGOGY_REVIEW.md`; implemented changes and timing are in `GENERATIVE_I_V3_STORYBOARD.md`.

The new probability fixtures, finite-model bound and sampling behavior are independently checked by `experiments/verify_pedagogy.py`. The Gaussian KL comparison uses numerical quadrature over [-12,12] with spacing .005; the two candidate approximations are not asserted to be global optima. The beta experiment has its own data hashes, checkpoints and protocol. No disentanglement or benchmark-quality claim is made.


## Probabilistic ELBO and GAN detours: 19 October 2026

The [linked resource index](resources.html) is the reading entry point. `src/resource_catalog.py` supplies the same links to slide notes and the teaching guide.

- [Lilian Weng, From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/): inspected the article and GAN diagram. Its discussion of distributions, optimal D, JS and training failures informed the teaching route. The lecture uses original figures and independently computed examples.
- [Goodfellow et al., original GAN paper](https://arxiv.org/abs/1406.2661): Sections 3 and 4.1, Algorithm 1 and Figure 1 support the classification interpretation, alternating updates, non-saturating objective, optimal D and classical JS identity. Equal source priors and the optimal-discriminator assumption are explicit. The numerical derivative comparison is with respect to the discriminator logit; it is not a guarantee about the full generator gradient.
- [Kingma & Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): the ELBO and its reconstruction/prior-KL form. The importance-weight and concave-log figures use the lecture’s existing exact finite example, not copied paper artwork.

`experiments/verify_probability.py` independently verifies evidence reweighting, Jensen, Bayes optimality under three source priors, the JS identity on 101 model mixtures, Gaussian density-ratio curves and loss derivatives in PyTorch. `results/probability.json` stores the exact fixtures. The continuous Gaussian generator is the analytical map G_mu(z)=z+mu; its displayed D* is an oracle calculation. The categorical interpolation toward the data is also analytical, not a saved training trajectory.

The resource index also links [Arjovsky & Bottou](https://arxiv.org/abs/1701.04862) and [Wasserstein GAN](https://arxiv.org/abs/1701.07875) as optional further reading. WGAN derivation and training are outside this lecture. All diagrams from the external articles remain external references; their raster files are not packaged.
