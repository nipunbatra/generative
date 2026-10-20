# Lecture 3: variational autoencoders

A 70-minute main route with **57 core scenes and 3 optional scenes**, each with two authored states. The lecture starts with Lecture 2's exact AE failure cloud and the same binary handwritten 3. The narrative derives the inference network from a specified generative model.

| Episode | Minutes |
|---|---:|
| Reopen the sampling question | 6 |
| Specify the generative story | 10 |
| Infer this image's code | 10 |
| Reuse the ELBO | 12 |
| Make the training path differentiable | 11 |
| One training step, then recorded model evidence | 13 |
| What did we gain? | 8 |

This includes per-image variational fitting with a frozen decoder, amortisation, the two distinct KL terms, numerical likelihood/KL contributions, explicit μ/log-variance/ε computation, independent pathwise gradient checks, an actual joint update, probability images versus observation draws and measured latent geometry. Timing is planned, not rehearsed.

## Open and run

Serve the shared series directory locally and open `/lecture3/`. The shared runtime generates the HTML, reading view, notes and PDFs from `content.js`. The existing lectures remain separate and nothing is published.

Run `lecture3.ipynb` with this lecture directory as its working directory. Data, checkpoints and `notebooks/lab.py` are bundled. No network, new packages or original repository access are needed. Requirements are Torch, NumPy and Matplotlib; the verified environment uses Python 3.12 and PyTorch 2.13.0. See the series README for installation.

The series-level notebook writer/executor at `../lecture2/scripts/make_notebooks.py` can reproduce both notebooks and their saved outputs. The individual notebook itself has no dependency on Lecture 2's Python files. Full visual reconstruction of the shared boundary figures belongs to the two-lecture evidence build; the shipped boundary SVG is already byte-identical to Lecture 2.

## Scientific conventions

- Observations are fixed thresholded binary MNIST. The same training and test snapshot as Lecture 2 is bundled. Held example: official test index90; it is illustrative and deliberately inherited.
- Prior: `N(0,I)`. Decoder: Bernoulli logits for 784 pixels. The displayed decoder probability image is different from a Bernoulli observation draw. Generation bypasses the encoder.
- Encoder: diagonal-Gaussian approximate posterior. `logvar=log σ²`; `std=exp(0.5*logvar)`. μ and logvar are computed per-image activations, while φ are shared weights.
- The posterior gap `KL(qφ(z|x)||pθ(z|x))` is distinct from the computable prior KL `KL(qφ(z|x)||p(z))`.
- Sum pixel contributions and latent-dimension KL contributions inside each image, then average examples. A one-draw reconstruction term is explicitly an estimate, while the diagonal-Gaussian prior KL is analytic.
- The held 3's chosen ε=(0.62,−1.15) gives approximately 122.262 nats reconstruction NLL plus 6.966 nats prior KL. Full-precision source values are in the fixture; presentation numbers are rounded.
- Per-image VI freezes a copied decoder and fits one free μ/logvar pair for 80 updates. It uses fresh 64-draw training estimates and a separate fixed 256-draw diagnostic. It is not claimed to find an exact posterior or to equal the amortised solution.
- The actual joint update copies the existing `vae2` checkpoint, uses eight training images and float64 SGD with η=0.000001. A fixed ε before/after diagnostic changes from about 126.017 to125.918 nats/image. Both encoder heads and decoder weights receive nonzero gradients and change. Ordinary training uses fresh noise.
- Reconstruction figures decode encoder means for visual inspection. This is not the expected reconstruction term under q. The latent scatter likewise plots means, not samples from the aggregate posterior.
- Recorded βKL=1 and βKL=8 d=2 checkpoints have the same architecture, seed and 40-epoch schedule. Their objectives differ. The optional comparison is a bounded observation of these runs, not a universal claim about sample quality.
- All model weights are existing recorded training results, replayed here. Only the stated inference/update computations are newly executed.

## Reproduction and evidence

`notebooks/lab.py` contains the full model class, loss conventions, inspected update, per-image fitting loop and independent scalar checks. The notebook includes actual outputs and figures. Optional tiny-subset fresh training is disabled by default and clearly labelled; it would not reproduce the recorded full-dataset quality.

`data/provenance.json` and `data/original-training.json` record sources, hashes and original training histories. `data/figure-manifest.json` hashes all figures. `data/fixtures.json` is the full-precision calculation source for the slides. `qa/science-checks.json` and `qa/notebook-execution.json` record executed checks. Shared browser/PDF QA is separate.

## Teaching sources

The sequence adapts Nipun Batra's [PML 2023 VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb), with the inference-first derivation made explicit. Mathematical reference: Kingma and Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). The compact diagram conventions were informed by inspection of [Lilian Weng's VAE article](https://lilianweng.github.io/posts/2018-08-12-vae/). All in-lecture topology diagrams are original redraws with the shared semantic palette; no blog raster figure is copied. The staged teaching style follows the [attention lectures](https://nipunbatra.github.io/attention/).
