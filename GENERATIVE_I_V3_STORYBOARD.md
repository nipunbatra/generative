# Generative Modelling I — edition 3 teaching route

19 October 2026. 78 conceptual screens, 181 main reveals, five backup screens. Preserve the original questions and experiments while making the reason for each new object visible.

| Chapter | Minutes | Intellectual transition |
|---|---:|---|
| Why generate? | 7 | Recognize an image → represent possible images → test simple samplers |
| Learning a distribution | 5 | Samples → mass and density → likelihood; quick recall with full worked states available |
| We already know one | 2 | The decoder-only language model already provides a generative mechanism |
| From randomness to data | 7 | Generate from latents → infer codes → reconstruct well → discover prior mismatch |
| Inference, KL and the bound | 15 | Generation versus inference → marginalization → approximate posterior → KL → ELBO |
| Training through randomness | 13 | Independent noise → forward computation → chain rule → executable autograd |
| What does the trained model learn? | 17 | Likelihood and KL → full VAE → measured decoder explorer → controlled beta comparison |
| Learning through a discriminator | 10 | Alternate D and G → trace gradients through frozen weights → inspect coverage |
| Invertible generation | 4 | Change of variables → shared dog/noise sequence → learned reverse-process question |

The source of truth for individual timings is `src/pedagogy_story.py`; `TEACHING_GUIDE.md` contains the generated route and teaching prompts. The latent explorer has 240 seconds and the controlled beta comparison 180 seconds. Timing still needs a classroom rehearsal.

## New diagram sequence

1. Observed digit and an empty output frame.
2. Recognition versus generation.
3. A partial authored sketch with several plausible completions.
4. Training replay versus samples from fitted independent pixels.
5. Generation and inference arrows around the same latent/data objects.
6. Three latent explanations whose contributions sum to evidence 0.48.
7. One shared encoder proposing a distribution for each image.
8. Discrete KL as expected extra log loss, then reversal of its arguments.
9. Two Gaussian candidates for a two-mode target, evaluated in both KL directions.
10. The exact evidence bound rising as the approximate posterior improves.
11. Posterior KL and prior KL in separate, labelled equations.
12. The existing persistent reparameterization graph followed by sample/rsample autograd behavior.
13. Three trained beta models with common inputs, prior vectors and latent axes.

All diagrams are authored SVG. Reference figures were inspected for their explanatory structure; their raster artwork is not embedded. The probabilistic diagrams use semantic colors already established in the lecture.

## Evidence boundaries

The three-state posterior and ELBO identity are exact finite calculations. The two-mode Gaussian comparison uses numerical quadrature and compares two chosen candidates; it does not claim they are global optima. The completion sketch is illustrative and does not claim a trained inpainting system. The beta comparison is a single-seed, 12-epoch run on the first 12,000 training images, evaluated on all 10,000 test images. It is distinct from the original 15-epoch, full-data checkpoints.

The GAN and diffusion bridge retain their existing evidence and qualifications. The final dog frames remain byte-identical to the Generative II bridge assets; no new diffusion training is claimed.
