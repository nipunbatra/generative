# Generative Modelling I: teaching review

19 October 2026. Review and proposed revision; the lecture has not been changed by this review.

## Main finding

The lecture has a clear opening question, but the motivation needs more development. It asks whether we can make another digit and then reveals the prior, decoder and Bernoulli mean before students have encountered the problem those objects solve. The next twelve minutes review distributions. The latent-variable chapter starts at 21:00; the VAE chapter starts at 29:00.

The most useful material to bring forward comes from Nipun's own VAE lecture: a model reconstructs well, yet sampling random codes produces poor images. Plotting the encoded data and proposed random codes together explains the mismatch. That experiment creates a reason to change the model.

There are already substantial strengths to preserve: measured AE/VAE outputs, saved decoder weights, a shared forward/backward reparameterization graph, explicit evidence labels, careful treatment of Bernoulli means versus sampled pixels, alternating GAN updates, and continuity with the diffusion lecture.

## A proposed seven-minute opening

| Time | What students see | Question and purpose |
|---|---|---|
| 0:00–1:00 | The same handwritten digit first receives a label, then sits beside an empty image frame. | We can recognize this digit. What would a model need to draw another plausible one? Establish the new task before notation. |
| 1:00–2:00 | One partially hidden digit with several hand-authored, explicitly illustrative completions. | Why might more than one answer be reasonable? Motivate a distribution over possible outputs. This is an application preview; the current unconditional MNIST model is not an inpainting system. |
| 2:00–3:15 | Training-image replay beside independent-pixel samples estimated from the training set. | Replay reproduces examples but does not demonstrate generalization. Matching each pixel's frequency does not capture the way strokes fit together. Compute these baselines; do not invent measured outputs. |
| 3:15–4:30 | Several real sevens with different stroke widths, slants and proportions. | Which aspects vary together? Introduce hidden causes as a modelling hypothesis; do not label learned latent axes as these properties without evidence. |
| 4:30–5:30 | Actual saved VAE samples, including imperfect ones. | What would count as success? Plausible examples, coverage of different kinds of examples, and generalization beyond replay. A small sample grid alone cannot establish all three. |
| 5:30–7:00 | A simple verbal diagram: random numbers → learned transformation → possible image. | How do we learn this transformation from examples? Introduce probability notation only after this question has meaning. |

The exact teacher wording can remain plain: “We have thousands of handwritten digits. We know how to recognize them. Can we learn enough about their structure to draw another plausible example? And how would we tell whether the model learned anything beyond storing the training set?”

Keep MNIST as the worked example. A short application preview gives students a reason to care; switching repeatedly between faces, digits, toy points and unrelated applications would weaken the through-line.

## Priority 1: make the AE failure an experiment

Use the same input images and trained model through four reveals:

1. Reconstruct held-out images.
2. Ask students to predict what decoding standard Gaussian codes will do.
3. Reveal the recorded outputs without selecting only failures.
4. Overlay the encoded data and the proposed sampling distribution in the same coordinate system.

The conclusion is precise: reconstruction training does not require the encoded data to match a chosen prior. An ordinary AE can generate images if supplied with a suitable distribution over codes; determinism alone does not prevent generation.

Nipun's [Lecture 25, approximately 20:40–32:10](https://www.youtube.com/watch?v=bo8wceOS6C0&t=1240s) already uses this progression. Bring its prediction questions into the deck. The notebook's latent-dimension sweep also supplies a useful challenge: better reconstruction need not imply better samples from an arbitrary prior.

## Priority 2: explain why an encoder is needed in a generative model

Use two views of the same model, following the distinction developed in [Jaan Altosaar's tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial):

- Generation: draw z from p(z), then draw x from pθ(x|z).
- Inference: after observing x, ask which latent values could explain it.

Reveal the marginal likelihood as a sum over a few candidate latent explanations before replacing the sum by an integral:

`pθ(x) = ∫ pθ(x|z) p(z) dz`.

For a nonlinear neural decoder, this integral and the resulting posterior are generally intractable to compute exactly. Introduce qφ(z|x) as a tractable approximation to pθ(z|x). Then explain amortization in one sentence: a shared encoder predicts the distribution parameters for each new image, instead of solving a separate optimization problem from scratch for every image.

Keep p(z), pθ(x|z), pθ(z|x), and qφ(z|x) distinct. The current three-distribution graphic is useful, but the true posterior needs a visible place in the explanation too.

## Priority 3: bring the ELBO's purpose into the main lecture

The existing main ELBO slide introduces two requirements and names their sum the negative ELBO. The identity explaining “lower bound” is in the backup. Move a concise version into the main path:

`log pθ(x) = ELBO(x) + KL(qφ(z|x) || pθ(z|x))`.

Show the evidence, bound, and nonnegative gap as a simple vertical diagram. Then expand the ELBO into reconstruction and prior KL terms. Mark the two KLs explicitly: the gap compares q with the true posterior; the computable objective contains a comparison with the prior.

For fixed decoder parameters, improving q tightens the bound. Updating the decoder changes the model and its evidence as well. Do not animate the evidence as a fixed ceiling while claiming to show both updates.

Nipun's [Lecture 24, around 26:40–29:40](https://www.youtube.com/watch?v=fQ8D0-tTCBU&t=1600s) explains the bound through the gap. [Lilian Weng's VAE section](https://lilianweng.github.io/posts/2018-08-12-vae/#loss-function-elbo) supplies the mathematical connection. Keep the full algebra in the backup; the main lecture needs the reason for the objective.

## Priority 4: connect the signature graph to executable autograd

Preserve the existing numeric forward and backward example. Add a small comparison tied directly to the highlighted graph:

- `Normal(mu, sigma).sample()`: this draw does not retain the pathwise autograd connection to mu and sigma.
- `mu + sigma * eps`: the dependence is explicit, with eps drawn independently.
- `Normal(mu, sigma).rsample()`: the library's reparameterized sampling interface.

Nipun's [Lecture 24, approximately 8:50–12:30](https://www.youtube.com/watch?v=fQ8D0-tTCBU&t=530s) asks students to predict the gradients. Reuse that question before revealing the numbers. Keep epsilon fixed during a single forward/backward pass; refresh it for a new Monte Carlo draw. Avoid the blanket claim that stochastic computation cannot be differentiated.

Once the ELBO is assembled, revisit the same graph to add the analytical KL gradient branch. Label the current hand-chosen numbers as reconstruction-path gradients rather than the full training gradient.

## Priority 5: give the measured explorer enough class time

The timing manifest assigns the latent explorer 31 seconds at 63:27. That does not support a meaningful prediction-and-observation activity. Reserve approximately four minutes for a guided exploration:

1. Predict the decoded image at a selected point.
2. Move locally and compare the measured neighborhood.
3. Compare a familiar region with a location farther into the prior's tails.
4. Sample from the prior and compare those draws with a uniformly spaced decoder grid.

Show the selected coordinates, decoded probability image, nearby decodes, and prior location together. A regular decoder grid is a map of responses, not an equally likely sample gallery. Nearby outputs may vary smoothly without yielding interpretable or disentangled axes.

Add a small measured β comparison only after running a controlled experiment. Use the same architecture, latent dimension, split, loss reductions, training budget, seeds and evaluation draws. Compare reconstructions, prior samples, and posterior geometry. A high β may reduce information carried by z; it does not guarantee useful disentanglement. A continuous β slider would be misleading unless it selects separately trained models with clear labels.

## Notebook details to fix before reusing experiments

Inspected `/Users/nipun/git/pml-teaching/notebooks/vae.ipynb` and fetched the linked public notebook. All 80 cell sources match. Public notebook SHA-256: `0a1523ff1537c1b78a70196c8b95b2a4b91683183ea6dc1068e4a00afbecff25`.

- `VAE_loss` averages MSE across pixels and KL across latent dimensions. The effective relative weighting changes with latent dimensionality. For an ELBO comparison, define an observation likelihood and use compatible per-example reductions, typically sums over pixel/latent axes followed by a batch average. The likelihood variance also controls the scale of an MSE term.
- Several variables named `std` actually receive the returned `logvar`. Rename them before showing the code to students.
- Generic scatter and distance helpers read the raw encoder output. For the VAE that output concatenates μ and log-variance. Plot and measure μ explicitly, or explicitly sample z; do not silently interpret all encoder outputs as a latent code. The automatic grid-bound helper also needs this distinction.
- The training curve records the last minibatch loss of each epoch. Label it accurately or accumulate an epoch average.
- β = 0 does not make the stochastic VAE exactly the same as the deterministic AE: the sampling operation remains.
- For the current diagonal-Gaussian VAE, KL is analytical. Monte Carlo sampling estimates the reconstruction expectation. A single sample per example is a valid estimator; poor images alone do not establish that more Monte Carlo samples would fix training.

These are reasons to port the teaching experiments carefully, not to replace the current measured results with unverified notebook outputs.

## Priority 6: preserve GAN and diffusion continuity

Introduce the GAN with a specific question: can a learned critic provide useful feedback on generated samples? Keep the alternating algorithm and the distinction between frozen discriminator weights and gradients passing through the discriminator to the generator.

Use coverage versus plausibility as the recurring evaluation question. Do not promise that a GAN automatically solves VAE blur or that reconstruction quality alone establishes generative quality.

Retain the shared dog photograph for the closing transition. The handoff question can be: “What if generating an image were a sequence of small denoising steps?” Show forward corruption, then leave the unknown reverse process for Generative II. Avoid suggesting that adding noise by itself gives a learned generator.

## Pacing and verification

The current probability recap occupies 12 minutes. If this class already knows basic Gaussian distributions and likelihood, shorten the main recap and keep the full worked material available as an optional route. Spend the recovered time on the opening, posterior/ELBO bridge, and measured explorer. Preserve the reparameterization activity and the alternating GAN sequence.

Before calling a revision finished:

- Rehearse the actual 80-minute route with prediction pauses; assigning seconds in a manifest is not timing evidence.
- Inspect each changed reveal at presentation size, including formulas, edge labels and gradient paths.
- Test the saved-weight decoder and recorded fallback separately.
- Recompute any new baselines and β experiments; label analytical illustrations and measured results separately.
- Use consistent symbols and plain prompts. Replace generic chapter questions with questions about the exact object on screen.

## Reference coverage

Reviewed the current source, timing manifest, teaching notes and opening render; both linked articles; the local and matching public course notebook; and selected English-caption sections of all three videos. Video timestamps are approximate because captions are automatic. This was a content and sequencing review, not a fresh exhaustive visual audit or a rerun of model training.

- [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI): opening audience prediction, applications before architecture, and the transition to latent variables. Borrow the audience involvement; do not copy broad claims that deterministic AEs cannot generate.
- [Nipun Batra, Lecture 24: Variational Inference: Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU): autograd experiment and ELBO gap.
- [Nipun Batra, Lecture 25: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0): reconstruction versus sampling experiment, latent geometry, interpolation, and the β = 0 question.
- [Course notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb): useful experimental sequence, with the implementation qualifications above.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): primary reference for approximate inference and reparameterized optimization.

An optional external second-opinion call was rejected by automatic approval review because its payload contained private project details. The review above was completed directly, without transmitting those details through an alternate route.
