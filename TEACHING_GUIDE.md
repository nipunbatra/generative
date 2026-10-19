# Teaching guide — Generative Modelling I

87 conceptual screens, 208 physical builds, 95 minutes. Five backup screens are excluded from the main route.

Full route: 95 minutes. For the 80-minute core, skip the nine screens labelled in GENERATIVE_I_V4_STORYBOARD.md; existing measured-demo allocations remain unchanged. [All linked resources](resources.html).

Controls: arrows/Space for builds; N next concept; S notes; Q discussion; O overview; P present/read. Phone opens in reading mode. Controls menu links PDF and notebook.

## 01. Generative Modelling I

Start 00:00; 20 seconds; 1 builds.

We have learned to recognize and compare inputs. Today: learn the structure and variation needed to create another plausible example. Start with the task; postpone the architecture.

**Ask:** What must a useful generator learn?

**Expected answer:** Learn a distribution over possible data, then sample from it.

**Further reading:** [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) · [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0)

## 02. Why generate?

Start 00:20; 10 seconds; 1 builds.

Chapter plan: 7 minutes. Ask: What would a useful generator have to learn?

**Ask:** What would a useful generator have to learn?

**Expected answer:** Use the following examples to develop the answer.

## 03. Recognizing a digit is one task. Drawing one is another.

Start 00:30; 45 seconds; 2 builds.

First reveal the label for a supplied image. Then ask for a new image without supplying an input image. The decoder-only language model was already generative; we return to that familiar example shortly.

**Ask:** Recognizing a digit is one task. Drawing one is another.

**Expected answer:** These models predict something about a supplied input. Generating asks for a new input-like object.

**Further reading:** [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) · [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0)

## 04. What if several answers are reasonable?

Start 01:15; 60 seconds; 2 builds.

Ask for possible completions of the authored partial sketch. Several answers can be plausible. This motivates distributions over outputs. It is an application preview, not a claim that our unconditional MNIST model performs inpainting.

**Ask:** What if several answers are reasonable?

**Expected answer:** Represent possible answers and their uncertainty instead of insisting on a single average.

**Further reading:** [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) · [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0)

## 05. What have these two generators missed?

Start 02:15; 105 seconds; 3 builds.

Replay uses the first eight training examples. Independent pixels use frequencies estimated from all 60,000 binary training images and seed 667. Ask students to predict the second panel before revealing it. Both are samplers; replay alone does not demonstrate generalization, and independent pixels discard stroke dependencies.

**Ask:** What have these two generators missed?

**Expected answer:** We need to learn structure and variation beyond replay or independent pixel frequencies.

**Further reading:** [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) · [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0)

## 06. Which properties of these sevens vary together?

Start 04:00; 60 seconds; 1 builds.

Look at slant, thickness and proportions in actual examples. Hidden causes are a modelling hypothesis. We do not claim the learned latent axes correspond to these named properties.

**Ask:** Which properties of these sevens vary together?

**Expected answer:** Several hidden factors could vary; learned latent coordinates need not correspond one-to-one to those factors.

## 07. What would count as a good generator?

Start 05:00; 70 seconds; 2 builds.

These are the same first eight fixed-seed VAE-16 decoder means. Inspect imperfections. Ask separately about plausible individual samples, coverage, and generalization. A small panel cannot prove coverage or absence of memorization. No labels were used in training.

**Ask:** What would count as a good generator?

**Expected answer:** Plausible samples, adequate coverage, and generalization beyond replay.

**Further reading:** [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) · [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0)

## 08. What does it mean to learn a distribution?

Start 06:10; 50 seconds; 1 builds.

Introduce the recurring eight-mode ring. The observations are simulated from a known mixture; the contour overlay is an eight-component Gaussian mixture fitted by EM without component labels. It is a learned non-neural density. The fit is evaluated on its training cloud, not claimed as held-out generalization. This is a conceptual preview.

**Ask:** What does it mean to learn a distribution?

**Expected answer:** Learn where probability mass belongs so new draws have the right structure and coverage.

## 09. Learning a distribution

Start 07:00; 10 seconds; 1 builds.

Chapter plan: 5 minutes. Ask: Where should probability mass go?

**Ask:** Where should probability mass go?

**Expected answer:** Use the following examples to develop the answer.

## 10. If we sample 100 times, what might happen?

Start 07:10; 50 seconds; 5 builds.

First ask for predictions with P(red)=.2, P(blue)=.5, P(green)=.3. Reveal the actual fixed-seed 100-draw histogram. Reset and use 10 draws to show fluctuation. Expected counts 20/50/30 are expectations, not guaranteed results.

**Ask:** If we sample 100 times, what might happen?

**Expected answer:** Usually near 20 red, 50 blue and 30 green, with random variation.

## 11. What do μ and σ control?

Start 08:00; 20 seconds; 1 builds.

Move mean while keeping sigma fixed, then sigma while keeping mean fixed. The same standard noise is reused during parameter changes; Fresh draws changes it. Variance is sigma squared, not sigma.

**Ask:** What do μ and σ control?

**Expected answer:** Mean translates the distribution; standard deviation sets its scale.

## 12. Is the height of this curve a probability?

Start 08:20; 40 seconds; 4 builds.

Reveal the interval shading. Density has units inverse to x and can exceed one. For a continuous variable the probability at a single exact value is zero. The curve is not a histogram of individual real numbers.

**Ask:** Is the height of this curve a probability?

**Expected answer:** No. Integrate density over an interval to get probability.

## 13. Can one Gaussian describe these eight groups?

Start 09:00; 60 seconds; 6 builds.

Begin with a single Gaussian matching population first and second moments. Toggle to the known mixture. Both views use the identical 280 points and axes. Eight clusters, not eight digit labels.

**Ask:** Can one Gaussian describe these eight groups?

**Expected answer:** A single Gaussian cannot represent separated high-density modes with low-density gaps.

## 14. Could a simple recipe already generate data?

Start 10:00; 40 seconds; 4 builds.

Use component k=0 with centre (2.4,0), and offset (0.1,-0.2), giving x=(2.5,-0.2), orally. Component labels are latent variables too. This is a real sampling recipe.

**Ask:** Could a simple recipe already generate data?

**Expected answer:** Yes. A mixture is already a probability model with a precise sampling algorithm.

## 15. Which parameters explain our observations?

Start 10:40; 60 seconds; 6 builds.

Hold the observations fixed. Start at mean zero and calculate summed log density. Advance to move the mean to .25, then .5. The exact maximum is .5. Ask students whether increasing every density without preserving normalization would define a probability model. The final build introduces maximum likelihood in general.

**Ask:** Which parameters explain our observations?

**Expected answer:** Choose parameters giving high likelihood to the observations. Here the best mean is .5.

## 16. Can we sample it? Can we evaluate its density?

Start 11:40; 20 seconds; 2 builds.

A generator can transform random samples without an easy normalized density formula. A VAE decoder likelihood is conditional; its marginal p(x) integrates over z and is usually intractable. This prepares the later GAN/flow comparison.

**Ask:** Can we sample it? Can we evaluate its density?

**Expected answer:** No. Having a sampler does not imply tractable normalized density evaluation.

## 17. We already know one

Start 12:00; 15 seconds; 1 builds.

Chapter plan: 2 minutes. Ask: Our language model already generates. How?

**Ask:** Our language model already generates. How?

**Expected answer:** Use the following examples to develop the answer.

## 18. Our decoder-only language model was generative

Start 12:15; 30 seconds; 2 builds.

Recall the Indian-name generator and next-token exercises without reteaching attention. At each step a categorical distribution supplies the next draw. The displayed words are an authored sequence, not a recorded model output.

**Ask:** Our decoder-only language model was generative

**Expected answer:** It defines a joint distribution through conditional distributions and samples sequentially.

## 19. Could we do the same for image pixels?

Start 12:45; 45 seconds; 3 builds.

The grid is schematic and small enough to read. A flattened binary MNIST image has 784 variables. Other models use image tokens rather than pixels. The factorization is exact; tractability depends on conditional models.

**Ask:** Could we do the same for image pixels?

**Expected answer:** Yes. Autoregressive image models use pixels or tokens in a chosen order.

## 20. Must every part wait for the previous part?

Start 13:30; 30 seconds; 1 builds.

Contrast sequential sampling with a latent generator. This is a question about generation structure, not a blanket speed ranking. Training an autoregressive model can still parallelize across observed sequence positions.

**Ask:** Must every part wait for the previous part?

**Expected answer:** Not necessarily. A latent generator can map a random vector to all output coordinates in one pass.

## 21. From randomness to data

Start 14:00; 30 seconds; 1 builds.

Chapter plan: 7 minutes. Ask: Why can reconstruction succeed while sampling fails?

**Ask:** Why can reconstruction succeed while sampling fails?

**Expected answer:** Use the following examples to develop the answer.

## 22. Where will our randomness come from?

Start 14:30; 60 seconds; 2 builds.

The unit Gaussian is a modeling assumption here, not something an ordinary AE automatically learns. Show one purple point. The prior is not the data distribution and is not conditioned on the particular x.

**Ask:** Where will our randomness come from?

**Expected answer:** We choose a simple, sampleable prior over latent vectors, often a standard multivariate Gaussian.

## 23. What should a neural generator do with z?

Start 15:30; 60 seconds; 2 builds.

A probabilistic decoder outputs parameters of p(x|z); drawing x can introduce additional randomness. In our Bernoulli example the image of probabilities is a decoder mean, not itself a binary sample. A deterministic GAN generator instead directly outputs x.

**Ask:** What should a neural generator do with z?

**Expected answer:** It maps a latent draw to output values or to parameters of an observation distribution.

## 24. During training we see x. Which z explains it?

Start 16:30; 60 seconds; 1 builds.

Let the missing z feel like a practical problem. We can invent a network going from x back toward a code. Do not display a VAE graph yet.

**Ask:** During training we see x. Which z explains it?

**Expected answer:** We need an inference mechanism; an encoder is one way to learn it.

## 25. Could an encoder find a useful code?

Start 17:30; 60 seconds; 2 builds.

This AE uses BCE on binary data and a 2D deterministic code. It has no KL or prior matching penalty. The displayed reconstructions are limited by two-dimensional capacity.

**Ask:** Could an encoder find a useful code?

**Expected answer:** Yes. Optimize an encoder and decoder for reconstruction; that does not yet solve prior sampling.

## 26. Reconstruct a digit. Then try a prior draw.

Start 18:30; 90 seconds; 3 builds.

Compare the observed digit encoded by the trained AE with a standard Gaussian latent decoded by that same network. Switch to the VAE using the same input and proposed prior draw. These actual outputs are not selected to make the AE look bad. Reconstruction training alone makes no promise about a chosen prior.

**Ask:** Reconstruct a digit. Then try a prior draw.

**Expected answer:** A reconstruction uses a code the encoder produces. A proposed prior may visit different regions.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 27. Will a random Gaussian code decode well?

Start 20:00; 60 seconds; 2 builds.

Compare the actual AE and VAE code scatter with the chosen standard Gaussian reference. Empty regions and mismatch are observed geometry here, not a theorem that all AEs fail. The browser can switch the measured recorded model. Prior samples are in the notes/experiment outputs.

**Ask:** Will a random Gaussian code decode well?

**Expected answer:** Not necessarily. Reconstruction alone imposes no standard-Gaussian organization on encoded points.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 28. Inference, KL and the bound

Start 21:00; 30 seconds; 1 builds.

Chapter plan: 20 minutes. From the generative model to posterior inference, KL, Jensen’s inequality and a computable evidence bound.

**Ask:** Which latent values explain this image?

**Expected answer:** Use the following examples to develop the answer.

## 29. The same model, viewed in two directions

Start 21:30; 90 seconds; 3 builds.

Start from the generative process: prior z, decoder distribution, observed x. Reverse the question after observing x: which latent values explain it? The inference arrow denotes a conditional question, not an inverse decoder.

**Ask:** The same model, viewed in two directions

**Expected answer:** Generation samples x given z. Inference asks for a distribution over z given x.

**Further reading:** [Jaan Altosaar: What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## 30. How probable is this observation?

Start 23:00; 90 seconds; 3 builds.

Use the exact three-state latent example. Prior probabilities .2,.5,.3 and observation likelihoods .1,.8,.2 yield contributions .02,.40,.06, so p(x)=.48. Normalize those contributions to obtain the posterior. In a continuous neural latent model the sum becomes an integral, generally without an exact tractable solution.

**Ask:** How probable is this observation?

**Expected answer:** Average the conditional likelihood over the prior: p(x)=.48 in this toy.

**Further reading:** [Jaan Altosaar: What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## 31. Let one encoder learn to propose explanations

Start 24:30; 90 seconds; 3 builds.

The true posterior is defined by the generative model but is generally hard to compute. q is a tractable approximation. A shared neural encoder predicts its parameters for each image; we amortize inference across images. Learned network weights are distinct from the per-image latent variables.

**Ask:** Let one encoder learn to propose explanations

**Expected answer:** qφ(z|x) approximates pθ(z|x); the same learned encoder serves many images.

**Further reading:** [Jaan Altosaar: What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## 32. What if the encoder predicts a distribution?

Start 26:00; 60 seconds; 2 builds.

The encoder outputs 2d numbers: d means and d log variances, not a sampled z directly. For batch B these arrays are B×d. Exponentiating half log variance gives positive standard deviation. Each x has its own q.

**Ask:** What if the encoder predicts a distribution?

**Expected answer:** Predict parameters of a conditional approximate posterior, then sample a latent vector from it.

## 33. Which distribution are we talking about?

Start 27:00; 90 seconds; 3 builds.

Ask which object depends on x and which has learned theta. q approximates the generally intractable true posterior p_theta(z|x); do not call q the true posterior. The decoder likelihood is not the marginal data density.

**Ask:** Which distribution are we talking about?

**Expected answer:** The prior supplies z, q infers z from x, and the decoder models x given z.

**Further reading:** [Jaan Altosaar: What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## 34. How expensive is the wrong probability model?

Start 28:30; 90 seconds; 3 builds.

Samples come from q. At each outcome compute log q minus log p, then average under q. Individual contributions may be negative; their sum is nonnegative. With q=(.8,.2), p=(.5,.5), KL=.192745 nats. Natural logs give nats. The logarithmic scoring/coding interpretation is idealized expected extra log loss.

**Ask:** How expensive is the wrong probability model?

**Expected answer:** KL(q||p) is expected extra negative log probability when data drawn from q are scored using p.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 35. Why does the direction of KL matter?

Start 30:00; 60 seconds; 2 builds.

Keep q=(.8,.2), p=(.5,.5). Swapping direction changes both the log ratio and the averaging distribution. KL(q||p)=.192745; KL(p||q)=.223144. If p assigns zero mass to an event with positive q mass, KL(q||p) diverges. KL is not a symmetric distance. In VAE inference we use the q-to-posterior direction.

**Ask:** Why does the direction of KL matter?

**Expected answer:** The first distribution chooses which outcomes receive weight in the average.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 36. Cover both groups, or concentrate on one?

Start 31:00; 60 seconds; 2 builds.

Original two-mode figure inspired by the KL comparison discussed in Lilian Weng and Eric Jang. Target p is an equal mixture of N(-2,.55²) and N(2,.55²). Compare q=N(0,2.1²) with q=N(-2,.65²). Display both KL directions using quadrature on [-12,12] with spacing .005. These are two candidates, not claimed globally optimal fits. Reverse KL can prefer a concentrated approximation in this restricted family; it does not universally choose one mode.

**Ask:** Cover both groups, or concentrate on one?

**Expected answer:** Forward KL penalizes poor coverage of target mass; reverse KL also penalizes placing approximate-posterior mass in target gaps.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 37. Can we average over a distribution we can sample?

Start 32:00; 90 seconds; 3 builds.

Return to the exact three-state model. Define w(z)=p(x,z)/q(z|x), with q positive wherever the joint contributes. Multiplying each weight by q recovers the same joint contributions .02,.40,.06. Thus E_q[w]=p(x)=.48. We have changed the averaging distribution, not the model or its evidence. For continuous z the sum becomes an integral. This identity alone does not make the logarithm of that expectation easy to optimize.

**Ask:** Can we average over a distribution we can sample?

**Expected answer:** Multiply and divide by q: p(x)=E_q[p(x,z)/q(z|x)]. The expectation is over the approximate posterior.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 38. What changes when we move the log inside?

Start 33:30; 120 seconds; 3 builds.

The weights are .05, 4/3 and .2 with averaging probabilities .4,.3,.3. First show log applied after averaging: log(.48)=-.733969. Then average the three log weights: -1.594820. The logarithm is concave, so E[log w] <= log E[w]. The lower point is the q-weighted average of the three points on the log curve; it is below the curve at the average weight. This is Jensen’s inequality. The lower expression is the Evidence Lower BOund. Equality occurs when w is constant q-almost surely, as when q is the true posterior in this toy.

**Ask:** What changes when we move the log inside?

**Expected answer:** The expectation of log weights is a lower bound on the log of their expectation: ELBO <= log p(x).

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 39. How can we improve a bound on the evidence?

Start 35:30; 90 seconds; 3 builds.

Exact discrete toy from the preceding evidence slide. Hold the generative model fixed, so log p(x)=log .48 is fixed. Improve q through three explicitly chosen distributions. The computed gap is KL(q||true posterior). The final q equals the true posterior and closes the gap. A neural model generally cannot compute this gap exactly; we can in this finite toy.

**Ask:** How can we improve a bound on the evidence?

**Expected answer:** log p(x)=ELBO+KL(q||posterior). Nonnegative KL makes ELBO a lower bound.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 40. Two KL terms. Two different comparisons.

Start 37:00; 60 seconds; 2 builds.

The gap compares q with the true posterior. Expanding the bound yields expected log likelihood minus KL to the prior. The latter is computable for our Gaussian family. For fixed θ, improving q tightens the bound; when θ changes the evidence changes too.

**Ask:** Two KL terms. Two different comparisons.

**Expected answer:** Posterior KL explains the bound; prior KL appears in its computable expansion.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 41. Why does the VAE loss have two parts?

Start 38:00; 90 seconds; 3 builds.

Now name the complete negative ELBO. The reconstruction term is an expectation under q; the KL to the unit Gaussian prior is analytical. β=1 gives the standard objective. The preceding exact toy explains why these terms occur.

**Ask:** Why does the VAE loss have two parts?

**Expected answer:** Minimizing the negative ELBO optimizes a lower bound on marginal log likelihood.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 42. Which parts can we compute and differentiate?

Start 39:30; 90 seconds; 2 builds.

For each observed x, sample z from the encoder distribution, evaluate the decoder log likelihood and subtract KL(q||prior). Approximate the reconstruction expectation with K samples; K=1 is common. Our Gaussian KL is analytic. Average the negative bound across the minibatch and update both theta and phi. The exact true-posterior KL gap is generally unavailable. Holding theta fixed tightens the bound by changing q; updating theta also changes the evidence. Improving a bound does not guarantee every stochastic update improves true marginal likelihood. Beta=1 gives this ELBO; other beta weights define a different objective. This creates the need for the following reparameterization graph.

**Ask:** Which parts can we compute and differentiate?

**Expected answer:** Estimate the reconstruction expectation, compute the Gaussian prior KL, and differentiate the negative bound with respect to both networks.

**Further reading:** [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 43. Training through randomness

Start 41:00; 34 seconds; 1 builds.

Chapter plan: 13 minutes. Ask: How does a gradient pass through a random draw?

**Ask:** How does a gradient pass through a random draw?

**Expected answer:** Use the following examples to develop the answer.

## 44. Where should the gradient go?

Start 41:34; 68 seconds; 1 builds.

Avoid claiming sampling is mathematically impossible to differentiate. Several gradient estimators exist. We want a low-variance pathwise estimator for this location-scale Gaussian family.

**Ask:** How will a reconstruction gradient reach the encoder?

**Expected answer:** We must specify how the random draw depends on encoder parameters and choose a gradient estimator.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 45. Put the randomness in a separate input

Start 42:42; 68 seconds; 2 builds.

Introduce epsilon first. Hold that draw fixed while differentiating the deterministic computation. The operation is elementwise for a diagonal Gaussian. On the second build reveal add and z in the graph.

**Ask:** Could the random draw be independent of the encoder?

**Expected answer:** Yes. Sample standard noise independently, then scale and shift it using the encoder outputs.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 46. Why does scale-and-shift give the right distribution?

Start 43:50; 102 seconds; 3 builds.

Start with standard Gaussian draws. Multiply those same draws by sigma=.5. Add mu=2. Scaling before shifting gives the desired mean mu; reversing these operations without adjustment would give sigma times mu. The identity is in distribution.

**Ask:** Why does scale-and-shift give the right distribution?

**Expected answer:** Affine transformation preserves Gaussianity and gives mean mu and variance sigma squared.

## 47. One graph, from image to reconstruction loss

Start 45:32; 68 seconds; 1 builds.

First show multiply and add. Then connect z to decoder and reconstruction loss. Ask students to redraw this before looking at backward arrows. The decoder loss depends on the observed x target, described in the notes.

**Ask:** Can you draw the actual computation graph?

**Expected answer:** Noise multiplies sigma; add mu; pass z through the decoder; compare its output distribution to the observed x.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 48. Follow one draw through the graph

Start 46:40; 102 seconds; 3 builds.

Pause before revealing the product and sum. Ask whether drawing epsilon again would yield the same z. It would generally not. The parameters stay fixed but the sample varies.

**Ask:** What latent value do these numbers produce?

**Expected answer:** 1.6. The distribution is N(2, .25), but this particular sample is 1.6.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 49. Now follow the same graph backwards

Start 48:22; 102 seconds; 3 builds.

Apply the chain rule one edge at a time. Stop at the partial derivatives before showing answers. The KL contributes additional gradients in the full objective. No gradient is needed with respect to the random-number generator.

**Ask:** Where do reconstruction gradients go?

**Expected answer:** The pathwise reconstruction derivatives are 3 for mu and -2.4 for sigma, holding epsilon fixed.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 50. Why does the encoder return log-variance?

Start 50:04; 68 seconds; 2 builds.

The scalar derivative -0.2 comes from .5 times .5 times -.8. Multiplying upstream gradient 3 gives -.6. This is still the reconstruction path only. Practical code uses log variance for unconstrained outputs and positive scales.

**Ask:** What does the log-variance parameterization change?

**Expected answer:** The chain rule adds a factor one-half sigma when differentiating with respect to log variance.

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 51. Four lines implement the same graph

Start 51:12; 68 seconds; 1 builds.

Point to the corresponding nodes on the preceding graph. The encoder learns mu and logvar. Exponentiation computes a positive standard deviation. randn_like supplies independent standard Gaussian noise. The addition and multiplication preserve the gradient path.

**Ask:** Four lines implement the same graph

**Expected answer:** 

**Further reading:** [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 52. Which draw keeps the gradient path?

Start 52:20; 100 seconds; 3 builds.

The measured PyTorch check confirms that Normal(mu,sigma).sample() does not retain this pathwise autograd connection; explicit mu+sigma*eps does, and rsample supplies the reparameterized interface. Hold eps fixed during a backward pass; draw fresh noise for a new Monte Carlo sample. Other gradient estimators exist.

**Ask:** Which draw keeps the gradient path?

**Expected answer:** Use an explicit differentiable transformation of independent noise, or rsample().

**Further reading:** [PyTorch: Pathwise derivative and rsample](https://pytorch.org/docs/stable/distributions.html#pathwise-derivative) · [Nipun Batra: Variational Inference, Part 2](https://www.youtube.com/watch?v=fQ8D0-tTCBU)

## 53. What does the trained model learn?

Start 54:00; 10 seconds; 1 builds.

Chapter plan: 17 minutes. Ask: How do reconstruction and generation change together?

**Ask:** How do reconstruction and generation change together?

**Expected answer:** Use the following examples to develop the answer.

## 54. Why not optimize reconstruction alone?

Start 54:10; 40 seconds; 3 builds.

Return to the measured AE mismatch. We are not adding a penalty merely for attractive scatter plots: the KL follows from the variational bound on marginal log likelihood.

**Ask:** Why not optimize reconstruction alone?

**Expected answer:** Reconstruction alone does not connect the inferred latent distributions to the prior used for generation.

## 55. What does “reconstruction loss” actually mean?

Start 54:50; 20 seconds; 1 builds.

The reconstruction term is not universally MSE. A Gaussian with fixed observation variance produces scaled squared error plus a constant. We threshold MNIST so BCE is a literal Bernoulli log likelihood, not a claim about original grayscale values.

**Ask:** What does “reconstruction loss” actually mean?

**Expected answer:** It is negative log likelihood under the chosen decoder observation model.

## 56. What probability did we assign to these four pixels?

Start 55:10; 105 seconds; 4 builds.

For an observed one use p; for zero use 1-p. Multiply the four probabilities or add their negative logarithms. We sum over pixels in the MNIST experiment; a mean would divide by four here.

**Ask:** What probability did we assign to these four pixels?

**Expected answer:** The sum is 1.196 nats; the mean is 0.299.

## 57. How far is this posterior from the prior?

Start 56:55; 30 seconds; 2 builds.

Move mean toward zero and sigma toward one. Curve overlap is an intuition; the displayed KL uses the exact formula, not a visual area distance. KL is asymmetric and measured in nats. The second build reveals its value.

**Ask:** How far is this posterior from the prior?

**Expected answer:** For these Gaussian distributions KL reaches zero exactly when mean is zero and standard deviation is one.

## 58. Can we calculate one KL by hand?

Start 57:25; 105 seconds; 6 builds.

Substitute the mean, variance, and log variance in separate steps. Compute the bracket, then multiply by one-half. Only after that move to mu=0 and sigma=1. Equality is the minimum of this KL term alone, not a requirement that a trained VAE ignore x.

**Ask:** Can we calculate one KL by hand?

**Expected answer:** About .818 nats; zero at the unit Gaussian prior.

## 59. Now we can assemble the VAE

Start 59:10; 85 seconds; 3 builds.

This is the first full VAE architecture. The encoder produces mu and log variance, epsilon supplies independent noise, z feeds the decoder, and two loss paths supply gradients. q for each example is compared to p; this is not a class-cluster constraint.

**Ask:** Now we can assemble the VAE

**Expected answer:** Every component solves a previously introduced problem: inference, differentiable sampling, observation modeling, and prior compatibility.

## 60. What stays when we only want a new sample?

Start 60:35; 70 seconds; 2 builds.

At generation time sample z from the prior. The decoder produces Bernoulli probabilities; sample each output pixel if a binary image is needed. A reconstruction instead conditions its inferred latent on x.

**Ask:** What stays when we only want a new sample?

**Expected answer:** The decoder and prior suffice for generation. The encoder is used for inference and reconstruction.

## 61. How well do the two models reconstruct the same digits?

Start 61:45; 25 seconds; 1 builds.

Use at most 45 seconds here. Greater latent capacity changes this run’s quality; this is not a universal benchmark. All panels are decoder probability means. Actual binary draws appeared in the opening. Held-out BCE/KL and exact architecture are in the backup and metrics manifest.

**Ask:** What did the small models actually learn?

**Expected answer:** The 2D model is useful for seeing geometry; the 16D model has more capacity but still makes imperfect outputs.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 62. What happens when we draw from the prior?

Start 62:10; 35 seconds; 2 builds.

Use the same prior vectors for the mean images and their binary draws within each model. Dimensions differ across models. These eight draws were fixed before inspection. Blurry means express uncertain Bernoulli pixels; binary draws can be noisy.

**Ask:** What happens when we draw from the prior?

**Expected answer:** 

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 63. What lives at this point in latent space?

Start 62:45; 240 seconds; 1 builds.

Spend four minutes here. Predict the decode before moving. Inspect a nearby point and its 3 by 3 neighborhood. Compare the selected location with its prior density, then draw from the prior. The evenly spaced grid is a map, not a gallery of equally probable samples. Saved-weight inference and nearest-grid fallback are separately labelled. Learned axes need not have named meanings.

**Ask:** What lives at this point in latent space?

**Expected answer:** The grid maps z to decoder probabilities; the prior tells us how frequently we visit regions of that grid.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 64. What changes between this 7 and this 2?

Start 66:45; 50 seconds; 3 builds.

The path linearly interpolates the 16-dimensional posterior means of the first two test examples. All 41 displayed decoder means were computed with the checkpoint. A two-dimensional line represents t, not a projection of the full latent space. The two endpoints and selected point stay marked.

**Ask:** What changes along a latent path?

**Expected answer:** The decoder changes along a continuous latent path; training balances reconstruction and KL terms.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 65. What changes when we train a larger latent representation?

Start 67:35; 25 seconds; 1 builds.

Read both vertical axes in nats per training example. The larger latent representation reduces reconstruction error in this run while using more KL. These are training curves. Held-out metrics remain in the appendix.

**Ask:** What changes when we train a larger latent representation?

**Expected answer:** The reconstruction and KL terms change together; neither alone measures sample quality.

## 66. What changes when we strengthen the prior penalty?

Start 68:00; 180 seconds; 3 builds.

Train with reconstruction + β × prior KL. β=1 is the negative ELBO; β≠1 changes the objective. Three new 2D VAEs, same initialization/data order/noise seeds, first 12,000 training images, 12 epochs. BCE sums pixels; KL sums latent dimensions; both average examples. Same held-out images and prior draws. These are separate trained checkpoints, not a continuous model slider. β=0 retains stochastic sampling. No claim of disentanglement or universal optimal β.

**Ask:** What changes when we strengthen the prior penalty?

**Expected answer:** Inspect reconstruction, prior samples and measured KL together. Stronger pressure can reduce information carried by z.

**Further reading:** [Nipun Batra: Variational Autoencoders](https://www.youtube.com/watch?v=bo8wceOS6C0) · [Nipun Batra: PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) · [TensorFlow: Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

## 67. Learning a distribution through classification

Start 71:00; 26 seconds; 1 builds.

Chapter plan: 20 minutes. Construct the source-classification experiment, derive its objective and Bayes optimum, then connect alternating updates, gradient choices and distribution coverage.

**Ask:** Can source classification teach a generator a distribution?

**Expected answer:** Use the following examples to develop the answer.

## 68. Must we train with that particular objective?

Start 71:26; 52 seconds; 1 builds.

The VAE used a probability model and a variational objective. GANs provide an alternative training signal. Introduce only the generator initially; no full game diagram yet.

**Ask:** Must we train with that particular objective?

**Expected answer:** No. A learned discriminator can supply adversarial feedback.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 69. What probability does the discriminator predict?

Start 72:18; 90 seconds; 3 builds.

Define the training experiment before introducing its loss. Choose source S=1 or S=0 with equal probability. If S=1, draw x from the data distribution; otherwise draw z from the noise prior and set x=G(z). This mapping induces p_G over x even when its density is unavailable. Hide S and ask D to estimate P(S=1|x). This is source classification in a constructed mixture, not a general probability that an image is real or a density p_data(x). Equal source priors matter for the later D* formula.

**Ask:** What probability does the discriminator predict?

**Expected answer:** D(x) estimates the probability that x came from the data source in this balanced classification experiment.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 70. How could generated points get feedback?

Start 73:48; 52 seconds; 2 builds.

The first view introduces real and fake inputs, the second emphasizes learning the discriminator. The score is useful for the adversarial objective; do not call it calibrated truth or general realism probability.

**Ask:** How could generated points get feedback?

**Expected answer:** Train a discriminator on the source labels of real and generated examples.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 71. Where does the discriminator loss come from?

Start 74:40; 90 seconds; 3 builds.

Use the Bernoulli conditional model P(S|x)=D(x)^S(1-D(x))^(1-S). Its log likelihood is S log D+(1-S)log(1-D). Average over the balanced source experiment. The result is V/2, where V is the sum of real and generated expectations in the original GAN objective. Multiplying by two preserves the optimum, so the implementation minimizes L_D=-V. Keep this factor explicit: an average across all real and fake examples differs by two from a sum of the two class means.

**Ask:** Where does the discriminator loss come from?

**Expected answer:** The discriminator objective is ordinary binary classification log likelihood on a balanced mixture of data and generated examples.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 72. What would the best discriminator say?

Start 76:10; 120 seconds; 3 builds.

Use exact illustrative categorical probabilities p_data=(.6,.3,.1), p_G=(.1,.3,.6). Bayes gives D*(x)=pi p_data(x)/(pi p_data(x)+(1-pi)p_G(x)); pi is the data-source prior. At pi=.5, D*=(6/7,.5,1/7). The middle bin has score .5 even though the full distributions differ. Change pi to show why an unbalanced sampling experiment changes the posterior. Under equal priors the odds D*/(1-D*) recover the density ratio p_data/p_G. In actual GANs those densities are unavailable; D learns from samples. The formula describes the population optimum, not every trained neural discriminator.

**Ask:** What would the best discriminator say?

**Expected answer:** At equal source priors, D*(x)=p_data(x)/(p_data(x)+p_G(x)); it compares the two densities at the same x.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 73. How does the discriminator change as the model moves?

Start 78:10; 120 seconds; 4 builds.

Original distribution diagram inspired by Goodfellow Figure 1 and Lilian Weng’s probability discussion. Here p_data=N(0,1), z~N(0,1), G_mu(z)=z+mu, so p_G=N(mu,1) is analytically known. Reveal p_G, then the exact Bayes discriminator on a separate probability axis. D*(x)=sigmoid(mu^2/2-mu*x). Move mu toward zero and inspect a fixed x. At mu=0 the distributions agree and optimal D is .5 everywhere. A trained network constantly outputting .5 could instead be underfit, so its output alone is not proof of convergence. These are controlled distributions and an oracle D*, not neural training snapshots.

**Ask:** How does the discriminator change as the model moves?

**Expected answer:** The optimal discriminator follows the density ratio. When both distributions match, it cannot infer the source better than the source prior.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 74. What learns during the discriminator step?

Start 80:10; 78 seconds; 3 builds.

Hold G fixed. Generate examples and detach them. Compute the summed real/fake BCE terms, then backpropagate into D and update only its weights. The displayed scores are hand-chosen teaching values; the later snapshots come from a separately recorded training run.

**Ask:** What learns during the discriminator step?

**Expected answer:** Only D parameters update. Fake examples are treated as fixed inputs for this step.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 75. Can fixed discriminator weights still pass gradients?

Start 81:28; 78 seconds; 3 builds.

Do not wrap D(G(z)) in no_grad during the G step. Its parameters can have requires_grad false while operations still track dependence on G output. This loss differs from minimizing log(1-D(G(z))) in the classical minimax formulation. Backup contains that reference.

**Ask:** Can fixed discriminator weights still pass gradients?

**Expected answer:** Yes. Fixed parameter values still define a differentiable function of the input.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 76. Why change the generator’s loss in practice?

Start 82:46; 90 seconds; 3 builds.

Write D(G(z))=sigmoid(a), where a is the discriminator logit. For minimax generator loss log(1-D), the derivative with respect to a is -D. For non-saturating loss -log D, it is D-1. At a=-4, D=.017986, so the two derivatives are -.017986 and -.982014. Both updates try to increase the data-source logit, but the latter avoids the small loss-to-logit derivative when D confidently rejects the sample. The complete generator gradient also includes the derivative of the discriminator logit with respect to its input and of G with respect to its parameters; the new objective does not guarantee useful gradients in every setting. D parameters stay fixed during this update.

**Ask:** Why change the generator’s loss in practice?

**Expected answer:** The non-saturating loss provides a larger loss-to-logit derivative when D strongly rejects a generated example.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) · [Arjovsky & Bottou: Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862)

## 77. How do we train both networks?

Start 84:16; 52 seconds; 2 builds.

Use the two buttons once each. The diagram is a teaching simulation, not a recorded GAN run. There is no guaranteed monotonic global progress from alternating gradient updates.

**Ask:** How do we train both networks?

**Expected answer:** Update D with G outputs detached; then update G using gradients through a fixed D.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 78. What do these two losses ask for?

Start 85:08; 78 seconds; 3 builds.

Sum one real and one fake negative log probability for D. The non-saturating generator minimizes minus log D(fake). D is a classical sigmoid probability of source membership under its training problem, not a calibrated measure of objective realism. A WGAN critic has a different objective and interpretation.

**Ask:** What do these two losses ask for?

**Expected answer:** D wants real scores high and fake scores low. G wants D(fake) high.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 79. How is the ideal game connected to KL?

Start 86:26; 90 seconds; 3 builds.

Return to the exact three-bin distributions. Form m=(p_data+p_G)/2 and average KL(p_data||m) and KL(p_G||m): this is Jensen–Shannon divergence. Substitute the optimal discriminator into the classical minimax value to obtain V(G,D*)=-log 4+2 JS. Change the categorical model along a chosen interpolation toward the data distribution. JS reaches zero and V reaches -log 4. This is an analytical distribution path, not a learned G update. The result assumes optimal D and the original balanced minimax objective. Finite alternating updates with the non-saturating generator loss are not literally gradient descent on this JS expression.

**Ask:** How is the ideal game connected to KL?

**Expected answer:** Under an optimal balanced discriminator, the classical minimax value differs from twice JS only by -log 4.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 80. What does alternating training do to the points?

Start 87:56; 130 seconds; 5 builds.

Advance through steps 0, 100, 500, 1000 and 2000. Both axes remain fixed. Coverage counts centers with at least ten of one thousand fixed prior samples within three component standard deviations. This threshold is a teaching diagnostic, not a calibrated benchmark. We show the single run as recorded, whether or not it collapses.

**Ask:** What does alternating training do to the points?

**Expected answer:** Inspect both the shape of generated points and their coverage; the loss alone is insufficient.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) · [Arjovsky & Bottou: Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862)

## 81. If samples look plausible, have we succeeded?

Start 90:06; 54 seconds; 2 builds.

Retain this deliberately constructed three-mode sampler to isolate the difference between plausible individual points and coverage. It is distinct from the measured GAN run on the preceding screen.

**Ask:** If samples look plausible, have we succeeded?

**Expected answer:** No. A generator can make plausible points while missing most modes.

**Further reading:** [Goodfellow et al.: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) · [Arjovsky & Bottou: Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862)

## 82. Invertible generation

Start 91:00; 15 seconds; 1 builds.

Chapter plan: 4 minutes. Ask: Could many small transformations make generation easier?

**Ask:** Could many small transformations make generation easier?

**Expected answer:** Use the following examples to develop the answer.

## 83. What if our generator were invertible?

Start 91:15; 30 seconds; 2 builds.

Move deformation slowly. The concrete map is x1=z1 and x2=z2+a*.6*sin(1.25*z1), which has an explicit inverse and determinant one. It illustrates invertibility, not a successful learned multimodal density. It does not equal flow matching.

**Ask:** What if our generator were invertible?

**Expected answer:** We can map observations back to the base distribution and account for local volume change.

**Further reading:** [Rezende & Mohamed: Normalizing Flows](https://arxiv.org/abs/1505.05770)

## 84. Where does exact density come from?

Start 91:45; 45 seconds; 3 builds.

The forward map x=2z doubles interval length. Its probability mass stays fixed, so density halves. The inverse derivative is one-half. This explains the plus log absolute inverse Jacobian convention. Ordinary generators need not be invertible; a VAE encoder and decoder are not generally inverses. This is normalizing flow, not flow matching.

**Ask:** Where does exact density come from?

**Expected answer:** Change of variables combines the base density with the inverse map’s local volume correction.

**Further reading:** [Rezende & Mohamed: Normalizing Flows](https://arxiv.org/abs/1505.05770)

## 85. Four answers to the same question

Start 92:30; 30 seconds; 1 builds.

Ask students to supply each mechanism before you say it. We have not ranked the families by universal sharpness or quality. No family automatically solves all coverage, likelihood and computation requirements.

**Ask:** Four answers to the same question

**Expected answer:** They differ in factorization, latent inference, training signal, or invertibility.

## 86. Could many small transformations help?

Start 93:00; 90 seconds; 6 builds.

These exact image files open Generative II. They come from its seeded forward Markov trajectory with fresh noise at each step, grayscale and a blue-to-purple display tint. Clipping and tint are display operations. No denoiser output is claimed.

**Ask:** Could many small transformations help?

**Expected answer:** Adding Gaussian corruption is easy to specify. Learning to undo it is the new question.

**Further reading:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 87. Can we learn to remove a little noise at a time?

Start 94:30; 30 seconds; 1 builds.

The reverse arrow poses a question. Replaying stored frames does not learn a reverse process or recover an original from pure noise. Leave the DDPM equations for Generative II.

**Ask:** Next: learning to reverse noise

**Expected answer:** That is the next lecture’s question: generation through many learned small steps.

**Further reading:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 88. Why is it a lower bound?

Start Optional; 0 seconds; 1 builds.

Distinguish the true posterior in the gap from the prior in the objective. Supports the negative-ELBO sign convention.

**Ask:** Why is it a lower bound?

**Expected answer:** KL is nonnegative, so ELBO is no larger than log marginal likelihood.

## 89. How does diagonal-Gaussian KL scale to d dimensions?

Start Optional; 0 seconds; 1 builds.

This is the implementation used in the experiment. Reconstruction sums 784 pixel losses per example; KL sums d latent coordinates.

**Ask:** How does diagonal-Gaussian KL scale to d dimensions?

**Expected answer:** Sum the independent coordinate contributions.

## 90. Classical minimax versus practical generator loss

Start Optional; 0 seconds; 1 builds.

Both target fooling D, but have different gradients. Do not silently replace the minimax G objective with non-saturating loss while claiming identical optimization.

**Ask:** Classical minimax versus practical generator loss

**Expected answer:** The non-saturating objective changes G gradients while preserving the intended adversarial goal.

## 91. What exactly was measured?

Start Optional; 0 seconds; 1 builds.

One posterior sample per test example for reported VAE metrics, seed 668. Test data never trained on. Image protocol and complete training curves are recorded. Small instructional run, not tuned state-of-the-art performance.

**Ask:** What exactly was measured?

**Expected answer:** Three locally trained MLP models with fixed data preprocessing and saved evidence.

## 92. Sources and reproducibility

Start Optional; 0 seconds; 1 builds.

The figures are original teaching constructions or measured outputs except the credited recurring photograph. The website needs no network for lecture interactions.

**Ask:** Sources and reproducibility

**Expected answer:** All claims and measurements have source or reproduction pointers.

