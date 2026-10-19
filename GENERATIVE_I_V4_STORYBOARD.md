# Generative Modelling I: probability detours

19 October 2026. **87 main screens, 208 main reveals, five backups.** The full route is planned for 95 minutes. The existing 80-minute route remains available by skipping the nine detour screens below; the latent explorer and beta comparison keep their four- and three-minute allocations.

## Route

| Chapter | Full route | 80-minute core |
|---|---:|---:|
| Why generate? | 7 min | 7 min |
| Learning a distribution | 5 min | 5 min |
| We already know one | 2 min | 2 min |
| From randomness to data | 7 min | 7 min |
| Inference, KL and the bound | 20 min | 15 min |
| Training through randomness | 13 min | 13 min |
| What does the trained model learn? | 17 min | 17 min |
| Learning through a discriminator | 20 min | 10 min |
| Invertible generation and diffusion handoff | 4 min | 4 min |

These are authored allocations, not rehearsal results. Use the full route for the probabilistic treatment requested here.

## ELBO detour: five minutes

1. **Can we average over a distribution we can sample?** (`elbo-reweight`, 90 s): retain the exact three-state joint model. Multiply and divide by q, show the importance weights, and recover evidence 0.48.
2. **What changes when we move the log inside?** (`elbo-jensen`, 120 s): the same weights become points on a concave log curve. Compare log E[w] = −0.734 with E[log w] = −1.595. Jensen explains the direction of the bound.
3. **Which parts can we compute and differentiate?** (`elbo-learning`, 90 s): sample latent explanations, score the observed x, subtract analytic prior KL and update both networks. This leads into reparameterization.

The existing posterior-gap animation and two-KL distinction remain between these steps. The finite toy can compute its evidence and posterior gap exactly; a neural VAE generally cannot. A change to q tightens the bound for fixed model parameters. A change to the model also changes the evidence. Beta 1 is the ELBO; other beta weights define different objectives.

## GAN detour: ten minutes

1. **What probability does the discriminator predict?** (`gan-source`, 90 s): choose a source, draw x, hide the source, predict it. G induces a distribution over its outputs.
2. **Where does the discriminator loss come from?** (`gan-logloss`, 90 s): Bernoulli source likelihood becomes the two-expectation objective. The factor of two between the balanced average and V is explicit.
3. **What would the best discriminator say?** (`gan-bayes`, 120 s): derive the source posterior with Bayes. Change the source prior and compare three outcomes. A score of one half at one outcome does not establish global agreement.
4. **How does the discriminator change as the model moves?** (`gan-density`, 120 s): move an analytically known Gaussian generator and inspect the exact Bayes discriminator on a separate probability axis. The final reveal matches the distributions.
5. **Why change the generator’s loss in practice?** (`gan-gradient`, 90 s): keep D fixed and compare loss-to-logit derivatives −D and D−1, including a checked numerical example.
6. **How is the ideal game connected to KL?** (`gan-js`, 90 s): form the midpoint distribution, average two KLs, and verify V(G,D*) = −log 4 + 2 JS as a chosen model distribution approaches the data distribution.

The existing D/G parameter-update diagrams and measured 2D GAN snapshots remain in the route. Analytical curves are labelled separately from the saved training run. The JS result concerns an optimal balanced discriminator and the original minimax value; practical non-saturating updates are not described as direct JS gradient descent.

## Linked reading

[Resources and further reading](resources.html) groups all 17 references and links back to the lecture concepts. Relevant resources also appear in slide footers, reading-mode notes, presenter notes, the PDF reference page, the [teaching guide](TEACHING_GUIDE.md) and the [executed notebook](notebooks/generative_i_lab.ipynb).

Main teaching references: [Lilian Weng on VAEs](https://lilianweng.github.io/posts/2018-08-12-vae/), [Lilian Weng on GANs](https://lilianweng.github.io/posts/2017-08-20-gan/), [Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial), [Nipun’s variational inference lecture](https://www.youtube.com/watch?v=fQ8D0-tTCBU), [Nipun’s VAE lecture](https://www.youtube.com/watch?v=bo8wceOS6C0), [PML VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb) and [MIT 6.S191](https://www.youtube.com/watch?v=R8V8CbuxryI).

The new figures are original SVG drawings. No blog raster figure is included in the release.
