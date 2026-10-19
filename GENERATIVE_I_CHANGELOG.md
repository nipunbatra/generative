# Edition 4: probabilistic ELBO and GAN detours

19 October 2026.

- Added nine original diagram screens: evidence reweighting, Jensen, computable ELBO, GAN source experiment, classifier likelihood, Bayes-optimal D, changing density ratios, generator-loss gradients and JS.
- Expanded the full route to 95 minutes while retaining the original 80-minute route as a documented option.
- Added a 17-reference resource page and contextual links in footers, reading notes, presenter notes, the guide, PDF references and notebook.
- Added three executed notebook exercises and independent probability/browser checks.
- Preserved the measured VAE/GAN checkpoints, decoder explorer, alternating-update graphs and diffusion bridge.

# Generative Modelling I — changelog

## 19 October 2026 — edition 3

- Rebuilt the opening around a generative task, multiple plausible answers, and measured replay/independent-pixel baselines.
- Added original generation/inference, marginalization, shared-encoder, discrete-KL, two-mode-KL and exact-ELBO diagrams.
- Brought posterior approximation and the ELBO gap into the main route before reparameterization.
- Connected the persistent computation graph to verified PyTorch sample/rsample behavior.
- Trained and saved a controlled beta 0/1/4 experiment, with common inputs, latent axes and evaluation draws.
- Expanded the companion notebook and executed all 15 code cells without retraining.
- Retained every original conceptual ID, the original measured evidence, GAN sequence and exact diffusion bridge images.
- Rebalanced the planned 80 minutes; 78 screens, 181 main reveals and five backups.
- Re-exported the HTML and PDFs; verification details are in the current QA report.

## 12 October 2026

- Created the separate local project at `~/git/generative`, with the supplied brief, unchecked audit checklist and slide-by-slide storyboard saved before implementation.
- Inspected the Beyond Attention source and rendered opening, plus later CLIP/VLM typography, layout, controls, reveal logic and photograph provenance. Adapted the shell to the requested exact semantic palette.
- Authored 50 conceptual screens and 80 physical builds for the 80-minute route. Separated five backup screens. Added questions, expected answers and timing notes throughout.
- Built the fourteen requested interactive ideas with useful saved static states. Added a local VAE decoder and an explicitly labelled nearest-recorded-grid fallback.
- Trained AE-2, VAE-2 and VAE-16 on 60,000 binary MNIST examples for 15 epochs, seed 667. Saved checkpoints, fixed held-out outputs, prior draws/means, latent scatter/grid, interpolations, curves, metrics and hashes.
- Fitted a full-covariance eight-component Gaussian mixture by EM to the recurring simulated cloud; kept that learned density separate from analytical mixture contours. Mode collapse and the flow deformation are explicitly illustrative; no optional GAN run is presented.
- Checked sample/gradient arithmetic, log-variance derivatives, Gaussian KL, interval probability, MLE mean, flow inverse/Jacobian and gradients through frozen discriminator parameters.
- Rendered all builds. Fixed mobile SVG overflow, diagram/axis labels, a flex-layout overlap, dense digit panel scaling and slider step alignment. Added checks for exact Gaussian/KL defaults and control endpoints. Redrew backward paths to trace the actual graph edges.
- Exported the compact PDF, every-build teaching PDF and separate backup PDF. Executed the offline notebook against the included measured outputs. Preserved diagram sources and figure/metrics manifests.
- Local delivery only; no course-site deployment or external upload was performed. The independently present `diffusion/` directory was left unchanged and excluded from this lecture's packaging.

## Major rebuild, 12 October 2026

- Preserved all 50 original concept IDs and five backups. Added nine chapter dividers and seven focused screens, for 66 main screens and 150 builds in the 80-minute plan.
- Replaced the sampling boxes with a persistent graph from observed image through encoder, scale, multiply, add, latent, decoder and loss. Forward values and red backward paths use the same geometry; the log-variance derivative and four-line implementation stay explicit.
- Enlarged the measured latent plane, selected decoder output, local neighborhood and prior plot. Added AE/VAE prior comparisons, separate reconstruction/generation panels, a 41-state interpolation control and a dedicated training-curve screen.
- Rebuilt GAN updates around one held graph with frozen/trainable weights and a gradient through fixed D. Added checked toy losses and the actual 2,000-step GAN run. Its remaining density mismatch is visible.
- Corrected MLE reveals, partial-ELBO labelling, Bernoulli arithmetic, KL pacing, and the source architecture used for autoregressive recall. Enlarged categorical/Gaussian figures and added interval, sample-count and mixture-component states.
- Reused the exact Generative II corruption frames and reversed the visual route for the closing question.
- Applied the humanizer skill to the new slide text, notes and documentation, retaining technical terms, measured limitations and provenance labels.
- Added shared numerical fixtures, independent checkpoint checks, an updated executed notebook, browser interaction tests and renewed PDF/visual QA. The initial unchecked rebuild checklist remains in the baseline snapshot.

### Figure collision pass

- Rerouted the observed-target and loss edges; separated numeric values from sampling nodes and noise arrows. Moved KL axis captions and the posterior axis label clear of every slider endpoint.
- Checked all 155 main/backup reveal states and 36 extra control states for text, stroke and node-border collisions.
- Enlarged the selected corruption frame, retained the six-frame comparison, and enlarged the closing reverse sequence. Added axes and units to the measured GAN loss history.
- Re-exported all three PDFs after these changes; final evidence is in `output/rebuild-qa/QA_REPORT.md`.
