# Generative Modelling I: verification

Verified locally on 19 October 2026. This report describes the new lecture in this folder.

## Content and teaching

- 60 core scenes, 145 core reveal states, and three optional appendix scenes with six states.
- Both the 120-minute and 90-minute routes retain the required KL, ELBO, reparameterisation, VAE and GAN material. The difference is discussion and demonstration time.
- Each scene has a visible **Say** line. Each final reveal labels its **Takeaway**. Pressing N opens a fuller explanation; Q opens a question with an answer. The saved reading edition includes all of these.
- The wording was revised with the humanizer skill. Equations, numerical fixtures, source attribution and distinctions between measured and illustrative content were retained.
- The visual reference was the local **Beyond Generation: Models That Make Decisions** deck. Its rendered diagrams informed the revised encoder branches, shared discriminator, conditioning paths and forward/backward explanations. This is a description of the design work, not an independent rating of visual quality.
- The sampling figures distinguish a deterministic generator output, decoder probabilities and a Bernoulli observation draw. The reparameterisation builds introduce noise, scaling and shifting in order. The conditional GAN returns to the opening shoe example. The satellite scene enlarges one August/January comparison rather than shrinking the complete paper figure.

## Mathematical and model checks

`math-checks.json` records passing checks for the numerical fixtures, support handling, the ELBO gap identity at 101 q values, entropy/cross-entropy, fixed-noise finite differences, GAN saturation derivatives, seeded randomness, MLE variance and Gaussian KL at the prior.

`lab-checks.json` records 14 passing PyTorch checks. These cover tensor and target shapes at batch sizes 3 and 17, loss reductions, encoder-free prior sampling, the two GAN update boundaries, gradient flow through a frozen discriminator and the deliberately detached generator failure. The two-pixel loss agrees with the shared JavaScript fixtures.

`decoder-parity.json` compares 12,544 decoder probabilities with the PyTorch export. The maximum absolute difference is about 2.40e-7, below the 1e-5 threshold. The notebook has eight executed code cells with no recorded execution errors.

The mathematical source is `src/math.mjs`. The neural runs are recorded in `assets/recorded/provenance.json`; their model weights are in `models/`. Interface values come from these calculations and recordings, not authored loss curves.

## Browser and visual checks

`browser-checks.json` records every one of the 151 reveal states, with no JavaScript/resource errors or detected layout collisions. Checks cover image loading, content boundaries, the visible speaking cue and separation from the source footer.

The interaction checks cover KL support infinity and equality, keyboard focus in range controls, fixed epsilon while changing parameters, deterministic reset, checkpoint stepping, stopping playback on scene changes, compressed-route timing, scene jumps, deep links, desktop scaling and operation after requests are blocked following a warm load. Additional checks verify GAN phase defaults/reset and pixel-for-pixel agreement between the reconstruction canvas in presentation mode and the all-build export.

Every completed scene was inspected in the six contact sheets. The revised architecture, likelihood, conditional GAN, gradient and loss-curve figures were also inspected individually. Exported mathematical pages were rasterised for visual inspection. Automated bounds checks do not by themselves establish pedagogical quality.

`reading-checks.json` records the 390-pixel reading view: 63 scene sections, no page-width overflow and no reported errors. The saved notes contain 63 note sections, no broken HTML images and no remaining canvas elements.

## Export checks

`pdf-checks.json` records:

| File | Pages | Contents |
|---|---:|---|
| `completed-scenes.pdf` | 64 | 60 core scenes, three appendix scenes, one complete ELBO derivation reference |
| `all-builds.pdf` | 151 | Every core and appendix reveal state |
| `recap.pdf` | 2 | Notation, objectives, reductions and training code reminders |

The lecture PDFs preserve selectable text, source hyperlinks and internal section destinations. All 63 completed teaching scenes and all 151 reveal pages include speaking cues. The extra derivation reference page does not add a separate cue. The native SVG figures are also exported under `figures/`.

## Evidence boundaries

The local data are procedural binary seven-segment glyphs, not MNIST. The opening MNIST examples are published AEVB figures. The small VAE produces visibly imperfect reconstructions and prior samples; the short image-GAN run is not a convergence result. The two-Gaussian toy GAN uses a defined geometric coverage measure, not a general image-quality score.

The sleep traces are labelled illustrations of a proposed application. Generated satellite images are possibilities, not measurements or forecasts. Published figures retain their individual source attribution and rights information in the asset manifest. No site was deployed or lecture published during this task.
