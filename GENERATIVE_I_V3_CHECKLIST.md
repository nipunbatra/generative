# Generative Modelling I — edition 3 checklist

19 October 2026. The 80-minute schedule is a teaching plan, not a measured classroom rehearsal.

## Motivation and sequence

- [x] Open with the task before the architecture or probability notation.
- [x] Preview multiple plausible completions using explicitly authored sketches.
- [x] Compute training-image replay and a fitted independent-pixel baseline.
- [x] Ask separately about plausibility, coverage and generalization.
- [x] Preserve the original conceptual IDs and the existing measured models.
- [x] Put the AE sampling experiment before its latent-geometry explanation.
- [x] Explain generation, inference, marginalization and amortized approximation.
- [x] Bring the ELBO gap and its interpretation into the main lecture.
- [x] Allocate four minutes to the measured latent explorer and three to the beta experiment.

## Original diagrams and KL detour

- [x] Draw new generation/inference and shared-encoder diagrams.
- [x] Work a discrete KL example using actual averaging weights.
- [x] Show KL asymmetry with the reverse numerical calculation.
- [x] Draw a two-mode target and compare two Gaussian candidates in both KL directions.
- [x] Label quadrature, illustrations, observations and measured outputs accurately.
- [x] Construct an exact three-state evidence example and close its ELBO gap.
- [x] Distinguish the posterior KL gap from the prior KL in the objective.
- [x] Keep model parameters fixed in the gap animation and explain the scope.
- [x] Preserve the shared forward/backward reparameterization graph.
- [x] Verify sample(), explicit reparameterization and rsample() in PyTorch.
- [x] Attribute the pedagogical references without copying their raster diagrams.

## Measured evidence

- [x] Train beta 0, 1 and 4 with identical architecture, initialization and data/noise protocols.
- [x] Sum pixel and latent losses before averaging examples.
- [x] Use the same held-out images and prior vectors for each model.
- [x] Keep decoder means distinct from binary samples.
- [x] Save checkpoint hashes, metrics, posterior means and all training histories.
- [x] Label the 12,000-image experiment separately from the original full-data run.
- [x] Avoid claims of guaranteed disentanglement or universal sample-quality improvement.
- [x] Execute the companion notebook against the saved evidence.

## Verification and handoff

- [x] Run the original functional checks, plus new interaction checks.
- [x] Independently reproduce new arithmetic, baseline samples and checkpoint outputs.
- [x] Inspect all reveals for header/footer overflow, broken assets and equation errors.
- [x] Check SVG text, node boundaries and edge intersections across reveals and parameter extremes.
- [x] Check presentation scaling at 1600×900 and 1024×768, plus 390px reading mode.
- [x] Confirm offline opening, browser decoder parity and recorded fallback.
- [x] Export concept, every-build and backup PDFs.
- [x] Inspect all 186 PDF states in 21 contact sheets.
- [x] Verify the complete release bundle against its SHA-256 manifest.

See `output/rebuild-qa/QA_REPORT.md` for measured check counts and scope. The edition 2 audit/checklist remain historical records.
