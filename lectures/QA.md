# Verification record

Build verification recorded on 20 October 2026. These checks cover the local artifacts; hosted delivery is verified separately. Full fresh image-model training is not claimed.

## Delivered scope

| Lecture | Main slides | Optional slides | Completed PDF pages | All-reveal PDF pages |
|---|---:|---:|---:|---:|
| 1. KL and variational inference | 66 | 6 | 72 | 157 |
| 2. From generative models to autoencoders | 54 | 3 | 57 | 114 |
| 3. Variational autoencoders | 57 | 3 | 60 | 120 |
| 4. Generative adversarial networks | 53 | 3 | 56 | 113 |
| 5. Denoising diffusion probabilistic models | 60 | 3 | 63 | 126 |

Total: **308 logical slides**, **630 authored reveal states**, **10 PDFs / 938 pages**.

## Browser and visual review

Every authored state was rendered and captured at the 1280 × 720 logical slide size. All decks pass mathematical typesetting, missing-asset, content overflow, slide-margin, code-width and inline-SVG viewBox text checks. Keyboard navigation, complete/reset, notes, deep links and 390px reading mode pass. Long formulas and diagrams scroll within their reading-mode container rather than widening the page.

The primary author inspected all Lecture 1 state contact sheets, followed up on full-resolution frames and inspected PDF output. Independent cross-review covered every AE/VAE and GAN/DDPM state. It identified and resolved equations with rounded values presented as exact, decoder-logit/probability inconsistencies, arrow/label crossings, misleading GAN update labels, clipped density axes, the U-Net skip destination, the latent-diffusion sampling start and the attention matrix convention. Corrected states were inspected again.

The reports are [AE/VAE cross-review](lecture1/qa/cross-review-l2-l3.md) and [GAN/DDPM cross-review](lecture2/qa/cross-review-l4-l5.md). Browser reports are published under each lecture's `qa/`. The larger state screenshots and contact sheets are retained locally and can be regenerated with the QA scripts. Automated geometry checks support this inspection; they do not establish teaching effectiveness.

## PDF checks

Every page of both PDFs for every lecture was rasterised. Page counts, page dimensions and nonempty text pass. For every logical slide, the completed export matches the corresponding final reveal page outside the page-number footer. Every completed PDF body was also compared with its browser screenshot, allowing one pixel of global registration for subpixel rounding and a mean absolute pixel difference below 5/255. Image-heavy frames initially exceeded the unregistered threshold; full-size comparison confirmed the rounding difference and the registered check passes. Representative PDF pages were visually inspected. Reports are published at `lectureN/qa/pdf-checks.json`; rendered pages are retained locally under `lectureN/qa/pdf/` and can be regenerated.

## Notebook execution

| Lecture | Executed code cells | Saved outputs | Errors |
|---|---:|---:|---:|
| 1 | 18 / 18 | 22 | 0 |
| 2 | 9 / 9 | 16 | 0 |
| 3 | 10 / 10 | 15 | 0 |
| 4 | 9 / 9 | 12 | 0 |
| 5 | 14 / 14 | 17 | 0 |

Notebooks were executed on CPU with the existing Python 3.12 / PyTorch 2.13 environment. They bundle the small data snapshots, definitions and checkpoints required by their class paths. Full image-model training is explicitly optional and was not run anew. Exact execution reports are under each lecture's `qa/`.

## Mathematical and empirical checks

- **VI:** 2,780 independent assertions, including 2,005 numerical comparisons. Gaussian KL, derivatives, controlled-noise gradients, quadrature, direct Beta ELBO and evidence-gap identity pass. Largest comparison discrepancy is approximately 6.00e−12. The measured coin fit has a small positive residual KL; it is not labelled exact.
- **AE/VAE:** explicit event sums and batch means; binary-pixel BCE arithmetic; diagonal Gaussian KL against the library; actual parameter updates; identical AE code-cloud evidence across the lecture transition. Reconstruction-only gradients separately reach the VAE mean head, log-variance head and decoder.
- **GAN:** stable logit losses agree with scalar arithmetic; finite differences check the derivative; actual parameter changes verify D/G ownership. Saved image-generator replay differs from the original-device result by at most approximately 5.28e−6.
- **DDPM:** schedule identities, forward moments, the epsilon-parameterised reverse mean, posterior variance and the final-step zero-noise boundary pass. A real small-batch update and the full 1,000-step saved-model sampling replay run. Maximum replay difference from the original device is approximately 1.88e−5.

Per-lecture numerical fixtures, science reports, seeds and provenance record the scope of these claims. Generated samples and training traces are measured or replayed evidence; conceptual network diagrams are labelled schematics.

## Practical limits

The 60–70 minute main routes are planned budgets, not classroom-rehearsed timings. Forty-five-minute alternatives reduce scope. Rendering and mathematical verification do not establish generalisation, sample quality or optimal pedagogy. Hosted-delivery checks, larger model training and classroom rehearsal are separate from this build verification.
