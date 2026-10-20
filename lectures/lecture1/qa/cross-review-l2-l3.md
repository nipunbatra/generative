# Independent cross-review: Lectures 2 and 3

Reviewed on 20 October 2026 by the Lecture 1 author. Review only: no Lecture 2/3 files were edited.

## Visual scope

- Lecture 2: all 114 desktop presentation states, across ten contact sheets in `lecture2/qa/contacts/`, from `states-001-012.jpg` through `states-109-114.jpg`.
- Lecture 3: all 120 desktop presentation states, across ten contact sheets in `lecture3/qa/contacts/`, from `states-001-012.jpg` through `states-109-120.jpg`.
- Full-size expansion of `lecture2/qa/screenshots/036-ae-gradient-2.jpg` and `lecture3/qa/screenshots/038-pathwise-graph-2.jpg` to inspect diagram labels.
- Full-size inspection of the subsequent correction `lecture3/qa/pathwise-corrected.png`.
- Full-size inspection of all four subsequent L2 corrections: `lecture2/qa/corrected-code-three.png`, `corrected-fit-code-density.png`, `corrected-objective-says.png` and `corrected-ae-gradient.png`.
- An actual `getBBox()` check of visible inline SVG text against the enclosing viewBox in every state: L2 128 text checks and L3 102 text checks, with no clipping. Saved as `cross-svg-bounds-l2-l3.json` alongside this report. External SVG images are outside this particular automated check.

This pass checks the saved desktop slide/build renders. It does not claim a separate mobile, PDF or classroom-rehearsal review. The root agent owns export and browser QA.

## Findings sent to the owning author

| Lecture / scene | Finding | Requested correction | State |
|---|---|---|---|
| L2, 25 `code-three` | Three-decimal measured code coordinates are presented with an equality sign. | Use approximate equality for the displayed rounded vector. | Fixed by owner; independently checked in the corrected PNG. |
| L2, 47 `fit-code-density` | The expression for reconstructed probabilities uses `xhat = Gθ(znew)`, although scene 23 defines `Gθ` as outputting logits `a`. | Apply sigmoid to decoder logits when writing `xhat`. | Fixed by owner; independently checked in the corrected PNG. |
| L2, 49 `objective-says` | The earlier probability-based reconstruction loss is shown consuming raw `Gθ` logits. | Include sigmoid, or explicitly introduce and name a logits-based loss. | Fixed with explicit sigmoid; independently checked in the corrected PNG. |
| L2, 36 `ae-gradient`, both builds | The backward curves cross the text “gradient” and “compare with x”. The problem is visible in the full-size render, not only in a contact sheet. | Place labels away from curves or on the relevant nodes, while preserving the backward-path explanation. | Fixed by owner; independently checked in the corrected PNG. |
| L3, 38 `pathwise-graph`, both builds | The noise-to-sample curve crosses its own “sample ε” label; the μ/noise inputs visually merge at the same port. | Move sampling label onto the noise source and use separate input ports. | Fixed by owner; independently checked in the full-size corrected PNG. |

No other clipping, overlap, unreadable plot labelling, broken image, or unintended answer reveal was observed across the 234 saved states. This is a bounded observation from the rendered evidence, not a guarantee that every possible browser or print configuration is free of layout faults.

All findings in this cross-review are now closed. The root agent is regenerating final all-state and PDF evidence from the corrected content.

The L2 reconstruction, interpolation, prior-sampling and fitted-code-density experiments are visibly distinguished. The held-out 3 and shared code-cloud frame make the L2-to-L3 transition concrete. L3's probability images versus sampled binary images are labelled separately; the fixed axes make the AE/VAE code comparison interpretable.

## Related mathematical review

The earlier independent source review checked the L3 ELBO signs, posterior-gap versus prior-KL distinction, log-variance conversion, scalar/vector pathwise calculations, and pixel/latent sum followed by batch mean. Two corrections were requested and reported complete by the owner:

- Rounded latent substitution now uses approximate equality.
- A reconstruction-only BCE check, excluding KL, backpropagates through a fixed-noise sample and asserts nonzero gradients in the encoder mean head, encoder log-variance head and decoder. This separates the claimed reconstruction path from gradients that could come directly from the KL term.

An independent float64 calculation from saved float32 tensors reproduced the L3 KL within approximately `1.5e-7`; the displayed reconstruction-plus-KL sum differs by approximately `4.8e-6`, consistent with float32 rounding.
