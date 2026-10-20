# Independent rendered review: GAN and diffusion lectures

20 October 2026. Reviewer: the AE/VAE author, independently inspecting the rendered GAN/DDPM material. Read-only for Lecture 4 and Lecture 5; fixes belong to their author. This report separates observed rendered evidence from requested changes and subsequent verification.

## Coverage

- Lecture 4: all ten contact sheets, covering all 113 authored states, including optional states.
- Lecture 5: all eleven contact sheets, covering all 126 authored states, including optional states.
- Full-resolution follow-up on `two-players/1`, `toy-vote/1`, `unet-shape/2`, `cross-attention/2`, and `latent-diffusion/2` confirmed the issues below.
- Root had already identified seven Lecture 5 image layouts with height overflow. Their initial rendered images visibly overlap headings/prompts. These are recorded separately from the additional findings below and require a fresh rendered pass after repair.
- This is a visual/pedagogical review, not an independent execution of the GAN/DDPM notebooks or a claim of classroom rehearsal.

Evidence is in `lecture4/qa/screenshots/`, `lecture4/qa/contacts/`, `lecture5/qa/screenshots/` and `lecture5/qa/contacts/` under the shared series directory.

## Findings delivered to the author

| ID | Rendered evidence | Finding and requested action | Current verification status |
|---|---|---|---|
| R1 | L4 `two-players/1–2`; screenshot `005-two-players-1.jpg` | Both modules say their parameters are “updated” while the terminal node says “G loss”. This implies a simultaneous update before alternating phases are introduced. Use neutral learned-parameter labels in the overview; reserve updated/fixed labels for explicit phases. | Resolved: neutral learned θ/ω labels and feedback overview confirmed in the fresh rendered diagram. |
| R2 | L4 `toy-vote`, `g-only` initial state, `toy-progress` initial state; screenshot `004-toy-vote-1.jpg` | Initial generated KDE peaks are visibly clipped at the top of the density plots. Keep fixed comparison axes, but make the range include the peak or explicitly communicate clipping. | Resolved: complete KDE peaks visible in toy-vote, g-only initial, and toy-progress initial; legends sit above the data. |
| R3 | L4 `g-only/2` | The final generated KDE is almost absent inside −5…5. If outputs moved outside the plotted window, annotate their range or out-of-window mass. Students otherwise see a disappearing curve rather than the actual exploit of the fixed critic. | Resolved: before/after axes fixed at [−30,30]; final state explicitly reports 56.6% outside the former [−5,5] window. |
| R4 | L5 `unet-shape/1–2`; screenshot `032-unet-shape-2.jpg` | “Skip features” goes directly from encoder to the predicted-noise value node, bypassing the decoder. Match the actual checkpoint topology: feed the skip into a decoder/merge computation before the output. | Resolved: fresh rendered U-Net sends skip features into the decoder. Shared paper-coloured label halo keeps the curve from crossing the glyphs. |
| R5 | L5 `latent-diffusion/2`; screenshot `056-latent-diffusion-2.jpg` | The text explains generation from latent noise, but the visible main path still starts from image→encoder. Use separate training-preparation and sampling states; the sampling diagram should start with latent noise. | Resolved: first state shows training preparation; second starts with latent noise, then reverse diffusion, decoder, new image. |
| R6 | L5 `cross-attention/1–2`; screenshot `055-cross-attention-2.jpg` | Q=WQ h, K=WK c, V=WV c is shown beside softmax(QKᵀ/√d)V, which normally assumes tokens in rows. Use Q=hWQ, K=cWK, V=cWV or explicitly supply a consistent alternative orientation. | Resolved: fresh formula uses Q=h_image W_Q, K=cW_K, V=cW_V, consistent with token-row attention. |

## Pedagogical observations

The scalar GAN chain is strong: one fixed real/fake example leads through logits, both classification terms, a literal discriminator update, recomputed values, a generator derivative and a separate literal generator update. The freeze-versus-detach comparison and actual parameter-change checks make update ownership concrete. Loss curves are appropriately paired with coverage diagnostics, not presented as sample-quality certificates. The held 3 and the change from binary VAE data to greyscale GAN data are explicitly identified.

The diffusion scalar chain similarly exposes the forward sample, prediction, residual, scalar loss, derivative and update before the tensor implementation. It distinguishes variance from standard deviation, α from cumulative α-bar, known training noise from the network input, and a clean-image estimate from a one-step reverse mean. The explicit final-step noise boundary is valuable. The training/posterior/ordinary-sampling distinctions and the simplified-MSE-versus-variational-objective scope are stated carefully.

The reported diagram and image-layout issues have been corrected and visually rechecked. Remaining slides in the inspected contact sheets did not reveal additional blocking visual defects; that is bounded by this rendered review, not a universal guarantee of instructional effectiveness.

## Follow-up verification

Completed after the final 20 October 2026 rerender (screenshots refreshed at approximately 15:15 local time). R1–R6 are resolved as recorded above. Full-size corrected screenshots inspected include L4 `005-two-players-1`, `004-toy-vote-1`, `029-g-only-1/2`, and `038-toy-progress-1`; L5 `032-unet-shape-2`, `055-cross-attention-2`, and `056-latent-diffusion-1/2`.

Also inspected both builds of all seven previously overflowing L5 image scenes at full resolution:

- `002-show-trajectory`
- `004-forward-film`
- `015-image-forward`
- `019-noise-levels`
- `022-noise-not-image`
- `037-estimate-image`
- `046-reverse-replay`

All fourteen corrected image states keep images, labels, explanatory captions and question/answer text within their intended regions. The held image, noise tensors, prediction and generated trajectory retain visible provenance and role distinctions. No remaining blocking visual issue was found in this follow-up. This review covers rendered slides; automated geometry, mathematical execution, mobile and PDF checks remain separately recorded by the responsible reviewers.
