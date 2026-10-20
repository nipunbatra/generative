# Author review: Lecture 5

20 October 2026.

- Checked schedule products and forward mean/variance, scalar autodiff against finite differences, two algebraically equivalent reverse means, explicit posterior variance and no final-step noise.
- Executed a fresh update on four official training images; held test index 0 remained inspection-only. Replayed all 1,000 reverse steps for all 8 checkpoint samples. CPU/original-device maximum difference was 1.88e−5.
- Executed the full notebook with 14 code cells and 4 actual figures. Schedule, reverse step and complete sampler definitions are visible executable cells copied from the companion source by the notebook generator. Inspected every notebook figure.
- Independently rendered and inspected trajectory, image tuple, U-Net, latent generation and cross-attention after reviewer fixes. Three/four-image rows now reserve one horizontal row and remain within the teaching area.
- U-Net skip features enter the decoder rather than bypass it to the predicted-noise value. Input and output edges use separate attachment points.
- Latent-diffusion reveal 1 shows image→encoder training preparation; reveal 2 starts from latent noise and ends through the decoder in a generated image. Cross-attention uses consistent row-token matrix orientation.
- Simple MSE includes the declared 1/(CHW) reduction. The Gaussian reverse-kernel frame states t>1; t=1 is the deterministic mean readout for this sampler. Rounded numerical values use approximation signs.
- Independent all-state visual review and mathematical review supplied the corrections; root records full browser/PDF verification separately.

The original 2,000-update model is reused, not newly fully trained. Fresh arithmetic, one neural update and checkpoint sampling are separately labelled. Sampling quality is qualitative teaching evidence; no image-quality metric or full distribution-match claim is made.
