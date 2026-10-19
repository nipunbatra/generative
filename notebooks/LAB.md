# Generative modelling practical · 45–60 minutes

Use `lecture-lab.ipynb`, or run the equivalent scripts from the repository root. All essential exercises run on CPU and need no network after dependencies are installed. The supplied checkpoints are labelled recorded training runs.

1. **KL and evidence, 8–10 minutes.** Run the shared calculator (`node src/math.mjs`). Write both KL directions. Set q₀ to 0.2, 0.6 and 0.8 in the lecture's exact calculator. Explain why evidence stays fixed and when the bound is tight. Test a zero-probability event. State which distribution weights each expectation.
2. **Pathwise gradient, 7–10 minutes.** Differentiate `(mu + exp(logvar/2)*eps)**2` with μ=1, logvar=log(0.25), ε=−0.4. Compare autograd with central finite differences, holding ε fixed. Average 10,000 seeded draws and compare the result with the analytic expected gradient. Distinguish one draw from an expectation.
3. **VAE, 15–20 minutes.** Inspect `TinyVAE` in `models.py`. Train on binary glyphs or load `models/vae.pt`. Sum BCE over pixels and KL over coordinates, then average the batch totals. Compare a held-out observation, posterior-mean reconstruction, prior probability image and a Bernoulli sample. Move through a uniformly spaced 2D grid and explain why the grid is not prior sampling. The two coordinates have no guaranteed semantic interpretation.
4. **GAN, 15–20 minutes.** Inspect `gan_step` and replay the actual toy checkpoints. Verify that only D changes in its phase, and only G in its phase. Compare the two-mode reference with an intentionally one-mode sample set. Detach G's fake input during its update and diagnose the absent gradient. Restore the correct function. Never use `retain_graph=True` to cover up stale forwards.

## Commands

```sh
python notebooks/validate.py
python notebooks/train.py --epochs 65 --steps 1800
node src/math.mjs
```

The fresh recorded run uses 2,048 procedural seven-segment binary glyphs, ten classes, not MNIST. VAE: latent dimension 2, hidden width 128, Adam 0.001, batch 128, 65 epochs. Toy GAN: 2D latent and data, hidden widths 32, one D/G update per iteration, 1,800 iterations. A separate short image-GAN run uses the specified 32→128→256→784 generator and 784→256→128→1 discriminator. No quality or convergence guarantee follows from completing the run.

Train/test glyph distributions share the same procedural generator with different seeds. These data establish code behaviour and a small controlled example, not performance on handwritten digits. Shapes are binary with no grayscale Bernoulli ambiguity. The image GAN uses real and generated values on the same [0,1] scale.

## Deliverables

Submit your KL arithmetic, gradient calculation, one annotated VAE generation graph and a GAN phase table. Include one failure example and a diagnostic that detects it. Answer notes are in `ANSWERS.md`; consult them after attempting the exercises.
