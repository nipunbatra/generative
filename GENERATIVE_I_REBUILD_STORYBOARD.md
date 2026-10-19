# Revised storyboard

80 minutes, nine chapters. Each chapter begins with a large question.

| Chapter | Minutes | Visual argument |
|---|---:|---|
| Why generate? | 5 | Observed digits, new draws, same finite cloud versus learned density |
| Learning a distribution | 12 | Draws accumulate; interval expands; mixture gains components; likelihood rises |
| Autoregression | 4 | Reused decoder architecture; three pixel steps |
| Randomness to data | 8 | Prior point enters decoder; AE reconstructs but its chosen prior mismatches codes |
| VAE distributions | 5 | One observed 7, encoder distribution, latent draw, observation distribution |
| Training through randomness | 15 | Persistent computation graph; multiply, add, decoder; reverse exactly those edges |
| Two losses and measured models | 17 | Two failure modes; full objective; pixel BCE; KL; reconstruction, generation, latent exploration, interpolation, curves |
| Learning through a discriminator | 10 | Persistent GAN; D update, G update, numeric objectives, measured toy training, coverage |
| Invertible generation and next lecture | 4 | Deformed grid, conserved probability mass, exact Generative II dog frames, reverse question |

Keep all 50 original concept IDs and five backups. Add focused screens for Bernoulli arithmetic, AE/VAE prior samples, generation types, training curves, and measured GAN snapshots. Preserve live controls, meaningful PDF states, and offline weights/grid fallbacks. The distribution identity scales epsilon before shifting it so the final mean is mu.
