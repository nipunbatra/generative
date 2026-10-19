# Generative Modelling I — storyboard before implementation

Central question: How can we learn to turn simple randomness into valid data?

50 conceptual screens; 80 planned physical builds; 80 minutes. Build counts include the first visible state. Questions precede mechanisms. Each row has one primary visual; mathematical answers are hidden until their build. Section timings include discussion and demonstrations.

| Screen | Minutes / section | Question and visual construction | Builds |
|---|---|---|---:|
| 01 | 00–06 Why generate? | From randomness to data — cloud, learned map, digits | 1 |
| 02 | | What did our earlier models produce? Image/class, text/intent, CLIP/match | 2 |
| 03 | | Could we produce another digit? Observed digits, then measured prior samples | 2 |
| 04 | | What does learning a distribution mean? Data cloud and model contour | 1 |
| 05 | 06–18 Distributions | What happens in 100 draws? Category probabilities then sampled histogram | 2 |
| 06 | | What do mean and spread control? Gaussian sliders and fixed standard noise | 1 |
| 07 | | Is curve height probability? Point versus shaded interval | 2 |
| 08 | | Can one Gaussian capture eight groups? Same cloud, switch density overlay | 1 |
| 09 | | Can a recipe generate data? Sample component, sample offset, return point | 1 |
| 10 | | Which distribution fits observations? Likelihood curve and worked density | 2 |
| 11 | | Is sampling the same as density evaluation? Two separate abilities | 1 |
| 12 | 18–24 Familiar generator | Was our language model generative? The / dog / ran and factorization | 2 |
| 13 | | Could we do this for pixels? Raster sequence, partial grid | 1 |
| 14 | | Must every part wait? Sequential path versus one latent transformation | 1 |
| 15 | 24–32 Latents | Why do two sevens differ? Real sevens; motivating factors, no disentanglement claim | 1 |
| 16 | | Where does randomness live? Chosen Gaussian prior and one sampled point | 2 |
| 17 | | What does the generator do? z to learned map to x | 1 |
| 18 | | Which z explains this x? Inverse inference question, no answer yet | 1 |
| 19 | | Could an encoder find a code? Ordinary AE, measured reconstruction | 2 |
| 20 | | Can we sample the AE? Measured code scatter, prior region, recorded decoded samples | 2 |
| 21 | 32–49 VAE | What if the encoder predicts a distribution? Means and log variances; shape labels | 2 |
| 22 | | Which distribution is which? Prior / approximate posterior / likelihood | 1 |
| 23 | | How does the gradient reach the encoder? Opaque stochastic node | 1 |
| 24 | | Can randomness enter independently? Standard epsilon, then deterministic transform | 2 |
| 25 | | Why does the transform work? Same fixed noise, scale and shift | 2 |
| 26 | | Can we draw the computation graph? Multiply/add, decoder, reconstruction loss | 2 |
| 27 | | What z do these numbers produce? 2, .5, -.8; expression; 1.6 | 3 |
| 28 | | Where do reconstruction gradients go? Upstream 3; partials; 3 and -2.4 | 3 |
| 29 | | What does log variance change? PyTorch code; derivative .5 sigma epsilon | 2 |
| 30 | 49–59 Objective and measured demo | Why not reconstruction alone? Two competing requirements | 1 |
| 31 | | Why two loss terms? Expected negative log likelihood, then KL | 2 |
| 32 | | What does reconstruction loss mean? Gaussian versus Bernoulli observation model | 1 |
| 33 | | How far is q from the prior? KL sliders and overlapping densities | 2 |
| 34 | | Can we calculate KL? Inputs/formula, then .818 nats and zero case | 2 |
| 35 | | Can we assemble the VAE now? Complete graph with two losses | 1 |
| 36 | | What stays at generation time? Training path, then prior-to-decoder | 2 |
| 37 | | What did our small models actually learn? Measured d=2 and d=16 reconstructions/samples | 1 |
| 38 | | What lives at this point? Clickable measured 2D decoder plane | 1 |
| 39 | | What changes along a path? Measured interpolation, then recon/KL curves | 2 |
| 40 | 59–71 GAN | Could a learned critic supply feedback? Generator alone and mixed points | 1 |
| 41 | | How do we judge generated points? Introduce discriminator | 2 |
| 42 | | What learns during the D step? Detached fake samples and D gradients | 1 |
| 43 | | Can fixed D weights still pass gradients? Through D to G; non-saturating loss | 2 |
| 44 | | How do we train the pair? Alternating D/G controls; reference minimax in notes | 1 |
| 45 | | Is realism enough? Same eight modes versus illustrative three-mode coverage | 2 |
| 46 | 71–75 Flows | What if the map is invertible? Continuous grid deformation and inverse | 2 |
| 47 | | Where does exact density come from? Change of variables and local area | 1 |
| 48 | 75–80 Cliffhanger | Four answers to one question — concise comparison | 1 |
| 49 | | Could many small transformations help? Clean photo, slight noise, heavy noise, almost noise | 4 |
| 50 | | Could we learn the reverse? Reverse arrow; next lecture title; stop | 1 |

Backup: ELBO identity and gap; diagonal KL derivation; minimax versus non-saturating objective; experiment protocol and full metrics; sources. These are not part of the 80-minute route.

## Implementation and reference mapping

- Primary shell: `dl-teaching/attention-followups/transformer-outputs/src/lecture.css`, `lecture.js`, `story.py` (Beyond Attention).
- Later shell: `attention-followups/from-clip-to-vlm/src/clip-theme.css` and `beyond-token-generation/src/lecture.css`: warm paper, Avenir Next, stable 1280×720 stage, 40px question headings, large figures, restrained footer controls, notes and overview.
- Adapt these patterns into a standalone vanilla HTML/JS build with local KaTeX. No React runtime is needed for the existing slide-mode pattern.
- Palette is the user's exact semantic palette. Motion only follows an explicit teaching action.
- Photograph: existing Vision I / CLIP Newfoundland image, with original provenance retained.
- Recurring 2D example: seeded eight-Gaussian ring throughout density fitting, generators and mode collapse. One Gaussian and known mixture are explicitly analytical/illustrative, not claimed as a trained GAN.
- Experiments: reproducible binarized MNIST AE-2, VAE-2, VAE-16. Store weights, seeds, data hashes, train curves, held-out outputs and browser decoder weights. No retouching.
- PDF: both final-state conceptual slides and every main physical build, plus separate backup.
