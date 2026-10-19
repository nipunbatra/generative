# Generative Modelling I: edition 4 verification

19 October 2026. Local interactive lecture, analytical examples, linked resources, executed notebook and PDF exports.

## Scope

**87 main screens, 208 main reveal states, five backup screens.** The full route is planned for 95 minutes. Nine marked detour screens account for exactly 900 seconds; omitting them leaves the existing 80-minute core. The measured latent explorer keeps four minutes and the beta comparison three. Neither route has been timed in a classroom rehearsal.

The nine new screens cover evidence reweighting, Jensen’s inequality, the computable ELBO, the GAN source experiment, Bernoulli classifier likelihood, Bayes-optimal D, changing density ratios, generator-loss derivatives and the Jensen–Shannon connection. Existing conceptual IDs, measured experiments and the Generative II handoff remain.

## Verification results

| Evidence | Result | Scope |
|---|---|---|
| `browser-report.json` | 37 passing checks | Every main reveal and backup, overflow, assets, equations, decoder parity, offline fallback, keyboard navigation, 390px reading mode, original-topic retention and full-route timing |
| `geometry-report.json` | 265 states; no detected collisions | All reveals, slider endpoints and existing Gaussian parameter corners; SVG text, strokes, node borders and viewBox bounds |
| `pedagogy-browser.json` | 17 passing checks | Existing KL, posterior-gap and beta controls; measured explorer time; 1600×900 and 1024×768 stage bounds |
| `probability-browser.json` | 27 passing checks | Evidence/Jensen reveals, source probabilities, source-prior slider, density probe, matching distributions, logit gradients, JS interpolation, both timing routes and resource links |
| `probability-science.json` | 27 passing checks | Independent exact calculations and PyTorch autograd |
| `resource-links.json` | 6 passing checks | Local HTML paths/fragments and portable PDF citations, including all seven requested article/video/notebook links |
| `pdf-report.json` | 87 / 208 / 5 pages | Concept / every-build / backup PDFs; no extracted words outside pages |
| Companion notebook | 18 executed code cells | Saved outputs, no error outputs; optional training disabled |

The independent science checks cover evidence reweighting and Jensen in all three posterior states; constant importance weights at the true posterior; Bayes optima and strict concavity at three source priors; the JS identity and bounds across 101 model mixtures; Gaussian density-ratio curves at four means; and both generator-loss derivatives at four logits. `results/probability.json` stores the checked fixtures.

## Visual review

All **213 teaching-build and backup PDF pages** were rasterized into 24 complete contact sheets. This revision visually reviewed **all 27 new reveal states** in three focused PDF contact sheets, plus the revised reference page and the ending/backup pages. Full-size browser review covered the reweighting table, Jensen graph, computable ELBO, source experiment, Bayes table, density/discriminator plot, generator gradients and JS comparison. The resource page was inspected at desktop size and checked at mobile width.

The retained lecture states had received the previous complete contact-sheet review. Their current browser bounds and SVG geometry were rechecked in this revision. The new figure pass fixed one Jensen annotation-spacing collision, clarified the average marker and labelled the categorical probability-mass axis. Checks are complemented by visual review; they do not prove correctness at every browser, font, zoom level or possible parameter combination.

## Resources and portability

`resources.html` groups 17 references, including both Lilian Weng articles, Jaan Altosaar’s tutorial, Nipun’s two lectures and PML notebook, MIT 6.S191, primary papers, implementation references and dataset attribution. The same catalog supplies contextual reading links in slide footers, reading-mode notes, presenter notes and the generated teaching guide. The executed notebook also links the relevant references.

PDF footers link directly to their external readings. The conceptual, every-build and backup exports contain 99, 255 and 15 link annotations respectively, with no machine-specific `file:` citation links. All seven requested article/video/notebook references appear as clickable links on the PDF reference page. Link checks establish presence, valid local targets and portable annotations, not continuous availability of every remote server.

## Scientific interpretation

The finite evidence model is exact. Reweighting requires q to cover the joint’s contributing support. Jensen explains the lower-bound direction; the posterior-gap example changes q while holding the generative model fixed. Updating model parameters changes the evidence as well. The reconstruction expectation is sampled, the chosen Gaussian prior KL is analytic, and beta 1 gives the standard ELBO.

The GAN examples define a source-classification experiment. D estimates its source posterior; its output is not a data density or a universal authenticity probability. The equal-prior formula and density-ratio interpretation concern an optimal discriminator. A score of one half at one location, or from an underfit network, does not establish a matched distribution.

The displayed Gaussian generator is the analytical map G_mu(z)=z+mu, with an oracle discriminator. The categorical path toward the data distribution is a chosen interpolation. Neither is presented as neural training. The JS identity assumes optimal D and the classical balanced minimax objective. Practical non-saturating updates are not described as direct JS gradient descent. The derivative comparison is with respect to D’s logit; the complete parameter gradient also includes the discriminator-input and generator derivatives.

The saved AE/VAE and beta checkpoints, measured GAN snapshots and diffusion bridge are retained. Browser decoder parity and retained interactions pass. Their protocols and earlier independent evidence remain in `metrics/`, `results/`, the numerical reports and `SOURCES.md`. No new GAN training claim is made.

## Reproduction and release

`README.md` contains build, verification and reproduction commands. All new figures are original SVG drawings; external blog raster artwork is not distributed. The package includes source, checkpoints, recorded outputs, notebook, guide, resources and PDFs. It excludes cached training data, local environments, scratch and the separate `diffusion/` project. The packaging script checks ZIP integrity and each file against its SHA-256 manifest.

The edition 4 bundle passed ZIP integrity and file-by-file SHA-256 verification: 599 packaged files, approximately 75.0 MB.
