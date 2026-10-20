# Lecture 2: from generative models to autoencoders

A 65-minute main route with **54 core scenes and 3 optional scenes**. Each scene has two authored presentation states. The opening distinguishes copying, reconstruction and generation; the closing code-cloud frame is reused byte-for-byte in Lecture 3.

The lecture includes a fitted two-mode example, binary-pixel modelling, a PCA/linear-AE bridge, the same held handwritten 3, a four-pixel BCE calculation, an actual AE update, held-out bottleneck comparisons, prior decoding, interpolation and a separately fitted code density. Timing is planned, not classroom-rehearsed.

| Episode | Minutes |
|---|---:|
| What counts as a new example? | 7 |
| Fit, then sample | 10 |
| What changes for images? | 8 |
| Learn a compact representation | 10 |
| One AE update | 12 |
| Does this give us a sampler? | 11 |
| Identify what is missing | 7 |

## Open and run

The shared runtime generates `index.html`, reading mode, notes and PDFs from `content.js`. Serve the series directory locally and open `/lecture2/`. The published series is available at <https://nipunbatra.github.io/generative/lectures/>.

Open `lecture2.ipynb` with this directory as its working directory. Use the Python environment described in the series README.

The notebook is self-contained within this lecture folder: data, checkpoints and Python model definitions are bundled. No downloads, new packages or access to the old lecture repository are needed to run it. Torch, NumPy and Matplotlib are required. Saved outputs contain an executed training update and plots.

For repeatable execution of both notebooks from the series workspace:

```sh
python lecture2/scripts/make_notebooks.py
```

The script supplies a local kernel specification and records execution results under each lecture's `qa/`. Notebook cells may also be run directly in a Jupyter environment with the listed dependencies.

## Scientific conventions

- Fixed binary MNIST observations: `raw / 255 > 0.5`. Training snapshot is official train0–255; test snapshot is official test0–1999. These splits do not overlap.
- The held 3 is official test index90, inherited from the previous lecture's deliberately selected illustrative example. It is not a random or representative performance estimate.
- AE checkpoints are the existing `ae2` and `ae16` models. The recorded checkpoint training metadata and hashes are bundled. Replay is not fresh full training.
- Decoder output is 784 logits. Displayed grey images are sigmoid probabilities; they are not binary observation draws.
- BCE sums all 784 pixels per image and averages images. Reconstruction comparisons use identical binary test observations and reductions.
- The inspected step copies `ae2`, uses eight training images, float64 and SGD with η=0.0001. Both parameter groups change. The measured loss is about 121.438 →121.087 nats/image.
- The code-cloud/prior comparison keeps exactly the same axes. The Lecture 3 opening receives the same SVG bytes. The AE and VAE mean plots share these limits.
- The additional code Gaussian is fitted on **training** codes only, with a small covariance jitter. Its sample grid is a qualitative diagnostic, not a generative benchmark.
- Binary-image rasters use nearest display interpolation. Probability-image rasters use bilinear display interpolation. Underlying numerical arrays and model computations are unchanged.

## Reproduction and evidence

`notebooks/lab.py` contains the model architecture, loss terms and numerical checks. `scripts/build_evidence.py` replays checkpoints in both Lecture 2 and Lecture 3 and regenerates their figures and shared fixtures. Run it from anywhere with both lecture folders present. `scripts/prepare_data.py` is the original packaging step: it requires the old local MNIST and checkpoint sources and is not needed for ordinary notebook use.

`data/provenance.json` records data/checkpoint origins and hashes. `data/original-training.json` preserves the recorded training histories. `data/figure-manifest.json` records generated figure hashes. `qa/science-checks.json` records BCE, gradient and parameter-update checks. `qa/notebook-execution.json` records the actual notebook execution. Browser and PDF checks are produced separately by the shared runtime tooling; numerical checks alone do not establish visual quality.

## Teaching sources

The sequence adapts Nipun Batra's [PML 2023 VAE notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb), especially reconstruction versus random-latent decoding. The progressive layout follows the [attention series](https://nipunbatra.github.io/attention/). Original topology drawings use the compact local-label approach reviewed in [Lilian Weng's AE/VAE article](https://lilianweng.github.io/posts/2018-08-12-vae/); no blog image is copied into the lecture. MNIST is the source dataset. The encoder/decoder diagrams and all current explanatory compositions are original teaching redraws.
