# Generative modelling: four complete lectures

Slides lead; short, executed notebooks provide the experiments. The sequence follows the concrete distribution-fitting and debugging approach in PML 2023, with progressive visual reasoning inspired by the attention lectures.

## Open

From this directory:

```sh
python3 -m http.server 8767 --bind 127.0.0.1
```

Open <http://127.0.0.1:8767/>. Serve the files over HTTP: the modules load numerical fixtures, so opening `index.html` with `file://` is insufficient. The slide application, mathematical fonts, data and class demonstrations use local files.

- Right/left arrows: next/previous reveal.
- Shift + arrow: next/previous slide.
- N: speaker notes. O: outline. R: reset this slide. Home/End: beginning/end.
- `?reading`: every completed slide, optional extensions and expandable notes.
- `?all`: include optional extensions during presentation.
- `#slide-id/2`: open a specific reveal.
- Each lecture has editable `content.js`, figures, numerical data, a notebook, speaker notes, and two PDFs under `exports/`.
- `completed.pdf` has one completed page per slide; `all-builds.pdf` includes every authored reveal. Both include labelled optional extensions at the end.

The recommended main routes are 140, 145, 60 and 70 minutes; Lectures 1 and 2 each fill two class sessions. These include student responses and demonstrations; they are not classroom-rehearsed timings. See [the teaching guide](TEACHING-GUIDE.md) for shorter routes and demonstration choices.

## Run a notebook

Use Python 3.12 and a Jupyter environment with PyTorch, NumPy and Matplotlib. The saved notebooks have execution outputs. Open a notebook from its own lecture directory so that its local data and checkpoint paths resolve. See each lecture's README for exact rebuild commands, seeds, data conventions and the scope of the measured evidence.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

Installation needs network access; the prepared class path does not download MNIST or weights. Full fresh image-model training is an optional offline task and is distinguished from the freshly executed toy calculations, single updates and saved-checkpoint replay. The notebooks document how to provide local MNIST files for longer runs.

## Rebuild and inspect

The browser scripts use Puppeteer and Chrome. In the development checkout, Puppeteer is supplied by the parent `dl-teaching` workspace. Elsewhere, make an existing Puppeteer installation available to Node and set `PUPPETEER_EXECUTABLE_PATH` if Chrome is in a different location. No browser dependency is required to teach from the HTML or PDFs.

```sh
node scripts/scaffold.mjs
# Start the local server in a separate terminal.
node scripts/verify.mjs
node scripts/export.mjs
node scripts/package.mjs
```

Set `SERIES_URL` to use another server address. The QA and export scripts also accept lecture numbers, for example `node scripts/verify.mjs 1 3`.

The shared palette, mathematical typesetting, diagram helpers and reveal engine are in `shared/`. Figures and numerical fixtures remain lecture-local. [Maths and diagrams](MATHS-AND-DIAGRAMS.md) specifies the conventions; [QA](QA.md) records what was actually checked. `asset-manifest.json` records file hashes. Reference material is attributed in slide footers, notes and lecture data provenance.

The four-lecture series is hosted at <https://nipunbatra.github.io/generative/lectures/>. The hub links to every HTML lecture, reading view, notebook, speaker notes and both PDF editions. The earlier decks remain at their existing URLs.

For notebook execution, clone <https://github.com/nipunbatra/generative> and keep the `lectures/` directory structure: Lectures 3 and 4 import their bundled helper code and load their bundled data/checkpoints; Lecture 2 loads its three trained networks from `lecture2/models/` (in Colab it downloads them and MNIST). Downloading only an `.ipynb` file is sufficient to inspect its saved outputs, but not to run those four notebooks.
