import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
for(let n=1;n<=5;n++){
 const dir=path.join(root,`lecture${n}`);fs.mkdirSync(dir,{recursive:true});
 fs.writeFileSync(path.join(dir,'index.html'),`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#FAFAFB"><link rel="icon" href="data:,"><title>Lecture ${n} · Generative modelling</title><link rel="stylesheet" href="../shared/katex/katex.min.css"><link rel="stylesheet" href="../shared/deck.css"><script src="../shared/katex/katex.min.js"></script><script type="module" src="../shared/deck.js"></script></head>
<body><main id="lecture" aria-label="Lecture ${n}"><p class="loading">Preparing the lecture…</p></main><nav class="controls" aria-label="Lecture controls"><span id="position"></span><button data-action="prev" aria-label="Previous reveal">←</button><button data-action="next" aria-label="Next reveal">→</button><button data-action="reset">Reset</button><button data-action="finish">Complete</button><button data-action="notes">Notes</button><button data-action="scenes">Scenes</button><a id="read-link" href="?reading">Read</a><a href="../">Series</a></nav><aside class="panel" id="notes" aria-label="Speaker notes" hidden></aside><aside class="panel" id="overview" aria-label="Scenes" hidden></aside><noscript>This lecture requires JavaScript. Use the exported PDFs for a static version.</noscript></body></html>\n`);
}
console.log('Five lecture shells prepared.');
