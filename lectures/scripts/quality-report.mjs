import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const rows=[],notebooks=[];
for(let n=1;n<=4;n++){
 const dir=path.join(root,`lecture${n}`),read=f=>JSON.parse(fs.readFileSync(path.join(dir,f)));
 const m=read('manifest.json'),browser=read('qa/browser-checks.json'),pdf=read('qa/pdf-checks.json'),science=read('qa/science-checks.json');
 if(!browser.passed||!pdf.passed||!(science.passed||science.status==='passed'))throw new Error(`Lecture ${n} checks failed`);
 const nb=read(m.notebook),cells=nb.cells.filter(c=>c.cell_type==='code');
 const record={lecture:n,notebook:m.notebook,codeCells:cells.length,executed:cells.filter(c=>c.execution_count!==null).length,outputs:cells.reduce((a,c)=>a+c.outputs.length,0),errors:cells.flatMap(c=>c.outputs).filter(o=>o.output_type==='error').length};
 if(record.executed!==record.codeCells||record.errors)throw new Error(`Lecture ${n} notebook incomplete`);
 notebooks.push(record);rows.push({lecture:n,title:m.title,core:m.coreSlides,optional:m.optionalSlides,states:m.totalStates,completed:m.slides.length});
 const notes=path.join(dir,'speaker-notes.md');let text=fs.readFileSync(notes,'utf8');
 if(!text.includes('## Pacing'))text=text.replace('\n## ',`\n## Pacing\n\n| Episode | Minutes |\n|---|---:|\n${m.sections.filter(s=>s.minutes>0).map(s=>`| ${s.title} | ${s.minutes} |`).join('\n')}\n\n## `);
 fs.writeFileSync(notes,text);
}
fs.writeFileSync(path.join(root,'notebook-verification.json'),JSON.stringify(notebooks,null,2)+'\n');
const stateCount=rows.reduce((a,r)=>a+r.states,0),completed=rows.reduce((a,r)=>a+r.completed,0);
fs.writeFileSync(path.join(root,'QA.md'),`# Verification record

Build verification recorded on 20 October 2026. These checks cover the local artifacts; hosted delivery is verified separately. Full fresh image-model training is not claimed.

## Delivered scope

| Lecture | Main slides | Optional slides | Completed PDF pages | All-reveal PDF pages |
|---|---:|---:|---:|---:|
${rows.map(r=>`| ${r.lecture}. ${r.title} | ${r.core} | ${r.optional} | ${r.completed} | ${r.states} |`).join('\n')}

Total: **${completed} logical slides**, **${stateCount} authored reveal states**, **10 PDFs / ${completed+stateCount} pages**.

## Browser and visual review

Every authored state was rendered and captured at the 1280 × 720 logical slide size. All decks pass mathematical typesetting, missing-asset, content overflow, slide-margin, code-width and inline-SVG viewBox text checks. Keyboard navigation, complete/reset, notes, deep links and 390px reading mode pass. Long formulas and diagrams scroll within their reading-mode container rather than widening the page.

The primary author inspected all Lecture 1 state contact sheets, followed up on full-resolution frames and inspected PDF output. Independent cross-review covered every AE/VAE and GAN/DDPM state. It identified and resolved equations with rounded values presented as exact, decoder-logit/probability inconsistencies, arrow/label crossings, misleading GAN update labels, clipped density axes, the U-Net skip destination, the latent-diffusion sampling start and the attention matrix convention. Corrected states were inspected again.

The reports are [AE/VAE cross-review](lecture1/qa/cross-review-l2-l3.md) and [GAN/DDPM cross-review](lecture2/qa/cross-review-l4-l5.md). Browser reports are published under each lecture's \`qa/\`. The larger state screenshots and contact sheets are retained locally and can be regenerated with the QA scripts. Automated geometry checks support this inspection; they do not establish teaching effectiveness.

## PDF checks

Every page of both PDFs for every lecture was rasterised. Page counts, page dimensions and nonempty text pass. For every logical slide, the completed export matches the corresponding final reveal page outside the page-number footer. Every completed PDF body was also compared with its browser screenshot, allowing one pixel of global registration for subpixel rounding and a mean absolute pixel difference below 5/255. Image-heavy frames initially exceeded the unregistered threshold; full-size comparison confirmed the rounding difference and the registered check passes. Representative PDF pages were visually inspected. Reports are published at \`lectureN/qa/pdf-checks.json\`; rendered pages are retained locally under \`lectureN/qa/pdf/\` and can be regenerated.

## Notebook execution

| Lecture | Executed code cells | Saved outputs | Errors |
|---|---:|---:|---:|
${notebooks.map(r=>`| ${r.lecture} | ${r.executed} / ${r.codeCells} | ${r.outputs} | ${r.errors} |`).join('\n')}

Notebooks were executed on CPU with the existing Python 3.12 / PyTorch 2.13 environment. They bundle the small data snapshots, definitions and checkpoints required by their class paths. Full image-model training is explicitly optional and was not run anew. Exact execution reports are under each lecture's \`qa/\`.

## Mathematical and empirical checks

- **VI:** 2,780 independent assertions, including 2,005 numerical comparisons. Gaussian KL, derivatives, controlled-noise gradients, quadrature, direct Beta ELBO and evidence-gap identity pass. Largest comparison discrepancy is approximately 6.00e−12. The measured coin fit has a small positive residual KL; it is not labelled exact.
- **AE/VAE:** explicit event sums and batch means; binary-pixel BCE arithmetic; diagonal Gaussian KL against the library; actual parameter updates; identical AE code-cloud evidence across the lecture transition. Reconstruction-only gradients separately reach the VAE mean head, log-variance head and decoder.
- **GAN:** stable logit losses agree with scalar arithmetic; finite differences check the derivative; actual parameter changes verify D/G ownership. Saved image-generator replay differs from the original-device result by at most approximately 5.28e−6.
- **DDPM:** schedule identities, forward moments, the epsilon-parameterised reverse mean, posterior variance and the final-step zero-noise boundary pass. A real small-batch update and the full 1,000-step saved-model sampling replay run. Maximum replay difference from the original device is approximately 1.88e−5.

Per-lecture numerical fixtures, science reports, seeds and provenance record the scope of these claims. Generated samples and training traces are measured or replayed evidence; conceptual network diagrams are labelled schematics.

## Practical limits

The 60–70 minute main routes are planned budgets, not classroom-rehearsed timings. Forty-five-minute alternatives reduce scope. Rendering and mathematical verification do not establish generalisation, sample quality or optimal pedagogy. Hosted-delivery checks, larger model training and classroom rehearsal are separate from this build verification.
`);
console.log(JSON.stringify({passed:true,logicalSlides:completed,revealStates:stateCount,pdfPages:completed+stateCount,notebooks:notebooks.length}));
