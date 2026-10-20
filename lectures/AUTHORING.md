# Shared authoring contract

Each lecture owns its own directory. Root owns `shared/`, `scripts/`, the landing page and integration. Existing lectures/prototype remain intact.

## Content module

Each lecture provides `content.js`:

```js
import {m, eq, at, only, cols, code, image, table, flow, plot, palette} from '../shared/author.js';
export const meta = {
  id:'lecture1', title:'KL and variational inference', number:1, minutes:70,
  subtitle:'Fit an approximation to a posterior',
  sections:[{id:'compare', title:'Compare distributions', minutes:9}],
  notebook:'lecture1.ipynb',
};
export const slides = [
  {
    id:'closer', section:'compare', title:'Which approximation is closer?',
    builds:2,
    body:cols('<div>an editable SVG or image</div>', at(2,eq(String.raw`q_\psi`))),
    prompts:['Predict before calculating.','Compare the calculated values.'],
    notes:{
      ask:['Which is closer?','What supports the answer?'],
      expected:['A proposed answer.','The numerical comparison.'],
      point:'Point to the fixed reference.',
      misconception:'State the misconception this scene resolves.',
      transition:'Why the next scene is needed.',
      demo:'Optional notebook cell/section and exact action.',
    },
    source:'Exact calculation, or measured output with its source.',
    optional:false,
  }
];
```

The engine builds notes and both PDFs from this same source. Every slide needs an authored question, expected response and transition. Notes may use strings or arrays for ask/expected. `builds` counts the initial state. The main route excludes `optional:true`; the full reading/PDF package includes clearly labelled optional slides after the core route.

If fixtures are needed: `const data=await fetch(new URL('./data/fixtures.json',import.meta.url)).then(r=>r.json());`. Paths inside images are relative to the lecture's page, so use `figures/name.svg`.

## Helpers

- `m(tex)` produces inline mathematical source; `eq(tex)` produces display maths. Pass raw TeX with `String.raw`. The engine renders with local KaTeX. Supported macros: `\E`, `\KL`, `\N`, `\ELBO`, `\pcol{}`, `\qcol{}`, `\dcol{}`, `\ncol{}`, `\lcol{}`.
- `at(build,html)` reveals cumulatively from that build.
- `only('1-2',html)` shows only the specified build(s). Adjacent `only` siblings share one stable layout slot.
- `cols(left,right,ratio='1fr 1fr')` makes a two-column composition. Mobile stacks it.
- `code(string)` renders escaped, readable code. Keep active slide snippets to 3–6 short lines; put complete runnable code in the notebook.
- `image(src,alt,caption='')` renders an evidence image with its caption. Preserve aspect ratio.
- `table(headers,rows)` produces a small table. Use no more than about five rows in a main slide.
- `flow(nodes,edges,options={})` produces an editable SVG. Each node has `{id,x,y,w,h,label,sub,role,shape}`. Roles: `data,q,p,noise,loss,d,ink`; shape is `box` or `circle`. Edges have `{from,to,label,role,dashed,fromSide,toSide}`; sides are `left,right,top,bottom`, default right→left. Options `{width:1000,height:330,label:'accessible description'}`. The labels are plain text; add maths as adjacent `m()` labels if needed. Prefer custom SVG when an explanatory figure needs precise geometry.
- `plot(series,options)` draws checked data on fixed axes. Each series `{label,role,x:[...],y:[...],dashed}`. Options `{xMin,xMax,yMin,yMax,xLabel,yLabel,width:1000,height:360,ticksX,ticksY}`. Includes a compact labelled legend above the chart; custom SVG may use direct labels.
- `palette` has `{data:'#245EDB',q:'#8B2CDE',p:'#0F766E',noise:'#AA4E08',loss:'#BE123C',d:'#147737',ink:'#14171F',paper:'#FAFAFB'}`.

Useful authored classes: `.lead` (28 px), `.small` (25 px), `.muted`, `.equation-stack`, `.centre`, `.diagram-scroll`, `.formula-scroll`, `.image-pair`, `.result` (large number), `.caption` (22 px), `.callout` (28 px). Titles are 40 px; principal maths 32 px; code 25 px. A slide's content area is about 1152 × 440 px. Split dense content across slides rather than shrinking it.

## Package files

- `content.js`, `README.md`, `lectureN.ipynb`, `data/`, `figures/`, reproducibility scripts and local data/checkpoints where needed.
- A notebook is executable top-to-bottom in the documented environment, includes figures and observed results, and has no intentional unhandled failure. Catch and explain expected sampling errors.
- Scripts record actual execution checks in `qa/` with seeds, dependencies and output provenance. Do not claim recorded checkpoints are newly trained.
- Root generates `index.html`, `speaker-notes.md`, `exports/completed.pdf`, `exports/all-builds.pdf`, `manifest.json`, and all-state browser/PDF QA.

Core routes stay 60–70 minutes. More slides should expose intermediate reasoning, examples and diagrams, not introduce unrelated topics. Optional extensions are explicitly separated.
