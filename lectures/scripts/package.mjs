import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import './stamp-assets.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const records=[];
for(let n=1;n<=4;n++){
 const dir=path.join(root,`lecture${n}`),manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 const notebook=manifest.notebook;
 for(const rel of [notebook,'speaker-notes.md','exports/completed.pdf','exports/all-builds.pdf']){
  if(!fs.existsSync(path.join(dir,rel)))throw new Error(`Missing lecture${n}/${rel}`);
 }
 records.push(manifest);
}
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const rows=records.map(m=>`<article><div class="num">0${m.number}</div><div><p class="meta">${m.minutes} min · ${m.coreSlides} main slides + ${m.optionalSlides} optional · ${m.totalStates} reveal states</p><h2><a href="lecture${m.number}/">${esc(m.title)}</a></h2><p>${esc(m.subtitle)}</p><nav aria-label="Lecture ${m.number} resources"><a class="present" href="lecture${m.number}/">Present →</a><a href="lecture${m.number}/?reading">Read with notes</a><a href="lecture${m.number}/${esc(m.notebook)}" download>Notebook</a><a href="lecture${m.number}/speaker-notes.md">Speaker notes</a><a href="lecture${m.number}/exports/completed.pdf">Completed PDF</a><a href="lecture${m.number}/exports/all-builds.pdf">All reveals PDF</a></nav></div></article>`).join('\n');
fs.writeFileSync(path.join(root,'index.html'),`<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Generative modelling · Four lectures</title>
<style>
:root{font-family:"Avenir Next",Avenir,"Helvetica Neue",Arial,sans-serif;color:#14171f;background:#fafafb}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:72px 36px 48px}.kicker{color:#0f766e;text-transform:uppercase;font-size:13px;font-weight:650;letter-spacing:1.8px}h1{font-size:56px;line-height:1.08;letter-spacing:-2px;margin:18px 0 22px;max-width:900px}.intro{max-width:780px;font-size:22px;line-height:1.55;color:#46515d}.route{font-size:17px;line-height:1.6;color:#68717b;padding:24px 0 28px;border-bottom:1px solid #d7dce1}article{display:grid;grid-template-columns:68px 1fr;gap:24px;padding:32px 0;border-bottom:1px solid #d7dce1}.num{font-size:28px;color:#8b2cde;padding-top:4px}h2{font-size:28px;letter-spacing:-.6px;margin:7px 0 10px}h2 a{text-decoration:none;color:inherit}.meta{color:#68717b;font-size:13px;margin:0}article p{font-size:17px;line-height:1.5;margin:8px 0;color:#46515d}nav{display:flex;gap:12px 24px;flex-wrap:wrap;padding-top:12px}a{color:#0f766e;text-underline-offset:4px}nav a{font-size:15px}.present{font-weight:650}a:focus-visible{outline:2px solid #8b2cde;outline-offset:4px}footer{font-size:14px;line-height:1.65;color:#68717b;margin-top:32px}footer a{margin-right:18px}@media(max-width:650px){main{padding:34px 22px}h1{font-size:39px;letter-spacing:-1px}.intro{font-size:19px}article{grid-template-columns:35px 1fr;gap:12px}h2{font-size:25px}nav{gap:14px 18px}}
</style><main><p class="kicker">Nipun Batra · Deep learning</p><h1>Generative modelling,<br>one reasoning step at a time.</h1><p class="intro">Start with fitting a distribution. Build a generative model, inspect its learning signal, and test what its samples actually show.</p><p class="route">KL & variational inference → generative models, autoencoders and VAEs → GANs → diffusion<br>Slides lead. Short notebook demonstrations make the calculations visible.</p>${rows}<footer><p>Use ← / → for reveals, Shift + arrows to change slides, N for notes, O for the outline, and R to reset a slide. “Read with notes” includes optional extensions and works on a phone. Both PDFs include the clearly marked optional slides.</p><p>Timing is a teaching budget, not a classroom rehearsal measurement. Checkpoint replay, fresh updates and full training are labelled separately.</p><a href="TEACHING-GUIDE.md">Teaching guide</a><a href="MATHS-AND-DIAGRAMS.md">Maths & diagram conventions</a><a href="README.md">Run & reproduce</a><a href="QA.md">Verification</a></footer></main></html>`);
let files=[];
function walk(dir){for(const e of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,e.name),rel=path.relative(root,p);if(e.isDirectory()){if(!['qa','__pycache__','.ipynb_checkpoints','.cache'].includes(e.name))walk(p);}else if(!['asset-manifest.json'].includes(e.name))files.push({path:rel,bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')});}}
walk(root);
fs.writeFileSync(path.join(root,'asset-manifest.json'),JSON.stringify({created:new Date().toISOString(),scope:'Deliverable source, notebooks, figures, data, checkpoints and exports. QA screenshots omitted from this inventory.',files},null,2)+'\n');
console.log(JSON.stringify({lectures:records.map(m=>({number:m.number,core:m.coreSlides,optional:m.optionalSlides,states:m.totalStates})),files:files.length},null,2));
