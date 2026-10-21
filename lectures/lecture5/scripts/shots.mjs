// Screenshot every build of every slide (presentation mode, 1280×720) and record overflow and errors.
//   SERIES_URL=http://127.0.0.1:8779 node lecture5/scripts/shots.mjs OUT_DIR [first-id] [last-id]
import fs from 'node:fs';
import path from 'node:path';
import {browser,base} from '../../scripts/browser.mjs';
const out=process.argv[2],from=process.argv[3],to=process.argv[4];fs.mkdirSync(out,{recursive:true});
const b=await browser(),p=await b.newPage(),errors=[];
p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
await p.setViewport({width:1280,height:720});
await p.goto(`${base}/lecture5/${process.env.ALL?'?all':''}`,{waitUntil:'networkidle0'});await p.waitForFunction(()=>window.deckReady);
const slides=await p.evaluate(()=>lecture.slides.map(s=>({id:s.id,builds:s.builds})));
let on=!from,report=[];
for(const [i,s] of slides.entries()){
 if(s.id===from)on=true;if(!on)continue;
 for(let k=1;k<=s.builds;k++){
  await p.evaluate((id,k)=>lecture.goTo(id,k),s.id,k);await new Promise(r=>setTimeout(r,k===1?450:250));
  const st=await p.evaluate(()=>{const sc=document.querySelector('.scene.current'),c=sc.querySelector('.content');return {over:c.scrollHeight-c.clientHeight,wide:Math.max(0,...[...sc.querySelectorAll('.content *')].map(e=>e.getBoundingClientRect().right))-(sc.getBoundingClientRect().right)};});
  report.push({i:i+1,id:s.id,build:k,...st});
  await p.screenshot({path:path.join(out,`${String(i+1).padStart(3,'0')}-${s.id}-${k}.png`)});
 }
 if(s.id===to)break;
}
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({errors,report},null,1));
console.log('shots',report.length,'errors',errors.length,'overflow',report.filter(r=>r.over>4).map(r=>`${r.id}/${r.build}:${r.over}`).join(' '));
await b.close();
