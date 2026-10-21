import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {browser,base} from './browser.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const chosen=process.argv.slice(2).filter(x=>/^[1-5]$/.test(x));const lectures=chosen.length?chosen:[1,2,3,4,5];
const b=await browser();let allPassed=true;
try{
 for(const n of lectures){
  const dir=path.join(root,`lecture${n}`),out=path.join(dir,'qa');fs.mkdirSync(path.join(out,'screenshots'),{recursive:true});
  for(const name of fs.readdirSync(path.join(out,'screenshots')))if(name.endsWith('.jpg'))fs.unlinkSync(path.join(out,'screenshots',name));
  const p=await b.newPage(),errors=[],failures=[],checks=[];let captures=0;
  p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error')errors.push(m.text());});p.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  await p.setViewport({width:1280,height:800,deviceScaleFactor:1});await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);
  await p.goto(`${base}/lecture${n}/?all`,{waitUntil:'networkidle0'});
  try{await p.waitForFunction(()=>window.deckReady,{timeout:30000});}catch(e){fs.writeFileSync(path.join(out,'browser-checks.json'),JSON.stringify({passed:false,startup:e.message,errors},null,2));await p.close();allPassed=false;continue;}
  const state=await p.evaluate(()=>({meta:lecture.meta,slides:lecture.allSlides,mathsErrors:lecture.mathsErrors}));errors.push(...state.mathsErrors);
  const names=state.slides;
  for(let i=0;i<names.length;i++){
   const s=names[i];for(let k=1;k<=s.builds;k++){
    await p.evaluate((id,k)=>lecture.goTo(id,k),s.id,k);
    const faults=await p.evaluate(()=>{
     const el=document.querySelector('.scene.current'),r=el.getBoundingClientRect(),c=el.querySelector('.content'),scale=r.width/1280,f=[];
     for(const x of el.querySelectorAll('h2,p,pre,.display,svg text,table')){
      if(getComputedStyle(x).visibility==='hidden')continue;const z=x.getBoundingClientRect();if(!z.width||!z.height)continue;
      if(z.left<r.left+60*scale-1||z.right>r.right-60*scale+1||z.top<r.top||z.bottom>r.bottom)f.push('outside margins: '+x.textContent.trim().slice(0,95));
     }
     if(c.scrollHeight>c.clientHeight+4)f.push('content height overflow');
     for(const box of el.querySelectorAll('.formula-scroll,.diagram-scroll,.table-scroll'))if(getComputedStyle(box).visibility!=='hidden'&&box.scrollWidth>box.clientWidth+2)f.push('content wider than its box: '+box.textContent.trim().slice(0,70));
     for(const img of el.querySelectorAll('img'))if(!img.complete||!img.naturalWidth)f.push('missing image '+img.getAttribute('src'));
     for(const pre of el.querySelectorAll('pre'))if(getComputedStyle(pre).visibility!=='hidden'&&pre.scrollWidth>pre.clientWidth+2)f.push('code width overflow');
     for(const svg of el.querySelectorAll('svg[viewBox]'))for(const t of svg.querySelectorAll('text')){
      if(getComputedStyle(t).visibility==='hidden')continue;const v=svg.viewBox.baseVal,z=t.getBBox();
      if(z.x<v.x-1||z.y<v.y-1||z.x+z.width>v.x+v.width+1||z.y+z.height>v.y+v.height+1)f.push('SVG text outside viewBox: '+t.textContent);
     }
     return [...new Set(f)];
    });
    if(faults.length)failures.push({id:s.id,build:k,faults});
    await (await p.$('.scene.current')).screenshot({path:path.join(out,'screenshots',`${String(i+1).padStart(3,'0')}-${s.id}-${k}.jpg`),type:'jpeg',quality:85});captures++;
   }
  }
  const check=(name,pass)=>{checks.push({name,pass});if(!pass)failures.push({check:name});};
  const multi=names.find(s=>s.builds>1);await p.evaluate(id=>lecture.goTo(id,1),multi.id);await p.keyboard.press('ArrowRight');check('right arrow advances',await p.evaluate(()=>lecture.current.build===2));await p.keyboard.press('ArrowLeft');check('left arrow returns',await p.evaluate(()=>lecture.current.build===1));await p.click('[data-action="finish"]');check('complete reveals final state',await p.evaluate(()=>lecture.current.build===lecture.slides[lecture.current.index].builds));await p.click('[data-action="reset"]');check('reset restores first state',await p.evaluate(()=>lecture.current.build===1));await p.click('[data-action="notes"]');check('speaker notes available',await p.$eval('#notes',e=>!e.hidden&&e.textContent.includes('Expected')));await p.click('[data-action="notes"]');
  await p.goto(`${base}/lecture${n}/?all#${multi.id}/2`,{waitUntil:'networkidle0'});await p.waitForFunction(()=>window.deckReady);check('fresh deep link',await p.evaluate(id=>lecture.current.id===id&&lecture.current.build===2,multi.id));
  await p.setViewport({width:390,height:844});await p.goto(`${base}/lecture${n}/?reading`,{waitUntil:'networkidle0'});await p.waitForFunction(()=>window.deckReady);check('mobile reading has no page overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));check('reading has all scenes',await p.$$eval('.scene',(e,count)=>e.length===count,names.length));
  // A full-page capture of a long deck exceeds Chrome's screenshot limit; keep the first 16,000 px.
  const readingHeight=await p.evaluate(()=>document.documentElement.scrollHeight);
  await p.screenshot({path:path.join(out,'reading-mobile.jpg'),type:'jpeg',quality:80,clip:{x:0,y:0,width:390,height:Math.min(readingHeight,16000)},captureBeyondViewport:true});
  const manifest={...state.meta,coreSlides:names.filter(s=>!s.optional).length,optionalSlides:names.filter(s=>s.optional).length,slides:names.map(s=>({id:s.id,title:s.title,section:s.section,builds:s.builds,optional:!!s.optional})),totalStates:names.reduce((a,s)=>a+s.builds,0)};
  fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  const notes=[`# ${state.meta.title}: speaker notes`,`\nPlanned main route: ${state.meta.minutes} minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.\n`];
  notes.push('## Pacing\n\n| Episode | Minutes |\n|---|---:|\n'+(state.meta.sections||[]).filter(s=>s.minutes>0).map(s=>`| ${s.title} | ${s.minutes} |`).join('\n')+'\n');
  for(const s of names){notes.push(`## ${s.id}: ${s.title}${s.optional?' (optional)':''}\n`);const no=s.notes||{};const asks=Array.isArray(no.ask)?no.ask:[no.ask];const expected=Array.isArray(no.expected)?no.expected:[no.expected];asks.forEach((a,i)=>notes.push(`**Build ${i+1}, ask:** ${a||''}\n\n**Expected:** ${expected[i]||expected.at(-1)||''}\n`));for(const [k,l] of [['point','Point here'],['misconception','Common misconception'],['demo','Notebook demonstration'],['transition','Transition']])if(no[k])notes.push(`**${l}:** ${no[k]}\n`);if(s.source)notes.push(`**Source / evidence:** ${s.source}\n`);}
  fs.writeFileSync(path.join(dir,'speaker-notes.md'),notes.join('\n'));
  const report={passed:!errors.length&&!failures.length,renderedStates:captures,coreSlides:manifest.coreSlides,optionalSlides:manifest.optionalSlides,errors,failures,checks,scope:'Every authored state captured. Geometry checks do not replace visual inspection or classroom rehearsal.'};fs.writeFileSync(path.join(out,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({lecture:n,passed:report.passed,renderedStates:captures,coreSlides:manifest.coreSlides,optionalSlides:manifest.optionalSlides,errorCount:errors.length,failureCount:failures.length,firstErrors:errors.slice(0,3),firstFailures:failures.slice(0,4)}));allPassed&&=report.passed;await p.close();
 }
}finally{await b.close();}
if(!allPassed)process.exitCode=1;
