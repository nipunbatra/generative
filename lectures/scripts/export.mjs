import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {browser,base} from './browser.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),b=await browser();
const chosen=process.argv.slice(2).filter(x=>/^[1-5]$/.test(x));const lectures=chosen.length?chosen:[1,2,3,4,5];
try{
 for(const n of lectures){
  const out=path.join(root,`lecture${n}`,'exports');fs.mkdirSync(out,{recursive:true});
  for(const [query,name] of [['print','completed.pdf'],['print=builds','all-builds.pdf']]){
   const p=await b.newPage(),errors=[];p.on('pageerror',e=>errors.push(e.message));await p.setViewport({width:1280,height:720});await p.goto(`${base}/lecture${n}/?${query}`,{waitUntil:'networkidle0'});await p.waitForFunction(()=>window.deckReady);await p.evaluate(()=>document.fonts.ready);await p.emulateMediaType('print');
   const state=await p.evaluate(()=>({pages:document.querySelectorAll('.scene').length,maths:lecture.mathsErrors,overflow:[...document.querySelectorAll('.scene')].filter(s=>{const c=s.querySelector('.content');return c.scrollHeight>c.clientHeight+4;}).map(s=>s.id)}));
   if(errors.length||state.maths.length||state.overflow.length)throw new Error(JSON.stringify({lecture:n,query,errors,...state}));
   await p.pdf({path:path.join(out,name),preferCSSPageSize:true,printBackground:true,margin:{top:0,right:0,bottom:0,left:0},timeout:180000});
   console.log(`Lecture ${n}: ${name}, ${state.pages} authored pages`);await p.close();
  }
 }
}finally{await b.close();}
