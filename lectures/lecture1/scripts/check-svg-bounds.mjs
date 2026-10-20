import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {browser,base} from '../../scripts/browser.mjs';

// DOM element bounds can miss text clipped inside a nested SVG viewport.
// Inspect the text's own SVG coordinates in every revealed state instead.
const output=fileURLToPath(new URL('../qa/svg-text-bounds.json',import.meta.url));
const b=await browser();
try{
 const page=await b.newPage();
 await page.setViewport({width:1280,height:800,deviceScaleFactor:1});
 await page.goto(`${base}/lecture1/?all`,{waitUntil:'networkidle0'});
 await page.waitForFunction(()=>window.deckReady);
 const slides=await page.evaluate(()=>lecture.allSlides.map(s=>({id:s.id,builds:s.builds})));
 const findings=[];
 let textChecks=0;
 for(const slide of slides){
  for(let build=1;build<=slide.builds;build++){
   await page.evaluate((id,k)=>lecture.goTo(id,k),slide.id,build);
   const result=await page.evaluate(()=>{
    let count=0;
    const issues=[];
    for(const svg of document.querySelectorAll('.scene.current svg')){
     if(getComputedStyle(svg).visibility==='hidden')continue;
     const v=svg.viewBox.baseVal;
     if(!v.width)continue;
     for(const text of svg.querySelectorAll('text')){
      if(getComputedStyle(text).visibility==='hidden')continue;
      count++;
      const r=text.getBBox();
      if(r.x<v.x-.5||r.x+r.width>v.x+v.width+.5||
         r.y<v.y-.5||r.y+r.height>v.y+v.height+.5){
       issues.push({
        text:text.textContent,
        bbox:{x:r.x,y:r.y,width:r.width,height:r.height},
        viewBox:svg.getAttribute('viewBox'),
       });
      }
     }
    }
    return{count,issues};
   });
   textChecks+=result.count;
   if(result.issues.length)findings.push({id:slide.id,build,issues:result.issues});
  }
 }
 const report={
  passed:findings.length===0,
  states:slides.reduce((n,s)=>n+s.builds,0),
  textChecks,
  findings,
  scope:'Text bounds inside inline SVG viewBoxes at a desktop viewport. External SVG images, mathematical glyphs, mobile layout and visual semantics require separate review.',
 };
 fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
 if(!report.passed)process.exitCode=1;
}finally{
 await b.close();
}
