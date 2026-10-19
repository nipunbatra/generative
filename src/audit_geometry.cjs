/* Rendered SVG geometry guard. Visual inspection is still required for plot semantics. */
const puppeteer=require('puppeteer'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await require('./browser.cjs').launch(puppeteer),page=await browser.newPage();
 await page.setViewport({width:1280,height:720});await page.goto('file://'+root+'/index.html');
 await page.waitForFunction(()=>window.lectureReady);await page.evaluate(()=>document.fonts.ready);
 await page.addStyleTag({content:'*{animation:none!important}'});
 const slides=await page.evaluate(()=>lecture.slides);let states=0,failures=[];
 const inspect=async(id,state)=>{states++;const hits=await page.evaluate(()=>{
  const out=[];
  for(const svg of document.querySelectorAll('.slide.active svg')){
   if(!svg.getClientRects().length)continue;
   const texts=[...svg.querySelectorAll('text')].filter(t=>t.getClientRects().length&&!t.closest('.future,.only-hidden'));
   const bs=texts.map(t=>t.getBBox()),v=svg.viewBox.baseVal;
   for(let i=0;i<texts.length;i++){
    const a=bs[i],label=texts[i].textContent;
    if(a.x< -3||a.y< -3||a.x+a.width>v.width+3||a.y+a.height>v.height+3)out.push({kind:'text-outside-viewbox',label});
    for(let j=i+1;j<texts.length;j++){
     const b=bs[j],w=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x),h=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
     if(w>2&&h>2)out.push({kind:'text-text',label,other:texts[j].textContent,overlap:[w,h]});
    }
    for(const shape of svg.querySelectorAll('path[stroke],rect[stroke],circle[stroke],ellipse[stroke]')){
     if(shape.closest('defs'))continue;
     const len=shape.getTotalLength();
     for(let d=0;d<=len;d+=2){const q=shape.getPointAtLength(d);
      if(q.x>a.x+2&&q.x<a.x+a.width-2&&q.y>a.y+3&&q.y<a.y+a.height-3){out.push({kind:'text-stroke',label,stroke:(shape.getAttribute('d')||shape.outerHTML).slice(0,160)});break;}
     }
    }
   }
  }
  return out;
 });if(hits.length)failures.push({id,state,hits});};
 for(let i=0;i<slides.length;i++){
  await page.evaluate(()=>lecture.showBackup(true));
  for(let b=0;b<slides[i].builds;b++){
   await page.evaluate(([i,b])=>{lecture.reset();lecture.go(i,b)},[i,b]);await inspect(slides[i].id,'build '+b);
  }
  const ranges=await page.$$eval('.slide.active input[type=range]',es=>es.map(e=>({key:e.dataset.param,min:e.min,max:e.max})));
  for(const r of ranges)for(const v of [r.min,r.max]){
   await page.evaluate(([i,key,v])=>{lecture.reset();lecture.go(i,lecture.slides[i].builds-1);let e=document.querySelector('.slide.active [data-param="'+key+'"]');e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}))},[i,r.key,v]);await inspect(slides[i].id,`${r.key}=${v}`);
  }
  if(['gaussian','kl-interactive'].includes(slides[i].id))for(const mu of [-2,2])for(const sigma of [.25,2]){
   await page.evaluate(([i,mu,sigma])=>{lecture.reset();lecture.go(i,lecture.slides[i].builds-1);for(const [key,v]of [['mu',mu],['sigma',sigma]]){let e=document.querySelector('.slide.active [data-param="'+key+'"]');e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));}},[i,mu,sigma]);await inspect(slides[i].id,`mu=${mu},sigma=${sigma}`);
  }
 }
 const report={states,failures,method:'SVG text bounding boxes, 2-unit sampled strokes and node borders, viewBox containment; all reveals plus slider endpoints and Gaussian parameter corners. Two units horizontal / three units vertical tolerance excludes font metric padding. Does not replace full-size visual review or inspect glyph outlines.'};
 fs.writeFileSync(root+'/output/rebuild-qa/geometry-report.json',JSON.stringify(report,null,2));await browser.close();console.log(JSON.stringify({states,failures},null,2));process.exitCode=failures.length?1:0;
})().catch(e=>{console.error(e);process.exit(1)});
