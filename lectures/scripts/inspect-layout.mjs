import {browser,base} from './browser.mjs';
const b=await browser(),p=await b.newPage();
try{
 await p.setViewport({width:390,height:844});
 await p.goto(`${base}/lecture${process.argv[2]||2}/?reading`,{waitUntil:'networkidle0'});
 await p.waitForFunction(()=>window.deckReady);
 console.log(JSON.stringify(await p.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.right>innerWidth+1&&r.width>0&&!e.closest('.formula-scroll,.diagram-scroll,.table-scroll,pre')}).map(e=>({tag:e.tagName,cls:e.className,id:e.closest('.scene')?.id,text:e.textContent.slice(0,70),right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).slice(0,30)})),null,2));
}finally{await b.close();}
