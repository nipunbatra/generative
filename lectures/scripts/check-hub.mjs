import fs from 'node:fs';
import {browser,base} from './browser.mjs';
const b=await browser(),p=await b.newPage(),errors=[];
fs.mkdirSync('qa',{recursive:true});
try{
 p.on('pageerror',e=>errors.push(e.message));
 await p.setViewport({width:1280,height:1000});
 await p.goto(base+'/',{waitUntil:'networkidle0'});
 await p.evaluate(()=>document.fonts.ready);
 const links=await p.$$eval('a',els=>els.map(e=>({label:e.textContent,href:e.href})));
 const results=[];
 for(const link of links)results.push({...link,...await p.evaluate(async url=>{const r=await fetch(url,{method:'HEAD'});return {status:r.status};},link.href)});
 await p.screenshot({path:'qa/series-desktop.jpg',type:'jpeg',quality:90,fullPage:true});
 await p.setViewport({width:390,height:844});await p.reload({waitUntil:'networkidle0'});
 const mobile=await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
 await p.screenshot({path:'qa/series-mobile.jpg',type:'jpeg',quality:90,fullPage:true});
 const report={passed:!errors.length&&results.every(x=>x.status===200)&&mobile,errors,mobileNoPageOverflow:mobile,links:results};
 fs.writeFileSync('qa/hub-checks.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({passed:report.passed,links:results.length,mobileNoPageOverflow:mobile,errors}));
 if(!report.passed)process.exitCode=1;
}finally{await b.close();}
