const puppeteer=require('puppeteer'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await require('./browser.cjs').launch(puppeteer),page=await browser.newPage();
 await page.setViewport({width:1280,height:720});await page.goto('file://'+root+'/index.html');await page.waitForFunction(()=>window.lectureReady);
 const fixture=JSON.parse(fs.readFileSync(root+'/results/probability.json'));
 const checks=[];const check=(name,ok)=>{checks.push({name,ok});if(!ok)throw Error(name);};
 const f=x=>(Math.abs(x)<.0005?0:x).toFixed(3).replace('-','−');
 const go=async(id,b=0)=>page.evaluate(([id,b])=>{lecture.reset();lecture.go(lecture.slides.findIndex(s=>s.id===id),b)},[id,b]);
 const text=()=>page.$eval('.slide.active .plot',e=>e.textContent);
 const param=async(k,v)=>page.$eval(`.slide.active [data-param="${k}"]`,(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}))},v);
 await go('elbo-reweight',2);check('Reweighted evidence remains .48',(await text()).includes('0.48 = p(x)'));
 await go('elbo-jensen',0);check('Lower bound withheld on first reveal',!(await text()).includes('−1.595'));
 await go('elbo-jensen',2);check('Jensen values match independent calculation',(await text()).includes(f(fixture.toy.log_evidence))&&(await text()).includes(f(fixture.toy.states[0].elbo)));
 await go('gan-source',0);check('Source posterior withheld initially',!(await text()).includes('P(S = 1 | x)'));
 await go('gan-source',2);check('Source posterior revealed',(await text()).includes('P(S = 1 | x)'));
 await go('gan-logloss',2);check('Balanced-class factor of two stated',(await text()).includes('Balanced average = V / 2'));
 await go('gan-bayes',2);
 for(const row of fixture.bayes){await param('sourcePrior',row.pi);let value=await text();check(`All three Bayes scores at ${row.pi}`,row.optimal.every(v=>value.includes(f(v))));}
 await go('gan-density',2);check('Gaussian Bayes score at x=0', (await text()).includes('D*(x) = 0.881'));
 await param('probe',2);check('Probe changes density-ratio score',(await text()).includes('D*(x) = 0.119'));
 await page.click('.slide.active [data-action=match-density]');check('Matching models gives half',(await text()).includes('D* = 0.5 everywhere')&&(await text()).includes('D*(x) = 0.500'));
 await go('gan-density',3);check('Final reveal matches distributions',(await text()).includes('D* = 0.5 everywhere'));
 await go('gan-gradient',2);
 for(const row of fixture.gradients){await param('logit',row.logit);const value=await text();check(`Both loss derivatives agree at logit ${row.logit}`,value.includes(f(row.minimax))&&value.includes(f(row.non_saturating)));}
 await go('gan-js',2);
 for(const row of fixture.js){await param('match',row.t);const value=await text();check(`JS identity displayed at model fraction ${row.t}`,value.includes(`JS = ${f(row.js)} nats`)&&value.includes(`= ${f(row.value)}`));}
 check('Nine detours add 15 minutes',await page.evaluate(()=>{const s=lecture.slides.filter(s=>s.detour);return s.length===9&&s.reduce((n,x)=>n+x.seconds,0)===900}));
 check('Core route retains 80 minutes',await page.evaluate(()=>lecture.slides.filter(s=>!s.appendix&&!s.detour).reduce((n,s)=>n+s.seconds,0)===4800));
 await go('gan-bayes');await page.keyboard.press('s');check('Presenter notes link primary paper and Lilian',await page.$eval('#dialog-content',e=>[...e.querySelectorAll('a')].some(a=>a.href==='https://arxiv.org/abs/1406.2661')&&[...e.querySelectorAll('a')].some(a=>a.href.includes('2017-08-20-gan'))));await page.click('#dialog-close');
 check('Reading notes contain resources',await page.$$eval('#gan-bayes .reading-notes a',es=>es.length===2));
 const resource=await browser.newPage();await resource.goto('file://'+root+'/resources.html');
 const urls=await resource.$$eval('article h3 a',es=>es.map(e=>e.href));
 check('Resource index contains all 17 references',urls.length===17);
 check('All user-requested text/video resources linked',['2018-08-12-vae','2017-08-20-gan','jaan.io','fQ8D0-tTCBU','bo8wceOS6C0','R8V8CbuxryI','notebooks/vae.ipynb'].every(part=>urls.some(url=>url.includes(part))));
 await resource.setViewport({width:390,height:844});check('Resource page fits mobile width',await resource.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await resource.screenshot({path:root+'/output/rebuild-qa/resources-mobile.png',fullPage:true});await resource.close();
 fs.writeFileSync(root+'/output/rebuild-qa/probability-browser.json',JSON.stringify({checks},null,2));await browser.close();console.log(checks.length+' probability and resource checks passed');
})().catch(e=>{console.error(e);process.exit(1)});
