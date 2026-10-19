const puppeteer=require('puppeteer'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await require('./browser.cjs').launch(puppeteer),page=await browser.newPage();
 await page.setViewport({width:1280,height:720});await page.goto('file://'+root+'/index.html');await page.waitForFunction(()=>window.lectureReady);
 const checks=[];const check=(name,ok)=>{checks.push({name,ok});if(!ok)throw Error(name);};
 const go=async(id,b=0)=>page.evaluate(([id,b])=>{lecture.reset();lecture.go(lecture.slides.findIndex(x=>x.id===id),b)},[id,b]);
 const text=()=>page.$eval('.slide.active .plot',e=>e.textContent);
 await go('kl-meaning',2);check('Initial discrete KL', (await text()).includes('0.193 nats'));
 await page.$eval('.slide.active [data-param=qmass]',e=>{e.value=.5;e.dispatchEvent(new Event('input',{bubbles:true}))});
 check('Matching distributions produce zero KL',(await text()).includes('0.000 nats'));
 await go('kl-modes');check('Wide candidate scores',(await text()).includes('0.635')&&(await text()).includes('1.653'));
 await page.click('.slide.active [data-action=kl-narrow]');check('Narrow candidate scores',(await text()).includes('8.800')&&(await text()).includes('0.721'));
 await page.click('.slide.active [data-action=kl-wide]');check('KL mode reset',(await text()).includes('0.635'));
 await go('elbo-gap');check('Initial exact gap',(await text()).includes('0.861'));
 await page.$eval('.slide.active input',e=>{e.value=2;e.dispatchEvent(new Event('input',{bubbles:true}))});
 check('Exact posterior closes evidence gap',(await text()).includes('The gap is zero.')&&(await text()).includes('−0.734'));
 await go('autograd-sampling');check('rsample withheld initially',!(await text()).includes('rsample'));
 await go('autograd-sampling',2);check('rsample revealed',(await text()).includes('rsample'));
 await go('beta-experiment');let before=await page.$eval('.slide.active .plot',e=>e.innerHTML);
 await page.click('.slide.active [data-action=beta-4]');check('Beta checkpoint button updates output',before!==await page.$eval('.slide.active .plot',e=>e.innerHTML));
 check('Beta metrics from saved checkpoint',(await text()).includes('162.8')&&(await text()).includes('3.04'));
 check('Beta selected state',await page.$eval('.slide.active [data-action=beta-4]',e=>e.getAttribute('aria-pressed')==='true'));
 check('Posterior geometry includes 600 means',await page.$$eval('.slide.active circle',es=>es.length===601));
 check('Explorer receives four minutes',await page.evaluate(()=>lecture.slides.find(x=>x.id==='latent-plane').seconds===240));
 check('Full route has 95 minutes',await page.evaluate(()=>lecture.slides.filter(x=>!x.appendix).reduce((n,x)=>n+x.seconds,0)===5700));
 // Print the new diagrams at both common projector aspect ratios.
 for(const size of [{width:1600,height:900},{width:1024,height:768}]){
  await page.setViewport(size);await page.waitForFunction(()=>Math.abs(parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--scale'))-Math.min(innerWidth/1280,innerHeight/720))<1e-6);await go('kl-modes',1);
  check(`Presentation fits ${size.width}x${size.height}`,await page.$eval('.slide.active',e=>{let r=e.getBoundingClientRect();return r.x>=-1&&r.y>=-1&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1}));
 }
 fs.writeFileSync(root+'/output/rebuild-qa/pedagogy-browser.json',JSON.stringify({checks},null,2));
 await browser.close();console.log(checks.length+' new interaction checks passed');
})().catch(e=>{console.error(e);process.exit(1)});
