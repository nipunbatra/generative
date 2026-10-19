'use strict';
const DATA=JSON.parse(document.getElementById('lesson-data').textContent);
const slides=[...document.querySelectorAll('.slide')];
let current=0,build=0,showBackup=false;
const state=new WeakMap(), mainCount=DATA.slides.filter(s=>!s.appendix).length;
const {text,arrow,box,path,circle,wrap,C}=Fig;
function st(el){if(!state.has(el)){let a={};el.querySelectorAll('[data-param]').forEach(i=>a[i.dataset.param]=+i.value);state.set(el,a);}return state.get(el);}
function stage(el){return document.body.classList.contains('present')?+(el.closest('.slide').dataset.build||0):DATA.slides[slides.indexOf(el.closest('.slide'))].builds-1;}
function sourceFigure(el,b){const id=el.dataset.figure,opt=JSON.parse(el.dataset.options||'{}');let html='';
if(id==='pipeline')html=Fig.pipeline(opt.labels,opt.colors);
else if(id==='density')html=Fig.density(opt.mu,opt.sig,{...opt.opts,area:b>0});
else if(id==='reparam'){let v=opt.stage;if(el.closest('#independent-noise'))v=b?2:1;if(el.closest('#computation-graph'))v=b?3:2;html=Fig.reparam(v,opt.grad);}
else if(id==='gan')html=Fig.gan(opt.mode,opt.stage);
else html=Fig[id]();
el.innerHTML=html;}
function decode(model,z){let a=z;model.layers.forEach((l,k)=>{a=l.weight.map((w,i)=>{let s=l.bias[i];for(let j=0;j<w.length;j++)s+=w[j]*a[j];return k<2?Math.max(0,s):1/(1+Math.exp(-s));});});return a;}
function paintDigit(canvas,values){let ctx=canvas.getContext('2d'),im=ctx.createImageData(28,28);for(let i=0;i<784;i++){let v=Math.round(255*(1-values[i]));im.data[i*4]=im.data[i*4+1]=im.data[i*4+2]=v;im.data[i*4+3]=255;}ctx.putImageData(im,0,0);}
function recordedPixels(index){const raw=atob(DATA.vae2.recorded_pixels);return Array.from(raw.slice(index*784,(index+1)*784),c=>c.charCodeAt(0)/255);}
function updateDecoder(el){let s=st(el);const hasWeights=Array.isArray(DATA.vae2.layers)&&DATA.vae2.layers.length===3;if(!hasWeights)s.fallback=true;s.x??=0;s.y??=0;let x=s.x,y=s.y,values;
if(s.fallback){let i=Math.round((x+3)*4),j=Math.round((3-y)*4);i=Math.max(0,Math.min(24,i));j=Math.max(0,Math.min(24,j));x=i/4-3;y=3-j/4;values=recordedPixels(j*25+i);}else values=decode(DATA.vae2,[x,y]);
el.querySelector('.cross').style.left=(x+3)/6*100+'%';el.querySelector('.cross').style.top=(3-y)/6*100+'%';
paintDigit(el.querySelector('canvas'),values);el.querySelector('.readout').textContent=`z = (${x.toFixed(2)}, ${y.toFixed(2)})`;
el.querySelector('.decoder-status').textContent=s.fallback?'RECORDED FALLBACK · nearest measured grid point; quantized to 8-bit for display.':'LIVE LOCAL DECODE · saved VAE-2 weights; decoder Bernoulli mean.';
el.querySelector('[data-action=fallback]').disabled=!hasWeights;el.querySelector('[data-action=fallback]').textContent=!hasWeights?'Recorded grid (weights unavailable)':s.fallback?'Use local decoder':'Use recorded fallback';el.querySelector('.decoder-grid').setAttribute('aria-label',`Latent plane, z ${x.toFixed(2)}, ${y.toFixed(2)}. Use arrow keys.`);
}
let noiseOriginal,noisePixels;
const photo=new Image();photo.src=DATA.photo;photo.onload=()=>{let c=document.createElement('canvas');c.width=c.height=224;let ctx=c.getContext('2d');ctx.drawImage(photo,0,0,224,224);noiseOriginal=ctx.getImageData(0,0,224,224);noisePixels=Fig.normals(224*224*2,899).flat().slice(0,224*224*3);document.querySelectorAll('[data-kind=noise]').forEach(draw);};
function drawNoise(el,s,b){if(!noiseOriginal)return;const a=s.manual?s.noise:[0,.18,.65,.999][Math.min(b,3)], signal=Math.sqrt(1-a),noise=Math.sqrt(a);const plot=el.querySelector('.plot');if(!plot.querySelector('canvas'))plot.innerHTML='<div class="noise-view"><div><canvas class="clean-photo" width="224" height="224" aria-label="Clean Newfoundland photograph"></canvas><p class="noise-label">same photograph</p></div><div class="noise-arrow">→</div><div><canvas class="noisy-photo" width="224" height="224" aria-label="Photograph after Gaussian corruption"></canvas><p class="noise-label">corrupted observation</p></div></div>';
plot.querySelector('.clean-photo').getContext('2d').putImageData(noiseOriginal,0,0);let ctx=plot.querySelector('.noisy-photo').getContext('2d'),im=ctx.createImageData(224,224);for(let i=0;i<224*224;i++){for(let c=0;c<3;c++){let clean=noiseOriginal.data[4*i+c]/127.5-1;im.data[4*i+c]=Math.max(0,Math.min(255,127.5*(1+signal*clean+noise*noisePixels[3*i+c])));}im.data[4*i+3]=255;}ctx.putImageData(im,0,0);el.querySelector('.readout').textContent=['Clean','Slightly noisy','Very noisy','Almost pure noise'][a===0?0:a<.3?1:a<.9?2:3];if(!s.manual){let input=el.querySelector('input');input.value=a;el.querySelector('output').value=a.toFixed(3);}}
function draw(el){let s=st(el),b=stage(el),kind=el.dataset.kind,plot=el.querySelector('.plot'),out=el.querySelector('.readout'),html='',read='';
if(Probability.draw(el,s,b))return;
if(Pedagogy.draw(el,s,b))return;
if(Rebuild.draw(el,s,b))return;
if(kind==='decoder'){updateDecoder(el);return;}
if(kind==='discrete'){let count=s.count??(b?100:0);html=Fig.discrete(count);read=count?'Solid bars: observed frequencies · dashed amber: probabilities':'Predict the counts before advancing.';}
if(kind==='gaussian'||kind==='location-scale'){let mu=kind==='location-scale'&&b===0?0:s.mu,sig=kind==='location-scale'&&b===0?1:s.sigma;html=Fig.density(mu,sig,{draws:false});let dots=Fig.normals(35,667+(s.seed||0)).filter(v=>Math.abs(mu+sig*v[0])<=4).map(v=>circle(440+(mu+sig*v[0])*85,299,3,'purple',.6)).join('');html=html.replace('</svg>',dots+'</svg>');read=`Gaussian: μ = ${mu.toFixed(2)}, σ = ${sig.toFixed(2)}, variance = ${(sig*sig).toFixed(2)} · plot shows x ∈ [−4, 4]`;}
if(kind==='sample-density'){html=s.show?Fig.fittedDensity(DATA.mixture):Fig.mixture('samples');read=s.show?'MEASURED FIT: Gaussian mixture learned by EM on this illustrative dataset':'Same fixed sample cloud';}
if(kind==='mixture'){html=Fig.mixture(s.kind||'single');read=(s.kind||'single')==='single'?'Population mean and covariance matched by one Gaussian':'Equal mixture weights; σ = 0.23 per component';}
if(kind==='fit'){html=Fig.density(s.mu,1);let xs=[-.5,.5,1.5];html=html.replace('</svg>',xs.map(x=>circle(440+x*85,299,7,'blue')).join('')+'</svg>');let ll=xs.reduce((a,x)=>a+Math.log(Fig.normal(x,s.mu,1)),0);read=b?`Sum log density = ${ll.toFixed(3)} nats · maximum at μ = 0.5`:'Three fixed observations: −0.5, 0.5, 1.5';}
if(kind==='prior'){html=Fig.prior(s.index||0);read='Each coordinate has mean 0 and variance 1.';}
if(kind==='posterior'){html=Fig.posterior(s.mu,s.sigma);read='μ = (μ₁, 0.4), σ = (σ₁, 0.7) · illustrative parameters';}
if(kind==='ae'){let model=(s.model==='vae'?DATA.vae2:DATA.ae2),ps=model.test_latents,colors=['#3478E5','#D45555','#178F82','#B98224','#8B5BB5','#566875','#8F6149','#3F879D','#BD6590','#78863A'];let o=Fig.axes(300,180,6).replaceAll('x₁','z₁').replaceAll('x₂','z₂')+`<circle cx="300" cy="180" r="12" fill="${C.amber}12" stroke="${C.amber}" stroke-width="2"/>`;ps.forEach((p,i)=>{let x=300+p[0]*6,y=180-p[1]*6;if(x>85&&x<515&&y>16&&y<330)o+=circle(x,y,3,colors[model.test_labels[i]],.65);});o+=text(730,105,s.model==='vae'?'VAE posterior means':'Ordinary AE codes','teal',29)+text(730,169,'Same scale; all 600 codes shown','gray',23)+text(730,224,'Amber circle: prior radius 2','amber',23)+text(730,280,'Labels colour points after training','gray',19);html=wrap(o,'Measured latent codes versus Gaussian prior');}
if(kind==='kl'){html=Fig.density(s.mu,s.sigma,{prior:true}).replace('possible value x','latent value z');let kl=.5*(s.mu*s.mu+s.sigma*s.sigma-Math.log(s.sigma*s.sigma)-1);read=b?`KL(q ‖ p) = ${kl.toFixed(3)} nats`:'Move the posterior toward the fixed amber prior.';}
if(kind==='gan'){html=Fig.gan(s.mode||'D',1);read=(s.mode||'D')==='D'?'D step: fake x is detached; update D only.':'G step: freeze D parameters; backpropagate through D into G.';}
if(kind==='collapse'){html=Fig.collapse(s.modes);read=`Constructed coverage: ${s.modes} / 8 modes · this is not a GAN training run.`;}
if(kind==='flow'){let amount=s.manual?s.amount:(b?.85:0);html=Fig.flow(amount);read='Invertible shear; determinant 1. This illustrative map preserves local area.';if(!s.manual){el.querySelector('input').value=amount;el.querySelector('output').value=amount.toFixed(2);}}
if(kind==='noise'){drawNoise(el,s,b);return;}
plot.innerHTML=html;if(out)out.textContent=read;let ps=el.querySelector('.print-state');if(ps)ps.textContent=read;
}
function bindInteractions(){document.querySelectorAll('.interactive').forEach(el=>{
 st(el);el.querySelectorAll('input').forEach(i=>i.addEventListener('input',()=>{let s=st(el);s[i.dataset.param]=+i.value;s.manual=true;el.querySelector(`[data-value="${i.dataset.param}"]`).value=(+i.value).toFixed(i.step==='1'?0:2);draw(el);}));
 el.querySelectorAll('[data-action]').forEach(btn=>btn.addEventListener('click',()=>{let s=st(el),a=btn.dataset.action;
 Probability.action(el,s,a);
 Pedagogy.action(el,s,a);
 Rebuild.action(el,s,a);
 if(a==='toggle')s.show=!s.show;if(a==='ten')s.count=(s.count??0)+10;if(a==='hundred')s.count=100;if(a==='reset')s.count=0;if(a==='resample')s.seed=(s.seed||0)+1;if(a==='single'||a==='mixture')s.kind=a;if(a==='sample')s.index=(s.index||0)+1;if(a==='ae'||a==='vae')s.model=a;if(a==='D'||a==='G')s.mode=a;if(a==='fallback')s.fallback=!s.fallback;draw(el);
 }));
 const grid=el.querySelector('.decoder-grid');if(grid){grid.addEventListener('click',ev=>{let r=grid.getBoundingClientRect(),s=st(el);s.x=Math.max(-3,Math.min(3,(ev.clientX-r.left)/r.width*6-3));s.y=Math.max(-3,Math.min(3,3-(ev.clientY-r.top)/r.height*6));draw(el);});grid.addEventListener('keydown',ev=>{let s=st(el);if(!['ArrowRight','ArrowLeft','ArrowUp','ArrowDown'].includes(ev.key))return;ev.stopPropagation();ev.preventDefault();s.x=Math.max(-3,Math.min(3,(s.x||0)+(ev.key==='ArrowRight'?.1:ev.key==='ArrowLeft'?-.1:0)));s.y=Math.max(-3,Math.min(3,(s.y||0)+(ev.key==='ArrowUp'?.1:ev.key==='ArrowDown'?-.1:0)));draw(el);});}
});}
function updateSlide(slide){let b=+(slide.dataset.build||0);slide.querySelectorAll('[data-r]').forEach(el=>el.classList.toggle('future',+el.dataset.r>b));slide.querySelectorAll('[data-only]').forEach(el=>el.classList.toggle('only-hidden',+el.dataset.only!==b));slide.querySelectorAll('[data-figure]').forEach(el=>sourceFigure(el,b));slide.querySelectorAll('.interactive').forEach(draw);}
function resize(){document.documentElement.style.setProperty('--scale',Math.min(innerWidth/1280,innerHeight/720));}
function go(i,b=0,hash=true){current=Math.max(0,Math.min(showBackup?slides.length-1:mainCount-1,i));build=Math.max(0,Math.min(DATA.slides[current].builds-1,b));slides.forEach((s,j)=>s.classList.toggle('active',j===current));slides[current].dataset.build=build;updateSlide(slides[current]);document.querySelector('.counter').textContent=`${current+1} / ${mainCount} · ${build+1}/${DATA.slides[current].builds}`;document.getElementById('prev').disabled=current===0&&build===0;document.getElementById('next-btn').disabled=current===(showBackup?slides.length-1:mainCount-1)&&build===DATA.slides[current].builds-1;if(hash)history.replaceState(null,'',`#${DATA.slides[current].id}/${build+1}`);if(document.body.classList.contains('read'))slides[current].scrollIntoView();}
function advance(d){if(d>0){if(build<DATA.slides[current].builds-1)go(current,build+1);else go(current+1);}else{if(build>0)go(current,build-1);else if(current>0)go(current-1,DATA.slides[current-1].builds-1);}}
function toggleMode(){let present=!document.body.classList.contains('present');document.body.classList.toggle('present',present);document.body.classList.toggle('read',!present);if(!present)slides.forEach((s,i)=>{s.dataset.build=DATA.slides[i].builds-1;updateSlide(s);});else go(current,build);resize();}
function showDialog(title,body){let d=document.getElementById('dialog');document.getElementById('dialog-title').textContent=title;document.getElementById('dialog-content').innerHTML=body;d.showModal();}
function notes(){const s=DATA.slides[current];showDialog('Presenter note',`<p>${s.notes}</p><p><strong>Ask:</strong> ${s.question}</p><p><strong>Expected answer:</strong> ${s.answer}</p><p>${s.start_label} · ${s.seconds} seconds</p>${s.readings?.length?'<p class="reading-links"><strong>Further reading:</strong> '+s.readings.map(r=>`<a href="${r.url}" target="_blank" rel="noopener">${r.title}</a>`).join(' · ')+'</p>':''}`);}
function overview(){showDialog('Lecture overview',DATA.slides.filter(s=>showBackup||!s.appendix).map((s,i)=>`<button data-jump="${i}">${String(i+1).padStart(2,'0')} · ${s.title}${s.appendix?' (backup)':''}</button>`).join(''));document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>{document.getElementById('dialog').close();go(+b.dataset.jump);});}
bindInteractions();slides.forEach((s,i)=>{s.dataset.build=document.body.classList.contains('read')?DATA.slides[i].builds-1:0;updateSlide(s);});
document.getElementById('prev').onclick=()=>advance(-1);document.getElementById('next-btn').onclick=()=>advance(1);document.getElementById('mode').onclick=toggleMode;document.getElementById('notes').onclick=notes;document.getElementById('overview').onclick=overview;document.getElementById('tools-button').onclick=()=>{let el=document.getElementById('tools');el.hidden=!el.hidden;document.getElementById('tools-button').setAttribute('aria-expanded',String(!el.hidden));};document.getElementById('backup').onclick=()=>{showBackup=!showBackup;document.querySelectorAll('.appendix').forEach(s=>s.hidden=!showBackup);document.getElementById('backup').textContent=showBackup?'Hide backup':'Show backup';if(!showBackup&&current>=mainCount)go(mainCount-1);};document.getElementById('dialog-close').onclick=()=>document.getElementById('dialog').close();
addEventListener('keydown',ev=>{if(document.getElementById('dialog').open)return;if(ev.target.closest('input,select,textarea,.decoder-grid'))return;if(ev.target.closest('button')&&[' ','Enter'].includes(ev.key))return;if(['ArrowRight',' ','PageDown'].includes(ev.key)){ev.preventDefault();advance(1);}if(['ArrowLeft','PageUp'].includes(ev.key)){ev.preventDefault();advance(-1);}if(ev.key.toLowerCase()==='p')toggleMode();if(ev.key.toLowerCase()==='o')overview();if(ev.key.toLowerCase()==='s')notes();if(ev.key.toLowerCase()==='n')go(current+1);if(ev.key==='Home')go(0);if(ev.key==='End')go(mainCount-1);if(ev.key.toLowerCase()==='q'){let s=DATA.slides[current];showDialog('Discuss',`<p>${s.question}</p><details><summary>Reveal answer</summary><p>${s.answer}</p></details>`);}});
addEventListener('resize',resize);resize();let hash=location.hash.slice(1).split('/'),idx=DATA.slides.findIndex(s=>s.id===hash[0]);if(idx>=mainCount){showBackup=true;document.querySelectorAll('.appendix').forEach(s=>s.hidden=false);}go(idx<0?0:idx,Math.max(0,(+hash[1]||1)-1),false);
// Authoring/QA API; avoids synthetic keypresses for exhaustive build exports.
window.lecture={go,advance,slides:DATA.slides,decode,draw,reset(){document.querySelectorAll('.interactive').forEach(el=>{state.delete(el);st(el);});},setMode(mode){let p=mode==='present';if(document.body.classList.contains('present')!==p)toggleMode();},showBackup(value){showBackup=value;document.querySelectorAll('.appendix').forEach(s=>s.hidden=!value);},get current(){return {slide:current,build};}};
window.lectureReady=true;

// Keep direct links useful in reading mode after fonts and images settle.
function followFragment(){const [id,k]=location.hash.slice(1).split('/'),i=DATA.slides.findIndex(s=>s.id===id);if(i>=0)go(i,Math.max(0,(+k||1)-1),false);}
addEventListener('hashchange',followFragment);
addEventListener('load',()=>{document.fonts.ready.then(()=>requestAnimationFrame(()=>{if(location.hash&&document.body.classList.contains('read'))slides[current].scrollIntoView();}));},{once:true});
