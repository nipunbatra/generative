import {m,eq,at,only,cols,code,table,palette,esc} from '../shared/author.js';
import {panel,label,legend,fig,svg,widget,num,lin,TINT,MUTED,AX,digitSVG,svgFrame,pic,photo,arrowHead,CSS,CSS_BASE,mount as mountWidgets} from './widgets.js';
import {LIN,COS,T,TOY,TOY_EPS,TOY_Z,npdf,pInk,epsStar,reverseKernel,irreducible,forwardStep,reverseStep} from './math.mjs';

const load=f=>fetch(new URL(`./data/${f}`,import.meta.url),{cache:'no-cache'}).then(r=>r.json());
const [px,mn,toy,W,ex,anchors]=await Promise.all(['pixel.json','mnist.json','toy.json','toy-weights.json','extra.json','notebook-anchors.json'].map(load));
const ANCHORS=new Set(anchors),D={px,mn,toy,W,ex};
if(typeof document!=='undefined'&&!document.getElementById('l5-css'))document.head.insertAdjacentHTML('beforeend',`<style id="l5-css">${CSS_BASE}${CSS}</style>`);
export const mount=(stage,ctx)=>mountWidgets(stage,ctx,D);
const R=String.raw;
/** A number for KaTeX: fixed decimals, ASCII minus (KaTeX typesets it), no "-0.00". */
const tn=(v,n=2)=>{const s=Number(v).toFixed(n);return /^-0(\.0*)?$/.test(s)?s.slice(1):s;};
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const r=px.run,two=px.two_steps,PIX=px.pixel,PI=px.pi;
const fr=k=>mn.film.rows[mn.film.t.indexOf(k)];
export const meta={
 id:'lecture4',number:4,title:'Diffusion models',
 subtitle:'Create by learning to undo noise',minutes:138,
 notebook:'lecture4-diffusion.ipynb',
 sections:[
  {id:'map',title:'Lecture map',minutes:1},
  {id:'open',title:'Introduction · Destruction as creation',minutes:7},
  {id:'fwd',title:'Part 1 · The forward process',minutes:22},
  {id:'back',title:'Part 2 · The reverse process',minutes:10},
  {id:'train',title:'Part 3 · What the network learns, and training',minutes:20},
  {id:'gen',title:'Part 4 · Sampling: birth of an image',minutes:20},
  {id:'score',title:'Part 5 · The score: arrows to the data',minutes:8},
  {id:'fast',title:'Part 6 · Noise schedules and faster sampling',minutes:14},
  {id:'steer',title:'Part 7 · Steering: labels, guidance and text',minutes:28},
  {id:'wrap',title:'Part 8 · Wrap-up',minutes:8},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
/** Numbered sections, in the order of Paras Chopra's explainer. The notebook uses the same numbers, with anchors sec-0-1 … sec-8-1. */
const CHAPTERS=[
 ['0.1','Static in, a cat out'],['0.2','Destruction as creation'],
 ['1.1','One noising step'],['1.2','A toy schedule, by hand'],['1.3','A beautiful shortcut'],['1.4','The real schedule'],['1.5','Watch it dissolve'],
 ['2.1','Run the film backwards'],['2.2','Small steps back are simple'],['2.3','Guess the noise'],['2.4','One step back'],
 ['3.1','What does the network learn?'],['3.2','The network'],['3.3','The training loop'],['3.4','Training, live'],['3.5','What the network believes'],
 ['4.1','One step back, by hand'],['4.2','The toy, all the way back'],['4.3','The sampling loop'],['4.4','Watch it'],['4.5','Why the fresh noise?'],['4.6','New digits'],
 ['5.1','Arrows to the data'],['5.2','Follow the arrows'],
 ['6.1','Noise schedules'],['6.2','DDIM: fewer steps'],['6.3','Faster still'],
 ['7.1','A 3 on demand'],['7.2','Classifier-free guidance'],['7.3','The guidance dial'],['7.4','From pixels to latents'],['7.5','Text in'],['7.6','A sentence in, a picture out'],
 ['8.1','Wrap-up'],
];
const PARTS=[['0','Introduction: destruction as creation'],['1','The forward process'],['2','The reverse process'],['3','What the network learns, and training'],['4','Sampling: birth of an image'],['5','The score: arrows to the data'],['6','Noise schedules and faster sampling'],['7','Steering: labels, guidance and text'],['8','Wrap-up']];
const COLAB='https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture4/lecture4-diffusion.ipynb';
const secAnchor=k=>`sec-${k.replace('.','-')}`;
const firstOf={};
/** Section markers in the slide definitions below are reading aids only: the route at the end of the file
 * puts every slide in its numbered section (or in optional depth) and gives it its notebook link. */
const chap=k=>{if(!CHAPTERS.some(ch=>ch[0]===k)&&!/^[0-9]\.[0-9]$/.test(k))throw new Error('Unknown section '+k);};
/** A link to one notebook cell in Colab; the anchor must exist in the built notebook. */
function nbLink(anchor,number){
 if(!ANCHORS.has(anchor))throw new Error(`Notebook anchor ${anchor} is missing: rebuild the notebook`);
 return {href:`${COLAB}#scrollTo=${anchor}`,label:number?`▶ Notebook ${number}`:'▶ Notebook'};
}
/** The bottom bar carries one story line per build; questions and answers live in the notes. */
function S(id,section,title,o){
 const builds=o.builds||1;
 for(const k of ['lines','ask','expected'])if(!Array.isArray(o[k])||o[k].length!==builds)throw new Error(`${id}: ${k} needs ${builds} entries`);
 if(slides.some(s=>s.id===id))throw new Error('Duplicate slide '+id);
 slides.push({id,section,title,builds,body:o.body,prompts:o.lines,chapter:'',nbAnchor:o.nb||null,notebook:null,
  notes:{ask:o.ask,expected:o.expected,point:o.point||'',misconception:o.misconception||'',transition:o.transition||'',demo:o.demo||''},
  source:o.source||'',optional:!!o.optional});
}
const ACTS=['Forward','Reverse','Train','Sample','Score','Faster','Steer','Wrap-up'];
const acts=k=>`<div class="act-strip" role="img" aria-label="Lecture map: part ${k} of ${ACTS.length}">${ACTS.map((a,i)=>`<span class="${i+1<k?'done':i+1===k?'now':''}">${i+1} · ${a}</span>`).join('')}</div>`;
const act=(k,...a)=>`${acts(k)}<div class="act-body">${a.join('')}</div>`;
const tags={pub:['published',palette.d],ddpm:['the original DDPM model',palette.d],ours:['our model',palette.q],exact:['exact',palette.p],illus:['illustration',MUTED]};
const tag=k=>`<span class="tag" style="color:${tags[k][1]}">${tags[k][0]}</span>`;

// ---------------------------------------------------------------- sources, credited on the slides that use them
const SRC={
 paras:'<a href="https://paraschopra.github.io/explainers/diffusion/index.html">Paras Chopra, “Diffusion Models from First Principles” (2025)</a>',
 seff:'<a href="https://www.youtube.com/watch?v=fbLgFrlTnGU">Ari Seff, “What are Diffusion Models?” (2022)</a>',
 deepia:'<a href="https://www.youtube.com/watch?v=EhndHhIvWWw">Deepia, “Diffusion Models: DDPM, Generative AI Animated” (2025)</a>',
 outlier:'<a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier, “Diffusion Models, Paper Explanation, Math Explained” (2022)</a>',
 vizuara:'<a href="https://www.youtube.com/watch?v=bVrweqE-r38">Vizuara, “Principles of Diffusion Models”, Lecture 1 (2025)</a>',
 mit:k=>`<a href="https://www.youtube.com/playlist?list=PL_1TbuIu65A_G908tHHvTnyQsueR17rMh">MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lecture ${k}</a>`,
 ddpm:'<a href="https://arxiv.org/abs/2006.11239">Ho, Jain &amp; Abbeel, “Denoising Diffusion Probabilistic Models” (2020)</a>',
 sohl:'<a href="https://arxiv.org/abs/1503.03585">Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” (2015)</a>',
 song:'<a href="https://arxiv.org/abs/1907.05600">Song &amp; Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution” (2019)</a>',
 sde:'<a href="https://arxiv.org/abs/2011.13456">Song et al., “Score-Based Generative Modeling through SDEs” (2021)</a>',
 ddim:'<a href="https://arxiv.org/abs/2010.02502">Song, Meng &amp; Ermon, “Denoising Diffusion Implicit Models” (2021)</a>',
 iddpm:'<a href="https://arxiv.org/abs/2102.09672">Nichol &amp; Dhariwal, “Improved Denoising Diffusion Probabilistic Models” (2021)</a>',
 cg:'<a href="https://arxiv.org/abs/2105.05233">Dhariwal &amp; Nichol, “Diffusion Models Beat GANs on Image Synthesis” (2021)</a>',
 cfg:'<a href="https://arxiv.org/abs/2207.12598">Ho &amp; Salimans, “Classifier-Free Diffusion Guidance” (2022)</a>',
 ldm:'<a href="https://arxiv.org/abs/2112.10752">Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models” (2022)</a>',
 weng:'<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">Lilian Weng, “What are Diffusion Models?” (2021)</a>',
 luo:'<a href="https://arxiv.org/abs/2208.11970">Calvin Luo, “Understanding Diffusion Models: A Unified Perspective” (2022)</a>',
 film:'Companion film “Diffusion models, from noise to images, step by step” (2026): google/ddpm-cat-256 run on the CC0 photo “Chelsea” (scikit-image)',
 sd:'Stable Diffusion v1.5, run for the companion film “Diffusion models, from noise to images, step by step” (2026)',
 lcm:'<a href="https://arxiv.org/abs/2310.04378">Luo et al., “Latent Consistency Models” (2023)</a>',
 p2p:'<a href="https://arxiv.org/abs/2208.01626">Hertz et al., “Prompt-to-Prompt Image Editing with Cross Attention Control” (2022)</a>',
};
const by=(...k)=>k.join(' · ');

// ---------------------------------------------------------------- figures: our 3 and its pixel
const held=mn.held,xtOf=(t,S=LIN)=>held.x0.map((v,i)=>Math.sqrt(S.ab[t])*v+Math.sqrt(1-S.ab[t])*held.eps[i]);
const pixIdx=PIX[0]*28+PIX[1];
function pixelFig(){
 const H=300;let s=digitSVG(held.x0,{size:280,mark:PIX,x:10,y:10,label:'Our 3, with one pixel outlined'});
 s+=`<line x1="${10+(PIX[1]+1.2)*10}" y1="${10+(PIX[0]+.5)*10}" x2="345" y2="150" stroke="${palette.noise}" stroke-width="2"/>`;
 s+=label(355,140,`pixel (${PIX[0]}, ${PIX[1]})`,{fill:palette.noise,size:26,weight:650})+label(355,180,`its value x₀ = +1: ink`,{fill:palette.data,size:24})+label(355,215,'paper is −1, ink is +1',{size:20});
 return fig(700,H,s,'Our 3, with one ink pixel outlined.');
}
/** The 3 at several noise levels, computed with one fixed noise draw: the forward chain. */
function chainFig(b){
 const ts=[0,100,250,500,1000],W=1150,H=330,sz=140,gap=(W-40-5*sz)/4;let s=`<defs>${arrowHead('ch-f',palette.noise)}${arrowHead('ch-r',palette.q)}</defs>`;
 ts.forEach((t,k)=>{const x=20+k*(sz+gap);s+=digitSVG(xtOf(t),{size:sz,x,y:95})+label(x+sz/2,268,t===0?'x₀':t===T?`x_T, T = ${T}`:`x_${t}`,{size:23,anchor:'middle',fill:palette.ink,weight:600});
  if(k<4){const x1=x+sz+8,x2=x+sz+gap-8;s+=`<path d="M${x1},120 C${(x1+x2)/2},70 ${(x1+x2)/2},70 ${x2},120" fill="none" stroke="${palette.noise}" stroke-width="2.6" marker-end="url(#ch-f)"/>`;
   if(b>=2)s+=`<path d="M${x2},210 C${(x1+x2)/2},262 ${(x1+x2)/2},262 ${x1},210" fill="none" stroke="${palette.q}" stroke-width="2.6" stroke-dasharray="7 5" marker-end="url(#ch-r)"/>`;}});
 s+=label(20,40,'q: add noise, step by step. Fixed: nothing to learn.',{fill:palette.noise,size:24,weight:600});
 if(b>=2)s+=label(20,318,'p_θ: remove noise, step by step. Learned: today’s network.',{fill:palette.q,size:24,weight:600});
 return fig(W,H,s,'Our 3 at t = 0, 100, 250, 500 and 1000: noise is added left to right; the network learns to go right to left.');
}
function pixelHistFig(){
 const h=px.hist,tot=h.reduce((a,b)=>a+b,0),P=panel({W:520,H:300,L:52,R:44,T:50,B:52,xMin:-1,xMax:1,yMax:.62,ticksX:[-1,0,1],fmtX:v=>v===-1?'−1':v===1?'+1':'0',yAxis:false,xLabel:'paper (−1) … ink (+1)'});
 let s=P.axes;h.forEach((c,k)=>{const x1=P.x(-1+2*k/20),x2=P.x(-1+2*(k+1)/20),v=c/tot;s+=`<rect x="${(x1+1).toFixed(1)}" y="${P.y(v).toFixed(1)}" width="${(x2-x1-2).toFixed(1)}" height="${(P.y(0)-P.y(v)).toFixed(1)}" fill="${k<10?palette.noise:palette.data}" fill-opacity=".75"/>`;});
 return fig(520,300,s+label(16,30,`pixel (${PIX[0]}, ${PIX[1]}) in 60,000 training images`,{size:20}),'Histogram of one pixel across the training set: mostly paper, sometimes ink, a little grey in between.');
}
function bumpsFig(b){
 const t=r.t,a=r.alpha_bar,v=1-a,sg=Math.sqrt(a),xs=lin(-3,3,301),P=panel({W:620,H:320,L:16,R:14,T:64,B:52,xMin:-3,xMax:3,yMax:.62,ticksX:[-2,-1,0,1,2],yAxis:false,xLabel:'x_t'});
 let s=P.axes+P.area(xs,xs.map(u=>npdf(u,sg,v)),TINT.data)+P.line(xs,xs.map(u=>npdf(u,sg,v)),palette.data)+P.area(xs,xs.map(u=>npdf(u,-sg,v)),TINT.noise)+P.line(xs,xs.map(u=>npdf(u,-sg,v)),palette.noise);
 s+=legend(16,30,[[`if ink: N(+${num(sg,2)}, ${num(v,2)})`,palette.data],[`if paper: N(−${num(sg,2)}, ${num(v,2)})`,palette.noise]],20);
 if(b>=1){s+=P.vline(r.xt,palette.ink,'stroke-dasharray="6 5"')+P.dot(r.xt,r.lik_ink,palette.data)+P.dot(r.xt,r.lik_paper,palette.noise)+label(P.x(r.xt)+10,P.y(r.lik_ink)-6,num(r.lik_ink,2),{fill:palette.data,size:22,weight:600})+label(P.x(r.xt)+10,P.y(r.lik_paper)+24,num(r.lik_paper,2),{fill:palette.noise,size:22,weight:600})+label(P.x(r.xt)+8,P.y(.58),`xₜ = ${num(r.xt,2)}`,{fill:palette.ink,size:20});}
 return fig(620,320,s,`The two possible distributions of the noisy pixel at t = ${t}, and the observed value.`);
}
function spikesFig(){
 const K=reverseKernel(0,T,0,PI),P=panel({W:560,H:300,L:16,R:14,T:44,B:52,xMin:-1.6,xMax:1.6,yMax:.8,ticksX:[-1,-.3,0,1],fmtX:v=>v===-1?'−1':v===1?'+1':num(v,v===0?0:1),yAxis:false,xLabel:'x₀'}),xs=lin(-1.6,1.6,301);
 let s=P.axes+[[1,K.weights[0]],[-1,K.weights[1]]].map(([v,w])=>`<line x1="${P.x(v)}" y1="${P.y(0)}" x2="${P.x(v)}" y2="${P.y(w)}" stroke="${palette.p}" stroke-width="8"/>`+label(P.x(v),P.y(w)-12,num(w,2),{fill:palette.p,size:22,anchor:'middle',weight:600})).join('');
 const fy=xs.map(K.fit),mx=Math.max(...fy);s+=P.line(xs,fy.map(v=>v/mx*.42),palette.q,'stroke-dasharray="9 6"')+P.vline(K.fitMean,palette.q)+label(P.x(K.fitMean)+8,P.y(.66),`mean ${num(K.fitMean,2)}: grey`,{fill:palette.q,size:21});
 return fig(560,300,s+legend(16,24,[['where x₀ could be',palette.p],['best single Gaussian',palette.q,true]],19),'From pure noise straight back to the clean pixel: two spikes, and the best single Gaussian between them.');
}
function productFig(){
 const pi=PI,t=500,S=LIN,xt=.2,xs=lin(-2.2,2.2,401),a=S.alpha[t],b=S.beta[t],ab1=S.ab[t-1];
 const prior=x=>pi*npdf(x,Math.sqrt(ab1),1-ab1)+(1-pi)*npdf(x,-Math.sqrt(ab1),1-ab1),lik=x=>npdf(xt,Math.sqrt(a)*x,b);
 const pr=xs.map(prior),li=xs.map(lik),post=xs.map((x,i)=>pr[i]*li[i]),mp=Math.max(...pr),ml=Math.max(...li),mq=Math.max(...post);
 const P=panel({W:900,H:205,L:16,R:14,T:44,B:52,xMin:-2.2,xMax:2.2,yMax:1.08,ticksX:[-2,-1,0,1,2],yAxis:false,xLabel:'xₜ₋₁'});
 return fig(900,205,P.axes+P.area(xs,pr.map(v=>v/mp*.55),TINT.p)+P.line(xs,pr.map(v=>v/mp*.55),palette.p)+P.line(xs,li.map(v=>v/ml),palette.noise,'style="stroke-width:2"')+P.line(xs,post.map(v=>v/mq*.95),palette.q,'stroke-dasharray="8 5"')+
  legend(16,26,[['prior: wide',palette.p],['likelihood: narrow',palette.noise],['product: one bell',palette.q,true]],19),'A wide smooth prior times a narrow likelihood gives a narrow, nearly Gaussian product.');
}
function scheduleFig(b){
 const ts=lin(0,1000,201),S=LIN,P=panel({W:560,H:300,L:58,R:14,T:46,B:52,xMin:0,xMax:1000,yMax:1.02,ticksX:[0,250,500,750,1000],ticksY:[0,.5,1],xLabel:'t'});
 let s=P.axes+P.line(ts,ts.map(t=>Math.sqrt(S.ab[Math.round(t)])),palette.data)+P.line(ts,ts.map(t=>Math.sqrt(1-S.ab[Math.round(t)])),palette.noise)+P.line(ts,ts.map(t=>S.beta[Math.round(t)]*25),MUTED,'stroke-dasharray="7 5" style="stroke-width:2"');
 s+=legend(58,26,[['signal √ᾱₜ',palette.data],['noise √(1−ᾱₜ)',palette.noise],['βₜ × 25',MUTED,true]],19);
 if(b>=2){const Q=panel({ox:590,W:560,H:300,L:58,R:14,T:46,B:52,xMin:0,xMax:1000,yMin:-5,yMax:4,ticksX:[0,250,500,750,1000],ticksY:[-4,-2,0,2,4],fmtY:v=>v===0?'1':`10${v<0?'⁻':''}${'⁰¹²³⁴⁵'[Math.abs(v)]}`,xLabel:'t'});
  const snr=ts.map(t=>{const a=S.ab[Math.max(1,Math.round(t))];return Math.log10(a/(1-a));});s+=Q.axes+Q.line(ts,snr,palette.q)+label(648,26,'signal-to-noise ᾱ/(1−ᾱ), log scale',{fill:palette.q,size:19})+Q.dot(250,Math.log10(r.alpha_bar/(1-r.alpha_bar)),palette.q);}
 return fig(1150,300,s,'The linear schedule: beta grows slowly; the signal coefficient falls from 1 to almost 0, the noise coefficient rises to 1.');
}
function noiseImg(src,k,n,sz,title,color,x){return svgFrame(src,k,n,x,34,sz)+label(x,24,title,{fill:color,size:20,weight:600});}
function tupleFig(b){
 const W=1150,H=330,sz=150;let s=`<defs>${arrowHead('tp-a',palette.ink)}</defs>`;
 s+=digitSVG(held.x0,{size:sz,x:0,y:34})+label(0,24,'x₀: a training image',{fill:palette.data,size:20,weight:600});
 s+=noiseImg('mnist-eps.png',0,1,sz,'ε ~ N(0, I): fresh noise',palette.noise,190);
 s+=label(372,118,'t = 250',{fill:palette.ink,size:26,weight:650})+label(372,148,'(a random level)',{size:18});
 if(b>=2){s+=`<path d="M520,110 H560" stroke="${palette.ink}" stroke-width="2.4" marker-end="url(#tp-a)"/>`+digitSVG(xtOf(250),{size:sz,x:572,y:34})+label(572,24,'x_t = √ᾱ x₀ + √(1−ᾱ) ε',{fill:palette.ink,size:20,weight:600});}
 if(b>=3){s+=`<path d="M732,110 H772" stroke="${palette.ink}" stroke-width="2.4" marker-end="url(#tp-a)"/><rect x="780" y="70" width="120" height="80" rx="8" fill="#FFFFFF" stroke="${palette.q}" stroke-width="2.5"/>`+label(840,108,'network',{fill:palette.q,size:22,anchor:'middle',weight:600})+label(840,134,'sees x_t, t',{fill:palette.q,size:17,anchor:'middle'})+
  `<path d="M900,110 H940" stroke="${palette.ink}" stroke-width="2.4" marker-end="url(#tp-a)"/>`+noiseImg('mnist-film-eps.png',9,16,sz,'ε̂: its guess',palette.q,950)+
  label(640,262,'loss = ‖ε − ε̂‖²: the target is the noise we drew',{fill:palette.loss,size:23,weight:600,anchor:'middle'})+`<path d="M265,192 C265,300 1025,300 1025,192" fill="none" stroke="${palette.loss}" stroke-width="2" stroke-dasharray="7 5"/>`;}
 return fig(W,H,s,'One training example: an image, a noise draw and a level make a noisy input; the network guesses the noise; the loss compares.');
}
function timeFeatFig(){
 const W=1150,H=190,L=60,dim=16,half=8,freq=Array.from({length:half},(_,k)=>Math.exp(-Math.log(10000)*k/(half-1)));
 let s='';const cols=200;for(let i=0;i<cols;i++){const t=1+Math.round(i/(cols-1)*999);for(let j=0;j<dim;j++){const v=j<half?Math.sin(t*freq[j]):Math.cos(t*freq[j-half]);const c=v>=0?`rgba(139,44,222,${(v*.85).toFixed(2)})`:`rgba(170,78,8,${(-v*.85).toFixed(2)})`;s+=`<rect x="${(L+i*(W-L-20)/cols).toFixed(1)}" y="${(30+j*8).toFixed(1)}" width="${((W-L-20)/cols+.5).toFixed(1)}" height="8.2" fill="${c}"/>`;}}
 s+=label(L-8,46,'fast',{size:16,anchor:'end'})+label(L-8,156,'slow',{size:16,anchor:'end'})+[1,250,500,750,1000].map(t=>label(L+(t-1)/999*(W-L-20),184,String(t),{size:18,anchor:'middle'})).join('');
 return fig(W,H,s+label(L,20,'16 time features of t: sin and cos at 8 frequencies (purple +, orange −)',{size:19}),'Sinusoidal time features for every step t: fast and slow waves, like token positions in a transformer.');
}
function irrFig(){
 const P=panel({W:560,H:220,L:62,R:28,T:46,B:52,xMin:0,xMax:1000,yMax:.5,ticksX:[0,250,500,750,1000],ticksY:[0,.25,.5],xLabel:'t'}),ir=px.irreducible;
 const Q=panel({ox:590,W:560,H:220,L:62,R:28,T:46,B:52,xMin:0,xMax:1000,yMax:.2,ticksX:[0,250,500,750,1000],ticksY:[0,.1,.2],xLabel:'t'}),lt=mn.loss_by_t;
 return fig(1150,220,P.axes+P.line(ir.map(x=>x[0]),ir.map(x=>x[1]),palette.loss)+P.dot(r.t,irreducible(r.t,PI),palette.loss)+label(62,26,'one pixel: the least possible loss',{fill:palette.loss,size:20})+
  Q.axes+Q.line(lt.map(x=>x[0]),lt.map(x=>x[1]),palette.q)+label(652,26,'our MNIST network: its loss at each t',{fill:palette.q,size:20}),'Left: the loss no predictor can beat, for one pixel, by noise level. Right: our trained network’s loss by noise level.');
}
function mlpFig(){
 const W=1150,H=320;let s=`<defs>${arrowHead('ml-a',palette.ink)}</defs>`;
 const col=(x,n,c,lab,sub)=>{let g='';for(let k=0;k<n;k++)g+=`<circle cx="${x}" cy="${80+k*(200/(n-1||1))}" r="9" fill="#FFFFFF" stroke="${c}" stroke-width="2.4"/>`;return g+label(x,52,lab,{fill:c,size:20,anchor:'middle',weight:600})+label(x,312,sub,{size:17,anchor:'middle'});};
 s+=col(80,3,palette.data,'input','x, y, time')+col(270,6,palette.ink,'128','SiLU')+col(420,6,palette.ink,'128','SiLU')+col(570,6,palette.ink,'128','SiLU')+col(740,2,palette.q,'output','ε̂ₓ, ε̂ᵧ');
 for(const [a,b] of [[95,255],[285,405],[435,555],[585,725]])s+=`<path d="M${a},180 H${b}" stroke="${AX}" stroke-width="2" marker-end="url(#ml-a)"/>`;
 const D=toy.data.x,Qx=v=>860+(v+1.2)/2.4*270,Qy=v=>40+(1.2-v)/2.4*270;
 s+=`<rect x="860" y="40" width="270" height="270" fill="#FFFFFF" stroke="#E5E7EB"/>`+D.filter((_,k)=>k%2===0).map(([x,y])=>`<circle cx="${Qx(x).toFixed(1)}" cy="${Qy(y).toFixed(1)}" r="1.8" fill="${palette.data}" fill-opacity=".6"/>`).join('')+label(860,28,'the data: 2-D points',{fill:palette.data,size:19});
 return fig(W,H,s,'A small MLP: the point and its time features in, three hidden layers of 128, the predicted 2-D noise out.');
}
function unetFig(b){
 const W=1150,H=330;let s=`<defs>${arrowHead('un-a',palette.ink)}${arrowHead('un-s',palette.d)}</defs>`;
 const blk=(x,y,w,h,txt,sub,c=palette.ink)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="#FFFFFF" stroke="${c}" stroke-width="2.4"/>`+label(x+w/2,y+h/2+2,txt,{fill:c,size:20,anchor:'middle',weight:600})+(sub?label(x+w/2,y+h/2+24,sub,{size:16,anchor:'middle'}):'');
 s+=digitSVG(xtOf(250),{size:96,x:0,y:40})+label(48,160,'x_t, 28×28',{size:17,anchor:'middle'});
 s+=blk(130,40,120,90,'28×28','32 channels')+blk(300,120,120,80,'14×14','64')+blk(470,200,120,70,'7×7','128')+blk(640,120,120,80,'14×14','64')+blk(810,40,120,90,'28×28','32');
 for(const [a,b2] of [[[250,95],[300,150]],[[420,170],[470,225]],[[590,225],[640,170]],[[760,150],[810,95]]])s+=`<path d="M${a[0]},${a[1]} L${b2[0]},${b2[1]}" stroke="${palette.ink}" stroke-width="2.2" marker-end="url(#un-a)"/>`;
 s+=`<path d="M950,85 H990" stroke="${palette.ink}" stroke-width="2.2" marker-end="url(#un-a)"/>`+svgFrame('mnist-film-eps.png',9,16,1000,40,96)+label(1048,160,'ε̂, 28×28',{size:17,anchor:'middle',fill:palette.q});
 if(b>=2)s+=`<path d="M250,62 H810" stroke="${palette.d}" stroke-width="2.4" stroke-dasharray="8 5" marker-end="url(#un-s)"/><path d="M420,135 H640" stroke="${palette.d}" stroke-width="2.4" stroke-dasharray="8 5" marker-end="url(#un-s)"/>`+label(530,56,'skip connections: fine detail goes straight across',{fill:palette.d,size:18,anchor:'middle'});
 if(b>=3)s+=`<rect x="300" y="285" width="460" height="38" rx="6" fill="${TINT.noise}" stroke="${palette.noise}" stroke-width="2"/>`+label(530,310,'t → time features → MLP → added inside every block',{fill:palette.noise,size:19,anchor:'middle',weight:600})+[190,360,530,700,870].map(x=>`<path d="M${x},285 V${x===530?272:x===360||x===700?202:132}" stroke="${palette.noise}" stroke-width="1.6" stroke-dasharray="4 4"/>`).join('');
 return fig(W,H,s,'A U-Net: the noisy image is shrunk to 7×7 to see the whole digit, then grown back to 28×28, with skip connections and the time embedding in every block.');
}


// ---------------------------------------------------------------- the toy: five big steps, every number by hand
const toyF=(()=>{const xs=[1];for(let k=1;k<=5;k++)xs.push(forwardStep(xs[k-1],k,TOY_EPS[k],TOY));return xs;})();
const toyFp=(()=>{const xs=[-1];for(let k=1;k<=5;k++)xs.push(forwardStep(xs[k-1],k,TOY_EPS[k],TOY));return xs;})();
const toyR=(()=>{let x=toyF[5];const rows=[];for(let k=5;k>=1;k--){const e=epsStar(x,k,PI,TOY),R=reverseStep(x,k,e,TOY_Z[k],TOY);rows.push({t:k,x,e,...R});x=R.next;}return rows;})();
const f1=v=>num(v,1),f3=v=>num(v,3),fp=(v,n=3)=>v<0?`(${num(v,n)})`:num(v,n);
const cellAt=(b,html)=>b>1?`<span data-at="${b}">${html}</span>`:html;
function toyScheduleTable(){
 const ex4=v=>String(+Number(v).toFixed(4));
 const rows=[1,2,3,4,5].map(k=>{const prod=k===1?'α₁':`${ex4(TOY.ab[k-1])} × ${f1(TOY.alpha[k])}`;
  return [String(k),f1(TOY.beta[k]),`1 − ${f1(TOY.beta[k])} = ${f1(TOY.alpha[k])}`,cellAt(2,`${prod} = <b>${ex4(TOY.ab[k])}</b>`),cellAt(3,f3(Math.sqrt(TOY.ab[k]))),cellAt(3,f3(Math.sqrt(1-TOY.ab[k])))];});
 return `<div class="hand">${table(['t','β<sub>t</sub>','α<sub>t</sub> = 1 − β<sub>t</sub>',cellAt(2,'ᾱ<sub>t</sub> = ᾱ<sub>t−1</sub> × α<sub>t</sub>'),cellAt(3,'signal √ᾱ<sub>t</sub>'),cellAt(3,'noise √(1 − ᾱ<sub>t</sub>)')],rows)}</div>`;
}
function toyForwardTable(){
 const by={1:1,2:1,3:2,4:2,5:3};
 const rows=[1,2,3,4,5].map(k=>[String(k),cellAt(by[k],f3(toyF[k-1])),cellAt(by[k],num(TOY_EPS[k],1)),cellAt(by[k],`${f3(Math.sqrt(TOY.alpha[k]))} × ${fp(toyF[k-1])} + ${f3(Math.sqrt(TOY.beta[k]))} × ${fp(TOY_EPS[k],1)}`),cellAt(by[k],`<b>${f3(toyF[k])}</b>`)]);
 return `<div class="hand">${table(['t','x<sub>t−1</sub>','ε<sub>t</sub>','√α<sub>t</sub> · x<sub>t−1</sub> + √β<sub>t</sub> · ε<sub>t</sub>','x<sub>t</sub>'],rows)}</div>`;
}
function toyReverseTable(){
 const rows=toyR.map((r,k)=>[String(r.t),cellAt(k+1,f3(r.x)),cellAt(k+1,f3(r.e)),cellAt(k+1,`(${f3(r.x)} − ${num(r.c,3)} × ${fp(r.e)}) / ${f3(Math.sqrt(TOY.alpha[r.t]))} = ${f3(r.mu)}`),cellAt(k+1,r.t>1?`${f3(r.sigma)} × ${fp(r.z,1)}`:'none'),cellAt(k+1,`<b>${f3(r.next)}</b>`)]);
 return `<div class="hand">${table(['t','x<sub>t</sub>','ε̂ (network)','mean μ = (x<sub>t</sub> − c<sub>t</sub> ε̂) / √α<sub>t</sub>','fresh noise √β<sub>t</sub> z','x<sub>t−1</sub>'],rows)}</div>`;
}
/** One step of the film: forwards is easy, backwards is the question. */
function backFig(){
 const W=760,H=290,sz=190;let s=`<defs>${arrowHead('bk-f',palette.noise)}${arrowHead('bk-r',palette.q)}</defs>`;
 s+=`<rect x="20" y="50" width="${sz}" height="${sz}" rx="8" fill="#F3F4F6" stroke="#D7DCE1"/>`+label(20+sz/2,50+sz/2+34,'?',{fill:palette.q,size:100,anchor:'middle',weight:700})+label(20,34,'x₂₄₉: one step earlier',{fill:palette.q,size:25,weight:600});
 s+=digitSVG(xtOf(250),{size:sz,x:550,y:50})+label(550,34,'x₂₅₀: what we see',{fill:palette.ink,size:25,weight:600});
 s+=`<path d="M225,112 H535" stroke="${palette.noise}" stroke-width="2.8" marker-end="url(#bk-f)"/>`+label(380,98,'add noise: easy',{fill:palette.noise,size:25,anchor:'middle',weight:600});
 s+=`<path d="M535,184 H225" stroke="${palette.q}" stroke-width="2.8" stroke-dasharray="9 6" marker-end="url(#bk-r)"/>`+label(380,222,'remove noise: ?',{fill:palette.q,size:25,anchor:'middle',weight:600});
 return fig(W,H,s,'One step of the film: adding noise from step 249 to 250 is easy; going back is the question.');
}
/** Real outputs of the original DDPM cat network on one photo, from the companion film. */
const photoRow=(items,sz)=>`<div class="eqimg" style="gap:22px">${items.map(([f,c,col])=>`<figure class="eqimg-f">${photo(0,1,sz,f)}<figcaption style="color:${col}">${c}</figcaption></figure>`).join('')}</div>`;
function guessGrid(){
 const ts=[950,800,600,400,250,100],sz=150,cap=(t,c)=>`<span class="caption" style="color:${c};text-align:right">${t}</span>`;
 return `<div class="diagram-scroll"><div style="display:grid;grid-template-columns:auto repeat(6,${sz}px);gap:8px 12px;align-items:center"><span></span>${ts.map(t=>`<span class="caption" style="text-align:center">t = ${t}</span>`).join('')}`+
  cap('input xₜ',palette.ink)+ts.map((t,k)=>photo(k,6,sz,'cat-guess-xt.jpg')).join('')+cap('its guess x̂₀',palette.data)+ts.map((t,k)=>photo(k,6,sz,'cat-guess-x0.jpg')).join('')+'</div></div>';
}
/** A row of frames cut from one strip of the companion film, each with its caption; optionally labelled on the left. */
const stripRow=(file,n,caps,sz)=>`<div class="eqimg" style="gap:14px">${caps.map((c,k)=>`<figure class="eqimg-f">${photo(k,n,sz,file)}<figcaption>${c}</figcaption></figure>`).join('')}</div>`;
const stripLine=(lab,col,row)=>`<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap"><span class="caption" style="width:150px;flex:none;color:${col};font-weight:650">${lab}</span>${row}</div>`;
const sdArrow=t=>`<span class="op" style="display:flex;flex-direction:column;align-items:center;font-size:20px;gap:2px">${t}<span style="font-size:34px;line-height:1">→</span></span>`;
/** One forward step as a flow of numbers: x0 → × √α → + √β ε → x1. */
function stepFig(){
 const W=700,H=170,x1=Math.sqrt(TOY.alpha[1]),k1=Math.sqrt(TOY.beta[1])*TOY_EPS[1];let s=`<defs>${arrowHead('sf1',palette.ink)}</defs>`;
 const val=(x,v,c,lab)=>`<circle cx="${x}" cy="80" r="44" fill="#FFFFFF" stroke="${c}" stroke-width="3"/>`+label(x,88,v,{fill:c,size:24,anchor:'middle',weight:650})+label(x,152,lab,{size:18,anchor:'middle'});
 s+=val(60,'1.000',palette.data,'x₀ (ink)')+val(350,f3(x1),palette.ink,'shrunk')+val(640,f3(toyF[1]),palette.ink,'x₁');
 s+=`<path d="M108,80 H300" stroke="${palette.ink}" stroke-width="2.4" marker-end="url(#sf1)"/><path d="M398,80 H590" stroke="${palette.noise}" stroke-width="2.4" marker-end="url(#sf1)"/>`;
 s+=label(204,40,'× √α₁',{fill:palette.ink,size:21,anchor:'middle',weight:600})+label(204,66,`× ${f3(x1)}`,{fill:palette.ink,size:19,anchor:'middle'})+label(494,40,'+ √β₁ ε₁',{fill:palette.noise,size:21,anchor:'middle',weight:600})+label(494,66,`+ ${f3(Math.sqrt(TOY.beta[1]))} × ${num(TOY_EPS[1],1)}`,{fill:palette.noise,size:19,anchor:'middle'});
 return fig(W,H,s,'One noising step on the ink pixel: shrink by the square root of alpha, then add a small noise kick.');
}
/** One reverse step as a picture: the network guesses the noise, the rest is arithmetic. */
function reverseFig(){
 const W=1150,H=190,r0=toyR[0];let s=`<defs>${arrowHead('rf1',palette.ink)}</defs>`;
 const box=(x,w,top,bot,c)=>`<rect x="${x}" y="40" width="${w}" height="86" rx="10" fill="#FFFFFF" stroke="${c}" stroke-width="2.6"/>`+label(x+w/2,76,top,{fill:c,size:21,anchor:'middle',weight:650})+label(x+w/2,106,bot,{fill:c,size:18,anchor:'middle'});
 s+=box(0,120,'x₅',f3(r0.x),palette.ink)+box(170,150,'network','sees x₅ and t = 5',palette.q)+box(370,130,'ε̂',`guess ${f3(r0.e)}`,palette.q)+box(550,270,'mean μ',`(x₅ − ${num(r0.c,3)} ε̂)/√α₅ = ${f3(r0.mu)}`,palette.q)+box(870,280,'x₄ = μ + √β₅ z',`${f3(r0.mu)} + ${f3(r0.sigma)} × ${fp(r0.z,1)} = ${f3(r0.next)}`,palette.ink);
 for(const [a,b] of [[120,170],[320,370],[500,550],[820,870]])s+=`<path d="M${a+2},83 H${b-6}" stroke="${palette.ink}" stroke-width="2.4" marker-end="url(#rf1)"/>`;
 s+=label(0,170,'Only the purple box is learned. The fresh noise z is drawn, like ε in the forward step.',{size:19});
 return fig(W,H,s,'One reverse step: the network guesses the noise from x_t and t; the mean removes a little of it; fresh noise is added.');
}

// ================================================================ Opening
chap('0.1');
S('hook','open','Static in, a cat out',{builds:2,nb:'sec-0-1',
 body:widget('filmstrip','Four sampling runs of the original DDPM cat model, from static to cats',{runs:'1,3,4,5',size:220})+at(2,small(`${tag('ddpm')} Real samples from the network of the original DDPM paper (Ho, Jain &amp; Abbeel 2020), trained on photos of cats.`)),
 lines:['The network of the original DDPM paper, trained on photos of cats, starts from pure static: 196,608 random numbers per image…','…and a thousand small steps later there is a cat. Today: how it works, built from scratch on handwritten digits, where we can see every number.'],
 ask:['What could turn random numbers into a photo?','What has the network learned, if not the photos themselves?'],
 expected:['Collect guesses: a decoder (VAE), a generator (GAN)… Today: many small denoising steps.','How to remove a little noise from a noisy image, at every noise level. That is the whole lecture.'],
 point:'Four of eight runs of google/ddpm-cat-256 (the original DDPM LSUN-cat checkpoint, 114 million parameters): 1,000 steps of ancestral sampling, shown every 50 steps. All eight runs, with the network’s guess at every step, are in §{cat-film}.',
 source:by(SRC.ddpm,SRC.film),
 transition:'What this same loop does today.'});
S('apps','open','The same loop runs today’s image generators',{builds:2,
 body:`<div class="thumbs">${[['pub-cherries-1.png','“a basket of cherries”'],['pub-repaint-1.png','fill a hole'],['pub-sr3-face-3.png','sharpen a blurry face'],['pub-sdedit-out1.png','a sketch → a scene'],['pub-control-2.png','edges → a bird'],['pub-dreambooth-acropolis.png','your dog, at the Acropolis']].map(([f,c],k)=>`<figure>${`<img src="figures/pub/${f}" alt="${esc(c)}" style="width:150px;height:150px;object-fit:cover;border-radius:6px">`}<figcaption>${c}</figcaption></figure>`).join('')}</div>`+
  at(2,small(`${tag('pub')} Video, audio, 3-D shapes, protein structures and robot motions are made the same way.`)),
 lines:['Text to image, filling a hole, sharpening a face, a sketch into a scene, edges into a bird, one dog in a new place. Different inputs, one recipe.','The recipe goes beyond images: video, audio, proteins, robot policies. By the end you can say what each network was trained to predict.'],
 ask:['What is the same across all six?','Which of them do you think is hardest?'],
 expected:['Each starts from noise and removes it step by step, steered by a different input.','Open question; return to it at the end.'],
 source:'Published samples: LDM (Rombach et al. 2022) · RePaint (Lugmayr et al. 2022) · SR3 (Saharia et al. 2022) · SDEdit (Meng et al. 2022) · ControlNet (Zhang et al. 2023) · DreamBooth (Ruiz et al. 2023). Showcase after '+SRC.mit(1),
 transition:'The idea in one sentence.'});
chap('0.2');
S('three-gens','open','Two generators so far, and a third idea',{builds:3,
 body:`<div class="columns" style="grid-template-columns:1fr 1fr 1fr;gap:28px;align-items:start">${[
  ['VAE (Lecture 2)','vae-samples.png','one decoder step from z: honest likelihood, blurry digits',palette.p,1],
  ['GAN (Lecture 3)','gan-samples.png','one generator step: sharp, but a fragile game that can collapse',palette.loss,2],
  ['Diffusion (today)','mnist-samples-16.png','a thousand small denoising steps, each one an easy regression',palette.q,3]].map(([h,f,c,col,b])=>at(b,`<div><p class="lead" style="color:${col};font-weight:650">${h}</p>${pic(f,330,h)}<p class="caption" style="margin-top:8px">${c}</p></div>`)).join('')}</div>`,
 lines:['The VAE decodes a random code in one step: it has a likelihood and trains stably, but its digits are blurry.','The GAN also makes a digit in one step: sharp, but training is a game that can collapse.','Diffusion makes a digit in a thousand small steps. Each step is a simple regression problem, and the result is sharp.'],
 ask:['Why are VAE samples blurry?','What went wrong with GANs?','What might we pay for a thousand steps?'],
 expected:['The decoder predicts an average over the plausible images (a Gaussian or Bernoulli mean).','Mode collapse and unstable alternating training; no likelihood to monitor.','Time: a thousand network calls per image. Part ¶{slow} cuts that.'],
 point:'VAE: the 16-number VAE of Lecture 2 (β = 1), prior samples (mean images shown). GAN: Lecture 3’s MNIST GAN after 30 epochs. Diffusion: our model, Part ¶{samples64}.',
 transition:'The idea in one sentence.'});
chap('0.3');
S('core-idea','open','The idea in one sentence',{builds:2,
 body:stack(`<p class="big-quote">If you learn to undo each tiny step of a destruction process, you can create from scratch.<span class="by">Paras Chopra, “Diffusion Models from First Principles”</span></p>`,
  at(2,`<figure class="illus">${`<img src="figures/illustrations/ink-drop.jpg" style="max-height:215px;object-fit:contain" alt="A drop of ink spreading through five glasses of water until the water is a uniform faint blue.">`}<figcaption>${tag('illus')} Generated with Gemini. A drop of ink spreads until the water is uniform; nothing pulls it back together.</figcaption></figure>`)),
 lines:['Destroying is easy; creating is hard. The trick is to connect them.','Film a drop of ink spreading through water. Played forwards, structure disappears. Between two neighbouring frames the change is tiny, and tiny changes can be predicted. Learn to undo each one, and you can run the film backwards.'],
 ask:['Can you un-mix ink from water?','Why would small steps be easier to undo than one big one?'],
 expected:['Not physically: diffusion only runs forwards.','Between neighbouring frames almost nothing changes, so the previous frame is almost determined by the next one. Part ¶{reverse-explorer} makes this exact.'],
 point:'This physical process gave the method its name: Sohl-Dickstein et al. (2015) borrowed it from non-equilibrium thermodynamics. Paras Chopra’s explainer uses a different everyday example; we use ink in water throughout.',
 source:by(SRC.paras,SRC.sohl),
 transition:'The plan for today.'});
S('route','open','The plan: forwards, backwards, train, sample',{builds:2,
 body:`<div class="state-slot"><div data-only="1">${chainFig(1)}</div><div data-only="2">${chainFig(2)}</div></div>`,
 lines:['Left to right: add a little noise, a thousand times, until our 3 is pure static. That direction is fixed and needs no learning (Part 1).','Right to left: remove a little noise at a time. That direction is learned (Parts 2–4). Then the score view (5), faster sampling (6), and steering with labels and text (7).'],
 ask:['Which direction is easy?','What does the network have to learn?'],
 expected:['Adding noise: one line of code.','How to undo one small step of noise, at every noise level.'],
 source:SRC.ddpm,
 transition:'Part 1: the forward process.'});

// ================================================================ Part 1 · The forward process
chap('1.1');
S('one-step','fwd','One noising step: shrink a little, add a little noise',{builds:3,
 body:act(1,cols(pixelFig(),stack(eq(R`x_t=\sqrt{\alpha_t}\;x_{t-1}+\sqrt{\beta_t}\;\ncol{\varepsilon_t}`),eq(R`\alpha_t=1-\beta_t,\qquad \ncol{\varepsilon_t}\sim\N(0,1)`),
  at(2,stepFig()),
  at(3,small('β<sub>t</sub> is how much noise step t adds. Real models take 1,000 tiny steps; to see every number we first use a toy with five big steps.'))),'0.8fr 1.2fr')),
 lines:['Our running example: one ink pixel of our 3 (ink is +1, paper is −1). One noising step multiplies the pixel by √αₜ, a little less than 1, and adds a small Gaussian kick of size √βₜ.',`Toy step 1, with β₁ = 0.1 and the noise draw ε₁ = 0.3: shrink 1 to ${f3(Math.sqrt(.9))}, add ${f3(Math.sqrt(.1)*.3)}, and get x₁ = ${f3(toyF[1])}.`,'Repeat with a growing β. But first: why shrink at all?'],
 ask:['Why √β and not β?',`Check: √0.9 × 1 + √0.1 × 0.3 = ?`,'What would happen without the shrink?'],
 expected:['β is a variance; its square root is the size (standard deviation) of the kick.',`${f3(Math.sqrt(.9))} + ${f3(Math.sqrt(.1)*.3)} = ${f3(toyF[1])}.`,'Next slide: the values spread out more and more.'],
 source:by(SRC.paras,SRC.ddpm),
 transition:'Without the shrink.'});
chap('1.1');
S('explode','fwd','Keep adding noise: what happens to the spread?',{builds:2,
 body:act(1,widget('explode','Histograms of our 3’s pixels after t noising steps, with and without the shrink')),
 lines:['All 784 pixels of our 3, after t steps of adding noise only. Paper and ink blur together, and the spread keeps growing: at t = 1,000 the standard deviation is 3.3, more than three times the image’s range.','Now shrink a little before adding: multiply by √(1 − β), then add √β ε. The spread settles at 1 and stays there.'],
 ask:['Where does 3.3 come from?','Why does the shrink keep the spread at 1?'],
 expected:[`The variances add up: 0.8 (the image) + Σβ = ${num(px.sum_beta,1)}. Every extra step adds more.`,'The next slide: (1 − β) of the old variance plus β of new noise.'],
 point:`Naive noising: Var(x_T) = Var(x₀) + Σβₜ ≈ ${num(px.sum_beta+.8,1)}. With more steps or bigger β it grows without bound, and where the noise ends depends on the image.`,
 source:SRC.deepia,
 transition:'Why the shrink works, in one line.'});
S('shrink-math','fwd','Shrink a little, then add: the spread stays at 1',{builds:3,
 body:act(1,stack(eq(R`x_t=\sqrt{1-\beta_t}\;x_{t-1}+\sqrt{\beta_t}\;\ncol{\varepsilon_t}`),
  at(2,eq(R`\operatorname{Var}(x_t)=(1-\beta_t)\operatorname{Var}(x_{t-1})+\beta_t`)),
  at(2,small('Rule: Var(aX + bε) = a² Var X + b² Var ε when X and ε are independent, and Var ε = 1.')),
  at(3,eq(R`\operatorname{Var}(x_{t-1})=1\;\Rightarrow\;\operatorname{Var}(x_t)=(1-\beta_t)+\beta_t=1`)),
  at(3,small(`And if the variance is not yet 1, the gap shrinks: Var(xₜ) − 1 = (1 − βₜ)(Var(xₜ₋₁) − 1). Our pixel’s first step: √0.9999 × 1 + 0.01 × (−0.5) = 0.995.`)))),
 lines:['The real noising step: shrink the previous value by √(1 − β), then add noise of variance β.','Variance of a sum of independent parts: the shrunk part keeps (1 − β) of the old variance; the noise adds β.','If the variance was 1, it stays exactly 1. If it was not, every step moves it closer to 1.'],
 ask:['Why √(1 − β) and not (1 − β)?','Which rule did we use?','What does this guarantee at the end?'],
 expected:['So that the variance, which goes with the square, is multiplied by exactly 1 − β.','Variances of independent terms add; a constant factor comes out squared.','Whatever the image, after many steps the values have mean 0 and variance 1.'],
 source:by(SRC.deepia,SRC.ddpm),
 transition:'Watch it on the whole 3.'});
S('film','fwd','Slide t: watch our 3 dissolve',{builds:2,
 body:act(1,widget('film','Our 3 at noise level t')),
 lines:['Move t. The image is a mix of two things: the signal, our 3 scaled by √ᾱₜ, and noise scaled by √(1 − ᾱₜ). The pixel we follow is outlined.',`Around t = 600 the 3 is gone: √ᾱ has fallen to ${num(Math.sqrt(LIN.ab[600]),2)}. Our ink pixel (+1) reads ${num(xtOf(500)[pixIdx],2)} at t = 500: from the value alone you can no longer tell ink from paper.`],
 ask:['At what t can you no longer see the 3?','Our pixel is ink. Why can it read 0 or below?'],
 expected:['Around t = 500–600.','The noise part √(1 − ᾱ)ε (here ε = −0.5) outweighs the shrunk signal.'],
 point:'The bars are the shortcut’s two coefficients (§{shortcut}), √ᾱₜ and √(1 − ᾱₜ). The film is computed live from the clean 3 and one fixed noise draw.',
 source:SRC.paras,
 transition:'Every image ends up in the same place.'});
S('pixel-hist','fwd','Every image ends in the same place: N(0, 1)',{builds:2,
 body:act(1,widget('pixhist','Pixel values at noise level t, against N(0, 1)')),
 lines:['At t = 0 the 784 pixels of our 3 sit at −1 (paper) and +1 (ink). Move t: the two bumps blur and merge, and by t = 1,000 they are the standard bell N(0, 1). Every MNIST pixel does the same (green).','Why a Gaussian? Not the Central Limit Theorem: the signal part √ᾱₜ x₀ is shrunk to zero, and what is left, √(1 − ᾱₜ) ε, is Gaussian because we made it Gaussian.'],
 ask:['Would a dataset of faces end in the same place?','Why does it matter that every dataset ends at N(0, 1)?'],
 expected:['Yes: the end point does not depend on the data.','Because then we know how to start the reverse process: draw from N(0, I).'],
 misconception:'Explainers often credit the Central Limit Theorem. The forward noise is Gaussian by construction and the signal is shrunk away; no averaging of many non-Gaussian pieces is involved.',
 source:by(SRC.paras,'(endpoint argument corrected)'),
 transition:'The same on a real photograph.'});
S('river','fwd','One pixel, a thousand steps: the density flows',{builds:2,
 body:act(1,widget('river','The density of our pixel over the thousand steps')),
 lines:[`Our pixel across the dataset: ink (+1) with probability ${num(PI,2)}, paper (−1) otherwise. Its density over time, exactly: two thin lines that widen and merge into one bell.`,'Fourteen noising runs, step by step. Each starts at ink or paper and wanders; by the end you cannot tell where any of them started.'],
 ask:['Where is the density at t = 0?','Pick one path at t = 1,000. Did it start at ink or paper?'],
 expected:['Two spikes: at −1 with probability 0.65 and at +1 with 0.35.','You cannot tell. That is the problem Part ¶{reverse-q} faces.'],
 source:SRC.deepia,
 transition:'Do we need all thousand steps to make one noisy example?'});
chap('1.2');
S('toy-schedule','fwd','A toy schedule you can do by hand: five steps',{builds:3,nb:'cell-toy',
 body:act(1,toyScheduleTable()),
 lines:['Five steps with growing β: 0.1, 0.2, 0.3, 0.5, 0.8. Each α is simply 1 − β: the share of the previous value that a step keeps.',`ᾱₜ (“alpha-bar”) multiplies the α’s so far: 0.9, then 0.9 × 0.8 = 0.72, then × 0.7 = 0.504, × 0.5 = 0.252, × 0.2 = ${tn(TOY.ab[5],4)}. It measures how much of the original image survives.`,`The two numbers we use all lecture: √ᾱₜ, the size of the signal, and √(1 − ᾱₜ), the size of the noise. By step 5 they are ${f3(Math.sqrt(TOY.ab[5]))} and ${f3(Math.sqrt(1-TOY.ab[5]))}: almost pure noise.`],
 ask:['Why does α fall as β grows?','Why multiply the α’s?','Do the squares of the two coefficients add to 1?'],
 expected:['α = 1 − β: more noise means keeping less of the old value.','Every step shrinks the signal again: after t steps, by √(α₁ α₂ ⋯ αₜ) = √ᾱₜ.','Yes, ᾱ + (1 − ᾱ) = 1: signal and noise share a fixed budget, so the spread stays 1.'],
 point:'A toy for the arithmetic only. Real schedules take 1,000 small steps, so that each step back is easy (Part ¶{reverse-explorer}).',
 transition:'Push our pixel through the five steps.'});
S('toy-forward','fwd','Forward, step by step: every number',{builds:3,nb:'cell-toy',
 body:act(1,toyForwardTable(),at(3,p(`From x₅ = ${f3(toyF[5])} alone you cannot tell the pixel was ink. (A paper pixel with the same draws would give ${f3(toyFp[5])}.)`))),
 lines:[`Start at the ink pixel, x₀ = 1. Step 1: ${f3(toyF[1])}. Step 2 shrinks by √0.8 and adds √0.2 × (−0.5): ${f3(toyF[2])}.`,`Steps 3 and 4 have bigger kicks. After step 4 the ink pixel is negative, ${f3(toyF[4])}: it looks like paper.`,`Step 5: x₅ = ${f3(toyF[5])}. The original value is almost gone.`],
 ask:['Compute x₂ yourself.','How can an ink pixel become negative?','Could you recover x₀ from x₅?'],
 expected:[`√0.8 × ${f3(toyF[1])} + √0.2 × (−0.5) = ${f3(toyF[2])}.`,'The noise kick is as large as the shrunk signal.','Not reliably: ink and paper pixels end up in overlapping places. That is Part ¶{reverse-q}’s problem.'],
 transition:'Do we have to run every step?'});
chap('1.3');
S('shortcut-q','fwd','Do we need a 1,000-step loop for one noisy example?',{builds:3,
 body:act(1,stack(eq(R`x_1=\sqrt{\alpha_1}\,\dcol{x_0}+\sqrt{\beta_1}\,\ncol{\varepsilon_1}\qquad(\alpha_t=1-\beta_t)`),
  at(2,eq(R`x_2=\sqrt{\alpha_2}\big(\sqrt{\alpha_1}\,\dcol{x_0}+\sqrt{\beta_1}\,\ncol{\varepsilon_1}\big)+\sqrt{\beta_2}\,\ncol{\varepsilon_2}`)),
  at(3,eq(R`x_3=\sqrt{\alpha_3}\Big(\sqrt{\alpha_2}\big(\sqrt{\alpha_1}\,\dcol{x_0}+\sqrt{\beta_1}\,\ncol{\varepsilon_1}\big)+\sqrt{\beta_2}\,\ncol{\varepsilon_2}\Big)+\sqrt{\beta_3}\,\ncol{\varepsilon_3}\;\cdots`)))),
 lines:['Name the shrink factor: αₜ = 1 − βₜ. One step: x₁ = √α₁ x₀ + √β₁ ε₁.','Two steps: substitute x₁ into the rule for x₂.','Three steps, and it keeps nesting. Training needs millions of noisy examples at random t: we cannot afford this loop each time.'],
 ask:['What kind of random variable is x₂?','How many noise draws does x₁₀₀₀ involve?','Any way to collapse this?'],
 expected:['A constant times x₀ plus a sum of independent Gaussians.','A thousand.','A sum of independent Gaussians is one Gaussian: §{two-steps}.'],
 source:SRC.seff,
 transition:'Collapse two steps by hand.'});
S('two-steps','fwd','Two steps by hand: two noises become one',{builds:4,
 body:act(1,stack(eq(R`x_2=\underbrace{\sqrt{\alpha_2\alpha_1}\,\dcol{x_0}}_{\text{signal}}+\underbrace{\sqrt{\alpha_2\beta_1}\,\ncol{\varepsilon_1}+\sqrt{\beta_2}\,\ncol{\varepsilon_2}}_{\text{two independent Gaussians}}`),
  at(2,eq(R`\operatorname{Var}(\text{noise})=\alpha_2\beta_1+\beta_2=1-\alpha_1\alpha_2`)),
  at(3,eq(R`x_2=\sqrt{\alpha_1\alpha_2}\,\dcol{x_0}+\sqrt{1-\alpha_1\alpha_2}\;\ncol{\varepsilon},\qquad\ncol{\varepsilon}\sim\N(0,1)`)),
  at(4,eq(R`\beta_1=0.1,\ \beta_2=0.2:\quad x_2=\sqrt{0.72}\,(1)+\sqrt{0.28}\,(${tn(two.eps,2)})=${tn(two.x2,2)}`)))),
 lines:['Training needs millions of noisy examples at random t: must we run every step for each one? Substitute x₁ into the rule for x₂ and open the brackets: a signal part, and two independent noise terms.','Their variances add (Lecture 1): α₂β₁ + β₂. Since β = 1 − α, that is exactly 1 − α₁α₂.','So the two noises are one Gaussian with that variance: x₂ = √(α₁α₂) x₀ + √(1 − α₁α₂) ε.',`Check with two big illustrative steps, β₁ = 0.1 and β₂ = 0.2, and the draws ε₁ = 0.3, ε₂ = −0.5: step by step x₂ = ${num(two.x2,2)}, and the formula with the combined ε = ${num(two.eps,2)} gives the same.`],
 ask:['Why may we add the variances?','Check: α₂β₁ + β₂ = 1 − α₁α₂?','What is the signal coefficient after two steps?','Is the combined ε a new random draw?'],
 expected:['ε₁ and ε₂ are independent.','α₂(1 − α₁) + 1 − α₂ = 1 − α₁α₂. Yes.','√(α₁α₂): the product of the shrink factors.','It is a fixed combination of ε₁ and ε₂, with the right N(0, 1) distribution. To make a training example we can draw it directly.'],
 point:`Illustrative numbers only: real steps are tiny (β₁ = 0.0001). Step by step: x₁ = √0.9 + √0.1 × 0.3 = ${num(two.x1,4)}; x₂ = √0.8 × ${num(two.x1,4)} + √0.2 × (−0.5) = ${num(two.x2,4)}. Combined ε = (√0.8 √0.1 × 0.3 + √0.2 × (−0.5)) / √0.28 = ${num(two.eps,4)}.`,
 source:SRC.seff,
 transition:'The same argument for any t.'});
S('shortcut','fwd','The shortcut: jump straight to any step',{builds:3,
 body:act(1,stack(eq(R`x_t=\sqrt{\bar\alpha_t}\;\dcol{x_0}+\sqrt{1-\bar\alpha_t}\;\ncol{\varepsilon},\qquad \bar\alpha_t=\alpha_1\alpha_2\cdots\alpha_t`),
  at(2,eq(R`\text{toy, }t=2:\quad\sqrt{0.72}\times 1+\sqrt{0.28}\times(${tn(two.eps,3)})=${tn(two.x2,3)}\qquad\text{(the same as two steps)}`)),
  at(3,code(`def q_sample(x0, t, eps, alpha_bar):
    a = alpha_bar[t]                        # one number per example
    return a.sqrt() * x0 + (1 - a).sqrt() * eps`)))),
 lines:['The two-step argument works for any t: the signal is shrunk by √ᾱₜ (the product of all the α’s), and all the noises together are one Gaussian with variance 1 − ᾱₜ.',`Check on the toy: one jump with the combined draw ε = ${num(two.eps,3)} gives x₂ = ${num(two.x2,3)}, exactly what two steps gave.`,'In code: one line. Any noise level, any image, in one shot.'],
 ask:['Which columns of the toy table are the two coefficients?','What is x₂ with ε = 0?','How many lines of code to make a noisy example?'],
 expected:['√ᾱₜ (signal) and √(1 − ᾱₜ) (noise).',`√0.72 = ${f3(Math.sqrt(.72))}: just the shrunk signal.`,'One.'],
 misconception:'The shortcut gives the same distribution as stepping, not the same draw: a stepwise run uses t noise draws, the jump uses one. The next slide checks it.',
 source:by(SRC.ddpm,SRC.paras+' (IV)'),
 transition:'Check the shortcut by simulation.'});
S('shortcut-check','fwd','Step by step or one jump: the same distribution',{nb:'cell-shortcut',
 body:act(1,widget('shortcut','Histograms of the noisy pixel made step by step and with one jump')),
 lines:['4,000 runs each way, from our ink pixel: t small steps with their own noise draws, or one jump with one draw. Same histogram, same mean and spread, matching the formula. Move t.'],
 ask:['Are the two runs giving the same noisy pixel?'],
 expected:['No, different draws; the same distribution. For training that is all we need.'],
 source:SRC.paras,
 transition:'How fast should we destroy?'});
chap('1.6');
chap('1.4');
S('schedule','fwd','The real schedule: 1,000 small steps',{builds:2,
 body:act(1,scheduleFig(1),at(2,eq(R`t=250:\ \ \bar\alpha=${tn(r.alpha_bar,3)},\quad x_{250}=${tn(r.signal,3)}\times 1+${tn(r.noise,3)}\times(${tn(r.eps,1)})=${tn(r.xt,3)}`))),
 lines:['The schedule of the original DDPM: 1,000 steps, β rising from 0.0001 to 0.02. The signal coefficient √ᾱₜ falls from 1 to almost 0; the noise coefficient √(1 − ᾱₜ) rises to 1.',`Our running example from now on: t = 250, where signal and noise are about equal. The ink pixel with the noise draw ε = −0.5 becomes x₂₅₀ = ${num(r.xt,3)}.`],
 ask:['Why 1,000 small steps rather than 5 big ones?','At t = 250, which is bigger, signal or noise?'],
 expected:['Small steps are easy to undo (Part ¶{reverse-explorer}).',`About equal: ${num(r.signal,3)} and ${num(r.noise,3)}.`],
 point:'ᾱ₂₅₀ is a product of 250 α’s, each a little below 1. The calculator on the next slide shows the product for any t.',
 source:by(SRC.ddpm,SRC.iddpm),
 transition:'Try any step yourself.'});
S('calc-forward','fwd','Try it: every number of a forward jump',{nb:'cell-toy',
 body:act(1,widget('calc-forward','A calculator for the forward jump')),
 lines:['Pick the toy or the real schedule, a step t, ink or paper, and a noise draw ε. Every number is shown: β, α, the product ᾱ, the two coefficients and the noisy value, with the bell of all values xₜ could take.'],
 ask:['Real schedule, t = 250, ink, ε = −0.5: what is xₜ?'],
 expected:[`${num(r.xt,3)}, our running example.`],
 transition:'Now on a whole image.'});
chap('1.5');
S('fwd-recap','fwd','Destroying is easy, and it hands us labels for free',{builds:2,
 body:act(1,table(['we have','how'],[['a clean image x₀','the dataset'],['a noise level t','pick 1 … 1,000 at random'],['the noise ε','draw N(0, I)'],['the noisy image xₜ','one line: √ᾱₜ x₀ + √(1 − ᾱₜ) ε']]),
  at(2,p('Next: can we go back? Given x<sub>t</sub>, where was x<sub>t−1</sub>?'))),
 lines:['Part 1 in one table. No learning anywhere: every quantity is chosen, drawn or computed.','The hard direction is the other one. Part 2: the reverse process.'],
 ask:['Which of these will the network see?','What would we need to go back?'],
 expected:['xₜ and t. Part ¶{tuple} makes ε its target.','The distribution of x_{t−1} given xₜ.'],
 transition:'Part 2.'});
S('photo-jump','fwd','The same jump on a real photograph',{builds:2,
 body:act(1,widget('photojump','A photograph of a cat, one noise image, and their mix at noise level t')),
 lines:['Nothing in the shortcut is special to digits. A 256 × 256 colour photo is 196,608 numbers; the same two coefficients mix it with one noise image. Move t.','By t ≈ 600 only the layout is left, light and dark patches; by t = 1,000 the photo is gone. Every photo ends as the same static.'],
 ask:['At what t can you no longer tell it is a cat?','Which survives longer: the fine fur, or the big dark patches?'],
 expected:['Around t = 500–700 with this schedule.','The big patches: fine detail is small and drowns first. Generation will work in the opposite order, coarse first.'],
 point:'The photo is “Chelsea” from scikit-image (CC0), 256 × 256; the noise is one fixed draw and the schedule is the linear DDPM schedule of §{schedule}. Frames every 50 steps, made for the companion film.',
 source:by(SRC.ddpm,SRC.film),
 transition:'Part 1 in one table.'});
// ================================================================ Part 2 · The reverse process
const pct=v=>`${Math.round(v*100)}%`;
function knowFig(b){
 const W=1150,H=320;let s=`<defs>${arrowHead('kn-a',palette.p)}</defs>`+digitSVG(xtOf(250),{size:220,x:40,y:50})+label(40,36,'x₂₅₀: noisy',{fill:palette.ink,size:22,weight:600});
 if(b<2)[[300,90],[320,170],[290,250],[350,130],[340,230]].forEach(([x,y],k)=>{s+=label(x,y,'?',{fill:palette.loss,size:40+(k%2)*8,weight:700});});
 else s+=`<path d="M560,160 H300" stroke="${palette.p}" stroke-width="3" marker-end="url(#kn-a)"/>`+digitSVG(held.x0,{size:220,x:580,y:50})+label(580,36,'x₀: known',{fill:palette.data,size:22,weight:600})+
  label(830,130,'The step back is pinned:',{fill:palette.p,size:24,weight:650})+label(830,165,'one narrow Gaussian',{fill:palette.p,size:24,weight:650})+label(830,215,'pulled a little towards x₀',{size:21});
 if(b<2)s+=label(420,150,'Where was x₂₄₉?',{fill:palette.loss,size:26,weight:650})+label(420,185,'It could have come from',{size:21})+label(420,212,'many different images.',{size:21});
 return fig(W,H,s,b<2?'A noisy 3 with question marks: many images could have produced it.':'The same noisy 3 next to the clean 3: given the clean image, the step back is pinned down.');
}
S('reverse-q','back','Run the film backwards: where was xₜ₋₁?',{builds:3,
 body:act(2,cols(backFig(),stack(eq(R`\text{we want:}\quad p(x_{t-1}\mid x_t)`),
  at(2,p('The exact answer needs the distribution of <b>every possible image</b>, one step noisier. Nobody has that.')),
  at(3,p('So we <b>learn</b> it. When the step is small, the answer is one bell curve: the network only has to find its centre.'))),'1.15fr 1fr')),
 lines:['Forwards was one line of code. Backwards: we see a noisy image at step t. What did it look like one step earlier? Many images could have produced it.','The exact step back is Bayes’ rule, and it needs the distribution of all images at that noise level: as hard as the generative model itself.','So we learn it, one small step at a time. Next: why a small step back is just a Gaussian, on our one pixel, where we can compute everything exactly.'],
 ask:['Why is going forwards easy?','What would we need to compute the step back exactly?','Why might small steps help?'],
 expected:['We chose the noising: one shrink and one Gaussian kick.','The distribution of every possible noisy image one step earlier: the very thing we are trying to learn.','Between two neighbouring frames almost nothing changes, so the earlier frame is almost pinned down by the later one.'],
 point:'In symbols: p(xₜ₋₁ | xₜ) = q(xₜ | xₜ₋₁) q(xₜ₋₁) / q(xₜ). The likelihood is our noising step; the prior q(xₜ₋₁) averages the noising over every image in the world. Optional depth works through this Bayes calculation for one pixel (§{pixel-bayes}).',
 source:SRC.paras+' (V)',
 transition:'Shrink the jump, and watch the step back become one Gaussian.'});
chap('2.2');
S('pixel-prior','back','One pixel: paper or ink',{builds:2,
 body:act(2,cols(pixelHistFig(),stack(p(`Across the 60,000 training images, our pixel is ink (darker than mid-grey) in ${pct(PI)} of them.`),
  at(2,eq(R`x_0=+1\ \text{(ink) with probability }\pi=${tn(PI,2)}`)),at(2,eq(R`x_0=-1\ \text{(paper) otherwise}`)),
  at(2,small(`A simplification: about ${pct(px.share_between)} of the time this pixel is a grey in between. Ignoring the greys makes every formula exact.`))),'1fr 1fr')),
 lines:[`Our pixel across the whole training set: mostly paper, often ink (${pct(PI)} of images), sometimes in between.`,'Model it as a coin: ink with probability π = 0.35, paper otherwise. This is the prior over the clean pixel, and it is all Bayes needs.'],
 ask:['Why is this pixel ink so often?','Is a coin a good model of one pixel?'],
 expected:['It sits where many digits have a stroke (the bottom curve of 3, 5, 8, 0…).','For one pixel, yes, up to the greys. Lecture 2: one probability per pixel gets each pixel right but makes poor images, because strokes are coordinated. Remember that for §{coordinate}.'],
 transition:'We observe the noisy pixel. Was it ink?'});
S('pixel-bayes','back',`We see x₂₅₀ = ${num(r.xt,2)}. Was the pixel ink?`,{builds:4,
 body:act(2,cols(bumpsFig(1),`<div class="eq-sm">`+stack(eq(R`p(x_{250}\mid\text{ink})=${tn(r.lik_ink,2)},\qquad p(x_{250}\mid\text{paper})=${tn(r.lik_paper,2)}`),
  at(2,eq(R`P(\text{ink}\mid x_{250})=\frac{${tn(PI,2)}\times${tn(r.lik_ink,2)}}{${tn(PI,2)}\times${tn(r.lik_ink,2)}+${tn(1-PI,2)}\times${tn(r.lik_paper,2)}}=\dcol{${tn(r.post_ink,2)}}`)),
  at(3,eq(R`\qcol{\hat x_0}=\E[x_0\mid x_{250}]=${tn(r.post_ink,2)}\,(+1)+${tn(r.post_paper,2)}\,(-1)=\qcol{${tn(r.x0_hat,2)}}`)),
  at(4,eq(R`P(\text{ink}\mid x_t)=\sigma\!\Big(\log\tfrac{\pi}{1-\pi}+\tfrac{2\sqrt{\bar\alpha_t}}{1-\bar\alpha_t}\,x_t\Big)`)))+'</div>','1fr 1fr')),
 lines:[`If the pixel was ink, x₂₅₀ comes from N(+${num(r.signal,2)}, ${num(r.var,2)}); if paper, from N(−${num(r.signal,2)}, ${num(r.var,2)}). The observed ${num(r.xt,2)} is three times more likely under ink.`,`Bayes, with the prior ${num(PI,2)} for ink: P(ink | x₂₅₀) = ${num(r.post_ink,2)}. The prior pulls it down: paper is more common.`,`The best single guess of x₀ under squared error is the posterior average: ${num(r.x0_hat,2)}. A grey: probably ink, but not surely.`,'In general the posterior is a sigmoid of xₜ: the log of the likelihood ratio is a straight line in xₜ, as in logistic regression.'],
 ask:['Which bump is taller at 0.38?','Why is the answer below the likelihood ratio’s 0.76?','Why the average, and not the more likely value +1?','Where is the sigmoid centred?'],
 expected:['The ink bump: 0.51 against 0.16.','The prior: paper is 0.65 likely before we look.','Squared error is minimised by the mean (Lecture 1: the mean minimises the expected squared distance).','Where the two posterior probabilities are equal; the prior shifts it from 0 towards ink.'],
 point:`ᾱ₂₅₀ = ${num(r.alpha_bar,4)}, so the bumps sit at ±√ᾱ = ±${num(r.signal,3)} with variance 1 − ᾱ = ${num(r.var,3)}. Log-likelihood ratio: −[(x − s)² − (x + s)²]/(2v) = 2sx/v.`,
 transition:'Move t and xₜ yourself.'});
S('inkpaper','back','Move t and x_t: the posterior and the best guess',{nb:'cell-inkpaper',
 body:act(2,widget('inkpaper','Bayes for one pixel under sliders for t and x_t')),
 lines:['Move xₜ: the posterior slides from paper to ink. Move t: at small t the two bumps are far apart and any xₜ is decisive; at large t they overlap and the answer falls back to the prior.'],
 ask:['At t = 50, what is the best guess for xₜ = 0.38? At t = 900?'],
 expected:[`At t = 50: ink, almost surely (x̂₀ ≈ ${num(2*pInk(.38,50,PI)-1,2)}). At t = 900: almost the prior, x̂₀ ≈ ${num(2*pInk(.38,900,PI)-1,2)}.`],
 transition:'What if we jump all the way back in one step?'});
chap('2.3');
S('big-jump','back','From pure noise straight back to the pixel: two spikes',{builds:3,
 body:act(2,cols(spikesFig(),stack(at(2,small('One Gaussian has one bump: its best fit sits at the average, a grey that is neither paper nor ink.')),
  at(3,`<div class="thumbs" style="flex-wrap:nowrap"><figure>${pic('mnist-mean-3.png',130,'The average of all training 3s')}<figcaption>average of ${ex.n_threes.toLocaleString('en')} training 3s</figcaption></figure><figure>${pic('mnist-mean-all.png',130,'The average of all training digits')}<figcaption>average of all digits</figcaption></figure></div>`)),'1.1fr 0.9fr')),
 lines:[`From x_T, pure noise, back to x₀ in one jump: x_T says nothing about the pixel, so the answer is the prior: paper with ${num(1-PI,2)}, ink with ${num(PI,2)}. Two spikes.`,`One Gaussian cannot be two spikes. Its best fit is centred at the average, ${num(2*PI-1,2)}: a grey pixel nobody drew.`,'For whole images the same thing happens: a one-step denoiser asked for “a digit” returns the average digit, a blur. Average a hundred cats and you get no cat at all.'],
 ask:['What does x_T tell us about the pixel?','Why is the average a bad sample?','What did the VAE decoder output, and why was it blurry?'],
 expected:['Nothing: at t = 1,000 the signal coefficient is 0.006.','It is not a value the data ever takes: no image is grey there.','A mean over the plausible images given z: the same averaging.'],
 source:SRC.mit(2)+' (the “100 cats” question)',
 transition:'So take smaller steps back.'});
chap('2.4');
S('reverse-explorer','back','Shrink the jump: the step back becomes one Gaussian',{builds:3,nb:'cell-reverse',
 body:act(2,widget('reverse','The exact distribution of the pixel several steps back, and the best single Gaussian')),
 lines:[`Our pixel is ink in ${pct(PI)} of training digits and paper otherwise: simple enough to compute the step back exactly. From t = 500, given xₜ = 0, a jump all the way back gives two spikes, ink or paper (green). One Gaussian (purple) fits badly.`,`Back 200 steps, to t = 300: two bumps, ${num(px.reverse[2].separation,1)} widths apart. One Gaussian is still a poor fit.`,`One step back, 500 → 499: the bumps are ${num(px.reverse[5].separation,2)} widths apart, one bell. Press play to shrink the jump continuously.`],
 ask:['Where are the spikes, and why those heights?','What makes the two bumps merge?','So what should one learned step look like?'],
 expected:['At ±1, with the posterior probabilities of paper and ink given xₜ.','Each bump is narrow when the jump is short, but the bumps also move together: one step changes x by very little.','A single Gaussian, N(μ, σ²), with a mean that depends on xₜ and t.'],
 point:'Everything here is exact: q(x_s | xₜ) = Σ over ink/paper of P(x₀ | xₜ) · q(x_s | xₜ, x₀), each term a Gaussian (optional depth derives it for one step).',
 source:by(SRC.seff,'(the 1-D picture, here computed exactly)'),
 transition:'So learn one Gaussian per step. What should the network guess?'});
S('why-gaussian','back','Why: a narrow likelihood times a smooth prior',{builds:3,
 body:act(2,stack(eq(R`q(x_{t-1}\mid x_t)\;\propto\;\underbrace{\N\big(x_t;\sqrt{\alpha_t}\,x_{t-1},\,\beta_t\big)}_{\text{narrow: }\beta_t\le 0.02}\;\times\;\underbrace{q(x_{t-1})}_{\text{smooth}}`),
  at(2,productFig()),
  at(3,eq(R`p_\theta(x_{t-1}\mid x_t)=\N\big(x_{t-1};\ \qcol{\mu_\theta(x_t,t)},\ \sigma_t^2\big)`)))),
 lines:['Bayes again for one step: the likelihood is a narrow bump, because βₜ is at most 0.02.','Over such a narrow range the prior hardly changes, so the product is one narrow bell, centred near xₜ and nudged towards where the prior is higher.','So we will learn one Gaussian per step: its mean comes from a network μ_θ(xₜ, t); its width σₜ is fixed by the schedule. A thousand such steps, and the network is shared.'],
 ask:['Why does a small β make the likelihood narrow?','What if β were large?','Which part needs learning: the mean or the width?'],
 expected:['β is the variance of one step’s noise.','The likelihood would be wide and the product would inherit the prior’s two bumps: §{big-jump}.','The mean. DDPM fixes σₜ² to βₜ or to β̃ₜ of §{posterior-2}.'],
 point:'The general result is old (Feller 1949): for small steps, the reverse of a Gaussian diffusion has the same Gaussian form. Sohl-Dickstein et al. (2015) built on it.',
 source:by(SRC.seff,SRC.sohl),
 transition:'One case where the step back is exactly Gaussian.'});
chap('2.5');
S('know-x0','back','If we knew the clean image, the step back would be exact',{builds:2,
 body:act(2,`<div class="state-slot"><div data-only="1">${knowFig(1)}</div><div data-only="2">${knowFig(2)}</div></div>`,at(2,eq(R`q(x_{t-1}\mid x_t,\dcol{x_0})=\N\big(x_{t-1};\ \pcol{\tilde\mu_t(x_t,x_0)},\ \tilde\beta_t\big)`))),
 lines:['Given only the noisy 3, the step back is uncertain: many clean images could have produced it.','Given the clean 3 as well, the step back is pinned down: one narrow Gaussian. During training we always have x₀, so this is a target we can compute.'],
 ask:['Why does knowing x₀ help?','When do we know x₀?'],
 expected:['It removes the uncertainty about the image: what is left is only the noise of one step.','During training, never during generation.'],
 source:SRC.outlier+' (18:23)',
 transition:'Derive it: Bayes with three Gaussians.'});
S('posterior-1','back','Bayes with three Gaussians',{builds:3,
 body:act(2,stack(eq(R`q(x_{t-1}\mid x_t,\dcol{x_0})=\frac{q(x_t\mid x_{t-1},\dcol{x_0})\;q(x_{t-1}\mid\dcol{x_0})}{q(x_t\mid\dcol{x_0})}`),
  at(2,eq(R`=\frac{\N\big(x_t;\sqrt{\alpha_t}\,x_{t-1},\beta_t\big)\;\N\big(x_{t-1};\sqrt{\bar\alpha_{t-1}}\,\dcol{x_0},1-\bar\alpha_{t-1}\big)}{\N\big(x_t;\sqrt{\bar\alpha_t}\,\dcol{x_0},1-\bar\alpha_t\big)}`)),
  at(3,eq(R`\log q(x_{t-1}\mid x_t,\dcol{x_0})=-\frac12\Big[\frac{(x_t-\sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,\dcol{x_0})^2}{1-\bar\alpha_{t-1}}\Big]+\text{const}`)))),
 lines:['Bayes’ rule with x₀ given throughout: likelihood of xₜ, times the prior of x_{t−1} given x₀, over the evidence.','All three are Gaussians we know. The first is one noising step (x₀ adds nothing once x_{t−1} is known: the Markov property). The other two are the shortcut of §{shortcut}.','Take logs. The denominator has no x_{t−1}: a constant. What is left is a quadratic in x_{t−1}, and an exponentiated quadratic is a Gaussian.'],
 ask:['Why may we drop x₀ from q(xₜ | x_{t−1}, x₀)?','Which of the three is the shortcut?','Why does a quadratic in the exponent mean Gaussian?'],
 expected:['Each step only looks at the previous value.','The second and the third: x_{t−1} and xₜ given x₀.','N(x; μ, s²) ∝ exp(−(x − μ)²/(2s²)): a quadratic in x.'],
 source:by(SRC.outlier,SRC.weng),
 transition:'Read off the mean and the width.'});
S('posterior-2','back','Complete the square: the mean and the width',{builds:3,
 body:act(2,stack(eq(R`\frac{1}{\tilde\beta_t}=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\quad\Rightarrow\quad\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t`),
  at(2,eq(R`\pcol{\tilde\mu_t}=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\dcol{x_0}+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t`)),
  at(3,eq(R`t=250:\quad\pcol{\tilde\mu}=${tn(r.coef_x0,4)}\times 1+${tn(r.coef_xt,4)}\times ${tn(r.xt,3)}=\pcol{${tn(r.mu_true,3)}},\qquad\sqrt{\tilde\beta}=${tn(r.sigma,3)}`)))),
 lines:['Match the quadratic with −(x − μ̃)²/(2β̃): the coefficient of x_{t−1}² gives the width. It simplifies because βₜ = 1 − αₜ.','The coefficient of x_{t−1} gives the mean: a weighted mix of the clean value and the current noisy value.',`Our pixel: the step back keeps ${pct(r.coef_xt)} of xₜ and adds ${num(r.coef_x0*100,1)}% of x₀: μ̃ = ${num(r.mu_true,3)}, with width ${num(r.sigma,3)}. A tiny step.`],
 ask:['Check: α/β + 1/(1 − ᾱₜ₋₁) = (1 − ᾱₜ)/(β(1 − ᾱₜ₋₁))?','Is μ̃ closer to xₜ or to x₀?','How far does one step move the pixel?'],
 expected:['Common denominator: α(1 − ᾱₜ₋₁) + β = α − ᾱₜ + 1 − α = 1 − ᾱₜ. Yes.','To xₜ: 99% of the weight.',`From ${num(r.xt,3)} to a mean of ${num(r.mu_true,3)}, plus noise of size ${num(r.sigma,3)}.`],
 point:'Linear term: the x_{t−1} coefficient in the bracket is −2[√αₜ xₜ/βₜ + √ᾱₜ₋₁ x₀/(1 − ᾱₜ₋₁)]; times β̃ₜ gives μ̃. Same algebra as Lecture 1’s Gaussian KL, one completion of the square.',
 source:by(SRC.ddpm,'eq. 6–7'),
 transition:'But at generation time we do not have x₀.'});
chap('2.3');
S('guess-noise','back','If we knew the noise, we could undo the jump exactly',{builds:3,
 body:act(2,stack(eq(R`x_t=\sqrt{\bar\alpha_t}\,\dcol{x_0}+\sqrt{1-\bar\alpha_t}\,\ncol{\varepsilon}\quad\Longleftrightarrow\quad\dcol{x_0}=\frac{x_t-\sqrt{1-\bar\alpha_t}\;\ncol{\varepsilon}}{\sqrt{\bar\alpha_t}}`),
  at(2,eq(R`\text{toy, }t=2:\quad\frac{${tn(two.x2,3)}-\sqrt{0.28}\times(${tn(two.eps,3)})}{\sqrt{0.72}}=${tn((two.x2-Math.sqrt(.28)*two.eps)/Math.sqrt(.72),3)}`)),
  at(3,eq(R`\qcol{\hat\varepsilon}=\text{network}(x_t,\,t)\;\approx\;\ncol{\varepsilon}`)))),
 lines:['Solve the shortcut for x₀: if someone told us the noise ε, the clean pixel would follow in one line.',`Check on the toy: x₂ = ${num(two.x2,3)} with its noise ${num(two.eps,3)} gives back x₀ = 1 exactly.`,'When generating, nobody tells us ε. So a network guesses it from what it can see: xₜ and t. Everything else is arithmetic.'],
 ask:['Why can we solve for x₀?','What does the network see?','Can its guess be exact?'],
 expected:['The shortcut is a straight line in x₀ and ε: given one, the other follows.','xₜ and t only.','Not always: the same xₜ can come from different (x₀, ε) pairs, so it guesses an average (§{average}, optional depth).'],
 source:SRC.paras+' (VI)',
 transition:'Put the guess into one step back.'});

S('guess-x0','back','We do not have x₀ when generating, so guess it',{builds:3,
 body:act(2,stack(eq(R`\qcol{\mu_\theta(x_t,t)}=\tilde\mu_t\big(x_t,\ \qcol{\hat x_0(x_t,t)}\big)`),
  at(2,eq(R`\text{pixel:}\quad \qcol{\mu}=${tn(r.coef_x0,4)}\times\qcol{${tn(r.x0_hat,2)}}+${tn(r.coef_xt,4)}\times${tn(r.xt,3)}=\qcol{${tn(r.mu_hat,3)}}\qquad(\text{exact: }\pcol{${tn(r.mu_true,3)}})`)),
  at(3,eq(R`\qcol{\hat x_0}=\frac{x_t-\sqrt{1-\bar\alpha_t}\;\qcol{\hat\varepsilon}}{\sqrt{\bar\alpha_t}}\qquad\text{(the shortcut, solved for }x_0)`)))),
 lines:['The plan: replace the unknown x₀ in μ̃ by a guess x̂₀ computed from xₜ. That guess is the network.',`With the best possible guess for our pixel, x̂₀ = ${num(r.x0_hat,2)} from §{pixel-bayes}, the mean is ${num(r.mu_hat,3)}: within ${num(Math.abs(r.mu_true-r.mu_hat),3)} of the exact ${num(r.mu_true,3)}. Small errors per step, a thousand steps.`,'Guessing x₀ and guessing the noise are the same thing: solve the shortcut for x₀. DDPM’s network guesses the noise ε̂.'],
 ask:['What should the network output, ideally?','Is a small error per step harmless?','Why might guessing ε be easier than guessing x₀?'],
 expected:['E[x₀ | xₜ]: the posterior mean, as in §{pixel-bayes}.','It accumulates over a thousand steps; the noise term in each step lets later steps correct it.','ε always has mean 0 and variance 1 at every t: a well-scaled target. §{lsimple} shows the choice reweights the loss.'],
 transition:'Write the mean directly in terms of ε.'});
S('mean-eps','back','Substitute: the mean in terms of the noise',{builds:3,
 body:act(2,stack(eq(R`\tilde\mu_t=a\,\dcol{x_0}+b\,x_t,\qquad a=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t},\ \ b=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`),
  at(2,eq(R`\dcol{x_0}=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\ncol{\varepsilon}}{\sqrt{\bar\alpha_t}}\ \Rightarrow\ \tilde\mu_t=\Big(\frac{a}{\sqrt{\bar\alpha_t}}+b\Big)x_t-\frac{a\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\,\ncol{\varepsilon}`)),
  at(3,eq(R`\pcol{\tilde\mu_t}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\ncol{\varepsilon}\Big)\qquad\frac{1}{${tn(Math.sqrt(r.alpha),4)}}\Big(${tn(r.xt,3)}-\frac{${tn(r.beta,4)}}{${tn(r.noise,3)}}(${tn(r.eps,1)})\Big)=${tn(r.mu_true,3)}`)))),
 lines:['The mean is a mix of x₀ and xₜ, with weights a and b from the schedule.','Replace x₀ by the shortcut solved for x₀: everything becomes xₜ and ε.',`Simplify the two coefficients (√ᾱₜ = √αₜ √ᾱₜ₋₁ and β + α = 1): μ̃ = (xₜ − βₜ/√(1 − ᾱₜ) · ε)/√αₜ. Our pixel with its true ε = −0.5: ${num(r.mu_true,3)} again.`],
 ask:['Why is the x₀-weight a written with √ᾱₜ₋₁?','Expand a/√ᾱₜ + b: what do you get?','What does the step do, in words?'],
 expected:['x_{t−1} given x₀ is the shortcut at t − 1.','[β + α − ᾱₜ]/((1 − ᾱₜ)√αₜ) = 1/√αₜ.','Remove a small multiple of the noise from xₜ, then undo the shrink.'],
 source:by(SRC.ddpm,'eq. 11'),
 transition:'Three names for one guess.'});
S('three-targets','back','Three ways to say the same guess',{builds:2,
 body:act(2,table(['the network outputs','meaning','our pixel (best guess)'],[
  [m(R`\qcol{\hat\varepsilon}`),'the noise: “what is the noise? subtract it”',num(r.eps_star,2)],
  [m(R`\qcol{\hat x_0}`),'the clean value: “what is under the noise?”',num(r.x0_hat,2)],
  [m(R`\qcol{s}=\nabla_x\log q_t`),'the score: “which way is more likely?” (Part ¶{score-def})',num(r.score,2)]]),
  at(2,p('Each can be converted into the others with the shortcut. DDPM predicts ε.'))),
 lines:['Three equivalent things a network can output. Knowing one, xₜ and t, you can compute the others.','DDPM picks the noise: it has the same scale at every t, so it is an easy target. The network has one job: guess the noise.'],
 ask:['Convert: given x̂₀ = 0.26, what is ε̂?','What is the network’s only job?'],
 expected:[`ε̂ = (xₜ − √ᾱ x̂₀)/√(1 − ᾱ) = (${num(r.xt,3)} − ${num(r.signal,3)} × ${num(r.x0_hat,2)})/${num(r.noise,3)} = ${num(r.eps_star,2)}.`,'Guess the noise in xₜ (equivalently the clean image), at every t.'],
 source:by(SRC.paras,SRC.outlier+' (4:48)'),
 transition:'What network makes the guess?'});
chap('2.4');
S('reverse-step','back','One step back, in one line',{builds:3,
 body:act(2,stack(eq(R`x_{t-1}=\underbrace{\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\;\qcol{\hat\varepsilon}\Big)}_{\qcol{\mu}\text{: a little less noise}}+\underbrace{\sqrt{\beta_t}\;z}_{\text{fresh noise}},\qquad z\sim\N(0,1)`),
  at(2,reverseFig()))),
 lines:['The reverse step in one line. Inside the bracket: remove a small multiple of the guessed noise. Dividing by √αₜ undoes the forward shrink. Then add a little fresh noise.',`As a picture, toy step 5: from x₅ = ${f3(toyR[0].x)} the network guesses ε̂ = ${f3(toyR[0].e)}; the mean is ${f3(toyR[0].mu)}; with the fresh draw z = ${tn(toyR[0].z,1)} we land at x₄ = ${f3(toyR[0].next)}.`,'That is the whole generator: start from pure noise and repeat this step from t = T down to 1. Only the network is learned; everything else is the schedule’s numbers.'],
 ask:['Which part needs the network?','Why divide by √αₜ?','Why add noise when we want to remove it?'],
 expected:['Only ε̂.','The forward step shrank by √αₜ; going back we grow by the same factor.','Each step back is a draw from a narrow Gaussian (§{reverse-explorer}), not a single point. §{why-noise} shows what goes wrong without it.'],
 point:'Where the mean comes from: it is the exact step back given x₀, with x₀ replaced by the guess from ε̂ (derived in optional depth by completing the square). Here σₜ = √βₜ; our MNIST runs use the slightly smaller √β̃ₜ, β̃ₜ = βₜ(1 − ᾱₜ₋₁)/(1 − ᾱₜ). Both work (Ho et al. 2020).',
 source:by(SRC.ddpm+' (Algorithm 2)',SRC.paras+' (V)'),
 transition:'Part 3: how does the network learn to guess the noise?'});
// ================================================================ Part 3 · What the network learns, and training
chap('3.1');
S('tuple','train','A training example costs one line',{builds:3,
 body:act(3,`<div class="state-slot">${[1,2,3].map(b=>`<div data-only="${b}">${tupleFig(b)}</div>`).join('')}</div>`),
 lines:['Ingredients: a training image, a fresh noise image and a random level t.','Make the noisy input with the shortcut: xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε.','The network sees xₜ and t and guesses the noise. The answer is the ε we drew ourselves: labels for free, as many as we like.'],
 ask:['Which of these does the network see?','Is this supervised learning?','How many training examples can we make from one image?'],
 expected:['Only xₜ and t.','Yes: input (xₜ, t), target ε. We manufactured the labels.','As many as we like: every (t, ε) pair is a new example.'],
 source:SRC.ddpm+' (Algorithm 1)',
 transition:'The loss, once by hand.'});
S('total-noise','train','ε is all the noise since x₀, not the last step’s',{builds:2,
 body:act(3,stack(eq(R`x_t=\sqrt{\bar\alpha_t}\,\dcol{x_0}+\underbrace{\sqrt{1-\bar\alpha_t}\;\ncol{\varepsilon}}_{\text{every step's noise, combined}}`),
  at(2,table(['t = 250','size'],[['all noise added since x₀: √(1 − ᾱ)',num(r.noise,3)],['the last step’s noise alone: √β₂₅₀',num(Math.sqrt(r.beta),3)]])))),
 lines:['The target ε is the combined noise of all t steps, as in §{two-steps}: not the noise added in the last step.',`At t = 250 the last step’s noise has size ${num(Math.sqrt(r.beta),3)}; all the noise together, ${num(r.noise,3)}. The network predicts the whole thing.`],
 ask:['Why not predict the last step’s noise?','Then how do we take one step?'],
 expected:['Training never runs the steps: it jumps with the shortcut, so there is no “last step” noise to target.','From the total-noise guess we get the mean μ (§{reverse-step}): Part ¶{step-hand}.'],
 misconception:'A common confusion: ε is not the small noise between xₜ₋₁ and xₜ.',
 source:SRC.mit(2)+' (16:15)',
 transition:'The network also needs t.'});
chap('3.2');
S('why-t','train','One network for every noise level: tell it t',{builds:3,
 body:act(3,stack(`<div class="table-sm">`+table(['the same input xₜ = 0.38','best guess of x₀'],[['at t = 50',num(2*pInk(.38,50,PI)-1,2)+': ink, surely'],['at t = 250',num(2*pInk(.38,250,PI)-1,2)+': unsure'],['at t = 900',num(2*pInk(.38,900,PI)-1,2)+': just the prior']])+'</div>',
  at(3,timeFeatFig()))),
 lines:['The same pixel value means different things at different noise levels: at t = 50 it is surely ink, at t = 900 it says nothing.','So the network must be told t. One set of weights handles every level; t tells it how much noise to expect.','Like token positions in a transformer, t becomes a vector of sines and cosines at different frequencies, so nearby steps look alike and far ones differ.'],
 ask:['Could one network without t work?','Why not feed t as a single number?','Where have you seen these features before?'],
 expected:['Only by guessing the noise level from the image, which is hard at both ends. Feeding t is free.','A plain number is a poor input for a network to make many different behaviours from; sinusoids give it a rich, smooth code.','Positional encodings in “Attention is all you need”.'],
 source:by(SRC.outlier+' (7:01)',SRC.mit(1)+' (22:21)'),
 transition:'The training loop.'});
chap('3.3');
S('loss-hand','train','The loss: squared error between the noise and its guess',{builds:3,
 body:act(3,stack(eq(R`\lcol{L(\theta)}=\big\|\ncol{\varepsilon}-\qcol{\hat\varepsilon_\theta(x_t,t)}\big\|^2\quad(\text{averaged over pixels and examples})`),
  at(2,eq(R`\text{pixel:}\quad(\ncol{${tn(r.eps,1)}}-\qcol{${tn(r.eps_star,2)}})^2=\lcol{${tn((r.eps-r.eps_star)**2,2)}}`)),
  at(3,`<div class="thumbs"><figure>${pic('mnist-eps.png',150,'The noise we drew')}<figcaption>ε: the noise we drew</figcaption></figure><figure><svg width="150" height="150" viewBox="0 0 150 150">${svgFrame('mnist-film-eps.png',9,16,0,0,150)}</svg><figcaption>ε̂: the network’s guess, t = 250</figcaption></figure><figure style="justify-content:center"><p class="lead" style="color:${palette.loss}">mean squared error ${num(fr(250).eps_mse,3)}</p></figure></div>`))),
 lines:['The loss: how far the guessed noise is from the noise we drew, squared and averaged.',`Our pixel with the best possible guess, ${num(r.eps_star,2)}: the error is (−0.5 − ${num(r.eps_star,2)})² = ${num((r.eps-r.eps_star)**2,2)}. Even the best guess is wrong for this draw.`,`Our 3 at t = 250 through the trained network: the guessed noise looks just like the real one; the average squared error over 784 pixels is ${num(fr(250).eps_mse,3)}.`],
 ask:['Why squared error?','Why is the best guess wrong for our pixel?','Why can the network do so much better on the image than we did on one pixel?'],
 expected:['Squared error is minimised by the average (Lecture 1): the network learns the average noise that could have produced xₜ. It is also a reweighted ELBO (§{train-code}).','The same xₜ could have come from paper with a different noise (§{average}, optional depth).','It sees all 784 pixels: neighbours reveal whether this one is a stroke (§{x0film}).'],
 transition:'The training step in code.'});
chap('3.4');
S('average','train','The same noisy value can come from different noises',{builds:3,
 body:act(3,stack(eq(R`x_{250}=${tn(r.xt,2)}:\quad\underbrace{\text{ink},\ \ncol{\varepsilon}=${tn(r.eps,2)}}_{P=${tn(r.post_ink,2)}}\qquad\text{or}\qquad\underbrace{\text{paper},\ \ncol{\varepsilon}=${tn((r.xt+r.signal)/r.noise,2)}}_{P=${tn(r.post_paper,2)}}`),
  at(2,eq(R`\qcol{\hat\varepsilon^*}=\E[\ncol{\varepsilon}\mid x_t]=${tn(r.post_ink,2)}\times(${tn(r.eps,2)})+${tn(r.post_paper,2)}\times${tn((r.xt+r.signal)/r.noise,2)}=\qcol{${tn(r.eps_star,2)}}`)),
  at(3,irrFig()))),
 lines:[`Two stories explain x₂₅₀ = ${num(r.xt,2)}: ink with noise −0.50, or paper with noise +${num((r.xt+r.signal)/r.noise,2)}. The network cannot tell which.`,`The squared error is smallest for the average noise, weighted by how likely each story is: ${num(r.eps_star,2)}. The network learns this conditional mean.`,'So the loss cannot reach zero. Left: for our pixel the unavoidable loss is largest in the middle, where both stories are plausible. Right: our trained network’s loss by noise level.'],
 ask:['Why the average, again?','At which t is the pixel’s unavoidable loss zero?','Why is the network’s loss largest at small t?'],
 expected:['Squared error is minimised by the mean (Lecture 1).','At both ends: at small t the pixel is obvious; at large t the noise ≈ xₜ itself.','Tiny noise on grey stroke edges is hard to tell apart from the stroke itself: real pixels are not just ink or paper.'],
 point:'Irreducible loss = E[Var(ε | xₜ)] = ᾱ/(1 − ᾱ) · E[1 − x̂₀²] for a ±1 pixel. Computed by numerical integration over xₜ.',
 source:SRC.mit(2)+' (homework hint: the conditional expectation minimises squared error)',
 transition:'What network makes the guess?'});
chap('3.5');
S('mlp','train','For points in 2-D: a small MLP',{nb:'cell-toy-train',
 body:act(3,mlpFig(),small(`Our 2-D example: points on a smiley face. ${toy['toy-uncond'].params.toLocaleString('en')} parameters; the time features are simply concatenated to the input.`)),
 lines:['Before images, a dataset we can see completely: points on a smiley face. The network: the point (x, y) and the 16 time features in, three layers of 128, two numbers out: the guessed noise.'],
 ask:['What are the inputs and outputs?'],
 expected:['Inputs: x, y and 16 features of t. Outputs: the two components of ε̂.'],
 source:by(SRC.mit(1)+' (2-D toy model)','smiley after '+SRC.paras),
 transition:'Press play.'});
S('unet','train','For images: a U-Net, so every pixel can see the whole digit',{builds:3,
 body:act(3,`<div class="state-slot">${[1,2,3].map(b=>`<div data-only="${b}">${unetFig(b)}</div>`).join('')}</div>`),
 lines:['The output must be the same size as the input: one guessed noise value per pixel. Shrink to 7×7 so that each unit sees the whole digit, then grow back to 28×28.',`Skip connections carry fine detail straight across, so the output can be sharp. ${mn.model.params.toLocaleString('en')} parameters.`,'The time embedding is added inside every block: the same weights, told how noisy the input is.'],
 ask:['Why shrink the image at all?','What would be lost without the skip connections?','Where does t enter?'],
 expected:['To coordinate pixels far apart: whether a pixel is a stroke depends on the whole digit.','Exact positions of edges: the decoder would have to reinvent them from 7×7.','In every block, as a learned per-channel shift.'],
 point:'Channels 32 → 64 → 128 → 64 → 32, GroupNorm and SiLU, time features → MLP → a per-block projection added to the activations. The code is in scripts/diffusion.py and the notebook.',
 source:by(SRC.ddpm,SRC.outlier+' (7:01)'),
 transition:'The network also needs t.'});
chap('3.6');
S('train-code','train','The training step in eight lines',{builds:2,nb:'cell-train-step',
 body:act(3,stack(`<div class="code-sm">`+code(`t   = torch.randint(1, T + 1, (B,))          # a level per image
eps = torch.randn_like(x0)                   # the noise: also the target
a   = alpha_bar[t].view(B, 1, 1, 1)
xt  = a.sqrt() * x0 + (1 - a).sqrt() * eps   # the shortcut
eps_hat = model(xt, t)                       # the guess
loss = ((eps - eps_hat) ** 2).mean()         # squared error
opt.zero_grad(); loss.backward()
opt.step()                                   # one gradient step`)+'</div>',at(2,p(`<b>That is all of training:</b> no adversary, no encoder, no sampling. ${mn.model.steps.toLocaleString('en')} such steps trained our MNIST model in about 50 minutes. And the loss is a reweighted ELBO.`)))),
 lines:['Draw a level and a noise image for each training image, jump with the shortcut, guess the noise, take the squared error, and step.','Compare with the GAN of Lecture 3: one network, one loss, no game. And the loss is not arbitrary: it is the ELBO of Lectures 1 and 3 with its weights set to 1. Diffusion pays later, at sampling time.'],
 ask:['Which line is the shortcut?','What does a batch contain?'],
 expected:['xt = a.sqrt() * x0 + (1 - a).sqrt() * eps.','Different images, each at its own random noise level.'],
 point:`Batch ${mn.model.batch}, laptop GPU. Why squared error on the noise: the ELBO is a sum of KL terms, each a weighted squared noise error; Ho et al. found that setting every weight to 1 gives better samples. Optional depth derives it (§{elbo-simple}).`,
 source:SRC.ddpm+' (Algorithm 1)',
 transition:'Watch it learn, on points we can see.'});
S('playground','train','Press play: the network learns the smiley',{builds:2,nb:'cell-toy-train',
 body:act(3,widget('playground','The smiley network during training: arrows, samples and loss')),
 lines:['Before training: arrows from each grid point to the network’s guess of the clean point (at t = 100) point nowhere useful, and its samples are a blob.','Press play: as the loss falls the arrows turn towards the face, and the samples, drawn with Part ¶{sample-code}’s loop, become a smiley.'],
 ask:['What should the arrows point at?','Why does the loss stay well above zero?'],
 expected:['The nearest part of the face: where a clean point near this noisy one most likely was.','Some of the noise cannot be guessed: the same noisy point can come from different clean points (§{average}, optional depth).'],
 point:`${toy['toy-uncond'].steps.toLocaleString('en')} Adam steps on fresh smiley points, batch ${toy['toy-uncond'].batch}; frames at steps 0, 200, 500, 1k, 2k, 5k, 10k, 20k, 40k.`,
 source:SRC.paras+' (“Training Playground”, here with a real network)',
 transition:'And on images: what does the trained network believe?'});
chap('3.7');
S('x0film','train','What the network believes our 3 is, at each noise level',{builds:2,nb:'cell-x0-film',
 body:act(3,widget('x0film','Our 3 at a noise level: input, predicted noise and the network’s guess')),
 lines:['Our 3 at t = 250, its predicted noise, and the clean image that guess implies. Move the slider: at t = 1,000 the guess is a blurry average digit; by t = 500 it is clearly a 3; at small t it is exactly our 3.','Each pixel guessed on its own (the best one-pixel guess, from optional depth) stays a grey blur much longer. The network’s guesses are far better.'],
 ask:['Why is the guess blurry at high noise?','Why is the network so much better than one pixel at a time?'],
 expected:['It is an average over every digit that could hide under this much noise.','It sees all 784 pixels together: a stroke here makes a stroke there likely (§{coordinate}, optional depth).'],
 transition:'The same on a real photo, with the original DDPM network.'});
S('photo-guess','train','The original DDPM network on a real photograph',{builds:2,
 body:act(3,`<div class="state-slot"><div data-only="1">${photoRow([['pair-xt-0400.jpg','x₄₀₀: the input',palette.ink],['eps.jpg','ε: the noise we added',palette.noise],['pair-epshat-0400.jpg','ε̂: the network’s guess',palette.q],['pair-x0hat-0400.jpg','x̂₀: the photo it implies',palette.data]],220)}</div><div data-only="2">${guessGrid()}</div></div>`),
 lines:['The network of the original DDPM paper, given our photo at t = 400. Its guess of the noise (third) is almost exactly the noise we added (second): correlation 0.998. Subtract it, and the photo comes back (right).','The same at other noise levels. At t = 950 its guess is a grey blur; by t = 600, a blurry cat; by t = 250, this cat. The same story as our 3, on a photo.'],
 ask:['Why can it guess the noise so well at t = 400?','Why is the guess at t = 950 blurry?'],
 expected:['Most of the photo is still there; noise and photo look very different; and it has seen millions of cat photos.','Many photos fit what is left. The best squared-error guess is their average: a blur.'],
 point:'Error of the noise guess (mean squared, pixel values in [−1, 1]) at t = 100, 400, 950: 0.035, 0.0034, 0.00002, easier as the noise grows. Error of the photo guess: 0.004, 0.014, 0.18, harder as the noise grows, because a small noise error is divided by √ᾱₜ.',
 source:by(SRC.ddpm,SRC.film),
 transition:'Training in one table.'});
S('coordinate','train','One pixel alone gives a grey; all 784 together give a 3',{builds:2,
 body:act(3,(()=>{const k=mn.film.t.indexOf(500),row=fr(500);const cell=(src,i,title,col,note,x)=>svgFrame(src,i,16,x,40,210)+label(x,28,title,{fill:col,size:21,weight:600})+label(x,280,note,{size:19});
  return `<div class="state-slot"><div data-only="1">${fig(1150,300,cell('mnist-film-xt.png',k,'x₅₀₀',palette.ink,'the input',0)+cell('mnist-film-alone.png',k,'each pixel alone',palette.p,`error ${num(row.alone_mse,3)}`,290),'The noisy 3 and the pixel-by-pixel guess.')}</div><div data-only="2">${fig(1150,300,cell('mnist-film-xt.png',k,'x₅₀₀',palette.ink,'the input',0)+cell('mnist-film-alone.png',k,'each pixel alone',palette.p,`error ${num(row.alone_mse,3)}`,290)+cell('mnist-film-x0.png',k,'the network',palette.q,`error ${num(row.x0_mse,3)}`,580)+`<svg x="870" y="40" width="210" height="210" viewBox="0 0 28 28">${digitSVG(held.x0,{size:28})}</svg>`+label(870,28,'our 3',{fill:palette.data,size:21,weight:600}),'The noisy 3, the pixel-by-pixel guess, the network’s guess and the truth.')}</div></div>`;})()),
 lines:[`At t = 500, guessing each pixel on its own from its own noisy value gives a grey cloud: average squared error ${num(fr(500).alone_mse,3)}.`,`The network looks at all of them at once and returns a clean 3: error ${num(fr(500).x0_mse,3)}, three times smaller. Lecture 2’s lesson again: strokes are coordinated choices, and the U-Net can see them.`],
 ask:['Which is the per-pixel model of Lecture 2?','Why can the network see the stroke?'],
 expected:['Each pixel alone: one probability per pixel, independent.','Its 7×7 layer sees the whole digit; neighbouring noisy pixels together are much less noisy than one.'],
 transition:'Recap of training.'});
S('train-recap','train','Training: a regression problem we made ourselves',{builds:2,
 body:act(3,table(['step','what happens'],[['draw','an image x₀, a level t, a noise ε'],['make','xₜ with the shortcut'],['guess','ε̂ = network(xₜ, t)'],['learn','minimise ‖ε − ε̂‖²: the network learns E[ε | xₜ], the best average guess']]),
  at(2,p('<b>Next:</b> generate. Start from pure noise and use the guesses to walk back, one small Gaussian step at a time.'))),
 lines:['Part 3 in four rows: draw, make, guess, learn. The network ends up predicting the average noise consistent with what it sees.','Part 4: put the guess into Part 2’s step back, and repeat a thousand times.'],
 ask:['What is the target, in one word?','What do we still need to generate an image?'],
 expected:['The noise.','A starting point (pure noise) and the reverse step of §{reverse-step}.'],
 transition:'Part 4.'});

// ================================================================ Part 4 · Sampling: birth of an image
chap('4.1');
S('step-hand','gen','One step back, by hand, on the real schedule',{builds:4,
 body:act(4,stack(eq(R`\qcol{\hat\varepsilon}=\text{network}(x_{250}=${tn(r.xt,3)},\ t=250)=\qcol{${tn(r.eps_star,3)}}`),
  at(2,eq(R`\qcol{\mu}=\frac{1}{\sqrt{${tn(r.alpha,4)}}}\Big(${tn(r.xt,3)}-\frac{${tn(r.beta,4)}}{${tn(r.noise,3)}}\times${tn(r.eps_star,3)}\Big)=\qcol{${tn(r.mu_from_eps,3)}}`)),
  at(3,eq(R`x_{249}=\qcol{\mu}+\sqrt{\beta_{250}}\;z=${tn(r.mu_from_eps,3)}+${tn(Math.sqrt(r.beta),3)}\times${tn(r.z,1)}=${tn(r.mu_from_eps+Math.sqrt(r.beta)*r.z,3)}`)),
  at(4,eq(R`\text{the network's guess of the clean pixel: }\ \qcol{\hat x_0}=\frac{${tn(r.xt,3)}-${tn(r.noise,3)}\times${tn(r.eps_star,3)}}{${tn(r.signal,3)}}=\qcol{${tn(r.x0_hat,2)}}`)))),
 lines:[`Our pixel at t = 250, value ${num(r.xt,3)}. Suppose the trained network says the noise is ${num(r.eps_star,2)} (the best possible guess for this pixel).`,`The mean: remove ${num(r.beta,4)}/${num(r.noise,3)} of the guessed noise and undo the shrink: ${num(r.mu_from_eps,3)}. One step moves the pixel very little.`,`Add fresh noise of size √β = ${num(Math.sqrt(r.beta),3)}: with z = ${num(r.z,1)}, x₂₄₉ = ${num(r.mu_from_eps+Math.sqrt(r.beta)*r.z,3)}. One step done; 249 to go.`,`Along the way we can read off the network’s guess of the clean pixel: ${num(r.x0_hat,2)}. Probably ink, not sure yet.`],
 ask:['Where does ε̂ come from?','Why does one step move so little?','What changes in the next step?','What will x̂₀ do as t falls?'],
 expected:['From the trained network, given xₜ and t.','βₜ is tiny: each real step removes only a little noise.','t becomes 249: new βₜ, ᾱₜ, and a new guess from the network.','Sharpen towards ink or paper as the noise disappears.'],
 source:SRC.ddpm+' (Algorithm 2)',
 transition:'The toy, all the way back.'});
S('step-image','gen','The same step on an image, at t = 600',{builds:3,
 body:act(4,(()=>{const o=mn.onestep,cell=(f,title,col,x)=>`<image href="figures/${f}" x="${x}" y="44" width="190" height="190" style="image-rendering:pixelated"/><rect x="${x}" y="44" width="190" height="190" fill="none" stroke="#D7DCE1"/>`+label(x,32,title,{fill:col,size:20,weight:600});
  const f=b=>fig(1150,270,cell('mnist-onestep-xt.png','x₆₀₀',palette.ink,0)+(b>=2?cell('mnist-onestep-eps.png','ε̂: guessed noise',palette.noise,232)+cell('mnist-onestep-x0.png','x̂₀: guessed digit',palette.q,464):'')+(b>=3?cell('mnist-onestep-mean.png','μ: the mean',palette.q,696)+cell('mnist-onestep-next.png','x₅₉₉',palette.ink,928):'')+label(0,262,`β = ${num(o.beta,3)}, ᾱ = ${num(o.alpha_bar,3)}, step width ${num(o.sigma,3)}`,{size:18}),'One reverse step on an image.');
  return `<div class="state-slot">${[1,2,3].map(b=>`<div data-only="${b}">${f(b)}</div>`).join('')}</div>`;})()),
 lines:['A sampling run of our model at t = 600: to us, pure static.','The network’s guessed noise, and the digit it implies: already a 7.','The mean moves the static only very slightly towards that digit, and fresh noise is added. To the eye nothing changes; over 600 more steps the 7 appears.'],
 ask:['Can you see the 7 in x₆₀₀?','Why do μ and x₅₉₉ look the same as x₆₀₀?','So where is the progress?'],
 expected:['No.','One step changes each pixel by about 1%.','In x̂₀: the guess sharpens step by step (§{mnist-traj}).'],
 transition:'The loop.'});
chap('4.2');
S('toy-reverse','gen','The toy, all the way back: every number',{builds:5,nb:'cell-toy-reverse',
 body:act(4,toyReverseTable(),small(`c<sub>t</sub> = β<sub>t</sub>/√(1 − ᾱ<sub>t</sub>). ε̂ is the best possible guess for this one-pixel dataset (optional depth shows how to compute it).`)),
 lines:[`Start from x₅ = ${f3(toyR[0].x)}, where our forward run ended. Step 5: the network guesses ${f3(toyR[0].e)}, the mean is ${f3(toyR[0].mu)}, a big fresh kick ${f3(toyR[0].sigma)} × (${tn(toyR[0].z,1)}): x₄ = ${f3(toyR[0].next)}.`,`Step 4: guess ${f3(toyR[1].e)}, mean ${f3(toyR[1].mu)}, kick ${f3(toyR[1].sigma)} × ${tn(toyR[1].z,1)}: x₃ = ${f3(toyR[1].next)}.`,`Step 3: x₂ = ${f3(toyR[2].next)}. The pixel hovers near 0: still undecided.`,`Step 2: x₁ = ${f3(toyR[3].next)}. Now it leans towards paper.`,`Step 1, no fresh noise: x₀ = ${f3(toyR[4].next)}. Paper. Generation does not undo our forward run: it draws a new pixel. Run it many times (next slide) and about 35% come out ink, as in the data.`],
 ask:['Check row 5: (0.360 − c₅ × 0.420)/√0.2?','Why is the kick at step 4 smaller than at step 5?','Why can it hover near 0?','Which way will step 1 go?','We started from an ink run. Why paper?'],
 expected:[`c₅ = 0.8/0.974 = ${tn(toyR[0].c,3)}; (0.360 − ${tn(toyR[0].c*toyR[0].e,3)})/0.447 = ${f3(toyR[0].mu)}.`,'√β₄ = 0.707 < √β₅ = 0.894: the toy’s β shrinks going back.','While the noise is large, both paper and ink are plausible.','Towards paper: x₁ is clearly negative.','x₅ is shared by ink and paper runs; going back is sampling, not recall.'],
 transition:'Try it yourself.'});
S('calc-reverse','gen','Try it: walk back from noise, one step at a time',{nb:'cell-toy-reverse',
 body:act(4,widget('calc-reverse','A calculator for the reverse steps')),
 lines:['Step back one click at a time and watch every number: the network’s guess, the mean, the fresh noise. Start new runs and count how often you end at ink. Then switch to the real schedule: 1,000 tiny steps.'],
 ask:['After 20 runs, what fraction ends at ink?'],
 expected:[`About ${pct(PI)}, the share of ink at this pixel in the data: the sampler reproduces the data’s distribution.`],
 point:'The “network” here is the best possible denoiser for the one-pixel dataset (ink with probability 0.35, else paper), computed exactly with Bayes’ rule; see optional depth.',
 transition:'The same loop, in code, for images.'});
chap('4.3');
S('sample-code','gen','The sampling loop: start from noise, repeat a thousand times',{builds:2,nb:'cell-sample',
 body:act(4,stack(`<div class="code-sm">`+code(`x = torch.randn(n, 1, 28, 28)                # start: pure noise
for t in range(T, 0, -1):                    # 1000, 999, ..., 1
    eps_hat = model(x, t)                    # guess the noise
    c = beta[t] / (1 - alpha_bar[t]).sqrt()
    mean = (x - c * eps_hat) / alpha[t].sqrt()   # the mean of 2.4
    z = torch.randn_like(x) if t > 1 else 0  # no noise at the last step
    x = mean + sigma[t] * z                  # sample the Gaussian step
return x`)+'</div>',at(2,p(`<b>Training</b> uses one random level per example; <b>sampling</b> visits every level, in order: ${num(mn.sampling_seconds_64,1)} s for 64 digits on a laptop GPU.`)))),
 lines:['Start from pure noise, which Part 1 showed is where every image ends. Then a thousand times: guess the noise, compute the mean, add a little fresh noise. No noise at the last step.','Training never runs this loop; sampling always does. That is why sampling is slow, and Part 6 makes it faster.'],
 ask:['Why may we start from N(0, I)?','Why is there no noise at t = 1?'],
 expected:['x_T ≈ N(0, I) for every image (§{pixel-hist}).','We want the final, clean sample, not a noisy draw around it.'],
 point:'σₜ = √βₜ as in §{reverse-step}. Our MNIST film and samples use the slightly smaller √β̃ₜ; both work (Ho et al. 2020).',
 source:SRC.ddpm+' (Algorithm 2)',
 transition:'Watch it, in 2-D and on MNIST.'});
chap('4.3');
S('sampler2d','gen','Watch 300 particles find the smiley, and drop your own',{builds:2,nb:'cell-toy-sample',
 body:act(4,widget('sampler2d','Particles carried from noise to the smiley by the trained network')),
 lines:['Three hundred particles, each its own sampling run. At t = 1,000 they are pure noise.','Press play: the network pulls them onto the face. Click anywhere in the square to drop your own particle: the network, running in this page, carries it home.'],
 ask:['When do the particles start to look like a face?','Do two particles that start close end close?'],
 expected:['Late: most of the structure appears in the last few hundred steps.','Not necessarily: fresh noise at every step can send them to different parts of the face.'],
 source:by(SRC.seff+' (5:17)',SRC.mit(1)),
 transition:'The same on MNIST.'});
S('mnist-traj','gen','Eight runs from noise: x_t, and the network’s guess of the end',{builds:2,
 body:act(4,widget('trajectories','Eight sampling runs on MNIST')),
 lines:['Top: xₜ for eight runs; bottom: the network’s guess of the finished digit at that moment. Coarse first: a blob, then a shape, then strokes.','Watch the fourth run: its guess changes its mind, a 0, then a 9, a 5, a 3, and finally a 9. Early steps choose the rough shape; late steps add the details.'],
 ask:['Which appears first, the outline or the stroke details?','Why can the guess change its mind?'],
 expected:['The outline: high noise drowns fine detail first, so it is decided last.','Early on, many digits are consistent with xₜ; each step’s fresh noise and new evidence can tip the balance.'],
 source:by(SRC.mit(6)+' (coarse to fine)',SRC.paras+' (“The Generator”)'),
 transition:'And on the real cat model.'});
S('cat-film','gen','Eight runs of the real cat model: xₜ, and its guess of the end',{builds:2,
 body:act(4,widget('filmstrip','Eight sampling runs of the original DDPM cat model, with the network’s guess of the final photo',{runs:'0,1,2,3,4,5,6,7',both:1,size:124})),
 lines:['The opening film, now with a second row: at every step, the network’s guess of the finished photo. Press play.','Coarse to fine, as on MNIST: first blurry light and dark, then a layout, then fur and eyes. Not every run is a good cat: this 2020 model is small by today’s standards.'],
 ask:['When is the layout of each photo decided?','Which runs failed, and how?'],
 expected:['Early, by about t = 600: after that the guess changes only in detail.','Usually one or two: a very dark image, or two half-cats. LSUN cats contains many odd photos.'],
 point:'google/ddpm-cat-256, 1,000 steps of ancestral sampling (σₜ² = β̃ₜ, as in our MNIST runs), seed 2026, every 50th step shown. The second row is the clean-image guess x̂₀ computed from ε̂ at that step.',
 source:by(SRC.ddpm,SRC.film),
 transition:'Why the fresh noise at every step?'});
chap('4.4');
S('why-noise','gen','Take out the fresh noise, and every sample collapses',{builds:3,
 body:act(4,cols(stack(`<p class="lead" style="color:${palette.q};font-weight:650">with fresh noise</p>`,pic('mnist-with-noise.png',500,'Sixteen samples with fresh noise')),
  at(2,stack(`<p class="lead" style="color:${palette.loss};font-weight:650">same starts, no fresh noise</p>`,pic('mnist-mean-only.png',500,'Sixteen samples without fresh noise: all thin 1s'))),'1fr 1fr'),
  at(3,p(`The smiley does the same: without fresh noise all 300 particles end at one point. Each mean step pulls towards an average; only the noise lets a path commit, as our pixel must commit to paper or ink.`))),
 lines:['Sixteen runs of the sampler, as designed.',`The same sixteen starting noises, keeping only the mean of each step: sixteen identical thin 1s. Ink covers ${pct(mn.mean_only.ink_fraction_mean_only)} of the image instead of ${pct(mn.mean_only.ink_fraction_with_noise)}.`,'In 2-D every particle collapses to a single point. The step-back mean is a pull towards the average guess; the fresh noise is what keeps the samples spread like the data.'],
 ask:['Predict: what happens without the noise?','Why a 1?','So is the noise a nuisance?'],
 expected:['Students guess: blurry? the same digit?','A thin 1 is the least ink: the closest digit to the average, which every mean step pulls towards.','No: it is the sampling. It makes each step a draw from the Gaussian of §{reverse-explorer}, not its centre.'],
 misconception:'Removing the noise is not DDIM: DDIM (Part ¶{ddim-idea}) changes the update so that it stays correct without noise.',
 transition:'The payoff.'});
chap('4.5');
S('samples64','gen','64 new digits',{builds:2,
 body:act(4,pic('mnist-samples-64.png',1100,'Sixty-four samples from our model'),at(2,p('Sharp like the GAN’s, trained with one regression loss like the VAE. And none is a copy of a training image (optional depth measures it).'))),
 lines:[`64 samples from our model: each is a thousand steps from its own noise (${num(mn.sampling_seconds_64,1)} s for all 64).`,'This is what we were after: sharp digits from a stable, simple training loss.'],
 ask:['Find a bad one.','What did it cost?'],
 expected:['Usually a few odd strokes: the model is small and trained for 50 minutes.','A thousand network calls per image instead of one.'],
 transition:'The whole method on one slide.'});
S('napkin','gen','The whole method on a napkin',{builds:2,
 body:act(4,cols(`<div class="algo"><h4>Train</h4><ol><li>x₀ from the data, t ~ 1 … T, ε ~ N(0, I)</li><li>xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε</li><li>step on ‖ε − ε̂_θ(xₜ, t)‖²</li></ol></div>`,`<div class="algo"><h4>Sample</h4><ol><li>x_T ~ N(0, I)</li><li>for t = T … 1: ε̂ = ε̂_θ(xₜ, t)</li><li>xₜ₋₁ = (xₜ − βₜ/√(1−ᾱₜ) ε̂)/√αₜ + σₜ z</li></ol></div>`),
  at(2,p('<b>End of the first session.</b> Next: the score view, faster sampling, and steering with labels and text.'))),
 lines:['Everything so far, in six lines. Paras Chopra’s test: if you can derive these on a napkin, you have understood diffusion.','Next session: arrows to the data (the score), faster sampling, and steering with labels and text.'],
 ask:['Which line needs the network?','Which line is Part 1?'],
 expected:['Train step 3 and sample step 2.','Train step 2: the shortcut.'],
 source:SRC.paras+' (“derive it on a napkin”)',
 transition:'Part 5: the score.'});
// ================================================================ Part 5 · Why it works: a VAE, and a score
function vaeChainFig(b){
 const W=1150,H=340;let s=`<defs>${arrowHead('vc-a',palette.ink)}${arrowHead('vc-q',palette.noise)}${arrowHead('vc-p',palette.q)}</defs>`;
 const node=(x,y,t,c,w=70)=>`<rect x="${x-w/2}" y="${y-24}" width="${w}" height="48" rx="24" fill="#FFFFFF" stroke="${c}" stroke-width="2.4"/>`+label(x,y+8,t,{fill:c,size:22,anchor:'middle',weight:600});
 const arr=(x1,y1,x2,y2,c,m,txt,ty)=>`<path d="M${x1},${y1} L${x2},${y2}" stroke="${c}" stroke-width="2.4" marker-end="url(#${m})"/>`+(txt?label((x1+x2)/2,ty??y1-12,txt,{fill:c,size:19,anchor:'middle'}):'');
 s+=label(20,34,'VAE (Lecture 2)',{fill:palette.ink,size:22,weight:650})+node(170,80,'x',palette.data)+arr(206,80,370,80,palette.noise,'vc-q','encoder q_φ(z | x)')+node(410,80,'z',palette.ink)+arr(446,80,610,80,palette.q,'vc-p','decoder p_θ(x | z)')+node(650,80,'x',palette.data)+label(760,86,'prior p(z) = N(0, I)',{size:19});
 s+=label(20,190,'Diffusion',{fill:palette.ink,size:22,weight:650});
 const xs=[170,330,490,650,810],labs=['x₀','x₁','x₂','…','x_T'];
 xs.forEach((x,k)=>{s+=node(x,250,labs[k],k===0?palette.data:palette.ink);if(k<4)s+=arr(x+36,236,xs[k+1]-36,236,palette.noise,'vc-q',k===0?'q: add noise':'',220)+arr(xs[k+1]-36,266,x+36,266,palette.q,'vc-p',k===0?'p_θ: remove noise':'',292);});
 s+=label(860,256,'p(x_T) = N(0, I)',{size:19});
 if(b>=2)s+=`<path d="M288,96 C288,160 400,180 400,226" fill="none" stroke="${palette.noise}" stroke-width="1.8" stroke-dasharray="6 5"/><path d="M410,104 C410,160 560,190 560,226" fill="none" stroke="${palette.ink}" stroke-width="1.8" stroke-dasharray="6 5"/><path d="M528,96 C528,170 470,250 420,280" fill="none" stroke="${palette.q}" stroke-width="1.8" stroke-dasharray="6 5"/>`+
  label(1130,140,'encoder: fixed, no φ',{fill:palette.noise,size:19,anchor:'end'})+label(1130,166,'latents: x₁ … x_T, 784 numbers each',{fill:palette.ink,size:19,anchor:'end'})+label(1130,192,'decoder: the network',{fill:palette.q,size:19,anchor:'end'});
 if(b>=3)s+=`<path d="M860,90 C960,120 960,200 860,240" fill="none" stroke="${palette.p}" stroke-width="1.8" stroke-dasharray="6 5"/>`+label(980,320,'same prior: a standard Gaussian',{fill:palette.p,size:19,anchor:'middle'});
 return fig(W,H,s,'A VAE above the diffusion chain: the forward noising plays the encoder, the noisy images are the latents, the denoiser is the decoder.');
}
chap('5.1');
S('vae-map','why','Diffusion is a VAE with a fixed encoder',{builds:3,
 body:act(5,`<div class="state-slot">${[1,2,3].map(b=>`<div data-only="${b}">${vaeChainFig(b)}</div>`).join('')}</div>`),
 lines:['Lecture 2’s VAE above, today’s chain below. Does diffusion fit the VAE picture?','Yes: the noisy images x₁ … x_T are the latents, the forward noising is the encoder (fixed, nothing to learn) and the reverse chain is the decoder.','Both start sampling from the same prior: a standard Gaussian. So we can train diffusion the way we trained the VAE: maximise the ELBO.'],
 ask:['What plays z?','What is unusual about this encoder?','What should the training objective be, then?'],
 expected:['All of x₁ … x_T.','It has no parameters: q is chosen, not learned. And the latents are as big as the image.','The ELBO, as in Lectures 1 and 3.'],
 source:by(SRC.seff+' (6:28)',SRC.luo),
 transition:'Its ELBO, in one line.'});
chap('5.2');
S('elbo-chain','why','The same bound as Lecture 1',{builds:3,
 body:act(5,stack(eq(R`\text{Lecture 1:}\quad\log p(\mathcal D)\;\ge\;\E_{q}\big[\log p(\mathcal D,\theta)-\log q(\theta)\big]=\ELBO`),
  at(2,eq(R`\text{now:}\quad\log p_\theta(\dcol{x_0})\;\ge\;\E_{q}\big[\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid\dcol{x_0})\big]`)),
  at(3,eq(R`p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}\qcol{p_\theta(x_{t-1}\mid x_t)},\qquad q(x_{1:T}\mid \dcol{x_0})=\prod_{t=1}^{T}q(x_t\mid x_{t-1})`)))),
 lines:['Lecture 1: the log evidence is at least the ELBO, an average under q of log joint minus log q.','Here the data is x₀, the unknowns are x₁ … x_T and q is the forward chain: the same bound, letter for letter.','Both chains factorise into steps: the model’s joint is the prior on x_T times a thousand learned steps back; q is a thousand fixed steps forward.'],
 ask:['What plays θ in Lecture 1’s version?','Which distribution do we average under?','Why does each chain factorise?'],
 expected:['The latents x₁ … x_T.','q(x₁:T | x₀): run the noising from the training image.','Each is Markov: every step depends only on the previous state.'],
 source:by(SRC.ddpm,'Lecture 1 §4.1'),
 transition:'Turn the forward steps around.'});
S('elbo-flip','why','Turn each forward step around, given x₀',{builds:3,
 body:act(5,stack(eq(R`\ELBO=\E_q\Big[\log p(x_T)+\sum_{t=1}^{T}\log\frac{\qcol{p_\theta(x_{t-1}\mid x_t)}}{q(x_t\mid x_{t-1})}\Big]`),
  at(2,eq(R`q(x_t\mid x_{t-1})=q(x_t\mid x_{t-1},\dcol{x_0})=\frac{\pcol{q(x_{t-1}\mid x_t,\dcol{x_0})}\;q(x_t\mid \dcol{x_0})}{q(x_{t-1}\mid \dcol{x_0})}\qquad(t\ge2)`)),
  at(3,eq(R`\prod_{t=2}^{T}\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}=\frac{q(x_2\mid x_0)}{q(x_1\mid x_0)}\cdot\frac{q(x_3\mid x_0)}{q(x_2\mid x_0)}\cdots=\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}`)))),
 lines:['Put the factorisations in: one log ratio per step, the network’s step back over the fixed step forward.','The ratio compares steps in opposite directions. Fix that with Bayes, exactly as in §{posterior-1}: the forward step equals the step back given x₀, times a ratio of shortcut densities.','Those ratios telescope: every middle term cancels, leaving only the two ends.'],
 ask:['Why compare in the same direction?','Why may we add x₀ to q(xₜ | xₜ₋₁)?','What is left after the cancellation?'],
 expected:['The network’s step goes backwards; to compare like with like we need the backward step of q.','Markov: x₀ adds nothing once xₜ₋₁ is known.','q(x_T | x₀) / q(x₁ | x₀).'],
 source:by(SRC.outlier+' (16:23)',SRC.luo),
 transition:'Collect the terms.'});
S('elbo-terms','why','Three kinds of term; the middle ones are the training loss',{builds:3,
 body:act(5,stack(`<div class="eq-sm">`+eq(R`-\ELBO=\underbrace{\KL\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T}+\sum_{t=2}^{T}\underbrace{\E_q\,\KL\big(\pcol{q(x_{t-1}\mid x_t,x_0)}\,\|\,\qcol{p_\theta(x_{t-1}\mid x_t)}\big)}_{L_{t-1}}\;\underbrace{-\,\E_q\log p_\theta(x_0\mid x_1)}_{L_0}`)+'</div>',
  at(2,table(['term','what it compares','depends on θ?'],[['L_T','the noised image with the N(0, I) prior','no: and it is ≈ 0'],['L₀','the very last step back','yes, a small decoder term']])),
  at(3,p(`Each L<sub>t−1</sub> compares the exact step back given x₀ (§{posterior-2}) with the network’s step back. For our pixel at t = 250 it is ${num(r.kl,4)} nats.`)))),
 lines:['Group the logs into KL divergences. Three kinds of term.','L_T has no parameters, and the shortcut makes it tiny. L₀ is the final step, a small reconstruction term.','The heart of it: a thousand KLs, each between the exact step back given x₀ and the network’s step back. Both are Gaussians.'],
 ask:['Why is L_T ≈ 0?','Which terms does training change?','What are the two Gaussians in L_{t−1}?'],
 expected:['q(x_T | x₀) = N(√ᾱ_T x₀, 1 − ᾱ_T) ≈ N(0, 1): ᾱ_T ≈ 0.00004.','L₀ and the L_{t−1}.','N(μ̃ₜ(xₜ, x₀), β̃ₜ) from §{posterior-2}, and N(μ_θ(xₜ, t), σₜ²) with σₜ² = β̃ₜ.'],
 point:'The KL form appears by grouping log p_θ(xₜ₋₁ | xₜ) − log q(xₜ₋₁ | xₜ, x₀) under the expectation; the two leftover ends make L_T and L₀.',
 source:by(SRC.ddpm,'eq. 5'),
 transition:'A KL of two Gaussians: Lecture 1 has the formula.'});
chap('5.3');
function klFig(){
 const s=r.sigma,xs=lin(r.mu_true-4*s,r.mu_true+4*s,301),P=panel({W:620,H:250,L:16,R:14,T:50,B:52,xMin:xs[0],xMax:xs.at(-1),yMax:npdf(0,0,s*s)*1.15,ticksX:[.2,.3,.4,.5,.6].filter(v=>v>xs[0]&&v<xs.at(-1)),yAxis:false,xLabel:'x₂₄₉'});
 return fig(620,250,P.axes+P.area(xs,xs.map(x=>npdf(x,r.mu_true,s*s)),TINT.p)+P.line(xs,xs.map(x=>npdf(x,r.mu_true,s*s)),palette.p)+P.line(xs,xs.map(x=>npdf(x,r.mu_hat,s*s)),palette.q,'stroke-dasharray="8 5"')+
  legend(16,26,[[`exact step back, mean ${num(r.mu_true,3)}`,palette.p],[`network’s step, mean ${num(r.mu_hat,3)}`,palette.q,true]],19),'Two Gaussians with the same width and slightly different means.');
}
S('kl-gauss','why','Lecture 1’s formula: two Gaussians of the same width',{builds:3,
 body:act(5,stack(eq(R`\KL\big(\N(\mu_1,\sigma_1^2)\,\|\,\N(\mu_2,\sigma_2^2)\big)=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12`),
  at(2,eq(R`\sigma_1=\sigma_2=\sigma:\qquad \KL=0+\frac{\sigma^2+(\mu_1-\mu_2)^2}{2\sigma^2}-\frac12=\frac{(\pcol{\mu_1}-\qcol{\mu_2})^2}{2\sigma^2}`)),
  at(3,cols(klFig(),eq(R`\frac{(${tn(r.mu_true,4)}-${tn(r.mu_hat,4)})^2}{2\times${tn(r.beta_tilde,5)}}=${tn(r.kl,4)}\ \text{nats}`),'1.2fr 1fr')))),
 lines:['The Gaussian KL derived step by step in Lecture 1, section 2.7.','When the widths are equal the log term vanishes and the variances cancel: only the squared distance between the means is left.',`Our pixel at t = 250: the exact step back has mean ${num(r.mu_true,3)}, the network’s (with the best guess) ${num(r.mu_hat,3)}, both with width ${num(r.sigma,3)}. KL = ${num(r.kl,4)} nats. Minimising it slides one bell onto the other.`],
 ask:['Which terms of Lecture 1’s formula vanish?','So what does minimising each L_{t−1} do?','Why are the two means so close?'],
 expected:['log(σ₂/σ₁) = 0, and σ₁²/(2σ₂²) − ½ = 0.','Moves the network’s mean onto the exact mean μ̃ₜ(xₜ, x₀).','Both are 99% xₜ; they differ only in the 1% that depends on x₀ versus its guess.'],
 source:by(SRC.deepia+' (17:23)','Lecture 1 §2.7'),
 transition:'Both means are written in noise.'});
chap('5.4');
S('kl-mse','why','Write both means with noise: the KL is a squared noise error',{builds:3,
 body:act(5,stack(eq(R`\pcol{\tilde\mu_t}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\ncol{\varepsilon}\Big),\qquad\qcol{\mu_\theta}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\qcol{\hat\varepsilon_\theta}\Big)`),
  at(2,eq(R`\pcol{\tilde\mu_t}-\qcol{\mu_\theta}=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\big(\qcol{\hat\varepsilon_\theta}-\ncol{\varepsilon}\big)`)),
  at(3,eq(R`L_{t-1}=\frac{(\tilde\mu_t-\mu_\theta)^2}{2\tilde\beta_t}=\underbrace{\frac{\beta_t^2}{2\tilde\beta_t\,\alpha_t(1-\bar\alpha_t)}}_{w_t}\big\|\ncol{\varepsilon}-\qcol{\hat\varepsilon_\theta}\big\|^2`)),at(3,eq(R`w_{250}\,(${tn(r.eps,1)}-${tn(r.eps_star,2)})^2=${tn(r.weight,4)}\times${tn((r.eps-r.eps_star)**2,3)}=${tn(r.kl_check,4)}`)))),
 lines:['From §{mean-eps}: the exact mean uses the true noise ε, the network’s mean its guess ε̂. Everything else is identical.','Subtract: xₜ cancels, and what remains is a constant times the noise error.',`Square and divide by 2β̃: each KL term is a weight wₜ times the squared noise error. Our pixel: ${num(r.weight,4)} × ${num((r.eps-r.eps_star)**2,3)} = ${num(r.kl_check,4)}, the same as the previous slide.`],
 ask:['Why does xₜ cancel?','What depends on θ?','Check the number against §{kl-gauss}.'],
 expected:['It appears identically in both means.','Only ε̂: the weight is fixed by the schedule.',`${num(r.kl_check,4)} = ${num(r.kl,4)}: the same.`],
 source:by(SRC.ddpm,'eq. 12'),
 transition:'And the weights?'});
function weightFig(){
 const S=LIN,ts=[],ws=[];for(let t=2;t<=T;t+=3){ts.push(t);ws.push(Math.log10(S.beta[t]**2/(2*S.btil[t]*S.alpha[t]*(1-S.ab[t]))));}
 const P=panel({W:620,H:300,L:62,R:30,T:46,B:52,xMin:0,xMax:1000,yMin:-3,yMax:0,ticksX:[0,250,500,750,1000],ticksY:[-3,-2,-1,0],fmtY:v=>v===0?'1':`10${'⁻'}${'⁰¹²³'[-v]}`,xLabel:'t'});
 return fig(620,300,P.axes+P.line(ts,ws,palette.loss)+P.dot(250,Math.log10(r.weight),palette.loss)+label(62,26,'ELBO weight wₜ (log scale)',{fill:palette.loss,size:20}),'The ELBO weight of each step falls steeply with t.');
}
S('lsimple','why','Drop the weights: the simple loss we trained in Part 3',{builds:2,
 body:act(5,cols(weightFig(),stack(eq(R`L_{\text{simple}}=\E_{t,\,x_0,\,\varepsilon}\big\|\ncol{\varepsilon}-\qcol{\hat\varepsilon_\theta(x_t,t)}\big\|^2`),at(2,p('Ho et al. (2020) found that equal weights give <b>better samples</b> than the true ELBO weights: they move effort from the easy small-noise steps to the harder large-noise ones.')),at(2,small('So Part 3’s loss is the ELBO with its weights changed: training is, up to this reweighting, maximum likelihood.'))),'1fr 1.05fr')),
 lines:['The ELBO weights wₜ are huge at small t and tiny at large t: the true bound mostly cares about the last, almost-clean steps.','DDPM drops them: every t counts equally. That is exactly the loss we trained in Part 3, now with a reason.'],
 ask:['Which steps does the true ELBO emphasise?','Is L_simple still a bound on log p(x₀)?'],
 expected:['The small-t ones: almost clean images, tiny noise.','Not exactly: it is a reweighted version. It trains the same network to the same optimum E[ε | xₜ] at every t; only the effort per t changes.'],
 source:by(SRC.ddpm,'§3.4'),
 transition:'A second way to see the same network: as arrows.'});
chap('5.5');
function scoreFig(){
 const t=r.t,pi=PI,xs=lin(-2.6,2.6,301),P=panel({W:900,H:250,L:16,R:14,T:46,B:56,xMin:-2.6,xMax:2.6,yMax:.62,ticksX:[-2,-1,0,1,2],yAxis:false,xLabel:'x (our pixel at t = 250)'});
 let s=P.axes+P.area(xs,xs.map(x=>pi*npdf(x,r.signal,r.var)+(1-pi)*npdf(x,-r.signal,r.var)),TINT.p)+P.line(xs,xs.map(x=>pi*npdf(x,r.signal,r.var)+(1-pi)*npdf(x,-r.signal,r.var)),palette.p);
 s+=`<defs>${arrowHead('sc-a',palette.q)}</defs>`;
 for(const x of [-2.2,-1.6,-1.1,-.6,-.2,.2,.6,1.1,1.6,2.2]){const P1=pInk(x,t,pi),sc=-((x-r.signal*(2*P1-1)))/r.var;const dx=Math.max(-.45,Math.min(.45,sc*.12));s+=`<line x1="${P.x(x).toFixed(1)}" y1="${P.y(.05)}" x2="${P.x(x+dx).toFixed(1)}" y2="${P.y(.05)}" stroke="${palette.q}" stroke-width="3" marker-end="url(#sc-a)"/>`;}
 return fig(900,250,s+label(16,26,'density of the noisy pixel (green) and its score (purple arrows: uphill)',{size:20}),'The density of the noisy pixel and arrows of its score, pointing uphill.');
}
S('score-def','why','The score: an arrow at every point, pointing uphill',{builds:3,
 body:act(5,stack(eq(R`s(x)=\nabla_x\log p(x)`),at(2,scoreFig()))),
 lines:['The score of a distribution: the gradient of its log density, an arrow at every point.','Our noisy pixel at t = 250: two bumps. Every arrow points uphill, towards the nearer bump; between them, towards the more probable one.','Why it is useful: the gradient of log p ignores the normalising constant, the integral that made Lecture 1 hard. Arrows are enough to sample.'],
 ask:['Where is the score zero?','Why does the normaliser not matter?','How could you sample using only arrows?'],
 expected:['At the peaks, and at the dip between them.','log(p/Z) = log p − log Z, and log Z has no x: its gradient is zero.','Walk uphill with some noise: Langevin dynamics (§{langevin}).'],
 point:'Energy view (Vizuara): p ∝ exp(−E). The score is −∇E: downhill in energy, uphill in probability.',
 source:by(SRC.paras+' (IX)',SRC.vizuara+' (39:27)',SRC.song),
 transition:'The smiley’s arrows, from our network.'});
S('scorefield','why','The smiley’s arrows, at different noise levels',{builds:2,nb:'cell-score',
 body:act(5,widget('scorefield','Arrows of the learned score of the smiley at a chosen noise level')),
 lines:['From the trained network: flip the guessed noise and scale it, s ≈ −ε̂/√(1 − ᾱₜ). The noise guess points away from the data; the score points towards it. At t = 100 every arrow points to a slightly blurred, slightly shrunk smiley.','Move t. At t = 10 the arrows are sharp near the face; far from it the network saw no training points at this level, and the arrows are less trustworthy. At t = 500 the density is blurred everywhere and the arrows are smooth.'],
 ask:['Why does the arrow length grow far from the face?','Why use many noise levels?'],
 expected:['The arrow is √ᾱ x̂₀ − x: the further from the face, the further to go.','Low noise: precise but only near the data. High noise: defined everywhere but blurry. Start high and lower it: the schedule.'],
 source:by(SRC.song,SRC.mit(1)+' (23:00)'),
 transition:'Follow the arrows.'});
S('langevin','why','Follow the arrows, with a little noise: Langevin sampling',{builds:2,nb:'cell-langevin',
 body:act(5,widget('langevin','Particles moved by Langevin dynamics with the learned score')),
 lines:['Langevin dynamics: a small step uphill along the score, plus a small random kick. Run it at one fixed noise level, starting from pure noise.','Now anneal: begin at t = 1,000 where the arrows exist everywhere, and lower the noise as the particles settle. This is what DDPM’s sampler does, with a step size set by the schedule.'],
 ask:['Why the random kick?','How does this compare with Part 4’s loop?'],
 expected:['Without it every particle climbs to a peak and stops: no spread, just like §{why-noise}.','Same structure: a nudge along the network’s guess plus fresh noise, at decreasing noise levels.'],
 point:'Step size here: η = 0.1 (1 − ᾱₜ), one step per level, every fourth level. DDPM’s step is a particular choice of this nudge and kick.',
 source:by(SRC.song,SRC.paras+' (IX)'),
 transition:'Part 6: noise schedules and faster sampling.'});
chap('5.6');
S('tweedie','why','The noise predictor is a score estimator',{builds:4,
 body:act(5,stack(eq(R`\text{one bump, }x_0\text{ known:}\quad\nabla_{x}\log\N\big(x;\sqrt{\bar\alpha_t}\,x_0,1-\bar\alpha_t\big)=-\frac{x-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}=-\frac{\ncol{\varepsilon}}{\sqrt{1-\bar\alpha_t}}`),
  at(2,eq(R`\text{all bumps:}\quad\nabla_x\log q_t(x)=\E\Big[-\frac{\ncol{\varepsilon}}{\sqrt{1-\bar\alpha_t}}\;\Big|\;x\Big]=-\frac{\E[\ncol{\varepsilon}\mid x]}{\sqrt{1-\bar\alpha_t}}`)),
  at(3,eq(R`\qcol{\hat\varepsilon_\theta(x_t,t)}\approx\E[\ncol{\varepsilon}\mid x_t]\quad\Longrightarrow\quad\qcol{\hat\varepsilon_\theta(x_t,t)}\approx-\sqrt{1-\bar\alpha_t}\;\nabla_{x}\log q_t(x_t)`)),
  at(4,eq(R`\text{pixel, }t=250:\quad-\frac{${tn(r.eps,1)}}{${tn(r.noise,3)}}=+${tn(r.score_conditional,2)}\ \ (\text{if we knew it was ink}),\qquad -\frac{${tn(r.eps_star,3)}}{${tn(r.noise,3)}}=${tn(r.score,2)}`)))),
 lines:['If we knew x₀, the noisy value would come from one Gaussian bump, and its score is the noise, scaled: −ε/√(1 − ᾱ).','The real noisy density mixes one bump per possible x₀. Its score is the average of the bumps’ scores, weighted by how likely each x₀ is given x: the average noise (Tweedie’s formula).','Our network learned exactly that average noise (§{average}). So, up to a scale and a sign, the noise predictor is the score of the noisy data: the denoiser knows the arrows.',`Our pixel: the one-bump formula would say +${num(r.score_conditional,2)} (push up, towards ink) only if we knew it was ink. The true score is ${num(r.score,2)}: it points from xₜ towards √ᾱ · x̂₀ = ${num(r.signal*r.x0_hat,2)}.`],
 ask:['Where does the minus sign come from?','Why may the gradient move inside the average?','So what does the network estimate?','Why do the two numbers differ?'],
 expected:['Gaussian log density: −(x − mean)²/(2 var), whose gradient is −(x − mean)/var.','∇q_t = ∫ p(x₀) ∇q(x | x₀) dx₀; divide by q_t and Bayes turns p(x₀) q(x | x₀)/q_t(x) into p(x₀ | x).','The score of the noisy data distribution at every noise level, scaled by −√(1 − ᾱ).','One uses the true noise, unknown at generation time; the score averages over both stories, ink and paper.'],
 misconception:'Paras Chopra’s explainer writes ∇ log p(xₜ) = −ε/√(1 − ᾱₜ). That is the score of q(xₜ | x₀), for a known clean image. The score of the noisy data needs the posterior average E[ε | xₜ], which is what the network learns.',
 source:by(SRC.song,SRC.luo,SRC.paras+' (IX, corrected)'),
 transition:'Four ways to use the same trained network.'});
S('map-2x2','why','Four ways to run the same trained network',{builds:2,
 body:act(5,table(['','random (fresh noise)','deterministic'],[['discrete steps','DDPM: Part 4','DDIM: Part 6'],['continuous time','reverse SDE','probability-flow ODE']]),
  at(2,p('In two sentences: diffusion is a VAE whose encoder is fixed noising; its ELBO is a weighted sum of noise MSEs, and we train the unweighted one. The noise predictor is the score of the noisy data, so sampling climbs the arrows while the noise level falls.'))),
 lines:['One trained ε̂-network, four samplers: random or deterministic, in steps or in continuous time. Optional depth has the continuous versions.','The deterministic samplers are the faster ones: Part ¶{ddim-idea}.'],
 ask:['Which of these is the DDPM loop?','Does changing the sampler need retraining?'],
 expected:['Discrete and random.','No: all four use the same ε̂-network.'],
 source:by(SRC.mit(2)+' (39:04)',SRC.sde),
 transition:'Diffusion is also a VAE.'});

S('elbo-simple','learn','Its ELBO is a weighted noise error',{builds:2,
 body:act(7,stack(eq(R`-\ELBO=\sum_{t}w_t\;\E\big\|\ncol{\varepsilon}-\qcol{\hat\varepsilon_\theta(x_t,t)}\big\|^2+\text{const}`),
  at(2,eq(R`L_{\text{simple}}=\E\big\|\ncol{\varepsilon}-\qcol{\hat\varepsilon_\theta(x_t,t)}\big\|^2\qquad(\text{every }w_t=1)`)),
  at(2,p('Part 3’s loss is this ELBO with its weights changed; Ho et al. found that equal weights give better samples. The derivation, one rule per line, is in optional depth.')))),
 lines:['Write the ELBO of this VAE as in Lectures 1 and 3: each step contributes a KL divergence between two Gaussians, and that KL is a weighted squared error between the true noise and the guess.','Set every weight to 1 and you get exactly the loss we trained in Part 3. So training is, up to this reweighting, maximum likelihood.'],
 ask:['What are the latents and the encoder here?','Why drop the weights?'],
 expected:['x₁ … x_T; the fixed noising.','Empirically better samples: the true weights concentrate on the almost-clean steps.'],
 source:by(SRC.ddpm,'§3.4'),
 transition:'Does it just copy its training set?'});

// ================================================================ Part 6 · Noise schedules and faster sampling
chap('6.1');
const ddimX=(()=>{const a=LIN.ab[200];return Math.sqrt(a)*r.x0_hat+Math.sqrt(1-a)*r.eps_star;})();
S('slow','fast','A thousand network calls per image',{builds:2,
 body:act(6,table(['our model, laptop GPU','time'],[['one network call, 8 digits',`${num(mn.ddim.find(x=>x.steps===1).seconds*1000,0)} ms`],['DDPM: 1,000 calls',`${num(mn.ddim.find(x=>x.steps===1000).seconds,1)} s`],['a 512×512 text-to-image model','about a thousand times more work per call']]),
  at(2,p('Can we take fewer, bigger steps with the same trained network?'))),
 lines:['Sampling costs one network call per step: a thousand calls for each batch. Fine for 28×28 digits, painful for large images.','Part 2 said big jumps back are not Gaussian. Is there another way to jump?'],
 ask:['Where does the time go?','Why not just use fewer DDPM steps?'],
 expected:['Into the network calls: everything else is arithmetic.','The step back is only Gaussian for small steps (§{reverse-explorer}); big DDPM steps get the distribution wrong.'],
 transition:'Jump to the guess, then re-noise.'});
S('ddim-idea','fast','DDIM: jump to the guess, then re-noise it to an earlier level',{builds:3,
 body:act(6,stack(eq(R`\qcol{\hat x_0}=\frac{x_t-\sqrt{1-\bar\alpha_t}\;\qcol{\hat\varepsilon}}{\sqrt{\bar\alpha_t}}`),
  at(2,eq(R`x_{t'}=\sqrt{\bar\alpha_{t'}}\;\qcol{\hat x_0}+\sqrt{1-\bar\alpha_{t'}}\;\qcol{\hat\varepsilon}\qquad(t'<t)`)),
  at(3,eq(R`\text{pixel, }250\to200:\ \ \sqrt{${tn(LIN.ab[200],3)}}\times${tn(r.x0_hat,2)}+\sqrt{${tn(1-LIN.ab[200],3)}}\times${tn(r.eps_star,2)}=${tn(ddimX,3)}`)))),
 lines:['Guess the clean image from xₜ, as before.','Then use the shortcut of §{shortcut} in reverse: noise the guess back up to an earlier level t′, reusing the same guessed noise instead of a fresh draw.','No fresh noise, so the path is deterministic, and t′ can be far earlier: 50 jumps instead of 1,000, with the same trained network.'],
 ask:['Which formula is the second line?','Why reuse ε̂ rather than draw new noise?','What do we lose?'],
 expected:['The shortcut xₜ = √ᾱ x₀ + √(1 − ᾱ) ε with the two guesses.','It keeps the noisy image consistent with the guess; DDIM shows these jumps keep the same marginals as the forward process.','The randomness of each step: the same starting noise always gives the same sample.'],
 misconception:'DDIM is not “DDPM with the noise term removed” (§{why-noise} showed that collapses). It is a different update, with a non-Markov forward process that has the same marginals q(xₜ | x₀).',
 source:by(SRC.ddim,SRC.mit(2)+' (24:35)'),
 transition:'See the paths.'});
S('ddim-2d','fast','Same network: DDPM paths jitter, DDIM paths glide',{builds:2,nb:'cell-ddim',
 body:act(6,widget('paths','Forty sampling paths of the same network: DDPM and DDIM')),
 lines:['Forty particles, same starting noise, same trained network. Left: DDPM, a thousand random steps. Right: DDIM, fifty deterministic jumps.','DDPM paths wander and then settle; DDIM paths are nearly straight. Both end on the smiley.'],
 ask:['Why are DDIM paths so straight?','Do the two samplers give the same final points?'],
 expected:['Each jump aims at the current guess of the clean point, and the guess changes slowly.','No: different particles end in different places, but both sets follow the data distribution.'],
 point:`Mean distance to the face: DDPM ${num(toy.paths.ddpm_to_data,4)}, DDIM ${num(toy.paths.ddim_to_data,4)}; fresh data points ${num(toy.paths.data_to_data,4)}.`,
 transition:'And on MNIST?'});
S('ddim-steps','fast','Fewer steps, same network: from 1 to 1,000',{nb:'cell-ddim',
 body:act(6,widget('ddimsteps','MNIST samples from DDIM with a chosen number of steps')),
 lines:['Eight digits from the same eight noises. One step gives the blurry average guess of §{x0film}; about twenty steps already give clean digits, fifty times faster than DDPM.'],
 ask:['Why does one step give blobs, and how few steps can you get away with?'],
 expected:['One jump returns x̂₀ at t = 1,000: the average digit. Here about 20 steps suffice; distilled and consistency models get to 1–4 (§{frontier}).'],
 source:SRC.paras+' (XI, “Speed vs Quality”)',
 transition:'Faster still, on a real image model.'});
S('sd-speed','fast','Stable Diffusion in fewer steps, and in one',{builds:2,
 body:act(6,stack(stripLine('DDIM, the same network',palette.p,stripRow('sd/steps.jpg',6,['1 step','2 steps','5 steps','10 steps','25 steps','50 steps'],132)),
  at(2,stripLine('a distilled model (LCM)',palette.q,stripRow('sd/lcm.jpg',5,['1 step','2 steps','4 steps','8 steps',`<span style="color:${palette.p}">DDIM, 4 steps</span>`],132))))),
 lines:['The real Stable Diffusion: one sentence, one starting noise, fewer and fewer DDIM steps. From about five steps there is a fox; one or two big jumps are not enough. Twenty-five steps look as good as fifty.','A distilled model, a latent consistency adapter trained to jump far, draws a recognisable fox in one step and a sharp one in four. Plain DDIM at four steps (right) is still smeared.'],
 ask:['How many DDIM steps would you pay for?','What did the consistency model learn that DDIM did not?'],
 expected:['About 20–25: as good as 50 at half the cost. Our MNIST model needed about 20 as well (§{ddim-steps}).','To map a noisy point straight to the end of its path in one jump: a student trained to land where many small steps of the original model would.'],
 point:'Stable Diffusion v1.5: DDIM (η = 0), guidance w = 7.5, seed 1, prompt “a red fox sitting in fresh snow, soft morning light, wildlife photograph”. LCM: the LCM-LoRA adapter for SD 1.5 with its own scheduler and w = 1. Run on a laptop GPU for the companion film.',
 source:by(SRC.ddim,SRC.lcm,SRC.sd),
 transition:'Where the field went.'});
chap('6.2');
S('cosine','fast','The schedule matters: linear destroys too early',{builds:2,
 body:act(6,widget('schedules','Our 3 under the linear and the cosine schedule')),
 lines:['Our 3 at t = 600 under the linear schedule of DDPM: gone. Under the cosine schedule: still visible. Move t.','With the linear schedule the last few hundred steps are almost pure noise, so the network spends many of its levels learning very little. The cosine schedule spreads the destruction evenly.'],
 ask:['At which t does the 3 vanish under each?','Why does that matter for training?'],
 expected:['Linear about 600, cosine about 800–850.','Training samples t uniformly: with linear, about a third of the examples are nearly pure noise.'],
 point:'Our models use the linear schedule (simpler, and enough for MNIST). The cosine schedule is ᾱₜ = cos²(((t/T) + s)/(1 + s) · π/2), s = 0.008.',
 source:by(SRC.iddpm,SRC.outlier+' (6:26)'),
 transition:'Even with a good schedule: a thousand steps. Can we take fewer?'});
chap('6.3');
S('label','fast','A 3 on demand: give the network the label',{builds:3,
 body:act(6,stack(code(`h = time_mlp(time_features(t))      # how noisy
h = h + label_embedding(c)            # which digit: one added line
eps_hat = unet(x_t, h)`),
  at(2,cols(pic('mnist-cfg-1.png',500,'Sixteen samples from the label-conditioned model asked for a 3'),stack(p(`Asked for a 3: <b>${pct(mn.guidance.find(g=>g.w===1).is_three)}</b> are classified as 3 by a separate MNIST classifier (${pct(mn.classifier_test_accuracy_2000)} test accuracy).`)),'1.1fr 1fr')),
  at(3,p('Good, but some are weak or ambiguous. Can we push the samples to be more clearly 3s?')))),
 lines:['Conditioning is one line: a learned embedding of the label is added to the time embedding, so every block knows both how noisy the image is and which digit to make.',`Asked for a 3, the conditional model gives ${Math.round(mn.guidance.find(g=>g.w===1).is_three*16)} of 16 that our classifier calls a 3.`,'Not all are convincing. Guidance pushes harder, and it comes from Bayes’ rule.'],
 ask:['Where does the label enter?','How do we check what the samples are?','How would you push towards “more 3-like”?'],
 expected:['Added to the time embedding, so it reaches every block.','With a separate classifier trained on clean MNIST.','Ask the network twice, with and without the label, and push along the difference: next slide.'],
 transition:'Push harder: guidance.'});
S('cg','fast','Classifier guidance: Bayes’ rule on scores',{builds:3,
 body:act(6,stack(eq(R`p(x\mid c)=\frac{p(c\mid x)\;p(x)}{p(c)}`),
  at(2,eq(R`\nabla_x\log p(x\mid c)=\nabla_x\log p(x)+\nabla_x\log p(c\mid x)-\underbrace{\nabla_x\log p(c)}_{=\,0}`)),
  at(3,eq(R`\text{guided score:}\quad\nabla_x\log p(x)+w\,\nabla_x\log p(c\mid x)`)))),
 lines:['Bayes’ rule for images of class c: likelihood of the label times the prior over images.','Take the log and the gradient in x. The evidence p(c) has no x, so it vanishes: the same trick that made the ELBO work. The conditional score is the unconditional score plus a classifier’s gradient.','Turn up the classifier’s voice with a weight w > 1. The catch: we need a classifier that works on noisy images at every t.'],
 ask:['Which term is the classifier?','Why does p(c) drop out?','What is hard about this classifier?'],
 expected:['∇ log p(c | x): which way makes the image more like class c.','No x in it: its gradient is zero.','It must classify xₜ at all noise levels; ordinary classifiers fail on noise.'],
 source:by(SRC.cg,SRC.mit(3)+' (6:19)'),
 transition:'Get rid of the classifier.'});
chap('6.4');
S('cfg-bayes','fast','Why guidance works: the classifier cancels out',{builds:4,
 body:act(6,stack(eq(R`\nabla_x\log p(c\mid x)=\nabla_x\log p(x\mid c)-\nabla_x\log p(x)`),
  at(2,eq(R`\nabla_x\log p(x)+w\big(\nabla_x\log p(x\mid c)-\nabla_x\log p(x)\big)`)),
  at(3,eq(R`\qcol{\tilde\varepsilon}=\hat\varepsilon_\theta(x_t,t,\varnothing)+w\big(\hat\varepsilon_\theta(x_t,t,c)-\hat\varepsilon_\theta(x_t,t,\varnothing)\big)`)),
  at(4,code(`e_uncond = model(x_t, t, NULL)        # label hidden
e_cond   = model(x_t, t, c)
eps_hat  = e_uncond + w * (e_cond - e_uncond)`)))),
 lines:['Rearrange Bayes once more: the classifier’s gradient is the difference between the conditional and the unconditional score.','Substitute it into the guided score. No classifier left: only two scores of the same diffusion model.','In noises (§{tweedie}: score = −ε̂/√(1 − ᾱ), the scale cancels): start from the unconditional guess and move w times along the difference. w = 0 ignores the label; w = 1 is the plain conditional model; w > 1 extrapolates.','How many networks? One: during training the label is replaced by “no label” some of the time (10% for MNIST), so the same network gives both guesses. Two calls per step at sampling.'],
 ask:['Where did the classifier go?','What is w = 1?','How many networks do we train?','What does guidance cost at sampling time?'],
 expected:['It is the difference of two denoisers: an implicit classifier.','The plain conditional model, ε̂(xₜ, t, c).','One, trained with label dropout.','Two network calls per step instead of one.'],
 source:by(SRC.cfg,SRC.mit(3)+' (17:28)'),
 transition:'Turn the dial.'});
S('cfg','steer','Guidance: ask twice, then exaggerate the difference',{builds:3,nb:'cell-cfg',
 body:act(5,stack(eq(R`\qcol{\hat\varepsilon_{\text{guided}}}=\hat\varepsilon(x_t,t,\varnothing)+w\,\big(\hat\varepsilon(x_t,t,c)-\hat\varepsilon(x_t,t,\varnothing)\big)`),
  at(2,p('w = 0 ignores the label · w = 1 is the conditional model · w &gt; 1 exaggerates it.')),
  at(3,code(`e_none = model(x_t, t, NULL)          # the label hidden
e_c    = model(x_t, t, c)             # the label shown
eps_hat = e_none + w * (e_c - e_none)`)))),
 lines:['Ask the same network twice: once with the label, once with it hidden. The difference points towards “more like the label”. Move w times along it.','w = 0 ignores the label, w = 1 is the conditional model itself, and larger w exaggerates the label.','One network gives both answers: in training, the label is replaced by “no label” 10% of the time. At sampling time: two calls per step.'],
 ask:['What is the guided guess when w = 1?','What might go wrong at w = 20?','How many networks do we train?'],
 expected:['The conditional guess ε̂(xₜ, t, c).','Over-exaggerated, unnatural samples, all very alike.','One, with label dropout.'],
 point:'Why the difference of two guesses acts like a classifier: Bayes’ rule (optional depth).',
 source:by(SRC.cfg,SRC.paras+' (XII)'),
 transition:'See the two guesses as arrows.'});
S('cfgvec','steer','Guidance as arrows: two guesses, and a push past them',{nb:'cell-cfg',
 body:act(5,widget('cfgvec','Classifier-free guidance at one point of the smiley')),
 lines:['The part-conditional smiley network at one point. Black: its guess with no label. Purple: its guess for “mouth”. Red: the guided guess. Move w: past 1, the red arrow overshoots the purple one, pushing harder towards the mouth. Click to move the point.'],
 ask:['Where is the red arrow at w = 0? At w = 1?'],
 expected:['On the black arrow; on the purple arrow.'],
 transition:'Turn the dial on whole samples.'});
chap('6.5');
S('dial','fast','The guidance dial: more w, more 3, less variety',{builds:2,nb:'cell-cfg',
 body:act(6,widget('guidance','Samples guided towards a 3, with measurements, and the guided smiley')),
 lines:[`Sixteen samples asked for a 3. w = 0 ignores the label (${pct(mn.guidance[0].is_three)} are 3s, about the base rate). w = 1: ${pct(mn.guidance[1].is_three)}. From w = 2 every sample is a 3, but the variety falls and the strokes get bolder.`,'The smiley, guided towards the mouth: higher w pulls the samples into the middle of the mouth. Guidance trades variety for fidelity.'],
 ask:['Which w would you choose?','Is w = 7 sampling from p(x | c)?'],
 expected:['Somewhere around 2–3 here; image generators use roughly 3–8.','No: it samples a sharpened distribution, more typical than real 3s. That is why images look “generated” at high w.'],
 point:`Variety is the mean distance between pairs of the 16 samples: ${mn.guidance.map(g=>`w = ${g.w}: ${num(g.diversity,1)}`).join(', ')}. Classifier: a separate CNN, ${pct(mn.classifier_test_accuracy_2000)} on test digits.`,
 source:by(SRC.mit(3)+' (13:20)',SRC.cfg),
 transition:'Every digit on demand.'});
S('digits-demand','fast','Every digit on demand',{builds:2,
 body:act(6,pic('mnist-cfg-digits-cols.png',520,'Six samples of each digit, one column per digit, guided with w = 3'),at(2,p(`At w = 3 our classifier agrees with the requested digit for <b>${pct(mn.digits_on_demand_accuracy)}</b> of these 60 samples. <b>Next:</b> scale it up, from a digit label to a sentence.`))),
 lines:['Six samples of each digit, guided with w = 3: one network, one label, ten classes.',`All ${Math.round(mn.digits_on_demand_accuracy*60)} are classified as the digit we asked for. Next: bigger images (latent diffusion), and a sentence instead of a label.`],
 ask:['What would change for a text prompt?','What stays the same?'],
 expected:['The label embedding becomes a text encoder, and conditioning uses cross-attention.','The training loop, the sampler and the guidance formula.'],
 transition:'Bigger images: latent diffusion.'});
// ================================================================ Part 7 · From digits to text-to-image (sections 7.4 and 7.5)
chap('7.1');
S('latent','big','Pixels are expensive: diffuse a compressed code',{builds:3,
 body:act(7,stack(p('A 512 × 512 colour image: <b>786,432</b> numbers through the U-Net, at every one of 50 steps, twice with guidance.'),
  at(2,cols(stack(pic('mnist-ae-3.png',230,'Our 3 and its reconstruction from a 4×7×7 code'),small(`3 → 4 × 7 × 7 code (${mn.latent.numbers} numbers) → decoded`)),stack(pic('mnist-latent-3.png',260,'The four channels of the 3’s latent code'),small('the code: four 7 × 7 channels')),'1fr 1fr')),
  at(3,cols(pic('mnist-latent-samples.png',360,'Samples from diffusion on the latent codes'),small('Diffusion on the codes, then the decoder. Stable Diffusion: 512 × 512 × 3 → 64 × 64 × 4, 48 times fewer numbers.'),'1fr 1fr')))),
 lines:['Running a U-Net on every pixel of a large image, at every step, is the main cost.','Lecture 2 to the rescue: an autoencoder compresses our 3 into a 4 × 7 × 7 code and decodes it back almost perfectly.','Train the same diffusion model on the codes, then decode its samples: latent diffusion, the idea behind Stable Diffusion.'],
 ask:['What does the autoencoder have to get right?','What does the diffusion model now learn?','Why do we need the decoder at the end?'],
 expected:['Reconstruction: the code must keep everything visible.','The distribution of codes: smaller, and with the pixel-level detail handled by the decoder.','The samples are codes; the decoder turns them into images.'],
 point:`Our autoencoder: convolutions down to 4 × 7 × 7, codes scaled by their standard deviation (${num(mn.latent.latent_std,2)}) before diffusion. Latent DDPM: 4 time-conditioned blocks, 20,000 updates.`,
 source:by(SRC.ldm,SRC.paras+' (XII)'),
 transition:'The real thing: Stable Diffusion’s autoencoder.'});
S('sd-vae','big','Stable Diffusion’s autoencoder: 48 times fewer numbers',{builds:2,
 body:act(7,`<div class="eqimg" style="gap:22px;justify-content:center">`+
  `<figure class="eqimg-f">${photo(0,2,250,'sd/vae.jpg')}<figcaption>our photo: 512 × 512 × 3<br>786,432 numbers</figcaption></figure>`+sdArrow('encode')+
  `<figure class="eqimg-f"><span style="display:grid;grid-template-columns:repeat(2,121px);gap:8px">${[0,1,2,3].map(c=>photo(c,4,121,'sd/vae-latent.png')).join('')}</span><figcaption>the code: 4 × 64 × 64<br>16,384 numbers</figcaption></figure>`+
  at(2,sdArrow('decode'))+at(2,`<figure class="eqimg-f">${photo(1,2,250,'sd/vae.jpg')}<figcaption>decoded: almost the same<br>(30.9 dB)</figcaption></figure>`)+'</div>'),
 lines:['The real autoencoder of Stable Diffusion on our photo: 786,432 numbers in, four 64 × 64 channels out. That is 16,384 numbers, 48 times fewer, and the channels still look like a small cat.','Decoded back, the photo is almost unchanged (30.9 dB). Stable Diffusion’s U-Net works only on these small codes; the decoder runs once, at the very end.'],
 ask:['Why does the code still look like a cat?','What does the decoder have to invent?'],
 expected:['Each code position summarises an 8 × 8 patch of the photo, so the layout survives: the code is a small, strange image.','Fine texture such as fur and sharp edges: restored plausibly, not exactly.'],
 point:'The encoder’s mean, times the scaling factor 0.18215; each channel stretched to its 1–99% range for display. Autoencoder: 83.7M parameters (34.2M encoder, 49.5M decoder). Round trip: 30.9 dB PSNR.',
 source:by(SRC.ldm,SRC.sd),
 transition:'From a label to a sentence.'});
chap('7.2');
function sdFig(b){
 const W=1150,H=310;let s=`<defs>${arrowHead('sd-a',palette.ink)}</defs>`;
 const box=(x,y,w,h,t,sub,c,lec)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#FFFFFF" stroke="${c}" stroke-width="2.4"/>`+label(x+w/2,y+h/2,t,{fill:c,size:21,anchor:'middle',weight:650})+label(x+w/2,y+h/2+24,sub,{size:16,anchor:'middle'})+(lec?label(x+w/2,y+h+24,lec,{fill:palette.p,size:16,anchor:'middle',weight:600}):'');
 const ar=(x1,y1,x2,y2)=>`<path d="M${x1},${y1} L${x2},${y2}" stroke="${palette.ink}" stroke-width="2.2" marker-end="url(#sd-a)"/>`;
 s+=label(20,60,'“a red fox sitting in the snow”',{fill:palette.ink,size:21,weight:600})+ar(110,72,110,104)+box(20,108,180,70,'text encoder','CLIP or T5','#245EDB','CLIP lecture');
 s+=box(20,212,180,64,'noise','64 × 64 × 4','#AA4E08','Part 1');
 s+=ar(200,244,300,226)+ar(200,143,300,190);
 s+=box(304,150,330,110,'U-Net denoiser','× 50 steps, with guidance','#8B2CDE','today');
 s+=label(469,140,'cross-attention: image asks, words answer',{fill:palette.q,size:17,anchor:'middle'});
 s+=ar(634,205,720,205)+box(724,170,180,70,'VAE decoder','64 × 64 × 4 → image','#0F766E','Lecture 2')+ar(904,205,960,205);
 s+=`<image href="figures/film/sd/fox.jpg" x="966" y="125" width="160" height="160"/>`;
 return fig(W,H,s,'The text-to-image pipeline: a text encoder, a U-Net denoiser in latent space, and a VAE decoder.');
}
S('text','big','Text in, through cross-attention',{builds:3,
 body:act(7,stack(sdFig(1),at(2,small('Cross-attention: each image position asks the words a question. Queries come from the U-Net’s features; keys and values from the word vectors.')))),
 lines:['A label was one embedding added everywhere. A sentence is many word vectors, and different parts of the image care about different words: cross-attention lets each position attend to the words, exactly the attention of our attention lectures.','The whole pipeline: text encoder, a U-Net denoiser on 64 × 64 × 4 latents with guidance, then the VAE decoder. Every box is a lecture you have had.','Nothing new in the training loop: image codes and captions instead of digits and labels.'],
 ask:['Where do queries and keys come from?','Which box is today’s lecture?','What does guidance do to a text prompt?'],
 expected:['Queries from image features, keys and values from words.','The U-Net denoiser and its sampler.','Pushes images to match the caption more strongly, at the cost of variety (§{sd-guidance}).'],
 source:by(SRC.ldm,SRC.paras+' (XII)','the fox: Stable Diffusion v1.5'),
 transition:'Which words go where, in the real model.'});
S('sd-attn','big','Which words go where: the real cross-attention',{builds:2,
 body:act(7,stripRow('sd/attn.jpg',5,['the picture','“red”','“fox”','“sitting”','“snow”'],205)+at(2,small('Bright: this part of the picture attends to the word. Each map is averaged over heads, layers and all 50 steps.'))),
 lines:['Stable Diffusion, asked for “a red fox sitting in the snow”. For each word, the parts of the picture that attend to it are lit up. The fox attends to “fox”, “red” and “sitting”.','The snow and the background attend to “snow”, and the fox does not. The sentence is not squeezed into one vector: each region reads the words it needs, at every step.'],
 ask:['Which part of the picture attends to “snow”?','How can different regions read different words?'],
 expected:['Everything except the fox: on average 0.26 off the fox and 0.04 on it, on a 0–1 scale.','Each image position sends its own query, so it gets its own attention weights over the words: cross-attention.'],
 point:'U-Net cross-attention at 16 × 16, the conditional half, mean over heads, layers and the 50 steps; each word’s map scaled to its own 0–1 range. Mean on the fox (its orange and dark pixels) / elsewhere: “fox” 0.50 / 0.14, “red” 0.39 / 0.17, “sitting” 0.64 / 0.27, “snow” 0.04 / 0.26. The small words “in” and “the” also light up the fox (0.61 / 0.18 and 0.51 / 0.19), so a single word’s map should not be over-read.',
 source:by(SRC.p2p,SRC.sd),
 transition:'Watch whole sentences become pictures.'});
S('sd-film','big','A sentence in, a picture out: 50 steps',{builds:2,
 body:act(7,widget('filmstrip','Five Stable Diffusion runs, one per sentence, from noise to a picture',{film:'sd',runs:'0,1,2,4,5',size:190,caps:'“a red fox sitting in fresh snow…”|“a peacock with its tail feathers open…”|“a lighthouse on a rocky cliff at sunset, oil painting”|“a watercolor painting of a hummingbird…”|“glowing jellyfish in the deep ocean…”'})),
 lines:['The real Stable Diffusion: five sentences, five starting noises, one network and the same 50-step loop with guidance. Press play.','The same story as our digits and the cats: blotches of colour first (noise in the latent space, decoded), then a layout, then detail. Only the sentence differs.'],
 ask:['Why do the first frames look like coloured blotches rather than grey static?','When is each layout decided?'],
 expected:['They are pure noise in the 4 × 64 × 64 code, passed through the decoder: decoded noise is blotchy and colourful.','Early: the network’s guess of the final picture shows each layout within the first ten or fifteen of the 50 steps; the rest adds detail.'],
 point:'Prompts: “a red fox sitting in fresh snow, soft morning light, wildlife photograph” · “a peacock with its tail feathers open in a garden, golden hour, photograph” · “a lighthouse on a rocky cliff at sunset, oil painting” · “a watercolor painting of a hummingbird and red flowers” · “glowing jellyfish in the deep ocean, underwater photograph”. Stable Diffusion v1.5, DDIM (η = 0), 50 steps, w = 7.5; every second step shown, decoded to pixels.',
 source:by(SRC.ldm,SRC.sd),
 transition:'Turn the guidance dial on a sentence.'});
S('sd-guidance','big','The guidance dial on a sentence: w = 0 to 15',{builds:2,
 body:act(7,stripRow('sd/guidance.jpg',5,['w = 0','w = 1','w = 3','w = 7.5 (default)','w = 15'],205)+at(2,eq(R`\qcol{\hat\varepsilon_{\text{guided}}}=\hat\varepsilon(x_t,t,\varnothing)+w\,\big(\hat\varepsilon(x_t,t,\text{sentence})-\hat\varepsilon(x_t,t,\varnothing)\big)`))),
 lines:['“A red fox sitting in fresh snow…”, the same starting noise, five guidance strengths. At w = 0 the sentence is ignored: an unrelated picture. At w = 1, the plain conditional model, a dull, greyish fox.','From w = 3 to 7.5, the fox we asked for; 7.5 is the usual default. At w = 15 the colours burn and the picture looks over-processed. The same trade as our 3s (§{dial}): more of the prompt, less natural.'],
 ask:['Why is w = 0 not a fox at all?','Which w would you ship?'],
 expected:['w = 0 keeps only the guess for the empty sentence: the unconditional model, which can draw anything.','Somewhere around 5–8. Paras Chopra’s explainer notes that most users settle around 7.5.'],
 point:'Stable Diffusion v1.5, DDIM 50 steps, seed 1. The guided noise is our formula of §{cfg}, with the sentence as the label and the empty sentence as “no label”; w = 1 uses only the conditional guess.',
 source:by(SRC.cfg,SRC.paras+' (XII)',SRC.sd),
 transition:'Part 8: wrap-up.'});
chap('7.3');
S('ideal','big','The best possible denoiser only copies its training set',{builds:3,
 body:act(7,stack(eq(R`\hat x_0^{*}(x_t)=\sum_{i=1}^{N}w_i\,x^{(i)},\qquad w_i\propto\exp\!\Big(-\frac{\|x_t-\sqrt{\bar\alpha_t}\,x^{(i)}\|^2}{2(1-\bar\alpha_t)}\Big)`),
  only('2',cols(stack(small('sampled with the perfect denoiser of 1,000 training images'),pic('mnist-ideal-samples.png',470,'Eight samples from the ideal denoiser'),small('their nearest training images'),pic('mnist-ideal-nearest.png',470,'The nearest training images')),p(`Distance to the nearest training image: <b>${mn.ideal.nearest_distance.map(v=>num(v,2)).join(', ')}</b>. Copies.`),'1.3fr 1fr')),
  only('3',cols(stack(small('sampled with our network'),pic('mnist-ours-8.png',470,'Eight samples from our network'),small('their nearest of 60,000 training images'),pic('mnist-ours-nearest.png',470,'Their nearest training images')),p(`Distances <b>${num(Math.min(...mn.ours_nearest.nearest_distance),1)}–${num(Math.max(...mn.ours_nearest.nearest_distance),1)}</b>; a real test digit is ${num(mn.ours_nearest.typical_test_to_train,1)} from its nearest training digit. New digits.`),'1.3fr 1fr')))),
 lines:['Our pixel formula with N possible clean images instead of two: for a finite training set the loss-minimising guess is a weighted average of training images. It drives the loss to its minimum.','Sample with it: every sample is exactly a training image. A perfect loss is a lookup table.','Our network samples digits that are about as far from the training set as fresh test digits are. It generalises, because it cannot represent the lookup table and is smooth instead.'],
 ask:['Why is this the best possible guess?','So is a lower training loss always better?','What stops our network from copying?'],
 expected:['Bayes with N equally likely clean images: the ink-or-paper calculation of optional depth, with N choices instead of two.','No: the global optimum memorises.','Its architecture and training: a smooth function cannot pick out 60,000 exact points (but very small datasets or huge models can memorise).'],
 source:SRC.mit(4)+' (13:04)',
 transition:'See it in 2-D.'});
S('copy-2d','big','In 2-D: the perfect denoiser copies, a network fills in',{builds:2,nb:'cell-copy',
 body:act(7,widget('copy','Samples from the ideal denoiser of 40 points and from a network trained on them')),
 lines:[`Forty training points on the smiley (circles). The perfect denoiser’s 400 samples land exactly on them: mean distance ${num(toy.copy.ideal_to_train,4)}.`,`A network trained on the same 40 points: its samples spread along the curves between them, ${num(toy.copy.net_to_train,3)} from the nearest training point but only ${num(toy.copy.net_to_shape,3)} from the true face. It invented the face between the points.`],
 ask:['Is the ideal denoiser wrong?','What did the network assume?'],
 expected:['No: it is optimal for these 40 points. It is a bad generator.','That nearby points behave alike: smoothness. That bias is what makes new samples.'],
 source:SRC.mit(4)+' (20:47)',
 transition:'Where the field went.'});
chap('7.4');
S('frontier','big','Where the field went: same idea, better parts',{
 body:act(7,`<div class="table-sm">`+table(['direction','what changes','examples'],[
  ['transformers as denoisers','the U-Net becomes a ViT on patches','DiT; Stable Diffusion 3; video models'],
  ['straight paths','predict a velocity along a straight line from noise to data','flow matching; rectified flow'],
  ['few steps','train a student to jump far','distillation; consistency models (1–4 steps)'],
  ['beyond images','the same loop on other data','audio, video, 3-D, protein structures, robot policies, weather'],
  ['text','undo masking instead of Gaussian noise','masked diffusion language models']])+'</div>'),
 lines:['Since 2020 the parts changed and the idea stayed: corrupt data with a known process, learn to undo it in small steps, sample by running the steps.'],
 ask:['Which of these would you try first for faster sampling?'],
 expected:['Distillation or consistency models for speed; flow matching for simpler training and straighter paths.'],
 source:by(SRC.mit(2),SRC.mit(6),'Peebles &amp; Xie (2023) · Lipman et al. (2023) · Liu et al. (2023) · Song et al. (2023)'),
 transition:'Part 7: steering with labels and text.'});
chap('7.5');
S('compare','big','Three generators, side by side',{
 body:act(7,`<div class="table-sm">`+table(['','VAE (Lecture 2)','GAN (Lecture 3)','Diffusion (today)'],[
  ['training signal','ELBO: reconstruct + KL','a discriminator’s verdict','ELBO, reweighted: predict the noise'],
  ['to sample','decode z ~ N(0, I)','generate from z ~ N(0, I)','denoise x_T ~ N(0, I), step by step'],
  ['network calls per sample','1','1','20–1,000'],
  ['likelihood','a lower bound','none','a lower bound'],
  ['typical failure','blurry samples','unstable training, collapse','slow sampling']])+'</div>'),
 lines:['The three generators of this course. Diffusion keeps the VAE’s stable, likelihood-based training and gets the GAN’s sharpness, and pays in sampling time.'],
 ask:['Which would you choose for a phone app that must generate in 10 ms?'],
 expected:['A GAN or a distilled diffusion model: one or a few network calls.'],
 transition:'Exit questions.'});
S('exit','big','Exit questions',{builds:5,
 body:act(7,`<div class="qa">${[
  ['What does the network see in training, and what is its target?','xₜ and t; the noise ε we drew.'],
  ['Why must t be an input?','One set of weights serves every noise level; the same xₜ means different things at different t.'],
  ['What happens in one reverse step?','Guess ε̂, turn it into x̂₀, compute the mean μ, add a little fresh noise.'],
  ['Why add noise while sampling?','Each step is a draw from a Gaussian; without the noise all samples collapse to one average.'],
  ['How does a sentence steer the image?','Its word vectors enter the U-Net through cross-attention; guidance pushes along the difference between the guesses with and without the text.']].map(([q,a],k)=>`<b>Q${k+1}</b><span>${q}${at(k+1,`<span style="color:${palette.p}"> ${a}</span>`)}</span>`).join('')}</div>`),
 lines:['Q1: training.','Q2: time.','Q3: one step back.','Q4: sampling noise.','Q5: text.'],
 ask:['Answer Q1 before the reveal.','Q2.','Q3.','Q4.','Q5.'],
 expected:['xₜ and t in; ε out.','Different noise levels need different behaviour from shared weights.','ε̂ → x̂₀ → μ → μ + σz.','Without it every sample is the same thin 1.','Cross-attention for the words, guidance for the push.'],
 transition:'References.'});
S('references','big','References and credits',{
 body:`<div class="refs">${[
  `<b>Explainer and videos this lecture follows.</b> ${SRC.paras} · ${SRC.seff} · ${SRC.deepia} · ${SRC.outlier} · ${SRC.vizuara} · MIT 6.S183 “A Practical Introduction to Diffusion Models” (IAP 2025), Lectures 1–6 (Yuan, Becker, Chen, Scarvelis, Lukoianov).`,
  `<b>Papers.</b> ${SRC.sohl} · ${SRC.ddpm} · ${SRC.song} · ${SRC.sde} · ${SRC.ddim} · ${SRC.iddpm} · ${SRC.cg} · ${SRC.cfg} · ${SRC.ldm} · Karras et al., “Elucidating the Design Space of Diffusion-Based Generative Models” (2022) · Lipman et al., “Flow Matching for Generative Modeling” (2023) · Liu et al., “Flow Straight and Fast” (2023) · Peebles &amp; Xie, “Scalable Diffusion Models with Transformers” (2023) · Kadkhodaie et al., “Generalization in diffusion models arises from geometry-adaptive harmonic representations” (2024) · ${SRC.lcm} · ${SRC.p2p}.`,
  `<b>Tutorials.</b> ${SRC.weng} · ${SRC.luo} · Sander Dieleman’s blog (sander.ai).`,
  `<b>Published images</b> (opening): LDM, RePaint, SR3, SDEdit, ControlNet, DreamBooth, each from the paper’s own figures. <b>Real photos</b> (§{hook}, §{photo-jump}, §{photo-guess}, §{cat-film}): the original DDPM cat checkpoint (google/ddpm-cat-256) and the CC0 photo “Chelsea” from scikit-image; (§{sd-speed}, §{sd-vae}, §{sd-attn}, §{sd-film}, §{sd-guidance}): Stable Diffusion v1.5 with the LCM-LoRA adapter. Both run for the companion film “Diffusion models, from noise to images, step by step” (2026). <b>Illustration:</b> ink in water, generated with Gemini.`,
  `<b>Our models and data.</b> MNIST (LeCun et al.); the MNIST DDPM, conditional DDPM, autoencoder, latent DDPM and classifier from the Generative Modelling II rebuild; the smiley models trained for this lecture. Gemini (Google) watched the videos and drafted a plan; Claude (Anthropic) built the lecture.`].map(t=>`<p>${t}</p>`).join('')}</div>`,
 lines:['Every source used, with the slides that use it credited in their footers.'],
 ask:['Which video would you watch next?'],
 expected:['Ari Seff for the overview in 15 minutes; Outlier for the full derivation; MIT 6.S183 for depth.'],
 transition:'End of the main route.'});

// ================================================================ Optional depth
const O=(id,title,o)=>S(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});
O('opt-sde','The continuous view: a stochastic differential equation',{builds:2,
 body:stack(eq(R`dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW\qquad\text{(forward: the noising, with many tiny steps)}`),
  at(2,eq(R`dx=\Big[-\tfrac12\beta(t)\,x-\beta(t)\,\nabla_x\log q_t(x)\Big]dt+\sqrt{\beta(t)}\,d\bar W\qquad\text{(reverse: needs the score)}`))),
 ask:['Where does the drift −½βx come from?','Why does the reverse need the score?'],
 expected:['From √(1 − β) x ≈ (1 − β/2) x for small β: the shrink.','Anderson (1982): running a diffusion backwards adds a drift along the score of the current density. Part ¶{score-def}’s network supplies it.'],
 point:'Dropping the noise and halving the score term gives the probability-flow ODE, with the same marginals; DDIM is a discretisation of it.',
 source:SRC.sde});
O('opt-targets','Other targets: x₀, v and flow matching',{builds:2,
 body:stack(eq(R`\hat\varepsilon,\quad \hat x_0,\quad \hat v=\sqrt{\bar\alpha_t}\,\varepsilon-\sqrt{1-\bar\alpha_t}\,x_0`),
  at(2,eq(R`\text{flow matching:}\quad x_t=(1-t)\,x_0+t\,\varepsilon,\qquad \text{predict }v=\varepsilon-x_0,\ \ \tfrac{dx}{dt}=v`))),
 ask:['Do these targets carry different information?','Why are flow-matching paths straighter?'],
 expected:['No: given xₜ and t each converts into the others. They weight the loss differently across t.','The noising path itself is a straight line, so the learned velocity field is close to straight.'],
 source:by(SRC.mit(6),'Lipman et al. (2023) · Liu et al. (2023)')});
O('opt-sds','A frozen diffusion model as a critic: score distillation',{
 body:stack(eq(R`\nabla_\psi L_{\text{SDS}}=\E_{t,\varepsilon}\Big[w(t)\big(\hat\varepsilon_\theta(x_t,t,c)-\varepsilon\big)\frac{\partial x}{\partial\psi}\Big],\quad x=g(\psi)`),small('Render an image x from 3-D parameters ψ, noise it, ask the frozen text-to-image model for its noise guess, and push ψ so that the guess matches (DreamFusion).')),
 ask:['What is being trained here?'],
 expected:['Only the 3-D parameters ψ; the diffusion model is frozen and acts as a critic.'],
 source:SRC.mit(5)});

// ================================================================ Lecture map, filled in once every section knows its first slide
S('contents','map','Lecture 4 at a glance: following Paras Chopra’s explainer',{nb:'top',
 body:'',
 lines:['Two sessions, in the order of Paras Chopra’s “Diffusion Models from First Principles”. Parts 0–4 build a working diffusion model, following every number by hand. Parts 5–8: the score view, faster sampling, and steering with labels and text. The ▶ Notebook link on every slide opens its section in Colab.'],
 ask:['Which part do you expect to be hardest?'],
 expected:['Usually Part 2, the step back; we follow every number.'],
 point:'Click a section to jump to its first slide; ▶ opens the matching notebook section. Run the notebook top to bottom once before class.',
 transition:'Introduction: static in, a cat out.'});
// ================================================================ The route: every slide in its numbered section, then optional depth
const ROUTE={
 '0.1':['hook','apps'],'0.2':['core-idea','route'],
 '1.1':['one-step','explode','shrink-math'],'1.2':['toy-schedule','toy-forward'],'1.3':['two-steps','shortcut','shortcut-check'],'1.4':['schedule','calc-forward'],'1.5':['film','pixel-hist','photo-jump','fwd-recap'],
 '2.1':['reverse-q'],'2.2':['reverse-explorer'],'2.3':['guess-noise'],'2.4':['reverse-step'],
 '3.1':['three-targets'],'3.2':['unet','why-t'],'3.3':['tuple','loss-hand','train-code'],'3.4':['mlp','playground'],'3.5':['x0film','photo-guess','train-recap'],
 '4.1':['step-hand'],'4.2':['toy-reverse','calc-reverse'],'4.3':['sample-code'],'4.4':['sampler2d','mnist-traj','cat-film'],'4.5':['why-noise'],'4.6':['samples64','napkin'],
 '5.1':['score-def','scorefield'],'5.2':['langevin'],
 '6.1':['cosine'],'6.2':['slow','ddim-idea','ddim-2d','ddim-steps'],'6.3':['sd-speed','frontier'],
 '7.1':['label'],'7.2':['cfg','cfgvec'],'7.3':['dial','digits-demand'],'7.4':['latent','sd-vae'],'7.5':['text','sd-attn'],'7.6':['sd-film','sd-guidance'],
 '8.1':['compare','exit','references'],
};
/** Depth for the curious, in the order of the main route. */
const OPTIONAL=['three-gens','river','shortcut-q','pixel-prior','pixel-bayes','inkpaper','big-jump','why-gaussian','know-x0','posterior-1','posterior-2','guess-x0','mean-eps',
 'total-noise','average','coordinate','step-image','tweedie','map-2x2','vae-map','elbo-simple','elbo-chain','elbo-flip','elbo-terms','kl-gauss','kl-mse','lsimple',
 'ideal','copy-2d','cg','cfg-bayes','opt-sde','opt-targets','opt-sds'];
const PART_ID={0:'open',1:'fwd',2:'back',3:'train',4:'gen',5:'score',6:'fast',7:'steer',8:'wrap'};
const STRIP=/<div class="act-strip"[^>]*>(?:<span[^>]*>[^<]*<\/span>)*<\/div>/;
{
 const byId=Object.fromEntries(slides.map(s=>[s.id,s])),ordered=[byId.contents],used=new Set(['contents']);
 const place=(s,{section,chapter='',anchor,number,optional,part})=>{s.section=section;s.chapter=chapter;s.optional=optional;
  s.notebook=anchor?nbLink(anchor,number):null;s.body=String(s.body).replace(STRIP,part?acts(part):'');ordered.push(s);used.add(s.id);};
 for(const [key,title] of CHAPTERS){
  const part=Number(key[0]);
  for(const id of ROUTE[key]||[]){const s=byId[id];if(!s)throw new Error('Route names a missing slide: '+id);if(used.has(id))throw new Error('Slide used twice: '+id);
   if(!firstOf[key])firstOf[key]=id;place(s,{section:PART_ID[part],chapter:`${key} ${title}`,anchor:s.nbAnchor||secAnchor(key),number:key,optional:false,part:part||0});}
 }
 for(const id of OPTIONAL){const s=byId[id];if(!s)throw new Error('Optional list names a missing slide: '+id);place(s,{section:'optional',anchor:s.nbAnchor,optional:true,part:0});}
 const left=slides.filter(s=>!used.has(s.id)).map(s=>s.id);if(left.length)throw new Error('Slides missing from the route: '+left.join(', '));
 slides.length=0;slides.push(...ordered);
 // cross-references: §{id} is the section of a slide (or the title of an optional one), ¶{id} its part
 const at_=Object.fromEntries(slides.map(s=>[s.id,s]));
 const secOf=id=>{const s=at_[id];if(!s)throw new Error('Reference to a missing slide: '+id);return s.optional?`“${s.title}”`:s.chapter.split(' ')[0];};
 const partOf=id=>{const s=at_[id];if(!s||s.optional)throw new Error('Part reference to a slide off the route: '+id);return s.chapter.split('.')[0];};
 const res=v=>typeof v==='string'?v.replace(/§\{([a-z0-9-]+)\}/g,(_,id)=>secOf(id)).replace(/¶\{([a-z0-9-]+)\}/g,(_,id)=>partOf(id)):v;
 for(const s of slides){for(const k of ['title','body','source'])s[k]=res(s[k]);s.prompts=s.prompts.map(res);
  for(const [k,v] of Object.entries(s.notes))s.notes[k]=Array.isArray(v)?v.map(res):res(v);}
}
slides[0].notebook=nbLink('top');

slides[0].body=`<div class="toc toc-flow">${PARTS.map(([k,t])=>`<h3>${k==='0'?'':`Part ${k} · `}${t}</h3><ol>${CHAPTERS.filter(([num])=>num.startsWith(k+'.')).map(([num,title])=>
 `<li><a href="#${firstOf[num]}/1">${num} ${title}</a><a class="toc-nb" href="${COLAB}#scrollTo=${secAnchor(num)}" target="_blank" rel="noopener" aria-label="notebook ${num}">▶</a></li>`).join('')}</ol>`).join('')}</div>`;
for(const [num] of CHAPTERS)if(!firstOf[num])throw new Error('Section '+num+' has no slides');
