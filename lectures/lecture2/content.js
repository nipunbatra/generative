import {m,eq,at,only,cols,code,table,palette} from '../shared/author.js';
import {init,mount as mountWidgets,panel,label,svg,fig,widget,legend,num,comma,lin,digits,digit,digitSrc,rawDigit,imgs,waitPanel,scatterSVG,classLegend,posteriorSVG,normalPdf,covEllipse,CLASS,BUMP,MUTED,TINT,LINE,AX} from './widgets.js';

const get=f=>fetch(new URL(f,import.meta.url),{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error(`${f}: ${r.status}`);return r.json();});
const [d,DG,NETS,NW,WAVE]=await Promise.all([get('./data/science.json'),get('./data/digits.json'),get('./data/nets.json'),get('./assets/examples/next-word.json'),get('./assets/examples/tts-wave.json')]);
const ANCHORS=new Set(await get('./data/notebook-anchors.json').catch(()=>[]));
const LLM=await Promise.all([1,2,3].map(i=>fetch(new URL(`./assets/examples/llm-sample-${i}.txt`,import.meta.url)).then(r=>r.text())));
init(d,DG,NETS);
export const mount=(stage,ctx)=>mountWidgets(stage,ctx);

// ---------------------------------------------------------------- lecture-local styles
const CSS=`
.dg{image-rendering:pixelated;border:1px solid ${LINE};border-radius:3px;background:#FAFAFB;display:block}
.dgrid{display:grid;justify-content:start;align-items:start}
.dg-cell{margin:0;display:flex;flex-direction:column;align-items:center;gap:3px}
.dg-cell figcaption{font-size:16px;color:${MUTED};line-height:1.1;text-align:center}
.dg-title{font-size:19px;color:${palette.ink};font-weight:600;margin-bottom:2px}
.row-label{font-size:19px;color:${MUTED};line-height:1.2}
.digit-rows{display:grid;grid-template-columns:auto 1fr;gap:10px 18px;align-items:center}
.pca-live{display:flex;align-items:center;gap:22px;font-size:20px}
.pca-live figure{margin:0;display:flex;flex-direction:column;align-items:center;gap:6px}
.pca-live figcaption{font-size:18px;color:${MUTED}}
.pca-live .arrow{font-size:40px;color:${MUTED}}
.pca-code{border:2px solid ${palette.noise};border-radius:12px;padding:12px 16px;color:${palette.noise};font-weight:650;display:flex;flex-direction:column;gap:4px;font-size:22px;max-width:220px}
.pca-code span{font-weight:400;font-size:17px}
.pca-read{display:flex;flex-direction:column;gap:14px;font-size:18px;color:${MUTED};max-width:220px}.pca-read b{display:block;font-size:30px;color:${palette.ink}}
.mog-live{display:flex;flex-direction:column;gap:12px}.mog-read{font-size:19px;color:${MUTED}}
.hover-card[hidden]{display:none!important}.eq-sm .display{font-size:26px}.code-xs .code{font-size:18px;line-height:1.38;padding:10px 16px}
@media(max-width:760px){.reading .dgrid,.reading .digit-rows>div,.reading .photo-row,.reading .pix-table,.reading .mog-live{overflow-x:auto;max-width:100%}.reading .cards{grid-template-columns:1fr!important}.reading .pca-live,.reading .dflex{flex-wrap:wrap}.reading .dflex>*{min-width:0;max-width:100%}.reading .digit-rows{grid-template-columns:1fr}.reading .pix-table{display:block}.reading .photo-row{grid-template-columns:repeat(2,minmax(0,1fr))!important}}.tsm table{font-size:21px}.tsm th,.tsm td{padding:7px 12px}
.hover-card{position:absolute;right:40px;top:150px;display:flex;flex-direction:column;gap:8px;font-size:20px;background:#fff;border:1px solid ${LINE};padding:12px;border-radius:10px;box-shadow:0 8px 30px #14171F14}
.cards{display:grid;gap:18px}.card{border:1.5px solid ${LINE};border-radius:14px;padding:14px 18px;display:flex;flex-direction:column;gap:6px;min-width:0}
.card h4{margin:0;font-size:22px;line-height:1.2}.card p{margin:0;font-size:19px;line-height:1.35}.card .src{font-size:14px;color:${MUTED}}
.prompt-box{font-size:21px;line-height:1.4;border-left:4px solid ${palette.q};background:#8B2CDE0D;padding:10px 16px;border-radius:6px}
.sample-text{font-size:20px;line-height:1.42;border:1px solid ${LINE};border-radius:10px;padding:10px 14px;background:#fff}
.sample-text b{color:${palette.q}}
.photo{border-radius:8px;display:block;width:100%;height:auto}
.photo-row{display:grid;gap:14px;align-items:end}.photo-row figure{margin:0;display:flex;flex-direction:column;gap:6px}.photo-row figcaption{font-size:17px;color:${MUTED};line-height:1.25}
.tag{display:inline-block;font-size:15px;font-weight:650;border-radius:999px;padding:2px 10px;margin-top:4px}
.tag.real{color:${palette.p};border:1.5px solid ${palette.p}}.tag.fake{color:${palette.loss};border:1.5px solid ${palette.loss}}
.big-number{font-size:40px;font-weight:650;color:${palette.q};line-height:1.1}
.pix-table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}.pix-table td{width:38px;height:30px;text-align:center;padding:0;border:1px solid #fff}
.audio-row{display:flex;align-items:center;gap:18px}.audio-row audio{height:38px}
.mini{font-size:20px;line-height:1.35}
.qa{font-size:24px;line-height:1.4}.qa b{color:${palette.q}}
`;
if(!document.getElementById('lecture2-style')){const st=document.createElement('style');st.id='lecture2-style';st.textContent=CSS;document.head.append(st);}

// ---------------------------------------------------------------- helpers
const R=String.raw;
const f=(v,n=2)=>num(v,n);
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const mini=s=>`<p class="mini">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const grid=(n,...a)=>`<div class="image-pair" style="grid-template-columns:repeat(${n},minmax(0,1fr));gap:20px">${a.join('')}</div>`;
const card=(title,body,color=palette.ink,src='')=>`<div class="card" style="border-color:${color}55"><h4 style="color:${color}">${title}</h4><p>${body}</p>${src?`<span class="src">${src}</span>`:''}</div>`;
const photo=(src,caption='',extra='')=>`<figure><img class="photo" src="${src}" alt="${caption.replace(/<[^>]+>/g,'')}" ${extra}>${caption?`<figcaption>${caption}</figcaption>`:''}</figure>`;
const dimg=(key,i,x,y,s,extra='')=>`<image href="${digitSrc(key,i)}" x="${x}" y="${y}" width="${s}" height="${s}" style="image-rendering:pixelated" ${extra}/><rect x="${x}" y="${y}" width="${s}" height="${s}" fill="none" stroke="${LINE}"/>`;
const rows=(...r)=>`<div class="digit-rows">${r.map(([lab,html])=>`<div class="row-label">${lab}</div><div>${html}</div>`).join('')}</div>`;
const F=d.faithful,M=d.mnist,A=d.ae,G=d.gen,V=d.vae,SW=A.sweep,BETA=V.beta;
const WATCH_LABELS=['0','1','2','3','4','5','6','7'];

// ---------------------------------------------------------------- sections, notebook links and the slide builder
export const meta={
 id:'lecture2',number:2,title:'From fitting a distribution to the VAE',
 subtitle:'Why generate, fit and sample, PCA and Gaussians on MNIST, autoencoders, and the variational autoencoder',minutes:150,
 notebook:'lecture2-generative-models.ipynb',
 sections:[
  {id:'map',title:'Lecture map',minutes:1},
  {id:'why',title:'Part 1 · Why generate?',minutes:16},
  {id:'fit',title:'Part 2 · Fit a distribution, then sample',minutes:24},
  {id:'pca',title:'Part 3 · Images: PCA and Gaussians',minutes:24},
  {id:'ae',title:'Part 4 · Autoencoders',minutes:22},
  {id:'gen',title:'Part 5 · Can an autoencoder generate?',minutes:18},
  {id:'vae',title:'Part 6 · The variational autoencoder',minutes:45},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
const CHAPTERS=[
 ['1.1','Things that do not exist'],['1.2','Text, pictures and sound'],['1.3','Beyond pictures'],['1.4','What they all share'],
 ['2.1','The data: Old Faithful'],['2.2','One Gaussian: fit, score, sample'],['2.3','Two bumps: a hidden choice'],['2.4','Fit the mixture, then sample it'],['2.5','How many bumps?'],['2.6','Two numbers per eruption'],
 ['3.1','An image is 784 numbers'],['3.2','Gaussians on raw pixels'],['3.3','PCA: a few numbers per image'],['3.4','Generate from PCA numbers'],
 ['4.1','From PCA to an autoencoder'],['4.2','Train one'],['4.3','How many numbers in the code?'],['4.4','What codes are good for'],
 ['5.1','Decode a random code'],['5.2','Where the codes are'],['5.3','Fit a density to the codes'],
 ['6.1','The generative story'],['6.2','Which code made this 3?'],['6.3','Variational inference for one image'],['6.4','An encoder does it for every image'],['6.5','The loss, term by term'],['6.6','What the VAE learned'],['6.7','The KL weight β'],['6.8','Wrap-up'],
];
const PARTS=[['1','Why generate?'],['2','Fit a distribution, then sample'],['3','Images: PCA and Gaussians'],['4','Autoencoders'],['5','Can an autoencoder generate?'],['6','The variational autoencoder']];
const COLAB='https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture2/lecture2-generative-models.ipynb';
const secAnchor=k=>`sec-${k.replace('.','-')}`;
let chapter=null;const firstOf={};
const chap=k=>{chapter=CHAPTERS.find(ch=>ch[0]===k);if(!chapter)throw new Error('Unknown section '+k);};
function nbLink(anchor,number){
 if(!ANCHORS.size)return null;
 if(!ANCHORS.has(anchor))throw new Error(`Notebook anchor ${anchor} is missing: rebuild the notebook`);
 return {href:`${COLAB}#scrollTo=${anchor}`,label:number?`▶ Notebook ${number}`:'▶ Notebook'};
}
/** The bottom bar carries one story line per build; questions and answers live in the notes. */
function S(id,section,title,o){
 const builds=o.builds||1;
 for(const k of ['lines','ask','expected'])if(!Array.isArray(o[k])||o[k].length!==builds)throw new Error(`${id}: ${k} needs ${builds} entries`);
 const ch=section==='optional'||section==='map'?null:chapter;
 if(ch&&!firstOf[ch[0]])firstOf[ch[0]]=id;
 const anchor=o.nb||(ch?secAnchor(ch[0]):null);
 slides.push({id,section,title,builds,body:o.body,prompts:o.lines,chapter:ch?`${ch[0]} ${ch[1]}`:'',notebook:anchor?nbLink(anchor,ch?.[0]):null,
  notes:{ask:o.ask,expected:o.expected,point:o.point||'',misconception:o.misconception||'',transition:o.transition||'',demo:o.demo||''},
  source:o.source||'',optional:!!o.optional});
}
const ACTS=['Why generate','Fit, then sample','PCA and Gaussians','Autoencoders','Can it generate?','The VAE'];
const acts=k=>`<div class="act-strip" role="img" aria-label="Lecture map: part ${k} of 6">${ACTS.map((a,i)=>`<span class="${i+1<k?'done':i+1===k?'now':''}">${i+1} · ${a}</span>`).join('')}</div>`;
const act=(k,...a)=>`${acts(k)}<div class="act-body">${a.join('')}</div>`;
const OURS='Computed for this lecture: <code>scripts/science.py</code> from the networks trained by <code>scripts/train.py</code> (MNIST, fixed seeds).';
const FAITHFUL_SRC='Old Faithful geyser data (Azzalini and Bowman, 1990), as distributed with R (<code>datasets::faithful</code>): 272 eruptions.';

// ---------------------------------------------------------------- figures
/** Randomness in, a learned network, a sample out; optionally a condition. */
function recipeDiagram(stage=1){
 const W=1100,H=stage>=2?300:230;let s='';
 const box=(x,y,w,h,t,sub,c)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,y+h/2+(sub?-4:9),t,{fill:c,size:26,anchor:'middle',weight:600})+(sub?label(x+w/2,y+h/2+24,sub,{size:19,anchor:'middle'}):'');
 const arrow=(x1,y1,x2,y2,c=palette.ink)=>`<line x1="${x1}" y1="${y1}" x2="${x2-8}" y2="${y2}" stroke="${c}" stroke-width="2.5"/><path d="M${x2},${y2} l-12,-7 v14 z" fill="${c}"/>`;
 s+=box(20,60,250,100,'random numbers','z ~ N(0, I)',palette.noise)+arrow(270,110,400,110)+box(400,50,300,120,'a learned network','weights θ, fitted to data',palette.p)+arrow(700,110,830,110)+box(830,60,250,100,'a sample x','text, image, sound…',palette.data);
 if(stage>=2)s+=box(400,214,300,72,'a condition c','prompt, label, sketch…',palette.q)+arrow(550,214,550,176,palette.q);
 return fig(W,H,s,'Random numbers go into a learned network, which outputs a sample; optionally a condition steers it.');
}
/** Examples in, a distribution out, new examples from it. */
function taskDiagram(stage=1){
 const W=1100,H=230;let s='';
 const arrow=(x1,x2,y,c,t)=>`<line x1="${x1}" y1="${y}" x2="${x2-8}" y2="${y}" stroke="${c}" stroke-width="2.5"/><path d="M${x2},${y} l-12,-7 v14 z" fill="${c}"/>`+(t?label((x1+x2)/2,y-14,t,{fill:c,size:21,anchor:'middle',weight:600}):'');
 const box=(x,w,t,sub,c)=>`<rect x="${x}" y="60" width="${w}" height="110" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,106,t,{fill:c,size:24,anchor:'middle',weight:600})+label(x+w/2,138,sub,{size:19,anchor:'middle'});
 s+=box(10,270,'examples','272 waiting times',palette.data);
 if(stage>=2)s+=arrow(280,420,115,palette.loss,'fit')+box(420,260,'a distribution p(x)','a formula with numbers',palette.p);
 if(stage>=3)s+=arrow(680,820,115,palette.q,'sample')+box(820,270,'new examples','times nobody measured',palette.q);
 return fig(W,H,s,'Examples are fitted by a distribution, which is then sampled to make new examples.');
}
/** Waiting time vs eruption length, with optional fitted ellipses and samples. */
function faith2d({gauss=false,mix=false,samples=null,colourBy=false,W=600,H=420}={}){
 const P=panel({W,H,L:64,R:14,T:28,B:56,xMin:1.3,xMax:5.5,yMin:40,yMax:100,ticksX:[2,3,4,5],ticksY:[50,60,70,80,90],xLabel:'eruption length (minutes)',fs:18,xAxisAtBottom:true});
 let s=P.axes+label(64,18,'next wait (minutes)',{size:18});
 const pts=samples||F.duration.map((x,i)=>[x,F.waiting[i]]);
 const col=i=>colourBy?BUMP[F.mix.resp_long[i]>.5?1:0]:samples?palette.q:palette.data;
 s+=`<g ${P.clip}>${pts.map(([x,y],i)=>`<circle cx="${P.x(x).toFixed(1)}" cy="${P.y(y).toFixed(1)}" r="4.2" fill="${col(i)}" fill-opacity="${samples?.55:.6}"/>`).join('')}</g>`;
 const sx=P.x(1)-P.x(0),sy=P.y(0)-P.y(1);
 if(gauss){const g=F.g2;for(const k of [1,2])s+=covEllipse(P.x(g.mean[0]),P.y(g.mean[1]),[g.cov[0][0],g.cov[0][1],g.cov[1][1]],k,sx,sy,palette.ink,k===1?'':'stroke-dasharray="7 5"');}
 if(mix){const g=F.m2;g.means.forEach((mu,k)=>{const c=g.covs[k];for(const kk of [1,2])s+=covEllipse(P.x(mu[0]),P.y(mu[1]),[c[0][0],c[0][1],c[1][1]],kk,sx,sy,BUMP[k],kk===1?'':'stroke-dasharray="7 5"');});}
 return {s,P};
}
/** PCA in two dimensions on the standardised eruptions: the line of most spread, projections, and rebuilt points. */
function pca2d(stage){
 const mu=[F.g2.mean[0],F.g2.mean[1]],sd=[Math.sqrt(F.g2.cov[0][0]),Math.sqrt(F.g2.cov[1][1])];
 const U=F.duration.map((x,i)=>[(x-mu[0])/sd[0],(F.waiting[i]-mu[1])/sd[1]]);
 const w=[Math.SQRT1_2,Math.SQRT1_2],rho=F.g2.corr;
 const W=1100,H=400,P=panel({W:560,H,L:50,R:14,T:28,B:50,xMin:-2.6,xMax:2.6,yMin:-2.6,yMax:2.6,ticksX:[-2,0,2],ticksY:[-2,0,2],fs:17,xAxisAtBottom:true,xLabel:'length, standardised'});
 let s=P.axes+label(50,18,'wait, standardised',{size:17});
 const z=U.map(([a,b])=>a*w[0]+b*w[1]);
 if(stage>=2)s+=`<line ${P.clip} x1="${P.x(-3*w[0])}" y1="${P.y(-3*w[1])}" x2="${P.x(3*w[0])}" y2="${P.y(3*w[1])}" stroke="${palette.noise}" stroke-width="3"/>`+label(P.x(1.75),P.y(2.25),'direction w₁',{fill:palette.noise,size:20,weight:600});
 if(stage>=3)s+=`<g ${P.clip}>${U.map(([a,b],i)=>`<line x1="${P.x(a).toFixed(1)}" y1="${P.y(b).toFixed(1)}" x2="${P.x(z[i]*w[0]).toFixed(1)}" y2="${P.y(z[i]*w[1]).toFixed(1)}" stroke="${MUTED}" stroke-width="1"/>`).join('')}</g>`;
 s+=`<g ${P.clip}>${U.map(([a,b])=>`<circle cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="4" fill="${palette.data}" fill-opacity="${stage>=4?.18:.6}"/>`).join('')}</g>`;
 if(stage>=4)s+=`<g ${P.clip}>${z.map(v=>`<circle cx="${P.x(v*w[0]).toFixed(1)}" cy="${P.y(v*w[1]).toFixed(1)}" r="4" fill="${palette.noise}" fill-opacity=".7"/>`).join('')}</g>`;
 // right: the one number per eruption
 if(stage>=3){const Q=panel({ox:610,W:490,H:150,L:20,R:14,T:40,B:46,xMin:-3,xMax:3,yMax:1,ticksX:[-2,-1,0,1,2],yAxis:false,fs:17,xLabel:'z = w₁ · x'});
  s+=Q.axes+label(630,26,'one number per eruption',{fill:palette.noise,size:20,weight:600})+z.map(v=>`<line x1="${Q.x(v).toFixed(1)}" y1="${Q.y(0)}" x2="${Q.x(v).toFixed(1)}" y2="${Q.y(0)-22}" stroke="${palette.noise}" stroke-opacity=".5" stroke-width="2"/>`).join('');}
 if(stage>=2)s+=label(630,210,`share of the spread along w₁: ${f(100*(1+rho)/2,0)}%`,{fill:palette.ink,size:21})+label(630,242,`the other direction keeps ${f(100*(1-rho)/2,0)}%`,{size:20});
 if(stage>=4)s+=label(630,300,'rebuild: x̂ = z · w₁ (orange)',{fill:palette.noise,size:21,weight:600})+label(630,332,'two numbers per eruption became one',{size:20});
 return fig(W,H,s,'PCA in two dimensions: the direction of most spread, each point projected to one number, and rebuilt on the line.');
}
/** A small line chart with direct labels. series: [{x, y, color, label, dashed}] */
function lineChart({series,W=640,H=340,xMin,xMax,yMin,yMax,ticksX,ticksY,xLabel,yLabel,fmtX,fmtY,ox=0,extra='',L=70}){
 const P=panel({ox,W,H,L,R:120,T:40,B:52,xMin,xMax,yMin,yMax,ticksX,ticksY,xLabel,fmtX,fmtY,fs:18,xAxisAtBottom:true});
 let s=P.axes+(yLabel?label(ox+L,22,yLabel,{size:18}):'');
 for(const sr of series){s+=P.line(sr.x,sr.y,sr.color,sr.dashed?'stroke-dasharray="7 5"':'')+(sr.dots?sr.x.map((x,i)=>P.dot(x,sr.y[i],sr.color,5)).join(''):'');
  const li=sr.labelAt??sr.x.length-1;
  if(sr.label)s+=label(P.x(sr.x[li])+8,P.y(sr.y[li])+6+(sr.dy||0),sr.label,{fill:sr.color,size:18,weight:600});}
 return {s:s+extra,P};
}

// ================================================================= Part 1 · Why generate?
chap('1.1');
S('faces','why','Who are these people?',{builds:2,
 body:act(1,`<figure class="evidence"><img src="assets/gallery/stylegan-faces.jpg" alt="A grid of photographs of faces" style="max-height:330px"></figure>`,at(2,p('None of them exist. A network turned random numbers into each face.'))),
 lines:['Look closely: ages, hair, glasses, backgrounds. Who are these people?','Nobody. Each face is a sample from a generative model (StyleGAN, 2019): random numbers in, a face out.'],
 ask:['Who are these people? Would you have known?','What did the network need, to make these?'],
 expected:['Students guess celebrities or stock photos. None exist.','Many real photographs to learn from, and a way to turn random numbers into new ones. That is today’s subject.'],
 transition:'A smaller test, on the digits we will use all lecture.',
 source:'Faces generated by StyleGAN: Karras, Laine and Aila, “A Style-Based Generator Architecture for GANs”, CVPR 2019. NVIDIA, CC BY-NC 4.0.'});
const RF=V.realfake_is_fake,rfMarks=RF.map(x=>x?palette.loss:palette.p);
S('realfake','why','Real or generated? Six of these twelve digits are fake',{builds:2,
 body:act(1,cols(`<div class="state-slot"><div data-only="1">${digits('realfake',{cols:6,size:104,gap:14})}</div><div data-only="2">${digits('realfake',{cols:6,size:104,gap:14,marks:rfMarks,labels:RF.map(x=>x?'<span class="tag fake">generated</span>':'<span class="tag real">real</span>')})}</div></div>`,
  stack(p('Six are handwritten MNIST test digits. Six come from a network.'),at(2,p(`The fakes: numbers ${RF.map((x,i)=>x?i+1:null).filter(Boolean).join(', ')} (counting left to right, top to bottom).`)),at(2,small('By the end of today you will have trained the network that made them: a VAE with 16 random numbers per digit.'))),'1.55fr 1fr')),
 lines:['Six are real, six are generated. Write down which six you think are fake.','Answer: the six outlined in red and labelled “generated”. Some are easy (broken strokes); some fool most people.'],
 ask:['Which six are fake? Write the positions.','Which fakes gave themselves away, and how?'],
 expected:['Answers vary; take a quick show of hands for two or three digits.','Broken or doubled strokes and blurry edges give some away; the cleanest fakes are hard to tell.'],
 point:'Real digits: six random MNIST test images not used elsewhere (seed 77). Fakes: the first six draws z ~ N(0, I) through the 16-number VAE of Part 6 (seed 78), not chosen by hand.',
 transition:'Generative models are everywhere now; a quick tour, then the common idea.',source:OURS});
chap('1.2');
S('llm','why','A chatbot is a generative model: one prompt, three different answers',{builds:2,
 body:act(1,`<div class="prompt-box"><b>Prompt:</b> Write one sentence that opens a short story about the first monsoon rain in Gandhinagar.</div>`,
  `<div class="cards" style="grid-template-columns:1fr 1fr 1fr">${LLM.map((t,i)=>(i===0?'':'<div data-at="2">')+`<div class="sample-text"><b>Sample ${i+1}.</b> ${t.trim()}</div>`+(i===0?'':'</div>')).join('')}</div>`),
 lines:['We sent the same prompt to a language model three times. Here is the first reply.','Same prompt, three different replies. Each is a sample from a distribution over sentences.'],
 ask:['If we send the same prompt again, do we get the same sentence?','Why would we want different answers each time?'],
 expected:['No: the model samples, so each run differs.','Variety: one prompt has many good answers. A model that always gave the single most likely sentence would be repetitive.'],
 transition:'How does it choose? One word at a time.',
 source:'Generated for this lecture on 21 October 2026 with Gemini 3.5 Flash, temperature 1 (<code>assets/examples/provenance.json</code>).'});
function nextWordChart(){
 const top=NW.top,W=1000,H=360,maxP=.2,x0=230,w=640;let s=label(0,26,`“${NW.prefix} …”`,{fill:palette.ink,size:22,weight:600});
 top.forEach(([t,pr],i)=>{const y=52+i*34;s+=label(x0-14,y+22,t.trim(),{fill:palette.ink,size:21,anchor:'end'})+`<rect x="${x0}" y="${y+4}" width="${(pr/maxP*w).toFixed(1)}" height="24" fill="${palette.q}" fill-opacity="${i===0?.85:.45}"/>`+label(x0+pr/maxP*w+10,y+23,`${f(100*pr,1)}%`,{size:19});});
 s+=label(x0,52+top.length*34+24,`every other word in the vocabulary: ${f(100*NW.rest,0)}% in total`,{size:19});
 return fig(W,H,s,'Probabilities of the next word after a prompt, from a small language model.');
}
S('next-word','why','Under the hood: a probability for every possible next word',{builds:2,
 body:act(1,cols(nextWordChart(),stack(at(2,p('Sampling = pick a word with these probabilities, append it, repeat.')),at(2,small(`The same model’s three continuations: “${NW.samples.map(x=>x.split('\n')[0].trim()).join('”, “')}”.`))),'1.45fr 1fr')),
 lines:['A small language model (SmolLM2, 360 million weights) scores every word it knows as the next word. “wet” gets 17%.','Sample a word, append it, repeat. That is all a chatbot does when it writes, at a much larger scale.'],
 ask:['Which word would the model pick if it always took the most likely one?','What is the model a distribution over?'],
 expected:['“wet”: 17%, but 83% of the time something else is sampled.','The next word given the words so far. Chaining them gives a distribution over whole sentences.'],
 point:'Probabilities are real: SmolLM2-360M-Instruct, computed locally. We do not train language models in this course; the point is only that generation = repeated sampling.',
 transition:'Pictures: one prompt, many images.',
 source:'SmolLM2-360M-Instruct (Hugging Face), next-token probabilities and three samples at temperature 1, computed for this lecture.'});
S('t2i','why','Text to image: one prompt, four samples',{builds:2,
 body:act(1,`<div class="prompt-box"><b>Prompt:</b> A watercolour painting of a camel walking across the white salt desert of the Rann of Kutch at sunrise, soft light, wide view.</div>`,
  `<div class="photo-row" style="grid-template-columns:repeat(4,minmax(0,1fr))">${[1,2,3,4].map(i=>(i===1?'':`<div data-at="2">`)+photo(`assets/examples/t2i-sample-${i}.jpg`,`sample ${i}`)+(i===1?'':'</div>')).join('')}</div>`),
 lines:['Same idea for pictures. One prompt, and the first picture the model returned.','Three more runs, same prompt: a rider appears, a walker, a different horizon. Four samples.'],
 ask:['What stayed the same across the four, and what changed?','Where does the variety come from?'],
 expected:['Same: camel, salt flat, sunrise, watercolour. Changed: rider, framing, sky.','From random numbers fed into the model; the prompt only narrows what is plausible.'],
 transition:'Images can also be the input.',
 source:'Generated for this lecture on 21 October 2026 with Gemini 3 Pro Image (<code>assets/examples/provenance.json</code>).'});
S('img2img','why','Image in, image out',{builds:2,
 body:act(1,`<div class="state-slot"><div data-only="1"><div class="photo-row" style="grid-template-columns:repeat(6,minmax(0,1fr))">${photo('assets/gallery/shoe-input.jpg','edges of a shoe')+photo('assets/gallery/shoe-output.jpg','pix2pix: a photo')+photo('assets/gallery/spade-layout.jpg','a label map')+photo('assets/gallery/spade-out-0.jpg','SPADE: a landscape')+photo('assets/gallery/horse.jpg','a horse')+photo('assets/gallery/zebra.jpg','CycleGAN: a zebra')}</div></div>`+
  `<div data-only="2"><div class="photo-row" style="grid-template-columns:repeat(4,minmax(0,1fr))">${photo('assets/gallery/spade-layout.jpg','one label map')+photo('assets/gallery/spade-out-0.jpg','sample 1')+photo('assets/gallery/spade-out-1.jpg','sample 2')+photo('assets/gallery/spade-out-2.jpg','sample 3')}</div></div></div>`),
 lines:['The input can be an image: a sketch becomes a photo, a coloured map becomes a landscape, a horse becomes a zebra.','The same label map, sampled three times: three different photographs. Still a distribution, now given an input.'],
 ask:['Is there one right answer for “photo of this sketch”?','So what is the model a distribution over?'],
 expected:['No: colour, texture and lighting are not in the sketch.','Over images, given the input image: p(photo | sketch).'],
 transition:'Sound works the same way.',
 source:'Isola et al., pix2pix (CVPR 2017); Park et al., SPADE (CVPR 2019); Zhu et al., CycleGAN (ICCV 2017). Attributed teaching excerpts; copyright with the authors.'});
function waveFig(){
 const W=1100,H=230,P=panel({W,H,L:20,R:14,T:20,B:46,xMin:0,xMax:WAVE.seconds,yMin:-1,yMax:1,ticksX:[0,2,4,6],yAxis:false,xLabel:'seconds',fs:18});
 const n=WAVE.max.length,xs=lin(0,WAVE.seconds,n);
 let pts=xs.map((x,i)=>`${P.x(x).toFixed(1)},${P.y(WAVE.max[i]).toFixed(1)}`).join(' ')+' '+xs.map((x,i)=>`${P.x(xs[n-1-i]).toFixed(1)},${P.y(WAVE.min[n-1-i]).toFixed(1)}`).join(' ');
 return fig(W,H,P.axes+`<polygon points="${pts}" fill="${palette.p}" fill-opacity=".55"/>`,'The waveform of a generated sentence of speech.');
}
S('speech','why','Sound: text in, 24,000 numbers per second out',{builds:2,
 body:act(1,`<div class="prompt-box"><b>Text:</b> “A generative model turns a few random numbers into a new example: a sentence, a picture, or this voice.”</div>`,waveFig(),
  `<div class="audio-row"><audio controls preload="none" src="assets/examples/tts-sample.mp3"></audio>${at(2,`<span class="mini">${comma(WAVE.samples)} numbers for ${f(WAVE.seconds,1)} seconds. A few of them: ${WAVE.first.slice(0,6).map(v=>f(v,3)).join(', ')}, …</span>`)}</div>`),
 lines:['A text-to-speech model read this sentence aloud. Press play.','The output is just a long list of numbers: the air pressure, 24,000 times a second. Generating sound = generating that list.'],
 ask:['What exactly does the model output?','Is there one right way to say this sentence?'],
 expected:['About 191,000 numbers: a waveform sampled 24,000 times a second.','No: speed, pitch, pauses and voice all vary. Another sample would sound different.'],
 transition:'Generative models also matter far from art and chat.',
 source:'Generated for this lecture on 21 October 2026 with Gemini 3.8 Flash TTS, voice “Kore”.'});
chap('1.3');
S('science','why','Beyond pictures: generative models in science',{builds:2,
 body:act(1,`<div class="cards" style="grid-template-columns:1fr 1fr">${card('Weather','GenCast samples 50 possible futures for the next 15 days. The spread of the samples says how sure the forecast is.',palette.data,'Price et al., Nature 2025')+card('Proteins','RFdiffusion generates new protein shapes, for example ones that bind a chosen target.',palette.p,'Watson et al., Nature 2023')}${at(2,card('Materials','MatterGen proposes new crystal structures with properties you ask for.',palette.q,'Zeni et al., Nature 2025'))}${at(2,card('Missing data','Sensors drop readings. A model of p(x) fills the gap with plausible values, not a flat average, and says how unsure it is.',palette.noise,'for example, air-quality and smart-meter records'))}</div>`),
 lines:['Weather and proteins: the output is a sample, and many samples together describe what could happen.','Materials, and the everyday case: filling in missing sensor readings with plausible values.'],
 ask:['Why would a forecaster want 50 samples instead of one best forecast?','Why not fill a gap in a sensor record with the average value?'],
 expected:['The samples show the range of possible futures: if they disagree, the forecast is uncertain.','The average is smooth and typical-looking but may never happen, and it hides the uncertainty. A sample looks like real data.'],
 transition:'Collect the uses.',
 source:'Price et al., “Probabilistic weather forecasting with machine learning” (GenCast), Nature 2025; Watson et al., “De novo design of protein structure and function with RFdiffusion”, Nature 2023; Zeni et al., “A generative model for inorganic materials design” (MatterGen), Nature 2025.'});
S('uses','why','What a model of p(x) is good for',{builds:2,
 body:act(1,`<div class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">${card('Create','new text, images, sound, designs',palette.q)+card('Fill in','missing pixels, missing readings, the rest of a sentence',palette.noise)+card('Simulate','many plausible futures, not one average',palette.data)}${at(2,card('Spot the odd one out','a reading with very low p(x) is suspicious: a faulty sensor, a fraud',palette.loss))}${at(2,card('Compress','likely data needs fewer bits: Lecture 1’s codes',palette.p))}${at(2,card('Make more data','extra training examples, or shareable synthetic records',palette.d))}</div>`),
 lines:['Three uses that need samples: create, fill in, simulate.','Three that need scores: spot anomalies, compress, and generate extra data when real data is scarce or private.'],
 ask:['Which of these needs only samples? Which needs a score p(x)?','Which use connects to Lecture 1?'],
 expected:['Create, simulate, augment: samples. Anomalies and compression: a score.','Compression: an event with probability p costs log₂(1/p) bits.'],
 transition:'What do all these models have in common?'});
chap('1.4');
S('recipe','why','What they share: random numbers in, a learned network, a sample out',{builds:2,
 body:act(1,`<div class="state-slot"><div data-only="1">${recipeDiagram(1)}</div><div data-only="2">${recipeDiagram(2)}</div></div>`),
 lines:['Every model on the tour: random numbers z go in, a network with learned weights θ turns them into a sample x.','Most also take a condition c that steers the sample: a prompt, a class label, a sketch. Then the model is p(x | c).'],
 ask:['What plays the role of z in the chatbot?','What is the condition for the camel pictures?'],
 expected:['The random choices made when sampling each next word.','The text prompt.'],
 transition:'The condition in each example.'});
S('conditions','why','Conditions steer the sample; today we learn p(x) without one',{builds:2,
 body:act(1,`<div class="tsm">${table(['','condition c','sample x'],[['chatbot','your prompt','a reply'],['text to image','a prompt','a picture'],['pix2pix, SPADE, CycleGAN','a sketch, a map, a photo','a photo'],['text to speech','a sentence','a waveform'],['today','none','a 28 × 28 digit']])}</div>`,at(2,p('Learning a distribution over images is the hard part, with or without a condition.'))),
 lines:['Each example on the tour fills in c and x differently. Same recipe: random numbers in, learned network, sample out.','Today we drop the condition and learn p(x) for handwritten digits: the core problem underneath all of these.'],
 ask:['What would a class label as the condition let you do with digits?','Why start without a condition?'],
 expected:['Ask for a particular digit: “draw me a 7”, with z choosing the style.','Conditioning adds an input; learning a distribution over images is the same problem.'],
 transition:'Two jobs such a model can do.'});
S('two-jobs','why','A model of p(x) can do two jobs: score and sample',{builds:2,
 body:act(1,cols(stack(p(`<b class="p">Score:</b> how plausible is this x?`),eq(R`p(\text{a clean }3)\ \text{high},\qquad p(\text{static})\ \text{tiny}`),at(2,p(`<b class="q">Sample:</b> make a new x that looks like the data.`))),
  `<div class="dflex" style="display:flex;gap:24px;align-items:end">${digit('example',0,150)}${digit('uniform',0,150)}${at(2,digit('vae16_prior_samples',3,150))}</div>`,'1.1fr 1fr')),
 lines:['Job one, score: a number that is high for a real-looking 3 and tiny for random static.','Job two, sample: produce a new x. Both jobs come from the same fitted distribution.'],
 ask:['Which job does the chatbot do when it writes?','Which job does anomaly detection need?'],
 expected:['Sample.','Score: flag inputs with low p(x).'],
 transition:'Why not just output the average example?'});
const mean3=(()=>{const s=imgs('display'),idx=M.display_labels.map((l,i)=>l===3?i:-1).filter(i=>i>=0),out=new Float32Array(784);for(const i of idx)for(let k=0;k<784;k++)out[k]+=s.px[i*784+k]/255/idx.length;return out;})();
S('why-not-mean','why','Why not just output the average?',{builds:2,
 body:act(1,cols(`<div class="dflex" style="display:flex;gap:18px;align-items:end">${digits('display',{idx:M.display_labels.map((l,i)=>l===3?i:-1).filter(i=>i>=0).slice(0,5),cols:5,size:84})}</div>`,
  stack(`<div class="dflex" style="display:flex;gap:16px;align-items:center"><span class="lead">average of 100 threes:</span>${rawDigit(mean3,120)}</div>`,at(2,p('Nobody writes this 3. The average is the blur in the middle of many real 3s.'))),'1.1fr 1fr')),
 lines:['Five real 3s, and the pixel-by-pixel average of 100 of them.','The average looks like no real 3. We need the whole distribution, so we can draw one sharp 3 at a time.'],
 ask:['Would you accept the average as a new handwritten 3?','When is the average still useful?'],
 expected:['No: it is a ghost of many 3s.','As a summary, or a prediction scored by squared error. Not as a sample.'],
 transition:'The route for today.'});
S('route','why','Today’s route: six parts',{
 body:act(1,`<div class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">${card('1 · Why generate?','what generative models do, and the shared recipe',palette.ink)+card('2 · Fit, then sample','one number per example: a Gaussian, then two bumps',palette.data)+card('3 · PCA and Gaussians','784 numbers per digit: compress first, then fit',palette.noise)+card('4 · Autoencoders','a learned, bent version of PCA',palette.p)+card('5 · Can it generate?','random codes, and why they fail',palette.loss)+card('6 · The VAE','a recipe we can sample, trained with Lecture 1’s ELBO',palette.q)}</div>`),
 lines:['From one number per example to 784, from fitting by formula to training networks, ending with the variational autoencoder.'],
 ask:['Which part will need Lecture 1?'],expected:['Part 6: the posterior over codes is intractable, so we use the ELBO.'],
 transition:'Part 2: the simplest case, one number per example.'});

// ================================================================= Part 2 · Fit a distribution, then sample
chap('2.1');
const inRange=(a,b)=>F.waiting.filter(x=>x>=a&&x<=b).length;
S('faithful','fit','Old Faithful: 272 eruptions, and the wait before each one',{builds:2,
 body:act(2,cols(stack(p('A geyser in Yellowstone. Rangers record how long visitors wait between eruptions.'),p(`The first five waits, in minutes: <b>${F.first5.join(', ')}</b>, …`),at(2,small(`All ${F.n} waits as a histogram: two clumps, around 55 and 80 minutes.`))),
  `<div class="state-slot"><div data-only="1">${fig(700,330,waitPanel({W:700,H:330,data:false}).s+`<g>${F.waiting.map(x=>`<line x1="${(16+(x-40)/60*670).toFixed(1)}" y1="250" x2="${(16+(x-40)/60*670).toFixed(1)}" y2="276" stroke="${palette.data}" stroke-opacity=".35" stroke-width="2"/>`).join('')}</g>`,'All 272 waiting times as tick marks.')}</div><div data-only="2">${fig(700,330,waitPanel({W:700,H:330,dataFill:palette.data+'88'}).s,'Histogram of the 272 waiting times: two clumps.')}</div></div>`,'1fr 1.25fr')),
 lines:['One number per example: the wait until the next eruption. Each tick is one of 272 eruptions.','The histogram shows two clumps: short waits around 55 minutes, long ones around 80. Few waits near 70.'],
 ask:['If you had to invent one new plausible wait, what number would you say?','What shape does the histogram have?'],
 expected:['Most say about 70, the average. Hold that thought.','Two bumps, with a dip near 65–70 minutes.'],
 transition:'The task, in three boxes.',source:FAITHFUL_SRC});
S('task','fit','The task: examples in, a distribution out, new examples from it',{builds:3,
 body:act(2,`<div class="state-slot"><div data-only="1">${taskDiagram(1)}</div><div data-only="2">${taskDiagram(2)}</div><div data-only="3">${taskDiagram(3)}</div></div>`),
 lines:['We start with examples: 272 waiting times.','Fit: choose a distribution p(x) from a family by maximum likelihood. Lecture 1: maximum likelihood minimises KL(data ‖ model).','Sample: run the fitted distribution forwards to get new waits that nobody measured.'],
 ask:['What do we have at the start?','What does “fit” mean here?','What does “sample” mean?'],
 expected:['Only examples, no formula.','Choose the parameters that make the observed waits most probable.','Draw new numbers from p(x), for example with torch.distributions.'],
 transition:'Why not just use the average?'});
S('mean-bad','fit',`The average wait, ${f(F.gauss.mu,1)} minutes, is a poor sample`,{builds:2,
 body:act(2,cols(fig(700,330,waitPanel({W:700,H:330,dataFill:palette.data+'88',marks:[[F.gauss.mu,palette.loss,`average ${f(F.gauss.mu,1)}`]]}).s,'The average falls in the dip between the clumps.'),
  stack(p(`Only <b>${inRange(66,76)}</b> of ${F.n} waits lie within 5 minutes of the average.`),at(2,p('The average sits in the dip. To make realistic waits we need the shape of the data, not one number.'))),'1.3fr 1fr')),
 lines:[`The average is ${f(F.gauss.mu,1)} minutes, in the dip between the clumps. Few real waits are near it.`,'A realistic new wait should come from one of the clumps, as often as the clumps occur. That needs a distribution.'],
 ask:['How many waits lie between 66 and 76 minutes?','What would a good generator do instead?'],
 expected:[`${inRange(66,76)} of 272.`,'Produce short waits about a third of the time and long waits otherwise, with realistic spread.'],
 transition:'Start with the simplest distribution: one Gaussian.',source:FAITHFUL_SRC});
chap('2.2');
S('gauss-fit','fit','One Gaussian: the best μ and σ have a formula',{builds:3,
 body:act(2,cols(`<div class="eq-sm">`+stack(eq(R`p(x)=\N(x;\,\mu,\sigma^2)`),at(2,eq(R`\text{maximise}\quad \frac1n\sum_{i=1}^{n}\log \N(x_i;\,\mu,\sigma^2)`)),
  at(3,eq(R`\hat\mu=\frac1n\sum_i x_i=${f(F.gauss.mu,1)}`)),at(3,eq(R`\hat\sigma^2=\frac1n\sum_i (x_i-\hat\mu)^2=${f(F.gauss.sd,1)}^2`)))+`</div>`,
  at(3,fig(620,330,waitPanel({W:620,H:330,dataFill:palette.data+'66',gauss:true}).s+legend(16,30,[[`N(${f(F.gauss.mu,1)}, ${f(F.gauss.sd,1)}²)`,palette.q]],19),'One fitted Gaussian over the histogram.')),'1.15fr 1fr')),
 lines:['Family: one Gaussian, two numbers μ and σ.','Fit: choose μ and σ to maximise the average log-likelihood of the 272 waits.','Set the derivatives to zero: μ̂ is the average and σ̂² the average squared deviation. One bump, centred in the dip.'],
 ask:['How many numbers describe this model?','Why maximise the log of the likelihood rather than the likelihood?','Where is the peak of the fitted bell?'],
 expected:['Two: μ and σ.','The log turns a product over 272 waits into a sum, and does not move the maximum.',`At ${f(F.gauss.mu,1)}: in the dip, where there is little data.`],
 transition:'Use the fitted bell to score waits.',source:FAITHFUL_SRC});
S('gauss-score','fit','Score: how probable is each wait under the bell?',{builds:2,
 body:act(2,cols(fig(620,330,waitPanel({W:620,H:330,dataFill:palette.data+'55',gauss:true,marks:[[54,palette.noise,'54'],[71,palette.loss,'71'],[80,palette.data,'80']]}).s,'The fitted bell, with three waits marked.'),
  stack(table(['wait x','p(x) under the bell'],[['54 minutes',f(F.gauss.at['54'],4)],['71 minutes',f(F.gauss.at['71'],4)],['80 minutes',f(F.gauss.at['80'],4)]]),at(2,p(`Average log-likelihood of all 272: <b>${f(F.gauss.avg_ll,3)}</b>`))),'1.15fr 1fr')),
 lines:['The bell gives a density for every wait. It rates 71 minutes, in the empty dip, as the most likely wait of all.','One number scores the whole model: the average log p(x) over the data. We will try to beat it.'],
 ask:['Which of the three waits does the bell think is most likely? Is that right?','Is a higher average log-likelihood better or worse?'],
 expected:['71: wrong, few eruptions have waits near 71.','Better: the data are more probable under the model.'],
 transition:'Now sample from the bell.',source:FAITHFUL_SRC});
S('gauss-sample','fit','Sample from the bell: the draws fill the empty dip',{builds:2,
 body:act(2,widget('faith-sample','Draw waiting times from the fitted Gaussian.',{model:'gauss'})),
 lines:['Press +1 a few times, then +272: draws from the fitted Gaussian, over the real waits.','The draws pile up in the dip where real eruptions rarely are, and miss the two clumps.'],
 ask:['Before pressing: where will most draws land?','What is wrong with these samples?'],
 expected:['Near 71, the centre of the bell.','Too many waits in the dip, too few in the clumps: one bump cannot make two clumps.'],
 demo:'Notebook 2.2: the sampling cell draws 272 waits with torch.distributions.Normal(mu, sigma).sample((272,)) and plots both histograms.',
 transition:'Why are there two clumps?',source:FAITHFUL_SRC});
chap('2.3');
S('two-kinds','fit','The geyser has two kinds of eruption',{builds:2,
 body:act(2,cols(`<div class="state-slot"><div data-only="1">${fig(600,420,faith2d().s,'Eruption length against the next wait.')}</div><div data-only="2">${fig(600,420,faith2d({colourBy:true}).s,'The same points coloured by kind of eruption.')}</div></div>`,
  stack(p('Add a second measurement: how long each eruption lasted.'),at(2,p(`Short eruptions (about ${f(F.mix.dur_short,1)} min) come before short waits; long ones (about ${f(F.mix.dur_long,1)} min) before long waits.`)),at(2,small('The kind of eruption is a hidden choice. If we only see the wait, we never see the kind directly.'))),'1.2fr 1fr')),
 lines:['Plot each eruption’s length against the wait that follows it. Two groups.','Nature first “chooses” a short or a long eruption; the wait then depends on that choice.'],
 ask:['How many groups do you see?','If you saw only the wait, could you always tell the kind of eruption?'],
 expected:['Two.','Usually, from which clump it is in; near the dip, not for sure.'],
 transition:'Turn that story into a recipe.',source:FAITHFUL_SRC});
S('mixture-story','fit','A recipe with a hidden step: pick a bump, then draw',{builds:3,
 body:act(2,stack(eq(R`k\sim\text{Categorical}(\pi_1,\pi_2)\qquad\text{(short or long)}`),eq(R`x\mid k\sim\N(\mu_k,\sigma_k^2)\qquad\text{(the wait, given the kind)}`),
  at(2,p('<b>k</b> is a <b>latent</b> (hidden) variable: it shapes the data, but we never observe it.')),at(3,eq(R`p(x)=\sum_{k=1}^{2}\pi_k\,\N(x;\,\mu_k,\sigma_k^2)\qquad\text{a mixture of Gaussians}`)))),
 lines:['Two steps: choose a kind k with probabilities π₁, π₂; then draw the wait from that kind’s Gaussian.','k is a latent variable: hidden, but part of the recipe. This idea returns in every later part.','The distribution of x alone adds up both ways x could have been made: a mixture of Gaussians.'],
 ask:['How many numbers describe this model?','Name another latent variable from everyday data.','Why a sum over k?'],
 expected:['Five: π₁ (π₂ = 1 − π₁), μ₁, μ₂, σ₁, σ₂.','For example: which person wrote a digit, the season behind a temperature, the topic behind a sentence.','x could come from either bump; the two ways are mutually exclusive, so their probabilities add.'],
 transition:'Score one wait under the mixture.'});
const T66=F.mix.terms['66'],T80=F.mix.terms['80'];
S('mixture-score','fit','To score a wait, add over both kinds',{builds:3,
 body:act(2,cols(fig(620,330,waitPanel({W:620,H:330,dataFill:palette.data+'44',mixture:true,marks:[[66,palette.ink,'66'],[80,palette.ink,'80']]}).s+legend(16,30,[['short',BUMP[0],true],['long',BUMP[1],true],['mixture',palette.p]],19),'The fitted mixture with two waits marked.'),
  `<div class="eq-sm">`+stack(eq(R`p(66)=\pi_1\N(66;\,\text{short})+\pi_2\N(66;\,\text{long})`),eq(R`\phantom{p(66)}=${f(T66[0],4)}+${f(T66[1],4)}=${f(T66[0]+T66[1],4)}`),
   at(2,eq(R`p(80)=${f(T80[0],6)}+${f(T80[1],4)}=${f(T80[0]+T80[1],4)}`)),
   at(3,p(`Average log-likelihood: <b>${f(F.mix.avg_ll,3)}</b>, up from ${f(F.gauss.avg_ll,3)} for one bump.`)))+`</div>`,'1fr 1.05fr')),
 lines:['At 66 minutes, between the clumps, both kinds contribute. Add them.','At 80 minutes, almost all the probability comes from the long kind.',`Over all 272 waits the mixture scores ${f(F.mix.avg_ll,3)} against ${f(F.gauss.avg_ll,3)}: the data are e^${f(F.ll_gain_total,0)} times more probable under it.`],
 ask:['Which kind explains a wait of 66 minutes better?','Why is the short term at 80 almost zero?','Is a difference of 0.23 per wait a lot?'],
 expected:['The short kind contributes slightly more here.','80 is 4 σ above the short bump’s centre.',`Over 272 waits it is ${f(F.ll_gain_total,0)} nats: a likelihood ratio of about 10^${f(F.ll_gain_total/Math.LN10,0)}.`],
 point:'These are the fitted values from the next section; showing the score first motivates fitting.',
 transition:'How were the five numbers found? The same loop as Lecture 1.',source:FAITHFUL_SRC});
chap('2.4');
S('mix-code','fit','Fit it by gradient descent, as Lecture 1 fitted q',{builds:2,
 body:act(2,cols(`<div class="code-sm">${code(`# x: the 272 waits, standardised
logits = torch.zeros(2, requires_grad=True)
mu = torch.tensor([-.25, .25], requires_grad=True)
log_sd = torch.zeros(2, requires_grad=True)
opt = Adam([logits, mu, log_sd], lr=0.03)
for step in range(600):
    log_pi = torch.log_softmax(logits, 0)
    comp = Normal(mu, log_sd.exp()).log_prob(x[:, None])
    loss = -torch.logsumexp(log_pi + comp, 1).mean()
    opt.zero_grad(); loss.backward(); opt.step()`)}</div>`,
  stack(at(2,mini('<code>log_softmax</code>: π₁, π₂ stay positive and add to 1.')),at(2,mini('<code>log_sd.exp()</code>: σ stays positive.')),at(2,mini('<code>logsumexp</code>: log of the sum over kinds, computed safely.'))),'1.9fr 1fr')),
 lines:['Same loop as Lecture 1: parameters, a loss (the negative average log-likelihood), Adam.','Three details keep the parameters valid: softmax for π, exp for σ, and logsumexp for the log of a sum.'],
 ask:['What is the loss?','Why not optimise σ directly?'],
 expected:['Minus the average of log p(x_i) over the 272 waits.','A gradient step could make σ negative; optimising log σ keeps σ = exp(log σ) positive.'],
 point:'Waits are standardised before fitting (as in the notebook); values shown are converted back to minutes. EM is a faster classic algorithm for mixtures; the notebook uses it in Part 3.',
 transition:'Watch it run.',nb:'sec-2-4'});
S('mix-fit','fit','Press play: the bumps separate onto the two clumps',{builds:2,
 body:act(2,widget('faith-fit','A two-bump mixture fitted by gradient descent.')),
 lines:['Both bumps start wide and near the middle.',`They pull apart onto the clumps; the score rises from ${f(F.mix.frames[0][6],2)} to ${f(F.mix.avg_ll,2)}.`],
 ask:['What should happen to the two bumps?','Why does the curve flatten at the end?'],
 expected:['Move apart, one onto each clump, and narrow.','Near the maximum the gradient is close to zero.'],
 transition:'Read off the fitted numbers.',source:FAITHFUL_SRC});
S('mix-result','fit','It found the two kinds of eruption without being told',{builds:2,
 body:act(2,cols(table(['','short kind','long kind'],[['share π',f(F.mix.pi[0],2),f(F.mix.pi[1],2)],['centre μ (min)',f(F.mix.mu[0],1),f(F.mix.mu[1],1)],['width σ (min)',f(F.mix.sd[0],1),f(F.mix.sd[1],1)]]),
  stack(at(2,p(`Waits it puts in the short bump follow eruptions of <b>${f(F.mix.dur_short,2)}</b> min on average; long bump: <b>${f(F.mix.dur_long,2)}</b> min.`)),at(2,small('It never saw eruption lengths. The hidden variable it invented matches a real one.'))),'1fr 1fr')),
 lines:[`About ${f(100*F.mix.pi[0],0)}% short waits near ${f(F.mix.mu[0],0)} minutes, ${f(100*F.mix.pi[1],0)}% long waits near ${f(F.mix.mu[1],0)}.`,'Check against the eruption lengths, which the fit never saw: the two bumps line up with short and long eruptions.'],
 ask:['What does π₁ = 0.36 mean in words?','Why is the match with eruption length remarkable?'],
 expected:['About 36% of eruptions are followed by a short wait.','The fit used only waits; it recovered a real hidden cause from the shape of the data.'],
 transition:'Now sample from the mixture.',source:FAITHFUL_SRC});
S('mix-sample','fit','Sample: pick a bump, then a time',{builds:2,
 body:act(2,widget('faith-sample','Draw waits from the two-bump mixture: pick a bump, then draw a time.',{model:'mix'})),
 lines:['Sampling runs the recipe: pick short or long with probabilities 0.36 and 0.64, then draw from that bump.','After 272 draws the histogram of samples matches the data: two clumps, few in the dip.'],
 ask:['What fraction of draws should pick the long bump?','How is this better than the single bell?'],
 expected:['About 64%.','The samples reproduce both clumps and the dip.'],
 demo:'Notebook 2.4: the sampling cell draws k from Categorical(pi), then x from Normal(mu[k], sd[k]).',
 transition:'Would three bumps be even better?',source:FAITHFUL_SRC});
chap('2.5');
function heldoutChart(){
 const h=F.heldout,K=h.map(r=>r.K);
 return lineChart({W:640,H:330,xMin:1,xMax:6,yMin:-4.1,yMax:-3.65,ticksX:[1,2,3,4,5,6],ticksY:[-4.1,-4,-3.9,-3.8,-3.7],xLabel:'number of bumps K',yLabel:'average log-likelihood',
  series:[{x:K,y:h.map(r=>r.train),color:MUTED,label:'200 training waits',dots:true,dy:14},{x:K,y:h.map(r=>r.test),color:palette.p,label:'72 held-out waits',dots:true}]});
}
S('how-many','fit','More bumps always fit the training waits better; new waits decide',{builds:2,
 body:act(2,cols(fig(780,330,heldoutChart().s,'Training and held-out average log-likelihood against the number of bumps.'),stack(p('Fit on 200 waits, score on the other 72.'),at(2,p('Held-out score is best at <b>K = 2</b>, then drops slightly.'))),'1.5fr 1fr')),
 lines:['Fit K = 1 to 6 bumps on 200 waits. The training score only goes up, or stays level.','On the 72 held-out waits the score peaks at two bumps. Extra bumps fit noise in the training waits.'],
 ask:['Why can the training score never get worse with more bumps?','How would you choose K for a new dataset?'],
 expected:['A model with K + 1 bumps can copy the K-bump fit (give the extra bump zero weight).','Hold out data, or use a score that penalises extra parameters.'],
 transition:'Two numbers per eruption.',source:FAITHFUL_SRC});
chap('2.6');
S('two-d','fit','Two numbers per eruption: Gaussians need a covariance',{builds:3,
 body:act(2,cols(`<div class="state-slot"><div data-only="1">${fig(600,420,faith2d({gauss:true}).s,'One 2-D Gaussian over the eruptions.')}</div><div data-only="2">${fig(600,420,faith2d({mix:true}).s,'A mixture of two 2-D Gaussians.')}</div><div data-only="3">${fig(600,420,faith2d({mix:true,samples:F.samples2.mix}).s,'272 samples from the 2-D mixture.')}</div></div>`,
  stack(p(`One 2-D Gaussian: a mean and a 2 × 2 covariance. Correlation ${f(F.g2.corr,2)}.`),at(2,p(`Two 2-D bumps, each with its own covariance: average log-likelihood ${f(F.m2.avg_ll,2)} against ${f(F.g2.avg_ll,2)}.`)),at(3,p('Samples from the 2-D mixture look like the eruptions.'))),'1.2fr 1fr')),
 lines:['With two numbers per example, a Gaussian needs a 2 × 2 covariance: how length and wait vary together (Lecture 1, section 3.3).','A mixture of two 2-D Gaussians: one tilted ellipse per kind of eruption.','Sampling works exactly as before: pick a kind, then draw a pair from its 2-D Gaussian.'],
 ask:['What does the tilt of the ellipse say?','How many numbers does each 2-D bump need?','What changes if each example had 784 numbers?'],
 expected:['Longer eruptions go with longer waits: positive correlation.','2 for the mean, 3 for the symmetric covariance, plus its share π.','A full covariance needs 784 × 785 / 2 ≈ 300,000 numbers per bump. That is Part 3’s problem.'],
 transition:'Recap Part 2.',source:FAITHFUL_SRC});
S('recap-2','fit','Part 2 in one slide',{builds:2,
 body:act(2,table(['step','one number per example','what we learned'],[['fit','choose parameters to maximise the average log p(x)','same loop as Lecture 1'],['score','evaluate log p(x) of any wait','held-out score picks the model'],['sample','run the recipe: pick a bump, then draw','hidden variables make samples realistic']]),at(2,p('Next: each example is a 28 × 28 image, 784 numbers. Same three steps, much harder.'))),
 lines:['Fit, score and sample, with a hidden choice that made the samples realistic.','Next, images: the same three steps with 784 numbers per example.'],
 ask:['Which step needed the hidden variable?','What will break first when x has 784 numbers?'],
 expected:['Sampling realistic waits, and fitting the two clumps.','The number of parameters (covariances), and finding good hidden variables.'],
 transition:'Part 3: images.'});

// ================================================================= Part 3 · Images: PCA and Gaussians
chap('3.1');
S('mnist','pca','MNIST: 70,000 handwritten digits, 28 × 28 pixels each',{builds:2,
 body:act(3,cols(digits('first_train',{cols:8,size:72,gap:8}),stack(p(`${comma(M.n_train)} for training, ${comma(M.n_test)} for testing.`),at(2,`<div class="dflex" style="display:flex;gap:18px;align-items:center">${digit('example',0,130)}<p class="lead">Our running example: test digit #${d.meta.example_index}, a 3.</p></div>`)),'1.15fr 1fr')),
 lines:['The first 16 training digits. Each is a 28 × 28 grey image of one handwritten digit.','We follow one digit all lecture: test digit #90, a 3.'],
 ask:['How many pixels does each image have?','Why keep one running example?'],
 expected:['28 × 28 = 784.','Every model can be checked on the same 3: its numbers, reconstruction and code.'],
 transition:'What the computer sees.',source:'MNIST (LeCun, Cortes and Burges), grey levels scaled to [0, 1].'});
function pixelTable(){
 const P=M.pixels3,r0=6,r1=16,c0=8,c1=19;let s='<table class="pix-table">';
 for(let r=r0;r<r1;r++){s+='<tr>';for(let c=c0;c<c1;c++){const v=P[r][c];s+=`<td style="background:rgba(20,23,31,${(v*.85).toFixed(2)});color:${v>.5?'#fff':MUTED}">${v===0?'0':f(v,1)}</td>`;}s+='</tr>';}
 return s+'</table>';
}
S('pixels','pca','Our 3 is 784 numbers between 0 and 1',{builds:2,
 body:act(3,cols(`<div class="dflex" style="display:flex;gap:26px;align-items:center">${digit('example',0,220)}${pixelTable()}</div>`,stack(p('Each pixel is a number: 0 is paper, 1 is full ink.'),at(2,eq(R`x\in[0,1]^{784}`)),at(2,p('One image = one point in a 784-dimensional space.'))),'1.5fr 1fr')),
 lines:['The pixels of a patch of our 3 (rows 6–15, columns 8–18), as numbers.','Flatten the 28 × 28 grid into a list: x is one point in a 784-dimensional space. A model of digits is a distribution over that space.'],
 ask:['How many of the 784 numbers are exactly 0 for a typical digit?','In Part 2, x was one number. What is x now?'],
 expected:['Most of them: the background.','A list of 784 numbers. The same three steps, fit, score and sample, now in 784 dimensions.'],
 transition:'What does a random point in that space look like?'});
S('random-points','pca','A random point in that space looks like static',{builds:2,
 body:act(3,cols(digits('uniform',{cols:4,size:110,gap:12}),stack(p('Each pixel drawn uniformly between 0 and 1.'),at(2,p('Almost every point in the space is static. Digits live in a tiny region; p(x) must find it.'))),'1.1fr 1fr')),
 lines:['Eight random points in the 784-dimensional space: every pixel uniform on [0, 1].','You would have to draw for longer than the age of the universe to hit a digit. A good p(x) concentrates on the tiny region where digits live.'],
 ask:['Would you ever expect a random point to look like a digit?','What must a model of digits learn, then?'],
 expected:['No: neighbouring pixels in digits are strongly related; random pixels are not.','Where in the space digits are, and how likely each region is.'],
 transition:'First attempt: fit Gaussians to raw pixels.'});
chap('3.2');
S('indep','pca','First try: one Gaussian per pixel, all independent',{builds:3,
 body:act(3,cols(stack(`<div class="dflex" style="display:flex;gap:22px;align-items:end">${`<figure class="dg-cell">${digit('pixel_mean',0,120)}<figcaption>mean of each pixel</figcaption></figure>`}${`<figure class="dg-cell">${digit('pixel_std',0,120)}<figcaption>spread of each pixel</figcaption></figure>`}</div>`,at(2,eq(R`p(x)=\prod_{j=1}^{784}\N(x_j;\,m_j,s_j^2)`)),at(2,p(`2 × 784 = ${comma(M.params.indep)} numbers.`))),
  at(3,stack(small('8 samples'),digits('indep_samples',{cols:4,size:96,gap:10}))),'1.1fr 1fr')),
 lines:['Fit a Gaussian to each pixel separately, like Part 2: its mean and its spread over the 60,000 training digits.','The model multiplies 784 independent bells: 1,568 numbers.','Samples are speckled noise around the average digit: every pixel decides on its own.'],
 ask:['Where is the spread largest?','What does “independent” assume about neighbouring pixels?','Why do the samples look speckled?'],
 expected:['In the middle, where strokes of different digits pass.','That knowing one pixel tells you nothing about its neighbour.','Neighbours are drawn independently, so no strokes form.'],
 transition:'Is the independence assumption reasonable?',source:'MNIST training set. '+OURS});
const corrURL=(()=>{const c=document.createElement('canvas');c.width=c.height=28;const g=c.getContext('2d'),im=g.createImageData(28,28),v=M.corr_centre.flat();
 for(let i=0;i<784;i++){const t=v[i],col=t>=0?[250-235*t,250-132*t,251-141*t]:[250+(190-250)*(-t)*2,250+(18-250)*(-t)*2,251+(60-251)*(-t)*2].map(x=>Math.max(0,Math.min(255,x)));im.data.set([...col.map(Math.round),255],i*4);}
 g.putImageData(im,0,0);return c.toDataURL();})();
S('neighbours','pca','Neighbouring pixels move together',{builds:2,
 body:act(3,cols(`<figure class="dg-cell"><img class="dg" src="${corrURL}" width="260" height="260" alt="Correlation of the centre pixel with every pixel"><figcaption>correlation of the centre pixel (14, 14) with every pixel: teal positive, red negative</figcaption></figure>`,
  stack(table(['pixel pair','correlation'],[['centre and its right neighbour',f(M.corr_pairs.right,2)],['two to the right',f(M.corr_pairs.two_right,2)],['five to the right',f(M.corr_pairs.five_right,2)],['a far corner pixel',f(M.corr_pairs.far,2)]]),at(2,p('An independent model throws all of this away.'))),'1fr 1.2fr')),
 lines:['If the centre pixel is inked, its neighbours usually are too: correlation 0.78 with the pixel to its right.','Strokes are made of neighbouring pixels that move together. A model of digits must capture that.'],
 ask:['Why is the correlation five pixels away negative?','What do we need instead of independent bells?'],
 expected:['A stroke through the centre often means white space a little further out.','A covariance between pixels: Part 2’s 2 × 2 covariance, now 784 × 784.'],
 transition:'Second try: one Gaussian with a full covariance.',source:'Correlations over the 60,000 MNIST training digits. '+OURS});
S('full-gauss','pca','Second try: one 784-dimensional Gaussian with a full covariance',{builds:2,
 body:act(3,cols(stack(eq(R`p(x)=\N(x;\,m,\Sigma),\qquad \Sigma\in\mathbb{R}^{784\times784}`),p(`${comma(M.params.full)} numbers: 784 for the mean, ${comma(M.params.full-784)} for the covariance.`),at(2,small('Samples have smoother, stroke-like blobs, but they are ghosts, not digits.'))),stack(small('8 samples'),digits('full_samples',{cols:4,size:96,gap:10})),'1fr 1fr')),
 lines:['One big Gaussian whose covariance links every pair of pixels: over 300,000 numbers.','Samples are smoother, because neighbouring pixels now move together. Still not digits: one bump for ten kinds of digit, and far too many numbers.'],
 ask:['Why are the samples smoother than before?','What is wrong with one bump? Think of Part 2.'],
 expected:['The covariance makes neighbours move together.','Digits form clusters (one per digit, at least); one Gaussian is Part 2’s single bell again.'],
 transition:'Idea: describe each image with a few numbers first.',source:OURS});
chap('3.3');
S('pca-2d','pca','Idea: describe each example with fewer numbers. PCA in 2-D first',{builds:4,
 body:act(3,`<div class="state-slot">${[1,2,3,4].map(k=>`<div data-only="${k}">${pca2d(k)}</div>`).join('')}</div>`),
 lines:['Back to the eruptions: two numbers each (standardised), strongly correlated.',`Find the direction along which the points spread most: here ${f(50*(1+F.g2.corr),0)}% of the spread.`,'Project each point onto that line: one number z per eruption instead of two.','Rebuild each point from its one number: it lands on the line, close to where it started. That is PCA.'],
 ask:['If you could keep one number per eruption, which would you keep?','How would you find the direction of most spread?','What is lost by projecting?','How many numbers did we save?'],
 expected:['A mix of length and wait: the position along the main diagonal.','The top eigenvector of the covariance matrix.','The distance from each point to the line: here only 5% of the spread.','Half: two numbers became one.'],
 point:'For standardised data with correlation ρ the directions are (1, 1)/√2 and (1, −1)/√2 with variances 1 ± ρ.',
 transition:'Now do this with 784 numbers.',source:FAITHFUL_SRC});
S('pca-mnist','pca','PCA on 60,000 digits: a mean image and directions',{builds:2,
 body:act(3,stack(`<div class="dflex" style="display:flex;gap:16px;align-items:end"><figure class="dg-cell">${digit('pca_mean',0,104)}<figcaption>mean</figcaption></figure><span class="lead">+</span>${digits('pca_components',{cols:8,size:92,gap:8,labels:['w₁','w₂','w₃','w₄','w₅','w₆','w₇','w₈']})}</div>`,
  at(2,eq(R`\underbrace{z=W^\top(x-m)}_{\text{encode: }784\to d}\qquad\qquad \underbrace{\hat x=m+Wz=m+z_1w_1+\dots+z_dw_d}_{\text{decode: }d\to784}`)))),
 lines:['PCA finds the directions of most spread in the 784-dimensional space: each direction is itself an image (teal adds ink, red removes it).','Encode: d numbers per image, z = Wᵀ(x − m). Decode: start from the mean and add z₁ times w₁, z₂ times w₂, and so on.'],
 ask:['What does w₁ seem to capture?','How many numbers per image after encoding?'],
 expected:['Roughly “a 0-like ring versus a 1-like stroke”: the biggest difference between digits.','d, for example 10 instead of 784.'],
 transition:'Rebuild our 3 from a few numbers.',source:'PCA of the 60,000 MNIST training digits (eigenvectors of the 784 × 784 covariance). '+OURS});
S('pca-slider','pca','Rebuild our 3 from its first d PCA numbers',{builds:1,
 body:act(3,widget('pca-slider','Reconstruct the running example from its first d PCA numbers.',{d:10})),
 lines:[`Drag d: 2 numbers give a blur (error ${f(M.pca.sse3['2'],1)}), 10 a rough 3 (${f(M.pca.sse3['10'],1)}), 50 a clear 3 (${f(M.pca.sse3['50'],1)}).`],
 ask:['How many numbers until you would call it a 3?'],
 expected:['Around 10–20 for most people; 50 looks almost exact.'],
 demo:'Notebook 3.3: the reconstruction cell prints the squared error for d = 1, 2, 5, 10, 20, 50, 100.',
 transition:'How much of all digits does d keep?'});
function varChart(){
 const v=M.pca.var_explained,xs=v.map((_,i)=>i+1);
 const marks=[2,10,50,100].map(k=>`<circle cx="0" cy="0" r="0"/>`);
 const C=lineChart({W:700,H:330,xMin:0,xMax:200,yMin:0,yMax:1,ticksX:[0,50,100,150,200],ticksY:[0,.25,.5,.75,1],fmtY:v=>`${Math.round(v*100)}%`,xLabel:'numbers kept d',yLabel:'share of the spread kept',series:[{x:xs,y:v,color:palette.noise}]});
 let s=C.s;for(const k of [2,10,50,100]){const y=v[k-1];s+=C.P.dot(k,y,palette.noise)+label(C.P.x(k)+8,C.P.y(y)+22,`${k}: ${f(100*y,0)}%`,{fill:palette.noise,size:18,weight:600});}
 return s;
}
S('pca-variance','pca','How many numbers do we need?',{builds:2,
 body:act(3,cols(fig(700,330,varChart(),'Share of the spread of all digits kept by the first d PCA directions.'),stack(p(`d = 10 keeps ${f(100*M.pca.at['10'],0)}% of the spread; d = 50 keeps ${f(100*M.pca.at['50'],0)}%.`),at(2,p('784 numbers per digit, but about 50 carry most of the information.'))),'1.4fr 1fr')),
 lines:['The first few directions carry the most spread; the curve flattens fast.','Digits really live near a much lower-dimensional surface. That is why compressing first is a good idea.'],
 ask:['Why does the curve flatten?','What is a good d for generating digits?'],
 expected:['Later directions describe small details and pixel noise.','A trade-off: small d is easier to model, large d keeps detail. We will try several.'],
 transition:'A map of digits in two numbers.'});
S('pca-scatter','pca','Each digit as 2 numbers: PCA’s map of 1,000 test digits',{builds:2,
 body:act(3,widget('scatter','The first two PCA numbers of 1,000 test digits.',{src:'pca',lim:'-6,7.5,-6,6',ticks:'-4,0,4|-4,0,4',title:'z₁, z₂ from PCA',side:'hover any point to see its digit'})),
 lines:['Each point is one test digit, placed at its first two PCA numbers and coloured by its label. Hover to see the digit.','0s and 1s separate; 4, 7 and 9 overlap heavily. Two straight-line numbers cannot pull every digit apart.'],
 ask:['Which digits are easy to tell apart here?','Why do some digits overlap?'],
 expected:['0 and 1: they sit at opposite ends of w₁.','Two linear directions keep only 17% of the spread; 4, 7 and 9 differ in details these directions do not capture.'],
 transition:'Now generate from PCA numbers.',source:'1,000 MNIST test digits, 100 per class (seed 2024). '+OURS});
chap('3.4');
function genDiagram(kind='pca'){
 const W=1100,H=190,dec=kind==='pca'?['decode','x̂ = m + W z',palette.noise]:kind==='ae'?['decoder','a neural network',palette.p]:['decoder','fθ(z), learned',palette.p];
 const prior=kind==='pca'?['z ~ p(z)','fitted to 60,000 codes']:['z ~ N(0, I)',kind==='ae'?'a guess':'chosen in advance'];
 const box=(x,w,t,sub,c)=>`<rect x="${x}" y="50" width="${w}" height="110" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,97,t,{fill:c,size:24,anchor:'middle',weight:600})+label(x+w/2,129,sub,{size:19,anchor:'middle'});
 const arrow=(x1,x2)=>`<line x1="${x1}" y1="105" x2="${x2-8}" y2="105" stroke="${palette.ink}" stroke-width="2.5"/><path d="M${x2},105 l-12,-7 v14 z" fill="${palette.ink}"/>`;
 return fig(W,H,label(150,30,'step 1: sample a code',{fill:palette.ink,size:20,anchor:'middle',weight:600})+box(20,260,prior[0],prior[1],palette.data)+arrow(280,420)+label(540,30,'step 2: turn it into an image',{fill:palette.ink,size:20,anchor:'middle',weight:600})+box(420,240,dec[0],dec[1],dec[2])+arrow(660,800)+`<image href="${digitSrc(kind==='pca'?'gen_d20k10':kind==='ae'?'ae2_prior_samples':'vae2_prior_samples',0)}" x="800" y="40" width="130" height="130" style="image-rendering:pixelated"/><rect x="800" y="40" width="130" height="130" fill="none" stroke="${LINE}"/>`+label(960,110,'a new digit',{size:20}),'Generation in two steps: sample a code, then decode it into an image.');
}
S('pca-gen-recipe','pca','Generate in two steps: sample a few numbers, then rebuild an image',{builds:2,
 body:act(3,genDiagram('pca'),at(2,p('Part 2’s recipe again: first a hidden thing (the code z), then the data. Here step 2 is PCA’s decoder.'))),
 lines:['Fit a distribution to the d PCA numbers of the 60,000 training digits, sample a new code, and decode it with x̂ = m + W z.','The same shape as the mixture: pick a hidden value first, then produce x from it. The code z plays the role of the bump k.'],
 ask:['Where do the 60,000 codes come from?','Which distribution should we fit to them?'],
 expected:['Encode every training digit: z = Wᵀ(x − m).','Start with one Gaussian; Part 2 suggests a mixture.'],
 transition:'One Gaussian over the codes.'});
S('pca-gauss','pca','One Gaussian over d PCA numbers',{builds:2,
 body:act(3,rows(['d = 2',digits('gen_d2k1',{cols:16,size:60,gap:4})],['d = 10',digits('gen_d10k1',{cols:16,size:60,gap:4})],['d = 50',at(2,digits('gen_d50k1',{cols:16,size:60,gap:4}))])),
 lines:['A Gaussian over 2 numbers gives blurry average-looking digits; over 10 numbers, rough digits appear.','Over 50 numbers: more detail, but many samples are mixtures of two digits. One bump for ten kinds of digit.'],
 ask:['Why are the d = 2 samples so blurry?','Why do some d = 50 samples look like two digits on top of each other?'],
 expected:['Two PCA numbers can only describe a blurry digit, whatever their values.','The single Gaussian puts a lot of probability between the clusters of different digits.'],
 transition:'Use a mixture over the codes.',source:OURS});
S('pca-mog','pca','A mixture of K Gaussians over d numbers: try it',{builds:1,
 body:act(3,widget('mog-gen','Sample digits from a mixture of Gaussians over PCA numbers.',{d:20,k:10})),
 lines:['Choose d and K, then press Sample: 16 codes from the mixture, each decoded with PCA. Compare K = 1 and K = 20 at d = 20.'],
 ask:['What happens as K grows? As d grows?'],
 expected:['More bumps: cleaner, more digit-like samples. More numbers: sharper, but the mixture needs many more parameters.'],
 point:'Fitted by maximum likelihood with EM (full covariances), on the PCA numbers of the training digits; held-out scores are on the 10,000 test digits and can be compared only for the same d.',
 demo:'Notebook 3.4: the mixture cell fits K = 10 bumps to the 20 PCA numbers and samples 16 digits.',
 transition:'What limits this recipe?',source:OURS});
S('pca-limits','pca','What limits the PCA recipe?',{builds:2,
 body:act(3,cols(rows(['8 test digits',digits('watch',{cols:8,size:64,gap:6})],['PCA, d = 2',digits('pca_recon_watch_2',{cols:8,size:64,gap:6})],['PCA, d = 16',digits('pca_recon_watch_16',{cols:8,size:64,gap:6})]),
  stack(p('PCA’s decoder is a straight-line map: x̂ = m + W z.'),at(2,p('Bend it: replace both maps with small neural networks. That is an autoencoder.'))),'1.5fr 1fr')),
 lines:[`With 2 numbers, PCA’s rebuilt digits are blurs (test error ${f(M.pca.test_sse['2'],1)} per image); with 16 they are recognisable but soft (${f(M.pca.test_sse['16'],1)}).`,'A flexible encoder and decoder should keep more from the same few numbers. Part 4.'],
 ask:['Why can a straight-line decoder not draw a sharp 3 from 2 numbers?','What would a better decoder do?'],
 expected:['It can only add 2 fixed images to the mean; every digit is built from the same two strokes.','Turn a few numbers into a sharp digit along a curved surface, not a flat plane.'],
 transition:'Part 3 recap.',source:OURS});
S('recap-3','pca','Part 3 in one slide',{builds:2,
 body:act(3,table(['model','numbers','samples'],[['independent pixel Gaussians',comma(M.params.indep),'speckle'],['one full Gaussian',comma(M.params.full),'ghosts'],['Gaussian on 50 PCA numbers',comma(M.gens.d50k1.params),'blended digits'],['20 bumps on 20 PCA numbers',comma(M.gens.d20k20.params),'rough but real digits']]),at(2,p('Compress first, then fit: the code is the hidden variable. Next, learn the compression itself.'))),
 lines:['Raw pixels need huge Gaussians and still fail. Compressing to a few PCA numbers and fitting a mixture there works much better.','The weak link is PCA’s straight-line decoder. Part 4 learns a better one.'],
 ask:['Which step of “compress, then fit” is PCA?','What would improve the samples most?'],
 expected:['The compression: encode to d numbers, decode back.','A decoder that turns few numbers into sharp digits.'],
 transition:'Part 4: autoencoders.'});

// ================================================================= Part 4 · Autoencoders
chap('4.1');
/** image → encoder → code → decoder → image, as PCA, an autoencoder, or a VAE. */
function aeDiagram({kind='ae',stage=9,loss=false,zText,showOut=true}={}){
 const W=1100,H=kind==='vae'?330:280,y0=kind==='vae'?60:40,mid=y0+100;
 const encC=kind==='pca'?palette.noise:palette.p,decC=encC;
 const encT=kind==='pca'?['encode','Wᵀ(x − m)']:kind==='vae'?['encoder','784 → 400 → 4']:['encoder','784 → 400 → 2'];
 const decT=kind==='pca'?['decode','m + W z']:['decoder','2 → 400 → 784'];
 const trap=(x,wide,narrowRight,c,t)=>{const h1=wide?200:80,h2=wide?80:200,x2=x+170;return `<polygon points="${x},${mid-h1/2} ${x2},${mid-h2/2} ${x2},${mid+h2/2} ${x},${mid+h1/2}" fill="${c}14" stroke="${c}" stroke-width="2.5"/>`+label(x+85,mid-6,t[0],{fill:c,size:22,anchor:'middle',weight:600})+label(x+85,mid+22,t[1],{size:17,anchor:'middle'});};
 let s=dimg('example',0,10,mid-60,120)+label(70,mid+88,'x: 784 numbers',{size:18,anchor:'middle'});
 if(stage>=2)s+=trap(150,true,true,encC,encT);
 if(kind==='vae'){
  if(stage>=2)s+=`<rect x="350" y="${mid-88}" width="96" height="54" rx="8" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(398,mid-53,'μ (2)',{fill:palette.q,size:21,anchor:'middle',weight:600})+`<rect x="350" y="${mid+34}" width="96" height="54" rx="8" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(398,mid+69,'log σ² (2)',{fill:palette.q,size:19,anchor:'middle',weight:600});
  if(stage>=3)s+=`<circle cx="520" cy="${mid+130}" r="34" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/>`+label(520,mid+138,'ε',{fill:palette.noise,size:26,anchor:'middle',weight:600})+label(574,mid+150,'ε ~ N(0, I)',{fill:palette.noise,size:17})+
   `<rect x="480" y="${mid-36}" width="130" height="72" rx="10" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/>`+label(545,mid+8,'z = μ + σ ⊙ ε',{fill:palette.noise,size:19,anchor:'middle',weight:600})+
   `<line x1="446" y1="${mid-60}" x2="480" y2="${mid-20}" stroke="${palette.ink}" stroke-width="2"/><line x1="446" y1="${mid+60}" x2="480" y2="${mid+20}" stroke="${palette.ink}" stroke-width="2"/><line x1="520" y1="${mid+96}" x2="530" y2="${mid+36}" stroke="${palette.noise}" stroke-width="2"/>`;
  if(stage>=4){s+=trap(640,false,false,decC,decT);if(showOut)s+=dimg(stage>=5?'vae2_mean_recon3':'example',0,840,mid-60,120)+label(900,mid+88,'x̂: the mean image',{size:18,anchor:'middle'});}
 }else{
  if(stage>=3)s+=`<rect x="334" y="${mid-36}" width="142" height="72" rx="10" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(405,mid+8,zText||(kind==='pca'?'z: d numbers':'z: 2 numbers'),{fill:palette.q,size:19,anchor:'middle',weight:600});
  if(stage>=4)s+=trap(490,false,false,decC,decT);
  if(stage>=4&&showOut)s+=dimg(kind==='pca'?'pca_recon3':'ae2_recon3',kind==='pca'?1:0,690,mid-60,120)+label(750,mid+88,'x̂: rebuilt',{size:18,anchor:'middle'});
  if(loss&&stage>=4)s+=`<rect x="850" y="${mid-40}" width="240" height="80" rx="10" fill="${palette.paper}" stroke="${palette.loss}" stroke-width="2.5"/>`+label(970,mid-6,'loss',{fill:palette.loss,size:22,anchor:'middle',weight:600})+label(970,mid+22,'Σⱼ (xⱼ − x̂ⱼ)²',{fill:palette.loss,size:19,anchor:'middle'});
 }
 return fig(W,H,s,'An image goes through an encoder to a short code and back through a decoder.');
}
S('pca-as-ae','ae','PCA is an encoder and a decoder',{builds:2,
 body:act(4,aeDiagram({kind:'pca',zText:'z: 2 numbers'}),at(2,p('Both maps are straight lines (matrix products), chosen to make the rebuilt image close to the original.'))),
 lines:['PCA already has the shape we want: squeeze 784 numbers into a short code, then expand back.',`PCA’s two maps are linear. For our 3 with 2 numbers the squared error is ${f(M.pca.sse3['2'],1)}.`],
 ask:['Which part is the encoder, which the decoder?','What does PCA minimise?'],
 expected:['Encoder Wᵀ(x − m); decoder m + W z.','The squared reconstruction error, summed over pixels and averaged over images, among all linear maps.'],
 transition:'Replace the straight lines with networks.'});
S('ae-arch','ae','An autoencoder: replace both maps with small neural networks',{builds:3,
 body:act(4,`<div class="state-slot"><div data-only="1">${aeDiagram({kind:'ae',stage:2,showOut:false})}</div><div data-only="2">${aeDiagram({kind:'ae',stage:3,showOut:false})}</div><div data-only="3">${aeDiagram({kind:'ae',stage:4})}</div></div>`),
 lines:['The encoder is a small network: 784 inputs, 400 hidden units with ReLU, 2 outputs.','Its output is the code z, the bottleneck: like the neck of a bottle, everything must squeeze through 2 numbers.','The decoder mirrors it: 2 → 400 → 784, then a sigmoid. Train both so that x̂ ≈ x.'],
 ask:['Why squeeze through so few numbers?','What would happen with a 784-number code?','What is the target output?'],
 expected:['To force it to keep only what matters about the digit.','It could copy the input and learn nothing useful.','The input itself: the network is trained to reconstruct x.'],
 point:'Hinton and Salakhutdinov (Science, 2006) made deep autoencoders work; the idea goes back to the 1980s. One hidden layer of 400 units follows Kingma and Welling (2014).',
 transition:'The same in PyTorch.'});
S('ae-code','ae','Two small networks in PyTorch',{builds:2,
 body:act(4,cols(`<div class="code-sm">${code(`class AutoEncoder(nn.Module):
    def __init__(self, z=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, z))
        self.decoder = nn.Sequential(
            nn.Linear(z, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)     # 784 -> z numbers
        return self.decoder(z)  # z -> 784 numbers`)}</div>`,
  stack(at(2,p('Why a sigmoid at the end? Are we classifying?')),at(2,small('No: it keeps every output pixel between 0 and 1, like the data.'))),'1.6fr 1fr')),
 lines:['Two nn.Sequential blocks. The forward pass encodes, then decodes.','The final sigmoid squashes each of the 784 outputs into (0, 1), the range of our pixels.'],
 ask:['How many weights does the encoder have?','Why no activation after the encoder’s last layer?'],
 expected:['784 × 400 + 400 + 400 × 2 + 2 ≈ 315,000.','The code can be any real number; a ReLU would forbid negative codes.'],
 transition:'What does it minimise?',nb:'sec-4-1'});
S('ae-loss','ae','The loss: squared error, summed over pixels, averaged over images',{builds:3,
 body:act(4,stack(eq(R`L(\phi,\theta)=\frac1n\sum_{i=1}^{n}\ \sum_{j=1}^{784}\big(x_{ij}-\hat x_{ij}\big)^2,\qquad \hat x_i=f_\theta\big(g_\phi(x_i)\big)`),
  at(2,p(`Our 3, through a trained 2-number autoencoder: squared error <b>${f(A.ae2_sse3,1)}</b>.`)),at(3,p(`PCA on the same 3 with the same 2 numbers: <b>${f(M.pca.sse3['2'],1)}</b>. Same budget, bent maps, less error.`)))),
 lines:['g_φ is the encoder with weights φ; f_θ the decoder with weights θ. Both are trained together on one loss.','For our 3 the trained autoencoder leaves a squared error of 26.3, summed over the 784 pixels.','PCA, the best straight-line version, leaves 41.3. The bend buys a lot.'],
 ask:['Which weights does the loss depend on?','Why sum over pixels but average over images?','Why must the autoencoder do at least as well as PCA, in principle?'],
 expected:['Both φ (encoder) and θ (decoder).','The sum is the error of one image; averaging over images makes the loss independent of the batch size.','With no ReLUs and no sigmoid it could represent PCA exactly; the nonlinearities only add options.'],
 transition:'Before training: what comes out?'});
chap('4.2');
S('ae-untrained','ae','Before training: always look at what the network outputs',{builds:2,
 body:act(4,cols(rows(['8 test digits',digits('watch',{cols:8,size:64,gap:6})],['untrained output',digits('ae2_train_frames',{idx:[0,1,2,3,4,5,6,7],cols:8,size:64,gap:6})]),
  stack(p(`Every output pixel is between ${f(A.init_pixel_range[0],2)} and ${f(A.init_pixel_range[1],2)}: flat grey.`),at(2,p(`Test squared error: <b>${f(A.init_test_sse,0)}</b> per image.`))),'1.45fr 1fr')),
 lines:['With random weights the sigmoid outputs about 0.5 everywhere: grey squares, whatever the input.','The starting loss is about 185 per image. This is the number training must bring down; a sanity check before any training.'],
 ask:['Why is every output close to 0.5?','Why check the untrained model at all?'],
 expected:['Small random weights make the pre-sigmoid values near 0, and sigmoid(0) = 0.5.','To catch bugs early and know the starting loss: if training ends near 185, it learned nothing.'],
 transition:'Train it: one step at a time.',source:OURS});
S('ae-step','ae','One training step: forward, loss, backward, update',{builds:2,
 body:act(4,cols(`<div class="code-sm">${code(`model = AutoEncoder(z=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(30):
    for x in loader:        # 128 images at a time
        x_hat = model(x)
        loss = ((x_hat - x)**2).sum(1).mean()
        opt.zero_grad()
        loss.backward()     # encoder AND decoder
        opt.step()`)}</div>`,
  stack(p(`${comma(A.steps_per_epoch)} steps per pass over the 60,000 digits; 30 passes.`),at(2,p('One loss, one backward pass: the gradient flows from the error back through the decoder, the code, and the encoder.'))),'1.5fr 1fr')),
 lines:['Batches of 128 digits, the explicit loss from the last slide, Adam.','loss.backward() sends gradients through the decoder, into the code, and on into the encoder: both are trained together.'],
 ask:['How many updates in total?','Does the encoder get a gradient even though the loss compares only images?'],
 expected:[`${comma(30*A.steps_per_epoch)}.`,'Yes: the reconstruction depends on the code, which depends on the encoder weights.'],
 transition:'Watch the reconstructions form.',nb:'sec-4-2'});
S('ae-train','ae','Press play: reconstructions sharpen as it trains',{builds:2,
 body:act(4,widget('ae-train','Reconstructions during training of the 2-number autoencoder.')),
 lines:['Frame 0: grey. Within the first 100 steps, blurry average digits appear.',`By the end, each digit is recognisable from 2 numbers. Training took ${f(A.seconds_ae2,0)} s on a laptop.`],
 ask:['What does the network learn first?','Which digits stay hardest to rebuild from 2 numbers?'],
 expected:['The average digit: a blurry blob that lowers the error everywhere.','Ones that resemble other digits, like 4 and 9, or unusual shapes.'],
 transition:'Did it just memorise the training digits?',source:OURS});
S('ae-test','ae','Did it memorise? Compare training and test error',{builds:2,
 body:act(4,cols(table(['2-number code','training error','test error'],[['autoencoder',f(SW['2'].train_sse,1),f(SW['2'].test_sse,1)],['PCA','',f(M.pca.test_sse['2'],1)]]),
  stack(p('Test digits were never used for training.'),at(2,p('Training and test errors are close: it learned digits in general, not these particular 60,000.'))),'1fr 1fr')),
 lines:['Average squared error per image on the training digits, and on the 10,000 test digits.','The gap is small (30.5 vs 31.1). And the autoencoder beats PCA by a wide margin on the same 2 numbers.'],
 ask:['What would memorising look like in these numbers?','Why is the test error a little higher?'],
 expected:['Very low training error and much higher test error.','The weights were tuned to the training digits; new digits fit a little less well.'],
 transition:'What if we take out the bends?',source:OURS});
S('linear-ae','ae','Take out the ReLUs and the sigmoid: a linear autoencoder rediscovers PCA',{builds:2,
 body:act(4,cols(rows(['8 test digits',digits('watch',{cols:8,size:60,gap:5})],['PCA, d = 2',digits('pca_recon_watch_2',{cols:8,size:60,gap:5})],['linear AE, 2',digits('lin2_recon_watch',{cols:8,size:60,gap:5})]),
  stack(p(`Test error: linear autoencoder ${f(A.lin2_test_sse,1)}, PCA ${f(M.pca.test_sse['2'],1)}.`),at(2,p(`Its decoder spans almost the same plane as PCA’s two directions (cosines of the angles between them: ${f(A.lin2_cos_angles[0],3)} and ${f(A.lin2_cos_angles[1],2)}).`))),'1.45fr 1fr')),
 lines:['Train the same way, but with straight-line encoder and decoder. The reconstructions match PCA’s.','Gradient descent found PCA’s plane on its own. The autoencoder is PCA with bendable maps.'],
 ask:['Why does a linear autoencoder end up like PCA?','So what do the ReLUs buy?'],
 expected:['PCA is the best linear encoder–decoder pair under squared error; training finds it (up to a rotation within the plane).','Curved maps: a better reconstruction from the same few numbers.'],
 transition:'How many numbers should the code have?',source:OURS});
chap('4.3');
const ZS=['2','4','8','16','32','64','128'],ZCOL=['#C4B5FD','#A78BFA','#8B5CF6','#7C3AED','#6D28D9','#5B21B6','#3B0F86'];
function bottleneckChart(){
 const ep=d.ae.ae2_curve.epoch,L10=Math.log10;
 return lineChart({W:660,H:340,xMin:1,xMax:30,yMin:L10(1),yMax:L10(60),ticksX:[1,10,20,30],ticksY:[1,2,5,10,20,50].map(L10),fmtY:v=>String(Math.round(10**v)),xLabel:'epoch',yLabel:'test squared error per image (log scale)',
  series:ZS.map((z,i)=>({x:ep,y:SW[z].curve.map(L10),color:ZCOL[i],label:`z = ${z}`}))}).s;
}
S('bottleneck','ae','Wider codes reconstruct better',{builds:2,
 body:act(4,cols(fig(780,340,bottleneckChart(),'Test error during training for codes of 2 to 128 numbers.'),stack(p('Seven autoencoders, codes of 2 to 128 numbers, trained the same way.'),at(2,p(`2 numbers: ${f(SW['2'].test_sse,1)}. 128 numbers: ${f(SW['128'].test_sse,1)}.`))),'1.5fr 1fr')),
 lines:['The same autoencoder with codes of 2, 4, 8, … 128 numbers. The curves stack in order.','More numbers in the code, less error: with 128 numbers each digit is rebuilt almost exactly.'],
 ask:['Why do the curves stack this way?','Is the 128-number autoencoder better at everything?'],
 expected:['A wider bottleneck keeps more detail of each individual digit.','At reconstruction, yes. Hold that question for Part 5.'],
 transition:'What the errors look like.',source:OURS});
S('bottleneck-recon','ae','The same 8 test digits through codes of 2, 8, 32 and 128 numbers',{builds:2,
 body:act(4,rows(['originals',digits('watch',{cols:8,size:60,gap:6})],['z = 2',digits('ae2_recon_watch',{cols:8,size:60,gap:6})],['z = 8',digits('ae8_recon_watch',{cols:8,size:60,gap:6})],['z = 32',at(2,digits('ae32_recon_watch',{cols:8,size:60,gap:6}))],['z = 128',at(2,digits('ae128_recon_watch',{cols:8,size:60,gap:6}))])),
 lines:['2 numbers keep the digit’s identity and rough shape; 8 numbers add the style.','32 and 128 numbers reproduce strokes almost exactly.'],
 ask:['What does the 2-number code lose?','At what size are you satisfied?'],
 expected:['Stroke thickness, slant and small details.','Around 16–32 for most people.'],
 transition:'Compare with PCA at every size.',source:OURS});
function aeVsPcaChart(){
 const xs=ZS.map(z=>Math.log2(Number(z)));
 return lineChart({W:660,H:330,xMin:1,xMax:7,yMin:0,yMax:46,ticksX:[1,2,3,4,5,6,7],fmtX:v=>String(2**v),ticksY:[0,10,20,30,40],xLabel:'numbers in the code',yLabel:'test squared error per image',
  series:[{x:xs,y:ZS.map(z=>M.pca.test_sse[z]),color:palette.noise,label:'PCA',dots:true,labelAt:1,dy:-12},{x:xs,y:ZS.map(z=>SW[z].test_sse),color:palette.q,label:'autoencoder',dots:true,labelAt:1,dy:22}]}).s;
}
S('ae-vs-pca','ae','Autoencoder against PCA, at every code size',{builds:2,
 body:act(4,cols(fig(780,330,aeVsPcaChart(),'Test error of PCA and the autoencoder against the code size.'),stack(p(`8 numbers: PCA ${f(M.pca.test_sse['8'],1)}, autoencoder ${f(SW['8'].test_sse,1)}.`),at(2,p('PCA needs about four times as many numbers for the same error.'))),'1.5fr 1fr')),
 lines:['At every code size the autoencoder rebuilds test digits with less error than PCA.','The biggest gain is for short codes, where bending matters most.'],
 ask:['Where is the gap largest?','When would PCA still be the better choice?'],
 expected:['Short codes: 2 to 16 numbers.','When you need speed, no training, or a guaranteed, interpretable linear map.'],
 transition:'What are the codes themselves good for?',source:OURS});
chap('4.4');
S('ae-scatter','ae','The 2-number codes of 1,000 test digits',{builds:2,
 body:act(4,widget('scatter','The 2-number autoencoder codes of 1,000 test digits.',{src:'ae2',lim:'-6,24,-9,14',ticks:'0,10,20|-5,0,5,10',title:'z₁, z₂ from the autoencoder',side:'hover any point to see its digit'})),
 lines:['The same 1,000 test digits, now placed at their autoencoder codes. Hover to see digits.','Compared with PCA’s map, the digits form tighter, better-separated groups. No labels were used in training.'],
 ask:['Which digits are still mixed?','The codes go from about −5 to 22. Is that a problem?'],
 expected:['4 and 9; some 3, 5 and 8.','Not for reconstruction. For generation, we will see (Part 5).'],
 transition:'Use codes to search.',source:OURS});
S('search','ae','Search by code: the nearest digits to a query',{builds:2,
 body:act(4,cols(`<div class="dflex" style="display:flex;gap:24px;align-items:end"><figure class="dg-cell">${digit('search_query',0,130)}<figcaption>query</figcaption></figure>${at(2,digits('search_code',{cols:5,size:100,gap:10,labels:['1st','2nd','3rd','4th','5th']}))}</div>`,
  stack(p('Encode the query, encode all 60,000 training digits, return the 5 nearest codes.'),at(2,small('Music apps and “vector databases” do the same with codes of songs, images and text.'))),'1.45fr 1fr')),
 lines:['Codes from a 16-number autoencoder. To search, compare 16 numbers instead of 784 pixels.','The nearest codes are other 1s with the same slant: the code keeps what matters about the digit.'],
 ask:['Why search with codes instead of pixels?','What would make two codes close?'],
 expected:['Fewer numbers, and distances that reflect shape rather than exact pixel positions.','Digits the decoder rebuilds similarly: same identity and similar style.'],
 transition:'Codes can also clean images.',source:OURS});
S('denoise','ae','Denoising: damage the input, ask for the clean image',{builds:3,
 body:act(4,cols(rows(['clean',digits('watch',{cols:8,size:58,gap:5})],['half the pixels removed',digits('damaged_watch',{cols:8,size:58,gap:5})],['plain autoencoder',at(2,digits('ae16_on_damaged',{cols:8,size:58,gap:5}))],['denoising autoencoder',at(2,digits('dae16_clean',{cols:8,size:58,gap:5}))]),
  stack(at(2,p(`Error against the clean digits: plain ${f(A.ae16_test_sse_damaged,1)}, denoising ${f(A.dae16_test_sse_damaged,1)}.`)),at(3,eq(R`L=\frac1n\sum_i\big\lVert x_i-f_\theta(g_\phi(\tilde x_i))\big\rVert^2`)),at(3,small('Input: the damaged x̃. Target: the clean x.'))),'1.5fr 1fr')),
 lines:['Remove a random half of the pixels of each test digit.','A plain autoencoder rebuilds the damage faithfully. A denoising autoencoder, trained on damaged inputs, fills in the strokes.','Its loss compares the output with the clean image, so it must learn what digits look like, not copy its input.'],
 ask:['Why does the plain autoencoder reproduce the holes?','What does the denoising version have to learn?','What does this have in common with dropout?'],
 expected:['It was trained to copy its input, holes and all.','What digits look like, so it can guess the missing pixels.','Both randomly remove inputs during training (denoising autoencoders came first, in 2008).'],
 point:'Both have 16-number codes. Vincent et al., “Extracting and composing robust features with denoising autoencoders”, ICML 2008.',
 transition:'Part 4 recap.',source:OURS});
S('recap-4','ae','Part 4 in one slide',{builds:2,
 body:act(4,table(['','PCA','autoencoder'],[['encoder, decoder','straight lines','small networks'],['trained by','an eigenvector formula','gradient descent on squared error'],['2-number test error',f(M.pca.test_sse['2'],1),f(SW['2'].test_sse,1)],['codes','overlapping','tighter groups']]),at(2,p('PCA gave us generation by “sample a code, decode it”. Can the autoencoder’s decoder do the same?'))),
 lines:['An autoencoder is a learned, bent PCA: much better reconstructions from the same few numbers.','Part 5: keep only the decoder, feed it random codes, and see what it draws.'],
 ask:['What did the autoencoder improve over PCA?','What do we need for generation that reconstruction does not give?'],
 expected:['Reconstruction from short codes, and better-separated codes.','A distribution over codes to sample from.'],
 transition:'Part 5: can an autoencoder generate?'});

// ================================================================= Part 5 · Can an autoencoder generate?
chap('5.1');
S('gen-question','gen','Keep the decoder, throw away the encoder, feed it random numbers',{builds:2,
 body:act(5,genDiagram('ae'),at(2,p('Which random numbers? The simplest choice: z ~ N(0, I).'))),
 lines:['The decoder turns 2 numbers into a digit. To generate, skip the encoder and invent the 2 numbers.','We need a distribution to draw codes from. Start with the simplest: the standard normal, N(0, I).'],
 ask:['What should a random code look like, to give a good digit?','Why might N(0, I) be a poor choice?'],
 expected:['Like the codes real digits get.','Nothing in training told the codes to look like N(0, I).'],
 transition:'What N(0, I) means, in two dimensions.'});
function stdNormalFig(){
 const pts=G.prior_points,P=panel({W:440,H:420,L:46,R:12,T:16,B:46,xMin:-4,xMax:4,yMin:-4,yMax:4,ticksX:[-4,-2,0,2,4],ticksY:[-4,-2,0,2,4],fs:17,xAxisAtBottom:true});
 let s=`<rect x="46" y="16" width="382" height="358" fill="#fff" stroke="${LINE}"/>`+P.axes;
 s+=pts.map(([a,b])=>`<circle cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="3.4" fill="${palette.loss}" fill-opacity=".6"/>`).join('');
 const sx=P.x(1)-P.x(0);for(const k of [1,2])s+=`<circle cx="${P.x(0)}" cy="${P.y(0)}" r="${k*sx}" fill="none" stroke="${palette.ink}" stroke-dasharray="6 5" stroke-width="2"/>`+label(P.x(0)+k*sx*.72+4,P.y(0)-k*sx*.72-4,`radius ${k}`,{fill:palette.ink,size:16});
 return s;
}
const inside=r=>G.prior_points.filter(([a,b])=>a*a+b*b<=r*r).length/G.prior_points.length;
S('std-normal','gen','N(0, I) in two dimensions: two independent standard bells',{builds:3,
 body:act(5,cols(fig(440,420,stdNormalFig(),'500 draws from the 2-D standard normal.'),stack(eq(R`z=(z_1,z_2)`),eq(R`z_1\sim\N(0,1),\quad z_2\sim\N(0,1),\ \text{independent}`),at(2,p(`Of these 500 draws, ${f(100*inside(1),0)}% land within distance 1 of the origin and ${f(100*inside(2),0)}% within 2 (in theory 39% and 86%).`)),at(3,p('In code: <code>torch.randn(500, 2)</code>. Draws pile up near (0, 0) and rarely go beyond 3.'))),'1fr 1.1fr')),
 lines:['N(0, I): each of the two numbers is a standard bell, drawn independently. “I” is the identity covariance.',`Most draws land close to the origin: ${f(100*inside(1),0)}% within distance 1 and ${f(100*inside(2),0)}% within 2.`,'That is all torch.randn does. Its draws live in a small disc around (0, 0).'],
 ask:['What does the I in N(0, I) say?','Where will almost all random codes land?','How far from the origin is a draw of N(0, I) in 16 dimensions, typically?'],
 expected:['The covariance is the identity: unit variance, no correlation between the numbers.','Within about 2 or 3 of the origin.','About 4 (the square root of 16): in more dimensions the draws sit on a shell.'],
 transition:'Feed such draws to three decoders.',source:OURS});
S('ae-random','gen','Random codes through three autoencoders',{builds:3,
 body:act(5,rows(['z = 2',digits('ae2_prior_samples',{cols:16,size:58,gap:4})],['z = 8',at(2,digits('ae8_prior_samples',{cols:16,size:58,gap:4}))],['z = 128',at(3,digits('ae128_prior_samples',{cols:16,size:58,gap:4}))])),
 lines:['2-number autoencoder: some digits, mostly the same few blurry shapes.','8 numbers: strokes that belong to no digit.','128 numbers, the best reconstructor of all: static. Better reconstruction, worse generation.'],
 ask:['Which of the three looks most like digits?','The 128-number model reconstructed best. Why does it generate worst?','What would you check to find out?'],
 expected:['The 2-number one.','Its codes for real digits are nowhere near where N(0, I) puts its draws.','Plot where real codes are, next to the random codes.'],
 transition:'Plot where the codes are.',source:OURS});
chap('5.2');
S('overlay','gen','Put the random codes on top of the real ones',{builds:2,
 body:act(5,cols(`<div class="state-slot"><div data-only="1">${fig(600,440,scatterSVG('ae2',{W:600,H:440,lim:[-6,24,-9,14],ticks:[[0,10,20],[-5,0,5,10]]}).s,'Autoencoder codes of 1,000 test digits.')}</div><div data-only="2">${fig(600,440,scatterSVG('ae2',{W:600,H:440,lim:[-6,24,-9,14],ticks:[[0,10,20],[-5,0,5,10]],prior:500}).s,'The same codes with 500 draws from N(0, I) in red.')}</div></div>`,
  stack(p(`Real codes spread from about ${f(A.ae2_box[0][0],0)} to ${f(A.ae2_box[1][0],0)} and ${f(A.ae2_box[0][1],0)} to ${f(A.ae2_box[1][1],0)}.`),at(2,p('The 500 red draws from N(0, I) all land in one small corner.')),at(2,small('Nothing in the loss asked the codes to sit near N(0, I). Random codes miss most digits.'))),'1.3fr 1fr')),
 lines:['The 2-number codes of 1,000 test digits, coloured by digit.','Add 500 draws from N(0, I) in red. They cover one corner: a few digit types, and nothing of the rest.'],
 ask:['Which digits would random codes ever produce?','For the 128-number model, where are the red points?'],
 expected:['Only those whose codes sit near (0, 0).','Even further from the real codes, in a space that is mostly empty.'],
 point:'This is the experiment improvised in the PML 2023 recording of Lecture 25, around 29–31 minutes.',
 transition:'Try it yourself.',source:OURS});
S('explore-ae','gen','Your turn: three ways to choose a code',{builds:1,
 body:act(5,widget('explorer','Choose a code for the autoencoder and see its decoder output.',{net:'ae2',lim:'-6,24,-9,14'})),
 lines:['(1) Draw a random code from N(0, I). (2) Encode a real test digit and decode its code. (3) Click anywhere on the plane. The decoder runs live in your browser.'],
 ask:['Which of the three ways gives the best digits? Click in an empty region: what do you get?'],
 expected:['Encoding a real digit; random codes give a few digit types; empty regions give smudges or strange shapes.'],
 point:'The decoder (2 → 400 → 784) runs in JavaScript from data/nets.json, weights stored in half precision.',
 transition:'Decode a whole grid of codes at once.',source:OURS});
S('ae-grid','gen','Decode a whole grid of codes',{builds:2,
 body:act(5,cols(fig(560,400,scatterSVG('ae2',{W:560,H:400,lim:[-12,30,-13,18],ticks:[[-10,0,10,20,30],[-10,0,10]],box:G.ae2_grid_box,r:3,opacity:.45}).s,'Codes with the decoded grid’s area marked.'),
  digits('ae2_grid',{cols:12,size:31,gap:1}),'1.15fr 1fr')),
 lines:['Decode a 12 × 12 grid of codes covering the dashed box: the codes on the left, the decoder’s output at each grid point on the right.','Over the real codes the grid shows digits blending into each other. Far from them the decoder only extrapolates: thick 1s and 0s, and broken strokes in the bottom-left corner.'],
 ask:['Which parts of the grid are useful for generation?','What does the decoder do far from any real code?'],
 expected:['Those over the clusters of real codes.','It extrapolates from the nearest digits: exaggerated or broken shapes. It was never trained on codes there, and N(0, I) would not even reach most of the box.'],
 transition:'Wider codes make it worse.',source:OURS});
S('spread','gen','Wider codes spread out',{builds:2,
 body:act(5,cols(table(['code size','two 0s apart','a 0 and a 1 apart','ratio'],ZS.filter(z=>['2','8','32','128'].includes(z)).map(z=>[z,f(SW[z].dist.same0,1),f(SW[z].dist.zero_one,1),f(SW[z].dist.ratio,2)])),
  stack(p('Average distance between codes of two 0s, and between a 0 and a 1.'),at(2,p(`With 128 numbers two 0s are almost as far apart as a 0 and a 1 (ratio ${f(SW['128'].dist.ratio,2)}).`))),'1.25fr 1fr')),
 lines:['As the code grows, codes of the same digit drift apart.','In the widest codes, similarity is nearly gone: every digit gets its own private spot. Great for reconstruction, useless for sampling.'],
 ask:['Why would a wider code let same-digit codes drift apart?','What would we want instead?'],
 expected:['The loss only cares about reconstruction; with room to spare, nothing keeps similar digits together.','Codes of similar digits close together, all within a region we know how to sample.'],
 point:'Distances averaged over the 100 0s and 100 1s of the display set. PML 2023 Lecture 25 shows the same trend for one pair of 0s.',
 transition:'Walk between two digits.',source:OURS});
S('interp','gen','Walk in a straight line from a 1 to a 0',{builds:2,
 body:act(5,rows(['ends',digits('interp_ends',{cols:2,size:62,gap:6})],['z = 2',digits('ae2_interp',{cols:9,size:62,gap:6})],['z = 16',at(2,digits('ae16_interp',{cols:9,size:62,gap:6}))])),
 lines:['Encode a 1 and a 0, take 9 evenly spaced codes on the line between them, decode each. With 2 numbers the path crosses the clusters of 8, 3 and 5 on its way.','With 16 numbers the endpoints are sharp, but the middle passes through blobs that are no digit: the straight line leaves the region where codes live.'],
 ask:['Why does the 2-number path pass through other digits?','Why might the 16-number path cross empty space?'],
 expected:['In 2 numbers the clusters are packed side by side; a straight line from 1 to 0 has to cross the 8, 3 and 5 clusters.','The straight line between two codes need not stay where real codes are; in 16 dimensions most of the space between them is empty.'],
 transition:'A fix we already know.',source:OURS});
chap('5.3');
S('posthoc','gen','A fix we already know: fit a mixture to where the codes are',{builds:3,
 body:act(5,cols(fig(520,380,scatterSVG('ae2',{W:520,H:380,lim:[-6,24,-9,14],ticks:[[0,10,20],[-5,0,5,10]],mog:G.ae2_mog,opacity:.35,r:3}).s,'Autoencoder codes with ten fitted Gaussian ellipses.'),
  stack(mini('Part 3’s recipe: fit 10 Gaussians to the codes of the 60,000 training digits, sample a code, decode.'),at(2,stack(small('2-number autoencoder + mixture'),digits('ae2_mog_samples',{cols:8,size:52,gap:4}))),at(3,stack(small('16-number autoencoder + mixture'),digits('ae16_mog_samples',{cols:8,size:52,gap:4})))),'1fr 1.15fr')),
 lines:['Fit a mixture of 10 Gaussians to the training codes: the ellipses follow the clusters.','Sample codes from it, decode: real-looking digits from the 2-number autoencoder.','The same with 16-number codes: sharper digits. A density fitted after training works.'],
 ask:['Do the ellipses cover the clusters?','Why do these samples look better than N(0, I) codes?','What is unsatisfying about this two-stage approach?'],
 expected:['Roughly: about one ellipse per digit group, some shared.','The codes are drawn where real codes are.','Two separate fits; the codes were never shaped for easy sampling; and there is no single model that scores p(x).'],
 transition:'What we would like instead.',source:OURS});
S('wishes','gen','What we would like instead',{builds:3,
 body:act(5,`<div class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">${card('1 · Codes we can sample','Codes that fill a simple, known shape, such as N(0, I), so that torch.randn gives good codes.',palette.data)}${at(2,card('2 · Smooth codes','Nearby codes decode to similar images, so the space between digits holds plausible digits too.',palette.p))}${at(3,card('3 · One probability model','One model, trained by maximum likelihood as in Part 2, that can both score and sample.',palette.q))}</div>`),
 lines:['First wish: decide the code distribution in advance, N(0, I), and make the codes fit it.','Second: each image should own a small region of codes, not a single point, so the space between is filled.','Third: write it as one probability model and train by maximum likelihood. The variational autoencoder does all three.'],
 ask:['How could we force codes to fill N(0, I)?','How could each image own a region instead of a point?','What is the likelihood of an image, if a code is drawn first?'],
 expected:['Penalise codes that stray from it, for example with a KL to N(0, I).','Let the encoder output a small cloud of codes: a mean and a spread.','The sum, or integral, over all codes of p(code) times p(image | code). Part 6.'],
 transition:'Part 6: the variational autoencoder.'});

// ================================================================= Part 6 · The VAE
chap('6.1');
S('vae-story','vae','Write the recipe first: pick a code, then draw an image',{builds:3,
 body:act(6,stack(eq(R`z\sim\N(0,I)\qquad\text{2 numbers, decided in advance}`),eq(R`x\mid z\sim\N\big(f_\theta(z),\ \sigma^2 I\big)\qquad\text{784 numbers, centred on the decoder's output}`),
  at(2,genDiagram('vae')))),
 lines:['Step 1: draw a 2-number code from N(0, I). Step 2: the decoder f_θ turns it into a mean image, and each pixel is that mean plus a little Gaussian noise of size σ.','To generate, run the two steps. No encoder needed: the prior N(0, I) is fixed in advance.','It is Part 2’s recipe with a neural network in step 2.'],
 ask:['What are the parameters of this model?','What is the role of σ?','What is new compared with PCA + Gaussian?'],
 expected:['The decoder weights θ (σ is fixed).','How far an image may stray from the decoder’s mean image; it sets how much reconstruction errors cost.','The decoder bends, and the code distribution is chosen in advance, not fitted afterwards.'],
 point:'When showing samples we display the mean image f_θ(z), not the noisy x: the noise is independent per pixel and would only add speckle.',
 transition:'Why σ = 0.2, and what the likelihood of one image is.'});
S('sigma','vae','The likelihood of an image, given its code, is a squared error',{builds:3,
 body:act(6,stack(eq(R`\log p_\theta(x\mid z)=\sum_{j=1}^{784}\log\N\big(x_j;\,f_\theta(z)_j,\sigma^2\big)`),at(2,eq(R`=-\frac{1}{2\sigma^2}\,\big\lVert x-f_\theta(z)\big\rVert^2\;-\;784\log\big(\sigma\sqrt{2\pi}\big)`)),
  at(3,p(`We use σ = ${f(d.meta.sigma,1)}: about the typical pixel error of a 2-number code, √(31 / 784) ≈ 0.2. The constant term, −784 log(σ√2π), is +${f(-V.loglik_const,1)}.`)))),
 lines:['Each pixel is an independent Gaussian around the decoder’s output, so the log-likelihood is a sum over 784 pixels.','Expand the Gaussian: minus the squared error over 2σ², plus a constant. The autoencoder’s loss was a likelihood all along.','σ = 0.2 matches the size of a typical pixel error; it is a modelling choice, and Section 6.7 shows what changing its weight does.'],
 ask:['What happens to the first term if σ is small?','Where have you seen squared error = Gaussian likelihood before?','Is σ learned here?'],
 expected:['Errors cost more: the model is pushed to reconstruct precisely.','Linear regression: least squares is maximum likelihood with Gaussian noise.','No, fixed at 0.2. It can be learned; we keep it fixed for clarity.'],
 transition:'Train by maximum likelihood, like Part 2.'});
S('vae-mle','vae','Train like Part 2: maximise log p(x) for every image',{builds:2,
 body:act(6,stack(eq(R`p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz`),at(2,cols(stack(small('Part 2: a sum over two kinds'),eq(R`p(x)=\sum_{k=1}^{2}\pi_k\N(x;\mu_k,\sigma_k^2)`)),stack(small('Now: an integral over every possible code'),eq(R`p_\theta(x)=\int \N(x;f_\theta(z),\sigma^2 I)\,\N(z;0,I)\,dz`)),'1fr 1fr')))),
 lines:['The likelihood of an image adds up all the ways it could have been made: every code z, weighted by its prior probability.','For the geyser the sum had two terms. Here it is an integral over all codes, through a neural network. No formula.'],
 ask:['Why do we need p(x) at all?','Why can we not compute this integral exactly?'],
 expected:['To train θ by maximum likelihood, and to score images.','The decoder is a neural network; the integral has no closed form, and for 16 numbers a grid is hopeless.'],
 transition:'Look at one image: which codes could have made it?'});
chap('6.2');
const wideMark=(()=>{const W_=posteriorSVG({S:400,prior:true}),P_=W_.P,cx=P_.x(V.grid.post_mean[0]),cy=P_.y(V.grid.post_mean[1]);return W_.s+`<circle cx="${cx}" cy="${cy}" r="14" fill="none" stroke="${palette.loss}" stroke-width="2.5"/>`+label(cx,cy+40,'the posterior: a tiny patch',{fill:palette.loss,size:17,weight:600,anchor:'middle'});})();
S('posterior3','vae','For our 3: which codes could have made it?',{builds:3,
 body:act(6,cols(`<div class="state-slot"><div data-only="1">${fig(400,400,wideMark,'The posterior over codes for our 3, within the prior.')}</div><div data-only="2-3">${fig(810,400,wideMark+posteriorSVG({ox:410,S:400,view:'zoom',title:'zoomed in about 27×'}).s,'The posterior, wide and zoomed.')}</div></div>`,
  stack(at(3,eq(R`p(z\mid x)=\frac{p(x\mid z)\,p(z)}{p(x)}`))),'1.6fr .7fr')),
 lines:['The trained VAE’s decoder, our 3, and every code on a fine grid: shade = how well that code explains the 3. Dashed circles: the prior N(0, I).',`Zoomed in: the codes that explain this 3 form a tiny patch, about ${f(V.grid.post_sd[0],3)} wide, near (${f(V.grid.post_mean[0],2)}, ${f(V.grid.post_mean[1],2)}).`,'This is the posterior over codes, by Bayes’ rule. Its denominator is the p(x) we want.'],
 ask:['Why is the patch so small?','Which part of Bayes’ rule is hard?','What is the prior here, and what is the likelihood?'],
 expected:['784 pixels with σ = 0.2 pin the code down very precisely.','The evidence p(x) in the denominator: the integral from the last slide.','Prior N(0, I) over codes; likelihood p(x | z) from the decoder.'],
 transition:'In two dimensions we can still add up rectangles.',source:OURS});
S('evidence-grid','vae','With 2 numbers we can add up rectangles; with 16 we cannot',{builds:2,
 body:act(6,stack(eq(R`\log p(x)\approx\log\sum_{\text{cells}}p(x\mid z_c)\,p(z_c)\,\Delta z=${f(V.grid.log_px,2)}`),p(`Checked by importance sampling with 100,000 draws: ${f(V.grid.log_px_is,2)}.`),at(2,p('A grid with 161 points per axis (from −4 to 4 in steps of 0.05) has 161² ≈ 26,000 cells for 2 numbers, and 161¹⁶ ≈ 10³⁵ cells for 16. Lecture 1’s intractable evidence, again.')))),
 lines:[`For our 3 the grid gives log p(x) = ${f(V.grid.log_px,2)}. (A density over 784 numbers can exceed 1, so its log can be positive.)`,'Grids grow exponentially with the number of code dimensions. We need another way.'],
 ask:['Why is log p(x) positive?','Why do grids fail in 16 dimensions?'],
 expected:['It is a density, not a probability: with σ = 0.2 per pixel, densities are large.','The number of cells is (points per axis) to the power of the dimension.'],
 point:`The fine grid is 241 × 241 cells over a window of ±0.15 around the posterior. A coarse grid over the whole prior gives ${f(V.grid.log_px_wide,2)}: too coarse to resolve the patch.`,
 transition:'Sample codes from the prior instead?',source:OURS});
function mcChart(){
 const mc=V.naive_mc,xs=mc.map(r=>Math.log10(r.S)),P=panel({W:680,H:290,L:70,R:20,T:40,B:52,xMin:0,xMax:5,yMin:-600,yMax:300,ticksX:[0,1,2,3,4,5],fmtX:v=>`10${'⁰¹²³⁴⁵'[v]}`,ticksY:[-600,-300,0,300],xLabel:'number of prior draws S',fs:18,xAxisAtBottom:true});
 let s=P.axes+label(70,22,'estimate of log p(x), 30 repeats',{size:18});
 s+=`<polygon points="${xs.map((x,i)=>`${P.x(x)},${P.y(Math.max(-600,mc[i].hi))}`).join(' ')} ${[...xs].reverse().map((x,i)=>`${P.x(x)},${P.y(Math.max(-600,mc[mc.length-1-i].lo))}`).join(' ')}" fill="${palette.loss}" fill-opacity=".15"/>`;
 s+=P.line(xs,mc.map(r=>Math.max(-600,r.median)),palette.loss)+xs.map((x,i)=>P.dot(x,Math.max(-600,mc[i].median),palette.loss,5)).join('');
 s+=`<line x1="${P.x(0)}" y1="${P.y(V.grid.log_px)}" x2="${P.x(5)}" y2="${P.y(V.grid.log_px)}" stroke="${palette.data}" stroke-dasharray="7 5" stroke-width="2"/>`+label(P.x(0)+8,P.y(V.grid.log_px)-10,`true value ${f(V.grid.log_px,1)}`,{fill:palette.data,size:18});
 return s;
}
S('naive-mc','vae','Average p(x | z) over random codes? Most draws are wasted',{builds:2,
 body:act(6,cols(stack(eq(R`p(x)=\E_{z\sim p(z)}\big[p(x\mid z)\big]\approx\frac1S\sum_{s=1}^{S}p(x\mid z_s)`),fig(680,290,mcChart(),'The Monte Carlo estimate of log p(x) against the number of prior draws.')),
  stack(p(`With 100 draws the estimate is still off by about ${f(V.grid.log_px-V.naive_mc[2].median,0)}.`),at(2,p(`Of 100,000 prior draws, the best 10 carry ${f(100*V.top10_share,0)}% of the total.`)),at(2,small('Almost every code from the prior explains this 3 terribly. We need codes from the posterior’s tiny patch.'))),'1.45fr 1fr')),
 lines:['Lecture 1’s Monte Carlo: p(x) is an average of p(x | z) over prior codes. It converges, but only after tens of thousands of draws.','A handful of lucky draws carry the whole sum. We should draw codes where the posterior is, not from the prior.'],
 ask:['Why is the estimate so low with few draws?','Where should we draw codes from instead?'],
 expected:['Most prior draws decode to a different digit; p(x | z) is astronomically small for them.','From near the posterior: codes that could have made this 3. That is what an encoder will provide.'],
 transition:'Lecture 1’s tool for an intractable posterior.',source:OURS});
chap('6.3');
S('vi-recall','vae','Lecture 1’s tool: fit a simple q to the posterior',{builds:3,
 body:act(6,stack(eq(R`q(z)=\N\big(m,\ \mathrm{diag}(s^2)\big)`),at(2,eq(R`\log p(x)=\underbrace{\E_{q}\big[\log p(x\mid z)\big]-\KL\big(q(z)\,\|\,p(z)\big)}_{\ELBO(q)}\;+\;\KL\big(q(z)\,\|\,p(z\mid x)\big)`)),
  at(3,`<div class="tsm">${table(['Lecture 1','today'],[['unknown: network weights θ','unknown: the code z of one image'],['data 𝒟: 17 points','data: one image x'],['prior N(0, 1) on each weight','prior N(0, I) on the code']])}</div>`))),
 lines:['Approximate the posterior over codes with a Gaussian q, with its own mean m and widths s.','Lecture 1’s identity: log p(x) = ELBO + KL(q ‖ posterior). Raising the ELBO pushes q towards the posterior, without ever computing p(x).','Same maths, new roles: the unknown is now the code of one image, not the weights of a network.'],
 ask:['Which two terms make up the ELBO, in words?','Why is the ELBO a lower bound on log p(x)?','What are the variational parameters here?'],
 expected:['Reconstruct x well from codes drawn from q; keep q close to the prior.','The last KL is never negative.','m and s: four numbers for a 2-number code.'],
 transition:'Fit q for our 3.'});
S('vi3','vae','Fit q for our 3: press play',{builds:2,
 body:act(6,widget('vi3','Variational inference for one image.')),
 lines:['Start q at the prior N(0, I) (a circle of radius 2 in the left panel) and climb the ELBO with reparameterised draws.',`q moves into the posterior’s patch and shrinks to fit it. The ELBO ends at ${f(V.vi3.final_elbo,2)}, just below log p(x) = ${f(V.grid.log_px,2)}.`],
 ask:['What happens to the ELBO at first? Why?','How big is the final gap?'],
 expected:['It wobbles and dips: q is still huge, and most codes it draws explain the 3 badly.',`About ${f(V.grid.log_px-V.vi3.final_elbo,2)} nats: q is nearly the exact posterior.`],
 point:'Adam, step size 0.05 then 0.01, 64 reparameterised draws per step, 1,200 steps (seed 23). The ELBO plotted is the noisy estimate at each step.',
 transition:'One image took 1,200 steps. Now 60,000 images.',source:OURS});
chap('6.4');
S('amortise','vae','60,000 images would mean 60,000 separate fits',{builds:3,
 body:act(6,stack(p('Each image has its own posterior, so its own m and s: 1,200 steps each, and again for every new image.'),at(2,p('Instead, train a network that reads an image and outputs its m and s in one pass: the <b>encoder</b>.')),at(3,eq(R`q_\phi(z\mid x)=\N\big(\mu_\phi(x),\ \mathrm{diag}\,\sigma_\phi(x)^2\big)`)))),
 lines:['Per-image VI works, but it is slow: 60,000 fits during training, and a fresh fit for every new image.','Learn the fit instead: an encoder network outputs the mean and spread of q for any image.','One network with weights φ, shared by all images. This is called amortised inference: pay once in training, then every image is one forward pass.'],
 ask:['How many variational parameters would per-image VI need for 60,000 images?','What does the encoder output?','What is the price of sharing one network?'],
 expected:['4 × 60,000 = 240,000.','Two means and two log-variances per image.','Its q may be a little worse than a dedicated fit for each image (next slides).'],
 transition:'Put the encoder in front of the decoder.'});
S('vae-arch','vae','The VAE: an encoder that outputs q, a code drawn from it, and the decoder',{builds:3,
 body:act(6,`<div class="state-slot"><div data-only="1">${aeDiagram({kind:'vae',stage:2,showOut:false})}</div><div data-only="2">${aeDiagram({kind:'vae',stage:3,showOut:false})}</div><div data-only="3">${aeDiagram({kind:'vae',stage:5})}</div></div>`),
 lines:['The encoder now ends in 4 numbers, not 2. A bug? No: a mean and a log variance for each code number.','Draw the code by reparameterisation (Lecture 1, section 3.5): z = μ + σ ⊙ ε with ε ~ N(0, I). Sampling z directly would block the gradient; written this way, backpropagation reaches μ and σ.','The decoder is unchanged: from z to a mean image.'],
 ask:['Why log σ² instead of σ?','Why reparameterise instead of sampling z directly?','At test time, which code do we use to reconstruct?'],
 expected:['Any real number is a valid log variance; σ = exp(½ log σ²) is then positive.','sample() blocks the gradient path from the loss to μ and σ.','Usually the mean μ.'],
 transition:'The encoder on our 3.'});
S('q3','vae','The encoder on our 3: one pass, almost as good as 1,200 steps',{builds:2,
 body:act(6,cols(fig(420,420,posteriorSVG({S:420,view:'zoom',enc:true,q:[V.vi3.final_mu[0],V.vi3.final_mu[1],V.vi3.final_sd[0],V.vi3.final_sd[1]]}).s,'The posterior, the per-image q and the encoder’s q, zoomed.'),
  stack(table(['','ELBO'],[['log p(x), the exact target',f(V.grid.log_px,2)],['q fitted for this image (1,200 steps)',f(V.vi3.final_elbo,2)],['q from the encoder (one pass)',f(V.q3.elbo,2)]]),at(2,p(`Encoder: μ = (${f(V.q3.mu[0],3)}, ${f(V.q3.mu[1],3)}), σ = (${f(V.q3.sd[0],3)}, ${f(V.q3.sd[1],3)}).`)),at(2,small('Dashed: the encoder’s q. Shaded: the per-image fit. Both 2 standard deviations.'))),'1fr 1.1fr')),
 lines:[`The per-image fit gets within ${f(V.grid.log_px-V.vi3.final_elbo,2)} of log p(x); the encoder, in one pass, within ${f(V.q3.gap,2)}.`,'The encoder’s q sits slightly off-centre and a little too narrow. That small loss is the price of amortisation.'],
 ask:['Which q is better, and by how much?','Why is the encoder’s q not perfect?'],
 expected:[`The per-image fit, by ${f(V.vi3.final_elbo-V.q3.elbo,2)} nats.`,'One network must serve all 60,000 images; it cannot be exactly right for each.'],
 transition:'The loss, term by term.',source:OURS});
chap('6.5');
S('vae-loss','vae','The loss for one image is the negative ELBO',{builds:3,
 body:act(6,stack(eq(R`L(\phi,\theta;x)=\underbrace{\E_{q_\phi(z\mid x)}\Big[\tfrac{1}{2\sigma^2}\lVert x-f_\theta(z)\rVert^2\Big]}_{\text{reconstruction}}+\underbrace{\KL\big(q_\phi(z\mid x)\,\|\,\N(0,I)\big)}_{\text{stay near the prior}}`),
  at(2,eq(R`\text{reconstruction}\approx\tfrac{1}{2\sigma^2}\lVert x-f_\theta(\mu+\sigma\odot\epsilon)\rVert^2,\qquad \epsilon\sim\N(0,I)\ \text{(one draw)}`)),
  at(3,eq(R`\KL=\tfrac12\sum_{k=1}^{2}\big(\mu_k^2+\sigma_k^2-\log\sigma_k^2-1\big)\qquad\text{(Lecture 1, section 2.7)}`)))),
 lines:['Minus the ELBO, up to the constant 541.4 from section 6.1: an expected scaled squared error, plus a KL from q to the prior.','The expectation is estimated with one reparameterised draw per image per step.','The KL between two diagonal Gaussians has a closed form: no sampling needed.'],
 ask:['Which term pulls the codes towards N(0, I)?','Why is one draw enough?','What does the KL do to a code with μ = 5?'],
 expected:['The KL term.','Over many steps and images the noise averages out; the estimate is unbiased.','Penalises it by about 12.5: strong pressure back towards the origin.'],
 point:'Shape of the loss: data fit + penalty, like ridge regression (squared error + λ‖w‖²). Here the penalty is a KL to the prior, and it falls out of the ELBO rather than being added by hand.',
 transition:'Put in the numbers for our 3.'});
const WK=V.worked;
S('vae-worked','vae','One step for our 3, with numbers',{builds:3,
 body:act(6,stack(eq(R`\mu=(${f(V.q3.mu[0],3)},\,${f(V.q3.mu[1],3)}),\quad\sigma=(${f(V.q3.sd[0],3)},\,${f(V.q3.sd[1],3)}),\quad\epsilon=(${f(WK.eps[0],1)},\,${f(WK.eps[1],1)})`),eq(R`z=\mu+\sigma\odot\epsilon=(${f(WK.z[0],3)},\,${f(WK.z[1],3)})`),
  at(2,`<div class="dflex" style="display:flex;gap:20px;align-items:center">${digit('example',0,92)}<span class="lead">vs</span>${digit('vae2_worked_recon',0,92)}${eq(R`\tfrac{1}{2(0.2)^2}\times${f(WK.sse,2)}=${f(WK.recon,1)}`)}</div>`),
  at(3,eq(R`\KL=${f(V.q3.kl_prior,2)}\qquad L=${f(WK.recon,1)}+${f(V.q3.kl_prior,2)}=${f(WK.loss,1)}`)),at(3,small(`Check: L − ${f(-V.loglik_const,1)} = ${f(WK.loss+V.loglik_const,1)}, close to −ELBO = ${f(-V.q3.elbo,1)}. The loss drops the constant, and one draw is noisy.`)))),
 lines:['The encoder gives μ and σ; draw ε; the code is z = μ + σ ⊙ ε.','Decode z and compare with our 3: squared error 23.2, scaled by 1/(2σ²) = 12.5.','Add the KL to the prior: the loss for this image on this step.'],
 ask:['Why is z so close to μ?','Which term is larger, and what does that say?','What does backward() update?'],
 expected:['σ is tiny (about 0.01): the code of this 3 is pinned down precisely.','Reconstruction, by far: fitting the pixels dominates this model.','The decoder’s θ through the reconstruction, the encoder’s φ through both μ, σ (via z) and the KL.'],
 transition:'The whole thing in PyTorch.',source:OURS});
S('vae-code','vae','The VAE in PyTorch: three changes to the autoencoder',{builds:2,
 body:act(6,cols(`<div class="code-xs">${code(`class VAE(AutoEncoder):          # same decoder
    def __init__(self, z=2):
        super().__init__(z)
        self.encoder[-1] = nn.Linear(400, 2 * z)  # (1)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps       # (2)
        return self.decoder(z), mu, logvar

x_hat, mu, logvar = model(x)
recon = ((x_hat - x)**2).sum(1) / (2 * 0.2**2)
kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(1)  # (3)
loss = (recon + kl).mean()`)}</div>`,stack(at(2,p('(1) the encoder outputs 2z numbers, (2) the code is a reparameterised draw, (3) the loss adds the KL.')),at(2,small('Training loop: identical to the autoencoder’s.'))),'1.75fr 1fr')),
 lines:['Same encoder and decoder as Part 4, with three changes.','Everything else, the batches, Adam and 50 passes over the data, is the autoencoder’s loop.'],
 ask:['What are the three changes?','How many lines of the training loop change?'],
 expected:['Two outputs per code number, a random draw of z, and the KL term.','None: only the model and the loss.'],
 transition:'How the two terms behave while training.',nb:'sec-6-5'});
function vaeCurves(){
 const c=V.curve,ep=c.epoch;
 const A_=lineChart({W:540,H:320,xMin:1,xMax:50,yMin:300,yMax:600,ticksX:[1,25,50],ticksY:[300,400,500,600],xLabel:'epoch',yLabel:'reconstruction term',series:[{x:ep,y:c.recon,color:palette.loss}]});
 const B_=lineChart({ox:560,W:540,H:320,xMin:1,xMax:50,yMin:0,yMax:12,ticksX:[1,25,50],ticksY:[0,4,8,12],xLabel:'epoch',yLabel:'KL term',series:[{x:ep,y:c.kl,color:palette.q}],L:630-560});
 return A_.s+B_.s;
}
S('vae-curves','vae','While training: reconstruction falls, the KL settles',{builds:2,
 body:act(6,fig(1100,320,vaeCurves(),'The two terms of the VAE loss during training.'),at(2,p(`After 50 epochs: reconstruction ${f(V.curve.recon.at(-1),0)}, KL ${f(V.curve.kl.at(-1),1)} per image.`))),
 lines:['The reconstruction term falls steadily, as for the autoencoder.','The KL stays near 9 nats per image: each code is allowed to say about 9 nats more than the prior, no more.'],
 ask:['Why does the KL not fall to zero?','What would a KL of zero mean?'],
 expected:['Then q would equal the prior for every image: the code would carry no information and reconstruction would suffer.','The code ignores the image: every 3 and 7 gets the same q. We will see this happen.'],
 transition:'Now sample.',source:OURS});
chap('6.6');
S('vae-samples','vae','Now sample: z ~ N(0, I), then decode',{builds:3,
 body:act(6,rows(['autoencoder, 2',digits('ae2_prior_samples',{cols:16,size:58,gap:4})],['VAE, 2',at(2,digits('vae2_prior_samples',{cols:16,size:58,gap:4}))],['VAE, 16',at(3,digits('vae16_prior_samples',{cols:16,size:58,gap:4}))])),
 lines:['The same 16 draws from N(0, I) as before, through the autoencoder’s decoder.','Through the 2-number VAE: almost every draw is a digit, and all ten digits appear.','Through the 16-number VAE: sharper and more varied. These include the fakes from the start of the lecture.'],
 ask:['What changed between the first two rows?','Why are the VAE samples still a little blurry?','Which digits are missing from the autoencoder row?'],
 expected:['Training made the codes fill N(0, I), so random codes land on digits.','Each pixel is modelled as an independent Gaussian around a mean image; the mean of plausible strokes is soft.','Most of them: random codes reach only the digits whose codes sit near the origin.'],
 transition:'Where the codes are now.',source:OURS});
S('vae-scatter','vae','The codes now fill N(0, I)',{builds:2,
 body:act(6,cols(`<div class="state-slot"><div data-only="1">${fig(560,440,scatterSVG('vae2',{W:560,H:440,lim:[-5,5,-5,5],ticks:[[-4,-2,0,2,4],[-4,-2,0,2,4]]}).s,'VAE codes of 1,000 test digits.')}</div><div data-only="2">${fig(560,440,scatterSVG('vae2',{W:560,H:440,lim:[-5,5,-5,5],ticks:[[-4,-2,0,2,4],[-4,-2,0,2,4]],prior:500}).s,'VAE codes with 500 draws from N(0, I).')}</div></div>`,
  stack(p(`Codes of all 10,000 test digits: mean (${f(V.agg.mean[0],1)}, ${f(V.agg.mean[1],1)}), spread (${f(V.agg.sd[0],1)}, ${f(V.agg.sd[1],1)}).`),at(2,p('Random draws (red) now land among real codes, on every digit.'))),'1.25fr 1fr')),
 lines:['The VAE’s codes (the means μ of q) for the same 1,000 test digits: one round cloud around the origin, digits side by side.','Overlay 500 draws from N(0, I): they fall on real codes of every digit. That is why sampling works.'],
 ask:['Compare with the autoencoder’s map (Part 5). What changed?','Is the cloud exactly N(0, I)?'],
 expected:['Codes now fill a round region around the origin with no large gaps; scales are about 1, not 20.','Not quite: a little wider (spread 1.7) and slightly off-centre. The KL is a pull, not a constraint.'],
 transition:'Try the three ways again.',source:OURS});
S('explore-vae','vae','Your turn again: the same three ways, now with the VAE',{builds:1,
 body:act(6,widget('explorer','Choose a code for the VAE or the autoencoder and see its decoder output.',{net:'vae2',both:'1',limae:'-6,24,-9,14',limvae:'-4.5,4.5,-4.5,4.5'})),
 lines:['Random codes, encoded test digits, or click anywhere. Switch between the VAE and the autoencoder to compare.'],
 ask:['Click in the gaps between digit groups. What do you see now?'],
 expected:['Plausible in-between shapes: the space between digits is filled.'],
 transition:'A grid, and a walk.',source:OURS});
S('vae-grid','vae','Decode a grid of codes: digits everywhere',{builds:2,
 body:act(6,cols(stack(small('VAE: codes from −3 to 3 in each direction'),digits('vae2_grid',{cols:12,size:30,gap:1})),at(2,stack(small('VAE: walk from the same 1 to the same 0'),digits('vae2_interp',{cols:9,size:50,gap:4}),small('16-number VAE'),digits('vae16_interp',{cols:9,size:50,gap:4}))),'1fr 1.1fr')),
 lines:['Every point of a 12 × 12 grid over [−3, 3]² decodes to a digit, morphing smoothly from one to the next.','Walk from the same 1 to the same 0: every step is a plausible digit.'],
 ask:['Compare with the autoencoder’s grid. What is different?','Why are the in-between images plausible now?'],
 expected:['No junk: the whole region that N(0, I) samples from is used.','Each image owned a small cloud of codes during training, so the decoder learned to draw something sensible everywhere in between.'],
 transition:'Section 6.7: a weight on the KL.',source:OURS});
chap('6.7');
S('beta-what','vae','Put a weight on the KL: β',{builds:2,
 body:act(6,stack(eq(R`L_\beta=\E_{q}\Big[\tfrac{1}{2\sigma^2}\lVert x-f_\theta(z)\rVert^2\Big]+\beta\,\KL\big(q(z\mid x)\,\|\,\N(0,I)\big)`),p('β = 1 is the true ELBO. Changing β is the same as assuming noisier or cleaner pixels (a different σ).'),
  at(2,cols(card('Small β','codes free to spread: close to an autoencoder',palette.p),card('Large β','codes squeezed towards the prior: closer to N(0, I), less detail',palette.q),'1fr 1fr')))),
 lines:['Weight the KL by β, as in β-VAE. Like ridge regression: data fit plus β times a penalty.','Predict before we look: what happens at β = 0, and at very large β?'],
 ask:['Which value of β gives back the real ELBO?','What do you predict at the two extremes?'],
 expected:['β = 1.','β = 0: codes spread out like an autoencoder’s. Huge β: codes collapse onto the prior and stop carrying information.'],
 point:'PML 2023 uses mean-over-pixels squared error plus β times the KL; its β = 0.001 corresponds to about β = 5 here.',
 transition:'Seven trained VAEs.'});
S('beta','vae','Seven VAEs, β from 0 to 256: try them',{builds:1,
 body:act(6,widget('beta','Codes, reconstructions and samples of VAEs trained with different KL weights.',{beta:'1'})),
 lines:['Click through β. Watch three things: how spread the codes are, the reconstructions, and the samples.'],
 ask:['At which β do samples look best? When do reconstructions start to blur?'],
 expected:['Around β = 1 to 4. By β = 64 reconstructions blur; at 256 everything is the same blob.'],
 transition:'Two extremes worth a closer look.',source:OURS});
S('beta-zero','vae','β = 0: is it just an autoencoder? (An exam question)',{builds:2,
 body:act(6,cols(table(['β','KL per image','average σ of q','spread of codes'],['0','1','256'].map(b=>[b,f(BETA[b].test_kl,1),f(BETA[b].sd_mean,4),f(BETA[b].code_spread,2)])),stack(p('With β = 0 nothing stops q from shrinking.'),at(2,p(`Structurally it still draws z = μ + σε, unlike an autoencoder. But training drives σ to ${f(BETA['0'].sd_mean,4)}, so in practice it behaves like one: spread-out codes, poor samples.`))),'1.3fr 1fr')),
 lines:['β = 0 keeps the random draw z = μ + σε, so it is not quite the same model as an autoencoder.','But training drives σ almost to zero, and then it behaves like one: spread-out codes and poor samples.'],
 ask:['Is a VAE with β = 0 the same as an autoencoder?','What does the KL term do to σ?'],
 expected:['Different by construction (the encoder outputs a distribution and z is sampled), but in practice it learns to make σ ≈ 0 and becomes one.','It stops σ shrinking: −log σ² grows as σ → 0.'],
 transition:'The other extreme.',source:OURS});
S('beta-collapse','vae','β = 256: every code decodes to the same blur',{builds:2,
 body:act(6,cols(rows(['β = 256, recon.',digits('beta256_recon_watch',{cols:8,size:58,gap:5})],['β = 256, samples',digits('beta256_prior_samples',{idx:[0,1,2,3,4,5,6,7],cols:8,size:58,gap:5})]),
  stack(p(`KL = ${f(BETA['256'].test_kl,2)}: q(z | x) equals the prior for every image.`),at(2,p('The decoder learns to ignore z and outputs the average digit. This is called <b>posterior collapse</b>.'))),'1.5fr 1fr')),
 lines:['With a huge weight on the KL, the cheapest solution is q = N(0, I) for every image.','Then z carries no information, so the decoder ignores it and draws the average digit every time.'],
 ask:['Why is KL = 0 the cheapest solution here?','How would you spot collapse during training?'],
 expected:['β × KL dominates the loss; reconstruction error is worth less than any information in the code.','The KL term falls to about 0 and samples all look the same.'],
 transition:'Wrap-up.',source:OURS});
chap('6.8');
S('arc','vae','The whole lecture in one table',{
 body:act(6,`<div class="tsm">`+table(['model','how it generates','how it is fitted','what goes wrong'],[
  ['Gaussian or mixture (Part 2)','pick a bump, draw a number','maximum likelihood','fine for a few numbers'],
  ['Gaussians on PCA numbers (3)','sample a code, decode linearly','maximum likelihood, then EM','straight-line decoder: blurry'],
  ['autoencoder (4–5)','no rule for choosing codes','squared error','random codes miss the data'],
  ['autoencoder + mixture (5)','sample a fitted code, decode','two separate fits','no single p(x)'],
  ['VAE (6)','z ~ N(0, I), decode','the ELBO: reconstruction + KL','soft samples; collapse if β is large']])+`</div>`),
 lines:['From one number per example to 784, from formulas to networks: always a hidden code first, then the data.'],
 ask:['Which models can both score and sample?'],expected:['The Part 2 models, the PCA mixtures and the VAE (through the ELBO, a bound). The plain autoencoder can do neither.'],
 transition:'What comes next.'});
S('next','vae','Next: can we drop the likelihood altogether?',{builds:2,
 body:act(6,stack(p('VAE samples are soft because each pixel is an independent Gaussian around a mean image.'),at(2,p('Next: a generator trained by a second network that judges real against fake (GANs). Then: many small denoising steps (diffusion).')))),
 lines:['The pixel-wise Gaussian likelihood is the VAE’s weak point: it rewards the safe average stroke.','Next lectures replace it: a learned critic (GANs) and repeated denoising (diffusion).'],
 ask:['Why does a pixel-wise Gaussian encourage soft images?','What else could tell a generator its images are wrong?'],
 expected:['Squared error favours the mean of plausible images, which is blurry.','A classifier trained to tell real from generated: a learned critic.'],
 transition:'End of the main route. Optional slides follow.'});

// ================================================================= Optional depth
const O=(id,title,o)=>S(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});
O('sparse','A sparse autoencoder: a KL between two coins',{builds:2,
 body:stack(p('Wide hidden layer, but each unit should be on only rarely, say for ρ = 5% of images.'),eq(R`L=\text{reconstruction}+\lambda\sum_{j}\KL\big(\text{Bernoulli}(\rho)\,\|\,\text{Bernoulli}(\hat\rho_j)\big)`),
  at(2,p('ρ̂ⱼ is unit j’s average activation over the data. Lecture 1’s Bernoulli KL, used as a penalty.'))),
 lines:['A sparse autoencoder penalises hidden units that are on too often.','The penalty is the KL between two coins: the target rate ρ and the unit’s actual rate.'],
 ask:['Why would we want few active units?','Where else did we add “loss + penalty”?'],
 expected:['Each unit specialises in a feature; this regularises the autoencoder.','The VAE: reconstruction + KL; ridge regression: squared error + λ‖w‖².'],
 transition:'Back to the main route.',source:'Ng, “Sparse autoencoder”, CS294A lecture notes; Lilian Weng, “From Autoencoder to Beta-VAE”.'});
O('heldout-mnist','Held-out scores of the PCA mixtures',{builds:1,
 body:table(['d','K = 1','K = 5','K = 10','K = 20'],[2,10,20,50].map(dd=>[String(dd),...[1,5,10,20].map(k=>f(M.gens[`d${dd}k${k}`].test_ll,1))])),
 lines:['Average log-likelihood of the 10,000 test codes. Compare along a row only: different d are densities in different spaces.'],
 ask:['Why can we not compare across rows?'],expected:['Each row is a density over a different number of dimensions, so their units differ.'],
 transition:'Back to the main route.',source:OURS});
O('em','EM in one slide: the classic way to fit a mixture',{builds:2,
 body:stack(eq(R`r_{ik}=\frac{\pi_k\N(x_i;\mu_k,\Sigma_k)}{\sum_j\pi_j\N(x_i;\mu_j,\Sigma_j)}\qquad\text{(E: who made each point?)}`),at(2,eq(R`\pi_k=\tfrac1n\textstyle\sum_i r_{ik},\quad \mu_k=\frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}},\quad \Sigma_k=\frac{\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{\sum_i r_{ik}}\qquad\text{(M)}`))),
 lines:['E step: for each point, the posterior probability that each bump made it (the eruption “kind”, given the wait).','M step: refit each bump to the points, weighted by those probabilities. Repeat: the likelihood never decreases.'],
 ask:['Where have you seen r_ik before?','Why is EM a cousin of the VAE?'],
 expected:['It is p(k | x), the posterior over the hidden variable: Bayes’ rule.','Both alternate between inferring hidden variables and refitting the model; the VAE uses an approximate posterior q because the exact one is intractable.'],
 transition:'Back to the main route.'});

// ---------------------------------------------------------------- the map slide, first
const toc=PARTS.map(([k,t])=>`<div><h3>${k} · ${t}</h3><ol>${CHAPTERS.filter(c=>c[0].startsWith(k+'.')).map(c=>`<li><a href="#${firstOf[c[0]]}/1" data-goto="${firstOf[c[0]]}">${c[0]} ${c[1]}</a></li>`).join('')}</ol></div>`).join('');
slides.unshift({id:'contents',section:'map',title:'Lecture 2 at a glance: six parts, twenty-nine sections',builds:1,
 body:`<div class="toc toc-flow">${toc}</div>`,prompts:['Click a section to jump to it. We start with why generative models matter, and end by training a VAE.'],chapter:'',notebook:ANCHORS.size?{href:COLAB,label:'▶ Notebook'}:null,
 notes:{ask:['What does each part add to the one before?'],expected:['A harder kind of data, or a better way to compress and sample it.'],point:'',misconception:'',transition:'Part 1: things that do not exist.',demo:''},source:'',optional:false});
