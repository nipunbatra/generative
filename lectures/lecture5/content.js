// Lecture 5: why generate, autoencoders and the VAE, for a class that already knows the ELBO (Lecture 1).
// A condensed route through Lecture 2's Parts 4–6. It reuses Lecture 2's trained networks, numbers, images
// and live figures (lecture2/data, lecture2/widgets.js), so every number here is one Lecture 2 computed.
import {m,eq,at,cols,code,table,palette} from '../shared/author.js';
import {init,mount as mountWidgets,panel,label,fig,widget,num,comma,lin,digits,digit,digitSrc,rawDigit,imgs,scatterSVG,posteriorSVG,squareW,CLASS,MUTED,LINE} from '../lecture2/widgets.js';
import {init as init5,mount as mount5,widget5} from './widgets.js';

const get=(f,base=import.meta.url)=>fetch(new URL(f,base),{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error(`${f}: ${r.status}`);return r.json();});
const L2=new URL('../lecture2/',import.meta.url);
const [d,DG,NETS,NW,WAVE]=await Promise.all([get('data/science.json',L2),get('data/digits.json',L2),get('data/nets.json',L2),get('assets/examples/next-word.json',L2),get('assets/examples/tts-wave.json',L2)]);
const LLM=await Promise.all([1,2,3].map(i=>fetch(new URL(`assets/examples/llm-sample-${i}.txt`,L2)).then(r=>r.text())));
const ANCHORS=new Set(await get('./data/notebook-anchors.json').catch(()=>[]));
init(d,DG,NETS);init5(d,NETS);
export const mount=(stage,ctx)=>{mountWidgets(stage,ctx);mount5(stage,ctx);};

// ---------------------------------------------------------------- lecture-local styles (the digit grids and cards of Lecture 2)
const CSS=`
.dg{image-rendering:pixelated;border:1px solid ${LINE};border-radius:3px;background:#FAFAFB;display:block}
.dgrid{display:grid;justify-content:start;align-items:start}
.dg-cell{margin:0;display:flex;flex-direction:column;align-items:center;gap:3px}
.dg-cell figcaption{font-size:16px;color:${MUTED};line-height:1.1;text-align:center}
.dg-title{font-size:19px;color:${palette.ink};font-weight:600;margin-bottom:2px}
.row-label{font-size:19px;color:${MUTED};line-height:1.2}
.digit-rows{display:grid;grid-template-columns:auto 1fr;gap:10px 18px;align-items:center}
.hover-card[hidden]{display:none!important}
.hover-card{position:absolute;right:40px;top:150px;display:flex;flex-direction:column;gap:8px;font-size:20px;background:#fff;border:1px solid ${LINE};padding:12px;border-radius:10px;box-shadow:0 8px 30px #14171F14}
.takeaway{font-size:22px;line-height:1.38;border-left:4px solid #0F766E;background:#0F766E0D;padding:9px 16px;border-radius:6px}.takeaway b{color:#0F766E}
.l1-link{display:inline-block;font-size:16px;font-weight:650;color:#8B2CDE;border:1px solid #8B2CDE66;background:#8B2CDE0D;border-radius:999px;padding:3px 12px;text-decoration:none;margin:2px 6px 2px 0}
.code-xs .code{font-size:18px;line-height:1.38;padding:10px 16px}
.tsm table{font-size:21px}.tsm th,.tsm td{padding:7px 12px}
.cards{display:grid;gap:18px}.cards>*{min-width:0}.card{border:1.5px solid ${LINE};border-radius:14px;padding:14px 18px;display:flex;flex-direction:column;gap:8px;min-width:0}
.card h4{margin:0;font-size:22px;line-height:1.2}.card p{margin:0;font-size:20px;line-height:1.4}.cards.big .card h4{font-size:25px}.cards.big .card p{font-size:23px}.card .display{font-size:28px}
.tag{display:inline-block;font-size:15px;font-weight:650;border-radius:999px;padding:2px 10px;margin-top:4px}
.tag.real{color:${palette.p};border:1.5px solid ${palette.p}}.tag.fake{color:${palette.loss};border:1.5px solid ${palette.loss}}
.pix-table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}.pix-table td{width:38px;height:30px;text-align:center;padding:0;border:1px solid #fff}
.mini{font-size:20px;line-height:1.35}
.dflex{display:flex;gap:22px;align-items:center}
.mh .diagram,.mh .plot{max-height:var(--mh)}
.mchain{display:grid;grid-template-columns:auto auto 1fr auto;gap:22px 20px;align-items:center;font-size:24px}.mchain>div{min-width:0}.mchain .math{font-size:1.12em;white-space:nowrap}.mread{grid-template-columns:auto 1fr auto;gap:26px 28px}.mread .mc-w{color:${palette.ink}}
.mc-h{font-size:17px;font-weight:650;color:${MUTED};border-bottom:1.5px solid ${LINE};padding-bottom:4px}.mc-n{font-weight:700}.mc-w{color:${MUTED}}.mc-s{font-size:18px;color:${palette.q};font-weight:600;white-space:nowrap}
@media(max-width:760px){.reading .mchain{grid-template-columns:auto 1fr;overflow-x:auto}.reading .mchain .mc-h:nth-child(-n+2){display:none}}
.card .src{font-size:14px;color:${MUTED}}
.prompt-box{font-size:21px;line-height:1.4;border-left:4px solid ${palette.q};background:#8B2CDE0D;padding:10px 16px;border-radius:6px}
.sample-text{font-size:20px;line-height:1.42;border:1px solid ${LINE};border-radius:10px;padding:10px 14px;background:#fff}.sample-text b{color:${palette.q}}
.photo{border-radius:8px;display:block;width:100%;height:auto}
.photo-row{display:grid;gap:14px;align-items:end}.photo-row figure{margin:0;display:flex;flex-direction:column;gap:6px}.photo-row figcaption{font-size:17px;color:${MUTED};line-height:1.25}
.audio-row{display:flex;align-items:center;gap:18px}.audio-row audio{height:38px}
@media(max-width:760px){.reading .photo-row{grid-template-columns:repeat(2,minmax(0,1fr))!important}.reading .dgrid,.reading .digit-rows>div,.reading .pix-table{overflow-x:auto;max-width:100%}.reading .cards{grid-template-columns:1fr!important}.reading .dflex{flex-wrap:wrap}.reading .dflex>*{min-width:0;max-width:100%}.reading .digit-rows{grid-template-columns:1fr}.reading .pix-table{display:block}}
`;
if(!document.getElementById('lecture5-style')){const st=document.createElement('style');st.id='lecture5-style';st.textContent=CSS;document.head.append(st);}

// ---------------------------------------------------------------- helpers
const R=String.raw;
const f=(v,n=2)=>num(v,n);
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const mini=s=>`<p class="mini">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const grid=(n,...a)=>`<div class="cards" style="grid-template-columns:repeat(${n},minmax(0,1fr))">${a.join('')}</div>`;
const bigGrid=(n,...a)=>`<div class="cards big" style="grid-template-columns:repeat(${n},minmax(0,1fr))">${a.join('')}</div>`;
/** A card holding display maths, a line of text and a link. */
const tool=(title,maths,text,link,color)=>`<div class="card" style="border-color:${color}66"><h4 style="color:${color}">${title}</h4>${maths}<p>${text}</p><div>${link}</div></div>`;
const card=(title,body,color=palette.ink,src='')=>`<div class="card" style="border-color:${color}66"><h4 style="color:${color}">${title}</h4><p>${body}</p>${src?`<span class="src">${src}</span>`:''}</div>`;
/** A file in Lecture 2's folder (its gallery and generated examples). */
const L2A=f=>`../lecture2/${f}`;
const photo=(src,caption='')=>`<figure><img class="photo" src="${L2A(src)}" alt="${caption.replace(/<[^>]+>/g,'')}">${caption?`<figcaption>${caption}</figcaption>`:''}</figure>`;
const takeaway=s=>`<div class="takeaway"><b>Takeaway.</b> ${s}</div>`;
const maxH=(h,html)=>`<div class="mh" style="--mh:${h}px">${html}</div>`;
/** A link back to the Lecture 1 slide where an idea was taught. */
const l1=(id,sec,text)=>`<a class="l1-link" href="../lecture1/#${id}/1" target="_blank" rel="noopener">↗ Lecture 1, ${sec}: ${text}</a>`;
const rows=(...r)=>`<div class="digit-rows">${r.map(([lab,html])=>`<div class="row-label">${lab}</div><div>${html}</div>`).join('')}</div>`;
const dimg=(key,i,x,y,s,extra='')=>`<image href="${digitSrc(key,i)}" x="${x}" y="${y}" width="${s}" height="${s}" style="image-rendering:pixelated" ${extra}/><rect x="${x}" y="${y}" width="${s}" height="${s}" fill="none" stroke="${LINE}"/>`;
/** A subscript inside SVG text, e.g. sub('g','φ'). */
const sub=(base,s_)=>`${base}<tspan baseline-shift="sub" font-size="70%">${s_}</tspan>`;
const GREEN='#147737';
// Lecture 1's colours for the pieces of Bayes' rule and the ELBO: q purple, likelihood orange, prior green, evidence blue, posterior teal, KL red.
const PRI=t=>R`\textcolor{#147737}{${t}}`;
const M=d.mnist,A=d.ae,G=d.gen,V=d.vae,SW=A.sweep,BETA=V.beta,WK=V.worked,PZ=V.prior_z;
const OURS='Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).';

// ---------------------------------------------------------------- sections and the slide builder
export const meta={
 id:'lecture5',number:5,title:'Autoencoders and the VAE, from the ELBO',
 subtitle:'For a class that knows the ELBO: why generate, an autoencoder, why it cannot generate, the VAE as Lecture 1’s objective with a code per image, and every term traced back to KL and the ELBO',minutes:86,
 notebook:'lecture5-autoencoders-vae.ipynb',
 sections:[
  {id:'map',title:'Lecture map',minutes:1},
  {id:'why',title:'Part 1 · Why generate?',minutes:14},
  {id:'ae',title:'Part 2 · Autoencoders: squeeze, then rebuild',minutes:15},
  {id:'gen',title:'Part 3 · Can an autoencoder generate?',minutes:8},
  {id:'idea',title:'Part 4 · From an autoencoder to a VAE: three fixes',minutes:12},
  {id:'vae',title:'Part 5 · Why this loss: Lecture 1’s ELBO, one image at a time',minutes:30},
  {id:'end',title:'Part 6 · Putting it together',minutes:6},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
/** Numbered sections. The notebook uses the same numbers, with anchors sec-1-1 … sec-6-2. */
const CHAPTERS=[
 ['1.1','Things that do not exist'],['1.2','Text, pictures and sound'],['1.3','What generative models are for'],['1.4','What they all share'],
 ['2.1','An image is 784 numbers'],['2.2','Encoder, code, decoder'],['2.3','The loss, and training'],['2.4','What the code holds'],
 ['3.1','Decode random codes'],['3.2','Where the codes are'],['3.3','What we want instead'],
 ['4.1','Fix 1: choose where codes live'],['4.2','Fix 2: a cloud for each image'],['4.3','Fix 3: pull the clouds to N(0, I)'],['4.4','The loss the fixes give'],
 ['5.1','What Lecture 1 gave us'],['5.2','A model with a hidden code'],['5.3','The same wall as Lecture 1'],['5.4','The ELBO for one image'],['5.5','The loss, term by term'],['5.6','Both terms, live'],['5.7','What the VAE learned'],['5.8','The KL weight β'],
 ['6.1','The whole lecture in maths'],['6.2','Wrap-up'],
];
const PARTS=[['1','Why generate?'],['2','Autoencoders: squeeze, then rebuild'],['3','Can an autoencoder generate?'],['4','From an autoencoder to a VAE: three fixes'],['5','Why this loss: Lecture 1’s ELBO, one image at a time'],['6','Putting it together']];
const COLAB='https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture5/lecture5-autoencoders-vae.ipynb';
const secAnchor=k=>`sec-${k.replaceAll('.','-')}`;
let chapter=null;const firstOf={};
const chap=k=>{chapter=CHAPTERS.find(ch=>ch[0]===k);if(!chapter)throw new Error('Unknown section '+k);};
/** A link to one notebook cell in Colab; once the notebook is built, the anchor must exist in it. */
function nbLink(anchor,number){
 if(!ANCHORS.size)return null;
 if(!ANCHORS.has(anchor))throw new Error(`Notebook anchor ${anchor} is missing: rebuild the notebook`);
 return {href:`${COLAB}#scrollTo=${anchor}`,label:number?`▶ Notebook ${number}`:'▶ Notebook'};
}
/** The bottom bar carries one story line per build; questions and answers live in the notes. */
function S(id,section,title,o){
 const builds=o.builds||1;
 for(const k of ['lines','ask','expected'])if(!Array.isArray(o[k])||o[k].length!==builds)throw new Error(`${id}: ${k} needs ${builds} entries`);
 const ch=section==='optional'||section==='map'||o.part?null:chapter;
 if(ch&&!firstOf[ch[0]])firstOf[ch[0]]=id;
 const anchor=o.nb||(ch?secAnchor(ch[0]):null);
 slides.push({id,section,title,builds,body:o.body,prompts:o.lines,chapter:ch?`${ch[0]} ${ch[1]}`:'',notebook:anchor?nbLink(anchor,ch?.[0]):null,
  notes:{ask:o.ask,expected:o.expected,point:o.point||'',misconception:o.misconception||'',transition:o.transition||'',demo:o.demo||''},
  source:o.source||'',optional:!!o.optional});
}
/** An optional slide: off the main route, shown in the reading view and the PDF. */
const O=(id,title,o)=>S(id,'optional',title,{nb:'optional',...o,optional:true});
const ACTS=['Why generate?','Autoencoders','Can it generate?','Three fixes','Why: the ELBO','Putting it together'];
const acts=k=>`<div class="act-strip" role="img" aria-label="Lecture map: part ${k} of 6">${ACTS.map((a,i)=>`<span class="${i+1<k?'done':i+1===k?'now':''}">${i+1} · ${a}</span>`).join('')}</div>`;
const act=(k,...a)=>`${acts(k)}<div class="act-body">${a.join('')}</div>`;
const partOpeners=[];
/** A full opening slide for a part: its question, a diagram, and its sections. The section list is filled in at the end. */
function partOpener(k,section,lead,ask,expected){
 S(`part-${k}`,section,`Part ${k} · ${PARTS[k-1][1]}`,{part:true,nb:`part-${k}`,body:'',lines:[lead],ask:[ask],expected:[expected],transition:`Section ${k}.1.`});
 partOpeners.push([slides.at(-1),k]);
}

// ---------------------------------------------------------------- figures
/** image → encoder → code → decoder → image, for the autoencoder or the VAE (Lecture 2's diagram). */
function aeDiagram({kind='ae',stage=9,showOut=true}={}){
 const W=1100,H=kind==='vae'?330:280,y0=kind==='vae'?60:40,mid=y0+100,encC=palette.p,decC=encC;
 const T={ae:[[`encoder ${sub('g','φ')}`,'weights φ · 784 → 400 → 2'],[`decoder ${sub('f','θ')}`,'weights θ · 2 → 400 → 784']],
  vae:[['encoder','weights φ · 784 → 400 → 4'],[`decoder ${sub('f','θ')}`,'weights θ · 2 → 400 → 784']]}[kind];
 const [encT,decT]=T;
 const trap=(x,wide,c,t)=>{const h1=wide?200:80,h2=wide?80:200,x2=x+170,sub_=t[1].split(' · ');return `<polygon points="${x},${mid-h1/2} ${x2},${mid-h2/2} ${x2},${mid+h2/2} ${x},${mid+h1/2}" fill="${c}14" stroke="${c}" stroke-width="2.5"/>`+label(x+85,mid-10,t[0],{fill:c,size:22,anchor:'middle',weight:600})+sub_.map((u,i)=>label(x+85,mid+16+i*20,u,{size:15,anchor:'middle'})).join('');};
 let s=dimg('example',0,10,mid-60,120)+label(70,mid+88,'x: 784 numbers',{size:18,anchor:'middle'});
 if(stage>=2)s+=trap(150,true,encC,encT);
 if(kind==='vae'){
  if(stage>=2)s+=`<rect x="350" y="${mid-88}" width="96" height="54" rx="8" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(398,mid-53,'μ (2)',{fill:palette.q,size:21,anchor:'middle',weight:600})+`<rect x="350" y="${mid+34}" width="96" height="54" rx="8" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(398,mid+69,'log σ² (2)',{fill:palette.q,size:19,anchor:'middle',weight:600});
  if(stage>=3)s+=`<circle cx="520" cy="${mid+130}" r="34" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/>`+label(520,mid+138,'ε',{fill:palette.noise,size:26,anchor:'middle',weight:600})+label(574,mid+150,'ε ~ N(0, I)',{fill:palette.noise,size:17})+
   `<rect x="480" y="${mid-36}" width="130" height="72" rx="10" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/>`+label(545,mid+8,'z = μ + σ ⊙ ε',{fill:palette.noise,size:19,anchor:'middle',weight:600})+
   `<line x1="446" y1="${mid-60}" x2="480" y2="${mid-20}" stroke="${palette.ink}" stroke-width="2"/><line x1="446" y1="${mid+60}" x2="480" y2="${mid+20}" stroke="${palette.ink}" stroke-width="2"/><line x1="520" y1="${mid+96}" x2="530" y2="${mid+36}" stroke="${palette.noise}" stroke-width="2"/>`;
  if(stage>=4){s+=trap(640,false,decC,decT);if(showOut)s+=dimg(stage>=5?'vae2_mean_recon3':'example',0,840,mid-60,120)+label(900,mid+88,'x̂: the mean image',{size:18,anchor:'middle'});}
 }else{
  if(stage>=3)s+=`<rect x="334" y="${mid-36}" width="142" height="72" rx="10" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2.5"/>`+label(405,mid+8,'z: 2 numbers',{fill:palette.q,size:19,anchor:'middle',weight:600});
  if(stage>=4)s+=trap(490,false,decC,decT);
  if(stage>=4&&showOut)s+=dimg('ae2_recon3',0,690,mid-60,120)+label(750,mid+88,'x̂: rebuilt',{size:18,anchor:'middle'});
 }
 return fig(W,H,s,'An image goes through an encoder to a short code and back through a decoder.');
}
/** Generation in two steps: sample a code, decode it. */
function genDiagram(kind='ae'){
 const W=1100,H=190,dec=kind==='ae'?['decoder','the trained network',palette.p]:['decoder','fθ(z), learned',palette.p];
 const prior=['z ~ N(0, I)',kind==='ae'?'a guess':'chosen in advance'];
 const box=(x,w,t,s_,c)=>`<rect x="${x}" y="50" width="${w}" height="110" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,97,t,{fill:c,size:24,anchor:'middle',weight:600})+label(x+w/2,129,s_,{size:19,anchor:'middle'});
 const arrow=(x1,x2)=>`<line x1="${x1}" y1="105" x2="${x2-8}" y2="105" stroke="${palette.ink}" stroke-width="2.5"/><path d="M${x2},105 l-12,-7 v14 z" fill="${palette.ink}"/>`;
 return fig(W,H,label(150,30,'step 1: sample a code',{fill:palette.ink,size:20,anchor:'middle',weight:600})+box(20,260,prior[0],prior[1],palette.data)+arrow(280,420)+label(540,30,'step 2: turn it into an image',{fill:palette.ink,size:20,anchor:'middle',weight:600})+box(420,240,dec[0],dec[1],dec[2])+arrow(660,800)+`<image href="${digitSrc(kind==='ae'?'ae2_prior_samples':'vae2_prior_samples',0)}" x="800" y="40" width="130" height="130" style="image-rendering:pixelated"/><rect x="800" y="40" width="130" height="130" fill="none" stroke="${LINE}"/>`+label(960,110,'a new digit?',{size:20}),'Generation in two steps: sample a code, then decode it into an image.');
}
/** Lecture 1's budget: log p(𝒟) = ELBO + KL to the posterior. With one=true, the same for one image. */
function budgetFig(one=false){
 const W=1150,H=250,x0=60,w=1030,e=x0+w,split=x0+w*.72;
 const s=`<rect x="${x0}" y="110" width="${split-x0}" height="64" fill="${palette.q}24" stroke="${palette.q}" stroke-width="2.5"/><rect x="${split}" y="110" width="${e-split}" height="64" fill="${palette.loss}2E" stroke="${palette.loss}" stroke-width="2.5"/>`+
  label((x0+split)/2,152,one?'ELBO(x): we can compute it':'ELBO(q): we can compute it',{fill:palette.q,size:24,anchor:'middle',weight:700})+label((split+e)/2,152,'KL to the posterior: ≥ 0',{fill:palette.loss,size:21,anchor:'middle',weight:700})+
  `<path d="M${x0},92 L${x0},78 L${e},78 L${e},92" fill="none" stroke="${palette.data}" stroke-width="2.5"/>`+label((x0+e)/2,64,one?'log p(x): the log evidence of one image':'log p(𝒟): one fixed number, the log evidence',{fill:palette.data,size:22,anchor:'middle',weight:700})+
  label((x0+e)/2,222,'raise the ELBO, and the KL to the posterior must fall',{fill:palette.ink,size:21,anchor:'middle'});
 return fig(W,H,s,'The log evidence splits into the ELBO, which we can compute, and the KL to the posterior, which we cannot.');
}
/** Lecture 1's pieces above, today's below: the same roles with new names. */
function castFig(){
 const W=1150,H=280,w=232,xs=[150,400,650,900],y1=36,y2=176;
 const chip=(x,y,t,c)=>`<rect x="${x}" y="${y}" width="${w}" height="50" rx="25" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,y+32,t,{fill:c,size:19,anchor:'middle',weight:600});
 const top=[[`prior p(θ)`,GREEN],[`likelihood p(𝒟 | θ)`,palette.noise],[`guide ${sub('q','ψ')}(θ)`,palette.q],['maximise the ELBO',palette.ink]];
 const bot=[[`prior p(z) = N(0, I)`,GREEN],[`decoder ${sub('p','θ')}(x | z)`,palette.noise],[`encoder ${sub('q','φ')}(z | x)`,palette.q],['maximise the ELBO',palette.ink]];
 let s=label(10,y1+32,'Lecture 1',{fill:palette.ink,size:20,weight:650})+label(10,y2+32,'today',{fill:palette.ink,size:20,weight:650});
 xs.forEach((x,i)=>{s+=chip(x,y1,...top[i])+chip(x,y2,...bot[i])+`<line x1="${x+w/2}" y1="${y1+56}" x2="${x+w/2}" y2="${y2-12}" stroke="${MUTED}" stroke-width="2" stroke-dasharray="5 5"/><path d="M${x+w/2},${y2-4} l-6,-11 h12 z" fill="${MUTED}"/>`;});
 s+=label(150,y2+80,'swap the names: the weights θ become a code z for each image, and the data 𝒟 become one image x',{size:18});
 return fig(W,H,s,'Lecture 1’s prior, likelihood, guide and objective, and today’s: prior N(0, I), decoder, encoder, the same objective.');
}
/** The VAE as a graphical model (Lecture 2). */
function vaePgm(stage){
 const W=1100,H=300;let g=`<rect x="60" y="16" width="560" height="250" rx="10" fill="none" stroke="${MUTED}" stroke-width="2"/>`+label(608,254,'60,000 images',{size:17,anchor:'end'});
 g+=`<circle cx="170" cy="120" r="46" fill="${palette.paper}" stroke="${GREEN}" stroke-width="3"/>`+label(170,130,'z',{fill:GREEN,size:30,anchor:'middle',weight:650})+label(170,196,'code (hidden)',{size:17,anchor:'middle'});
 g+=label(650,52,'the prior:',{fill:GREEN,size:20,weight:650})+label(650,80,'p(z) = N(0, I), 2 numbers, fixed in advance',{fill:GREEN,size:19});
 if(stage>=2){g+=`<line x1="216" y1="120" x2="414" y2="120" stroke="${palette.ink}" stroke-width="2.5"/><path d="M424,120 l-14,-7 v14 z" fill="${palette.ink}"/>`+label(318,104,'decoder network',{size:17,anchor:'middle'})+
  `<circle cx="470" cy="120" r="46" fill="#D9DEE5" stroke="${palette.data}" stroke-width="3"/>`+label(470,130,'x',{fill:palette.data,size:30,anchor:'middle',weight:650})+label(470,196,'image (observed)',{size:17,anchor:'middle'});
  g+=label(650,132,'the likelihood:',{fill:palette.noise,size:20,weight:650})+label(650,160,'p(x | z) = N(x; decoder(z), σ² I)',{fill:palette.noise,size:19})+label(650,186,'each pixel: the decoder’s value + noise of size σ',{size:17});}
 if(stage>=3){g+=`<circle cx="318" cy="290" r="9" fill="${palette.ink}"/>`+label(332,296,'θ: the decoder’s weights, shared by all images (learned)',{size:17})+`<line x1="318" y1="281" x2="318" y2="142" stroke="${palette.ink}" stroke-width="2" stroke-dasharray="4 4"/>`;}
 return fig(W,H+12,g,'The VAE as a graphical model: a hidden code z with prior N(0, I), an arrow through the decoder to the observed image x, decoder weights θ shared.');
}
/** The graphical model with the dashed inference arrow: the encoder. */
function inferPgm(stage){
 const W=1100,H=290;let g=`<rect x="60" y="16" width="520" height="240" rx="10" fill="none" stroke="${MUTED}" stroke-width="2"/>`+label(568,244,'60,000 images',{size:17,anchor:'end'});
 g+=`<circle cx="170" cy="120" r="46" fill="${palette.paper}" stroke="${GREEN}" stroke-width="3"/>`+label(170,130,'z',{fill:GREEN,size:30,anchor:'middle',weight:650})+
  `<circle cx="450" cy="120" r="46" fill="#D9DEE5" stroke="${palette.data}" stroke-width="3"/>`+label(450,130,'x',{fill:palette.data,size:30,anchor:'middle',weight:650})+
  `<line x1="216" y1="104" x2="394" y2="104" stroke="${palette.ink}" stroke-width="2.5"/><path d="M404,104 l-14,-7 v14 z" fill="${palette.ink}"/>`+label(310,90,`${sub('p','θ')}(x | z): decoder`,{fill:palette.noise,size:17,anchor:'middle'});
 if(stage>=2)g+=`<path d="M404,140 Q310,190 222,142" fill="none" stroke="${palette.q}" stroke-width="2.5" stroke-dasharray="8 6"/><path d="M216,138 l14,0 l-6,12 z" fill="${palette.q}"/>`+label(310,206,`${sub('q','φ')}(z | x): encoder`,{fill:palette.q,size:18,anchor:'middle',weight:650});
 g+=label(630,46,'the posterior p(z | x): which codes made this x?',{fill:palette.p,size:19,weight:650})+label(630,74,'it needs p(x), the integral of Section 5.3',{size:17});
 if(stage>=2)g+=label(630,124,'approximate it, as in Lecture 1, with',{fill:palette.q,size:19,weight:650})+label(630,152,`${sub('q','φ')}(z | x) = N(${sub('μ','φ')}(x), ${sub('σ','φ')}(x)²), a Gaussian whose`,{fill:palette.q,size:17})+label(630,176,'mean and spread the encoder computes from x',{fill:palette.q,size:17});
 if(stage>=3)g+=label(630,226,'train θ and φ together: maximise the ELBO',{fill:palette.ink,size:19,weight:650})+label(630,252,'summed over the 60,000 images',{size:17});
 return fig(W,H,g,'The VAE graphical model with the dashed inference arrow: the encoder q_φ(z | x) approximates the posterior.');
}
/** Sketch: reconstruction alone, KL alone, and both together (Lecture 2). */
function tugFig(){
 const W=1100,H=300,PW=340,R_=70;
 const frame=(ox,title,col)=>`<rect x="${ox}" y="40" width="${PW}" height="210" rx="8" fill="#fff" stroke="${LINE}"/>`+label(ox,26,title,{fill:col,size:19,weight:650})+`<circle cx="${ox+PW/2}" cy="145" r="${R_}" fill="none" stroke="${MUTED}" stroke-width="1.8" stroke-dasharray="6 5"/>`+label(ox+PW/2+R_*.72+4,145-R_*.72-4,'N(0, I)',{size:14});
 const cloud=(ox,dx,dy,rx,ry,k)=>`<ellipse cx="${ox+PW/2+dx}" cy="${145+dy}" rx="${rx}" ry="${ry}" fill="${CLASS[k]}2E" stroke="${CLASS[k]}" stroke-width="2"/>`+`<circle cx="${ox+PW/2+dx}" cy="${145+dy}" r="3.5" fill="${CLASS[k]}"/>`;
 let g='';
 g+=frame(0,'reconstruction alone',palette.noise)+cloud(0,-120,-70,4,4,0)+cloud(0,125,-60,4,4,1)+cloud(0,95,75,4,4,3)+cloud(0,-110,70,4,4,0)+
  label(0,276,'tiny clouds, far apart:',{size:16})+label(0,296,'an autoencoder again',{size:16});
 g+=`<g data-at="2">${frame(380,'KL alone',palette.q)+cloud(380,0,0,R_,R_,0)+cloud(380,3,-2,R_,R_,1)+cloud(380,-3,2,R_,R_,3)+label(380,276,'every image gets the same cloud N(0, I):',{size:16})+label(380,296,'z says nothing about the image',{size:16})}</g>`;
 g+=`<g data-at="3">${frame(760,'both together',palette.p)+cloud(760,-34,22,24,16,0)+cloud(760,36,14,22,16,1)+cloud(760,2,-34,24,15,3)+
  `<path d="M${760+PW/2+14},${145-10} l3,8 9,0 -7,5 3,9 -8,-6 -8,6 3,-9 -7,-5 9,0z" fill="${palette.loss}" stroke="#fff"/>`+label(760,276,'clouds near the centre, still apart:',{size:16})+label(760,296,'a random z (star) lands near some cloud',{size:16})}</g>`;
 return fig(W,H+10,g,'Sketch: reconstruction alone gives tiny separated clouds; KL alone gives identical N(0, I) clouds; together, clouds sit near the centre but remain distinct.');
}
/** Sketch: points for the autoencoder, overlapping clouds for the VAE. */
function cloudsFig(){
 const W=1100,H=330,PW=520;
 const draw=(ox,isVae)=>{const P=panel({ox,W:PW,H,L:30,R:10,T:40,B:30,xMin:-4,xMax:4,yMin:-3,yMax:3,xAxis:false,yAxis:false});
  const pts=[[-2.2,1.1,0],[-1.6,1.6,0],[1.8,-.6,1],[2.3,-1.3,1],[.2,-1.9,2],[-.5,-1.5,2]],col=[CLASS[0],CLASS[1],CLASS[3]];
  let g=`<rect x="${ox+30}" y="40" width="${PW-40}" height="${H-70}" fill="#fff" stroke="${LINE}"/>`+label(ox+30,26,isVae?'VAE: each image → a small cloud':'autoencoder: each image → one point',{fill:isVae?palette.q:palette.p,size:20,weight:650});
  if(isVae)g+=`<circle cx="${P.x(0)}" cy="${P.y(0)}" r="${(P.x(2)-P.x(0)).toFixed(1)}" fill="none" stroke="${MUTED}" stroke-dasharray="6 5" stroke-width="2"/>`+label(P.x(0)+(P.x(2)-P.x(0))*.72,P.y(0)-(P.x(2)-P.x(0))*.72-6,'N(0, I)',{size:15});
  pts.forEach(([a,b,k])=>{const [x,y]=isVae?[a*.6,b*.6]:[a,b];
   g+=(isVae?`<ellipse cx="${P.x(x)}" cy="${P.y(y)}" rx="46" ry="30" fill="${col[k]}33" stroke="${col[k]}" stroke-width="2"/>`:'')+`<circle cx="${P.x(x)}" cy="${P.y(y)}" r="6" fill="${col[k]}"/>`;});
  return g+label(ox+40,H-8,isVae?'clouds overlap: the space between images is trained':'points: the space between them is never trained',{size:16});};
 return fig(W,H,draw(0,false)+`<g data-at="2">${draw(570,true)}</g>`,'Sketch: an autoencoder maps each image to a point; a VAE maps it to a cloud, and the clouds overlap.');
}
/** Our 3, its reconstruction and the squared error of every pixel, with three pixels marked. */
const LOSS_PX=[[24,10,'missed ink'],[5,11,'extra ink'],[3,24,'background']];
function lossFig(){
 const X=imgs('example').px,XH=imgs('ae2_recon3').px,S=214,gx=[6,292,578],y=46,cell=S/28;
 const err=Float32Array.from(X,(v,i)=>((v-XH[i])/255)**2),mx=Math.max(...err);
 const c=document.createElement('canvas');c.width=c.height=28;const g=c.getContext('2d'),im=g.createImageData(28,28);
 for(let i=0;i<784;i++){const t=Math.sqrt(err[i]/mx);im.data.set([...[250,250,251].map((v,k)=>Math.round(v+([190,18,60][k]-v)*t)),255],i*4);}
 g.putImageData(im,0,0);const errURL=c.toDataURL();
 const frame=(x,col)=>`<rect x="${x-2}" y="${y-2}" width="${S+4}" height="${S+4}" fill="none" stroke="${col}" stroke-width="3" rx="3"/>`;
 let s=`<image href="${digitSrc('example',0)}" x="${gx[0]}" y="${y}" width="${S}" height="${S}" style="image-rendering:pixelated"/>`+frame(gx[0],palette.p)+label(gx[0],30,'target x: our 3',{fill:palette.p,size:20,weight:650})+
  `<image href="${digitSrc('ae2_recon3',0)}" x="${gx[1]}" y="${y}" width="${S}" height="${S}" style="image-rendering:pixelated"/>`+frame(gx[1],palette.q)+label(gx[1],30,'prediction x̂',{fill:palette.q,size:20,weight:650})+
  `<image href="${errURL}" x="${gx[2]}" y="${y}" width="${S}" height="${S}" style="image-rendering:pixelated"/>`+frame(gx[2],palette.loss)+label(gx[2],30,'(x − x̂)², each pixel',{fill:palette.loss,size:20,weight:650});
 LOSS_PX.forEach(([r,cc],k)=>gx.forEach((x,j)=>{const cx=x+(cc+.5)*cell,cy=y+(r+.5)*cell;s+=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${(cell*1.7).toFixed(1)}" fill="none" stroke="${palette.ink}" stroke-width="2.5"/>`+(j===2?label(cx+(cc>20?-cell*2.2:cell*1.9),cy-cell*1.3,'①②③'[k],{fill:palette.ink,size:19,weight:700,anchor:cc>20?'end':'start',extra:'style="paint-order:stroke;stroke:#fff;stroke-width:5px"'}):'');}));
 s+=label(gx[2]+S/2,y+S+34,`sum of all 784 = ${f(A.ae2_sse3,1)}`,{fill:palette.loss,size:20,anchor:'middle',weight:650});
 return fig(800,310,s,'Our 3, the 2-number autoencoder’s reconstruction, and the squared error of each pixel; three pixels marked.');
}
function lossTable(){
 const X=imgs('example').px,XH=imgs('ae2_recon3').px;
 const r_=LOSS_PX.map(([r,c,what],k)=>{const a=X[r*28+c]/255,b=XH[r*28+c]/255;return [`${'①②③'[k]} ${what}`,`<span class="p">${f(a,2)}</span>`,`<span class="q">${f(b,2)}</span>`,`<span class="loss">${f((a-b)**2,2)}</span>`];});
 return `<div class="table-sm">${table(['pixel','<span class="p">x</span>','<span class="q">x̂</span>','<span class="loss">(x − x̂)²</span>'],[...r_,['all 784 pixels','','',`<b class="loss">${f(A.ae2_sse3,1)}</b>`]])}</div>`;
}
/** A patch of our 3's pixels as numbers (rows 6–15, columns 8–18). */
function pixelTable(){
 const P=M.pixels3;let s='<table class="pix-table">';
 for(let r=6;r<16;r++){s+='<tr>';for(let c=8;c<19;c++){const v=P[r][c];s+=`<td style="background:rgba(20,23,31,${(v*.85).toFixed(2)});color:${v>.5?'#fff':MUTED}">${v===0?'0':f(v,1)}</td>`;}s+='</tr>';}
 return s+'</table>';
}
/** 500 draws from the 2-D standard normal, with circles of radius 1 and 2. */
function stdNormalFig(){
 const pts=G.prior_points,P=panel({W:440,H:420,L:46,R:12,T:16,B:46,xMin:-4,xMax:4,yMin:-4,yMax:4,ticksX:[-4,-2,0,2,4],ticksY:[-4,-2,0,2,4],fs:17,xAxisAtBottom:true});
 let s=`<rect x="46" y="16" width="382" height="358" fill="#fff" stroke="${LINE}"/>`+P.axes;
 s+=pts.map(([a,b])=>`<circle cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="3.4" fill="${palette.loss}" fill-opacity=".6"/>`).join('');
 const sx=P.x(1)-P.x(0);for(const k of [1,2])s+=`<circle cx="${P.x(0)}" cy="${P.y(0)}" r="${k*sx}" fill="none" stroke="${palette.ink}" stroke-dasharray="6 5" stroke-width="2"/>`+label(P.x(0)+k*sx*.72+4,P.y(0)-k*sx*.72-4,`radius ${k}`,{fill:palette.ink,size:16,weight:650,extra:'style="paint-order:stroke;stroke:#fff;stroke-width:5px"'});
 return s;
}
const inside=r=>G.prior_points.filter(([a,b])=>a*a+b*b<=r*r).length/G.prior_points.length;
/** A small line chart with a direct label. */
function lineChart({series,W=640,H=340,xMin,xMax,yMin,yMax,ticksX,ticksY,xLabel,yLabel,ox=0,L=70}){
 const P=panel({ox,W,H,L,R:120,T:40,B:52,xMin,xMax,yMin,yMax,ticksX,ticksY,xLabel,fs:18,xAxisAtBottom:true});
 let s=P.axes+(yLabel?label(ox+L,22,yLabel,{size:18}):'');
 for(const sr of series)s+=P.line(sr.x,sr.y,sr.color);
 return s;
}
const AE_LIM=(()=>{const c=A.ae2_codes,xs=c.map(v=>v[0]),ys=c.map(v=>v[1]),pad=(a,b)=>[Math.floor(a-1.5),Math.ceil(b+1.5)];return [...pad(Math.min(...xs),Math.max(...xs)),...pad(Math.min(...ys),Math.max(...ys))];})();
const wideMark=()=>{const W_=posteriorSVG({S:400,prior:true}),P_=W_.P,cx=P_.x(V.grid.post_mean[0]),cy=P_.y(V.grid.post_mean[1]);return W_.s+`<circle cx="${cx}" cy="${cy}" r="14" fill="none" stroke="${palette.loss}" stroke-width="2.5"/>`+label(cx,cy+40,'the posterior: a tiny patch',{fill:palette.loss,size:17,weight:600,anchor:'middle'});};
const RF=V.realfake_is_fake,rfMarks=RF.map(x=>x?palette.loss:palette.p);
/** Randomness in, a learned network, a sample out; optionally a condition (Lecture 2). */
function recipeDiagram(stage=1){
 const W=1100,H=stage>=2?300:230;let s='';
 const box=(x,y,w,h,t,s_,c)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,y+h/2+(s_?-4:9),t,{fill:c,size:26,anchor:'middle',weight:600})+(s_?label(x+w/2,y+h/2+24,s_,{size:19,anchor:'middle'}):'');
 const arrow=(x1,y1,x2,y2,c=palette.ink)=>`<line x1="${x1}" y1="${y1}" x2="${x2-8}" y2="${y2}" stroke="${c}" stroke-width="2.5"/><path d="M${x2},${y2} l-12,-7 v14 z" fill="${c}"/>`;
 s+=box(20,60,250,100,'random numbers','z ~ N(0, I)',palette.noise)+arrow(270,110,400,110)+box(400,50,300,120,'a learned network','weights θ, fitted to data',palette.p)+arrow(700,110,830,110)+box(830,60,250,100,'a sample x','text, image, sound…',palette.data);
 if(stage>=2)s+=box(400,214,300,72,'a condition c','prompt, label, sketch…',palette.q)+arrow(550,214,550,176,palette.q);
 return fig(W,H,s,'Random numbers go into a learned network, which outputs a sample; optionally a condition steers it.');
}
/** A language model's probabilities for the next word (Lecture 2's SmolLM2 numbers). */
function nextWordChart(){
 const top=NW.top,W=1000,H=360,maxP=.2,x0=230,w=640;let s=label(0,26,`“${NW.prefix} …”`,{fill:palette.ink,size:22,weight:600});
 top.forEach(([t,pr],i)=>{const y=52+i*34;s+=label(x0-14,y+22,t.trim(),{fill:palette.ink,size:21,anchor:'end'})+`<rect x="${x0}" y="${y+4}" width="${(pr/maxP*w).toFixed(1)}" height="24" fill="${palette.q}" fill-opacity="${i===0?.85:.45}"/>`+label(x0+pr/maxP*w+10,y+23,`${f(100*pr,1)}%`,{size:19});});
 s+=label(x0,52+top.length*34+24,`every other word in the vocabulary: ${f(100*NW.rest,0)}% in total`,{size:19});
 return fig(W,H,s,'Probabilities of the next word after a prompt, from a small language model.');
}
/** The waveform of a generated sentence of speech. */
function waveFig(){
 const W=1100,H=230,P=panel({W,H,L:20,R:14,T:20,B:46,xMin:0,xMax:WAVE.seconds,yMin:-1,yMax:1,ticksX:[0,2,4,6],yAxis:false,xLabel:'seconds',fs:18});
 const n=WAVE.max.length,xs=lin(0,WAVE.seconds,n);
 const pts=xs.map((x,i)=>`${P.x(x).toFixed(1)},${P.y(WAVE.max[i]).toFixed(1)}`).join(' ')+' '+xs.map((x,i)=>`${P.x(xs[n-1-i]).toFixed(1)},${P.y(WAVE.min[n-1-i]).toFixed(1)}`).join(' ');
 return fig(W,H,P.axes+`<polygon points="${pts}" fill="${palette.p}" fill-opacity=".55"/>`,'The waveform of a generated sentence of speech.');
}
const threes=M.display_labels.map((l,i)=>l===3?i:-1).filter(i=>i>=0);
const mean3=(()=>{const s_=imgs('display'),out=new Float32Array(784);for(const i of threes)for(let k=0;k<784;k++)out[k]+=s_.px[i*784+k]/255/threes.length;return out;})();
/** Part 4 in one picture: three fixes, then a loss. */
function fixesFig(){
 const W=1150,H=230,w=250,h=96,y=40;
 const box=(x,t1,t2,c)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${palette.paper}" stroke="${c}" stroke-width="2.5"/>`+label(x+w/2,y+40,t1,{fill:c,size:21,anchor:'middle',weight:650})+label(x+w/2,y+70,t2,{size:18,anchor:'middle'});
 const arrow=(x1,x2)=>`<line x1="${x1}" y1="${y+h/2}" x2="${x2-8}" y2="${y+h/2}" stroke="${palette.ink}" stroke-width="2.5"/><path d="M${x2},${y+h/2} l-12,-7 v14 z" fill="${palette.ink}"/>`;
 const s=box(0,'① where codes live','z ~ N(0, I)',GREEN)+arrow(w,w+40)+box(w+40,'② a cloud per image','z = μ + σ ⊙ ε',palette.noise)+arrow(2*w+40,2*w+80)+box(2*w+80,'③ pull clouds to N(0, I)','a KL penalty',palette.q)+arrow(3*w+80,3*w+120)+
  `<rect x="${3*w+120}" y="${y}" width="${W-3*w-122}" height="${h}" rx="12" fill="${palette.loss}10" stroke="${palette.loss}" stroke-width="2.5"/>`+label((3*w+120+W)/2,y+40,'a loss',{fill:palette.loss,size:21,anchor:'middle',weight:650})+label((3*w+120+W)/2,y+70,'rebuild + KL',{size:18,anchor:'middle'})+
  label(W/2,y+h+52,'Part 5: this loss is exactly minus Lecture 1’s ELBO, one image at a time',{fill:palette.ink,size:20,anchor:'middle'});
 return fig(W,H,s,'Three fixes to the autoencoder (codes from N(0, I), a cloud per image, a KL pull towards N(0, I)) lead to a loss: rebuild plus KL.');
}
const klq=(m_,s_)=>.5*(m_*m_+s_*s_-Math.log(s_*s_)-1);
/** The VAE's loss with every term traced to where it came from. */
function lossMap(){
 const W=1150,H=420,A=[40,520],B=[630,500],y0=58;
 const chip=(x,y,w,t,src,c)=>`<rect x="${x}" y="${y}" width="${w}" height="40" rx="8" fill="${palette.paper}" stroke="${c}" stroke-width="1.8"/>`+label(x+14,y+27,t,{fill:palette.ink,size:18})+label(x+w-12,y+27,src,{fill:c,size:16,anchor:'end',weight:650});
 const branch=(x,ys,c)=>`<line x1="${x}" y1="${y0+70}" x2="${x}" y2="${ys.at(-1)+20}" stroke="${c}" stroke-width="2"/>`+ys.map(y=>`<line x1="${x}" y1="${y+20}" x2="${x+20}" y2="${y+20}" stroke="${c}" stroke-width="2"/>`).join('');
 let s=label(A[0],y0-18,'reconstruction',{fill:palette.noise,size:19,weight:650})+`<rect x="${A[0]}" y="${y0}" width="${A[1]}" height="70" rx="10" fill="${palette.noise}10" stroke="${palette.noise}" stroke-width="2.5"/>`+label(A[0]+A[1]/2,y0+45,'‖x − f(μ + σ ⊙ ε)‖² / 2σ²',{fill:palette.noise,size:27,anchor:'middle',weight:600})+
  label(A[0]+A[1]+35,y0+46,'+',{fill:palette.ink,size:34,anchor:'middle'})+
  label(B[0],y0-18,'KL(q(z | x) ‖ N(0, I))',{fill:palette.q,size:19,weight:650})+`<rect x="${B[0]}" y="${y0}" width="${B[1]}" height="70" rx="10" fill="${palette.q}10" stroke="${palette.q}" stroke-width="2.5"/>`+label(B[0]+B[1]/2,y0+45,'½ Σ (μ² + σ² − log σ² − 1)',{fill:palette.q,size:27,anchor:'middle',weight:600});
 const ys=[150,200,250];
 s+=`<g data-at="2">${branch(A[0]+16,ys,palette.noise)+chip(A[0]+36,ys[0],A[1]-36,'Gaussian pixels: it is −log p(x | z) + const','5.2',palette.noise)+chip(A[0]+36,ys[1],A[1]-36,'the autoencoder’s squared error, rescaled','2.3',palette.noise)+chip(A[0]+36,ys[2],A[1]-36,'z = μ + σ ⊙ ε: Fix 2, tool ①','Lecture 1, 3.5',palette.noise)}</g>`;
 s+=`<g data-at="3">${branch(B[0]+16,ys,palette.q)+chip(B[0]+36,ys[0],B[1]-36,'Fix 3; the formula is tool ②','Lecture 1, 2.6',palette.q)+chip(B[0]+36,ys[1],B[1]-36,'to the prior p(z) = N(0, I): Fix 1','4.1',palette.q)+chip(B[0]+36,ys[2],B[1]-36,'a KL: extra cost, never negative','Lecture 1, 2.4',palette.q)}</g>`;
 s+=`<g data-at="4"><rect x="${A[0]}" y="318" width="${B[0]+B[1]-A[0]}" height="76" rx="12" fill="${palette.data}0F" stroke="${palette.data}" stroke-width="2.5"/>`+
  label((A[0]+B[0]+B[1])/2,348,'together: −ELBO(x) + const, Lecture 1’s objective for one image (5.4)',{fill:palette.ink,size:21,anchor:'middle',weight:650})+
  label((A[0]+B[0]+B[1])/2,378,'a smaller loss means a higher lower bound on log p(x), and q closer to the posterior',{size:19,anchor:'middle'})+`</g>`;
 return fig(W,H,s,'The VAE loss: a reconstruction term and a KL term, each traced to the slide or Lecture 1 section it came from, and together minus the ELBO.');
}

// ================================================================= Part 1 · Why generate?
partOpener(1,'why','Generative models make new text, pictures and sound. A quick tour, what they are good for, and the one recipe they all share.',
 'What do a chatbot, an image generator and a weather model have in common?','Each turns random numbers into a new sample, through a network fitted to data.');
chap('1.1');
S('faces','why','Who are these people?',{builds:2,
 body:stack(`<figure class="evidence"><img src="${L2A('assets/gallery/stylegan-faces.jpg')}" alt="A grid of photographs of faces" style="max-height:330px"></figure>`,at(2,p('None of them exist. A network turned random numbers into each face.'))),
 lines:['Look closely: ages, hair, glasses, backgrounds. Who are these people?','Nobody. Each face is a sample from a generative model (StyleGAN, 2019): random numbers in, a face out.'],
 ask:['Who are these people? Would you have known?','What did the network need, to make these?'],
 expected:['Students guess celebrities or stock photos. None exist.','Many real photographs to learn from, and a way to turn random numbers into new ones. That is today’s subject.'],
 transition:'A smaller test, on the digits we will use all lecture.',
 source:'Faces generated by StyleGAN: Karras, Laine and Aila, “A Style-Based Generator Architecture for GANs”, CVPR 2019. NVIDIA, CC BY-NC 4.0.'});
S('realfake','why','Real or generated? Six of these twelve digits are fake',{builds:2,
 body:cols(`<div class="state-slot"><div data-only="1">${digits('realfake',{cols:6,size:104,gap:14})}</div><div data-only="2">${digits('realfake',{cols:6,size:104,gap:14,marks:rfMarks,labels:RF.map(x=>x?'<span class="tag fake">generated</span>':'<span class="tag real">real</span>')})}</div></div>`,
  stack(p('Six are handwritten MNIST test digits. Six come from a network.'),at(2,p(`The fakes: numbers ${RF.map((x,i)=>x?i+1:null).filter(Boolean).join(', ')}, counting left to right, top to bottom.`)),at(2,small('By the end of today you will know how the network that made them was trained: a VAE with 16 numbers per code.'))),'1.55fr 1fr'),
 lines:['Six are real, six are generated. Write down which six you think are fake.','The six outlined in red are generated. Some give themselves away with broken strokes; some fool most people.'],
 ask:['Which six are fake? Write the positions.','Which fakes gave themselves away, and how?'],
 expected:['Answers vary; take a quick show of hands for two or three digits.','Broken or doubled strokes and blurry edges give some away; the cleanest fakes are hard to tell.'],
 point:'Real digits: six random MNIST test images. Fakes: the first six draws z ~ N(0, I) through Lecture 2’s 16-number VAE, not chosen by hand.',
 transition:'Generative models are everywhere now: a quick tour, then the common idea.',source:OURS});
chap('1.2');
S('llm','why','A chatbot is a generative model: one prompt, three different answers',{builds:2,
 body:stack(`<div class="prompt-box"><b>Prompt:</b> Write one sentence that opens a short story about the first monsoon rain in Gandhinagar.</div>`,
  `<div class="cards" style="grid-template-columns:1fr 1fr 1fr">${LLM.map((t,i)=>(i===0?'':'<div data-at="2">')+`<div class="sample-text"><b>Sample ${i+1}.</b> ${t.trim()}</div>`+(i===0?'':'</div>')).join('')}</div>`),
 lines:['We sent the same prompt to a language model three times. Here is the first reply.','Same prompt, three different replies. Each is a sample from a distribution over sentences.'],
 ask:['If we send the same prompt again, do we get the same sentence?','Why would we want different answers each time?'],
 expected:['No: the model samples, so each run differs.','Variety: one prompt has many good answers. A model that always gave the single most likely sentence would be repetitive.'],
 transition:'How does it choose? One word at a time.',
 source:'Generated for Lecture 2 on 21 October 2026 with Gemini 3.5 Flash, temperature 1 (<code>lecture2/assets/examples/provenance.json</code>).'});
S('next-word','why','Under the hood: a probability for every possible next word',{builds:2,
 body:cols(nextWordChart(),stack(at(2,p('Sampling = pick a word with these probabilities, append it, repeat.')),at(2,small(`The same model’s three continuations: “${NW.samples.map(x=>x.split('\n')[0].trim()).join('”, “')}”.`))),'1.45fr 1fr'),
 lines:['A small language model (SmolLM2, 360 million weights) scores every word it knows as the next word. “wet” gets 17%.','Sample a word, append it, repeat. That is all a chatbot does when it writes, at a much larger scale.'],
 ask:['Which word would the model pick if it always took the most likely one?','What is the model a distribution over?'],
 expected:['“wet”: 17%, but 83% of the time something else is sampled.','The next word given the words so far. Chaining them gives a distribution over whole sentences.'],
 point:'Probabilities are real: SmolLM2-360M-Instruct, computed locally for Lecture 2. The point is only that generation = repeated sampling.',
 transition:'Pictures: one prompt, many images.',
 source:'SmolLM2-360M-Instruct (Hugging Face), next-token probabilities and three samples at temperature 1, computed for Lecture 2.'});
S('t2i','why','Text to image: one prompt, four samples',{builds:2,
 body:stack(`<div class="prompt-box"><b>Prompt:</b> A watercolour painting of a camel walking across the white salt desert of the Rann of Kutch at sunrise, soft light, wide view.</div>`,
  `<div class="photo-row" style="grid-template-columns:repeat(4,minmax(0,1fr))">${[1,2,3,4].map(i=>(i===1?'':`<div data-at="2">`)+photo(`assets/examples/t2i-sample-${i}.jpg`,`sample ${i}`)+(i===1?'':'</div>')).join('')}</div>`),
 lines:['Same idea for pictures. One prompt, and the first picture the model returned.','Three more runs, same prompt: a rider appears, a walker, a different horizon. Four samples.'],
 ask:['What stayed the same across the four, and what changed?','Where does the variety come from?'],
 expected:['Same: camel, salt flat, sunrise, watercolour. Changed: rider, framing, sky.','From random numbers fed into the model; the prompt only narrows what is plausible.'],
 transition:'Images can also be the input.',
 source:'Generated for Lecture 2 on 21 October 2026 with Gemini 3 Pro Image (<code>lecture2/assets/examples/provenance.json</code>).'});
S('img2img','why','Image in, image out',{builds:2,
 body:`<div class="state-slot"><div data-only="1"><div class="photo-row" style="grid-template-columns:repeat(6,minmax(0,1fr))">${photo('assets/gallery/shoe-input.jpg','edges of a shoe')+photo('assets/gallery/shoe-output.jpg','pix2pix: a photo')+photo('assets/gallery/spade-layout.jpg','a label map')+photo('assets/gallery/spade-out-0.jpg','SPADE: a landscape')+photo('assets/gallery/horse.jpg','a horse')+photo('assets/gallery/zebra.jpg','CycleGAN: a zebra')}</div></div>`+
  `<div data-only="2"><div class="photo-row" style="grid-template-columns:repeat(4,minmax(0,1fr))">${photo('assets/gallery/spade-layout.jpg','one label map')+photo('assets/gallery/spade-out-0.jpg','sample 1')+photo('assets/gallery/spade-out-1.jpg','sample 2')+photo('assets/gallery/spade-out-2.jpg','sample 3')}</div></div></div>`,
 lines:['The input can be an image: a sketch becomes a photo, a coloured map becomes a landscape, a horse becomes a zebra.','The same label map, sampled three times: three different photographs. Still a distribution, now given an input.'],
 ask:['Is there one right answer for “photo of this sketch”?','So what is the model a distribution over?'],
 expected:['No: colour, texture and lighting are not in the sketch.','Over images, given the input image: p(photo | sketch).'],
 transition:'Sound works the same way.',
 source:'Isola et al., pix2pix (CVPR 2017); Park et al., SPADE (CVPR 2019); Zhu et al., CycleGAN (ICCV 2017). Attributed teaching excerpts; copyright with the authors.'});
S('speech','why','Sound: text in, 24,000 numbers per second out',{builds:2,
 body:stack(`<div class="prompt-box"><b>Text:</b> “A generative model turns a few random numbers into a new example: a sentence, a picture, or this voice.”</div>`,waveFig(),
  `<div class="audio-row"><audio controls preload="none" src="${L2A('assets/examples/tts-sample.mp3')}"></audio>${at(2,`<span class="mini">${comma(WAVE.samples)} numbers for ${f(WAVE.seconds,1)} seconds. A few of them: ${WAVE.first.slice(0,6).map(v=>f(v,3)).join(', ')}, …</span>`)}</div>`),
 lines:['A text-to-speech model read this sentence aloud. Press play.','The output is just a long list of numbers: the air pressure, 24,000 times a second. Generating sound = generating that list.'],
 ask:['What exactly does the model output?','Is there one right way to say this sentence?'],
 expected:['About 191,000 numbers: a waveform sampled 24,000 times a second.','No: speed, pitch, pauses and voice all vary. Another sample would sound different.'],
 transition:'Generative models also matter far from art and chat.',
 source:'Generated for Lecture 2 on 21 October 2026 with Gemini 3.8 Flash TTS, voice “Kore”.'});
chap('1.3');
S('science','why','Beyond pictures: generative models in science',{builds:2,
 body:`<div class="cards big" style="grid-template-columns:1fr 1fr">${card('Weather','GenCast samples 50 possible futures for the next 15 days. The spread of the samples says how sure the forecast is.',palette.data,'Price et al., Nature 2025')+card('Proteins','RFdiffusion generates new protein shapes, for example ones that bind a chosen target.',palette.p,'Watson et al., Nature 2023')}${at(2,card('Materials','MatterGen proposes new crystal structures with properties you ask for.',palette.q,'Zeni et al., Nature 2025'))}${at(2,card('Missing data','Sensors drop readings. A model of p(x) fills the gap with plausible values instead of a flat average, and says how unsure it is.',palette.noise,'for example, air-quality and smart-meter records'))}</div>`,
 lines:['Weather and proteins: the output is a sample, and many samples together describe what could happen.','Materials, and the everyday case: filling in missing sensor readings with plausible values.'],
 ask:['Why would a forecaster want 50 samples instead of one best forecast?','Why not fill a gap in a sensor record with the average value?'],
 expected:['The samples show the range of possible futures: if they disagree, the forecast is uncertain.','The average is smooth and typical-looking but may never happen, and it hides the uncertainty. A sample looks like real data.'],
 transition:'Collect the uses.',
 source:'Price et al., “Probabilistic weather forecasting with machine learning” (GenCast), Nature 2025; Watson et al., “De novo design of protein structure and function with RFdiffusion”, Nature 2023; Zeni et al., “A generative model for inorganic materials design” (MatterGen), Nature 2025.'});
S('uses','why','What a model of p(x) is good for',{builds:2,
 body:`<div class="cards big" style="grid-template-columns:repeat(3,minmax(0,1fr))">${card('Create','new text, images, sound, designs',palette.q)+card('Fill in','missing pixels, missing readings, the rest of a sentence',palette.noise)+card('Simulate','many plausible futures instead of one average',palette.data)}${at(2,card('Spot the odd one out','a reading with very low p(x) is suspicious: a faulty sensor, a fraud',palette.loss))}${at(2,card('Compress','likely data needs fewer bits: Lecture 1’s codes',palette.p))}${at(2,card('Make more data','extra training examples, or shareable synthetic records',GREEN))}</div>`,
 lines:['Three uses that need samples: create, fill in, simulate.','Three that need scores: spot anomalies, compress, and generate extra data when real data is scarce or private.'],
 ask:['Which of these needs only samples? Which needs a score p(x)?','Which use connects to Lecture 1?'],
 expected:['Create, simulate, augment: samples. Anomalies and compression: a score.','Compression: an event with probability p costs log₂(1/p) bits, Lecture 1’s codes.'],
 transition:'What do all these models have in common?'});
chap('1.4');
S('recipe','why','What they share: random numbers in, a learned network, a sample out',{builds:2,
 body:`<div class="state-slot"><div data-only="1">${recipeDiagram(1)}</div><div data-only="2">${recipeDiagram(2)}</div></div>`,
 lines:['Every model on the tour: random numbers z go in, a network with learned weights θ turns them into a sample x.','Most also take a condition c that steers the sample: a prompt, a class label, a sketch. Today we drop c and learn p(x) for digits.'],
 ask:['What plays the role of z in the chatbot?','What is the condition for the camel pictures?'],
 expected:['The random choices made when sampling each next word.','The text prompt.'],
 transition:'Two jobs such a model can do.'});
S('two-jobs','why','A model of p(x) can do two jobs: score and sample',{builds:2,
 body:cols(stack(p(`<b class="p">Score:</b> how plausible is this x?`),eq(R`p(\text{a clean }3)\ \text{high},\qquad p(\text{static})\ \text{tiny}`),at(2,p(`<b class="q">Sample:</b> make a new x that looks like the data.`))),
  `<div class="dflex" style="align-items:end">${digit('example',0,150)}${digit('uniform',0,150)}${at(2,digit('vae16_prior_samples',3,150))}</div>`,'1.1fr 1fr'),
 lines:['Job one, score: a number that is high for a real-looking 3 and tiny for random static.','Job two, sample: produce a new x. Both jobs come from the same fitted distribution.'],
 ask:['Which job does the chatbot do when it writes?','Which job does anomaly detection need?'],
 expected:['Sample.','Score: flag inputs with low p(x).'],
 transition:'Why not just output the average example?'});
S('why-not-mean','why','Why not just output the average?',{builds:2,
 body:cols(digits('display',{idx:threes.slice(0,5),cols:5,size:84}),
  stack(`<div class="dflex"><span class="lead">average of ${threes.length} threes:</span>${rawDigit(mean3,120)}</div>`,at(2,p('Nobody writes this 3. The average is the blur in the middle of many real 3s.'))),'1.1fr 1fr'),
 lines:[`Five real 3s, and the pixel-by-pixel average of ${threes.length} of them.`,'The average looks like no real 3. We need the whole distribution, so we can draw one sharp 3 at a time.'],
 ask:['Would you accept the average as a new handwritten 3?','When is the average still useful?'],
 expected:['No: it is a ghost of many 3s.','As a summary, or a prediction scored by squared error. Not as a sample.'],
 transition:'Today’s route.'});
S('today','why','Today: two networks, then Lecture 1’s objective',{builds:3,
 body:stack(grid(3,card('Part 2 · Autoencoder','One network squeezes a digit into 2 numbers; a second rebuilds it.',palette.p),
  card('Part 3 · Can it generate?','Feed the second network random numbers. What comes out, and why?',palette.loss),
  at(2,card('Part 4 · Three fixes','Codes from N(0, I), a cloud per image, a KL pull: a new loss.',GREEN)),
  at(2,card('Part 5 · Why this loss','A model with a hidden code: the loss is minus Lecture 1’s ELBO.',palette.q)),
  at(3,card('Part 6 · Together','Every term of the loss traced back to KL and the ELBO.',palette.data))),
  at(3,small('The VAE’s loss for one image:')),at(3,`<div>${eq(R`\underbrace{\tfrac{1}{2\sigma^2}\big\lVert \pcol{x}-\qcol{\hat x}\big\rVert^2}_{\text{rebuild it}}+\underbrace{\KL\big(q_\phi(z\mid x)\,\|\,${PRI(R`\N(0,I)`)}\big)}_{\text{Lecture 1’s KL to the prior}}`)}</div>`)),
 lines:['First an autoencoder, and the question for generation: can its second network turn random numbers into digits?','Then three fixes that turn it into a VAE, and the reason they work: the VAE’s loss is minus Lecture 1’s ELBO.','A last part traces every term of that loss back to KL and the ELBO.'],
 ask:['What will the autoencoder’s 2 numbers have to keep?','What would “generate” mean for the second network?','Which piece of this loss have you seen before?'],
 expected:['Whatever is needed to rebuild the digit: which digit, its slant, its thickness.','Turn numbers we invent into a new, convincing digit.','The KL to N(0, I): Lecture 1’s KL to the prior. The squared error turns out to be the likelihood term.'],
 point:'The loss on this slide drops a constant (784 log(σ√2π)); Sections 5.5 and 6.1 show where every term comes from.',
 transition:'Part 2: autoencoders.'});

// ================================================================= Part 2 · Autoencoders
partOpener(2,'ae','Squeeze each digit into 2 numbers, then rebuild it from them. Two small networks, trained together on one squared-error loss.',
 'Why would squeezing a digit into 2 numbers be useful?','The 2 numbers must keep what matters about the digit. Later we will try inventing them.');
chap('2.1');
S('mnist','ae','MNIST: 70,000 handwritten digits, 28 × 28 pixels each',{builds:2,
 body:cols(`<div class="state-slot"><div data-only="1">${digits('first_train',{cols:8,size:72,gap:8})}</div><div data-only="2"><div class="dflex">${digit('example',0,200)}${pixelTable()}</div></div></div>`,
  stack(p(`${comma(M.n_train)} for training, ${comma(M.n_test)} for testing.`),at(2,p(`Our running example: test digit #${d.meta.example_index}, a 3. Each pixel is a number: 0 is paper, 1 is full ink.`)),at(2,eq(R`x\in[0,1]^{784}`))),'1.55fr 1fr'),
 lines:['The first 16 training digits: grey images of 28 × 28 pixels.','We follow one digit all hour: test digit #90, a 3. To a network it is a list of 784 numbers between 0 and 1.'],
 ask:['How many pixels does each image have?','Why keep one running example?'],
 expected:['28 × 28 = 784.','Every model can be checked on the same 3: its code, its reconstruction and its loss.'],
 transition:'Squeeze those 784 numbers into 2.',source:'MNIST (LeCun, Cortes and Burges), grey levels scaled to [0, 1].'});
chap('2.2');
S('ae-arch','ae','An autoencoder: encoder, short code, decoder',{builds:3,
 body:`<div class="state-slot"><div data-only="1">${aeDiagram({stage:2,showOut:false})}</div><div data-only="2">${aeDiagram({stage:3,showOut:false})}</div><div data-only="3">${aeDiagram({stage:4})}</div></div>`,
 lines:[`The encoder ${m(R`g_\phi`)} is a small network with weights φ: 784 inputs, 400 hidden units with ReLU, 2 outputs.`,'Its output is the code z, the bottleneck: everything about the digit must squeeze through 2 numbers.',`The decoder ${m(R`f_\theta`)}, with weights θ, mirrors it: 2 → 400 → 784, then a sigmoid. Train both so that x̂ ≈ x.`],
 ask:['Why squeeze through so few numbers?','What would happen with a 784-number code?','What is the target output?'],
 expected:['To force it to keep only what matters about the digit.','It could copy the input and learn nothing useful.','The input itself: the network is trained to rebuild x.'],
 point:'Hinton and Salakhutdinov (Science, 2006) made deep autoencoders work; the idea goes back to the 1980s. One hidden layer of 400 units follows Kingma and Welling (2014).',
 transition:'The same in PyTorch.'});
S('ae-code','ae','Two small networks in PyTorch',{builds:2,
 body:cols(`<div class="code-xs">${code(`class AutoEncoder(nn.Module):
    def __init__(self, z=2):
        super().__init__()
        self.encoder = nn.Sequential(  # g_φ, weights φ
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, z))
        self.decoder = nn.Sequential(  # f_θ, weights θ
            nn.Linear(z, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)     # 784 -> z numbers
        return self.decoder(z)  # z -> 784 numbers`)}</div>`,
  stack(at(2,p('Why a sigmoid at the end? Are we classifying?')),at(2,small('No: it keeps every output pixel between 0 and 1, like the data.'))),'2fr 1fr'),
 lines:['Two nn.Sequential blocks. The forward pass encodes, then decodes.','The final sigmoid squashes each of the 784 outputs into (0, 1), the range of our pixels.'],
 ask:['How many weights does the encoder have?','Why no activation after the encoder’s last layer?'],
 expected:['784 × 400 + 400 + 400 × 2 + 2 ≈ 315,000.','The code can be any real number; a ReLU would forbid negative codes.'],
 transition:'What does it minimise?'});
chap('2.3');
S('ae-loss','ae','The loss: how far each rebuilt pixel is from the real one',{builds:3,
 body:stack(eq(R`\lcol{L(\phi,\theta)}=\frac1n\sum_{i=1}^{n}\ \sum_{j=1}^{784}\big(\pcol{x_{ij}}-\qcol{\hat x_{ij}}\big)^2,\qquad \qcol{\hat x_i}=f_\theta\big(g_\phi(\pcol{x_i})\big)`),
  at(2,cols(lossFig(),at(3,lossTable()),'1.5fr 1fr'))),
 lines:['For every image: square the gap between each input pixel x and its rebuilt value x̂, add over 784 pixels. Then average over images.',
  `Our 3 through the trained 2-number autoencoder. Red marks the large squares; all 784 add up to ${f(A.ae2_sse3,1)}.`,
  'Three pixels: ① missed ink costs 0.63, ② ink where the 3 has none costs 0.42, and ③ clean background costs 0.'],
 ask:['Which weights does the loss depend on?','Where are the large errors?','Why sum over pixels but average over images?'],
 expected:['Both φ (encoder) and θ (decoder).','Along the strokes: the rebuilt 3 is blurrier and thinner than the real one.','The sum is the error of one image; averaging over images keeps the loss independent of the batch size.'],
 point:'Teal = target, purple = prediction, red = loss, as in every loss slide of the series.',
 transition:'Before training: what comes out?',source:OURS});
S('ae-untrained','ae','Before training: always look at what the network outputs',{builds:2,
 body:cols(rows(['8 test digits',digits('watch',{cols:8,size:60,gap:6})],['untrained output',digits('ae2_train_frames',{idx:[0,1,2,3,4,5,6,7],cols:8,size:60,gap:6})]),
  stack(p(`Every output pixel is between ${f(A.init_pixel_range[0],2)} and ${f(A.init_pixel_range[1],2)}: flat grey, whatever the input.`),
   at(2,p('Each pixel is off by about 0.5, since the target is 0 or 1. Squared: about 0.24.')),
   at(2,`<div class="eq-sm">${eq(R`\underbrace{784}_{\text{pixels}}\times\underbrace{${f(A.ae2_init_pixel_mse,3)}}_{\text{per pixel}}\approx${f(A.init_test_sse,0)}`)}</div>`)),'1.3fr 1fr'),
 lines:['With random weights the sigmoid outputs about 0.5 everywhere: grey squares, whatever the input.',`Each of 784 pixels is off by about a half, which squares to a quarter: about ${f(A.init_test_sse,0)} in all. Training must bring this far down.`],
 ask:['Why is every output close to 0.5?','What would it mean if training ended near 185?'],
 expected:['Small random weights make the pre-sigmoid values near 0, and sigmoid(0) = 0.5.','The network learned nothing: a bug, or a learning rate that is far off.'],
 transition:'Train it.',source:OURS});
S('ae-step','ae','One training step: forward, loss, backward, update',{builds:2,
 body:cols(`<div class="code-sm">${code(`model = AutoEncoder(z=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(30):
    for x in loader:        # 128 images at a time
        x_hat = model(x)
        loss = ((x_hat - x)**2).sum(1).mean()
        opt.zero_grad()
        loss.backward()     # encoder AND decoder
        opt.step()`)}</div>`,
  stack(p(`${comma(A.steps_per_epoch)} steps per pass over the 60,000 digits; 30 passes.`),at(2,p('One loss, one backward pass: the gradient flows from the pixel errors back through the decoder, the code, and the encoder.'))),'1.5fr 1fr'),
 lines:['Batches of 128 digits, the loss from the last slide, Adam.','loss.backward() sends gradients through the decoder, into the code, and on into the encoder: both are trained together.'],
 ask:['How many updates in total?','Does the encoder get a gradient even though the loss compares only images?'],
 expected:[`${comma(30*A.steps_per_epoch)}.`,'Yes: the reconstruction depends on the code, which depends on the encoder’s weights.'],
 transition:'Watch the reconstructions form.'});
S('ae-train','ae','Press play: reconstructions sharpen as it trains',{builds:2,
 body:widget('ae-train','Reconstructions during training of the 2-number autoencoder.'),
 lines:[`Frame 0: grey, error ${f(A.ae2_frames_sse[0],0)}. Press play: after 10 steps it is ${f(A.ae2_frames_sse[1],0)}; the network has learned the average blob.`,`The error keeps falling as each digit takes shape; after 30 epochs it is ${f(A.ae2_frames_sse.at(-1),1)}. Under each digit: its own error.`],
 ask:['What does the network learn first?','Which digits stay hardest to rebuild from 2 numbers?'],
 expected:['The average digit: a blurry blob that lowers the error everywhere.','Ones that resemble other digits, like 4 and 9, or unusual shapes.'],
 transition:'How much does the code size matter?',source:OURS});
chap('2.4');
S('code-size','ae','More numbers in the code, sharper rebuilds',{builds:2,
 body:rows(['originals',digits('watch',{cols:8,size:58,gap:6})],[`2 numbers · error ${f(SW['2'].test_sse,1)}`,digits('ae2_recon_watch',{cols:8,size:58,gap:6})],[`8 numbers · ${f(SW['8'].test_sse,1)}`,digits('ae8_recon_watch',{cols:8,size:58,gap:6})],[`32 numbers · ${f(SW['32'].test_sse,1)}`,at(2,digits('ae32_recon_watch',{cols:8,size:58,gap:6}))],[`128 numbers · ${f(SW['128'].test_sse,1)}`,at(2,digits('ae128_recon_watch',{cols:8,size:58,gap:6}))]),
 lines:['The same 8 test digits through autoencoders with codes of 2 and 8 numbers. The error is the average over all 10,000 test digits.','With 32 and 128 numbers the strokes come back almost exactly. Hold on to this: Part 3 asks which of them can generate.'],
 ask:['What does the 2-number code lose?','Which of these would you pick?'],
 expected:['Stroke thickness, slant and small details.','For rebuilding, the widest. For generating, wait for Part 3.'],
 transition:'Where do the 2-number codes land?',source:OURS});
S('ae-scatter','ae','Where the codes land: 1,000 test digits at their 2 numbers',{builds:2,
 body:widget('scatter','The 2-number autoencoder codes of 1,000 test digits.',{src:'ae2',lim:AE_LIM.join(','),ticks:'0,10,20,30|-10,0,10',title:'the whole code: z₁ and z₂',side:'This autoencoder has exactly 2 numbers|in its code, so nothing is hidden here.|Hover any point to see its digit.'}),
 lines:['Each dot is one test digit, placed at its 2-number code and coloured by its label. Hover to see the digit.','Digits of one kind cluster together, although no labels were used in training. The codes run from about −5 to 22.'],
 ask:['Which digits are still mixed?','The codes run from about −5 to 22. Is that a problem?'],
 expected:['4 and 9; some 3, 5 and 8.','Not for rebuilding. For generating, Part 3 will show it is.'],
 transition:'Part 3: can the decoder generate?',source:OURS});

// ================================================================= Part 3 · Can an autoencoder generate?
partOpener(3,'gen','Keep the decoder, throw the encoder away, and invent the code. Does a random code give a digit?',
 'What would a good random code have to look like?','Like the codes real digits get, so the decoder knows what to draw.');
chap('3.1');
S('gen-question','gen','Keep the decoder, feed it random numbers',{builds:2,
 body:stack(genDiagram('ae'),at(2,p(`Which random numbers? The simplest: z ~ N(0, I), two independent standard bells. In code: <code>torch.randn(2)</code>.`))),
 lines:['The decoder turns 2 numbers into a digit. To generate, skip the encoder and invent the 2 numbers.','We need a distribution to draw codes from. Start with the simplest, the standard normal N(0, I).'],
 ask:['What should a random code look like, to give a good digit?','Why might N(0, I) be a poor choice?'],
 expected:['Like the codes real digits get.','Nothing in training told the codes to look like N(0, I): they run from −5 to 22.'],
 transition:'Feed such draws to three decoders.'});
S('ae-random','gen','Random codes through three autoencoders',{builds:3,
 body:stack(rows(['2 numbers',digits('ae2_prior_samples',{cols:16,size:54,gap:4})],['8 numbers',at(2,digits('ae8_prior_samples',{cols:16,size:54,gap:4}))],['128 numbers',at(3,digits('ae128_prior_samples',{cols:16,size:54,gap:4}))]),at(3,takeaway('Better reconstruction did not give better samples: the best rebuilder makes static. Why? Look at where the codes are.'))),
 lines:['The 2-number autoencoder: some digits, mostly the same few blurry shapes.','8 numbers: strokes that belong to no digit.','128 numbers, the best rebuilder of all, gives static.'],
 ask:['Which row looks most like digits?','The 128-number model rebuilt best. Why does it generate worst?','What would you check to find out?'],
 expected:['The 2-number one.','Its codes for real digits are nowhere near where N(0, I) puts its draws.','Plot where real codes are, next to the random codes.'],
 transition:'Plot where the codes are.',source:OURS});
chap('3.2');
S('overlay','gen','Put the random codes on top of the real ones',{builds:2,
 body:stack(widget('scatter','Autoencoder codes of 1,000 test digits, 500 draws from N(0, I) in red, and the circles of N(0, I).',{src:'ae2',lim:AE_LIM.join(','),ticks:'0,10,20,30|-10,0,10',prior:500,contours:'1',decode:'ae2',h:300,title:'real codes and 500 random ones (red)',side:'Circles: N(0, I) at radius 1, 2 and 3.|Hover a dot to see the digit and what|the decoder draws from that code.'}),
  at(2,takeaway('N(0, I) covers a small corner of the real codes: random codes reach a few digit types and miss the rest.'))),
 lines:['The codes of 1,000 test digits, and 500 draws from N(0, I) in red inside its circles. Hover any dot to see what the decoder draws.','Nothing in the loss asked the codes to sit near N(0, I). The red draws land in one corner: some digits there, none of the rest.'],
 ask:['Hover a red dot near the origin, then one in an empty area. What does the decoder draw?','For the 128-number model, where would the red points fall?'],
 expected:['Near real codes: a digit; in empty places: blurs or odd strokes.','Even further from the real codes: in 128 numbers the space is mostly empty.'],
 point:'The decoder runs in the browser from lecture2/data/nets.json.',
 transition:'Try choosing codes yourself.',source:OURS});
S('explore-ae','gen','Your turn: three ways to choose a code',{builds:1,
 body:widget('explorer','Choose a code for the autoencoder and see its decoder output.',{net:'ae2',lim:AE_LIM.join(',')}),
 lines:['Draw a random code from N(0, I), encode a real test digit, or click anywhere on the plane. The decoder runs live in the browser.'],
 ask:['Which of the three ways gives the best digits?'],expected:['Encoding a real digit. Random codes and clicks far from the dots mostly give smudges.'],
 transition:'Decode a whole grid of codes at once.',source:OURS});
S('ae-grid','gen','Decode a whole grid of codes',{builds:2,
 body:cols(fig(squareW([-12,30,-13,18],400),400,scatterSVG('ae2',{W:squareW([-12,30,-13,18],400),H:400,lim:[-12,30,-13,18],ticks:[[-10,0,10,20,30],[-10,0,10]],box:G.ae2_grid_box,r:3,opacity:.45}).s,'Codes with the decoded grid’s area marked.'),
  stack(digits('ae2_grid',{cols:12,size:23,gap:1}),at(2,takeaway('Good digits only where real codes are. Away from them the decoder only extrapolates.'))),'1.15fr 1fr'),
 lines:['Decode a 12 × 12 grid of codes covering the dashed box: the codes on the left, the decoder’s output at each grid point on the right.','Over the real codes, digits blend into each other. Far from them, thick 1s and 0s and broken strokes: the decoder never trained there.'],
 ask:['Which parts of the grid are useful for generation?','What does the decoder do far from any real code?'],
 expected:['Those over the clusters of real codes.','It extrapolates from the nearest digits: exaggerated or broken shapes.'],
 transition:'What we want instead.',source:OURS});
chap('3.3');
S('wishes','gen','What we saw, and what we want instead',{builds:3,
 body:`<div>${table(['what we saw','what we want'],[
  ['random N(0, I) codes miss the real codes','codes that fill a region we know how to sample'],
  [at(2,'gaps between codes decode to junk'),at(2,'a decoder trained everywhere in that region, gaps included')],
  [at(3,'no way to say how likely an image is'),at(3,'one probability model that can score and sample')]])}</div>`,
 lines:['Problem one: random codes from N(0, I) land away from the real codes. We want codes that fill N(0, I).','Problem two: gaps between codes decode to junk. We want each image to own a small cloud, so the gaps get trained.','Problem three: the autoencoder cannot score an image. We want a probability model. Part 4 fixes the first two; Part 5 shows the fix is that model.'],
 ask:['How could we push codes towards N(0, I)?','How could each image own a cloud instead of a point?','Which tool from Lecture 1 will we need?'],
 expected:['Penalise codes that stray from it, for example with a KL to N(0, I).','Let the encoder output a mean and a spread, and draw the code from them.','Variational inference and the ELBO (Part 5), with its reparameterisation trick and Gaussian KL (Part 4).'],
 transition:'Part 4: three fixes, one at a time.'});

// ================================================================= Part 4 · From an autoencoder to a VAE: three fixes
partOpener(4,'idea','The autoencoder cannot generate. Three fixes, one at a time, turn it into a VAE; each one answers a problem we saw in Part 3.',
 'Which problem from Part 3 would you fix first?','Where the codes live: random codes from N(0, I) missed the real codes.');
chap('4.1');
S('fix-prior','idea','Fix 1: choose where the codes live, before training',{builds:3,
 body:cols(fig(440,420,stdNormalFig(),'500 draws from N(0, I), with circles of radius 1 and 2.'),stack(
  p('Decide the distribution of codes in advance: z ~ N(0, I), two independent standard bells.'),
  at(2,p('Why this one? It is easy to sample (<code>torch.randn(2)</code>), it has no gaps, and it is the same in every training run.')),
  at(3,p('Generating becomes two lines: draw z, decode it. The work moves into training: the codes of real digits must fill this disc.'))),'1fr 1.1fr'),
 lines:['The autoencoder put its codes wherever reconstruction was easiest. Turn it around: choose the region first, and make the codes go there.',
  `500 draws from N(0, I): ${f(100*inside(1),0)}% land within distance 1 of the origin and ${f(100*inside(2),0)}% within 2.`,
  'If training makes the codes of real digits fill this disc, every random draw lands near some digit. Fixes 2 and 3 make that happen.'],
 ask:['Why not let the network choose the code distribution itself?','Why N(0, I) and not, say, a uniform square?','What must training now achieve?'],
 expected:['Then we would not know how to sample from it: that was the autoencoder’s problem.','Any simple, known distribution would do. N(0, I) is smooth, has no edges, and its KL has a formula (Fix 3).','The codes of real images must spread over N(0, I), with no large holes.'],
 point:'In Part 5 this choice becomes the prior p(z), and it keeps the green colour of Lecture 1’s prior.',
 transition:'Fix 2: fill the gaps.',source:OURS});
chap('4.2');
S('fix-cloud','idea','Fix 2: send each image to a small cloud of codes, not a point',{builds:2,
 body:stack(cloudsFig(),small('A sketch, not data. Colours: three kinds of digit.')),
 lines:['The autoencoder sends each image to one point. Only those points are ever decoded in training, so the space between them is never trained.','Instead, send each image to a small cloud and decode a random point from it. Neighbouring clouds overlap, so the decoder learns the gaps too.'],
 ask:['Which parts of the code plane does an autoencoder’s decoder see in training?','What does the decoder learn when clouds overlap?'],
 expected:['Only the exact codes of training images.','To draw something sensible everywhere the clouds reach, including between images.'],
 point:'A sketch, not data. Lecture 2, Section 6.2 calls Part 4 “Road 1” to the VAE.',
 transition:'How a network outputs a cloud.'});
S('fix-net','idea','How a network outputs a cloud: a mean, a spread and a random draw',{builds:3,
 body:`<div class="state-slot"><div data-only="1">${aeDiagram({kind:'vae',stage:2,showOut:false})}</div><div data-only="2">${aeDiagram({kind:'vae',stage:3,showOut:false})}</div><div data-only="3">${aeDiagram({kind:'vae',stage:5})}</div></div>`,
 lines:['The encoder now ends in 4 numbers: a mean μ and a log variance log σ² for each of the 2 code numbers. Together they describe this image’s cloud.',`Draw a code from the cloud with Lecture 1’s reparameterisation trick, tool ①: z = μ + σ ⊙ ε, ε ~ N(0, I). Gradients still reach μ and σ. ${l1('reparam','3.5','the trick')}`,'The decoder is the autoencoder’s, unchanged: from z to a mean image.'],
 ask:['Why output log σ² instead of σ?','Why reparameterise instead of sampling z directly?','At test time, which code do we use to rebuild?'],
 expected:['Any real number is a valid log variance; σ = exp(½ log σ²) is then positive.','sample() blocks the gradient path from the loss to μ and σ.','Usually the mean μ.'],
 transition:'Fix 3: what stops the clouds drifting apart?'});
chap('4.3');
S('fix-kl','idea','Fix 3: a penalty that pulls every cloud towards N(0, I)',{builds:3,
 body:stack(p('With Fixes 1 and 2 alone, nothing stops the encoder from shrinking each cloud to a point and placing it far from the centre: an autoencoder again.'),
  at(2,eq(R`\text{penalty}=\KL\big(\N(\mu,\sigma^2)\,\|\,${PRI(R`\N(0,1)`)}\big)=\tfrac12\big(\mu^2+\sigma^2-\log\sigma^2-1\big)`)),
  at(2,`<div>${l1('diag-kl','2.6','tool ②: the Gaussian KL')}</div>`),
  at(3,`<div class="table-sm">${table(['one code number’s cloud','μ','σ','penalty'],[['exactly N(0, 1)','0','1','0'],['far from the centre','5','1',f(klq(5,1),1)],['shrunk almost to a point','0','0.01',f(klq(0,.01),1)],['our 3, first code number',f(V.q3.mu[0],2),f(V.q3.sd[0],3),f(klq(V.q3.mu[0],V.q3.sd[0]),1)]])}</div>`)),
 lines:['Two ways the encoder could cheat: shrink every cloud to a point, or push the clouds far apart. Either way we are back to an autoencoder.',
  'Charge each cloud for being unlike N(0, 1), with Lecture 1’s KL. Tool ② gives it by formula, one term per code number.',
  `Far away costs (μ = 5: ${f(klq(5,1),1)}); a cloud shrunk to a point costs too (σ = 0.01: ${f(klq(0,.01),1)}). Only N(0, 1) itself is free.`],
 ask:['What would the encoder do without this penalty?','Why a KL and not just μ²?','Which term punishes a shrinking cloud?'],
 expected:['Make clouds tiny and far apart: reconstruction is easiest that way.','μ² would pull clouds to the centre but let σ shrink to 0. The KL stops both, and Part 5 shows it is exactly what Lecture 1’s ELBO asks for.','−log σ²: it grows without limit as σ → 0.'],
 transition:'Put the three fixes together.',source:OURS});
chap('4.4');
S('fix-loss','idea','The three fixes give a loss',{builds:3,
 body:stack(`<div class="eq-sm">${eq(R`L=\underbrace{\big\lVert \pcol{x}-\qcol{f_\theta(\mu+\sigma\odot\epsilon)}\big\rVert^2}_{\text{rebuild from a draw of the cloud (Fix 2)}}\;+\;\underbrace{\KL\big(\N(\mu,\sigma^2)\,\|\,${PRI(R`\N(0,I)`)}\big)}_{\text{pull the cloud to }\N(0,I)\text{ (Fixes 1 and 3)}}`)}</div>`,
  at(2,p('Two questions remain. How much weight should each term get? And is this a sensible objective, or a hack that happens to work?')),
  at(3,p('Part 5 answers both with Lecture 1: this sum is minus the ELBO of a probability model, with a weight of 1/2σ² on the first term.'))),
 lines:['Collect the fixes: rebuild the image from a code drawn from its cloud, plus a KL that pulls the cloud towards N(0, I).',
  'The autoencoder had one term; this loss has two, pulling against each other. Which weight, and why this sum?',
  'Lecture 1 answers both: the sum is minus the ELBO, the weight comes from the pixel noise, and the KL appears by itself.'],
 ask:['Which term is the autoencoder’s loss?','What would a weight of 0 on the KL give?','What would a probability model add?'],
 expected:['The first: squared error between the image and its rebuild.','An autoencoder with noisy codes; in practice the clouds shrink to points (Section 5.8, β = 0).','A reason for the loss and its weights, and a score for images: a lower bound on log p(x).'],
 transition:'The two terms pull against each other.'});
S('vae-tug','idea','Two terms pull in opposite directions',{builds:4,
 body:stack(tugFig(),at(4,takeaway('A VAE is an autoencoder whose codes are noisy clouds held near N(0, I). To generate, draw z ~ N(0, I) and decode.'))),
 lines:['With only the rebuild term, the best clouds are tiny and far apart: each image keeps a private, exact code. That is Part 3’s autoencoder.','With only the KL term, every cloud becomes N(0, I) itself: all images share one cloud, and the decoder can only draw the average digit.','Together they balance: clouds sit near the centre and overlap a little, yet stay distinct enough to rebuild each image.','So a random draw from N(0, I) usually lands on or near some cloud, and the decoder has been trained to draw something there.'],
 ask:['What does the reconstruction term want the clouds to do?','What does the KL term want?','Where does the balance land?','Why does sampling now work?'],
 expected:['Shrink (less noise in z) and spread apart (no confusion between images).','Move to the centre and widen to σ = 1.','Clouds near the centre, small but not tiny, slightly overlapping.','The codes fill the region N(0, I) samples from, and the decoder was trained across it.'],
 point:'A sketch, not data. Section 5.6 is the real version for our 3, and Section 5.8 shows β = 0 (panel one) and β = 256 (panel two).',
 transition:'Part 5: why this loss is the right one.'});

// ================================================================= Part 5 · The VAE
partOpener(5,'vae','Why is Part 4’s loss the right one? Write a model with a hidden code, hit Lecture 1’s wall, and get past it the same way: the ELBO, one image at a time.',
 'Which wall did Lecture 1 hit?','The evidence p(𝒟): an integral with no formula, so the posterior was out of reach.');
chap('5.1');
S('recap-elbo','vae','From Lecture 1: one objective we can compute',{builds:3,
 body:stack(eq(R`\dcol{\log p(\mathcal D)}=\qcol{\ELBO(q)}+\lcol{\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)}`),
  small('log p(𝒟) is one fixed number. Whatever the ELBO gains, the KL to the posterior loses.'),
  at(2,eq(R`\qcol{\ELBO(q)}=\underbrace{\E_{\theta\sim q}\big[\ncol{\log p(\mathcal D\mid\theta)}\big]}_{\text{fit the data}}-\underbrace{\KL\big(\qcol{q(\theta)}\,\|\,${PRI(R`p(\theta)`)}\big)}_{\text{stay near the prior}}`)),
  at(3,eq(R`\text{train:}\quad\psi^*=\arg\max_\psi\ \qcol{\ELBO(q_\psi)}\qquad\text{loss}=-\qcol{\ELBO}`)),
  at(3,`<div>${l1('elbo-goal','4.1','the objective')}${l1('elbo-two-terms','4.1','the ELBO in two terms')}</div>`)),
 lines:['Lecture 1’s budget: the fixed log evidence splits into the ELBO, which we can compute, and the KL to the posterior, which we cannot.',
  'The ELBO in two terms: an average log-likelihood, to fit the data, minus a KL from q to the prior, to stay near it.',
  'The objective: tune q’s knobs ψ to maximise the ELBO, by gradient descent on minus the ELBO. Today reuses this line unchanged.'],
 ask:['Which part of log p(𝒟) could we compute?','What does each ELBO term ask of q?','What did gradient descent change in Lecture 1?'],
 expected:['The ELBO. The KL to the posterior needs p(𝒟), which we cannot compute.','The first: put q on θ values that explain the data. The second: do not stray far from the prior.','ψ: the means and log-widths of q, one bell per weight of the tiny network.'],
 point:'Colours follow Lecture 1’s Part 4: q purple, likelihood orange, prior green, evidence blue, the KL to the posterior red.',
 transition:'A model with a hidden code.'});
chap('5.2');
S('vae-pgm','vae','A generative model with a hidden code',{builds:3,
 body:`<div class="state-slot">${[1,2,3].map(k=>`<div data-only="${k}">${vaePgm(k)}</div>`).join('')}</div>`,
 lines:['Start with the prior: every image has a hidden code z, and before seeing any image we say z ~ N(0, I).','Then the likelihood: the decoder turns z into a mean image, and the image x is that mean plus a little Gaussian noise per pixel.','The decoder’s weights θ sit outside the box: one set, shared by every image, learned from the data.'],
 ask:['Which circle is observed?','What does the arrow mean?','How many sets of decoder weights are there?'],
 expected:['x, the image (shaded). z is never observed.','x is drawn given z: through the decoder, then pixel noise.','One, shared by all 60,000 images.'],
 point:'Graphical-model notation (Bishop, PRML chapter 8): shaded = observed, box = repeated for each example, a dot outside = a parameter. Prior green, likelihood orange, as in Lecture 1.',
 transition:'How to draw a new image from this model.'});
S('vae-draw','vae','To draw an image, follow the arrows: first z, then x',{builds:3,
 body:stack(`<div class="dflex" style="gap:26px">`+
  `<div class="card" style="border-color:${GREEN}66;min-width:210px"><h4 style="color:${GREEN}">① draw z</h4><p><code>z = torch.randn(2)</code></p><p>z = (${f(PZ[0][0],2)}, ${f(PZ[0][1],2)})</p></div>`+
  at(2,`<div class="dflex" style="gap:26px"><span class="lead">→</span><div class="card" style="border-color:${palette.noise}66"><h4 style="color:${palette.noise}">② decode</h4><figure class="dg-cell">${digit('vae2_prior_samples',0,120)}<figcaption>mean image f<sub>θ</sub>(z)</figcaption></figure></div></div>`)+
  at(3,`<div class="dflex" style="gap:26px"><span class="lead">→</span><div class="card" style="border-color:${palette.data}66"><h4 style="color:${palette.data}">③ add pixel noise</h4><figure class="dg-cell">${digit('vae2_noisy_sample',0,120)}<figcaption>x = mean + 0.2 × noise</figcaption></figure></div></div>`)+`</div>`,
  at(3,takeaway('We usually show the mean image: the pixel noise is independent speckle and adds nothing about the digit.'))),
 lines:['Step one, the prior: draw two numbers from N(0, I). No image is needed.','Step two, the arrow: the trained decoder turns the code into a mean image.','Step three: add independent noise of size σ = 0.2 to each pixel. That is a true sample of x; we display the mean image.'],
 ask:['Does generating need the encoder?','Why is this a 9 and not another digit?','Why display the mean image?'],
 expected:['No: only the prior and the decoder.','Because of where this z sits: the decoder has learned which region of codes means which digit.','The noise carries no information about the digit; it only makes speckle.'],
 transition:'The likelihood, in numbers.',source:OURS});
S('sigma','vae','The likelihood, in numbers: a squared error',{builds:3,
 body:stack(eq(R`\ncol{\log p_\theta(x\mid z)}=\sum_{j=1}^{784}\log\N\big(x_j;\,f_\theta(z)_j,\ \sigma^2\big)`),at(2,eq(R`=-\frac{1}{2\sigma^2}\,\big\lVert x-f_\theta(z)\big\rVert^2\;-\;784\log\big(\sigma\sqrt{2\pi}\big)`)),
  at(3,p(`We use σ = ${f(d.meta.sigma,1)}, about the typical pixel error of a 2-number code: √(31 / 784) ≈ 0.2. The constant −784 log(σ√2π) is +${f(-V.loglik_const,1)}.`))),
 lines:['Each pixel is an independent Gaussian around the decoder’s output, so the log-likelihood is a sum over 784 pixels.','Expand the Gaussian: minus the squared error over 2σ², plus a constant. The autoencoder’s loss was a likelihood all along.','σ = 0.2 is a modelling choice, matched to a typical pixel error. Section 5.8 shows what changing the balance does.'],
 ask:['What happens to the first term if σ is small?','Where have you seen squared error = Gaussian likelihood before?','Is σ learned here?'],
 expected:['Errors cost more: the model is pushed to rebuild precisely.','Linear regression: least squares is maximum likelihood with Gaussian noise.','No, fixed at 0.2. It can be learned; we keep it fixed for clarity.'],
 transition:'To learn θ we need p(x). Can we get it?'});
chap('5.3');
S('vae-wall','vae','Which codes made our 3? Bayes’ rule, and Lecture 1’s wall',{builds:3,
 body:stack(eq(R`\pcol{p(z\mid x)}=\frac{\ncol{p_\theta(x\mid z)}\ \ ${PRI('p(z)')}}{\dcol{p_\theta(x)}}`),
  at(2,eq(R`\dcol{p_\theta(x)}=\int \ncol{p_\theta(x\mid z)}\ ${PRI('p(z)')}\,dz`)),at(2,small('An integral over every possible code, through a neural network: no formula.')),
  at(3,p(`Lecture 1 hit this wall with p(𝒟), an integral over the network’s weights. We get past it the same way. ${l1('vi-idea','1.5','approximate the posterior')}`))),
 lines:['Training needs the posterior: which codes could have made this image? Bayes’ rule, with the prior in green and the likelihood in orange.','Its denominator, the evidence p(x), adds up every code weighted by its prior: an integral through a neural network.','Lecture 1’s problem exactly: an evidence we cannot compute. Lecture 1’s answer was variational inference.'],
 ask:['Why do we need the posterior?','Why can we not compute p(x)?','What did Lecture 1 do instead?'],
 expected:['To know which codes explain each image, so the decoder can be trained on them; and log p(x) is what training should maximise.','The decoder is a neural network: the integral has no closed form.','Chose a simple q and maximised the ELBO.'],
 transition:'What the posterior looks like for our 3.'});
S('posterior3','vae','For our 3: which codes could have made it?',{builds:2,
 body:cols(`<div class="state-slot"><div data-only="1">${fig(400,400,wideMark(),'The posterior over codes for our 3, within the prior.')}</div><div data-only="2">${fig(810,400,wideMark()+posteriorSVG({ox:410,S:400,view:'zoom',title:'zoomed in about 27×'}).s,'The posterior, wide and zoomed.')}</div></div>`,
  stack(at(2,p(`Centre (${f(V.grid.post_mean[0],2)}, ${f(V.grid.post_mean[1],2)}), width about ${f(V.grid.post_sd[0],3)}.`))),'1.6fr .7fr'),
 lines:['With only 2 numbers we can still check every code on a fine grid: shade = how well that code explains our 3. Dashed circles: the prior N(0, I).',`Zoomed in: the codes that explain this 3 form a tiny patch, about ${f(V.grid.post_sd[0],3)} wide. With 16 numbers no grid could find it.`],
 ask:['Why is the patch so small?','Why does the grid trick fail for 16 numbers?'],
 expected:['784 pixels with σ = 0.2 pin the code down very precisely.','A grid of 100 points per number would need 100¹⁶ points: the same count as Lecture 1’s 10¹⁴.'],
 transition:'Line Lecture 1 up with today.',source:OURS});
S('cast','vae','Same problem as Lecture 1, new names',{builds:2,
 body:stack(`<div class="table-sm">${table(['','Lecture 1','today, for one image x'],[
  ['the unknown, with a prior','θ: the tiny network’s 7 weights','z: this image’s 2-number code'],
  ['prior',m(PRI(R`p(\theta)`)),m(PRI(R`p(z)=\N(0,I)`))],
  ['likelihood',m(R`\ncol{p(\mathcal D\mid\theta)}`),m(R`\ncol{p_\theta(x\mid z)}`)+': the decoder'],
  ['posterior: out of reach',m(R`\pcol{p(\theta\mid\mathcal D)}`),m(R`\pcol{p(z\mid x)}`)],
  ['approximation',m(R`\qcol{q_\psi(\theta)}`)+': the guide',m(R`\qcol{q_\phi(z\mid x)}`)+': the encoder'],
  [at(2,'how many q’s?'),at(2,'one'),at(2,'one per image: 60,000, all computed by one encoder network')]])}</div>`,
  small('Careful with θ: today it names the decoder’s weights, learned as ordinary parameters. The unknown with a prior is z.')),
 lines:['Line them up. Lecture 1’s weights θ become a code z, and the data 𝒟 become one image x. Prior, likelihood and posterior keep their roles and colours.','One change: Lecture 1 fitted one q. Here every image needs its own q, so a network, the encoder, outputs it from the image.'],
 ask:['Which row changed the most?','Why not fit 60,000 separate q’s, as Lecture 1 fitted one?'],
 expected:['The approximation: one q became a q for every image, produced by a network.','It would take a separate optimisation for every image, and again for every new image. One encoder pass is instant.'],
 point:'Fitting one network that outputs q for any input is called amortised inference.',
 transition:'Lecture 1’s ELBO with the new names.'});
chap('5.4');
S('vae-elbo','vae','Lecture 1’s ELBO, for one image',{builds:3,
 body:stack(small('Lecture 1'),eq(R`\qcol{\ELBO(q)}=\E_{\theta\sim q}\big[\ncol{\log p(\mathcal D\mid\theta)}\big]-\KL\big(\qcol{q(\theta)}\,\|\,${PRI(R`p(\theta)`)}\big)`),
  at(2,small('today: swap the names')),at(2,eq(R`\qcol{\ELBO(x)}=\E_{z\sim q_\phi(z\mid x)}\big[\ncol{\log p_\theta(x\mid z)}\big]-\KL\big(\qcol{q_\phi(z\mid x)}\,\|\,${PRI(R`p(z)`)}\big)`)),
  at(3,eq(R`\dcol{\log p_\theta(x)}=\qcol{\ELBO(x)}+\lcol{\KL\big(q_\phi(z\mid x)\,\|\,p(z\mid x)\big)}\ \ge\ \qcol{\ELBO(x)}`)),
  at(3,small('Raising the ELBO does two jobs: θ makes the image more probable, and φ brings q closer to the posterior.'))),
 lines:['Lecture 1’s ELBO: fit the data, stay near the prior.','Swap the names: the same two terms for one image. Rebuild x from codes drawn from q, and keep q near the prior N(0, I).','And the same budget: log p(x) = ELBO + a KL that is never negative. Training maximises the ELBO summed over all 60,000 images.'],
 ask:['Which term is new compared with Lecture 1?','Which term rewards a good reconstruction?','What does maximising the ELBO do to θ, and to φ?'],
 expected:['None: every term has a counterpart. Only the names changed.','The first: the expected log-likelihood of the image.','θ: the images become more probable under the model. φ: q moves towards the true posterior.'],
 point:'The same derivation as Lecture 1, Section 4.1, with θ → z and 𝒟 → x; Lecture 2, Section 6.3 writes it out for one image.',
 transition:'Who computes q for each image?'});
S('elbo-read','vae','Read the ELBO term by term: it is Part 4’s three fixes',{builds:3,
 body:`<div class="mchain mread"><div class="mc-h">piece of the ELBO for one image</div><div class="mc-h">what it says</div><div class="mc-h">where we met it</div>${[
  [R`\E_{z\sim\qcol{q_\phi(z\mid x)}}\big[\,\cdots\big]`,'draw codes from this image’s cloud: the cloud is q','Fix 2'],
  [R`\ncol{\log p_\theta(x\mid z)}`,'decode z and score how well it explains x: the rebuild term','the autoencoder’s loss (5.2)'],
  [R`-\KL\big(\qcol{q_\phi(z\mid x)}\,\|\,${PRI('p(z)')}\big)`,'pull the cloud towards the prior p(z) = N(0, I)','Fixes 1 and 3']
 ].map(([t,say,src],k)=>{const c=x=>k?`<div data-at="${k+1}">${x}</div>`:`<div>${x}</div>`;return c(m(t))+c(`<span class="mc-w">${say}</span>`)+c(`<span class="mc-s">${src}</span>`);}).join('')}</div>`,
 lines:['The average over q: codes drawn from this image’s cloud. The encoder’s cloud of Fix 2 is Lecture 1’s q.',
  'The log-likelihood: decode the code and score the image. With Gaussian pixels it is the rebuild term (Section 5.2).',
  'The KL to the prior is Fix 3’s penalty, and the prior N(0, I) is Fix 1. Part 4’s guessed loss was minus the ELBO all along.'],
 ask:['Which fix does the average over q correspond to?','Which term was the autoencoder’s whole loss?','Where did Fix 1 go?'],
 expected:['Fix 2: drawing z from the image’s cloud, q_φ(z | x).','log p(x | z): the reconstruction, up to the 1/2σ² weight.','Into the prior p(z) = N(0, I), inside the KL.'],
 transition:'Fitting q for one image, as Lecture 1 did.'});
S('vi3','vae','Lecture 1’s VI on one image: fit q for our 3, press play',{builds:2,
 body:widget('vi3','Variational inference for one image.'),
 lines:['Lecture 1’s recipe on one image: start q at the prior N(0, I), draw codes with tool ①, and climb the ELBO with Adam.',`q moves into the posterior’s patch and shrinks to fit it. The ELBO ends at ${f(V.vi3.final_elbo,2)}, just below the ceiling log p(x) = ${f(V.grid.log_px,2)}.`],
 ask:['What happens to the ELBO at first? Why?','How big is the final gap?'],
 expected:['It wobbles and dips: q is still huge, and most codes it draws explain the 3 badly.',`About ${f(V.grid.log_px-V.vi3.final_elbo,2)} nats: q is nearly the exact posterior.`],
 point:'Adam, step size 0.05 then 0.01, 64 reparameterised draws per step, 1,200 steps (seed 23). The ELBO plotted is the noisy estimate at each step. Left: all codes, prior circles at radius 1 and 2; middle: zoomed on the posterior.',
 transition:'1,200 steps for one image. For 60,000 images, a network does it in one pass.',source:OURS});
S('encoder-q','vae','The encoder’s cloud is q: one network gives q for every image',{builds:3,
 body:`<div class="state-slot">${[1,2,3].map(k=>`<div data-only="${k}">${inferPgm(k)}</div>`).join('')}</div>`,
 lines:['The decoder is the solid arrow, from z to x. The posterior, which codes made this x, is out of reach.','The dashed arrow is Part 4’s encoder: its cloud N(μ, σ²) is q for that image, computed in one pass instead of 1,200 steps.','Decoder θ and encoder φ are trained together, by one objective: the ELBO, summed over all images.'],
 ask:['Which way does each arrow point?','What does the encoder output for one image?','What is variational about the VAE?'],
 expected:['Decoder: z to x (generate). Encoder: x to z (infer).','Two means and two log variances: q_φ(z | x) = N(μ, σ²).','q is a member of a simple family (diagonal Gaussians), fitted by maximising the ELBO: Lecture 1’s variational inference.'],
 point:'Kingma and Welling (2014), figure 1: solid = generative model p_θ, dashed = variational approximation q_φ.',
 transition:'Write the loss.'});
chap('5.5');
S('vae-loss','vae','Minus the ELBO is Part 4’s loss, with its weight decided',{builds:3,
 body:stack(`<div>${eq(R`-\qcol{\ELBO(x)}=\underbrace{\E_{q}\Big[\tfrac{1}{2\sigma^2}\big\lVert \pcol{x}-\qcol{f_\theta(z)}\big\rVert^2\Big]}_{\text{from }-\ncol{\log p_\theta(x\mid z)}}+\underbrace{\KL\big(q_\phi(z\mid x)\,\|\,${PRI(R`\N(0,I)`)}\big)}_{\text{stay near the prior}}+\underbrace{\text{const}}_{-541.4}`)}</div>`,
  at(2,`<div>${eq(R`\E_q\big[\cdots\big]\approx\tfrac{1}{2\sigma^2}\big\lVert \pcol{x}-\qcol{f_\theta(\mu+\sigma\odot\epsilon)}\big\rVert^2,\qquad\epsilon\sim\N(0,I)\ \text{(one draw, tool ①)}`)}</div>`),
  at(3,`<div>${eq(R`\KL=\tfrac12\sum_{k=1}^{2}\big(\mu_k^2+\sigma_k^2-\log\sigma_k^2-1\big)\qquad\text{(tool ②, no sampling)}`)}</div>`)),
 lines:['Flip the sign of the ELBO: Part 4’s loss, with the rebuild term weighted by 1/2σ², plus a constant that training can ignore.','The average over q is estimated with one reparameterised draw per image per step.','The KL between two diagonal Gaussians has a formula, so it needs no sampling. Both of Lecture 1’s tools, in one loss.'],
 ask:['Which term pulls the codes towards N(0, I)?','Why is one draw enough?','What does the KL do to a code with μ = 5?'],
 expected:['The KL term.','Over many steps and images the noise averages out; the estimate is unbiased.','Adds about ½ × 25 = 12.5 per number: strong pressure back towards the origin.'],
 point:'The constant is 784 log(σ√2π) = −541.4 for σ = 0.2. Shape of the loss: data fit + penalty, like ridge regression, but here the penalty comes out of the ELBO.',
 transition:'Put in the numbers for our 3.'});
S('vae-worked','vae','One step for our 3, with numbers',{builds:3,
 body:stack(eq(R`\mu=(${f(V.q3.mu[0],3)},\,${f(V.q3.mu[1],3)}),\quad\sigma=(${f(V.q3.sd[0],3)},\,${f(V.q3.sd[1],3)}),\quad\epsilon=(${f(WK.eps[0],1)},\,${f(WK.eps[1],1)})`),eq(R`z=\mu+\sigma\odot\epsilon=(${f(WK.z[0],3)},\,${f(WK.z[1],3)})`),
  at(2,`<div class="dflex">${digit('example',0,92,`style="outline:3px solid ${palette.p};outline-offset:1px"`)}<span class="lead">vs</span>${digit('vae2_worked_recon',0,92,`style="outline:3px solid ${palette.q};outline-offset:1px"`)}${eq(R`\tfrac{1}{2(0.2)^2}\times\lcol{${f(WK.sse,2)}}=${f(WK.recon,1)}`)}</div>`),
  at(3,eq(R`\KL=${f(V.q3.kl_prior,2)}\qquad \lcol{L}=${f(WK.recon,1)}+${f(V.q3.kl_prior,2)}=\lcol{${f(WK.loss,1)}}`))),
 lines:['The encoder gives μ and σ for our 3; draw ε; the code is z = μ + σ ⊙ ε.',`Decode z and compare with our 3 (teal) pixel by pixel: squared error ${f(WK.sse,1)}, scaled by 1/(2σ²) = 12.5.`,'Add the KL to the prior, from tool ②: the loss for this image on this step.'],
 ask:['Why is z so close to μ?','Which term is larger, and what does that say?','What does backward() update?'],
 expected:['σ is tiny (about 0.01): the code of this 3 is pinned down precisely.','Reconstruction, by far: fitting the pixels dominates this model.','The decoder’s θ through the reconstruction; the encoder’s φ through both μ, σ (via z) and the KL.'],
 point:`Check: L − ${f(-V.loglik_const,1)} = ${f(WK.loss+V.loglik_const,1)}, close to −ELBO = ${f(-V.q3.elbo,1)}. The loss drops the constant, and one draw is noisy.`,
 transition:'The whole thing in PyTorch.',source:OURS});
S('vae-code','vae','The VAE in PyTorch: three changes to the autoencoder',{builds:2,
 body:cols(`<div class="code-xs">${code(`class VAE(AutoEncoder):          # same decoder
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
loss = (recon + kl).mean()`)}</div>`,stack(at(2,p('(1) the encoder outputs 2z numbers, (2) the code is a reparameterised draw (tool ①), (3) the loss adds the KL (tool ②).')),at(2,small('The training loop is the autoencoder’s, unchanged.'))),'1.75fr 1fr'),
 lines:['The autoencoder of Part 2, with three changes.','Everything else, the batches, Adam and the passes over the data, is the autoencoder’s loop.'],
 ask:['What are the three changes?','How many lines of the training loop change?'],
 expected:['Two outputs per code number, a random draw of z, and the KL term.','None: only the model and the loss.'],
 transition:'Both terms on the real model, live.'});
chap('5.6');
S('elbo-explorer','vae','Your turn: choose q for our 3, and watch both terms',{builds:2,
 body:widget5('elbo-explorer','Choose q for our 3: the reconstruction term, the KL to the prior, and the ELBO below log p(x).'),
 lines:['Start at the encoder’s q. Widen σ: draws from q decode to other digits, so reconstruction rises while the KL to the prior falls.','Try q = N(0, I), then q on the 1s. Every q stays below the ceiling log p(x); the red gap is the KL to the posterior.'],
 ask:['What happens to each term as σ grows?','Can any q push the ELBO above log p(x)?'],
 expected:['Reconstruction rises as the draws turn into other digits; the KL to the prior falls until σ reaches 1.','No: the gap is a KL, never negative. The encoder’s q comes within about 0.7.'],
 point:'The decoder is Lecture 2’s 2-number VAE, run in the browser. Averages over ε use 8 × 8 Gauss–Hermite quadrature, so the numbers match the notebook to about 0.01. q has one σ for both code numbers here.',
 transition:'What the trained VAE does.',source:OURS});
chap('5.7');
S('vae-samples','vae','Now sample: z ~ N(0, I), then decode',{builds:3,
 body:rows(['autoencoder, 2',digits('ae2_prior_samples',{cols:16,size:58,gap:4})],['VAE, 2',at(2,digits('vae2_prior_samples',{cols:16,size:58,gap:4}))],['VAE, 16',at(3,digits('vae16_prior_samples',{cols:16,size:58,gap:4}))]),
 lines:['The same 16 draws from N(0, I) as in Part 3, through the autoencoder’s decoder.','Through the 2-number VAE: almost every draw is a digit, and all ten digits appear.','Through the 16-number VAE: sharper and more varied. The fakes at the start of the hour came from this model.'],
 ask:['What changed between the first two rows?','Why are the VAE samples still a little blurry?','Which digits are missing from the autoencoder row?'],
 expected:['Training made the codes fill N(0, I), so random codes land on digits.','Each pixel is an independent Gaussian around a mean image; the mean of plausible strokes is soft.','Most of them: random codes reach only the digits whose codes sit near the origin.'],
 transition:'Where the codes are now.',source:OURS});
S('vae-scatter','vae','The codes now fill N(0, I)',{builds:2,
 body:cols(`<div class="state-slot"><div data-only="1">${fig(squareW([-5,5,-5,5],420),420,scatterSVG('vae2',{W:squareW([-5,5,-5,5],420),H:420,lim:[-5,5,-5,5],ticks:[[-4,-2,0,2,4],[-4,-2,0,2,4]],contours:true}).s,'VAE codes of 1,000 test digits.')}</div><div data-only="2">${fig(squareW([-5,5,-5,5],420),420,scatterSVG('vae2',{W:squareW([-5,5,-5,5],420),H:420,lim:[-5,5,-5,5],ticks:[[-4,-2,0,2,4],[-4,-2,0,2,4]],prior:500,contours:true}).s,'VAE codes with 500 draws from N(0, I).')}</div></div>`,
  stack(p(`Codes of all 10,000 test digits: mean (${f(V.agg.mean[0],1)}, ${f(V.agg.mean[1],1)}), spread (${f(V.agg.sd[0],1)}, ${f(V.agg.sd[1],1)}).`),at(2,p('Random draws (red) now land among real codes, on every digit.'))),'1.25fr 1fr'),
 lines:['The VAE’s codes (the means μ) for the same 1,000 test digits: one round cloud around the origin, digits side by side.','Overlay 500 draws from N(0, I): they fall on real codes of every digit. That is why sampling works.'],
 ask:['Compare with the autoencoder’s map. What changed?','Is the cloud exactly N(0, I)?'],
 expected:['Codes fill a round region around the origin with no large gaps; the scale is about 1, not 20.','Not quite: a little wider (spread 1.7) and slightly off-centre, since the KL only pulls codes towards N(0, I).'],
 transition:'Decode a grid, and walk between two digits.',source:OURS});
S('vae-grid','vae','Decode a grid of codes: digits everywhere',{builds:2,
 body:cols(stack(small('VAE: codes from −3 to 3 in each direction'),digits('vae2_grid',{cols:12,size:30,gap:1})),at(2,stack(small('walk in a straight line from a 1 to a 0'),digits('vae2_interp',{cols:9,size:50,gap:4}),small('the same walk, 16-number VAE'),digits('vae16_interp',{cols:9,size:50,gap:4}))),'1fr 1.1fr'),
 lines:['Every point of a 12 × 12 grid over [−3, 3]² decodes to a digit, morphing smoothly into the next.','Walk from a 1 to a 0: every step is a plausible digit. The space between images was trained, because each image owned a cloud.'],
 ask:['Compare with the autoencoder’s grid. What is different?','Why are the in-between images plausible now?'],
 expected:['No junk: the whole region N(0, I) samples from is used.','During training each image owned a small cloud of codes, so the decoder learned to draw something sensible everywhere in between.'],
 transition:'Your turn: choose codes for the VAE.',source:OURS});
S('explore-vae','vae','Your turn again: choose a code for the VAE',{builds:1,
 body:widget('explorer','Choose a code for the VAE or the autoencoder and see its decoder output.',{net:'vae2',both:'1',limae:'-6,24,-9,14',limvae:'-4.5,4.5,-4.5,4.5'}),
 lines:['Random codes, encoded test digits, or click anywhere. Switch between the VAE and the autoencoder to compare.'],
 ask:['Click in the gaps between digit groups. What do you see now?'],expected:['Plausible in-between shapes: the space between digits is filled.'],
 transition:'One knob: the weight on the KL.',source:OURS});
chap('5.8');
S('beta-what','vae','Put a weight on the KL: β',{builds:2,
 body:stack(eq(R`L_\beta=\E_{q}\Big[\tfrac{1}{2\sigma^2}\big\lVert x-f_\theta(z)\big\rVert^2\Big]+\beta\,\KL\big(q_\phi(z\mid x)\,\|\,${PRI(R`\N(0,I)`)}\big)`),p('β = 1 is the true ELBO. Changing β is the same as assuming noisier or cleaner pixels, a different σ.'),
  at(2,grid(2,card('Small β','codes free to spread: close to an autoencoder',palette.p),card('Large β','codes squeezed towards the prior: closer to N(0, I), less detail',palette.q)))),
 lines:['Weight the KL by β, as in the β-VAE. β = 1 is minus the ELBO.','Predict before we look: what happens at β = 0, and at very large β?'],
 ask:['Which value of β gives back the real ELBO?','What do you predict at the two extremes?'],
 expected:['β = 1.','β = 0: codes spread out like an autoencoder’s. Huge β: codes collapse onto the prior and stop carrying information.'],
 transition:'Seven trained VAEs.'});
S('beta','vae','Seven VAEs, β from 0 to 256: try them',{builds:1,
 body:widget('beta','Codes, reconstructions and samples of VAEs trained with different KL weights.',{beta:'1'}),
 lines:['Click through β. Watch how spread the codes are, the reconstructions, and the samples.'],
 ask:['At which β do samples look best? When do reconstructions start to blur?'],
 expected:[`Around β = 1 to 4. By β = 64 reconstructions blur; at 256 the KL is ${f(BETA['256'].test_kl,2)}, every code decodes to the same blob (posterior collapse).`],
 point:`KL per image: β = 0 gives ${f(BETA['0'].test_kl,1)}, β = 1 gives ${f(BETA['1'].test_kl,1)}, β = 256 gives ${f(BETA['256'].test_kl,2)}. The optional slides take the two extremes apart.`,
 transition:'Wrap-up.',source:OURS});
// ================================================================= Part 6 · Putting it together
partOpener(6,'end','Every piece of the lecture on one page: the VAE’s loss, term by term, traced back to KL and the ELBO.',
 'Which Lecture 1 ideas does the VAE’s loss use?','The ELBO, the KL and its formula for Gaussians, and the reparameterisation trick.');
chap('6.1');
S('maths-chain','end','The whole lecture in five lines of maths',{builds:5,
 body:`<div class="mchain"><div class="mc-h"></div><div class="mc-h"></div><div class="mc-h">the maths</div><div class="mc-h">where we met it</div>${[
  ['①','KL',R`\KL(q\,\|\,p)=\E_{q}\big[\log q-\log p\big]\ \ge\ 0`,'Lecture 1, 2.4'],
  ['②','the budget',R`\dcol{\log p_\theta(x)}=\qcol{\ELBO(x)}+\lcol{\KL\big(q_\phi(z\mid x)\,\|\,p(z\mid x)\big)}`,'Lecture 1, 4.1 · today 5.4'],
  ['③','its two terms',R`\qcol{\ELBO(x)}=\E_q\big[\ncol{\log p_\theta(x\mid z)}\big]-\KL\big(q_\phi(z\mid x)\,\|\,${PRI('p(z)')}\big)`,'Lecture 1, 4.1 · today 5.4'],
  ['④','Gaussian pixels',R`-\ncol{\log p_\theta(x\mid z)}=\tfrac{1}{2\sigma^2}\lVert x-f_\theta(z)\rVert^2+\text{const}`,'today 5.2 and 2.3'],
  ['⑤','the two tools',R`z=\mu+\sigma\odot\epsilon,\qquad\KL\ \text{to}\ ${PRI(R`\N(0,1)`)}=\tfrac12\big(\mu^2+\sigma^2-\log\sigma^2-1\big)`,'Lecture 1, 3.5, 2.6 · today 4.2–4.3']
 ].map(([n,w,t,src],k)=>{const c=x=>k?`<div data-at="${k+1}">${x}</div>`:`<div>${x}</div>`;return c(`<span class="mc-n">${n}</span>`)+c(`<span class="mc-w">${w}</span>`)+c(m(t))+c(`<span class="mc-s">${src}</span>`);}).join('')}</div>`,
 lines:['① Lecture 1’s KL: the average extra cost of using q where p is right. Never negative.',
  '② The budget for one image: log p(x) splits into the ELBO and a KL to the posterior. So the ELBO is a lower bound on log p(x).',
  '③ The ELBO’s two terms: rebuild x from codes drawn from q, and keep q near the prior.',
  '④ With Gaussian pixels, the first term is the autoencoder’s squared error, scaled by 1/2σ².',
  '⑤ The two tools make the loss computable: draw z by reparameterisation, and take the KL by formula.'],
 ask:['Why is the KL never negative?','Why does ② make the ELBO a lower bound?','Which term would an autoencoder keep?','Where does the 1/2σ² come from?','Which of the two terms needs sampling?'],
 expected:['The right code is never beaten on average: Lecture 1, 2.4.','The KL on the right is ≥ 0, so the ELBO is at most log p(x).','The first, the reconstruction term.','The Gaussian density of each pixel around the decoder’s output, with σ = 0.2.','Only the reconstruction term, through z = μ + σ ⊙ ε. The KL has a formula.'],
 transition:'The same five lines as one picture.'});
S('maths-map','end','Where each term of the VAE’s loss comes from',{builds:4,
 body:lossMap(),
 lines:['The loss we train, written out: a reconstruction term plus a KL term.',
  'The reconstruction term: Gaussian pixels turn −log p(x | z) into the autoencoder’s squared error, and tool ① lets gradients through the draw.',
  'The KL term: a KL from q to the prior N(0, I), which tool ② gives by formula, and which is never negative.',
  'Together they are minus Lecture 1’s ELBO for one image. Making the loss smaller pushes up a lower bound on log p(x).'],
 ask:['Which term is the autoencoder’s loss?','Why is the draw of z written as μ + σ ⊙ ε?','What would happen without the KL term?','Why does a smaller loss help log p(x)?'],
 expected:['The reconstruction term, divided by 2σ².','So the gradient reaches μ and σ: Lecture 1’s reparameterisation trick.','The clouds shrink to points and spread out: an autoencoder again, which cannot generate (Part 3).','The loss is minus the ELBO, a lower bound on log p(x): raising the bound raises what it can certify, and the gap is the KL to the posterior.'],
 transition:'Wrap-up.'});
chap('6.2');
S('summary','end','Autoencoder and VAE, side by side',{builds:2,
 body:stack(`<div class="table-sm">${table(['','autoencoder','VAE'],[
  ['the encoder outputs','a code z: 2 numbers','a cloud: μ and log σ², that is q<sub>φ</sub>(z | x)'],
  ['the decoder gets','z','a draw z = μ + σ ⊙ ε (tool ①)'],
  ['the loss','squared error','squared error / 2σ² + KL to N(0, I) (tool ②) = −ELBO + const'],
  ['random z ~ N(0, I) gives','mostly junk','digits'],
  ['can it score an image?','no','yes, through the ELBO, a lower bound on log p(x)']])}</div>`,
  at(2,takeaway('A VAE is an autoencoder trained with Lecture 1’s objective. The KL to the prior makes the codes fill N(0, I), so random codes decode to digits.'))),
 lines:['Same two networks, two changes: a cloud instead of a point, and a KL to the prior in the loss.','Both changes come from Lecture 1: the ELBO with a code per image, the reparameterisation trick, and the Gaussian KL.'],
 ask:['Which row explains why sampling works?','Which Lecture 1 ideas did the VAE use?'],
 expected:['The loss row: the KL pulls codes into N(0, I), where random draws land.','Variational inference and the ELBO, reparameterisation, and the closed-form KL between Gaussians.'],
 transition:'What comes next.'});
S('next','end','Next: can we drop the pixel likelihood altogether?',{builds:2,
 body:stack(p('VAE samples are soft because each pixel is an independent Gaussian around a mean image.'),at(2,p('Next: a generator trained against a second network that judges real from fake (GANs). Then many small denoising steps (diffusion).'))),
 lines:['The pixel-wise Gaussian likelihood is the VAE’s weak point: it rewards the safe average stroke.','Lectures 3 and 4 replace it: a learned critic, then repeated denoising.'],
 ask:['Why does a pixel-wise Gaussian encourage soft images?','What else could tell a generator its images are wrong?'],
 expected:['Squared error favours the mean of plausible images, which is blurry.','A classifier trained to tell real from generated: a learned critic.'],
 transition:'End of the main route. Optional slides follow.'});

// ================================================================= Optional depth
O('opt-ae-test','Did the autoencoder memorise? Training against test error',{builds:2,
 body:cols(table(['2-number code','training error','test error'],[['autoencoder',f(SW['2'].train_sse,1),f(SW['2'].test_sse,1)],['PCA','',f(M.pca.test_sse['2'],1)]]),
  stack(p('Test digits were never used for training.'),at(2,p('Training and test errors are close: it learned digits in general, not these particular 60,000.'))),'1fr 1fr'),
 lines:['Average squared error per image after training, on the 60,000 training digits and the 10,000 test digits.',`The gap is small (${f(SW['2'].train_sse,1)} against ${f(SW['2'].test_sse,1)}), and the autoencoder beats PCA by a wide margin with the same 2 numbers.`],
 ask:['What would memorising look like in these numbers?','Why is the test error a little higher?'],
 expected:['Very low training error and much higher test error.','The weights were tuned to the training digits; new digits fit a little less well.'],source:OURS});
O('opt-linear-ae','Without the ReLUs and the sigmoid, an autoencoder rediscovers PCA',{builds:2,
 body:cols(rows(['8 test digits',digits('watch',{cols:8,size:60,gap:5})],['PCA, 2 numbers',digits('pca_recon_watch_2',{cols:8,size:60,gap:5})],['linear AE, 2',digits('lin2_recon_watch',{cols:8,size:60,gap:5})]),
  stack(p(`Test error: linear autoencoder ${f(A.lin2_test_sse,1)}, PCA ${f(M.pca.test_sse['2'],1)}.`),at(2,p(`Its decoder spans almost the same plane as PCA’s two directions (cosines of the angles: ${f(A.lin2_cos_angles[0],3)} and ${f(A.lin2_cos_angles[1],2)}).`))),'1.45fr 1fr'),
 lines:['Train the same way with straight-line maps. The reconstructions match PCA’s.','Gradient descent found PCA’s plane on its own: an autoencoder is PCA with bendable maps.'],
 ask:['Why does a linear autoencoder end up like PCA?','What do the ReLUs buy?'],
 expected:['PCA is the best linear encoder–decoder pair under squared error; training finds it, up to a rotation within the plane.','Curved maps: a better reconstruction from the same few numbers.'],source:OURS});
O('opt-ae-interp','The autoencoder: walk in a straight line from a 1 to a 0',{builds:1,
 body:rows(['ends',digits('interp_ends',{cols:2,size:60,gap:6})],['2 numbers',digits('ae2_interp',{cols:9,size:60,gap:6})],['16 numbers',digits('ae16_interp',{cols:9,size:60,gap:6})]),
 lines:['With 2 numbers the path crosses the clusters of other digits; with 16 the middle passes through blobs that are no digit.'],
 ask:['Why might the 16-number path cross empty space?'],expected:['The straight line between two codes need not stay where real codes are; in 16 dimensions most of the space between them is empty.'],source:OURS});
O('opt-q3','Check the encoder: its q for our 3, against the true posterior',{builds:2,
 body:cols(fig(420,420,posteriorSVG({S:420,view:'zoom',enc:true,q:[V.vi3.final_mu[0],V.vi3.final_mu[1],V.vi3.final_sd[0],V.vi3.final_sd[1]]}).s,'The posterior, the per-image q and the encoder’s q, zoomed.'),
  stack(`<div class="tsm">${table(['','value'],[['log p(x), the exact target',f(V.grid.log_px,2)],['ELBO, q fitted to this image alone',f(V.vi3.final_elbo,2)],['ELBO, q from the encoder (one pass)',f(V.q3.elbo,2)]])}</div>`,at(2,small('Dashed: the encoder’s q. Shaded: the per-image fit. Both 2 standard deviations.'))),'1fr 1.1fr'),
 lines:[`The per-image fit gets within ${f(V.grid.log_px-V.vi3.final_elbo,2)} of log p(x); the encoder, in one pass, within ${f(V.q3.gap,2)}.`,'The encoder’s q sits slightly off-centre and a little too narrow: the price of one network serving all 60,000 images.'],
 ask:['Which q is better, and by how much?','Why is the encoder’s q not perfect?'],
 expected:[`The per-image fit, by ${f(V.vi3.final_elbo-V.q3.elbo,2)} nats.`,'One network must serve every image; it cannot be exactly right for each.'],source:OURS});
O('opt-vae-curves','While training: reconstruction falls, the KL settles',{builds:1,
 body:fig(1100,320,lineChart({W:540,H:320,xMin:1,xMax:50,yMin:300,yMax:600,ticksX:[1,25,50],ticksY:[300,400,500,600],xLabel:'epoch',yLabel:'reconstruction term',series:[{x:V.curve.epoch,y:V.curve.recon,color:palette.noise}]})+lineChart({ox:560,W:540,H:320,xMin:1,xMax:50,yMin:0,yMax:12,ticksX:[1,25,50],ticksY:[0,4,8,12],xLabel:'epoch',yLabel:'KL term',series:[{x:V.curve.epoch,y:V.curve.kl,color:palette.q}]}),'The two terms of the VAE loss during training.'),
 lines:[`After 50 epochs: reconstruction ${f(V.curve.recon.at(-1),0)}, KL ${f(V.curve.kl.at(-1),1)} per image. The KL never falls to zero: then the code would say nothing about the image.`],
 ask:['What would a KL of zero mean?'],expected:['q equals the prior for every image: the code ignores the image. That is the β = 256 collapse.'],source:OURS});
O('opt-beta-zero','β = 0: is it just an autoencoder?',{builds:2,
 body:cols(table(['β','KL per image','average σ of q','spread of codes'],['0','1','256'].map(b=>[b,f(BETA[b].test_kl,1),f(BETA[b].sd_mean,4),f(BETA[b].code_spread,2)])),stack(p('With β = 0 nothing stops q from shrinking.'),at(2,p(`It still draws z = μ + σε, unlike an autoencoder. But training drives σ to ${f(BETA['0'].sd_mean,4)}, so it behaves like one.`))),'1.3fr 1fr'),
 lines:['β = 0 keeps the random draw z = μ + σε, so by construction it is a different model from an autoencoder.','But training drives σ almost to zero, and then it behaves like one: spread-out codes and poor samples.'],
 ask:['Is a VAE with β = 0 the same as an autoencoder?','What does the KL term do to σ?'],
 expected:['Different by construction, but in practice it learns σ ≈ 0 and becomes one.','It stops σ shrinking: −log σ² grows as σ → 0.'],source:OURS});
O('opt-beta-collapse','β = 256: every code decodes to the same blur',{builds:2,
 body:cols(rows(['β = 256, rebuilt',digits('beta256_recon_watch',{cols:8,size:58,gap:5})],['β = 256, samples',digits('beta256_prior_samples',{idx:[0,1,2,3,4,5,6,7],cols:8,size:58,gap:5})]),
  stack(p(`KL = ${f(BETA['256'].test_kl,2)}: q(z | x) equals the prior for every image.`),at(2,p('The decoder learns to ignore z and outputs the average digit. This is called <b>posterior collapse</b>.'))),'1.5fr 1fr'),
 lines:['With a huge weight on the KL, the cheapest solution is q = N(0, I) for every image.','Then z carries no information, so the decoder ignores it and draws the average digit every time.'],
 ask:['Why is KL = 0 the cheapest solution here?','How would you spot collapse during training?'],
 expected:['β × KL dominates the loss; reconstruction error is worth less than any information in the code.','The KL term falls to about 0 and samples all look the same.'],source:OURS});

// ================================================================= Lecture map, filled in once every section knows its first slide
S('contents','map','Lecture 5 at a glance: six parts, twenty-five sections',{nb:'top',
 body:'',
 lines:['Six parts, twenty-five numbered sections, about 85 minutes. The ▶ Notebook link on every slide opens its section in Colab.'],
 ask:['Which part leans most on Lecture 1?'],
 expected:['Part 5: the VAE’s loss is Lecture 1’s ELBO with a code per image. Part 6 traces every term back to it.'],
 point:'A 60-minute cut: in Part 1 skip next-word, img2img, speech and science; in Part 2 skip ae-untrained, ae-step and code-size; in Part 3 skip ae-grid; keep all of Part 4; in Part 5 skip posterior3, vae-code, explore-vae and beta-what. The optional slides carry the detours, and Lecture 2 has the full two-session version.',
 transition:'Part 1: why generate?'});
for(const [sl,k] of partOpeners){
 const figs={6:budgetFig(true),1:`<div class="photo-row" style="grid-template-columns:2.1fr 1fr 1fr 1fr">${photo('assets/gallery/stylegan-faces.jpg','faces (StyleGAN)')+photo('assets/examples/t2i-sample-1.jpg','text to image')+photo('assets/gallery/spade-out-0.jpg','a map to a landscape')}<figure>${digits('vae16_prior_samples',{idx:[0,1,2,3],cols:2,size:76,gap:6})}<figcaption>digits: today</figcaption></figure></div>`,2:maxH(250,aeDiagram({stage:4})),3:genDiagram('ae'),4:fixesFig(),5:castFig()};
 sl.body=act(k,figs[k],`<ol class="pb-grid">${CHAPTERS.filter(([n])=>n.startsWith(k+'.')).map(([n,t])=>`<li><span class="pb-num">${n}</span><a href="#${firstOf[n]}/1">${t}</a>${ANCHORS.size?`<a class="toc-nb" href="${COLAB}#scrollTo=${secAnchor(n)}" target="_blank" rel="noopener" aria-label="notebook ${n}">▶</a>`:''}</li>`).join('')}</ol>`);
}
slides.unshift(slides.pop());
slides[0].body=`<div class="toc toc-flow">${PARTS.map(([k,t])=>`<h3><a href="#part-${k}/1">Part ${k} · ${t}</a></h3><ol>${CHAPTERS.filter(([num])=>num.startsWith(k+'.')).map(([num,title])=>
 `<li><a href="#${firstOf[num]}/1">${num} ${title}</a>${ANCHORS.size?`<a class="toc-nb" href="${COLAB}#scrollTo=${secAnchor(num)}" target="_blank" rel="noopener" aria-label="notebook ${num}">▶</a>`:''}</li>`).join('')}</ol>`).join('')}</div>`;
for(const [num] of CHAPTERS)if(!firstOf[num])throw new Error('Section '+num+' has no slides');
