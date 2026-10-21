import {m,eq,at,only,cols,code,image,table,flow,plot,palette} from '../shared/author.js';
import {panel,label,legend,fig,widget,thetaBars,PARAM_NAMES,TINT,MUTED,AX,num,lin,covEllipse,matrix2,weather,mount as mountWidgets} from './widgets.js';
import {normalPdf,normalKL} from './math.mjs';

const d=await fetch(new URL('./data/science.json',import.meta.url),{cache:'no-cache'}).then(r=>r.json());
const ANCHORS=new Set(await fetch(new URL('./data/notebook-anchors.json',import.meta.url),{cache:'no-cache'}).then(r=>r.json()));
export const mount=(stage,ctx)=>mountWidgets(stage,ctx,d);
const R=String.raw;
const f=(v,n=4)=>Number(v).toFixed(n).replace('-','−');
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const n=d.normal,c=d.coin,g=d.gradients,mc=d.monte_carlo,st=d.story,b=st.bnn,ink=palette.ink;
const A=b.theta_A,Bt=b.theta_B,E=b.evidence,H=b.hmc,V=b.vi,VF=b.vi_full;
export const meta={
 id:'lecture1',number:1,title:'KL and variational inference',
 subtitle:'Bayes for a neural network, KL from scratch, and the ELBO',minutes:140,
 notebook:'lecture1-variational-inference.ipynb',
 sections:[
  {id:'map',title:'Lecture map',minutes:1},
  {id:'bayes',title:'Part 1 · Bayes’ rule for a neural network',minutes:22},
  {id:'kl',title:'Part 2 · How far apart are two distributions?',minutes:74},
  {id:'fit',title:'Part 3 · Find the closest q',minutes:22},
  {id:'elbo',title:'Part 4 · Back to Bayes: the ELBO',minutes:13},
  {id:'impl',title:'Part 5 · Implement it',minutes:8},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
/** Numbered sections. The notebook uses the same numbers, with anchors sec-1-1 … sec-5-2. */
const CHAPTERS=[
 ['1.1','Bayes’ rule'],['1.2','A tiny network and its prior'],['1.3','Likelihood and training'],['1.4','The evidence is intractable'],['1.5','Approximating the posterior'],
 ['2.1','Surprise'],['2.2','Entropy: average surprise'],['2.3','Codes: entropy in bits'],['2.4','Cross-entropy and KL'],['2.5','Classification loss'],['2.6','Maximum likelihood'],['2.7','KL at work: distillation, RLHF'],['2.8','Closed forms and PyTorch'],['2.9','KL by sampling'],
 ['3.1','Search a grid'],['3.2','Gradient descent'],['3.3','Covariance in 2-D'],['3.4','Mixtures: no formula'],['3.5','The reparameterisation trick'],['3.6','Which direction of KL?'],
 ['4.1','Deriving the ELBO'],['4.2','Checking it on a coin'],
 ['5.1','VI for the tiny network'],['5.2','Wrap-up'],
];
const PARTS=[['1','Bayes’ rule for a neural network'],['2','How far apart are two distributions?'],['3','Find the closest q'],['4','Back to Bayes: the ELBO'],['5','Implement it']];
const COLAB='https://colab.research.google.com/github/nipunbatra/generative/blob/main/lectures/lecture1/lecture1-variational-inference.ipynb';
const secAnchor=k=>`sec-${k.replace('.','-')}`;
let chapter=null;const firstOf={};
const chap=k=>{chapter=CHAPTERS.find(ch=>ch[0]===k);if(!chapter)throw new Error('Unknown section '+k);};
/** A link to one notebook cell in Colab; the anchor must exist in the built notebook. */
function nbLink(anchor,number){
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
const ACTS=['Bayes for a network','Distance and KL','Find the closest q','Back to Bayes','Implement it'];
const acts=k=>`<div class="act-strip" role="img" aria-label="Lecture map: part ${k} of 5">${ACTS.map((a,i)=>`<span class="${i+1<k?'done':i+1===k?'now':''}">${i+1} · ${a}</span>`).join('')}</div>`;
const act=(k,...a)=>`${acts(k)}<div class="act-body">${a.join('')}</div>`;
const grid3=(...a)=>`<div class="image-pair" style="grid-template-columns:repeat(3,minmax(0,1fr));gap:20px">${a.join('')}</div>`;
const TH={w1:'w₁',w2:'w₂',b1:'b₁',b2:'b₂',v1:'v₁',v2:'v₂',c:'c'};
const thin=(w,o=1)=>`style="stroke-width:${w};stroke-opacity:${o}"`;

// ---------------------------------------------------------------- figures kept from the previous version
function normalPlot(keys=['q1'],extra={}){
 const a=d.plots.normal;
 const series=[{label:'p · fixed target',role:'p',x:a.x,y:a.p}];
 keys.forEach(k=>series.push({label:k==='q1'?'q₁':k==='q2'?'q₂':`q · step ${k}`,role:'q',x:a.x,y:a[k]||a.snapshots[k],dashed:k==='q2'}));
 return plot(series,{width:640,height:340,xMin:-11,xMax:13,yMin:0,yMax:.23,xLabel:'θ',yLabel:'density',
  ticksX:[-10,-5,0,5,10],ticksY:[0,.1,.2],...extra,yLabel:''});
}
function coinPlot(index=0,snapshot=null,extra={}){
 const a=d.plots.coin;
 return plot([{label:'known posterior',role:'p',x:a.x,y:a.posterior},
  {label:snapshot===null?`q · Beta(${c.candidates[index].a},${c.candidates[index].b})`:`q · step ${snapshot}`,
   role:'q',x:a.x,y:snapshot===null?a.candidates[index]:a.snapshots[snapshot],dashed:true}],
  {xMin:0,xMax:1,yMin:0,yMax:6.5,xLabel:'r · chance of heads',yLabel:'density',ticksX:[0,.5,1],ticksY:[0,3,6],...extra});
}
function threeWay(k){
 const t=st.three,xs=t.plot.x,P=panel({W:370,H:300,L:14,R:12,T:60,B:52,xMin:-6,xMax:6,yMax:.44,ticksX:[-4,0,4],yAxis:false});
 let inner=P.axes+P.area(xs,t.plot.p,TINT.p)+P.line(xs,t.plot.p,palette.p);
 if(k==='laplace')inner+=P.line(xs,t.plot.laplace,palette.noise);
 if(k==='vi')inner+=P.line(xs,t.plot.vi,palette.q);
 if(k==='mcmc'){const e=t.mcmc.edges;t.mcmc.hist.forEach((h,i)=>{const x0=P.x(e[i]),x1=P.x(e[i+1]);inner+=`<rect x="${(x0+1).toFixed(1)}" y="${P.y(h).toFixed(1)}" width="${(x1-x0-2).toFixed(1)}" height="${(P.y(0)-P.y(h)).toFixed(1)}" fill="${palette.data}" fill-opacity=".45"/>`;});}
 const title={laplace:['Laplace',palette.noise],mcmc:['MCMC',palette.data],vi:['Variational inference',palette.q]}[k];
 return fig(370,300,inner+label(14,30,title[0],{fill:title[1],size:24,weight:600}),`${title[0]} approximation of a two-bump target.`);
}
function gridDots(){
 let row='',square='';
 for(let i=0;i<100;i++)row+=`M${(14+i*2.9).toFixed(1)},150h1.6v1.6h-1.6z`;
 for(let i=0;i<50;i++)for(let j=0;j<50;j++)square+=`M${(420+i*4.6).toFixed(1)},${(56+j*4.6).toFixed(1)}h2v2h-2z`;
 return fig(700,330,label(10,32,'1 unknown',{fill:palette.ink,size:24,weight:600})+`<path d="${row}" fill="${palette.ink}"/>`+
  label(10,230,'100 points',{size:22})+label(410,32,'2 unknowns',{fill:palette.ink,size:24,weight:600})+`<path d="${square}" fill="${palette.ink}"/>`+
  label(410,318,'100 × 100 = 10,000 points',{size:22}),'A grid of 100 points for one unknown, and 10,000 for two.');
}
function evidenceChart(){
 const k=st.coinkl.kl,ns=lin(0,30,61),P=panel({W:600,H:330,L:74,R:14,xMin:0,xMax:30,yMax:3,ticksX:[0,10,20,30],ticksY:[0,1,2,3],fmtY:v=>['1×','10×','100×','1000×'][v],xLabel:'tosses of the 0.8 coin'});
 const pt=(nn,text)=>P.dot(nn,nn*k/Math.LN10,palette.loss)+label(P.x(nn)-12,P.y(nn*k/Math.LN10)-14,text,{fill:palette.loss,size:22,anchor:'end',weight:600});
 return fig(600,330,P.axes+P.line(ns,ns.map(x=>x*k/Math.LN10),palette.loss)+label(74,30,'typical p(𝒟 | 0.8) / p(𝒟 | 0.5)',{fill:palette.loss,size:21})+
  `<g data-at="2">${pt(10,`${num(st.coinkl.x10,1)}×`)+pt(30,`${Math.round(st.coinkl.x30)}×`)}</g>`,'The evidence ratio grows exponentially with the number of tosses.');
}
function tangentChart(){
 const ts=lin(.05,3.2,200),P=panel({W:520,H:330,L:16,R:14,T:40,xMin:0,xMax:3.2,yMin:-2.6,yMax:2.4,ticksX:[2,3],ticksAtAxis:true,yAxis:false,xLabel:'t = p / q'});
 return fig(520,330,P.axes+P.line(ts,ts.map(t=>t-1),MUTED,'stroke-dasharray="8 6"')+P.line(ts,ts.map(Math.log),palette.q)+P.dot(1,0,palette.ink)+
  label(P.x(2.55),P.y(2.05),'t − 1',{fill:MUTED,size:23,anchor:'middle'})+label(P.x(2.75),P.y(.62),'log t',{fill:palette.q,size:23,anchor:'middle'})+
  label(P.x(.1),P.y(.42),'touch at t = 1',{fill:palette.ink,size:20}),'The curve log t lies below its tangent line t minus 1, touching at t = 1.');
}
function mixChart(){
 const a=d.plots.mixture,xs=a.x.filter(x=>x>=-4&&x<=8),py=xs.map(x=>a.p[a.x.indexOf(x)]),qy=xs.map(u=>normalPdf(u,{mu:3,sigma:1}));
 const P=panel({W:560,H:320,L:16,R:14,xMin:-4,xMax:8,yMax:.62,ticksX:[-4,0,4,8],yAxis:false,xLabel:'θ'});
 return fig(560,320,P.axes+P.area(xs,py,TINT.p)+P.line(xs,py,palette.p)+P.line(xs,qy,palette.q)+legend(16,30,[['p, the PML mixture',palette.p],['q = N(3, 1²)',palette.q]]),'A two-bump mixture target and a Gaussian q.');
}
function gapLine(row){
 const xmin=-22.2,xmax=-21.2,W=1000,L=65,Rm=45,x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-Rm);
 const a=x(row.elbo),b=x(c.log_evidence),close=b-a<110;
 return `<svg viewBox="0 0 1000 144" role="img" aria-label="ELBO and evidence on one fixed number line; their distance is the posterior KL.">
 <line x1="${L}" y1="69" x2="${W-Rm}" y2="69" stroke="#aab3bc"/>
 <circle cx="${a}" cy="69" r="5" fill="${palette.q}"/><circle cx="${b}" cy="69" r="5" fill="${palette.data}"/>
 <text x="${a}" y="49" text-anchor="${close?'end':'middle'}" fill="${palette.q}" font-size="24">ELBO ≈ ${f(row.elbo,3)}</text>
 <text x="${W-Rm}" y="25" text-anchor="end" fill="${palette.data}" font-size="24">log p(𝒟) ≈ ${f(c.log_evidence,3)}</text>
 <path d="M${a},78 V104 H${b} V78" fill="none" stroke="${palette.loss}" stroke-width="2"/>
 <text x="${(a+b)/2}" y="133" text-anchor="middle" fill="${palette.loss}" font-size="24">gap = KL ${row.posterior_kl<1e-10?'= 0':'≈ '+f(row.posterior_kl,3)}</text>
 <text x="${L}" y="96" font-size="20" fill="#68717b">−22.2</text><text x="${W-Rm}" y="96" text-anchor="end" font-size="20" fill="#68717b">−21.2 nats</text></svg>`;
}

// ---------------------------------------------------------------- Part 1 figures: the tiny network
function nnDiagram(theta=null){
 const W=640,H=330,X=[70,320,570],yH=[95,235],yO=165,rIn=34,rH=42;
 const node=(cx,cy,r,text,color,size=26)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${palette.paper}" stroke="${color}" stroke-width="3"/>`+label(cx,cy+9,text,{fill:color,size,anchor:'middle'});
 const edge=(x1,y1,x2,y2,name,lx,ly)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${palette.noise}" stroke-width="3"/>`+label(lx,ly,name,{fill:palette.noise,size:26,anchor:'middle',weight:600});
 const bias=(x,y,text)=>label(x,y,text,{fill:palette.noise,size:24,anchor:'middle',weight:600});
 const nv=(name,j)=>theta?`${name} = ${num(theta[j])}`:name;
 let s=edge(X[0]+rIn,yO,X[1]-rH,yH[0],nv('w₁',0),185,108)+edge(X[0]+rIn,yO,X[1]-rH,yH[1],nv('w₂',1),185,242)+
  edge(X[1]+rH,yH[0],X[2]-rIn,yO,nv('v₁',4),455,108)+edge(X[1]+rH,yH[1],X[2]-rIn,yO,nv('v₂',5),455,242);
 s+=node(X[0],yO,rIn,'x',palette.data)+node(X[1],yH[0],rH,'tanh',ink,22)+node(X[1],yH[1],rH,'tanh',ink,22)+node(X[2],yO,rIn,'f',palette.q);
 s+=bias(X[1],yH[0]-54,`+ ${nv('b₁',2)}`)+bias(X[1],yH[1]+70,`+ ${nv('b₂',3)}`)+bias(X[2],yO-50,`+ ${nv('c',6)}`);
 s+=label(X[0],yO+62,'input',{size:20,anchor:'middle'})+label(X[2],yO+62,'output',{size:20,anchor:'middle'});
 return fig(W,H,s,'A network with one input, two tanh hidden units and one output. Its seven parameters are the weights w1, w2, v1, v2 and the biases b1, b2, c.');
}
const fxPanel=(o={})=>panel({W:600,H:330,L:52,R:14,xMin:-2.5,xMax:2.5,yMin:-2.6,yMax:2.6,ticksX:[-2,-1,0,1,2],ticksY:[-2,0,2],xLabel:'x',...o});
const dataDots=(P,r=5.5)=>b.x.map((x,k)=>`<circle cx="${P.x(x).toFixed(1)}" cy="${P.y(b.y[k]).toFixed(1)}" r="${r}" fill="${palette.data}"/>`).join('');
const sub=t=>`<tspan baseline-shift="sub" font-size="72%">${t}</tspan>`;
/** Seven N(0, 1) bells, one per parameter, with one draw of θ marked on each. */
function bellsRow(theta){
 const W=1150,H=200,PW=150,G=(W-7*PW)/6,us=lin(-3.4,3.4,171),prior=us.map(u=>normalPdf(u,{mu:0,sigma:1}));let s='';
 b.names.forEach((nm,j)=>{
  const ox=j*(PW+G),P=panel({ox,W:PW,H,L:6,R:6,T:70,B:40,xMin:-3.4,xMax:3.4,yMax:.45,yAxis:false,ticksX:[-2,0,2]});
  const v=Math.max(-3.3,Math.min(3.3,theta[j])),top=normalPdf(v,{mu:0,sigma:1});
  s+=P.axes+P.area(us,prior,'#E5E7EB')+P.line(us,prior,'#9CA3AF',thin(2))+label(ox+PW/2,24,TH[nm],{fill:palette.noise,size:23,anchor:'middle',weight:600})+
   `<line x1="${P.x(v).toFixed(1)}" y1="${P.y(0).toFixed(1)}" x2="${P.x(v).toFixed(1)}" y2="${P.y(top).toFixed(1)}" stroke="${palette.noise}" stroke-width="3"/>`+P.dot(v,0,palette.noise,6.5)+
   label(Math.max(ox+22,Math.min(ox+PW-22,P.x(v))),56,num(theta[j]),{fill:palette.noise,size:19,anchor:'middle',weight:600});
 });
 return fig(W,H,s,'Seven N(0, 1) priors, one per parameter, with one draw of θ marked on each.');
}
/** Three prior draws as rows of seven numbers, each revealed with its curve. */
function oneDrawFigure(){
 const colors=[palette.noise,'#475569',palette.loss],W=1150,H=330,cx=j=>118+j*62;
 const P=panel({ox:550,W:600,H,L:52,R:14,T:56,B:56,xMin:-2.5,xMax:2.5,yMin:-4.2,yMax:4.2,ticksX:[-2,-1,0,1,2],ticksY:[-4,-2,0,2,4],xLabel:'x'});
 let s=P.axes+label(602,30,'f(x) for each draw',{fill:MUTED,size:21});
 PARAM_NAMES.forEach((nm,j)=>{s+=label(cx(j),96,nm,{fill:palette.noise,size:21,anchor:'middle',weight:600});});
 s+=`<line x1="0" y1="112" x2="520" y2="112" stroke="${AX}"/>`;
 [0,1,2].forEach(k=>{const y=154+k*56,c=colors[k];let row=label(0,y,`θ⁽${'¹²³'[k]}⁾`,{fill:c,size:23,weight:600});
  b.prior_thetas[k].forEach((v,j)=>{row+=label(cx(j),y,num(v),{fill:c,size:20,anchor:'middle'});});
  s+=`<g data-at="${k+1}">${row}${P.line(b.grid,b.prior_draws[k],c,thin(4))}</g>`;});
 s+=label(0,318,'Each row, put through the network, is one curve.',{size:19});
 return fig(W,H,s,'Three draws of θ from the prior as rows of seven numbers, and the function each one gives.');
}
function thetaACurve(){
 const P=fxPanel();
 return fig(600,330,P.axes+P.line(b.grid,b.f_A,palette.q)+dataDots(P)+label(52,30,`the network with θ${sub('A')}, and the data`,{fill:palette.q,size:21}),'The network with theta A drawn over the data.');
}
/** A one-parameter version of the evidence: only w₂ unknown, the other six fixed at θ_A. */
const W2=(()=>{
 const th=A.theta,lnN=(y,mu,sd)=>-.5*Math.log(2*Math.PI*sd*sd)-(y-mu)**2/(2*sd*sd);
 const logg=w=>{const t=[...th];t[1]=w;return b.x.reduce((a,x,i)=>a+lnN(b.y[i],t[4]*Math.tanh(t[0]*x+t[2])+t[5]*Math.tanh(t[1]*x+t[3])+t[6],b.noise_sd),0)+lnN(w,0,1);};
 const ws=lin(-3.2,-.8,241),top=Math.max(...ws.map(logg)),n=20,dw=2.4/n;
 return {ws,g:ws.map(w=>Math.exp(logg(w)-top)),n,dw,rect:Array.from({length:n},(_,k)=>{const w=-3.2+(k+.5)*dw;return [w,Math.exp(logg(w)-top)];})};
})();
function evidenceArea(){
 const P=panel({W:600,H:330,L:16,R:14,xMin:-3.2,xMax:-.8,yMax:1.18,ticksX:[-3,-2.5,-2,-1.5,-1],yAxis:false,xLabel:'w₂'});
 const rects=W2.rect.map(([w,h])=>`<rect x="${P.x(w-W2.dw/2).toFixed(1)}" y="${P.y(h).toFixed(1)}" width="${(P.x(W2.dw)-P.x(0)).toFixed(1)}" height="${(P.y(0)-P.y(h)).toFixed(1)}" fill="${TINT.q}" stroke="${palette.q}" stroke-width="1.5"/>`).join('');
 return fig(600,330,P.axes+`<g data-only="1">${P.area(W2.ws,W2.g,TINT.p)}</g><g data-at="2">${rects}</g>`+P.line(W2.ws,W2.g,palette.p)+
  label(16,30,`p(𝒟 | w₂) p(w₂), the other six fixed at θ${sub('A')}`,{fill:palette.p,size:21}),'The evidence for one unknown weight is the area under a curve, approximated by twenty rectangles.');
}
function dataChart(){
 const P=fxPanel();
 return fig(600,330,P.axes+dataDots(P)+label(52,30,'17 training points (x, y)',{fill:palette.data,size:21}),'Seventeen training points, with a gap between x = −0.4 and x = 0.5.');
}
function twoThetasChart(){
 const P=fxPanel();
 return fig(600,330,P.axes+P.line(b.grid,b.f_B,palette.noise)+P.line(b.grid,b.f_A,palette.q)+dataDots(P)+legend(52,30,[[`network with θ${sub('A')}`,palette.q],[`network with θ${sub('B')}`,palette.noise]]),'With theta A the network follows the data; with theta B it does not.');
}
function evidenceMcChart(){
 const e=b.evidence,xs=e.sizes.map(Math.log10),sup=['⁰','¹','²','³','⁴','⁵','⁶','⁷'];
 const P=panel({W:600,H:330,L:66,R:22,xMin:1,xMax:7,yMin:-60,yMax:5,ticksX:[1,3,5,7],fmtX:v=>`10${sup[v]}`,ticksY:[-60,-40,-20,0],xLabel:'prior draws S'});
 return fig(600,330,P.axes+`<line x1="${P.x(1)}" y1="${P.y(e.estimate).toFixed(1)}" x2="${P.x(7)}" y2="${P.y(e.estimate).toFixed(1)}" stroke="${palette.data}" stroke-width="2" stroke-dasharray="8 6"/>`+
  e.runs.map(r=>P.line(xs,r.estimates,palette.q,thin(2.5,.75))+r.estimates.map((v,k)=>v>=-60?P.dot(xs[k],v,palette.q,4.5):'').join('')).join('')+
  label(66,30,'estimate of log p(𝒟): four seeds',{fill:palette.q,size:21})+label(P.x(7),P.y(e.estimate)+28,`≈ ${num(e.estimate)}`,{fill:palette.data,size:20,anchor:'end'}),
  'Monte Carlo estimates of the log evidence against the number of prior draws, for four seeds.');
}
function predictiveChart(){
 const h=b.hmc,P=fxPanel();
 return fig(600,330,P.axes+P.band(b.grid,h.mean.map((v,k)=>v-2*h.sd[k]),h.mean.map((v,k)=>v+2*h.sd[k]),TINT.p)+h.draws.map(fy=>P.line(b.grid,fy,palette.p,thin(1.5,.45))).join('')+dataDots(P)+
  legend(52,30,[['mean ± 2 sd, and 20 posterior draws',palette.p]]),'The posterior predictive: a band that widens away from the data, and twenty functions drawn from the posterior.');
}
/** One Gaussian per parameter. With fitted values, drawn against the N(0, 1) prior in grey. */
function qFactors(fit=null){
 const W=1150,H=fit?190:150,PW=150,G=(W-7*PW)/6,us=lin(-3,3,161);let s='';
 const shape=[[.6,.5],[-.8,.35],[.2,.6],[-.3,.3],[.9,.45],[-.5,.4],[.1,.25]];
 b.names.forEach((nm,j)=>{
  const ox=j*(PW+G),mu=fit?fit.mu[j]:shape[j][0],sd=fit?fit.sigma[j]:shape[j][1];
  const P=panel({ox,W:PW,H,L:6,R:6,T:40,B:fit?38:12,xMin:-3,xMax:3,yMax:fit?2.2:1.7,yAxis:false,ticksX:fit?[-2,0,2]:[]});
  const prior=us.map(u=>normalPdf(u,{mu:0,sigma:1}));
  s+=P.axes+(fit?P.area(us,prior,'#E5E7EB')+P.line(us,prior,'#9CA3AF',thin(2)):'')+P.line(us,us.map(u=>normalPdf(u,{mu,sigma:sd})),palette.q)+
   label(ox+PW/2,24,TH[nm],{fill:palette.noise,size:23,anchor:'middle',weight:600});
 });
 return fig(W,H,s,fit?'The fitted Gaussian for each of the seven parameters, against the N(0, 1) prior.':'One Gaussian for each of the seven parameters.');
}

// ---------------------------------------------------------------- Part 2 figures: grades, codes, classifiers
const gr=st.grades,ce=st.cross_entropy,GC=[palette.d,palette.loss,MUTED],pct=v=>`${Math.round(v*100)}%`;
const RITVIK='KL intuition after <a href="https://www.youtube.com/watch?v=q0AkK8aYbLY">ritvikmath, “The KL Divergence: Data Science Basics”</a> (YouTube)';
const GERON='Coding story after <a href="https://www.youtube.com/watch?v=ErfnhcEV1O8">Aurélien Géron, “A Short Introduction to Entropy, Cross-Entropy and KL-Divergence”</a> (YouTube)';
function pretendChart(){
 const a=d.plots.normal,P=panel({W:600,H:330,L:16,R:14,xMin:-11,xMax:13,yMax:.23,ticksX:[-10,-5,0,5,10],yAxis:false,xLabel:'θ'});
 return fig(600,330,P.axes+P.area(a.x,a.p,TINT.p)+P.line(a.x,a.p,palette.p)+`<g data-at="2">${P.line(a.x,a.q1,palette.q)}${legend(250,30,[['q(θ), our choice',palette.q]])}</g>`+
  legend(16,30,[['p(θ | 𝒟), pretend',palette.p]]),'A pretend posterior N(1, 4²) and an approximation q = N(0, 2²).');
}
function gradeBars(q,p){
 const W=640,H=320,base=H-46,top=56,sc=v=>v/.6*(base-top);let s='';
 gr.names.forEach((name,i)=>{const cx=110+i*200;
  for(const [v,color,dx] of [[q[i],palette.q,-31],[p[i],palette.p,31]]){const h=Math.max(1.5,sc(v));
   s+=`<rect x="${cx+dx-28}" y="${(base-h).toFixed(1)}" width="56" height="${h.toFixed(1)}" rx="3" fill="${color}"/>`+label(cx+dx,base-h-10,pct(v),{fill:color,size:21,anchor:'middle'});}
  s+=label(cx,base+32,name,{fill:ink,size:24,anchor:'middle',weight:600});});
 s+=`<line x1="20" y1="${base}" x2="${W-20}" y2="${base}" stroke="${AX}"/><rect x="20" y="14" width="18" height="18" rx="3" fill="${palette.q}"/>`+label(46,30,'last year, q',{fill:palette.q,size:21})+
  `<rect x="230" y="14" width="18" height="18" rx="3" fill="${palette.p}"/>`+label(256,30,'this year, p',{fill:palette.p,size:21});
 return fig(W,H,s,'Grade shares in two years of the course.');
}
function ratioScale(logs){
 const W=560,H=160,y0=96,lo=logs?-1.7:0,hi=logs?1.7:4.4,X=v=>30+(v-lo)/(hi-lo)*500,vals=logs?gr.log_ratio:gr.ratio,one=X(logs?0:1);
 let s=`<line x1="${X(lo)}" y1="${y0}" x2="${X(hi)}" y2="${y0}" stroke="${AX}"/>`+label(14,24,logs?'log ratio, log(p / q)':'ratio, p / q',{fill:ink,size:22,weight:600});
 for(const t of logs?[-1,0,1]:[0,1,2,3,4])s+=`<line x1="${X(t)}" y1="${y0-5}" x2="${X(t)}" y2="${y0+5}" stroke="${AX}"/>`+label(X(t),y0+28,num(t,0),{size:19,anchor:'middle'});
 [0,1].forEach(i=>{s+=`<line x1="${one}" y1="${y0+46}" x2="${X(vals[i]).toFixed(1)}" y2="${y0+46}" stroke="${GC[i]}" stroke-width="5" stroke-linecap="round"/>`;});
 vals.forEach((v,i)=>{s+=`<circle cx="${X(v).toFixed(1)}" cy="${y0}" r="8" fill="${GC[i]}"/>`+label(X(v),y0-16,gr.names[i],{fill:GC[i],size:21,anchor:'middle',weight:600});});
 return fig(W,H,s,logs?'Log ratios are symmetric: plus and minus 1.39, and 0.':'Ratios are lopsided: 4, a quarter, and 1.');
}
function gradeValueBars(values,title,total=null){
 const W=560,H=360,y=v=>180-v*88;
 let s=`<line x1="20" y1="${y(0)}" x2="${W-20}" y2="${y(0)}" stroke="${AX}"/>`+label(20,26,title,{fill:ink,size:21});
 const bars=values.map((v,i)=>[gr.names[i],v,GC[i]]);if(total!==null)bars.push(['KL',total,ink]);
 bars.forEach(([name,v,color],i)=>{const cx=85+i*130,t=Math.min(y(0),y(v)),h=Math.max(2,Math.abs(y(v)-y(0)));
  s+=`<rect x="${cx-34}" y="${t.toFixed(1)}" width="68" height="${h.toFixed(1)}" rx="3" fill="${color}"/>`+label(cx,v<0?y(v)+28:y(v)-10,num(v,2),{fill:color,size:21,anchor:'middle'})+
   label(cx,352,name,{fill:ink,size:22,anchor:'middle',weight:600});});
 return fig(W,H,s,title);
}
function swapContrib(){
 const W=1150,H=330;let s='';
 [[gr.swap.terms_pq,gr.swap.kl_pq,'KL(this ‖ last): weights from this year'],[gr.swap.terms_qp,gr.swap.kl_qp,'KL(last ‖ this): weights from last year']].forEach(([terms,total,title],k)=>{
  const ox=k*590,y=v=>190-v*150;
  s+=`<line x1="${ox+20}" y1="${y(0)}" x2="${ox+540}" y2="${y(0)}" stroke="${AX}"/>`+label(ox+20,28,title,{fill:ink,size:21,weight:600});
  [...terms.map((v,i)=>[gr.names[i],v,GC[i]]),['total',total,ink]].forEach(([name,v,color],i)=>{const cx=ox+80+i*125,t=Math.min(y(0),y(v)),h=Math.max(2,Math.abs(y(v)-y(0)));
   s+=`<rect x="${cx-32}" y="${t.toFixed(1)}" width="64" height="${h.toFixed(1)}" rx="3" fill="${color}"/>`+label(cx,v<0?y(v)+26:y(v)-10,num(v,2),{fill:color,size:20,anchor:'middle'})+
    label(cx,318,name,{fill:ink,size:21,anchor:'middle',weight:600});});});
 return fig(W,H,s,'Each grade’s contribution to the KL, in both directions.');
}
const box=(x,y,w,h,color=ink)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${palette.paper}" stroke="${color}" stroke-width="2.5"/>`;
const arrow=(x1,y,x2,color=ink)=>`<line x1="${x1}" y1="${y}" x2="${x2-12}" y2="${y}" stroke="${color}" stroke-width="2.5"/><polygon points="${x2},${y} ${x2-14},${y-7} ${x2-14},${y+7}" fill="${color}"/>`;
const bitBox=(x,y,bit,color=palette.q)=>`<rect x="${x}" y="${y}" width="40" height="46" rx="5" fill="${TINT.q}" stroke="${color}" stroke-width="2"/>`+label(x+20,y+32,bit,{fill:color,size:24,anchor:'middle',weight:600});
function channelDiagram(){
 let s=box(10,40,270,170)+label(145,72,'weather station',{fill:ink,size:22,anchor:'middle',weight:600})+weather('sunny',75,135)+weather('cloudy',145,135)+weather('rainy',215,130)+label(145,196,'today: cloudy',{size:20,anchor:'middle'});
 s+=box(870,40,270,170)+label(1005,72,'receiver',{fill:ink,size:22,anchor:'middle',weight:600})+weather('cloudy',1005,135)+label(1005,196,'reads: cloudy',{size:20,anchor:'middle'});
 s+=arrow(290,125,860)+bitBox(530,102,'1')+bitBox(576,102,'0')+label(575,182,'bits on the wire: each one costs',{size:20,anchor:'middle'});
 return fig(1150,240,s,'A weather station sends the day’s weather to a receiver as bits.');
}
function prefixDecode(good){
 const W=1150,H=165;let s;
 if(good){
  s=label(0,26,'prefix code: sunny 0 · cloudy 10 · rainy 11',{fill:ink,size:21,weight:600});let x=10;
  [['0','sunny'],['10','cloudy'],['11','rainy'],['0','sunny'],['10','cloudy']].forEach(([w,name])=>{const start=x;[...w].forEach(bit=>{s+=bitBox(x,44,bit);x+=46;});
   s+=`<path d="M${start+2} 100 V108 H${x-8} V100" fill="none" stroke="${MUTED}" stroke-width="2"/>`+label((start+x-6)/2,134,name,{fill:ink,size:19,anchor:'middle'});x+=24;});
  s+=label(x+20,76,'→ the receiver reads left to right; each day is unambiguous',{size:20});
 }else{
  s=label(0,26,'not a prefix code: sunny 0 · cloudy 01 · rainy 1',{fill:palette.loss,size:21,weight:600})+bitBox(10,44,'0',palette.loss)+bitBox(56,44,'1',palette.loss);
  s+=label(130,76,'= sunny, then rainy?',{fill:palette.loss,size:21})+label(130,120,'or cloudy?',{fill:palette.loss,size:21})+label(400,98,'0 is the start of 01, so the receiver cannot tell.',{size:20});
 }
 return fig(W,H,s,good?'A bit stream decoded with a prefix code.':'An ambiguous bit stream from a code that is not a prefix code.');
}
function codeTree(){
 const W=520,H=320,node=(x,y)=>`<circle cx="${x}" cy="${y}" r="10" fill="${ink}"/>`,leaf=(x,y,name,pv)=>box(x-62,y-30,124,74)+weather(name,x,y-4,.55)+label(x,y+36,`${name} ${pv}`,{fill:ink,size:18,anchor:'middle'});
 const edge=(x1,y1,x2,y2,bit)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${MUTED}" stroke-width="2.5"/>`+label((x1+x2)/2+(x2<x1?-14:14),(y1+y2)/2,bit,{fill:palette.q,size:22,anchor:'middle',weight:600});
 let s=edge(260,40,130,130,'0')+edge(260,40,380,120,'1')+edge(380,120,300,230,'0')+edge(380,120,460,230,'1')+node(260,40)+node(380,120);
 s+=leaf(130,160,'sunny','½')+leaf(300,262,'cloudy','¼')+leaf(460,262,'rainy','¼');
 return fig(W,H,s,'A Huffman code tree: sunny is 0, cloudy 10, rainy 11.');
}
function catGlyph(cx,cy,k=1){
 return `<g stroke="${ink}" stroke-width="${2.5*k}" fill="none" stroke-linejoin="round"><path d="M${cx-30*k},${cy-14*k} L${cx-24*k},${cy-44*k} L${cx-8*k},${cy-26*k} Q${cx},${cy-29*k} ${cx+8*k},${cy-26*k} L${cx+24*k},${cy-44*k} L${cx+30*k},${cy-14*k} Q${cx+36*k},${cy+18*k} ${cx},${cy+26*k} Q${cx-36*k},${cy+18*k} ${cx-30*k},${cy-14*k}Z"/><path d="M${cx-4*k},${cy+6*k} L${cx},${cy+10*k} L${cx+4*k},${cy+6*k}"/></g><circle cx="${cx-11*k}" cy="${cy-4*k}" r="${3*k}" fill="${ink}"/><circle cx="${cx+11*k}" cy="${cy-4*k}" r="${3*k}" fill="${ink}"/>`;
}
function probBars(ox,oy,vals,names,color,title,bw=200){
 let s=label(ox,oy,title,{fill:color,size:21,weight:600});
 vals.forEach((v,i)=>{const y=oy+24+i*48;s+=label(ox,y+23,names[i],{fill:ink,size:21})+`<rect x="${ox+56}" y="${y}" width="${Math.max(2,v*bw).toFixed(1)}" height="32" rx="4" fill="${color}"/>`+label(ox+56+Math.max(2,v*bw)+10,y+23,num(v,1),{fill:color,size:21});});
 return s;
}
function classifierDiagram(){
 let s=`<rect x="10" y="50" width="170" height="160" rx="10" fill="#FFF7ED" stroke="${ink}" stroke-width="2"/>`+catGlyph(95,138,1.4)+label(95,240,'an image x',{size:20,anchor:'middle'});
 s+=arrow(188,130,256)+box(262,80,180,100,palette.q)+label(352,124,'classifier',{fill:palette.q,size:22,anchor:'middle',weight:600})+label(352,154,'weights θ',{size:19,anchor:'middle'})+arrow(450,130,510);
 s+=probBars(520,50,ce.pred,['cat','dog'],palette.q,'prediction q_θ(y | x)',180);
 s+=`<g data-at="2">${probBars(860,50,ce.hard,['cat','dog'],palette.p,'label p(y)',180)}${label(860,200,'all its probability on cat',{size:19})}</g>`;
 return fig(1150,250,s,'An image goes into a classifier, which outputs a distribution over labels; the true label is a distribution too.');
}
function labelVsPred(target){
 return fig(560,260,probBars(10,30,target,['cat','dog'],palette.p,'label p(y)',150)+probBars(290,30,ce.pred,['cat','dog'],palette.q,'prediction q_θ(y | x)',150),'The label distribution and the classifier’s prediction.');
}
function bernBars(){
 const W=600,H=300;let s='';
 [[.8,palette.p,'p = Bernoulli(0.8)'],[.5,palette.q,'q = Bernoulli(0.5)']].forEach(([a,color,title],k)=>{const ox=k*300,base=240,sc=v=>v*170;
  s+=label(ox+20,30,title,{fill:color,size:21,weight:600});
  [[0,1-a],[1,a]].forEach(([x,v],i)=>{const cx=ox+80+i*110,h=sc(v);s+=`<rect x="${cx-30}" y="${(base-h).toFixed(1)}" width="60" height="${h.toFixed(1)}" rx="3" fill="${color}"/>`+label(cx,base-h-10,num(v,1),{fill:color,size:21,anchor:'middle'})+label(cx,base+30,`x = ${x}`,{fill:ink,size:20,anchor:'middle'});});
  s+=`<line x1="${ox+30}" y1="${base}" x2="${ox+270}" y2="${base}" stroke="${AX}"/>`;});
 return fig(W,H,s,'Two Bernoulli distributions: probability 0.8 and 0.5 of x = 1.');
}
function diagEllipses(){
 const W=520,H=330,S=58,cx=260,cy=170;let s=`<line x1="${cx-230}" y1="${cy}" x2="${cx+230}" y2="${cy}" stroke="${AX}"/><line x1="${cx}" y1="${cy-150}" x2="${cx}" y2="${cy+150}" stroke="${AX}"/>`+label(cx+228,cy+26,'θ₁',{size:20,anchor:'end'})+label(cx+10,cy-132,'θ₂',{size:20});
 for(const k of [1,2])s+=`<circle cx="${cx}" cy="${cy}" r="${k*S}" fill="none" stroke="${palette.p}" stroke-width="3"/>`;
 for(const k of [1,2])s+=`<ellipse cx="${cx+1.1*S}" cy="${cy-.6*S}" rx="${(k*.9*S).toFixed(1)}" ry="${(k*.45*S).toFixed(1)}" fill="none" stroke="${palette.q}" stroke-width="3"/>`;
 return fig(W,H,s+legend(10,26,[['p = N(0, I)',palette.p],['q: axis-aligned',palette.q]]),'A standard Gaussian prior and an axis-aligned Gaussian q in two dimensions.');
}

// ---------------------------------------------------------------- Part 2 figures: surprise, entropy, codes, classifiers, data
const inf=st.info,wb=st.bits,cl=st.classify,ml=st.mle,half=v=>v===.5?'½':v===.25?'¼':num(v,2);
const log2=x=>Math.log(x)/Math.LN2,bitsOf=pr=>-log2(pr);
const HUANG='Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)';
const SUN='#E8A317';
/** The forecast: sun 0.9 and rain 0.1 as icons over probability bars, optionally with each surprise. */
function forecastFig(withSurprise=false){
 const W=520,H=320,base=250;let s='';
 [['sunny',inf.forecast[0]],['rainy',inf.forecast[1]]].forEach(([kind,pr],k)=>{const cx=130+k*240,h=pr*170;
  s+=weather(kind,cx,40,1.25)+`<rect x="${cx-42}" y="${(base-h).toFixed(1)}" width="84" height="${h.toFixed(1)}" rx="4" fill="${palette.p}"/>`+
   label(cx,base-h-10,`p = ${pr}`,{fill:palette.p,size:22,anchor:'middle',weight:600})+label(cx,base+30,kind,{fill:ink,size:21,anchor:'middle'});
  if(withSurprise)s+=label(cx,base+62,`surprise ${num(inf.surprise_bits[k],2)} bits`,{fill:palette.loss,size:20,anchor:'middle'});});
 s+=`<line x1="40" y1="${base}" x2="${W-40}" y2="${base}" stroke="${AX}"/>`;
 return fig(W,withSurprise?H:290,s,withSurprise?'Sun with probability 0.9 and surprise 0.15 bits; rain with probability 0.1 and surprise 3.32 bits.':'A forecast: sun with probability 0.9, rain with probability 0.1.');
}
/** Five days in a row: one probability and one surprise per day, their product and their sum. */
function fiveDays(withRain){
 const W=1150,H=150,kinds=['sunny','sunny',withRain?'rainy':'sunny','sunny','sunny'];let s='';
 kinds.forEach((kind,j)=>{const cx=60+j*120,pr=kind==='sunny'?.9:.1;
  s+=weather(kind,cx,34,.95)+label(cx,92,`p = ${pr}`,{fill:palette.p,size:19,anchor:'middle'})+label(cx,120,`${num(bitsOf(pr),kind==='sunny'?3:2)} bits`,{fill:palette.loss,size:19,anchor:'middle'});
  if(j<4)s+=label(cx+60,92,'×',{fill:palette.p,size:20,anchor:'middle'})+label(cx+60,120,'+',{fill:palette.loss,size:20,anchor:'middle'});});
 const pr=withRain?inf.one_rain.p:inf.five_sun.p,bt=withRain?inf.one_rain.bits:inf.five_sun.bits;
 s+=label(660,92,`= ${num(pr,withRain?4:3)}`,{fill:palette.p,size:22,weight:600})+label(660,120,`= ${num(bt,2)} bits`,{fill:palette.loss,size:22,weight:600})+
  label(820,92,'probabilities multiply',{fill:palette.p,size:20})+label(820,120,`surprises add: log₂ 1/${num(pr,withRain?4:3)} = ${num(bt,2)}`,{fill:palette.loss,size:20});
 return fig(W,H,s,withRain?'Four sunny days and one rainy day: the probabilities multiply to 0.066 and the surprises add to 3.93 bits.':'Five sunny days: the probabilities multiply to 0.59 and the surprises add to 0.76 bits.');
}
const qNode=(x,y,w,t)=>`<rect x="${x-w/2}" y="${y-26}" width="${w}" height="52" rx="26" fill="${palette.paper}" stroke="${ink}" stroke-width="2.5"/>`+label(x,y+8,t,{fill:ink,size:21,anchor:'middle',weight:600});
const leafNode=(x,y,kind,t)=>`<rect x="${x-62}" y="${y-36}" width="124" height="86" rx="10" fill="${palette.paper}" stroke="${MUTED}" stroke-width="2"/>`+weather(kind,x,y-6,.62)+label(x,y+38,t,{fill:ink,size:18,anchor:'middle'});
const treeEdge=(x1,y1,x2,y2,t,bit,at)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${MUTED}" stroke-width="2.5"/>`+label((x1+x2)/2+(x2<x1?-22:22),(y1+y2)/2,t,{fill:MUTED,size:18,anchor:'middle'})+
 (bit?`<g data-at="${at}">${label((x1+x2)/2+(x2<x1?-22:22),(y1+y2)/2+24,bit,{fill:palette.q,size:22,anchor:'middle',weight:600})}</g>`:'');
/** The question game: ask about the likeliest weather first; each answer is one bit. */
function questionTree(){
 let s=treeEdge(200,62,110,190,'yes','0',3)+treeEdge(200,62,390,140,'no','1',3)+treeEdge(390,140,300,250,'yes','0',3)+treeEdge(390,140,480,250,'no','1',3);
 s+=qNode(200,48,210,'Is it sunny?')+qNode(390,140,210,'Is it cloudy?')+leafNode(110,226,'sunny','1 question · ½')+leafNode(300,286,'cloudy','2 questions · ¼')+leafNode(480,286,'rainy','2 questions · ¼');
 return fig(560,340,s,'A yes/no question tree: is it sunny? if not, is it cloudy? Sunny takes one question, cloudy and rainy two.');
}
/** Huffman’s algorithm on sunny ½, cloudy ¼, rainy ¼, one merge per build. */
function huffmanSteps(){
 const L=[[90,'sunny','½'],[300,'cloudy','¼'],[470,'rainy','¼']],y=262;
 let s=L.map(([x,kind,pr])=>leafNode(x,y,kind,`${kind} ${pr}`)).join('');
 const node=(x,yy,t)=>`<circle cx="${x}" cy="${yy}" r="26" fill="${TINT.q}" stroke="${palette.q}" stroke-width="2.5"/>`+label(x,yy+8,t,{fill:palette.q,size:20,anchor:'middle',weight:600});
 const e=(x1,y1,x2,y2,bit)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${palette.q}" stroke-width="2.5"/><g data-at="4">${label((x1+x2)/2+(x2<x1?-16:16),(y1+y2)/2,bit,{fill:palette.q,size:22,anchor:'middle',weight:600})}</g>`;
 s+=`<g data-at="2">${e(385,150,300,226,'0')+e(385,150,470,226,'1')+node(385,150,'½')}</g>`;
 s+=`<g data-at="3">${e(240,50,90,226,'0')+e(240,50,385,124,'1')+node(240,50,'1')}</g>`;
 s+=`<g data-at="4">${[[90,'0'],[300,'10'],[470,'11']].map(([x,c])=>label(x,y+80,c,{fill:palette.q,size:24,anchor:'middle',weight:700})).join('')}</g>`;
 return fig(560,360,s,'Huffman’s algorithm: merge cloudy and rainy into a node of probability one half, merge that with sunny into the root, then read the codes 0, 10 and 11 off the tree.');
}
/** Weather distributions side by side, one bar per distribution for each kind of weather. */
function weatherBars(dists,titles,colors,{small:sm=false}={}){
 const W=sm?520:600,H=sm?260:300,base=H-60,top=60,n=dists.length,bw=n>1?48:64,sc=v=>v*(base-top)/1;let s='';
 wb.names.forEach((nm,i)=>{const cx=(sm?90:100)+i*(sm?170:200);
  s+=weather(nm,cx,base+34,.55);
  dists.forEach((dd,k)=>{const dx=n>1?(k?1:-1)*(bw/2+3):0,h=sc(dd[i]);
   s+=`<rect x="${(cx+dx-bw/2).toFixed(1)}" y="${(base-h).toFixed(1)}" width="${bw}" height="${Math.max(1.5,h).toFixed(1)}" rx="3" fill="${colors[k]}"/>`+label(cx+dx,base-h-8,half(dd[i]),{fill:colors[k],size:sm?17:20,anchor:'middle'});});});
 s+=`<line x1="20" y1="${base}" x2="${W-20}" y2="${base}" stroke="${AX}"/>`+legend(20,26,titles.map((t,k)=>[t,colors[k]]),sm?18:20);
 return fig(W,H,s,`Weather probabilities: ${titles.join(' and ')}.`);
}
/** A classifier: an image in, a distribution over four labels out; build 2 adds the one-hot label. */
function classifier4(){
 const bars=(ox,oy,vals,color,title,bw=170)=>{let g=label(ox,oy,title,{fill:color,size:20,weight:600});
  cl.classes.forEach((c,i)=>{const y=oy+18+i*44,w=Math.max(2,vals[i]*bw);g+=label(ox,y+24,c,{fill:ink,size:20})+`<rect x="${ox+60}" y="${y}" width="${w.toFixed(1)}" height="30" rx="4" fill="${color}"/>`+label(ox+66+w,y+23,num(vals[i],2),{fill:color,size:19});});
  return g;};
 let s=`<rect x="10" y="50" width="170" height="160" rx="10" fill="#FFF7ED" stroke="${ink}" stroke-width="2"/>`+catGlyph(95,138,1.4)+label(95,240,'an image of a cat',{size:19,anchor:'middle'});
 s+=arrow(188,130,248)+box(254,80,170,100,palette.q)+label(339,124,'classifier',{fill:palette.q,size:22,anchor:'middle',weight:600})+label(339,154,'+ softmax',{size:19,anchor:'middle'})+arrow(432,130,490);
 s+=bars(500,40,cl.pred,palette.q,'prediction q');
 s+=`<g data-at="2">${bars(850,40,cl.classes.map((_,i)=>i===cl.label?1:0),palette.p,'label p (one-hot)',150)}</g>`;
 return fig(1150,250,s,'A cat image goes into a classifier, which outputs probabilities 0.05, 0.05, 0.60 and 0.30 for ape, bear, cat and dog; the label puts all its probability on cat.');
}
/** A teacher’s soft label against the student’s prediction. */
function labelPair(teacher){
 const W=560,H=250;let s='';
 [[teacher,palette.p,'teacher’s soft label p'],[cl.pred,palette.q,'student’s prediction q']].forEach(([vals,color,title],k)=>{const ox=k*285;
  s+=label(ox,24,title,{fill:color,size:19,weight:600});
  cl.classes.forEach((c,i)=>{const y=42+i*48,w=Math.max(2,vals[i]*160);s+=label(ox,y+24,c,{fill:ink,size:19})+`<rect x="${ox+52}" y="${y}" width="${w.toFixed(1)}" height="30" rx="4" fill="${color}"/>`+label(ox+58+w,y+23,num(vals[i],2),{fill:color,size:18});});});
 return fig(W,H,s,'A teacher’s soft label and a student’s prediction over ape, bear, cat and dog.');
}
/** The classification loss, minus the log of the probability given to the true class. */
function lossCurve(){
 const xs=lin(.008,1,300),P=panel({W:600,H:330,L:58,R:16,T:44,B:56,xMin:0,xMax:1,yMin:0,yMax:5,ticksX:[0,.25,.5,.75,1],ticksY:[0,1,2,3,4,5],xLabel:'probability given to the true class'});
 let s=P.axes+P.line(xs,xs.map(x=>-Math.log(x)),palette.loss)+label(58,26,'loss = −ln q(true class), nats',{fill:palette.loss,size:20});
 for(const [k,v] of Object.entries(cl.curve)){const x=Number(k);s+=P.dot(x,v,palette.loss,6)+(x<1?label(P.x(x)+(x>=.9?-4:10),P.y(v)-(x>=.9?14:10),num(v,2),{fill:palette.loss,size:17,anchor:x>=.9?'end':'start'}):'');}
 return fig(600,330,s,'The cross-entropy loss against the probability given to the true class: 0 at 1, 0.51 at 0.6, 4.6 at 0.01.');
}
/** Twenty recorded days, in order, with their numbers. */
function dataGrid(){
 const W=1150,H=210,kinds=['sunny','cloudy','rainy'];let s='';
 ml.data.forEach((k,i)=>{const x=50+(i%10)*108,y=i<10?40:140;s+=weather(kinds[k],x,y,.8)+label(x,y+50,`day ${i+1}`,{size:15,anchor:'middle'});});
 return fig(W,H,s,`Twenty days of weather: ${ml.counts[0]} sunny, ${ml.counts[1]} cloudy and ${ml.counts[2]} rainy.`);
}

// ---------------------------------------------------------------- Section 2.7 figures: distillation and RLHF
const ds=st.distill,rl=st.rlhf;
const PI='<tspan font-family="Times New Roman,Georgia,serif" font-style="italic">π</tspan>',PIH='<span class="pi">π</span>';
const STUDENT0=[.25,.20,.40,.15];
/** Probability bars over ape, bear, cat and dog, drawn at (ox, oy). */
function classBars(ox,oy,vals,color,title,bw=150,rowH=34){
 let g=label(ox,oy,title,{fill:color,size:19,weight:600});
 cl.classes.forEach((c,i)=>{const y=oy+14+i*rowH,w=Math.max(2,vals[i]*bw);g+=label(ox,y+21,c,{fill:ink,size:18})+`<rect x="${ox+48}" y="${y}" width="${w.toFixed(1)}" height="26" rx="4" fill="${color}"/>`+label(ox+54+w,y+20,num(vals[i],2),{fill:color,size:17});});
 return g;
}
/** A network drawn as columns of nodes, scaled to look big or small. */
function netGlyph(x,y,w,h,cols,rows,color){
 let g='';const cx=c=>x+(c+.5)*w/cols,cy=(r,n)=>y+h/2+(r-(n-1)/2)*Math.min(h/(n+.5),22);
 for(let c=0;c<cols-1;c++){const n1=c===0?2:rows,n2=c===cols-2?3:rows;for(let a=0;a<n1;a++)for(let b2=0;b2<n2;b2++)g+=`<line x1="${cx(c).toFixed(1)}" y1="${cy(a,n1).toFixed(1)}" x2="${cx(c+1).toFixed(1)}" y2="${cy(b2,n2).toFixed(1)}" stroke="${color}" stroke-opacity=".22" stroke-width="1"/>`;}
 for(let c=0;c<cols;c++){const n=c===0?2:c===cols-1?3:rows;for(let r=0;r<n;r++)g+=`<circle cx="${cx(c).toFixed(1)}" cy="${cy(r,n).toFixed(1)}" r="5" fill="${palette.paper}" stroke="${color}" stroke-width="2"/>`;}
 return g;
}
/** Knowledge distillation: the same image through a big frozen teacher and a small student. */
function distillDiagram(stage=3){
 const W=1150,H=380,T=palette.p,Sq=palette.q,L=palette.loss;let s=graphDefs('dd');
 s+=`<rect x="10" y="104" width="130" height="120" rx="10" fill="#FFF7ED" stroke="${ink}" stroke-width="2"/>`+catGlyph(75,166,1.1)+label(75,248,'one image x',{size:18,anchor:'middle'});
 // the teacher: big and deep
 s+=`<rect x="200" y="10" width="300" height="128" rx="12" fill="${TINT.p}" stroke="${T}" stroke-width="2.5"/>`+netGlyph(214,22,272,104,6,5,T)+
  label(350,160,'teacher: big, accurate, slow',{fill:T,size:19,anchor:'middle',weight:600});
 s+=fwdEdge('dd','M140,136 L198,90')+fwdEdge('dd','M500,80 L566,80');
 s+=classBars(570,24,cl.teacher,T,'teacher’s p(y | x)');
 // the student: small and shallow
 const st2=`<rect x="250" y="192" width="190" height="90" rx="12" fill="${TINT.q}" stroke="${Sq}" stroke-width="2.5"/>`+netGlyph(262,200,166,74,3,3,Sq)+
  label(345,306,'student: small, fast',{fill:Sq,size:19,anchor:'middle',weight:600})+fwdEdge('dd','M140,196 L248,236')+fwdEdge('dd','M440,236 L566,262')+
  classBars(570,198,STUDENT0,Sq,'student’s q(y | x)');
 s+=stage>=2?st2:`<g opacity="0">${st2}</g>`;
 // the KL link and the gradient into the student
 if(stage>=3)s+=`<path d="M805,102 C870,102 870,276 805,276" fill="none" stroke="${L}" stroke-width="3"/>`+label(872,182,'KL(p ‖ q)',{fill:L,size:22,weight:600})+label(872,210,'make these the same',{fill:L,size:18});
 if(stage>=4)s+=backEdge('dd','M900,236 C930,372 520,360 444,272')+label(900,364,'gradient: only the student learns',{fill:L,size:17,anchor:'middle'})+
  `<rect x="452" y="16" width="42" height="24" rx="6" fill="${palette.paper}" stroke="${T}"/>`+label(473,33,'frozen',{fill:T,size:12,anchor:'middle',weight:600});
 return fig(W,H,s,'Knowledge distillation: the same image goes through a big teacher network and a small student network; the student is trained so its output distribution matches the teacher’s, minimising KL from the teacher to the student, and only the student receives gradients.');
}
/** One-hot label against the teacher’s soft labels, plain and softened by temperature. */
function softTemps(){
 const soft=t=>{const w=cl.teacher.map(v=>Math.pow(v,1/t)),z=w.reduce((a,b2)=>a+b2,0);return w.map(v=>v/z);};
 let s=classBars(0,26,cl.classes.map((_,i)=>i===cl.label?1:0),MUTED,'one-hot label',120);
 s+=classBars(380,26,soft(1),palette.p,'teacher, τ = 1',120)+classBars(760,26,soft(4),palette.p,'teacher softened, τ = 4',120);
 return fig(1150,220,s,'A one-hot label for cat, the teacher’s distribution, and the same distribution softened with temperature 4.');
}
/** RLHF: the policy writes a reply, a reward model scores it, and a frozen reference keeps it close. */
function rlhfDiagram(stage=3){
 const W=1150,H=340,Q=palette.q,T=palette.p,L=palette.loss;let s=graphDefs('rd');
 s+=box(10,60,170,90,ink)+label(95,96,'prompt',{fill:ink,size:19,anchor:'middle',weight:600})+label(95,124,'“Is the stove hot?”',{size:16,anchor:'middle'});
 s+=`<rect x="240" y="40" width="220" height="130" rx="12" fill="${TINT.q}" stroke="${Q}" stroke-width="2.5"/>`+netGlyph(252,52,196,86,5,4,Q)+label(350,160,`policy ${PI}, being trained`,{fill:Q,size:17,anchor:'middle',weight:600});
 s+=fwdEdge('rd','M180,105 L238,105')+fwdEdge('rd','M460,105 L528,105')+box(530,62,220,86,ink)+label(640,96,'reply y',{fill:ink,size:19,anchor:'middle',weight:600})+label(640,124,'“Yes, careful!”',{size:16,anchor:'middle'});
 s+=fwdEdge('rd','M750,105 L818,105')+box(820,62,170,86,palette.d)+label(905,96,'reward model',{fill:palette.d,size:18,anchor:'middle',weight:600})+label(905,124,'score r(y)',{size:16,anchor:'middle'});
 s+=fwdEdge('rd','M990,105 L1040,105')+label(1046,112,'+r',{fill:palette.d,size:24,weight:700});
 const ref=`<rect x="240" y="226" width="220" height="100" rx="12" fill="${TINT.p}" stroke="${T}" stroke-width="2.5"/>`+netGlyph(252,232,196,70,5,4,T)+label(350,320,`reference ${PI}<tspan baseline-shift="sub" font-size="70%">ref</tspan>, frozen`,{fill:T,size:16,anchor:'middle',weight:600})+
  `<path d="M350,172 L350,224" stroke="${L}" stroke-width="3" stroke-dasharray="6 5"/>`+label(364,204,`KL(${PI} ‖ ${PI}<tspan baseline-shift="sub" font-size="70%">ref</tspan>): stay close`,{fill:L,size:18,weight:600});
 s+=stage>=2?ref:`<g opacity="0">${ref}</g>`;
 if(stage>=3)s+=label(600,232,'maximise',{fill:ink,size:20})+label(600,266,`E[r(y)] − β · KL(${PI} ‖ ${PI}<tspan baseline-shift="sub" font-size="70%">ref</tspan>)`,{fill:ink,size:24,weight:600})+label(600,298,'reward, minus a leash to where it started',{size:17});
 return fig(W,H,s,'RLHF: a prompt goes to the policy, which writes a reply; a reward model scores the reply; a frozen reference copy of the model keeps the policy close through a KL penalty. The objective is expected reward minus beta times KL.');
}

// ---------------------------------------------------------------- Part 3 figures: covariance families, mixtures, computational graphs
const c3=st.cov3,COV_NAMES={iso:'isotropic: s²I',diag:'diagonal: diag(s₁², s₂²)',full:'full: LLᵀ'};
const COV_COUNTS={iso:'1 number, whatever the dimension',diag:'2 numbers; d in d dimensions',full:'3 numbers; d(d + 1)/2 in general'};
/** Example contours for the three covariance families, one panel per build. */
function covFamilies(){
 const W=1150,H=345,PW=370,G=20,k=44,ex={iso:[1,0,1],diag:[1.7,0,.36],full:[1.5,.95,.9]};let s='';
 ['iso','diag','full'].forEach((kind,j)=>{const ox=j*(PW+G),cx=ox+PW/2,cy=150;
  const g=`<line x1="${ox+24}" y1="${cy}" x2="${ox+PW-24}" y2="${cy}" stroke="${AX}"/><line x1="${cx}" y1="${cy-104}" x2="${cx}" y2="${cy+104}" stroke="${AX}"/>`+
   [1,2].map(r=>covEllipse(cx,cy,ex[kind],k,r,palette.q)).join('')+label(ox+PW-26,cy+24,'θ₁',{size:18,anchor:'end'})+label(cx+8,cy-88,'θ₂',{size:18})+
   label(ox+4,24,COV_NAMES[kind],{fill:ink,size:23,weight:600})+matrix2(ox+70,288,'Σ =',ex[kind],palette.q,20,2)+label(ox+4,338,COV_COUNTS[kind],{size:19});
  s+=j?`<g data-at="${j+1}">${g}</g>`:g;});
 return fig(W,H,s,'Example contours: an isotropic Gaussian (circles), a diagonal one (axis-aligned ellipses) and a full-covariance one (tilted ellipses), with their covariance matrices.');
}
/** The fitted diagonal and full q from Section 3.3, against the tilted target. */
function covPair(){
 const W=560,H=300,k=34,Sp=[c3.target.cov[0][0],c3.target.cov[0][1],c3.target.cov[1][1]];let s='';
 [['diag','diagonal q (mean-field)'],['full','full-covariance q']].forEach(([kind,title],j)=>{const ox=j*290,cx=ox+135,cy=165,f=c3.runs[kind].at(-1);
  s+=`<line x1="${ox+10}" y1="${cy}" x2="${ox+260}" y2="${cy}" stroke="${AX}"/><line x1="${cx}" y1="${cy-115}" x2="${cx}" y2="${cy+115}" stroke="${AX}"/>`+
   [1,2].map(r=>covEllipse(cx,cy,Sp,k,r,palette.p)).join('')+[1,2].map(r=>covEllipse(cx+f[1]*k,cy-f[2]*k,[f[3],f[4],f[5]],k,r,palette.q)).join('')+
   label(ox+4,24,title,{fill:palette.q,size:20,weight:600})+label(ox+4,288,f[6]<5e-4?'KL ≈ 0':`KL ${num(f[6],2)}`,{fill:palette.q,size:20});});
 return fig(W,H,s,'Section 3.3 again: a diagonal q shrinks inside the tilted target; a full-covariance q matches it.');
}
const MIX=st.mixture_sampling,COMP=[palette.d,palette.noise];
/** The two-step story of a mixture: pick a bump z, then draw θ from that bump. */
function mixtureStory(){
 const W=540,H=360;
 const stoch=(cx,cy,r,t,color)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${TINT.noise}" stroke="${palette.noise}" stroke-width="3"/>`+label(cx,cy+10,t,{fill:color,size:30,anchor:'middle',weight:600});
 const arrow=(x1,y1,x2,y2)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2-12}" stroke="#3B4048" stroke-width="2.5"/><polygon points="${x2},${y2} ${x2-7},${y2-14} ${x2+7},${y2-14}" fill="#3B4048"/>`;
 let s=label(0,26,'step 1: pick a bump',{fill:ink,size:22,weight:600});
 [0,1].forEach(k=>{s+=`<rect x="${10+k*170}" y="44" width="150" height="64" rx="10" fill="${palette.paper}" stroke="${COMP[k]}" stroke-width="2.5"/>`+
  label(85+k*170,72,`bump ${k+1}`,{fill:COMP[k],size:21,anchor:'middle',weight:600})+label(85+k*170,98,'probability ½',{fill:COMP[k],size:18,anchor:'middle'});});
 s+=stoch(430,76,38,'z',palette.noise)+label(430,140,'z ~ Categorical(½, ½)',{fill:palette.noise,size:18,anchor:'middle'});
 s+=arrow(430,152,430,206)+label(0,196,'step 2: draw θ from bump z',{fill:ink,size:22,weight:600});
 s+=stoch(430,248,38,'θ',palette.p)+label(430,312,`θ | z ~ N(μ${sub('z')}, σ${sub('z')}²)`,{fill:palette.p,size:18,anchor:'middle'});
 s+=label(0,240,`bump 1: N(${num(MIX.means[0],1)}, ${num(MIX.sds[0],1)}²)`,{fill:COMP[0],size:20})+label(0,270,`bump 2: N(${num(MIX.means[1],1)}, ${num(MIX.sds[1],1)}²)`,{fill:COMP[1],size:20});
 s+=label(0,350,'We see θ. We never see which bump made it.',{size:19});
 return fig(W,H,s,'A mixture generates θ in two steps: a random choice of bump z, then a draw of θ from that bump’s Gaussian.');
}
/** The two weighted bells and their sum, the mixture density. */
function mixtureChart(){
 const xs=lin(-2,3,251),w=MIX.weights,c=k=>xs.map(t=>w[k]*normalPdf(t,{mu:MIX.means[k],sigma:MIX.sds[k]})),c0=c(0),c1=c(1),sum=xs.map((_,i)=>c0[i]+c1[i]);
 const P=panel({W:600,H:330,L:16,R:14,T:62,B:56,xMin:-2,xMax:3,yMax:.75,ticksX:[-2,-1,0,1,2,3],yAxis:false,xLabel:'θ'});
 return fig(600,330,P.axes+P.line(xs,c0,COMP[0],'stroke-dasharray="8 5"')+P.line(xs,c1,COMP[1],'stroke-dasharray="8 5"')+
  label(P.x(-1.3),P.y(.56),'½ N(−0.2, 0.4²)',{fill:COMP[0],size:19,anchor:'middle'})+label(P.x(2.15),P.y(.46),'½ N(1, 0.5²)',{fill:COMP[1],size:19,anchor:'middle'})+
  `<g data-at="3">${P.area(xs,sum,TINT.p)}${P.line(xs,sum,palette.p)}${legend(16,30,[['p(θ): the sum of the two',palette.p]],20)}</g>`,
  'Two Gaussian bells, each weighted by one half, and their sum: the mixture density.');
}

// Computational graphs in the style of Lilian Weng’s VAE post (after Kingma, NIPS 2015):
// boxes are deterministic nodes, circles stochastic nodes; dark arrows compute forward, red dashed arrows carry gradients back.
let graphId=0;
const FWD='#3B4048';
function graphDefs(id){
 const mk=(name,color)=>`<marker id="${id}-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${color}"/></marker>`;
 return `<defs>${mk('f',FWD)}${mk('b',palette.loss)}</defs>`;
}
const detNode=(x,y,w,h,t,sub,color,size=26)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${palette.paper}" stroke="${color}" stroke-width="3"/>`+
 label(x+w/2,y+(sub?h/2-2:h/2+9),t,{fill:color,size,anchor:'middle',weight:600})+(sub?label(x+w/2,y+h/2+24,sub,{size:18,anchor:'middle'}):'');
const rndNode=(cx,cy,r,t)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${TINT.noise}" stroke="${palette.noise}" stroke-width="3"/>`+label(cx,cy+10,t,{fill:palette.noise,size:28,anchor:'middle',weight:600});
const fwdEdge=(id,d)=>`<path d="${d}" fill="none" stroke="${FWD}" stroke-width="2.5" marker-end="url(#${id}-f)"/>`;
const backEdge=(id,d)=>`<path d="${d}" fill="none" stroke="${palette.loss}" stroke-width="2.5" stroke-dasharray="8 6" marker-end="url(#${id}-b)"/>`;
const blockMark=(x,y)=>`<g stroke="${palette.loss}" stroke-width="4" stroke-linecap="round"><line x1="${x-11}" y1="${y-11}" x2="${x+11}" y2="${y+11}"/><line x1="${x-11}" y1="${y+11}" x2="${x+11}" y2="${y-11}"/></g>`;
/** kind 'sample': θ is a stochastic node and the gradient stops there. kind 'reparam': θ = μ + σε is deterministic. */
function compGraph(kind,{compact=false}={}){
 const id=`cg${++graphId}`,W=1150,L=palette.loss;let s=graphDefs(id),H;
 const title=compact?label(1130,22,kind==='sample'?'with sample()':'with rsample()',{fill:ink,size:22,weight:600,anchor:'end'}):'';
 if(kind==='sample'){
  const y=compact?80:128,r=compact?46:56,bh=compact?72:80,dy=compact?30:34,bend=compact?62:80,lab=compact?86:100;H=compact?172:238;
  s+=detNode(20,y-bh/2,170,bh,'μ, σ',compact?'':'parameters of q',palette.q)+rndNode(520,y,r,'θ')+detNode(870,y-bh/2,260,bh,'loss',compact?'':'log q(θ) − log p(θ)',L);
  s+=fwdEdge(id,`M190,${y} L${520-r-2},${y}`)+label(322,y-12,'sample()',{fill:FWD,size:21,anchor:'middle'})+fwdEdge(id,`M${520+r},${y} L868,${y}`);
  s+=backEdge(id,`M870,${y+dy} Q728,${y+bend} ${520+r-6},${y+dy}`)+label(728,y+lab,'∂loss/∂θ ✓',{fill:L,size:20,anchor:'middle'});
  s+=backEdge(id,`M${520-r+6},${y+dy} Q322,${y+bend} 192,${y+dy}`)+blockMark(322,y+(bend+dy)/2)+label(322,y+lab,'blocked: no path back',{fill:L,size:20,anchor:'middle'});
  if(!compact)s+=fwdEdge(id,`M105,${y-bh/2} C105,${y-96} 1000,${y-96} 1000,${y-bh/2-2}`)+label(560,y-104,'μ and σ also appear inside log q(θ); only that path survives, and it averages to ≈ 0',{size:19,anchor:'middle'});
 }else{
  const y=compact?96:136,yp=compact?44:62,ye=compact?140:196,re=compact?28:34,bh=compact?64:78;H=compact?178:262;
  s+=detNode(20,yp-bh/2+(compact?4:6),170,bh-(compact?8:12),'μ, σ',compact?'':'parameters',palette.q)+rndNode(105,ye,re,'ε')+detNode(420,y-bh/2,240,bh,'θ = μ + σε',compact?'':'deterministic',palette.noise,24)+detNode(870,y-bh/2,260,bh,'loss',compact?'':'log q(θ) − log p(θ)',L);
  if(!compact)s+=label(20,ye+re+26,'ε ~ N(0, 1): no parameters',{fill:palette.noise,size:19});
  s+=fwdEdge(id,`M190,${yp} L418,${y-18}`)+fwdEdge(id,`M${105+re},${ye} L418,${y+18}`)+fwdEdge(id,`M660,${y} L868,${y}`);
  s+=backEdge(id,`M870,${y+bh/2-6} Q765,${y+(compact?62:84)} 662,${y+bh/2-6}`)+label(765,y+(compact?70:94),'∂loss/∂θ ✓',{fill:L,size:20,anchor:'middle'});
  s+=backEdge(id,`M440,${y-bh/2} Q330,${compact?4:8} 192,${yp-(compact?14:16)}`)+label(compact?300:310,compact?18:24,'∂θ/∂μ = 1,  ∂θ/∂σ = ε  ✓',{fill:L,size:compact?18:20,anchor:'middle'});
 }
 return fig(W,H,title+s,kind==='sample'?'Computational graph with sample(): parameters mu and sigma feed a random node theta, which feeds the loss; the gradient from the loss reaches theta but is blocked between theta and the parameters.':
  'Computational graph with the reparameterisation: noise epsilon is a parameter-free random node; theta = mu + sigma epsilon is deterministic, so the gradient flows from the loss through theta to mu and sigma.');
}
function graphLegend(){
 const id=`cg${++graphId}`;
 return fig(1150,36,graphDefs(id)+`<rect x="0" y="6" width="34" height="26" rx="6" fill="${palette.paper}" stroke="${ink}" stroke-width="2.5"/>`+label(44,27,'deterministic node',{size:20})+
  `<circle cx="262" cy="19" r="14" fill="${TINT.noise}" stroke="${palette.noise}" stroke-width="2.5"/>`+label(284,27,'stochastic node',{size:20})+
  fwdEdge(id,'M470,19 L520,19')+label(530,27,'forward: compute',{size:20})+backEdge(id,'M730,19 L780,19')+label(790,27,'backward: gradient',{size:20})+
  blockMark(1000,19)+label(1020,27,'blocked',{size:20}),'Legend: boxes are deterministic nodes, circles stochastic nodes; dark arrows compute, red dashed arrows carry gradients.');
}

// ---------------------------------------------------------------- Part 4 figure: as q improves, the ELBO rises and the KL gap shrinks
function elboLadder(){
 const rows=c.candidates,lo=-22.2,hi=-21.2,W=1000,H=300,L=250,Rm=60,X=v=>L+(v-lo)/(hi-lo)*(W-L-Rm),xe=X(c.log_evidence);
 let s=`<line x1="${xe.toFixed(1)}" y1="28" x2="${xe.toFixed(1)}" y2="${H-16}" stroke="${palette.data}" stroke-width="2" stroke-dasharray="6 5"/>`+label(xe,22,`log p(𝒟) = ${num(c.log_evidence,2)}`,{fill:palette.data,size:20,anchor:'middle'});
 rows.forEach((r,k)=>{const yy=76+k*80,xq=X(r.elbo);
  s+=label(0,yy+7,`q = Beta(${r.a}, ${r.b})`,{fill:palette.q,size:21})+`<line x1="${L}" y1="${yy}" x2="${W-Rm}" y2="${yy}" stroke="${AX}"/>`+
   `<circle cx="${xq.toFixed(1)}" cy="${yy}" r="7" fill="${palette.q}"/><circle cx="${xe.toFixed(1)}" cy="${yy}" r="7" fill="${palette.data}"/>`+
   label(xq-12,yy-12,`ELBO ${num(r.elbo,2)}`,{fill:palette.q,size:18,anchor:'end'})+
   (r.posterior_kl>1e-6?`<path d="M${xq.toFixed(1)},${yy+12} V${yy+22} H${xe.toFixed(1)} V${yy+12}" fill="none" stroke="${palette.loss}" stroke-width="2"/>`+
    (xe-xq>120?label((xq+xe)/2,yy+42,`KL ${num(r.posterior_kl,2)}`,{fill:palette.loss,size:18,anchor:'middle'}):label(xe+14,yy+30,`KL ${num(r.posterior_kl,2)}`,{fill:palette.loss,size:18})):label(xe+14,yy+30,'KL = 0',{fill:palette.loss,size:18}));});
 return fig(W,H,s,'Three Beta q’s for the coin: as q improves the ELBO moves right towards the fixed log evidence, and the KL gap shrinks to zero.');
}

// ---------------------------------------------------------------- Part 3 figures kept
const CANDS=[[0,2],[1,3],[4,1],[-3,4],[1,8],[2,5]],P0={mu:1,sigma:4};
const candKL=CANDS.map(([mu,sigma])=>normalKL({mu,sigma},P0)),candBest=candKL.indexOf(Math.min(...candKL));
function sixCandidates(){
 const xs=lin(-16,18,241),py=xs.map(u=>normalPdf(u,P0)),W=1150,PW=370,PH=180,G=20;let s='';
 CANDS.forEach(([mu,sigma],k)=>{const ox=(k%3)*(PW+G),oy=Math.floor(k/3)*PH;
  const P=panel({ox,oy,W:PW,H:PH,L:10,R:10,T:46,B:14,xMin:-16,xMax:18,yMax:.42,yAxis:false});
  s+=P.axes+P.area(xs,py,TINT.p)+P.line(xs,py,palette.p)+P.line(xs,xs.map(u=>normalPdf(u,{mu,sigma})),palette.q)+
   label(ox+10,oy+30,`q = N(${num(mu,0)}, ${sigma}²)`,{fill:palette.q,size:22})+
   `<g data-at="2">${label(ox+PW-10,oy+30,`KL ${num(candKL[k])}`,{fill:k===candBest?palette.loss:ink,size:22,anchor:'end',weight:k===candBest?'700':''})}</g>`;});
 return fig(W,2*PH,s,'Six candidate Gaussians against the same target p = N(1, 4²), each with its KL.');
}
/** KL(q‖p) over a grid of (μ, σ). bestAt and pathAt are the builds that reveal the minimum and the optimiser’s path. */
function klGrid({bestAt=1,pathAt=0}={}){
 const W=600,H=340,L=66,R=16,T=44,B=56,nx=51,ny=40,mu0=-4.1,mu1=6.1,dl=Math.log(25)/ny,ls0=Math.log(4)-28.5*dl,ls1=ls0+ny*dl;
 const X=v=>L+(v-mu0)/(mu1-mu0)*(W-L-R),Y=ls=>H-B-(ls-ls0)/(ls1-ls0)*(H-T-B);
 const cuts=[.01,.03,.1,.2,.4,.7,1,1.5,2],ramp=['#F7F3FF','#EBDFFD','#DBC6FA','#C7A5F4','#AE7FEA','#9258D8','#7537BE','#5A2299','#421473','#2E0C52'];
 const color=v=>{const k=cuts.findIndex(t=>v<t);return ramp[k<0?cuts.length:k];};
 const cw=(W-L-R)/nx,chh=(H-T-B)/ny;let s='',best={v:Infinity};
 for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
  const mu=mu0+(i+.5)*(mu1-mu0)/nx,sigma=Math.exp(ls0+(j+.5)*(ls1-ls0)/ny),v=normalKL({mu,sigma},P0);
  if(v<best.v)best={v,mu,sigma};
  s+=`<rect x="${(L+i*cw).toFixed(1)}" y="${(H-B-(j+1)*chh).toFixed(1)}" width="${(cw+.5).toFixed(1)}" height="${(chh+.5).toFixed(1)}" fill="${color(v)}"/>`;}
 s+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="${AX}"/><line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="${AX}"/>`;
 for(const v of [-4,-2,0,2,4,6])s+=label(X(v),H-B+24,num(v,0),{size:20,anchor:'middle'});
 for(const v of [.5,1,2,4,8])s+=label(L-9,Y(Math.log(v))+7,String(v),{size:20,anchor:'end'});
 s+=label(W-R,H-9,'mean μ',{size:20,anchor:'end'})+label(L,T-16,'width σ, log scale',{size:20})+label(W-R,T-16,'darker = larger KL',{size:20,anchor:'end'});
 const wrap=(k,inner)=>k?`<g data-at="${k}">${inner}</g>`:inner;
 if(pathAt){const fr=st.olive.frames;s+=wrap(pathAt,`<polyline points="${fr.map(r=>`${X(r[1]).toFixed(1)},${Y(Math.log(r[2])).toFixed(1)}`).join(' ')}" fill="none" stroke="${palette.p}" stroke-width="3.5"/><circle cx="${X(4).toFixed(1)}" cy="${Y(Math.log(.5)).toFixed(1)}" r="7" fill="${palette.p}"/>${label(X(4)+12,Y(Math.log(.5))-10,'start',{fill:palette.p,size:20,weight:600})}`);}
 s+=wrap(bestAt>1?bestAt:0,`<circle cx="${X(best.mu).toFixed(1)}" cy="${Y(Math.log(best.sigma)).toFixed(1)}" r="9" fill="none" stroke="${palette.loss}" stroke-width="3"/>${label(X(best.mu)+14,Y(Math.log(best.sigma))-12,`best: N(${num(best.mu,1)}, ${num(best.sigma,1)}²)`,{fill:palette.loss,size:20,weight:600})}`);
 return fig(W,H,s,'KL from q = N(μ, σ²) to p = N(1, 4²) over a 51 by 40 grid of means and widths.');
}
function reparamPicture(){
 const W=1150,H=270,eps=-.75,us=lin(-8,8,241);let s='';
 [{title:'move μ from 0 to 1',from:{mu:0,sigma:2},to:{mu:1,sigma:2}},{title:'widen σ from 2 to 3',from:{mu:0,sigma:2},to:{mu:0,sigma:3}}].forEach((pp,k)=>{
  const ox=k*590,P=panel({ox,W:560,H,L:14,R:14,T:56,B:52,xMin:-8,xMax:8,yMax:.26,ticksX:[-6,-3,0,3,6],yAxis:false,xLabel:'θ'});
  const u0=pp.from.mu+pp.from.sigma*eps,u1=pp.to.mu+pp.to.sigma*eps,y=P.y(0)-24,dir=Math.sign(u1-u0);
  s+=P.axes+P.line(us,us.map(u=>normalPdf(u,pp.from)),'#C4B5FD',`style="stroke-dasharray:8 6"`)+P.line(us,us.map(u=>normalPdf(u,pp.to)),palette.q)+
   `<line x1="${P.x(u0).toFixed(1)}" y1="${y}" x2="${(P.x(u1)-dir*9).toFixed(1)}" y2="${y}" stroke="${palette.noise}" stroke-width="3"/>`+
   `<polygon points="${P.x(u1).toFixed(1)},${y} ${(P.x(u1)-dir*12).toFixed(1)},${y-7} ${(P.x(u1)-dir*12).toFixed(1)},${y+7}" fill="${palette.noise}"/>`+
   `<circle cx="${P.x(u0).toFixed(1)}" cy="${P.y(0).toFixed(1)}" r="6" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/><circle cx="${P.x(u1).toFixed(1)}" cy="${P.y(0).toFixed(1)}" r="6" fill="${palette.noise}"/>`+
   label(ox+14,30,pp.title,{fill:ink,size:22,weight:600})+label(ox+546,30,`θ: ${num(u0)} → ${num(u1)}`,{fill:palette.noise,size:22,anchor:'end'});
 });
 return fig(W,H,s,'With epsilon fixed at −0.75, moving mu shifts the draw theta by the same amount; widening sigma moves it by epsilon times the change.');
}
function viVsHmc(full=false){
 const h=b.hmc,v=full?{mean:VF.mean,sd:VF.sd_pred}:b.vi,P=fxPanel();
 const band=(mean,sd,fill)=>P.band(b.grid,mean.map((x,k)=>x-2*sd[k]),mean.map((x,k)=>x+2*sd[k]),fill);
 return fig(600,330,P.axes+band(h.mean,h.sd,'#0F766E30')+band(v.mean,v.sd,'#8B2CDE40')+P.line(b.grid,h.mean,palette.p)+P.line(b.grid,v.mean,palette.q,`style="stroke-dasharray:8 5"`)+dataDots(P)+
  legend(52,30,[['reference (MCMC)',palette.p],[full?'full-covariance VI':'mean-field VI',palette.q,true]]),`Predictive bands from ${full?'full-covariance':'mean-field'} VI and from the MCMC reference.`);
}

// ================================================================ Part 1 · Bayes’ rule for a neural network
const big=(v,k=2)=>Number(v).toLocaleString('en',{minimumFractionDigits:k,maximumFractionDigits:k}).replace('-','−');
const effective=Math.round(E.runs.reduce((s,r)=>s+r.effective_draws,0)/E.runs.length);
const lt=A.log_lik_points,lp=A.log_prior_terms;
chap('1.1');
S('bayes-rule','bayes','Bayes’ rule: update what you believe, using data',{builds:2,
 body:act(1,eq(R`\pcol{p(A\mid B)}=\frac{p(B\mid A)\,p(A)}{\dcol{p(B)}}`),
  at(2,eq(R`\underbrace{p(A\mid B)}_{\text{posterior}}=\dfrac{\overbrace{p(B\mid A)}^{\text{likelihood}}\ \ \overbrace{p(A)}^{\text{prior}}}{\underbrace{p(B)}_{\text{evidence}}}`))),
 lines:['Bayes’ rule: what to believe after seeing B, from what we believed before and how well each belief explains B.','Four names for the whole lecture: posterior, likelihood, prior and evidence.'],
 ask:['Which term is the belief before seeing B?','Which term only rescales?'],
 expected:['p(A), the prior.','The evidence p(B): it does not depend on A; it makes the posterior sum to one.'],
 transition:'In machine learning, what are A and B?'});
S('bayes-ml','bayes','In machine learning, the unknown is the parameters',{builds:2,
 body:stack(eq(R`\pcol{p(\theta\mid\mathcal D)}=\frac{p(\mathcal D\mid\theta)\,p(\theta)}{\dcol{p(\mathcal D)}}`),
  at(2,cols(stack(p('<b>Usual training</b>'),p(`keeps one ${m(R`\theta`)}: the best fit to the data ${m(R`\mathcal D`)}.`)),stack(p('<b>Bayesian learning</b>'),p(`keeps every ${m(R`\theta`)}, weighted by ${m(R`p(\theta\mid\mathcal D)`)}.`))))),
 lines:['A becomes θ, every weight and bias of the model; B becomes 𝒟, the training data.','Usual training keeps one θ. The Bayesian answer keeps a whole distribution over θ.'],
 ask:['What plays the role of A? Of B?','What does a distribution over θ buy us?'],
 expected:['A is θ, the parameters; B is the dataset 𝒟 of (x, y) pairs.','Uncertainty: predictions with error bars, by averaging over all plausible θ.'],
 transition:'Make θ concrete with the smallest interesting network.'});
chap('1.2');
S('tiny-net','bayes','A tiny network: one input, two hidden units, one output',{builds:2,
 body:cols(nnDiagram(),stack(eq(R`\begin{aligned}f_\theta(x)&=v_1\tanh(w_1x+b_1)\\&\quad+v_2\tanh(w_2x+b_2)+c\end{aligned}`),
  at(2,eq(R`\ncol{\theta}=(w_1,w_2,b_1,b_2,v_1,v_2,c)\in\mathbb R^7`)),at(2,p('Seven unknown numbers: four weights and three biases.'))),'1fr 1.1fr'),
 lines:['Our running example: one input x, two tanh hidden units and one output f.','Every weight and bias is unknown, so θ is seven numbers.'],
 ask:['How many weights? How many biases?','What would a bigger network add?'],
 expected:['Four weights (w₁, w₂, v₁, v₂) and three biases (b₁, b₂, c).','Each extra hidden unit adds a weight in, a bias and a weight out. Real networks have millions of parameters.'],
 transition:'Before any data: a prior on all seven.'});
S('priors','bayes','A prior on every parameter',{builds:2,
 body:stack(cols(eq(R`\theta_j\sim\N(0,1),\quad j=1,\dots,7`),eq(R`p(\theta)=\prod_{j=1}^{7}\N(\theta_j;\,0,\,1)`)),at(2,bellsRow(b.prior_thetas[0]))),
 lines:['Before seeing data, every parameter gets its own prior: θⱼ ~ N(0, 1), independently.','One draw from the prior is seven numbers, one from each bell: a single point θ in ℝ⁷.'],
 ask:['Why independent N(0, 1) priors?','How many numbers make one draw of θ?'],
 expected:['Simplicity: before data nothing ties one weight to another, and very large weights are unlikely.','Seven, one per parameter. This draw is θ⁽¹⁾ on the next slide.'],
 transition:'What does one draw of θ look like as a network?'});
S('one-draw','bayes','One draw of θ is one whole function',{nb:'cell-prior-draws',builds:3,
 body:oneDrawFigure(),
 lines:['Draw θ⁽¹⁾ from the prior, put its seven numbers into the network, and evaluate f at every x: one curve.','A second draw, θ⁽²⁾, gives a different curve.','Every point of ℝ⁷ is a whole function. Drawing θ from the prior draws functions.'],
 ask:['Look at θ⁽¹⁾. Which numbers decide where its curve steps, and how high?','Why does θ⁽²⁾ give a different shape?','So what does the prior on θ say about functions?'],
 expected:['b/w sets where each tanh steps, |w| how sharply, v how high; c shifts the whole curve.','Different weights and biases: steps in other places, of other heights and signs.','It is a distribution over functions: mostly smooth curves within about ±3.'],
 point:'θ⁽¹⁾ is the draw marked on the bells on the previous slide. θ⁽¹⁾, θ⁽²⁾ and θ⁽³⁾ are the first three of the twenty lines on the next slide.',
 transition:'Twenty draws at once.'});
S('prior-functions','bayes','Twenty draws, and the prior predictive',{nb:'cell-prior-draws',demo:'Notebook 1.2: the prior-draws cell prints θ⁽¹⁾ and draws the twenty functions with the prior predictive band; change the seed for new draws.',builds:2,
 body:widget('priorfuncs','Twenty prior draws of the network; hover over a line to see its seven parameter values'),
 lines:['Twenty draws of θ, twenty functions. Hover over a line, or step through, to see its point in ℝ⁷.','Average over every θ and add the 0.1 noise: the prior predictive, p(y | x) = ∫ p(y | x, θ) p(θ) dθ. What the model expects before any data.'],
 ask:['Which numbers make a line steep? Which shift it up or down?','Before any data, how sure is the model about y at x = 0?'],
 expected:['A large |w| makes a sharp step; c shifts the whole curve; v sets how high each step is.',`Not at all: the band is about ±${num(2*b.prior_predictive.sd[60],1)} wide everywhere.`],
 point:`Prior over functions: the individual curves. Prior predictive: their average with the noise added, the band (4,000 draws). Section 1.5 shows the same picture after the data: the posterior predictive.`,
 transition:'Now the data, and how well each θ explains them.'});
chap('1.3');
S('data-likelihood','bayes','The data, and how well one θ explains them',{builds:2,
 body:cols(dataChart(),stack(eq(R`y_i\mid x_i,\theta\ \sim\ \N\big(f_\theta(x_i),\,0.1^2\big)`),at(2,eq(R`p(\mathcal D\mid\theta)=\prod_{i=1}^{17}\N\big(y_i;\,f_\theta(x_i),\,0.1^2\big)`))),'1.05fr 1fr'),
 lines:['Seventeen training points with a gap in the middle. Each y is the network’s output plus Gaussian noise of size 0.1.','The likelihood of θ multiplies all seventeen densities: how well this one θ explains the whole dataset.'],
 ask:['What does the 0.1 describe?','Why a product?'],
 expected:['The noise standard deviation, assumed known here.','Given θ, the noise on different points is independent.'],
 point:`Data: ${b.data}.`,
 transition:'Compute the right-hand side of Bayes’ rule for one θ.'});
S('map-train','bayes','Finding a good θ: train the network',{nb:'cell-map',demo:'Switch to notebook 1.3 and run the training cell: it prints the final loss of all ten starts, then the log likelihood and log prior of θ_A and θ_B.',builds:2,
 body:stack(eq(R`\theta_A=\arg\max_\theta\ \big[\log p(\mathcal D\mid\theta)+\log p(\theta)\big]`),widget('mapfit','Training the network from ten random starts')),
 lines:['To pick one good θ, train the network the usual way: maximise log p(𝒟 | θ) + log p(θ). That is squared error plus weight decay.','From a random start, gradient descent bends the curve onto the data. Ten starts finish at slightly different losses; the best finish is θ<sub>A</sub>.'],
 ask:['What is the loss here, in words?','Why try ten starts?'],
 expected:['The squared error between the curve and the points, divided by 2 · 0.1², plus half the sum of squared parameters: the prior acts as weight decay.','Training can stop in different places. Here most starts reach 14.34; two stop a little higher.'],
 point:`Loss = Σᵢ (yᵢ − f(xᵢ))² / (2 · 0.1²) + ½ Σⱼ θⱼ², which is −log p(𝒟 | θ) − log p(θ) without its constants. Adam, learning rate 0.05, 2,000 steps per start. θ<sub>A</sub> is the MAP estimate.`,
 transition:'Meet θ<sub>A</sub>.'});
S('theta-a','bayes','Meet θ<sub>A</sub>: the network with its trained numbers',{nb:'cell-map',builds:2,
 body:cols(nnDiagram(A.theta),at(2,thetaACurve()),'1fr 1fr'),
 lines:['θ<sub>A</sub> is one point in ℝ⁷: these seven numbers, written on the network.','With these numbers the network’s curve follows the data. Next, score θ<sub>A</sub> with the likelihood and the prior.'],
 ask:['Which weight is largest?','Does this curve explain the data well?'],
 expected:[`w₂ = ${num(A.theta[1])}: the second hidden unit makes the sharp step.`,'Yes: it passes close to every point.'],
 transition:'The likelihood of θ<sub>A</sub>.'});
S('likelihood-formula','bayes','The likelihood of θ<sub>A</sub>: formula, then value',{nb:'cell-map',builds:3,
 body:stack(eq(R`p(\mathcal D\mid\theta_A)=\prod_{i=1}^{17}\N\big(y_i;\,f_{\theta_A}(x_i),\,0.1^2\big)`),
  at(2,eq(R`\log p(\mathcal D\mid\theta_A)=-\frac{1}{2\cdot0.1^2}\sum_{i=1}^{17}\big(y_i-f_{\theta_A}(x_i)\big)^2-\frac{17}{2}\log\big(2\pi\cdot0.1^2\big)`)),
  at(3,eq(R`\log p(\mathcal D\mid\theta_A)=${num(A.log_lik)}`))),
 lines:['Fix θ = θ<sub>A</sub>. The likelihood multiplies one Gaussian density per data point: how well the curve explains each y.','Take logs: the product becomes a sum of squared errors, scaled by the noise, plus a constant.','Put in θ<sub>A</sub>’s curve and the seventeen points: log p(𝒟 | θ<sub>A</sub>) = 12.17.'],
 ask:['Why a product?','Which part depends on θ?','Is a larger log likelihood better?'],
 expected:['Given θ, the noise on different points is independent.','Only the squared errors; the constant is the same for every θ.','Yes: the curve passes closer to the points.'],
 transition:'The prior of θ<sub>A</sub>.'});
S('prior-formula','bayes','The prior of θ<sub>A</sub>, then the evidence',{nb:'cell-map',builds:3,
 body:stack(eq(R`p(\theta_A)=\prod_{j=1}^{7}\N(\theta_j;\,0,\,1)`),
  at(2,eq(R`\log p(\theta_A)=-\frac12\sum_{j=1}^{7}\theta_j^2-\frac72\log2\pi=${num(A.log_prior)}`)),
  at(3,eq(R`p(\mathcal D)=\int p(\mathcal D\mid\theta)\,p(\theta)\,d\theta=\ \boldsymbol{?}`))),
 lines:['The prior multiplies one N(0, 1) density per parameter.','Take logs: minus half the sum of squared parameters, plus a constant. For θ<sub>A</sub> it is −9.41; the large w₂ costs the most.','The third term, the evidence, is not about θ<sub>A</sub>: it needs every θ at once.'],
 ask:['Which parameters make the prior small?','What is ½ Σ θⱼ² for θ<sub>A</sub>?','How could we compute the evidence?'],
 expected:['Large ones, positive or negative.',`About ${num(A.theta.reduce((a,v)=>a+v*v,0)/2)}: the weight-decay term from training.`,'Add up p(𝒟 | θ) p(θ) over all θ. Next slide.'],
 transition:'Compare θ<sub>A</sub> with a random θ from the prior.'});
S('two-thetas','bayes','Two settings of θ, side by side',{nb:'cell-map',builds:2,
 body:cols(twoThetasChart(),at(2,table(['',m(R`\theta_A`),`${m(R`\theta_B`)}, a prior draw`],[
  [m(R`\log p(\mathcal D\mid\theta)`),num(A.log_lik,1),big(Bt.log_lik,1)],
  [m(R`\log p(\theta)`),num(A.log_prior,1),num(Bt.log_prior,1)],
  ['sum of the two',num(Number(A.log_lik.toFixed(1))+Number(A.log_prior.toFixed(1)),1),big(Number(Bt.log_lik.toFixed(1))+Number(Bt.log_prior.toFixed(1)),1)]])),'1.1fr 1fr'),
 lines:['θ<sub>A</sub> follows the data. θ<sub>B</sub>, one random draw from the prior, does not.','Their priors are similar; their likelihoods differ by over 2,000 nats. A ratio of two posteriors does not even need p(𝒟).'],
 ask:['Which θ explains the data better, and by how much?','How many times more probable is θ<sub>A</sub> than θ<sub>B</sub> after seeing the data?'],
 expected:['θ<sub>A</sub>, by about 2,170 nats of log likelihood.','About e²¹⁷⁰ times. p(𝒟) cancels in a ratio of posteriors: it is needed only to normalise.'],
 transition:'Normalising is the hard part.'});
chap('1.4');
S('evidence-area','bayes','The evidence is an area: add up thin rectangles',{nb:'cell-evidence-1d',demo:'Notebook 1.4: the first cell computes this one-unknown evidence with 20 rectangles and with 200,000; their logs agree to three decimals.',builds:2,
 body:cols(evidenceArea(),stack(p(`Suppose only ${m(R`w_2`)} were unknown, with the other six numbers fixed at θ<sub>A</sub>:`),eq(R`p(\mathcal D)=\int p(\mathcal D\mid w_2)\,p(w_2)\,dw_2`),
  at(2,eq(R`\approx\sum_{k=1}^{20}p(\mathcal D\mid w_{2,k})\,p(w_{2,k})\,\Delta w`))),'1.05fr 1fr'),
 lines:['With one unknown, the evidence is the area under one curve: likelihood times prior.','Chop the area into thin rectangles and add them up. Twenty rectangles already match a much finer grid to three decimals.'],
 ask:['What is the height of each rectangle?','How many rectangles would two unknowns need?'],
 expected:['p(𝒟 | w₂) p(w₂) at its centre: the likelihood times the prior.','A grid of squares: 20 × 20 = 400, or 100 × 100 if each axis needs 100.'],
 point:'A one-dimensional slice of our network’s evidence, for illustration only: the curve is computed in the browser from θ<sub>A</sub> and the data.',
 transition:'And with all seven unknowns?'});
S('intractable','bayes','Intractable: why the rectangles do not scale',{builds:2,
 body:cols(gridDots(),stack(p('<b>Intractable</b>: no formula, and a cost that grows exponentially with the number of unknowns.'),
  at(2,table(['unknowns','grid points'],[['1','100'],['2','10,000'],['7, our network','10<sup>14</sup>: a day'],['10<sup>6</sup>','10<sup>2,000,000</sup>: never']]))),'1fr 1fr'),
 lines:['With 100 rectangles per unknown, two unknowns already need 10,000 boxes.','Our seven need 10¹⁴; a real network, 10<sup>2,000,000</sup>. That is what intractable means.'],
 ask:['How long would 10¹⁴ evaluations take at a billion a second?','So what does intractable mean?'],
 expected:['About 28 hours: possible, just. For 10^2,000,000: never.','No closed form, and no feasible numerical computation: the cost explodes with the number of unknowns.'],
 point:'Sampling θ from the prior and averaging p(𝒟 | θ) does not rescue it: almost every draw explains the data terribly. The optional slide on prior sampling shows the numbers: with ten draws the estimate is off by about 1,000 nats, and of ten million draws only about 100 matter.',
 transition:'Why do we want the posterior at all?'});
chap('1.5');
S('predictive','bayes','What we want in the end: predictions with error bars',{nb:'cell-hmc',builds:2,
 body:cols(predictiveChart(),stack(eq(R`\begin{aligned}p(y^*\mid x^*,\mathcal D)&=\int p(y^*\mid x^*,\theta)\\&\qquad\times\pcol{p(\theta\mid\mathcal D)}\,d\theta\end{aligned}`),
  at(2,p('Average every network’s prediction, weighted by the posterior. The band widens where there are no data.'))),'1.05fr 1fr'),
 lines:['The posterior is a means to an end. The posterior predictive averages every network’s prediction at a new x*.','Here it comes from a long MCMC run, used only as a reference. The band widens beyond the data and in the gap.'],
 ask:['Where should the model be least sure?','How would you compute this integral?'],
 expected:['Far from the data: beyond both ends, and in the gap.','Average f<sub>θ</sub>(x*) over samples of θ from the posterior, which we cannot draw directly. Yet.'],
 point:`Reference: Hamiltonian Monte Carlo, 4 chains of 2,000 samples, acceptance about ${num(H.acceptance.reduce((a,v)=>a+v,0)/H.acceptance.length)}. Predictive sd ${num(H.sd[0])} at x = −2.5, about ${num(Math.min(...H.sd))} in the data.`,
 transition:'How else could we approximate the posterior?'});
S('core-problem','bayes','The core problem of probabilistic machine learning',{builds:2,
 body:stack(eq(R`\pcol{p(\theta\mid\mathcal D)}=\frac{\overbrace{p(\mathcal D\mid\theta)}^{\text{easy}}\ \ \overbrace{p(\theta)}^{\text{easy}}}{\underbrace{\dcol{p(\mathcal D)}}_{\text{intractable}}}`),
  at(2,`<p class="result" style="font-size:34px;color:var(--ink)">How do we compute, or approximate, the posterior p(θ | 𝒟)?</p>`),
  at(2,small('Bayesian neural networks, VAEs, mixture models and topic models all meet this same question.'))),
 lines:['The likelihood and the prior are easy to evaluate for any θ. The evidence, which turns them into the posterior, is not.','So the central question of probabilistic machine learning: how do we compute, or approximate, the posterior?'],
 ask:['Which of the three pieces can we compute for one θ?','Where else will you meet this problem?'],
 expected:['The likelihood and the prior, as we did for θ<sub>A</sub>. Not the evidence.','In every Bayesian model with many unknowns, including the VAE in Lecture 3.'],
 transition:'Three ways to approach it.'});
S('three-ways','bayes','Three ways to approximate the posterior; we take the third',{nb:'cell-three-ways',builds:3,
 body:grid3(stack(threeWay('laplace'),small(`A Gaussian at the highest peak: N(${num(st.three.laplace[0],1)}, ${num(st.three.laplace[1],1)}²). It never sees the second bump.`)),
  at(2,stack(threeWay('mcmc'),small(`20,000 random-walk steps. Their mean, ${num(st.three.mcmc.mean)}, matches the true ${num(st.three.true_mean)}.`))),
  at(3,stack(threeWay('vi'),small(`The closest Gaussian, found by optimisation: N(${num(st.three.vi[0])}, ${num(st.three.vi[1])}²).`)))),
 lines:['Laplace: fit a Gaussian to the curvature at the highest peak.','MCMC: a random walk whose visits follow the posterior. Right in the long run, but slow.','Variational inference: the closest member of a simple family, found by optimisation. The rest of the lecture develops only this one.'],
 ask:['What does Laplace miss here?','What does MCMC cost?','What does VI trade away?'],
 expected:['The second bump: it only looks at the peak.','Many steps, each evaluating the model on the data; hard to scale to large models and datasets.','Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.'],
 point:'Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.',
 transition:'Variational inference, in one slide.'});
S('vi-idea','bayes','Variational inference: choose q, make it close to the posterior',{builds:3,
 body:stack(eq(R`q_\psi(\theta)=\prod_{j=1}^{7}\N(\theta_j;\,\mu_j,\,\sigma_j^2),\qquad\psi=(\mu_1,\dots,\mu_7,\ \sigma_1,\dots,\sigma_7)`),at(2,qFactors()),
  at(3,eq(R`\psi^*=\arg\min_\psi\ \text{distance}\big(q_\psi(\theta),\ \pcol{p(\theta\mid\mathcal D)}\big)`))),
 lines:['Pick a simple family: one Gaussian per parameter, each with its own mean and width. ψ is fourteen numbers.','One bell per parameter. Together they define q(θ), and drawing θ from q is easy.','Tune ψ until q is as close as possible to the posterior. Two questions: what is “close”, and how do we tune without the posterior?'],
 ask:['How many numbers does q have?','Why is drawing θ from q easy?','What do we need before we can minimise anything?'],
 expected:['Fourteen: a mean and a width for each of the seven parameters.','Each θⱼ is an independent Gaussian draw.','A number for the distance between two distributions. That is Part 2.'],
 transition:'Part 2: how far apart are two distributions?'});

// ================================================================ Part 2 · How far apart are two distributions?
chap('2.1');
S('pretend-posterior','kl','For now, pretend we know the posterior',{builds:3,
 body:act(2,cols(pretendChart(),stack(eq(R`\pcol{p(\theta\mid\mathcal D)}=\N(1,\,4^2)`),small('A stand-in for the real posterior: known, one-dimensional, one bump.'),
  at(2,eq(R`\qcol{q(\theta)}=\N(\mu,\,\sigma^2)`)),at(2,small('Our approximation: we choose μ and σ. Shown: μ = 0, σ = 2.')),
  at(3,p('How far apart are p and q? We need one number, so that we can make it small.'))),'1.1fr 1fr')),
 lines:['To learn the tools, pretend for a while that we know the posterior: one Gaussian, in one dimension.','Our approximation q is another Gaussian; we choose its mean and its width.','How far apart are p and q? We need one number, and we will build it from scratch, starting from surprise.'],
 ask:['What do we gain by pretending we know p?','What can we change about q?','What would you compare: means, widths, something else?'],
 expected:['We can check every tool against the right answer before facing a real posterior.','Its mean μ and its width σ.','Students suggest means and widths. The KL divergence compares the whole densities, and works for any shapes.'],
 point:'From here to the end of Part 3, p is a known stand-in for the posterior. Part 4 returns to the real one.',
 transition:'Start with the surprise of one outcome.'});
S('surprise-forecast','kl','How surprised would you be? A forecast of 90% sun',{builds:2,nb:'cell-surprise',
 body:cols(forecastFig(),stack(p('The forecast for tomorrow: sun with probability 0.9, rain with probability 0.1.'),
  at(2,p('If it rains you are surprised; if the sun shines, hardly at all. <b>Rare outcomes carry more surprise.</b>')),
  at(2,small('Surprise is information: the outcome you did not expect tells you the most.'))),'1fr 1fr'),
 lines:['Start with one outcome. The forecast says 90% sun, 10% rain.','Rain would surprise you; sun would not. We want a number for surprise: big for rare outcomes, small for common ones.'],
 ask:['Tomorrow it rains. How surprised are you, compared with a sunny day?','Should a 100% certain outcome surprise you at all?'],
 expected:['Much more: rain had only a 10% chance.','No: zero surprise. That is our first requirement.'],
 source:HUANG,transition:'What should the surprise function be?'});
S('surprise-candidates','kl','What should surprise be? Three requirements',{builds:3,nb:'cell-surprise',
 body:cols(stack(eq(R`I(1)=0`),small('a certain outcome is no surprise'),eq(R`p\downarrow\ \Rightarrow\ I(p)\uparrow`),small('rarer means more surprising'),
  eq(R`I(p_1\,p_2)=I(p_1)+I(p_2)`),small('two independent days: probabilities multiply, surprises add')),
  table(['candidate','I(1) = 0','rarer, more','days add: I(½ · 0.1) vs I(½) + I(0.1)'],[
   [at(2,m(R`1-p`)),at(2,'✓'),at(2,'✓'),at(2,'✗ 0.95 vs 1.4')],
   [at(2,m(R`1/p`)),at(2,'✗ 1'),at(2,'✓'),at(2,'✗ 20 vs 12')],
   [at(3,m(R`\log_2\frac1p`)),at(3,'✓'),at(3,'✓'),at(3,'✓ 4.32 = 1 + 3.32')]]),'0.75fr 1.35fr'),
 lines:['Surprise I(p) should be zero for a certain outcome, larger for rarer ones, and for two independent days, whose probabilities multiply, the surprises should add.','Test 1 − p and 1/p on two days with probabilities ½ and 0.1. Neither adds up, and 1/p even gives a certain outcome a surprise of 1.','The logarithm passes all three: log 1/(p₁p₂) = log 1/p₁ + log 1/p₂. Only the log turns products into sums. So surprise is I(x) = log 1/p(x).'],
 ask:['Why should surprises add for independent days?','Which requirement does 1/p fail?','Is the log the only choice?'],
 expected:['Independent days share nothing, so the information should simply add, just as the probabilities multiply.','The first, and the third: it multiplies instead of adding.','Up to the base, yes: any continuous f with f(ab) = f(a) + f(b) is a logarithm. The base sets the unit.'],
 source:HUANG,transition:'Check it on five days.'});
S('surprise-five','kl','Five days: probabilities multiply, surprises add',{builds:2,nb:'cell-surprise',
 body:stack(fiveDays(false),at(2,fiveDays(true))),
 lines:[`Five sunny days: the probability multiplies to 0.9⁵ = ${num(inf.five_sun.p,2)}, and the surprise adds: 5 × 0.152 = ${num(inf.five_sun.bits,2)} bits.`,`Make one day rainy: the probability drops to ${num(inf.one_rain.p,3)} and the surprise jumps to ${num(inf.one_rain.bits,2)} bits. The single rainy day carries most of it.`],
 ask:['Is log₂ 1/0.59 equal to 5 × 0.152?','Which day is the most informative?'],
 expected:['Yes: both are 0.76 bits. The log turns the product into a sum.','The rainy one: 3.32 of the 3.93 bits.'],
 source:HUANG,transition:'The surprise curve, and its unit.'});
S('surprise-curve','kl','Surprise in bits: I(x) = log₂ 1/p(x)',{builds:2,nb:'cell-surprise',
 body:stack(widget('surprise','The surprise log base 2 of one over p, with a slider for p'),at(2,small('Base 2 gives bits; the natural log gives nats; 1 nat ≈ 1.44 bits.'))),
 lines:['A certain outcome gives 0 bits, and halving the probability adds exactly one bit. Drag p, or try the buttons.','One bit is the surprise of a fair coin flip: one yes/no answer you could not guess. Information theory counts in bits, PyTorch in nats.'],
 ask:['How many bits for p = ⅛?','What is one bit, in words?'],
 expected:['3 bits: ⅛ = ½ × ½ × ½, three halvings.','The surprise of a fair coin flip: one yes/no answer you could not predict.'],
 source:HUANG,transition:'One outcome is surprising or not. What about a whole distribution?'});
chap('2.2');
S('entropy-average','kl','Entropy: how surprised are you, on average?',{builds:3,nb:'cell-entropy',
 body:cols(forecastFig(true),stack(eq(R`\text{sunny: }\log_2\tfrac{1}{0.9}=0.15\ \text{bits}`),eq(R`\text{rainy: }\log_2\tfrac1{0.1}=3.32\ \text{bits}`),
  at(2,eq(R`H=0.9\,(0.15)+0.1\,(3.32)=${num(inf.entropy.forecast,2)}\ \text{bits}`)),
  at(3,eq(R`H(p)=\sum_x p(x)\log_2\frac1{p(x)}=\E_{x\sim p}\Big[\log_2\frac1{p(x)}\Big]`))),'0.85fr 1.25fr'),
 lines:['Each kind of weather has its surprise: 0.15 bits for sun, 3.32 for rain.','How surprised are you on a typical day? 90% of days give 0.15 bits and 10% give 3.32: on average 0.47 bits.','That weighted average of surprise is the entropy: the surprise you expect from one draw.'],
 ask:['Which outcome is more surprising? Which happens more often?','Why weight by 0.9 and 0.1?','Is entropy about one outcome, or the whole distribution?'],
 expected:['Rain is far more surprising; sun far more common.','They are how often each surprise happens. Next slide: the long-run average over days.','The whole distribution: one number for how uncertain it is.'],
 source:HUANG,transition:'Why an expectation? Let days go by.'});
S('entropy-days','kl','Why an expectation? It is the average over many days',{nb:'cell-entropy',
 body:widget('days-entropy','Days drawn from the forecast, each with its surprise, and the running average'),
 lines:['Let days go by, each with its surprise: 0.15 bits if sunny, 3.32 if rainy. The running average settles on 0.47 bits. An expectation is a long-run average.'],
 ask:['After 10 days, is the average close to 0.47?'],
 expected:['Only by luck: one rainy day in ten gives 0.47, two give 0.8. It settles as days accumulate, because sun comes on 90% of them and rain on 10%.'],
 point:'The weighted sum Σ p(x) f(x) is the average of f over infinitely many draws; with S draws the plain average estimates it (Section 2.9).',
 demo:'Notebook 2.2: the entropy cell simulates 10,000 days and prints the running average at 10, 100, 1,000 and 10,000 days.',
 transition:'How does entropy change with the forecast?'});
S('entropy-explorer','kl','From certain to fully uncertain: between 0 and 1 bit',{nb:'cell-entropy',
 body:widget('entropy2','Entropy of two kinds of weather as P(sunny) changes'),
 lines:['Drag the chance of sun. A 50/50 forecast carries 1 bit, the most two outcomes can; 87.5/12.5 carries 0.54; a certain forecast, 0.'],
 ask:['Where is the entropy largest, and why?'],
 expected:['At 50/50: the weather is least predictable, so each day surprises you most on average.'],
 source:HUANG,transition:'Three kinds of weather.'});
S('entropy-three','kl','Three kinds of weather: 1.5 bits',{builds:2,nb:'cell-entropy',
 body:cols(weatherBars([wb.p],['our town, p'],[palette.p]),stack(table(['weather',m('p'),m(R`\log_2\frac1p`)],wb.names.map((nm,i)=>[nm,half(wb.p[i]),`${num(bitsOf(wb.p[i]),0)} bit${bitsOf(wb.p[i])===1?'':'s'}`])),
  at(2,eq(R`H=\tfrac12(1)+\tfrac14(2)+\tfrac14(2)=1.5\ \text{bits}`))),'1fr 1.1fr'),
 lines:['Our town: sunny half the time, cloudy a quarter, rainy a quarter. Their surprises are 1, 2 and 2 bits.','Average surprise: ½(1) + ¼(2) + ¼(2) = 1.5 bits per day.'],
 ask:['Why is cloudy worth 2 bits?','Is 1.5 bits more or less than a fair coin?'],
 expected:['¼ = ½ × ½: two halvings, two bits.','More: three outcomes leave more to learn each day than two equally likely ones.'],
 source:HUANG,transition:'What does 1.5 bits mean in practice?'});
S('entropy-questions','kl','Entropy counts yes/no questions',{builds:3,nb:'cell-entropy',
 body:cols(questionTree(),stack(p('Guess today’s weather with yes/no questions, asking about the likeliest first.'),
  at(2,eq(R`\tfrac12(1)+\tfrac14(2)+\tfrac14(2)=1.5\ \text{questions}`)),
  at(3,p('Write each answer as a bit, yes = 0 and no = 1: sunny is <b>0</b>, cloudy <b>10</b>, rainy <b>11</b>. That is a code.'))),'1.05fr 1fr'),
 lines:['Find today’s weather with yes/no questions. Ask “Is it sunny?” first: half the time you are done.','Otherwise ask “Is it cloudy?”. On average 1.5 questions: exactly the entropy.','Each answer is one bit. Writing the answers down gives a code for the weather: Section 2.3.'],
 ask:['How many questions on a sunny day? On a rainy day?','Can any strategy average fewer than 1.5 questions?','What is the code for rainy?'],
 expected:['One; two.','No: entropy is the least possible average. This strategy reaches it because every probability is a power of ½.','11: no to “sunny?”, then no to “cloudy?”.'],
 point:'Each question here is 50/50 given the earlier answers, so each answer carries exactly 1 bit.',
 source:HUANG,transition:'Section 2.3: send the weather as bits.'});
chap('2.3');
S('channel','kl','Way 2: send the weather, in bits',{builds:2,
 body:stack(channelDiagram(),at(2,table(['','sunny','cloudy','rainy'],[[`how often, ${m(R`p`)}`,'½','¼','¼']]))),
 lines:['Turn the question game into messages: a weather station sends each day’s weather to a receiver as 0s and 1s. Every bit costs, so we want as few as possible.','Sunny on half the days, cloudy on a quarter, rainy on a quarter: p = (½, ¼, ¼).'],
 ask:['What does the receiver need to recover?','What do we want to minimise?'],
 expected:['Which kind of day it was, exactly.','The average number of bits sent per day.'],
 source:GERON,transition:'First, what is a code?'});
S('uniform-code','kl','A code: a string of bits for each kind of day',{builds:2,
 body:cols(table(['weather',m(R`p`),'uniform code','bits'],wb.names.map((name,i)=>[name,half(wb.p[i]),`<code>${wb.uniform_code[i]}</code>`,'2'])),
  stack(p('The simplest code gives every kind of day the same length: 2 bits.'),at(2,p('Fair, but wasteful: half the days are sunny, yet a sunny day costs as much as a rare one.'))),'1.25fr 1fr'),
 lines:['A code assigns each kind of day a string of bits. The uniform code spends 2 bits on every day.','Wasteful: common days cost as much as rare ones. Could common days get shorter strings?'],
 ask:['How many bits does the uniform code send per day?','Where is it wasteful?'],
 expected:['2, every day.','On the common sunny days.'],
 source:GERON,transition:'Shorter strings need care.'});
S('prefix-code','kl','Prefix codes: the bits split up without separators',{builds:2,
 body:stack(prefixDecode(true),at(2,prefixDecode(false))),
 lines:['Give sunny 0, cloudy 10, rainy 11. No codeword is the start of another, so the receiver reads left to right and each day pops out.','Break that rule, say 0, 01 and 1, and “01” could mean cloudy, or sunny then rainy. A prefix code avoids the confusion.'],
 ask:['Decode 0 10 11 0 10.','Why is 0, 01, 1 a bad code?'],
 expected:['Sunny, cloudy, rainy, sunny, cloudy.','0 is the start of 01, so the receiver cannot tell where one day ends.'],
 source:GERON,transition:'Which prefix code is the most efficient?'});
S('huffman','kl','Huffman’s recipe: merge the two rarest, repeat',{builds:4,nb:'cell-huffman',
 body:cols(huffmanSteps(),stack(p('Huffman (1952) builds the best prefix code from the probabilities:'),
  at(2,p('1. Merge the two least likely outcomes into one node.')),at(3,p('2. Repeat until one node is left.')),at(4,p('3. Read each codeword off its path: left 0, right 1.'))),'1.1fr 1fr'),
 lines:['Start with the three outcomes and their probabilities.','Merge the two rarest, cloudy and rainy, into one node: ¼ + ¼ = ½.','Merge again: sunny and the new node make the root.','Read the codes off the tree: sunny 0, cloudy 10, rainy 11. The question game’s code.'],
 ask:['Which two do we merge first?','What is the merged node’s probability?','How many merges for K outcomes?','Why does the likeliest outcome get the shortest codeword?'],
 expected:['The two least likely: cloudy and rainy.','½.','K − 1: each merge removes one node.','It is merged last, so it sits closest to the root.'],
 point:'The notebook implements Huffman’s algorithm in ten lines with a heap, and recovers 0, 10, 11 for p and 10, 11, 0 for q.',
 transition:'How good is this code?'});
S('efficient-code','kl','Each codeword is exactly as long as its surprise',{builds:2,nb:'cell-huffman',
 body:stack(table(['weather',m('p'),'code','length','surprise'],wb.names.map((nm,i)=>[nm,half(wb.p[i]),`<code>${wb.right_code[i]}</code>`,String(wb.right_code[i].length),`${num(bitsOf(wb.p[i]),0)} bit${bitsOf(wb.p[i])===1?'':'s'}`])),
  at(2,eq(R`\text{average length}=\tfrac12(1)+\tfrac14(2)+\tfrac14(2)=1.5\ \text{bits}=H(p)`)),
  at(2,small('Shannon (1948): no code averages fewer than H(p) bits. Huffman reaches it when every probability is a power of ½, and comes within 1 bit otherwise.'))),
 lines:['Compare the code lengths with the surprises: 1, 2, 2 and 1, 2, 2. The best code spends exactly the surprise on each outcome.','So the average message length is the average surprise, the entropy: 1.5 bits per day instead of 2.'],
 ask:['How long is the codeword for an outcome with probability ⅛?','Can any code average fewer than 1.5 bits here?'],
 expected:['3 bits: log₂ 8.','No: by Shannon’s source coding theorem, H(p) is the minimum.'],
 source:HUANG,transition:'What if the code was built for the wrong weather?'});
chap('2.4');
S('wrong-belief','kl','A code built on a wrong belief',{builds:2,nb:'cell-huffman',
 body:cols(weatherBars([wb.p,wb.q],['our weather, p','their belief, q'],[palette.p,palette.q]),stack(p('The station was designed for a town where rain is the common weather: q = (¼, ¼, ½).'),
  at(2,table(['weather',m('q'),'their code',m(R`\log_2\frac1q`)],wb.names.map((nm,i)=>[nm,half(wb.q[i]),`<code>${wb.wrong_code[i]}</code>`,`${num(bitsOf(wb.q[i]),0)} bit${bitsOf(wb.q[i])===1?'':'s'}`])))),'1.1fr 1fr'),
 lines:['Suppose the station’s code was built for another town, where rain is common: q = (¼, ¼, ½). Our weather is still p = (½, ¼, ¼).','Huffman on q gives rainy the short codeword: sunny 10, cloudy 11, rainy 0. Each length is log₂ 1/q: the surprise under the wrong belief.'],
 ask:['Which of our days does their code make expensive?','Whose probabilities set the code lengths?'],
 expected:['Sunny days: half of our days now cost 2 bits instead of 1.','q’s: a code built for q spends log₂ 1/q(x) bits on x.'],
 source:HUANG,transition:'Average the cost over our weather.'});
S('cross-entropy','kl','Cross-entropy: the average cost when the code expects q',{builds:3,nb:'cell-cross-entropy',
 body:stack(eq(R`H(\pcol p,\qcol q)=\sum_x \pcol{p(x)}\,\log_2\frac{1}{\qcol{q(x)}}`),
  at(2,eq(R`=\tfrac12(2)+\tfrac14(2)+\tfrac14(1)=1.75\ \text{bits per day}`)),
  at(3,cols(p(`<b class="p">weights</b> from the truth p: how often each day happens`),p(`<b class="q">lengths</b> from the belief q: what each day costs`),'1fr 1fr'))),
 lines:['Cross-entropy: average, over our real weather p, of the code lengths built from q.','½ × 2 + ¼ × 2 + ¼ × 1 = 1.75 bits per day, against 1.5 with the right code.','The two play different roles: p decides how often, q decides how much. Entropy is the special case q = p.'],
 ask:['Which distribution supplies the weights?','Compute it.','What is H(p, p)?'],
 expected:['p, the real weather.','½(2) + ¼(2) + ¼(1) = 1.75.','H(p): the right code, 1.5 bits.'],
 source:HUANG,transition:'Watch it happen, day by day.'});
S('bits-days','kl','Watch it happen: our weather through both codes',{nb:'cell-huffman',
 body:widget('days-codes','Days of our weather sent through the code for p and the code for q'),
 lines:['Send days of our weather through both codes. With 10, 100, 1,000 days the right code settles at 1.5 bits per day and the wrong one at 1.75: 0.25 bits per day wasted.'],
 ask:['After 1,000 days, how many bits did the wrong code waste?'],
 expected:['About 250, 0.25 per day. The notebook’s run: 1,747 bits against 1,498.'],
 demo:'Notebook 2.3: the Huffman cell sends 1,000 days through both codes and prints the bit counts and the start of each bit stream.',
 transition:'Name the wasted bits.'});
S('kl-extra-bits','kl','KL divergence: the extra bits we waste',{builds:3,nb:'cell-cross-entropy',
 body:stack(eq(R`\KL(\pcol p\,\|\,\qcol q)=H(\pcol p,\qcol q)-H(\pcol p)=1.75-1.5=0.25\ \text{bits}`),
  at(2,eq(R`=\sum_x p(x)\Big(\log_2\frac1{q(x)}-\log_2\frac1{p(x)}\Big)`)),
  at(3,eq(R`=\sum_x p(x)\log_2\frac{p(x)}{q(x)}=\tfrac12(1)+\tfrac14(0)+\tfrac14(-1)=0.25`))),
 lines:['The KL divergence is the waste: cross-entropy minus entropy, here 0.25 bits per day.','Both are sums over the same outcomes with the same weights p(x), so subtract term by term.','A difference of logs is the log of a ratio: KL(p‖q) = Σ p log p/q. For our weather ½(1) + ¼(0) + ¼(−1) = 0.25.'],
 ask:['What would the waste be if q = p?','Why can the two sums be combined?','Which outcome saves bits? Does the total still lose?'],
 expected:['Zero: the right code wastes nothing.','Same outcomes, same weights p(x).','Rainy: its codeword got shorter (−1 bit, on a quarter of days). Sunny loses a bit on half the days, so the total is +0.25.'],
 source:HUANG,transition:'Two facts about KL.'});
S('kl-properties','kl','Never negative, and zero only for the right code',{builds:2,nb:'cell-cross-entropy',
 body:stack(eq(R`H(p,q)\ \ge\ H(p)\quad\Longrightarrow\quad\KL(p\,\|\,q)\ \ge\ 0`),small('No code averages fewer than H(p) bits, so a code built for q can only waste bits.'),
  at(2,eq(R`\KL(p\,\|\,q)=0\quad\Longleftrightarrow\quad q=p`)),at(2,small('Single terms can be negative, as rainy’s was; the weighted total cannot. Part 4 proves KL ≥ 0 for any distributions in one line.'))),
 lines:['The right code is the best possible, so any other code can only cost more: KL is never negative.','And it is zero only when the belief matches the truth.'],
 ask:['Can KL be negative?','If KL(p‖q) = 0, what do we know?'],
 expected:['No: that would beat the best possible code.','q = p: the belief is right.'],
 transition:'Does it matter which one is the truth?'});
S('kl-asymmetry','kl','Which one is the truth? KL is not symmetric',{builds:2,nb:'cell-cross-entropy',
 body:cols(weatherBars([wb.p,inf.kl.asym.q],['our weather, p','a sunny town, q'],[palette.p,palette.q]),
  table(['','bits wasted per day'],[[`our weather, their code: ${m(R`\KL(p\|q)`)}`,num(inf.kl.asym.fwd,2)],[at(2,`their weather, our code: ${m(R`\KL(q\|p)`)}`),at(2,num(inf.kl.asym.rev,2))]]),'1fr 1fr'),
 lines:['A sunny town expects sun 98% of the time, so its code gives cloudy and rainy 6.6-bit codewords. On our weather, where they come half the time, that wastes 1.84 bits a day: KL(p‖q).','Swap the roles: our code on their weather wastes only 0.86 bits a day, KL(q‖p). The truth supplies the weights, so swapping changes the answer.'],
 ask:['Why is their code so costly for us?','Is KL a distance in the everyday sense? Which direction will VI use?'],
 expected:['It spends 6.6 bits on our cloudy and rainy days, which are half of all days.','No: a distance is the same both ways; KL is a divergence. VI uses KL(q ‖ posterior), q first; Section 3.6 shows what that does to a fitted q.'],
 point:`Sunny town q = (0.98, 0.01, 0.01): codeword lengths log₂ 1/q = ${inf.kl.asym.codeword_bits.map(v=>num(v,2)).join(', ')} bits.`,
 source:HUANG,transition:'Section 2.5: where you have already met all this.'});
chap('2.5');
S('classifier','kl','A classifier outputs a distribution',{builds:2,nb:'cell-classifier',
 body:classifier4(),
 lines:['An image goes into a classifier; out comes a distribution over four labels: ape 0.05, bear 0.05, cat 0.60, dog 0.30. Call it q.','The true label is a distribution too: all of its probability on cat. Call it p.'],
 ask:['What does the classifier output?','How can a label be a distribution?'],
 expected:['A probability for each label, from a softmax.','One-hot: probability 1 on the true class, 0 on the others.'],
 source:HUANG,transition:'How surprised is the model by the truth?'});
S('ce-surprise','kl','The loss is the model’s surprise at the true label',{builds:3,nb:'cell-classifier',
 body:stack(eq(R`H(\pcol p,\qcol q)=\sum_y \pcol{p(y)}\,\log\frac{1}{\qcol{q(y)}}`),
  at(2,eq(R`=0\cdot\log\tfrac{1}{0.05}+0\cdot\log\tfrac{1}{0.05}+1\cdot\log\tfrac1{0.6}+0\cdot\log\tfrac1{0.3}`)),
  at(3,eq(R`=-\log q(\text{cat})=${num(cl.loss_nats,2)}\ \text{nats}=${num(cl.loss_bits,2)}\ \text{bits}`))),
 lines:['The cross-entropy between the label p and the prediction q: the label supplies the weights.','Write out all four terms. Three are multiplied by zero.','What is left is −log q(cat): the model’s surprise at the true label. That is the loss you have been minimising all along.'],
 ask:['Which terms survive?','In words, what is the loss?','What loss would q(cat) = 1 give?'],
 expected:['Only the true class.','How surprised the model is by the right answer.','0: no surprise.'],
 source:HUANG,transition:'And where is the KL?'});
S('ce-is-kl','kl','With a one-hot label, cross-entropy is exactly KL',{builds:3,nb:'cell-classifier',
 body:stack(eq(R`\KL(p\,\|\,q)=H(p,q)-H(p)`),at(2,eq(R`H(p)=1\cdot\log\tfrac11+0+0+0=0`)),at(2,small('A one-hot label is certain, so it has no surprise (0 · log 1/0 counts as 0).')),
  at(3,`<div class="code-sm">${code('F.cross_entropy(logits, target)   # 0.511 nats: here equal to KL(p ‖ q)')}</div>`)),
 lines:['From Section 2.4: KL is cross-entropy minus entropy.','The entropy of a one-hot label is 0: a certain outcome carries no surprise.','So the classification loss is exactly KL(label ‖ prediction). PyTorch computes it in nats: 0.511 for our cat.'],
 ask:['What is H(p) for a one-hot label?','So what does training minimise?','Why does PyTorch compute cross-entropy rather than KL?'],
 expected:['0.','KL(label ‖ prediction): the bits the prediction wastes on the true labels.','They are equal here, and in general differ by H(p), which training cannot change. Cross-entropy avoids the log of the labels.'],
 demo:'Notebook 2.5: F.cross_entropy prints 0.511 nats, −log 0.6, the same number as KL(one-hot ‖ q).',
 source:HUANG,transition:'What the loss looks like as the prediction changes.'});
S('loss-curve','kl','The loss curve is the surprise curve',{builds:2,nb:'cell-classifier',
 body:cols(lossCurve(),stack(p('The loss is −ln q(true class): the surprise curve of Section 2.1, in nats.'),at(2,p('Confident and wrong is expensive: q(true) = 0.01 costs 4.6 nats; q(true) = 0.6 costs 0.51.'))),'1.2fr 1fr'),
 lines:['Plot the loss against the probability the model gives to the true class: it is the surprise curve again.','Training pushes q(true) up the curve towards 1. Being confidently wrong is punished hardest.'],
 ask:['Where is the curve steepest?','Why might a model learn to be confident?'],
 expected:['Near 0: almost-impossible true labels cost the most.','Each step towards q(true) = 1 lowers the loss; with no other pressure it heads there.'],
 transition:'Soft labels.'});
S('soft-labels','kl','Soft labels: cross-entropy = entropy + KL',{builds:2,nb:'cell-classifier',
 body:cols(labelPair(cl.teacher),stack(eq(R`H(p,q)=\underbrace{H(p)}_{${num(cl.entropy_soft,2)}}+\underbrace{\KL(p\,\|\,q)}_{${num(cl.kl_soft,2)}}=${num(cl.ce_soft,2)}`),
  at(2,p('In knowledge distillation a small student learns from a big teacher’s predictions. H(p) is fixed by the teacher, so lowering the cross-entropy lowers the KL by exactly as much.'))),'1fr 1.15fr'),
 lines:['A soft label: a teacher network says cat 0.75, dog 0.20, bear 0.03, ape 0.02. Now H(p) is not 0.','Training changes only the student q, so minimising cross-entropy is still minimising KL(teacher ‖ student).'],
 ask:['Which part can training change?','Can the cross-entropy reach 0 here?'],
 expected:['Only the KL; H(p) is fixed by the teacher.',`No: its minimum is H(p) = ${num(cl.entropy_soft,2)} nats, reached when the student copies the teacher.`],
 transition:'Section 2.6: real data.'});
chap('2.6');
S('mle-data','kl','The data: twenty days of weather',{builds:2,nb:'cell-mle',
 body:stack(dataGrid(),at(2,p('Counts: 10 sunny, 5 cloudy, 5 rainy. Frequencies p̂ = (½, ¼, ¼).'))),
 lines:['Now real data: twenty recorded days. We want a model q for the weather. Which q explains these days best?','Count them: 10 sunny, 5 cloudy, 5 rainy. Their frequencies are p̂ = (½, ¼, ¼).'],
 ask:['What would you guess for tomorrow?','What should “explains the data best” mean?'],
 expected:['Sunny with probability ½: the frequency.','Gives the data the highest probability: maximum likelihood. Let us see why that is the frequency.'],
 transition:'How probable are these twenty days under a model q?'});
S('mle-likelihood','kl','How probable are these twenty days under q?',{builds:3,nb:'cell-mle',
 body:stack(eq(R`p(\mathcal D\mid q)=\prod_{i=1}^{20}q(x_i)=q(\text{sunny})^{10}\,q(\text{cloudy})^{5}\,q(\text{rainy})^{5}`),
  at(2,eq(R`q=(\tfrac14,\tfrac14,\tfrac12):\quad(\tfrac14)^{10}(\tfrac14)^{5}(\tfrac12)^{5}=2^{-35}\approx2.9\times10^{-11}`)),
  at(3,eq(R`\log_2 p(\mathcal D\mid q)=-35`))),
 lines:['Under the model the days are independent, so the probability of the record multiplies: one factor per day. Equal days give powers.','Take the wrong belief q = (¼, ¼, ½): the record has probability 2⁻³⁵, about 3 in a hundred billion.','Tiny numbers are why we take logs: log₂ p(𝒟 | q) = −35. The whole record is 35 bits of surprise.'],
 ask:['Why a product?','Why is it so small?','Why work with the log?'],
 expected:['Independent days: probabilities multiply.','Any one particular sequence of 20 days is unlikely, even under the right model.','Products of many small numbers underflow; the log turns them into a sum.'],
 transition:'Per day, that is a familiar number.'});
S('mle-surprise','kl','Per day: the data’s average surprise under q',{builds:3,nb:'cell-mle',
 body:stack(eq(R`-\frac1{20}\log_2 p(\mathcal D\mid q)=\frac1{20}\sum_{i=1}^{20}\log_2\frac1{q(x_i)}=\frac{35}{20}=1.75\ \text{bits}`),
  at(2,p('Have we seen 1.75 before?')),
  at(3,`<p class="result" style="font-size:38px">It is the cross-entropy H(p, q) of Section 2.4.</p>`)),
 lines:['Divide by the number of days: minus the log-likelihood per day is the average surprise of a recorded day under q. Here 1.75 bits.','Section 2.4 had a 1.75: the average code length when the code expects q but the weather is p.','Same number: the data’s average surprise under q is a cross-entropy. Why?'],
 ask:['What is minus the log-likelihood per day, in words?','Where have we seen 1.75?','Coincidence?'],
 expected:['The average surprise of a recorded day under q.','H(p, q): the wrong code on our weather.','No. Next slide: group the days by weather.'],
 transition:'Group the days by weather.'});
S('mle-group','kl','Why: group the days by weather',{builds:3,nb:'cell-mle',
 body:stack(eq(R`\frac1{20}\sum_{i=1}^{20}\log_2\frac1{q(x_i)}=\frac1{20}\Big[\,10\log_2\tfrac1{q(\text{sunny})}+5\log_2\tfrac1{q(\text{cloudy})}+5\log_2\tfrac1{q(\text{rainy})}\Big]`),
  at(2,eq(R`=\tfrac{10}{20}\log_2\tfrac1{q(\text{sunny})}+\tfrac5{20}\log_2\tfrac1{q(\text{cloudy})}+\tfrac5{20}\log_2\tfrac1{q(\text{rainy})}`)),
  at(3,eq(R`=\sum_x\hat p(x)\log_2\frac1{q(x)}=H(\hat p,\,q)`))),
 lines:['Sort the twenty terms by weather: ten equal sunny terms, five cloudy, five rainy. Equal terms add up to count × term.','Divide by 20: the counts become the frequencies, 10/20, 5/20, 5/20.','Those frequencies are p̂. The data’s average surprise is the cross-entropy H(p̂, q): the data became a distribution.'],
 ask:['How many sunny terms are there?','What do 10/20, 5/20 and 5/20 form?','So what is maximum likelihood minimising?'],
 expected:['Ten, all equal to log₂ 1/q(sunny).','The data’s frequencies p̂ = (½, ¼, ¼): the empirical distribution.','The cross-entropy H(p̂, q).'],
 transition:'Split it, as in 2.4.'});
S('mle-split','kl','The data’s surprise = entropy of the data + KL',{builds:2,nb:'cell-mle',
 body:stack(eq(R`H(\hat p,q)=\underbrace{H(\hat p)}_{\text{fixed by the data}}+\underbrace{\KL(\hat p\,\|\,q)}_{\text{depends on }q}`),
  at(2,eq(R`1.75=1.5+0.25\qquad\text{for }q=(\tfrac14,\tfrac14,\tfrac12)`))),
 lines:['Section 2.4’s split: cross-entropy = entropy + KL. The entropy of the data is fixed; only the KL depends on the model.','For q = (¼, ¼, ½): 1.75 = 1.5 + 0.25. The model can only remove the KL part.'],
 ask:['Which part can the model change?','What is the smallest possible average surprise?'],
 expected:['Only KL(p̂ ‖ q).','H(p̂) = 1.5 bits, when KL = 0, that is, q = p̂.'],
 transition:'Find the best q yourself.'});
S('mle-fit','kl','Find the q that makes the data least surprising',{nb:'cell-mle-fit',
 body:widget('mlefit','Move q and watch the data’s average surprise split into entropy and KL'),
 lines:['Move q. The teal part, H(p̂), never changes; the red part, the KL, shrinks to zero exactly at q = p̂ = (½, ¼, ¼). Maximum likelihood matches the frequencies.'],
 ask:['Start from (⅓, ⅓, ⅓). Which way should q(sunny) move?'],
 expected:['Up, to ½: the data have more sun than q expects. At q = p̂ the data are 32 times more probable than under (¼, ¼, ½).'],
 demo:'Notebook 2.6: the second cell runs gradient descent on the data’s average surprise from equal logits and lands on (0.5, 0.25, 0.25) without being told.',
 transition:'The general statement.'});
S('mle-kl','kl','Maximum likelihood minimises KL(data ‖ model)',{builds:3,nb:'cell-mle',
 body:stack(eq(R`\max_\theta\ \frac1N\sum_{i=1}^N\log q_\theta(x_i)\iff\min_\theta\ H(\hat p,q_\theta)\iff\min_\theta\ \KL(\hat p\,\|\,q_\theta)`),
  at(2,p('Any model, any data: the average log-likelihood is minus the data’s cross-entropy, and the data’s entropy is fixed.')),
  at(3,p('A classifier’s training set does the same, label by label: each example costs −log q<sub>θ</sub>(y<sub>i</sub> | x<sub>i</sub>), the surprise at its label.'))),
 lines:['Nothing in the derivation was special to weather: maximising the likelihood minimises KL(data ‖ model), with the data first.','With more data p̂ approaches the true distribution of the data, so the model is pulled towards it in this KL.','The classification loss of 2.5 is the same story, averaged over a training set: training a classifier is maximum likelihood.'],
 ask:['Which distribution comes first in this KL?','What happens with more data?','What is a training set’s cross-entropy loss?'],
 expected:['The data’s, p̂.','p̂ gets closer to the true distribution of the data.','The average surprise at the true labels: minus the average log-likelihood.'],
 transition:'Collect the pattern.'});
chap('2.7');
S('kl-everywhere','kl','KL at work: the same quantity in many losses',{nb:'sec-2-7',
 body:`<div class="table-sm">${table(['training …','minimises','first argument','where'],[['a classifier with cross-entropy','KL(labels ‖ predictions)','the labels','2.5 ✓'],['a model by maximum likelihood','KL(data ‖ model)','the data','2.6 ✓'],['a small student from a big teacher','KL(teacher ‖ student)','the teacher','next'],['a chatbot with RLHF','reward − β KL(policy ‖ reference)','the policy','after that'],['an approximation by VI','KL(q ‖ posterior)','our approximation q','Parts 3–5']])}</div>`,
 lines:['We have met two uses already: classification and maximum likelihood. Two more, step by step: distilling a big model into a small one, and keeping a chatbot honest in RLHF. VI is the rest of the lecture.'],
 ask:['Which distribution comes first in each row?'],
 expected:['The one that supplies the weights: labels, data, teacher, policy, q. Section 3.6 shows why the order matters for VI.'],
 point:'The video opens with distillation, generative models and RLHF as motivation.',
 source:HUANG,transition:'Distillation: a big teacher and a small student.'});
S('distill-why','kl','Knowledge distillation: a big teacher, a small student',{builds:2,nb:'cell-distill',
 body:`<div class="state-slot"><div data-only="1">${distillDiagram(1)}</div><div data-only="2">${distillDiagram(2)}</div></div>`,
 lines:['A big network, the teacher, is accurate but too slow and too expensive to run on a phone, or a million times a day.','We want a small network, the student, that behaves like it. Give both the same image: the teacher says cat 0.75, dog 0.20; the untrained student is vaguer.'],
 ask:['Why not just deploy the teacher?','How could the small model learn to behave like the big one?'],
 expected:['Cost and speed: memory, latency and energy grow with the number of parameters.','Copy its outputs: make the student’s distribution over labels match the teacher’s, input by input.'],
 point:'Hinton, Vinyals and Dean (2015), “Distilling the Knowledge in a Neural Network”. Smaller deployed models, such as DistilBERT, are trained this way.',
 transition:'Make the two distributions the same.'});
S('distill-match','kl','Make the student’s distribution match the teacher’s',{builds:2,nb:'cell-distill',
 body:`<div class="state-slot"><div data-only="1">${distillDiagram(3)}</div><div data-only="2">${distillDiagram(4)}</div></div>`,
 lines:['For each input, measure how far the student’s distribution is from the teacher’s: KL(teacher ‖ student). Average it over many inputs.','Minimise it. The teacher is frozen; gradients flow only into the student. At KL = 0 the student answers like the teacher on every input, doubts included.'],
 ask:['Why does the teacher come first in the KL?','Does the student need the true labels?'],
 expected:['The teacher plays the truth: its distribution supplies the weights, exactly like the labels in 2.5.','Not for this term: the teacher labels every input, even unlabelled ones. In practice a small label term is often added.'],
 transition:'The loss, term by term.'});
S('distill-loss','kl','The distillation loss, term by term',{builds:3,nb:'cell-distill',
 body:stack(eq(R`\min_{\text{student}}\ \frac1N\sum_{i=1}^{N}\KL\big(p_T(\cdot\mid x_i)\,\|\,q_S(\cdot\mid x_i)\big),\qquad\KL(p_T\,\|\,q_S)=H(p_T,q_S)-H(p_T)`),
  at(2,eq(R`\text{teacher frozen}\ \Rightarrow\ H(p_T)\ \text{fixed}\ \Rightarrow\ \text{minimise }H(p_T,q_S)=-\sum_y p_T(y)\log q_S(y)`)),
  at(3,eq(R`\mathcal L=(1-\alpha)\,H(y_{\text{true}},q_S)+\alpha\,\tau^2\,\KL\big(p_T^{(\tau)}\,\|\,q_S^{(\tau)}\big)`)),
  at(3,small('p<sup>(τ)</sup> = softmax(logits / τ): a temperature τ > 1 softens both distributions; τ² keeps the gradient scale comparable. α blends in the true labels.'))),
 lines:['Write the KL out: it is the cross-entropy with the teacher’s soft labels, minus the teacher’s entropy.','The teacher is frozen, so its entropy is a constant: minimising the KL is minimising cross-entropy with soft labels, the soft-label slide of 2.5.','In practice both distributions are softened with a temperature τ, and a small term with the true labels is added. For our cat the KL is 0.05 nats.'],
 ask:['Which part can the student change?','How is this related to the ordinary classification loss?','What does a temperature τ > 1 do?'],
 expected:['Only q_S: H(p_T) is fixed by the frozen teacher.','It is the same cross-entropy loss, with the teacher’s soft distribution in place of the one-hot label.','It flattens the teacher’s distribution so its small probabilities, which carry the extra information, count for more.'],
 transition:'What do soft labels teach that a one-hot label cannot?'});
S('distill-soft','kl','What soft labels teach: a cat looks a bit like a dog',{builds:2,nb:'cell-classifier',
 body:stack(softTemps(),at(2,p('A one-hot label says only “cat”. The teacher also says “a bit like a dog, not like an ape”: information about how classes relate, which Hinton called dark knowledge. Softening with τ = 4 makes it visible.'))),
 lines:['Compare what the student is shown. The label: all on cat. The teacher: mostly cat, some dog, almost no ape or bear.','The teacher’s small probabilities tell the student which mistakes are reasonable. A temperature τ = 4 turns them up, so they shape the loss.'],
 ask:['What does the student learn from the 0.20 on dog?','Why would τ = 1 hide that information?'],
 expected:['That cats and dogs look alike to the teacher, so confusing them is a small error, unlike cat versus ape.','The 0.02 and 0.03 barely change the loss; dividing the logits by τ evens the distribution out.'],
 transition:'Watch a real student learn.'});
S('distill-demo','kl','Press play: a tiny student learns a big teacher',{builds:2,nb:'cell-distill',
 body:stack(widget('distill','A student network learning a teacher’s distribution, with the KL from teacher to student'),
  at(2,small(`Test accuracy: teacher ${num(100*ds.teacher_acc,1)}%, student ${num(100*ds.student_acc,1)}%; the student agrees with the teacher on ${num(100*ds.agree,1)}% of test points, with ${Math.round(ds.teacher_params/ds.student_params)}× fewer parameters.`))),
 lines:[`Three overlapping classes in the plane. The teacher has ${ds.teacher_params.toLocaleString('en')} parameters, the student ${ds.student_params}. Colours show each network’s class probabilities. Predict: can the student copy the teacher?`,`Press play. The student’s map turns into the teacher’s, uncertain boundaries included, and KL(teacher ‖ student) falls from ${num(ds.frames[0][1],2)} to ${num(ds.frames.at(-1)[1],3)} nats.`],
 ask:[`Predict: can a ${ds.student_params}-parameter network copy a ${ds.teacher_params.toLocaleString('en')}-parameter one?`,'Why do the boundaries stay blurred?'],
 expected:['On this problem, nearly: the classes need only gently curved boundaries.','The student copies the teacher’s uncertainty too: where classes overlap, the teacher is unsure, and so is the student.'],
 demo:'Notebook 2.7: the distillation cell trains the teacher on labels, then the student on the teacher’s probabilities, and prints both accuracies, the agreement and the KL.',
 transition:'Second application: RLHF.'});
S('rlhf-setup','kl','RLHF: reward a chatbot, but keep it close to where it started',{builds:3,nb:'cell-rlhf',
 body:`<div class="state-slot"><div data-only="1">${rlhfDiagram(1)}</div><div data-only="2">${rlhfDiagram(2)}</div><div data-only="3">${rlhfDiagram(3)}</div></div>`,
 lines:['Reinforcement learning from human feedback: the language model, called the policy, writes a reply; a reward model, trained on human preferences, scores it. Training raises the reward.','A frozen copy of the starting model, the reference, stays alongside. A KL penalty keeps the policy’s next-word distributions close to the reference’s.',`The objective: expected reward minus β times KL(${PIH} ‖ ${PIH}<sub>ref</sub>). β sets how long the leash is.`],
 ask:['What does the reward model reward?','Why keep a frozen copy?','What does β control?'],
 expected:['Replies people preferred, as learned from human comparisons.','To measure how far the policy has drifted from a model that writes fluent, sensible text.','The trade-off: small β chases reward, large β stays near the reference.'],
 point:'This KL penalty is part of InstructGPT/ChatGPT-style RLHF (PPO) and of GRPO; DPO optimises the same KL-regularised objective without a separate reward model.',
 transition:'What goes wrong without the leash?'});
S('rlhf-hacking','kl','Without the KL: the policy fools the reward model',{builds:2,nb:'cell-rlhf',
 body:stack(`<div class="table-sm">${table(['reply to “Is the stove hot?”',m(R`\pi_{\text{ref}}`),'reward'],rl.replies.map((rep,i)=>[rep,num(rl.ref[i],2),(rl.reward[i]>0?'+':'')+num(rl.reward[i],1)]))}</div>`,
  at(2,p(`Maximise reward alone and everything goes to “${rl.replies.at(-1)}”: <b>reward hacking</b>.`))),
 lines:['Five possible replies. The reference model, trained on human text, mostly says sensible things. The reward model is imperfect: it loves enthusiasm.',`Chase the reward with no leash and the policy puts everything on the worst reply. The KL term charges for moving away from the reference: the less likely a reply was under ${PIH}<sub>ref</sub>, the more it costs.`],
 ask:['Which reply does the reward model prefer?','Why is the KL term a defence?'],
 expected:['The shouting one, +2.0: a flaw in the learned reward.',`A reply the reference almost never writes has a large log ratio ${PIH}/${PIH}<sub>ref</sub>, so piling probability on it costs a lot of KL.`],
 transition:'Slide the leash.'});
S('rlhf-beta','kl','Slide the leash: the best policy for each β',{builds:2,nb:'cell-rlhf',
 body:stack(widget('rlhfbeta','The best policy over five replies as beta changes'),at(2,small(`The best policy reweights the reference by the reward: ${PIH}*(y) ∝ ${PIH}<sub>ref</sub>(y) · e<sup>r(y)/β</sup>.`))),
 lines:['Move β. With a tight leash (large β) the policy stays the reference. Loosen it and probability flows to high-reward replies; with no leash it all goes to the hacked one.','The best policy has a closed form: reweight the reference by e<sup>r/β</sup>. KL keeps it a reweighting of what the model already says, never something new.'],
 ask:['At β = 1, which reply gains most?','What happens as β → 0? As β → ∞?'],
 expected:['“Yes, it is hot. Careful!”: probable under the reference and well rewarded.',`β → 0: all mass on the highest reward, the hacked reply. β → ∞: ${PIH}* = ${PIH}<sub>ref</sub>, no change.`],
 point:`Derivation: maximise Σ ${PIH} r − β Σ ${PIH} log(${PIH}/${PIH}<sub>ref</sub>) subject to Σ ${PIH} = 1; setting the derivative to zero gives log ${PIH} = log ${PIH}<sub>ref</sub> + r/β − const. The notebook checks it by gradient ascent.`,
 demo:`Notebook 2.7: the RLHF cell prints ${PIH}* for eight values of β and checks the closed form at β = 1 by gradient ascent.`,
 transition:'How is that KL computed for real text?'});
S('rlhf-tokens','kl','In practice: the KL is estimated token by token, from samples',{builds:2,nb:'cell-rlhf',
 body:stack(eq(R`\KL(\pi\,\|\,\pi_{\text{ref}})\approx\sum_{t}\Big[\log\pi(y_t\mid y_{<t})-\log\pi_{\text{ref}}(y_t\mid y_{<t})\Big],\qquad y\sim\pi`),
  at(2,p('A reply is a sequence of tokens, and there are far too many sequences to sum over. So sample replies from the policy and add the log ratios of the tokens: a Monte Carlo estimate (Section 2.9). GRPO uses the never-negative estimator r − 1 − log r from the appendix.'))),
 lines:[`For text, ${PIH} is a next-token distribution: the KL of a reply is the sum, over its tokens, of log ${PIH} − log ${PIH}<sub>ref</sub>.`,'There is no way to sum over all possible replies, so the KL is estimated from sampled replies, exactly the sampling idea of Section 2.9.'],
 ask:['Why not compute the KL exactly?','Which estimator can never be negative?'],
 expected:['Vocabulary^length possible replies: astronomically many.',`r − 1 − log r with r = ${PIH}<sub>ref</sub>/${PIH}: optional slide in the appendix; DeepSeek’s GRPO uses it.`],
 transition:'Back to continuous distributions: closed forms.'});
chap('2.8');
S('densities','kl','From sums to integrals',{builds:2,
 body:stack(eq(R`\text{discrete:}\qquad\KL(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}`),eq(R`\text{continuous:}\qquad\KL(p\|q)=\int p(\theta)\log\frac{p(\theta)}{q(\theta)}\,d\theta`),
  at(2,small('Probabilities become densities and the sum becomes an integral; the recipe is the same weighted average of log ratios.'))),
 lines:['For a continuous unknown like θ, the sum over grades becomes an integral over θ.','Nothing else changes. KL is never negative either; Part 4 proves it, where we need it.'],
 ask:['What plays the role of the weights now?','Can a density be bigger than 1?'],
 expected:['The density p(θ), times dθ.','Yes; but the KL is still a weighted average of log ratios, and is still never negative.'],
 transition:'Closed forms for the distributions we will use.'});
const TORCH_KL='https://github.com/pytorch/pytorch/blob/main/torch/distributions/kl.py';
const bk=[.8*Math.log(1.6),.2*Math.log(.4)];
S('bern-pmf','kl','Bernoulli: a coin with P(x = 1) = a',{builds:2,
 body:cols(bernBars(),stack(eq(R`p(x)=a^{x}(1-a)^{1-x},\qquad x\in\{0,1\}`),eq(R`p(1)=a,\qquad p(0)=1-a`),at(2,p(`Your turn: write ${m(R`\KL(p\,\|\,q)`)} for p = Bernoulli(a) and q = Bernoulli(b).`))),'1.05fr 1fr'),
 lines:['Bernoulli(a): x is 1 with probability a and 0 otherwise. Here p = Bernoulli(0.8), q = Bernoulli(0.5).','Your turn: put the PMF into Σₓ p(x) log p(x)/q(x). There are only two values of x.'],
 ask:['What is p(0) for Bernoulli(0.8)?','Write the KL.'],
 expected:['0.2.','Two terms: one for x = 1 and one for x = 0. Next slide.'],
 transition:'The answer, step by step.'});
S('bern-kl','kl','KL between two Bernoullis, step by step',{nb:'cell-closed-forms',builds:4,
 body:stack(eq(R`\KL(p\,\|\,q)=\sum_{x\in\{0,1\}}p(x)\log\frac{p(x)}{q(x)}`),at(2,eq(R`=p(1)\log\frac{p(1)}{q(1)}+p(0)\log\frac{p(0)}{q(0)}`)),
  at(3,eq(R`=a\log\frac{a}{b}+(1-a)\log\frac{1-a}{1-b}`)),at(4,eq(R`a=0.8,\ b=0.5:\quad0.8\log1.6+0.2\log0.4=${num(bk[0],3)}${num(bk[1],3)}=${num(bk[0]+bk[1],2)}`))),
 lines:['Start from the definition: a sum over the values of x.','There are two: write out the term for x = 1 and the term for x = 0.','Substitute p(1) = a, p(0) = 1 − a, and b for q.','For a = 0.8 and b = 0.5: 0.38 − 0.18 = 0.19 nats.'],
 ask:['How many terms does the sum have?','Which term is negative here, and why?','What is the KL if a = b?','Is 0.19 the same as KL(q‖p)?'],
 expected:['Two, one per value of x.','The x = 0 term: p(0) = 0.2 is smaller than q(0) = 0.5, so its log ratio is negative.','0: both log ratios are 0.','No: KL(Bernoulli(0.5) ‖ Bernoulli(0.8)) = 0.22.'],
 transition:'More than two outcomes.'});
S('cat-pmf','kl','Categorical: K outcomes with probabilities π₁, …, π<sub>K</sub>',{builds:2,
 body:cols(gradeBars(gr.q,gr.p),stack(eq(R`p(x=k)=\pi_k,\quad k=1,\dots,K`),eq(R`\pi_k\ge0,\qquad\textstyle\sum_k\pi_k=1`),small('Our grades: K = 3, with π = (0.4, 0.1, 0.5) this year and ρ = (0.1, 0.4, 0.5) last year.'),
  at(2,p(`Your turn: write ${m(R`\KL(p\,\|\,q)`)} for p = Categorical(π) and q = Categorical(ρ).`))),'1.3fr 1fr'),
 lines:['Categorical(π): one of K outcomes, outcome k with probability πₖ. Our grade distributions are categorical, with K = 3.','Your turn: the same recipe, now with K terms.'],
 ask:['What is π for this year’s grades?','Write the KL.'],
 expected:['(0.4, 0.1, 0.5).','A sum of K terms, πₖ log(πₖ/ρₖ). Next slide.'],
 transition:'The answer, step by step.'});
S('cat-kl','kl','KL between two categoricals, step by step',{nb:'cell-closed-forms',builds:3,
 body:stack(eq(R`\KL(p\,\|\,q)=\sum_{k=1}^{K}p(k)\log\frac{p(k)}{q(k)}=\sum_{k=1}^{K}\pi_k\log\frac{\pi_k}{\rho_k}`),
  at(2,eq(R`K=2:\quad\pi=(a,\,1-a),\ \rho=(b,\,1-b)\ \Rightarrow\ a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`)),
  at(3,eq(R`\text{grades: }0.4\log4+0.1\log\tfrac14+0.5\log1=${num(gr.kl,2)}`))),
 lines:['Substitute p(k) = πₖ and q(k) = ρₖ: one term per outcome.','With K = 2 this is exactly the Bernoulli formula: a Bernoulli is a categorical with two outcomes.','For the grades: 0.42 nats, the number we built by hand in Way 1.'],
 ask:['How many terms for K outcomes?','Why does K = 2 give the Bernoulli formula?','Which term is largest for the grades?'],
 expected:['K.','A Bernoulli is a categorical over {1, 0} with π = (a, 1 − a).','A: 0.4 log 4 ≈ 0.55.'],
 transition:'Continuous: the Gaussian.'});
S('gauss-pdf','kl','The Gaussian: a bell with mean μ and width σ',{builds:2,
 body:cols(normalPlot(['q1']),stack(eq(R`\N(\theta;\mu,\sigma^2)=\tfrac{1}{\sqrt{2\pi\sigma^2}}\,e^{-(\theta-\mu)^2/(2\sigma^2)}`),small('q = N(μ, σ²) is our approximation and p = N(m, s²) the pretend posterior. Shown: q = N(0, 2²), p = N(1, 4²).'),
  at(2,p(`Your turn: work out ${m(R`\KL(q\,\|\,p)`)}. Start with log q(θ) − log p(θ).`))),'1.2fr 1fr'),
 lines:['The Gaussian density: a normalising constant times e to the minus squared distance from the mean, scaled by the variance.','Your turn: work out KL(q ‖ p). The log makes the exponentials easy.'],
 ask:['What does σ control?','What is log N(θ; μ, σ²)?'],
 expected:['The width: about 68% of the mass lies within σ of μ.','−½ log(2πσ²) − (θ − μ)²/(2σ²).'],
 transition:'Step 1: the log ratio.'});
S('gauss-log-ratio','kl','Gaussian KL, step 1: the log ratio',{builds:3,
 body:stack(eq(R`\log q(\theta)=-\tfrac12\log(2\pi\sigma^2)-\frac{(\theta-\mu)^2}{2\sigma^2}`),at(2,eq(R`\log p(\theta)=-\tfrac12\log(2\pi s^2)-\frac{(\theta-m)^2}{2s^2}`)),
  at(3,eq(R`\log q(\theta)-\log p(\theta)=\log\frac{s}{\sigma}-\frac{(\theta-\mu)^2}{2\sigma^2}+\frac{(\theta-m)^2}{2s^2}`))),
 lines:['Take the log of q’s density: the exponential disappears.','The same for p, with its own mean m and width s.','Subtract. The 2π terms cancel, and the two log widths combine into log(s/σ).'],
 ask:['Why take logs first?','Where does log(s/σ) come from?','Which terms still depend on θ?'],
 expected:['The densities are exponentials; logs turn them into simple squares.','−½ log(2πσ²) + ½ log(2πs²) = log s − log σ.','The two squared terms.'],
 transition:'Step 2: average over q.'});
S('gauss-expectation','kl','Gaussian KL, step 2: average over q',{nb:'cell-closed-forms',builds:4,
 body:stack(eq(R`\KL(q\|p)=\E_q\Big[\log\frac{s}{\sigma}-\frac{(\theta-\mu)^2}{2\sigma^2}+\frac{(\theta-m)^2}{2s^2}\Big]`),
  at(2,eq(R`\E_q\big[(\theta-\mu)^2\big]=\sigma^2\quad\Rightarrow\quad-\frac{\sigma^2}{2\sigma^2}=-\frac12`)),
  at(3,eq(R`\E_q\big[(\theta-m)^2\big]=\E_q\big[\big((\theta-\mu)+(\mu-m)\big)^2\big]=\sigma^2+(\mu-m)^2`)),
  at(4,eq(R`\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12`))),
 lines:['Average the log ratio over θ drawn from q. The first term is a constant, so it stays as it is.','The second: the average squared distance from q’s own mean is its variance, σ². That term becomes −½.','The third: split θ − m into (θ − μ) + (μ − m). The cross term averages to zero, leaving σ² + (μ − m)².','Together: a width term, the squared shift between the means, and −½. That is the Gaussian KL.'],
 ask:['What is the average of a constant?','What is E_q[(θ − μ)²] called?','Why does the cross term vanish?','When is the KL zero?'],
 expected:['The constant itself.','The variance of q, σ².','It is 2(μ − m)·E_q[θ − μ], and E_q[θ − μ] = 0.','Only when μ = m and σ = s.'],
 transition:'Many dimensions at once.'});
S('diag-pdf','kl','A diagonal Gaussian: one bell per coordinate',{builds:2,
 body:cols(diagEllipses(),stack(eq(R`q(\boldsymbol\theta)=\prod_{j=1}^{d}\N(\theta_j;\,\mu_j,\,\sigma_j^2)`),eq(R`p(\boldsymbol\theta)=\prod_{j=1}^{d}\N(\theta_j;\,m_j,\,s_j^2)`),
  small('Independent coordinates, so the contours are axis-aligned. Our network’s q is exactly this, with d = 7.'),at(2,p('Your turn: what is KL(q ‖ p) in d dimensions?'))),'1fr 1.15fr'),
 lines:['A diagonal Gaussian is a product of one-dimensional Gaussians: each coordinate has its own mean and width.','Your turn: what happens to the log of a product?'],
 ask:['Why are the contours axis-aligned?','What is the log of a product?'],
 expected:['No correlation between coordinates: the covariance is diagonal.','The sum of the logs.'],
 transition:'The answer, step by step.'});
S('diag-kl','kl','KL between diagonal Gaussians, step by step',{nb:'cell-closed-forms',builds:3,
 body:stack(eq(R`\log\frac{q(\boldsymbol\theta)}{p(\boldsymbol\theta)}=\sum_{j=1}^{d}\log\frac{\N(\theta_j;\mu_j,\sigma_j^2)}{\N(\theta_j;m_j,s_j^2)}`),
  at(2,eq(R`\KL(q\|p)=\sum_{j=1}^{d}\KL\big(\N(\mu_j,\sigma_j^2)\,\|\,\N(m_j,s_j^2)\big)`)),
  at(3,eq(R`p=\N(0,I):\qquad\KL(q\|p)=\sum_{j=1}^{d}\Big[-\log\sigma_j+\frac{\sigma_j^2+\mu_j^2}{2}-\frac12\Big]`))),
 lines:['The log of a product is a sum of logs: the log ratio splits into one term per coordinate.','Each term involves only θⱼ, so its average under q is a one-dimensional Gaussian KL. The total is their sum.','Against a N(0, 1) prior on every coordinate: exactly the KL term in our network’s ELBO, and in every VAE.'],
 ask:['Why can each term be averaged separately?','How many one-dimensional KLs for our network?','What does each term reward?'],
 expected:['Each term depends on θⱼ alone, and q makes the coordinates independent.','Seven, one per parameter.','μⱼ near 0 and σⱼ near 1: staying close to the prior.'],
 transition:'Is this really what PyTorch computes?'});
S('torch-discrete','kl','The same formulas in PyTorch: Bernoulli and categorical',{nb:'cell-closed-forms',builds:2,
 body:stack(`<div class="code-sm">${code('@register_kl(Bernoulli, Bernoulli)\ndef _kl_bernoulli_bernoulli(p, q):\n    t1 = p.probs * (softplus(-q.logits) - softplus(-p.logits))\n    t2 = (1 - p.probs) * (softplus(q.logits) - softplus(p.logits))\n    return t1 + t2\n\n@register_kl(Categorical, Categorical)\ndef _kl_categorical_categorical(p, q):\n    t = p.probs * (p.logits - q.logits)\n    return t.sum(-1)')}</div>`,
  at(2,small(`softplus(−logit) = −log(probability), so t1 = a log(a/b) and t2 = (1 − a) log((1 − a)/(1 − b)). For a categorical, logits are log-probabilities: Σ πₖ (log πₖ − log ρₖ).`))),
 lines:['From PyTorch’s source, torch/distributions/kl.py: the Bernoulli and categorical KLs.','Read them against our formulas: a log(a/b) + (1 − a) log((1 − a)/(1 − b)), and Σₖ πₖ log(πₖ/ρₖ). The same.'],
 ask:['Where is a log(a/b) in the Bernoulli code?','Why does PyTorch use logits and softplus?'],
 expected:['t1: p.probs is a, and softplus(−q.logits) − softplus(−p.logits) = log a − log b.','Numerical stability: no log of probabilities that round to 0.'],
 point:'Shown as in PyTorch 2.13, with torch.nn.functional.softplus shortened to softplus and the lines handling probabilities of exactly 0 or 1 left out.',
 source:`PyTorch source: <a href="${TORCH_KL}">torch/distributions/kl.py</a>`,transition:'And the Gaussian.'});
S('torch-normal','kl','The same formulas in PyTorch: Gaussian and diagonal',{nb:'cell-closed-forms',demo:'Switch to notebook 2.4 and run the closed-forms cell: it checks all four formulas against kl_divergence and prints PyTorch’s own source for the Normal and Independent cases.',builds:2,
 body:stack(`<div class="code-sm">${code('@register_kl(Normal, Normal)\ndef _kl_normal_normal(p, q):\n    var_ratio = (p.scale / q.scale).pow(2)\n    t1 = ((p.loc - q.loc) / q.scale).pow(2)\n    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())\n\n@register_kl(Independent, Independent)\ndef _kl_independent_independent(p, q):\n    result = kl_divergence(p.base_dist, q.base_dist)\n    return _sum_rightmost(result, p.reinterpreted_batch_ndims)')}</div>`,
  at(2,small('½(σ²/s² + (μ − m)²/s² − 1 − log σ²/s²) = log(s/σ) + (σ² + (μ − m)²)/(2s²) − ½. Independent: the one-dimensional KLs, summed over coordinates.'))),
 lines:['PyTorch’s Gaussian KL, and the rule for independent coordinates.','Expand the Normal line: it is our formula. Independent sums the per-coordinate KLs: the diagonal-Gaussian rule.'],
 ask:['What is var_ratio in our notation?','Which line is the diagonal-Gaussian sum?'],
 expected:['σ²/s², with p the first argument (our q) and q the second (our p).','_sum_rightmost: it adds the one-dimensional KLs over the coordinates.'],
 point:'In PyTorch’s code p and q are just the first and second arguments; with our VI order, its p is our q. A NotImplementedError check is left out.',
 source:`PyTorch source: <a href="${TORCH_KL}">torch/distributions/kl.py</a>`,transition:'Use the formula: which Gaussian is closer?'});
S('compare-formula','kl','Use the formula: which Gaussian is closer to p?',{nb:'cell-compare',builds:2,
 body:cols(normalPlot(['q1','q2']),stack(eq(R`\pcol{p=\N(1,4^2)}`),table(['','substitute','KL'],[
  [`${m(R`q_1=\N(0,2^2)`)}`,m(R`\log2+\frac{4+1}{32}-\frac12`),at(2,num(n.candidates.q1.kl))],
  [`${m(R`q_2=\N(1,3^2)`)}`,m(R`\log\frac43+\frac{9+0}{32}-\frac12`),at(2,num(n.candidates.q2.kl))]])),'1.25fr 1fr'),
 lines:['The PML vote: which is closer to p, q₁ or q₂? Substitute into the formula.','KL(q₁‖p) = 0.35 nats and KL(q₂‖p) = 0.07. The formula settles the vote.'],
 ask:['Vote first: q₁ or q₂?','Which term made q₁ worse?'],
 expected:['Most pick q₂: same centre and a closer width.','Both: q₁ is off-centre, (0 − 1)² = 1, and narrower, log 2 against log 4/3.'],
 point:'PAUSE (PML L23 08:30).',
 transition:'Where does each number come from? Watch it as an area.'});
S('kl-area','kl','KL is an area you can watch',{nb:'cell-kl-area',
 body:widget('klarea','KL as the net area under its integrand, with sliders for q'),
 lines:['Purple area adds and red area subtracts; the net area is the KL. Drag q, or flip the direction.'],
 ask:['Make q narrow and centred on p. What happens in each direction?'],
 expected:['KL(q‖p) stays moderate: q only visits places where p is reasonable. KL(p‖q) explodes: p’s tails land where q is nearly zero.'],
 point:'Click q₁, then q₂, then flip to KL(p‖q). Then shrink σ in each direction.',
 transition:'What if there is no formula at all?'});
chap('2.9');
S('kl-expectation','kl','KL is an average: an expectation under p',{builds:2,
 body:stack(eq(R`\KL(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}`),at(2,eq(R`=\E_{x\sim p}\Big[\log\frac{p(x)}{q(x)}\Big]`)),
  at(2,small('E with x ~ p means: the average value when x is drawn from p.'))),
 lines:['Look again at the formula: each log ratio is weighted by p(x), and the weights add up to 1.','So KL is an expectation: the average log ratio when x is drawn from p.'],
 ask:['What do the weights p(x) add up to?','Which distribution do we draw x from?'],
 expected:['1: it is a weighted average.','p, the first argument.'],
 transition:'Two ways to compute an expectation.'});
S('expectation-sampling','kl','Two ways to compute an expectation',{nb:'cell-grades-sampling',builds:3,
 body:stack(eq(R`\E_{x\sim p}[f(x)]=\sum_x p(x)\,f(x)`),at(2,eq(R`\E_{\theta\sim p}[f(\theta)]=\int p(\theta)\,f(\theta)\,d\theta`)),
  at(3,eq(R`\E_{x\sim p}[f(x)]\approx\frac1S\sum_{s=1}^{S}f(x_s),\qquad x_s\sim p`)),
  at(3,small('Draws land on each x in proportion to p(x), so a plain average over draws weights each f(x) by p(x) automatically.'))),
 lines:['For a discrete x, an expectation is a weighted sum: each value f(x) times its probability.','For a continuous θ, the sum becomes an integral.','Or draw samples from p and take a plain average. Values that are more probable show up more often, so the weighting happens by itself.'],
 ask:['What is the weighted sum for our weather, with f = log₂ p/q?','When would you need the integral?','Why does a plain average over samples work?'],
 expected:['½(1) + ¼(0) + ¼(−1) = 0.25 bits.','When θ is continuous, like a network weight.','About half the draws are sunny, a quarter cloudy, a quarter rainy, so the plain average approaches the weighted sum.'],
 transition:'Try it with the weather.'});
S('mc-weather','kl','Sample days and watch the average find the KL',{nb:'cell-weather-sampling',
 body:widget('days-logratio','Days drawn from our weather and the running average of log₂ p/q'),
 lines:['Draw days from our weather p and average log₂ p/q: +1 bit on sunny days, 0 on cloudy, −1 on rainy. The average settles on the KL, 0.25 bits, with no formula needed.'],
 ask:['After 10 days, is the estimate close?'],
 expected:['Not necessarily: with few days the shares are off. It settles as they approach ½, ¼, ¼. This is a Monte Carlo estimate of KL.'],
 point:'The bits-per-day demo of 2.4 was already a Monte Carlo estimate: the wasted bits per day, averaged over simulated days.',
 demo:'Notebook 2.8: the weather-sampling cell draws 10,000 days and prints the running average at 10, 100, 1,000 and 10,000 days.',
 transition:'Back to our Gaussians, with the formula and with samples.'});
S('kl-sampling','kl','No formula needed: estimate KL by sampling',{nb:'cell-kl-sampling',demo:'Switch to notebook 2.5, “The code on the slide, run”: run it, then set S to 100 and to 1,000,000 and rerun. Come back for the live version.',builds:2,
 body:stack(eq(R`\KL(q\|p)=\E_{\theta\sim q}\big[\log q(\theta)-\log p(\theta)\big]\approx\frac1S\sum_{i=1}^{S}\big[\log q(\theta_i)-\log p(\theta_i)\big]`),
  cols(code('theta = q.sample((S,))\nlog_q = q.log_prob(theta)\nlog_p = p.log_prob(theta)\nest = (log_q - log_p).mean()\nexact = kl_divergence(q, p)'),
  at(2,table(['','KL(q₂‖p)'],[['exact formula',num(mc.exact_kl,3)],['estimate, S = 10,000',num(mc.repeats[3].estimates[0],3)]])),'1fr 1fr')),
 lines:['Draw θ from q, score each draw with both log densities, and average. No formula needed.','Checked on a pair that has a formula: 0.067 from 10,000 draws, against the exact 0.069.'],
 ask:['Which distribution supplies the draws?','Should the estimate match exactly?'],
 expected:['q, the first argument: it does the weighting.','No: it is random. It is close, and gets closer with more draws.'],
 point:'q₂ = N(1, 3²) against p = N(1, 4²), seed 1000.',
 transition:'Watch the estimate settle as draws arrive.'});
S('mc-live','kl','Watch the estimate settle',{nb:'cell-kl-sampling',
 body:widget('mclive','A Monte Carlo estimate of KL, built up one batch of draws at a time'),
 lines:['Add draws from q and watch the running average of log q − log p settle on the exact 0.349.'],
 ask:['Press +10 a few times. Is every estimate above the exact value?'],
 expected:['No: with few draws the estimate lands on either side, and can even be negative. The spread shrinks like 1/√S.'],
 point:'q = N(0, 2²) and p = N(1, 4²), the q₁ pair. Draws are seeded, so the sequence replays identically.',
 transition:'Part 3: use the score to find the best q.'});


// ================================================================ Part 3 · Find the closest q
chap('3.1');
S('six-candidates','fit','Six candidates: which q is closest to p?',{builds:2,nb:'cell-grid',
 body:act(3,sixCandidates()),
 lines:['Six Gaussians against the same target p = N(1, 4²). Vote: which is closest?',`Smallest KL: N(1, 3²) at ${num(candKL[1])}, then N(2, 5²) at ${num(candKL[5])}. Could another q do better?`],
 ask:['Vote: which q is closest?','Could another (μ, σ) do better?'],
 expected:['Most pick N(1, 3²) or N(2, 5²). The formula says N(1, 3²), narrowly.','Yes: N(1, 4²), which is p itself, has KL 0. We need a way to search.'],
 point:'PAUSE. All six KLs come from the closed-form Gaussian formula of 2.4.',
 transition:'Search systematically: a grid.'});
S('grid-search','fit','Search a grid of means and widths: hover over it',{builds:2,nb:'cell-grid',
 body:widget('klgrid','KL over a grid of means and widths; hover to see each q against p'),
 lines:['Every point of the grid is one q = N(μ, σ²), coloured by its KL to p. Hover over it: the right panel shows that q against p, and its score.',`The lightest point is q = p: 2,040 evaluations. For our network’s fourteen numbers a grid needs 40¹⁴ ≈ 3 × 10²². We need to walk downhill instead.`],
 ask:['Hover near the bottom right. Why is it so dark?','How many evaluations for fourteen parameters?'],
 expected:['Narrow and off-centre: q puts all its mass where p is small, and KL(q‖p) punishes exactly that.','40¹⁴ ≈ 2.7 × 10²², even with only 40 values per parameter.'],
 demo:'Notebook 3.1: the grid cell prints the six candidates’ KLs and draws the same heatmap with matplotlib.',
 transition:'Instead of evaluating everywhere, walk downhill.'});
chap('3.2');
S('walk-downhill','fit','Instead of a grid, walk downhill',{builds:2,nb:'cell-fit-code',
 body:cols(klGrid({bestAt:1,pathAt:2}),stack(eq(R`\psi^*=\arg\min_\psi\KL(q_\psi\,\|\,p)`),eq(R`\psi\leftarrow\psi-\eta\,\nabla_\psi\KL(q_\psi\,\|\,p)`),
  at(2,p('Gradient descent from N(4, 0.5²): the teal path. About a hundred steps, and the same recipe works for millions of parameters.'))),'1.05fr 1fr'),
 lines:['Cast the search as optimisation: follow the gradient of KL with respect to q’s parameters.','From N(4, 0.5²) the optimiser walks downhill to p. It needs only gradients, which autograd provides.'],
 ask:['What does the gradient tell us at each point?','Why does this scale where the grid did not?'],
 expected:['The direction in (μ, σ) that lowers KL fastest.','Each step costs one gradient evaluation, whatever the number of parameters.'],
 point:'The path is the recorded Adam run from the next two slides, drawn on the same grid.',
 transition:'The same search in code.'});
S('fit-code','fit','The same search in code',{builds:2,nb:'cell-fit-code',
 body:cols(code('mu  = Parameter(tensor(4.0))\nrho = Parameter(tensor(log(0.5)))\nopt = Adam([mu, rho], lr=0.05)\nfor step in range(200):\n    q = Normal(mu, rho.exp())\n    loss = kl_divergence(q, p)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()'),
  stack(eq(R`\sigma=\exp(\rho)>0\ \text{ for every }\rho`),at(2,p('Rebuild q, score it, differentiate, update. The notebook runs exactly this loop and plots the loss, μ, σ and the learned curve.'))),'1.25fr 1fr'),
 lines:['Store ρ = log σ, so that σ = exp(ρ) stays positive whatever an update does.','Rebuild q, score it, differentiate, update. The same five lines fit our network in Part 5.'],
 ask:['What stays fixed, and what does the optimiser change?','Why not update σ directly?'],
 expected:['p stays fixed; μ and ρ change.','A step could make σ zero or negative, which is not a valid Normal.'],
 demo:'Switch to notebook 3.2, “The code on the slide, run”. Run it: the printed table shows the loss, μ and σ every 25 steps, and the three panels show the loss falling, μ and σ arriving at 1 and 4, and the learned curve landing on p. Drag the step slider in the next cell, then come back for the animation.',
 transition:'Press play.'});
S('olive','fit','Press play: the loss falls as q slides onto p',{builds:2,nb:'cell-fit-slider',
 body:widget('olive','q moving onto p under gradient descent, with the loss'),
 lines:['Start q far away, at N(4, 0.5²). Predict: which reaches the target first, the centre or the width?','The width gets there first and overshoots a little; the centre slides in by step 100. Then the loss, KL(q‖p), is about 0. Drag the slider to replay any step.'],
 ask:['Predict: does the centre or the width arrive first?','Why can both move at once?'],
 expected:[`Students split. The width reaches 4 around step 45, overshoots to ${num(Math.max(...st.olive.frames.map(r=>r[2])))}, then settles; μ arrives by step 100.`,'Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.'],
 point:'PAUSE (PML L23 12:13). Adam, learning rate 0.05, analytic KL. The faint curves are q at earlier steps.',
 transition:'One dimension is easy. What changes with two correlated unknowns?'});
chap('3.3');
S('cov-families','fit','In two dimensions, q needs a covariance matrix: three families',{builds:3,
 body:stack(covFamilies(),at(3,small('Our network has d = 7 parameters: 1, 7 or 28 numbers. A network with a million weights: 1, a million, or 5 × 10¹¹.'))),
 lines:['Isotropic: one width for every direction, so the contours are circles. Σ = s²I is a single number.','Diagonal: its own width per coordinate but no correlation, so the ellipses are axis-aligned. This is mean-field: q(θ) = q(θ₁) q(θ₂).','Full: the ellipse can tilt. Writing Σ = LLᵀ with L lower-triangular keeps Σ a valid covariance for any L. It costs d(d + 1)/2 numbers.'],
 ask:['How many numbers describe an isotropic Σ?','What can a diagonal q not express?','Why parameterise Σ = LLᵀ instead of Σ directly?'],
 expected:['One, s², whatever the dimension.','Correlation: knowing θ₁ tells it nothing about θ₂.','Any lower-triangular L with a positive diagonal gives a valid covariance; a step on Σ’s entries directly could break that, like a negative σ in 1-D.'],
 transition:'Fit all three to the same tilted target.'});
S('cov-fit','fit','Fit all three to the same tilted target',{builds:2,nb:'cell-cov3',
 body:widget('cov3','Isotropic, diagonal and full-covariance q fitted to one correlated target'),
 lines:['The target p (teal) has widths 1.5 and 0.75 and correlation 0.8. All three q start from the same random Gaussian, tilted the wrong way. Predict where each ends.',`Press play. The full q matches p exactly: KL → 0. The diagonal q cannot tilt; it shrinks to widths ${num(c3.analytic.diag_sd[0])} and ${num(c3.analytic.diag_sd[1])}. The isotropic q shrinks to a circle of radius ${num(c3.analytic.iso_sd)}.`],
 ask:['Predict: which q can reach KL = 0?','Watch Σ under each panel. Which entries can each family change?'],
 expected:['Only the full one: p is tilted, and only the full family can tilt.','Isotropic: one number on both diagonal entries. Diagonal: the two diagonal entries. Full: all three, including the off-diagonal one.'],
 point:`Same random start (seed 33), Adam, learning rate 0.05, 600 steps of the closed-form KL. Final KL: isotropic ${num(c3.analytic.iso_kl,3)}, diagonal ${num(c3.analytic.diag_kl,3)}, full 0.`,
 demo:'Notebook 3.3: the covariance cell runs all three fits, prints each final Σ and checks it against the formulas on the next slide.',
 transition:'Read off the scores.'});
S('two-d','fit','A q that cannot tilt shrinks instead',{builds:2,nb:'cell-cov3',
 body:stack(table(['q family','widths it finds','KL(q‖p)'],[
  ['target p','1.50 and 0.75, tilted','—'],
  ['isotropic',`${num(c3.analytic.iso_sd)} and ${num(c3.analytic.iso_sd)}`,num(c3.analytic.iso_kl)],
  ['diagonal (mean-field)',`${num(c3.analytic.diag_sd[0])} and ${num(c3.analytic.diag_sd[1])}`,num(c3.analytic.diag_kl)],
  ['full','1.50 and 0.75, tilted','0']]),
  at(2,p('KL(q‖p) charges heavily for q-mass where p is small, off the tilted ridge. So a q that cannot tilt shrinks: it underestimates the uncertainty. Our network’s q is diagonal; Part 5 shows what that does to its error bars.'))),
 lines:['The full q is exact. The diagonal q is narrower than p in both coordinates; the isotropic q is narrower still in the wide direction.','Rather than spill mass off the ridge, a q that cannot tilt shrinks. Mean-field VI underestimates uncertainty.'],
 ask:['Why are the diagonal widths 0.90 and 0.45, not 1.5 and 0.75?','What should we expect of mean-field VI on a real posterior?'],
 expected:['They are p’s conditional widths, σᵢ√(1 − ρ²) = 1.5 × 0.6 and 0.75 × 0.6: how wide p is along one axis when the other coordinate is held fixed.','Error bars that are too narrow wherever the posterior correlates its parameters.'],
 point:'Under KL(q‖p) the best diagonal q has σᵢ² = 1/(Σₚ⁻¹)ᵢᵢ and the best isotropic q has s² = d / tr(Σₚ⁻¹). The notebook checks both against the fitted values.',
 transition:'Now a target with no formula at all.'});
chap('3.4');
S('mixture-what','fit','A mixture of Gaussians: first pick a bump, then draw from it',{builds:3,
 body:cols(mixtureStory(),stack(mixtureChart(),at(3,eq(R`p(\theta)=\sum_{k}\pi_k\,\N(\theta;\mu_k,\sigma_k^2)`))),'1fr 1.1fr'),
 lines:['A mixture of Gaussians makes θ in two steps. First pick a bump, like tossing a fair coin: z = 1 or 2. Then draw θ from that bump’s Gaussian.','Each bump contributes its own bell, scaled by how often it is picked: ½ N(−0.2, 0.4²) and ½ N(1, 0.5²).','We see θ but not the bump that made it, so the density adds both bells: p(θ) = Σₖ πₖ N(θ; μₖ, σₖ²). Two bumps, one distribution.'],
 ask:['If bump 1 is picked, where does θ land?','Why multiply each bell by ½?','Does p(θ) integrate to 1?'],
 expected:['Near −0.2: within about ±0.8, two widths.','Each bump makes only half of the draws, so it contributes half of its bell.','Yes: each bell integrates to 1 and the weights add to 1.'],
 point:'The PML 2023 mixture. z is a latent variable: part of how the data are made, never observed. Lecture 3’s VAE has the same structure, with a continuous z.',
 transition:'Draw from it yourself.'});
S('mixture-sample','fit','Draw from the mixture yourself',{nb:'cell-mixture-sampling',
 body:widget('mixsample','Two-step draws from a mixture of two Gaussians, building up a histogram'),
 lines:['Each draw first picks a bump (its colour), then draws θ from it. Add 10, 100, then 1,000 draws and watch the histogram fill in the teal density.'],
 ask:['After 1,000 draws, what fraction came from each bump?'],
 expected:['About half each, and the stacked histogram follows the teal curve: the weighted sum of the two bells.'],
 point:'Seeded, so the same draws replay. log p(θ) is easy to evaluate: the log of a weighted sum of two Gaussian densities.',
 demo:'Notebook 3.4: the sampling cell does the same two steps with Categorical and Normal, 2,000 draws, and colours the histogram by bump.',
 transition:'How far is a Gaussian q from this mixture?'});
S('mixture-no-formula','fit','PyTorch has no formula for this KL. The definition still works.',{builds:2,nb:'cell-mixture',
 body:stack(cols(mixChart(),stack(code('kl_divergence(q, mixture)'),at(2,code('NotImplementedError:\nNo KL(p || q) is implemented\nfor p type Normal and\nq type MixtureSameFamily','text'))),'1fr 1fr')),
 lines:['How far is a Gaussian q from this mixture? How would you compute KL(q ‖ p)? Then ask PyTorch.','No formula exists for a Gaussian against a mixture. But KL(q‖p) is an average under q, and averages can be estimated by sampling, as in 2.5.'],
 ask:['How would you compute KL(q‖p) for this p?','Is the KL undefined, then?'],
 expected:['Sample θ from q and average log q(θ) − log p(θ). log p(θ) of a mixture is easy to evaluate.','No. It is a perfectly good integral; we just cannot write it in closed form.'],
 point:`Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch ${d.provenance.torch}).`,
 transition:'Estimate the average by sampling, and optimise it.'});
S('now-optimise','fit','Now optimise it: the code runs, and q does not move',{builds:2,nb:'cell-sample-fit',
 body:stack(cols(code('theta = q.sample((5000,))\nlog_q = q.log_prob(theta)\nlog_p = p.log_prob(theta)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-sample','A fit that uses sample(): q stays where it started'),'1fr 1.06fr')),
 lines:['Use the sampled KL as the loss and run the same loop on the mixture. Predict: does q move?',`No error, no warning, and q stays at N(6, 0.1²): after 400 steps the KL is still about ${Math.round(st.mixfit.sample.at(-1)[4])}. Is this a bug? Hold that question: Section 3.5 answers it.`],
 ask:['Predict: where is q after 400 steps?','Is this a bug?'],
 expected:['Most expect it to slide onto the mixture. It does not move.','Not in the usual sense: every line does what it says. Something is missing from the gradient, and 3.5 finds out what.'],
 point:'PAUSE (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.',
 demo:'Notebook 3.4, “Optimise it with sample()”: run it; no error, and q ends where it started.',
 transition:'Section 3.5: draw the computation as a graph.'});
chap('3.5');
S('missing-path','fit','Why q does not move: the gradient stops at a random node',{builds:2,nb:'cell-missing-path',
 body:stack(compGraph('sample'),at(2,`<div class="table-sm">${table(['',m(R`\partial\KL/\partial\mu`),m(R`\partial\KL/\partial\sigma`)],[
  ['what autograd gets through sample()',num(g.detached_sample_grads[0],4),num(g.detached_sample_grads[1],4)],
  ['the true gradient',num(g.analytic_grads[0],3),num(g.analytic_grads[1],3)]])}</div>`)),
 lines:['Boxes are deterministic nodes, computed from their inputs; circles are stochastic, drawn at random. With sample(), θ is stochastic, and the backward pass (red) cannot get from θ back to μ and σ.','Only the path through log q(θ)’s formula survives, and it averages to zero: 0.0013 and −0.0003 against −0.063 and −0.375. Not a bug: a missing path in the graph.'],
 ask:['Which arrow can the gradient not travel along?','Why isn’t the surviving gradient exactly zero?'],
 expected:['From θ back to μ and σ: sample() returns plain numbers, with no record of how they depend on μ and σ.','log q(θ) contains μ and σ explicitly. That part averages to zero over draws; what remains is noise.'],
 point:'Gaussian pair q = N(0, 2²), p = N(1, 4²), 50,000 shared draws. A score-function estimator is another valid fix (optional slide).',
 demo:'Notebook 3.5: the first cell shows draw.backward() failing on a sample() draw, then the tiny surviving gradient against the true one.',
 source:'Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>, which follows Kingma’s NIPS 2015 talk',
 transition:'Give the gradient a path.'});
S('reparam','fit','The fix: move the randomness into ε, and θ = μ + σε',{builds:3,nb:'cell-reparam',
 body:stack(compGraph('reparam'),cols(eq(R`\ncol{\epsilon}\sim\N(0,1),\qquad\theta=\mu+\sigma\ncol{\epsilon}`),at(3,eq(R`\theta=-1.5,\quad\frac{\partial \theta}{\partial\mu}=1,\quad\frac{\partial \theta}{\partial\sigma}=\epsilon=-0.75`)),'0.85fr 1.15fr')),
 lines:['Draw the noise ε from a fixed N(0, 1), then shift and scale it. θ has the same distribution, N(μ, σ²), but it is now a deterministic node: a function of μ, σ and ε.','Pause: with ε = −0.75, μ = 0 and σ = 2, find θ, ∂θ/∂μ and ∂θ/∂σ.','θ = −1.5, ∂θ/∂μ = 1 and ∂θ/∂σ = ε = −0.75. The only random node, ε, has no parameters, so nothing blocks the path. In PyTorch this is rsample().'],
 ask:['Is θ still distributed as N(μ, σ²)?','Compute θ and both derivatives.','Can ∂θ/∂σ be negative?'],
 expected:['Yes: shifting and scaling a standard Normal gives N(μ, σ²).','θ = 0 + 2(−0.75) = −1.5; ∂θ/∂μ = 1; ∂θ/∂σ = ε = −0.75.','Yes, whenever ε < 0: widening q pushes this draw further left.'],
 point:'PAUSE (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.',
 source:'Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>',
 transition:'Picture what a fixed ε does.'});
S('reparam-picture','fit','What a fixed ε does when μ or σ changes',{builds:2,nb:'cell-reparam',
 body:stack(reparamPicture(),at(2,eq(R`\frac{\partial \theta}{\partial\mu}=1,\qquad\frac{\partial \theta}{\partial\sigma}=\epsilon=-0.75`))),
 lines:['Keep the same noise draw, ε = −0.75. Shift μ by 1: the draw moves right by exactly 1.','Widen σ by 1: the draw moves by ε = −0.75, further out on its side. These are the derivatives the gradient needs.'],
 ask:['If μ rises by 1, how far does θ move?','If σ rises by 1?'],
 expected:['By exactly 1.','By ε = −0.75: to the left, because this ε is negative.'],
 transition:'The two graphs side by side.'});
S('reparam-compare','fit','Side by side: where the randomness lives',{builds:2,
 body:stack(compGraph('sample',{compact:true}),at(2,compGraph('reparam',{compact:true})),graphLegend()),
 lines:['With sample(), the random node sits between the parameters and the loss, and the gradient cannot cross it.','With rsample(), the randomness enters as a parameter-free input ε and θ = μ + σε is deterministic: the gradient flows from the loss to μ and σ. Same distribution, different graph.'],
 ask:['Where is the stochastic node in each graph?','What does the reparameterisation trick change: the distribution of θ, or the graph?'],
 expected:['Top: θ itself. Bottom: ε, which has no parameters.','Only the graph. θ is N(μ, σ²) either way.'],
 point:'This is the reparameterisation trick of Kingma and Welling (2014), the step that makes VAEs trainable (Lecture 3).',
 source:'Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>',
 transition:'Change one word in the code.'});
S('rsample-fit','fit','Change one word: rsample() fits the mixture',{builds:2,nb:'cell-rsample-fit',
 body:cols(code('# one word changed\ntheta = q.rsample((5000,))\nlog_q = q.log_prob(theta)\nlog_p = p.log_prob(theta)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-rsample','A fit that uses rsample(): q moves onto the mixture'),'1fr 1.06fr'),
 lines:['Same loop, same start, same seed, with q.rsample. Press play.',`q settles at N(${num(st.mixfit.rsample.at(-1)[1])}, ${num(st.mixfit.rsample.at(-1)[2])}²), KL = ${num(st.mixfit.rsample.at(-1)[4],3)}: the best single Gaussian for this mixture. No formula was needed.`],
 ask:['Predict where q settles.','Why doesn’t the KL reach 0?'],
 expected:['Over the mixture’s mass, between and around the two bumps.','No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct numerical integral.'],
 point:'Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Sections 2.5 and 3.5.',
 demo:'Notebook 3.5, “Change one word”: run it and compare the two fits in the figure.',
 transition:'Section 3.6: which direction of KL?'});
chap('3.6');
S('fwdrev','fit','The price: reverse KL picks one bump',{builds:2,nb:'cell-fwdrev',
 body:widget('fwdrev','Forward and reverse KL fits of one Gaussian to two bumps'),
 lines:['Fit one Gaussian to two narrow bumps, three times. Predict where each q settles.','KL(p‖q) spreads over both bumps. KL(q‖p) locks onto whichever bump is nearer its start, and reports it confidently.'],
 ask:['Predict each final q.','Which fit would make you overconfident?'],
 expected:['Forward: wide, centred between the bumps. Reverse: one bump, depending on the start.','The reverse fits: each reports one bump with a narrow spread.'],
 point:`PAUSE. Narrow bumps: forward KL ${num(st.fwdrev['0.5'].fwd.at(-1)[3])}; reverse ${num(st.fwdrev['0.5'].revl.at(-1)[3])} (= −log 0.7) from the left and ${num(st.fwdrev['0.5'].revr.at(-1)[3])} (= −log 0.3) from the right. Then switch to wide bumps: overlapping bumps give one broad reverse-KL fit.`,
 transition:'Part 4: back to Bayes.'});

// ================================================================ Part 4 · Back to Bayes
chap('4.1');
S('can-we','elbo','Back to Bayes: minimise KL to the posterior directly?',{builds:2,
 body:act(4,eq(R`\KL\big(q_\psi(\theta)\,\|\,\pcol{p(\theta\mid\mathcal D)}\big)=\E_{\theta\sim q}\big[\log q_\psi(\theta)-\log\pcol{p(\theta\mid\mathcal D)}\big]`),
  at(2,table(['the loss needs','available?'],[['draws θ ~ q','yes: rsample, as in 3.5'],[m(R`\log q_\psi(\theta)`),'yes'],[m(R`\log p(\theta\mid\mathcal D)`),'no: it needs p(𝒟), the integral of 1.4']]))),
 lines:['The plan from Part 1: make q close to the posterior. Use KL(q ‖ posterior): an average under q, which we can sample.','Everything is available except log p(θ | 𝒟), which needs the evidence p(𝒟): the integral we cannot do. Four small steps get rid of it.'],
 ask:['Why this direction, KL(q‖p), and not KL(p‖q)?','Which piece is missing?'],
 expected:['KL(p‖q) averages under the posterior, so it needs posterior samples: the problem itself. KL(q‖p) averages under q, which we choose.','log p(θ | 𝒟): it contains log p(𝒟).'],
 source:'A careful written derivation: <a href="https://www.gregorygundersen.com/blog/2021/04/16/variational-inference/">Gregory Gundersen, “The ELBO in Variational Inference”</a>',
 transition:'Step 1: Bayes’ rule, in logs.'});
S('elbo-step1','elbo','Step 1: Bayes’ rule, in logs',{builds:3,
 body:stack(eq(R`p(\theta\mid\mathcal D)=\frac{p(\mathcal D\mid\theta)\,p(\theta)}{\dcol{p(\mathcal D)}}`),
  at(2,eq(R`\log p(\theta\mid\mathcal D)=\log p(\mathcal D\mid\theta)+\log p(\theta)-\dcol{\log p(\mathcal D)}`)),
  at(2,small('Rule: log(a · b / c) = log a + log b − log c.')),
  at(3,p(`Look at the last term: ${m(R`\dcol{\log p(\mathcal D)}`)} has no θ in it. Whatever θ is, it is the same number, fixed by our model and our data.`))),
 lines:['Bayes’ rule from Part 1: the posterior is likelihood times prior, divided by the evidence.','Take logs. A product becomes a sum, and a division a subtraction.','The last term does not contain θ: it is one fixed number. Remember that; it does the work in step 3.'],
 ask:['What is log of a product?','Which term does not depend on θ?','Is log p(𝒟) known?'],
 expected:['The sum of the logs.','log p(𝒟): it was integrated over every θ.','No: it is the intractable integral. But it is a constant, and that is enough.'],
 transition:'Step 2: put this into the KL.'});
S('elbo-step2','elbo','Step 2: substitute into the KL',{builds:3,
 body:stack(eq(R`\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)=\E_q\big[\log q(\theta)-\log p(\theta\mid\mathcal D)\big]`),
  at(2,eq(R`=\E_q\Big[\log q(\theta)-\big(\log p(\mathcal D\mid\theta)+\log p(\theta)-\dcol{\log p(\mathcal D)}\big)\Big]`)),
  at(3,eq(R`=\E_q\big[\log q(\theta)-\log p(\mathcal D\mid\theta)-\log p(\theta)+\dcol{\log p(\mathcal D)}\big]`)),
  at(3,small('E<sub>q</sub>[ · ] means: draw θ from q, evaluate the bracket, average. Rule used: −(a + b − c) = −a − b + c.'))),
 lines:['The KL we want: an average, under q, of log q minus log posterior.','Replace the log posterior by its three terms from step 1, in brackets.','Open the brackets: every sign flips, and log p(𝒟) now enters with a plus sign.'],
 ask:['What does E_q mean, in words?','What is inside the brackets now?','Which signs changed?'],
 expected:['Draw θ from q, evaluate, average: the Monte Carlo average of 2.5.','Step 1’s expression for log p(θ | 𝒟).','All three: the likelihood and prior become minus, the evidence becomes plus.'],
 transition:'Step 3: the constant leaves the average.'});
S('elbo-step3','elbo','Step 3: a constant comes out of an average',{builds:3,
 body:stack(eq(R`\E_q\big[f(\theta)+c\big]=\E_q\big[f(\theta)\big]+c\qquad\text{when }c\text{ does not depend on }\theta`),
  at(2,table(['','three draws','their average'],[[m(R`f(\theta)`),'2, 4, 6','4'],[m(R`f(\theta)+10`),'12, 14, 16','14 = 4 + 10']])),
  at(3,eq(R`\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)=\E_q\big[\log q(\theta)-\log p(\mathcal D\mid\theta)-\log p(\theta)\big]+\dcol{\log p(\mathcal D)}`))),
 lines:['A rule about averages: add the same number c to every value, and the average goes up by exactly c.','Check it: the average of 2, 4 and 6 is 4. Add 10 to each and the average is 14.','log p(𝒟) is such a constant: step 1 showed it has no θ. So it leaves the average.'],
 ask:['Why does adding c to every value add c to the average?','Check the table.','Could log p(θ) also leave the average?'],
 expected:['The weights of an average add up to 1, so c gets weight 1 in total.','(12 + 14 + 16)/3 = 14 = 4 + 10.','No: it changes with θ. Only terms with no θ can leave.'],
 transition:'Step 4: rearrange.'});
S('elbo-step4','elbo','Step 4: rearrange, and name the part we can compute',{builds:3,
 body:stack(eq(R`\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)=-\E_q\big[\log p(\mathcal D\mid\theta)+\log p(\theta)-\log q(\theta)\big]+\dcol{\log p(\mathcal D)}`),
  at(2,eq(R`\dcol{\log p(\mathcal D)}=\E_q\big[\log p(\mathcal D\mid\theta)+\log p(\theta)-\log q(\theta)\big]+\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)`)),
  at(3,eq(R`\dcol{\log p(\mathcal D)}=\underbrace{\E_q\big[\log p(\mathcal D\mid\theta)+\log p(\theta)-\log q(\theta)\big]}_{\qcol{\ELBO(q)}}+\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)`))),
 lines:['Take a minus sign outside the average: inside, the likelihood and the prior now have plus signs, and log q a minus.','Move that average to the other side: log p(𝒟) = average + KL.','Give the average a name: the ELBO, the evidence lower bound. Every term in it is computable: draws from q, the likelihood, the prior and log q. No p(𝒟) anywhere.'],
 ask:['Check the signs inside the bracket.','Move the average across. What sign does it take?','Which terms in the ELBO can we compute?'],
 expected:['−[a + b − c] = −a − b + c: matches the line above.','Plus: it was subtracted on the right, so it is added on the left.','All of them, for any θ drawn from q: log p(𝒟 | θ) by a forward pass, log p(θ) and log q(θ) by formula.'],
 point:'Notation: ELBO(q) = E_q[log p(𝒟, θ) − log q(θ)], since p(𝒟 | θ) p(θ) = p(𝒟, θ), the joint.',
 transition:'What is each of the three pieces?'});
S('elbo-pieces','elbo','One fixed number = what we can compute + what we cannot',{builds:2,
 body:stack(eq(R`\underbrace{\dcol{\log p(\mathcal D)}}_{\text{fixed}}=\underbrace{\qcol{\ELBO(q)}}_{\text{computable}}+\underbrace{\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)}_{\text{unknown, but}\ \ge0}`),
  at(2,table(['piece','what it is','changes with q?'],[
   [m(R`\dcol{\log p(\mathcal D)}`),'the log evidence: the data’s probability, averaged over every θ','no: fixed'],
   [m(R`\qcol{\ELBO(q)}`),'average under q of log likelihood + log prior − log q','yes; computable'],
   ['the KL','how far q is from the posterior','yes; ≥ 0, and 0 only at the posterior']]))),
 lines:['Read the identity as a budget: one fixed number, split into a part we can compute and a part we cannot.','The KL is the part we want small. We cannot compute it, but we know one thing about it: it is never negative.'],
 ask:['Which piece does not change when we change q?','What do we know about the KL even without computing it?'],
 expected:['log p(𝒟): it involves the model and the data, not q.','KL ≥ 0. Next slide: why.'],
 transition:'Why is KL never negative?'});
S('kl-nonneg','elbo','Why KL is never negative',{builds:3,
 body:cols(stack(eq(R`\log t\le t-1\quad\text{for every }t>0`),
  at(2,eq(R`-\KL(q\|p)=\E_q\Big[\log\frac{p(\theta)}{q(\theta)}\Big]\le\E_q\Big[\frac{p(\theta)}{q(\theta)}-1\Big]`)),
  at(3,eq(R`\E_q\Big[\frac{p}{q}\Big]=\int q(\theta)\,\frac{p(\theta)}{q(\theta)}\,d\theta=\int p(\theta)\,d\theta=1`)),
  at(3,p(`So ${m(R`-\KL\le1-1=0`)}: KL ≥ 0, and KL = 0 only when q = p.`))),tangentChart(),'1.25fr 1fr'),
 lines:['A fact about the logarithm: its curve never rises above its tangent line at t = 1, so log t ≤ t − 1, with equality only at t = 1.','Apply it to the ratio t = p(θ)/q(θ), inside the average under q. Minus the KL is the average of log(p/q).','The average of p/q under q is the integral of p, which is 1. So minus the KL is at most 1 − 1 = 0.'],
 ask:['Where do log t and t − 1 touch?','Can a single log ratio log(p/q) be positive?','So can the KL be negative?'],
 expected:['At t = 1, where both are 0.','Yes, wherever p > q. Single terms can have either sign; only the q-weighted average is guaranteed ≥ 0.','No. And it is 0 only when p/q = 1 everywhere, that is, q = p.'],
 point:'The same fact via Jensen’s inequality is an optional slide.',
 transition:'So the ELBO is a lower bound.'});
S('elbo-bound','elbo','The ELBO is a lower bound on the log evidence',{builds:3,
 body:stack(`<p class="lead">In plain words: whatever q we choose, the <b class="q">ELBO</b> never rises above the <b class="data">log evidence log p(𝒟)</b>. It is a lower bound <b>on log p(𝒟)</b>: an <b>E</b>vidence <b>L</b>ower <b>BO</b>und.</p>`,
  eq(R`\qcol{\ELBO(q)}=\dcol{\log p(\mathcal D)}-\lcol{\underbrace{\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)}_{\ge0}}\ \le\ \dcol{\log p(\mathcal D)}`),
  `<div class="state-slot"><div data-only="2">${gapLine(c.candidates[0])}</div><div data-only="3">${table(['','log p(𝒟)','it is the log of …'],[['the coin: discrete data',num(c.log_evidence,2),'a probability: never above 0'],['our network: continuous data',num(E.estimate,2),'a density: either sign']])}</div></div>`),
 lines:['Lower bound on what? On the log evidence, log p(𝒟). The red KL is never negative, so subtracting it can only lower the number: the purple ELBO always sits below the blue log p(𝒟).','A coin with q = Beta(10, 10): the ELBO sits 0.62 nats below log p(𝒟), and the gap is exactly the KL to the posterior.',`“Lower bound” says nothing about signs: a probability is at most 1, but a density can exceed it (log p(𝒟 | θ<sub>A</sub>) = +${num(A.log_lik,1)}). Only the order is guaranteed.`],
 ask:['Below what does the ELBO sit?','What is the gap between them?','Is the ELBO always negative? Is log p(𝒟)?'],
 expected:['The log evidence log p(𝒟), for every q: that is the “lower bound”.','The KL from q to the posterior: here 0.62 nats.','No, and no. For discrete data log p(𝒟) ≤ 0; for continuous data it can be positive. Only the order is guaranteed.'],
 point:'A finite-sample ELBO estimate can cross log p(𝒟) on an unlucky draw (optional slide); the exact expectation cannot.',
 transition:'What happens to the gap as q improves?'});
S('elbo-max','elbo','Raise the ELBO, and the KL falls by exactly as much',{builds:2,
 body:stack(elboLadder(),at(2,eq(R`\max_q\ \qcol{\ELBO(q)}\quad\Longleftrightarrow\quad\min_q\ \KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)`))),
 lines:['Three q’s for the same coin. log p(𝒟) does not move. As q improves, the ELBO climbs and the gap, the KL, shrinks by exactly the same amount.','So maximising the ELBO, which we can compute, minimises the KL to the posterior, which we cannot. At the best q the gap is 0.'],
 ask:['By how much did the KL fall from Beta(10, 10) to Beta(30, 20)?','Why does maximising one minimise the other?'],
 expected:['From 0.62 to 0.05: by 0.57 nats, exactly the ELBO’s rise.','Their sum is the fixed number log p(𝒟).'],
 transition:'What does the ELBO ask of q?'});
S('elbo-two-terms','elbo','The ELBO in two terms: fit the data, stay near the prior',{builds:3,
 body:stack(eq(R`\qcol{\ELBO(q)}=\E_q\big[\log p(\mathcal D\mid\theta)+\log p(\theta)-\log q(\theta)\big]`),
  at(2,eq(R`=\E_q\big[\log p(\mathcal D\mid\theta)\big]+\E_q\big[\log p(\theta)-\log q(\theta)\big]`)),
  at(3,eq(R`=\underbrace{\E_q\big[\log p(\mathcal D\mid\theta)\big]}_{\text{fit the data}}-\underbrace{\KL\big(q(\theta)\,\|\,p(\theta)\big)}_{\text{stay near the prior}}`))),
 lines:['Take the ELBO apart.','The average of a sum is the sum of the averages: a likelihood term and the rest.','The rest is minus a KL, from q to the prior. Fit the data, but stay near the prior. For our network the second term is the diagonal-Gaussian formula of 2.4; the first needs samples.'],
 ask:['Which rule splits the average?','Why is the second average minus a KL?','Which term pulls q towards a narrow, well-fitting θ?'],
 expected:['Linearity: E[a + b] = E[a] + E[b].','E_q[log p − log q] = −E_q[log q − log p] = −KL(q ‖ prior).','The data term. The KL to the N(0, 1) prior pulls the other way: wide, near zero.'],
 transition:'Check the identity where p(𝒟) is known.'});
chap('4.2');
S('elbo-slider','elbo','Check the identity where p(𝒟) is known: a coin',{nb:'cell-coin',
 body:widget('elboslider','Beta q sliders with the ELBO and KL split of the log evidence'),
 lines:['A coin with a Beta(5, 3) prior and 17 heads in 30 tosses has an exact posterior and an exact p(𝒟). Drag q: the ELBO and the KL trade off, and their sum never changes.'],
 ask:['Find the q with the highest ELBO. What is its KL?'],
 expected:['Beta(22, 16), the exact posterior: ELBO = log p(𝒟) = −21.40 and KL = 0.'],
 point:'Conjugate coin: posterior Beta(22, 16), log p(𝒟) = log B(22, 16) − log B(5, 3). The ELBO comes from analytic Beta moments, the KL from its own formula.',
 demo:'Notebook 4.2: the coin cells compute log p(𝒟) exactly, then fit a Beta q using only the joint, and check ELBO + KL = log p(𝒟) for every candidate.',
 transition:'Part 5: implement the ELBO for the network.'});

// ================================================================ Part 5 · Implement it
chap('5.1');
S('bnn-elbo-code','impl','The ELBO for our network, in code',{builds:2,nb:'cell-bnn-vi',
 body:act(5,code('mu  = Parameter(0.5 * randn(7))         # 7 means\nrho = Parameter(full((7,), log(0.1)))   # 7 log-widths\nq = Normal(mu, rho.exp())               # one bell per parameter\ntheta = q.rsample((16,))                # 16 sampled networks\nlog_lik = log_likelihood(theta).mean()  # forward pass\nkl = kl_divergence(q, prior).sum()      # closed form\nloss = -(log_lik - kl)                  # minus the ELBO'),
  at(2,p('Adam, learning rate 0.02, 4,000 steps; best of five random starts. No p(𝒟) anywhere.'))),
 lines:['Fourteen parameters, sixteen sampled networks per step, and the closed-form KL to the prior.','Every line is computable: rsample, a forward pass and a formula. Never p(𝒟).'],
 ask:['Which line uses the reparameterisation trick?','Which line uses the Gaussian KL formula?'],
 expected:['theta = q.rsample(...): gradients flow back to μ and ρ.','kl = kl_divergence(q, prior).sum(): seven closed-form terms.'],
 demo:'Switch to notebook 5.1 and run the VI cell (about 20 seconds): it prints the five restarts’ ELBOs and the split ELBO = E_q[log p(𝒟 | θ)] − KL(q ‖ prior). Then come back for the animation.',
 transition:'Press play.'});
S('bnn-vi','impl','Press play: q learns the network',{builds:2,nb:'cell-bnn-vi',
 body:widget('bnnvi','VI on the tiny network: the predictive band and the ELBO'),
 lines:['Start from random means and narrow widths. Watch q’s predictive band and the ELBO.',`The ELBO climbs from ${big(V.frames[0][1],0)} to about ${num(V.elbo,1)} and the band settles onto the data. It stays below log p(𝒟) ≈ ${num(E.estimate)}.`],
 ask:['Predict: will the ELBO reach log p(𝒟)?','What does the gap mean?'],
 expected:['It approaches, then stops below it; it can never cross.','The gap is KL(q ‖ posterior): how far the best mean-field q is from the true posterior.'],
 point:'PAUSE. Frames every 5 steps to step 200, then every 50; the ELBO for each frame uses 2,000 fixed draws.',
 transition:'What did q learn about each parameter?'});
S('bnn-q','impl','What q learned: one hidden unit switched off',{builds:2,nb:'cell-bnn-vi',
 body:stack(qFactors({mu:V.mu,sigma:V.sigma}),at(2,p(`v₁ ≈ ${num(V.mu[4])} with width ${num(V.sigma[4],3)}: the first hidden unit is switched off, and its w₁ and b₁ fall back to the prior.`))),
 lines:['The fitted bell for each parameter, against the N(0, 1) prior in grey.','VI switched the first hidden unit off. Mean-field VI often prunes units like this: every parameter kept away from the prior costs KL.'],
 ask:['Which bells look like the prior?','Why might the ELBO prefer to switch a unit off?'],
 expected:['w₁ and b₁: wide and centred near 0, like the prior.','The KL term charges for every parameter that moves away from the prior. Here one tanh unit already fits the data well.'],
 point:`Fitted widths: ${b.names.map((nm,j)=>`${TH[nm]} ${num(V.sigma[j],3)}`).join(', ')}.`,
 transition:'Seven separate bells: what kind of q is that?'});
S('vi-families','impl','Mean-field VI, and full-covariance VI',{builds:3,nb:'cell-bnn-vi-full',
 body:cols(covPair(),stack(eq(R`\text{mean-field: }q(\theta)=\prod_{j=1}^{7}\N(\theta_j;\mu_j,\sigma_j^2)`),small('One independent bell per parameter: the diagonal family of 3.3. 7 + 7 = 14 numbers.'),
  at(2,eq(R`\text{full covariance: }q(\theta)=\N\big(\mu,\,LL^\top\big)`)),at(2,small('One tilted bell in ℝ⁷: 7 + 28 = 35 numbers; d(d + 1)/2 in general.')),
  at(3,small(`Full q: correlation ${num(VF.largest_corr.value)} between ${TH[VF.largest_corr.pair[0]]} and ${TH[VF.largest_corr.pair[1]]}; ELBO ${num(V.elbo)} → ${num(VF.elbo)}.`))),'0.95fr 1.05fr'),
 lines:['Our q is mean-field: q factorises into one bell per parameter, so it cannot express correlations. In 3.3 that made a diagonal q shrink.','Full-covariance VI uses one tilted Gaussian over all seven parameters: Σ = LLᵀ, 35 numbers instead of 14. Fine for 7 parameters; hopeless for a million.',`Starting from the mean-field answer, it learns that a steeper ${TH[VF.largest_corr.pair[0]]} goes with a smaller ${TH[VF.largest_corr.pair[1]]}, and the ELBO rises by ${num(VF.elbo-V.elbo,2)} nats: the KL to the posterior falls by exactly that much.`],
 ask:['Why is it called mean-field?','What does full covariance cost for a million weights?','Why start the full fit from the mean-field answer?'],
 expected:['From physics: each factor of q responds only to the average (“mean field”) of the others. In practice: independent factors, a diagonal covariance.','5 × 10¹¹ numbers in L: far too many to store or optimise.','Full covariance contains mean-field as a special case, so from there it can only improve; from a random start, noisy gradients on 35 numbers can stop at a worse point.'],
 point:`ELBO: mean-field ${num(V.elbo,3)}, full covariance ${num(VF.elbo,3)}, log p(𝒟) ≈ ${num(E.estimate,3)}. Full fit: Adam, learning rate 0.01, 4,000 steps, 64 draws per step.`,
 demo:'Notebook 5.1, “Mean-field against full covariance”: run it (a few seconds); it prints both ELBOs and the largest correlation, and draws both predictive bands against the reference.',
 transition:'Do the error bars improve?'});
S('bnn-vs-ref','impl','Against the reference: the right curve, bands still too narrow',{builds:2,nb:'cell-bnn-vi-full',
 body:cols(`<div class="state-slot"><div data-only="1">${viVsHmc()}</div><div data-only="2">${viVsHmc(true)}</div></div>`,stack(`<div class="table-sm">${table(['','ELBO','sd at −2.5','sd at 2.5'],[
  ['mean-field',num(V.elbo,2),num(V.sd[0],3),num(V.sd.at(-1),3)],
  [at(2,'full covariance'),at(2,num(VF.elbo,2)),at(2,num(VF.sd_pred[0],3)),at(2,num(VF.sd_pred.at(-1),3))],
  ['MCMC reference',`log p(𝒟) ≈ ${num(E.estimate,2)}`,num(H.sd[0],3),num(H.sd.at(-1),3)]])}</div>`),'0.95fr 1.05fr'),
 lines:['Mean-field VI gets the mean curve right, but its bands are too narrow beyond the data: the diagonal q of 3.3 shrinking, in seven dimensions.',`Full covariance helps a little. The rest of the gap is the family: the posterior has several modes (swap the hidden units, flip signs), and one Gaussian sits on one.`],
 ask:['Where do VI and the reference disagree most?','Why doesn’t full covariance close the gap?'],
 expected:['Beyond the data at both ends, where VI is overconfident.','A Gaussian has one peak. The posterior has eight mirror-image copies of each mode, so up to log 8 ≈ 2.1 nats of the gap is that alone; the remaining shape is not Gaussian either.'],
 point:`Even tilted, one Gaussian stays about ${num(VF.gap,1)} nats below log p(𝒟). Predictive sd at x = 0: mean-field ${num(V.sd[60],3)}, full ${num(VF.sd_pred[60],3)}, reference ${num(H.sd[60],3)}.`,
 demo:'Notebook 5.1: the last figure shows both VI bands against the reference.',
 transition:'Collect the tools.'});
chap('5.2');
S('five-tools','impl','Six obstacles, six tools',{
 body:table(['obstacle','tool','where'],[
  ['p(𝒟) is an impossible integral','approximate the posterior with q','1.4 – 1.5'],
  ['“close” needs a number','KL(q ‖ p)','2.1 – 2.4'],
  ['a grid cannot search 14 numbers','gradient descent','3.1 – 3.2'],
  ['no formula for the average','Monte Carlo','2.5, 3.4'],
  ['no gradient through a draw','θ = μ + σε, rsample()','3.5'],
  ['KL to the posterior needs p(𝒟)','maximise the ELBO','4.1']]),
 lines:['Every obstacle today earned a tool, and our network used them all. In Lecture 3 a VAE does the same with one q per image.'],
 ask:['Cover the middle column. Which tool removed which obstacle?'],
 expected:['Students reconstruct it row by row.'],
 transition:'Next lecture.'});
S('next-lecture','impl','Generative models create this inference problem again',{builds:2,
 body:stack(p('A hidden description can generate an observation.'),at(2,p('Next: learn representations for images, then ask whether they can generate new ones.'))),
 lines:['In an image model, a hidden code z can generate an observed image x, just as the mixture’s bump z generated θ in 3.4.','Lecture 2 builds autoencoders; Lecture 3 returns to VI with an encoder that outputs q for each image.'],
 ask:['Where might a new unknown appear in an image model?','What should we ask of a good reconstruction?'],
 expected:['A latent description z can explain an observed image x.','Whether reconstructing observations also gives a useful way to generate new ones.'],
 transition:'Close the main route. Optional depth follows.'});

// ================================================================ Optional depth
const O=(id,title,o)=>S(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});
// Appendix: the hands-on route to KL with grades (after ritvikmath)
O('grades','By hand: two runs of this course, two grade distributions',{nb:'cell-grades',builds:2,
 body:cols(gradeBars(gr.q,gr.p),stack(p('<b>Appendix: KL by hand.</b> A second route to the same formula.'),p('Grades in last year’s run (q) and this year’s (p).'),at(2,p('A: 10% → 40%. B: 40% → 10%. C: 50% both years.'))),'1.3fr 1fr'),
 lines:['A second route to KL, by hand: the grade distributions of two runs of this course. How different are they, as one number?','Four times as many A’s, a quarter as many B’s, the same share of C’s.'],
 ask:['How would you score the difference between the two years?','A fourfold rise and a fourfold fall: should they count the same?'],
 expected:['Students propose differences of percentages. Note that C, unchanged, should contribute nothing.','By ratio, yes: both are a factor of 4. That suggests comparing ratios.'],
 point:'p is this year (the distribution we weight by); q is the reference, last year.',
 source:RITVIK,transition:'Compare grade by grade.'});

O('grades-ratios','Compare grade by grade: ratios',{builds:2,
 body:cols(table(['grade',m(R`p`),m(R`q`),m(R`p/q`)],gr.names.map((name,i)=>[name,num(gr.p[i],1),num(gr.q[i],1),gr.ratio[i]===.25?'¼':num(gr.ratio[i],0)])),
  stack(ratioScale(false),at(2,p(`Average the ratios? (4 + ¼ + 1) / 3 = ${num(gr.ratio_mean,2)}. Identical years would give 1, not 0, and ×4 sits 3 above 1 while ÷4 sits only 0.75 below.`))),'1fr 1.15fr'),
 lines:['How many times more common is each grade this year? A: 4. B: ¼. C: 1.','Averaging raw ratios is lopsided: a fourfold rise counts four times as much as a fourfold fall.'],
 ask:['On the ratio scale, which change looks bigger?','What should identical years score?'],
 expected:['The rise: +3 against −0.75, although both are fourfold.','Zero. Raw ratios give 1.'],
 source:RITVIK,transition:'Find a function of the ratio that behaves.'});

O('grades-f','What should a ratio count for? Find the function',{builds:3,
 body:cols(stack(p('We want a function f of the ratio r with:'),eq(R`f(1)=0`),eq(R`f(4)=+y,\qquad f(\tfrac14)=-y`),eq(R`f(2\cdot2)=f(2)+f(2)`)),
  table([m(R`f(r)`),m(R`f(\tfrac14)`),m(R`f(1)`),m(R`f(4)`),m(R`f(2)+f(2)`)],[
   [at(2,m(R`r-1`)),at(2,'−0.75'),at(2,'0'),at(2,'3'),at(2,'2 ≠ 3 ✗')],
   [at(2,m(R`r-\tfrac1r`)),at(2,'−3.75'),at(2,'0'),at(2,'3.75'),at(2,'3 ≠ 3.75 ✗')],
   [at(3,m(R`\log r`)),at(3,'−1.39'),at(3,'0'),at(3,'1.39'),at(3,'1.39 = 1.39 ✓')]]),'1fr 1.25fr'),
 lines:['What should a ratio count for? No change should count 0; a fourfold rise and a fourfold fall should cancel; and changes should add up.','Try a few functions. r − 1 is lopsided. r − 1/r balances ×4 against ÷4, but two doublings do not add up to a quadrupling.','The logarithm passes every test: log 4 = 1.39, log ¼ = −1.39, and log 2 + log 2 = log 4.'],
 ask:['Why should f(1) be 0?','Which test does r − 1/r fail?','Is the logarithm the only choice?'],
 expected:['No change in a grade should add any difference.','Adding up: f(2) + f(2) = 3, but f(4) = 3.75.','Up to its base, yes: a smooth f with f(a·b) = f(a) + f(b) must be a logarithm. The base sets the unit.'],
 source:RITVIK,transition:'Now combine the three log ratios.'});

O('grades-kl','Weight each grade by this year’s share: that is KL',{nb:'cell-grades',demo:'Notebook 2.1: the grades cell computes the ratios, the logs and the weighted sum, and checks it against torch.distributions.kl_divergence.',builds:3,
 body:cols(`<div class="state-slot"><div data-only="1">${gradeValueBars(gr.log_ratio,'log ratio of each grade, equal weights')}</div><div data-only="2-3">${gradeValueBars(gr.terms,'p(x) × log ratio, and their sum',gr.kl)}</div></div>`,
  stack(eq(R`\text{plain average: }\tfrac13\big(1.39-1.39+0\big)=0`),at(2,eq(R`\begin{aligned}\text{weighted: }&0.4(1.39)+0.1(-1.39)\\&+0.5(0)=${num(gr.kl,2)}\end{aligned}`)),
  at(3,eq(R`\KL(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}`))),'1fr 1.1fr'),
 lines:['Why not simply average the logs? Here the plain average is exactly 0, yet the two years clearly differ.','Weight each grade by how many students got it this year: the A’s, now common, count most. The weighted sum is 0.42.','That weighted average of log ratios is the Kullback–Leibler divergence of p from q.'],
 ask:['Why is the plain average 0 here?','Why weight by this year’s shares?','What would identical years give?'],
 expected:['The rise in A and the fall in B are both fourfold, so their logs cancel, and C did not change.','We are describing this year’s students: grades that many of them got should count more.','0: every ratio is 1 and every log is 0.'],
 source:RITVIK,transition:'What unit is 0.42 in?'});

O('nats-bits','The unit: nats, or bits',{nb:'cell-grades',builds:2,
 body:stack(eq(R`\KL(p\|q)=\sum_x p(x)\,\ln\frac{p(x)}{q(x)}=${num(gr.kl,2)}\ \text{nats}`),eq(R`\sum_x p(x)\,\log_2\frac{p(x)}{q(x)}=${num(gr.kl_bits,2)}\ \text{bits}`),
  at(2,p(`Read ${num(gr.kl,2)} nats as e<sup>${num(gr.kl,2)}</sup> ≈ ${num(gr.geometric_mean_ratio,1)}: on average, a grade from this year is ${num(gr.geometric_mean_ratio,1)} times more probable under this year’s distribution than under last year’s.`))),
 lines:['The log’s base sets the unit. Natural log: nats. Log base 2: bits. Here 0.42 nats is exactly 0.60 bits; 1 nat ≈ 1.44 bits.','0.42 nats means a typical this-year grade is about 1.5 times more probable under this year’s distribution than under last year’s.'],
 ask:['Why does changing the base only rescale the KL?','What would a KL of 0 nats mean?'],
 expected:['log₂ x = ln x / ln 2: every term scales by the same factor 1/ln 2 ≈ 1.44.','e⁰ = 1: grades are equally probable under both years, so the two distributions match.'],
 point:'e^KL is the geometric mean of p(x)/q(x) over x drawn from p.',
 transition:'Does it matter which year is the reference?'});

O('asymmetry','Swap the reference year: a different number',{nb:'cell-grades',builds:2,
 body:cols(gradeBars(gr.swap.q,gr.swap.p),stack(eq(R`\KL(\text{this}\,\|\,\text{last})=${num(gr.swap.kl_pq,2)}`),at(2,eq(R`\KL(\text{last}\,\|\,\text{this})=${num(gr.swap.kl_qp,2)}`)),at(2,p('Which distribution comes first matters: KL is not symmetric.'))),'1.3fr 1fr'),
 lines:['A different pair of years: last year 1% got an A, this year 25% did. Measured from this year: 0.64 nats.','Measured from last year instead: 0.30 nats. Changing the reference changes the answer: KL is not symmetric.'],
 ask:['Which direction do you expect to be larger?','Is KL a distance in the everyday sense?'],
 expected:['From this year’s view the A’s are a big change that many students have, so 0.64 is larger.','No: a distance is the same both ways. KL is a divergence.'],
 point:'The first grades example happens to give 0.42 in both directions, because it is a mirror image (A and B swap shares). That is why this slide uses a second pair.',
 transition:'Why does the reference matter?'});

O('asymmetry-why','Why the reference matters',{nb:'cell-grades',builds:2,
 body:stack(swapContrib(),at(2,small('The first distribution supplies the weights, so it decides which changes count.'))),
 lines:['From this year’s view, A is 25 times more common than before, and a quarter of the class has it: A contributes 0.80.','From last year’s view A was rare, so its change barely counts (−0.03); the drop in B dominates instead.'],
 ask:['Which grade dominates each direction?','So which order will VI use, and why might it matter?'],
 expected:['A from this year’s view; B from last year’s.','VI will put q first, KL(q ‖ p). Part 3 shows what that order does to the fit.'],
 transition:'Sample grades to estimate it.'});

O('mc-grades','Sample grades and watch the average find the KL',{nb:'cell-grades-sampling',demo:'Notebook 2.5: the grades-sampling cell draws 10,000 grades and prints the running average at 10, 100, 1,000 and 10,000 draws.',
 body:widget('mcgrades','Grades drawn from this year’s distribution, and the running average of their log ratios'),
 lines:['Draw grades from this year’s distribution and average their log ratios. The counts approach 40%, 10%, 50%, and the average approaches the exact 0.42.'],
 ask:['After 10 draws, is the estimate close?'],
 expected:['Not necessarily: with few draws the shares are off, so the estimate wobbles. It settles as the counts approach 40/10/50.'],
 point:'The exact value is the weighted sum (analytic); the running average is the Monte Carlo estimate. Draws are seeded, so the sequence replays identically.',
 transition:'Back to the optional slides.'});

O('kl-estimators','A KL estimator that is never negative',{builds:2,nb:'cell-estimators',
 body:stack(eq(R`r=\frac{p(\theta)}{q(\theta)},\ \theta\sim q:\qquad\underbrace{-\log r}_{\text{plain}}\qquad\text{vs}\qquad\underbrace{r-1-\log r}_{\ge0\text{ for every draw}}`),
  at(2,table(['draws S','plain: spread','plain: negative','r − 1 − log r: spread','negative'],st.estimators.rows.map(rw=>[String(rw.S),num(rw.plain_sd,3),`${Math.round(100*rw.plain_negative)}%`,num(rw.better_sd,3),`${Math.round(100*rw.better_negative)}%`])))),
 ask:['Why does adding r − 1 not change the average?','When can the better estimator be worse?'],
 expected:['E_q[r] = ∫ q · p/q = ∫ p = 1, so E_q[r − 1] = 0: it only removes noise.','When p has heavier tails than q: r = p/q can then have infinite variance. Here p is narrower than q, so r stays bounded.'],
 point:`${st.estimators.pair}, exact KL ${num(st.estimators.exact,3)}; 400 repeats per size. From Jia-Bin Huang’s video and John Schulman’s note “Approximating KL divergence”; GRPO uses this estimator for its KL penalty.`,
 source:'After <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang</a> and <a href="http://joschu.net/blog/kl-approx.html">John Schulman, “Approximating KL divergence”</a>'});
O('evidence-mc','Sampling from the prior: a few lucky draws carry everything',{nb:'cell-evidence-mc',builds:2,
 body:cols(evidenceMcChart(),stack(eq(R`p(\mathcal D)\approx\frac1S\sum_{s=1}^{S}p(\mathcal D\mid\theta_s)`),eq(R`\theta_s\sim p(\theta)`),
  at(2,p(`Ten draws: off by about 1,000 nats. Ten million: about ${effective} draws carry the estimate, ${num(E.estimate)}.`))),'1.05fr 1fr'),
 lines:['Average the likelihood over draws from the prior: four seeds, up to ten million draws each.','Almost every prior draw explains the data terribly. Seven parameters just about work; more parameters or more data make it hopeless.'],
 ask:['Why are the small-S estimates so low?','What happens with a million parameters?'],
 expected:['Prior draws almost never land where the likelihood is high, like θ<sub>B</sub>; the rare good draws dominate the average.','The chance that a prior draw lands near the posterior shrinks exponentially; no feasible S is enough.'],
 point:`Effective number of draws, out of 10⁷: ${E.runs.map(r=>Math.round(r.effective_draws)).join(', ')}. The largest single draw carries ${E.runs.map(r=>Math.round(100*r.largest_share)+'%').join(', ')} of the total.`,
 transition:'Why do we want the posterior at all?'});
O('coin-evidence','A KL value is how fast evidence piles up',{nb:'cell-coin-kl',builds:2,
 body:cols(evidenceChart(),stack(eq(R`\KL\big(\mathrm{Bern}(0.8)\,\|\,\mathrm{Bern}(0.5)\big)\approx${num(st.coinkl.kl)}\ \text{nats}`),
  at(2,eq(R`\frac{p(\mathcal D\mid0.8)}{p(\mathcal D\mid0.5)}\approx e^{\,${num(st.coinkl.kl)}\,n}`)),at(2,small(`The reverse direction, KL(0.5‖0.8) = ${num(st.coinkl.reverse)}, is a different number.`))),'1fr 1.15fr'),
 lines:['A coin lands heads 80% of the time. Each toss favours “0.8” over “fair” by 0.19 nats on average. After 30 tosses, by what factor?','About e^(30 × 0.19) ≈ 325. KL is the rate at which evidence piles up: 6.9× after 10 tosses, 325× after 30.'],
 ask:['Predict: after 30 tosses, how many times more likely are the data under 0.8 than under 0.5?','Why does a small KL eventually matter so much?'],
 expected:['Most guess far too low. It is about 325×.','Log evidence adds up toss by toss, so the ratio grows exponentially with n.'],
 point:`PAUSE. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of ${num(st.coinkl.simulated_mean_llr_30,3)}, against 30 × KL = ${num(30*st.coinkl.kl,3)}.`,
 transition:'Sums become integrals for continuous unknowns.'});
O('exit-transfer','A sensor has an unknown constant bias',{nb:'optional',builds:3,
 body:stack(eq(R`\theta\sim\N(0,1),\qquad y_i\mid \theta\sim\N(\theta,0.5^2)`),
 at(2,eq(R`p(\mathcal D,\theta)=p(\theta)\prod_i p(y_i\mid \theta)`)),
 at(3,eq(R`\max_\psi\E_{q_\psi(\theta)}[\log p(\mathcal D,\theta)-\log q_\psi(\theta)]`))),
 lines:['A sensor reads a hidden bias θ, with noise. Name the unknown, the data and a q.','The joint is the prior times the likelihood of each reading: available.','Maximise the ELBO over q’s parameters. The same tools apply.'],
 ask:['Identify the unknown, the observations and one possible q family.','Which available functions form the joint?','What objective would you implement?'],
 expected:['θ is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.','The prior density and the likelihood of each reading.','The expectation of log joint minus log q, maximised over ψ.'],
 point:'This conjugate toy has an exact answer too; the exercise is identifying the pieces, not needing approximation.',
 transition:'Where these unknowns appear next.'});
O('mc-variability','More samples, tighter estimates',{nb:'cell-mc-variability',
 body:image('figures/mc-variability.svg','Forty KL estimates for each of four sample counts, around the exact value.'),
 lines:['Forty repeats at each sample size. The spread shrinks like 1/√S; the exact value never moves.'],
 ask:['How many more draws halve the spread?'],
 expected:['Four times as many: the spread scales as 1/√S.'],
 point:'Same p and q at every size; 40 seeds each.',
 transition:'Act 4: we can estimate the loss. Now optimise it.'});
O('answer-key','A check with an answer key: the Beta coin',{nb:'cell-coin-fit',builds:3,
 body:stack(only('1',coinPlot(0,'0')),only('2',coinPlot(0,'100')),only('3',coinPlot(0,'600'))),
 lines:['With the Beta(5, 3) prior we know the posterior: Beta(22, 16). Fit a Beta q from Beta(10, 10) using only the joint.','By step 100, q has moved most of the way.',`After 600 steps: Beta(${num(c.training.trace.at(-1).a,1)}, ${num(c.training.trace.at(-1).b,1)}), with KL ≈ ${num(c.training.trace.at(-1).posterior_kl,4)} to the true posterior.`],
 ask:['Where does Beta(10, 10) put its mass?','What changed by step 100?','Is the final q exactly Beta(22, 16)?'],
 expected:['Centred at 0.5, and too broad.','It shifted towards the heads proportion and narrowed.','No, but very close: KL ≈ 0.0002. The answer key was never used by the optimiser.'],
 point:'Beta.rsample is differentiable through implicit reparameterisation, not μ + σε. Adam 0.025, 512 draws, seed 2401.',
 transition:'Move q yourself and watch the total.'});
O('optional-one-step','One literal update of the mean',{nb:'cell-one-step',builds:3,
 body:stack(eq(R`\sigma=2\text{ fixed}:\qquad L(\mu)=\text{const}+\frac{(\mu-1)^2}{32}`),
  at(2,eq(R`\frac{\partial L}{\partial\mu}=\frac{\mu-1}{16}=-0.0625\ \text{ at }\mu=0`)),
  at(3,eq(R`\mu^+=0-4(-0.0625)=0.25:\quad\KL\ ${num(n.one_step.kl_before,3)}\to${num(n.one_step.kl_after,3)}`))),
 ask:['Which term of the Gaussian KL depends on μ?','Differentiate it at μ = 0.','Take one step with η = 4. What is the best KL with σ stuck at 2?'],
 expected:['Only the squared shift, (μ − 1)²/(2·4²).','(μ − 1)/16 = −0.0625: a negative derivative, so descent moves μ right.',`μ⁺ = 0.25. Even at μ = 1 the KL is ${num(n.one_step.fixed_scale_best_kl,3)}: the width mismatch needs σ to move too.`],
 point:'η = 4 is chosen to make the arithmetic visible, not a recommended learning rate.'});
O('optional-nonnegative','KL ≥ 0 by Jensen’s inequality',{nb:'optional',builds:3,
 body:stack(eq(R`\KL(q\|p)=-\E_q\!\left[\log\frac{p(\theta)}{q(\theta)}\right]`),
 at(2,eq(R`\geq-\log\E_q\!\left[\frac{p(\theta)}{q(\theta)}\right]`)),
 at(3,eq(R`=-\log1=0`))),
 ask:['Which function makes Jensen’s inequality useful?','Why does the inequality point this way?','What makes the final expectation one?'],
 expected:['The convex function −log.','For a convex function, its expectation is at least its value at the expectation.','Integrating q · p/q gives the integral of p, which is 1 on a common support.'],
 point:'The tangent-line argument of Section 4.1 is the same fact without naming Jensen.'});
O('optional-negative-estimate','Five draws can give a negative KL estimate',{nb:'cell-negative',builds:2,
 body:stack(table(['θᵢ','log q(θᵢ) − log p(θᵢ)'],mc.negative_example.draws.map((x,i)=>[f(x,3),f(mc.negative_example.terms[i],3)])),
 at(2,eq(R`\widehat{\KL}_5\approx${num(mc.negative_example.estimate,3)},\qquad\KL(q\|p)\approx${num(mc.exact_kl,3)}`))),
 ask:['Some rows are negative. Could their average be negative?','Does this contradict KL ≥ 0?'],
 expected:['Yes: this five-draw average is about −0.093.','No. A finite-sample estimate is not the exact expectation.'],
 point:'Seed 0 was chosen deliberately as the first negative example in a seed search.'});
O('optional-mixture-candidates','The two PML candidates, scored by Monte Carlo',{nb:'cell-mixture-mc',builds:2,
 body:stack(table(['candidate','MC estimate ± 1 s.e.','numerical integral'],d.mixture.candidates.map(r=>[
  `N(3, ${r.sigma}²)`,`${f(r.mc,2)} ± ${f(r.standard_error,2)}`,f(r.quadrature,2)])),
 at(2,p('The narrow candidate scores far better. Both sit away from the mixture’s mass, so both are poor.'))),
 ask:['How large is the difference relative to the sampling uncertainty?','What role does the numerical integral play?'],
 expected:['Far larger than the standard errors.','An independent check, possible only because this toy is one-dimensional.'],
 point:'100,000 draws each, seeds 402 and 403.'});
O('optional-rsample-check','A fixed-noise finite difference checks the path',{nb:'cell-rsample-check',builds:2,
 body:stack(code('q = Normal(mu, sigma)\ntheta = q.rsample((S,))\nloss = (q.log_prob(theta) - p.log_prob(theta)).mean()'),
 at(2,table(['derivative','autograd','finite difference'],[
 ['μ',f(g.finite_difference.autograd[0],6),f(g.finite_difference.finite_difference[0],6)],
 ['σ',f(g.finite_difference.autograd[1],6),f(g.finite_difference.finite_difference[1],6)]]))),
 ask:['What is different about rsample?','What must stay fixed on both sides of the finite difference?'],
 expected:['It returns a draw with the pathwise gradient connection.','The same noise array. Fresh noise would confound the check with sampling variation.'],
 point:'Seed 88, S = 2048, h = 10⁻⁵.'});
O('optional-score-function','Pathwise gradients are one route, not the only one',{nb:'sec-3-5',builds:2,
 body:stack(eq(R`\nabla_\psi\E_{q_\psi}[f_\psi(\theta)]=\E_{q_\psi}[f_\psi(\theta)\nabla_\psi\log q_\psi(\theta)+\nabla_\psi f_\psi(\theta)]`),
 at(2,p('A score-function estimator accounts explicitly for the changing sampling distribution.'))),
 ask:['Which term was missing from the naive sample() gradient?','Does this identity require θ = μ + σε?'],
 expected:['The term accounting for the change in the probability of drawing θ.','No. It works for discrete q too, usually with higher variance.'],
 point:'Regularity conditions allow differentiating under the integral.'});
O('optional-noisy-elbo','A noisy ELBO estimate can cross the evidence',{nb:'sec-4-2',builds:2,
 body:stack(image('figures/coin-noisy-objective.svg','Sampled ELBO estimates and the analytic ELBO along the Beta coin fit.'),
 at(2,small(`${c.sampled_estimates_above_evidence} of 600 sampled estimates lie above the known evidence in this run.`))),
 ask:['Can a finite-sample ELBO estimate cross the reference line?','Which object remains a lower bound?'],
 expected:['Yes, on some draws.','The exact expected ELBO, not each sampled estimate.'],
 point:'Beta coin, seed 2401.'});
O('optional-certificate','A higher ELBO does not prove a zero gap',{nb:'sec-4-1',builds:2,
 body:stack(eq(R`\ELBO(q_{\mathrm{new}})>\ELBO(q_{\mathrm{old}})`),
 at(2,p('The family may not contain the posterior, and optimisation may stop short.'))),
 ask:['What does a true increase in the exact ELBO establish?','Does a stable loss establish q = p(θ | 𝒟)?'],
 expected:['For this fixed model and data, the posterior KL decreased.','No. The family or optimisation can leave a positive gap.'],
 point:`Recall the diagonal q in Section 3.3: its best possible KL was ${num(c3.analytic.diag_kl)}.`});
O('optional-coin-check','Check the fixed total with three Beta candidates',{nb:'cell-coin',builds:3,
 body:c.candidates.map((row,i)=>only(i+1,coinPlot(i,null,{height:260})+gapLine(row))).join(''),
 ask:['Add the ELBO and the gap. What is fixed?','Does the better candidate raise the ELBO by the amount its KL falls?','What happens at the exact posterior?'],
 expected:['The sum is the same log evidence, about −21.397.','Yes. The identity holds for each candidate.','The KL is zero, so the ELBO equals the log evidence.'],
 point:'ELBO computed directly from expected log joint minus expected log q; KL computed separately.'});
O('optional-reductions','A mean must say which dimension it averages',{nb:'optional',builds:2,
 body:stack(table(['object','shape','correct reduction'],[
 ['S scalar Beta draws','[S]','mean over S'],
 ['S d-dimensional joint log densities','[S]','mean over S'],
 ['S factorised coordinate log densities','[S, d]','sum over d, then mean over S'],
 ]),at(2,p('MultivariateNormal.log_prob already sums the event dimension. Do not sum the sample axis as well.'))),
 ask:['Does every log_prob output still have an event dimension?','What goes wrong if you sum over S before averaging?'],
 expected:['No. A multivariate distribution’s log_prob already reduces its event dimensions.','It rescales the objective, often multiplying an entropy contribution by the number of samples.']});
O('optional-count-data','An ordered sequence and a count have different evidence',{nb:'optional',builds:2,
 body:stack(eq(R`p(H=17\mid r)=\binom{30}{17}r^{17}(1-r)^{13}`),
 at(2,table(['observed object','log evidence'],[
 ['one ordered sequence',f(c.log_evidence,3)],
 ['only the head count',f(c.count_convention_comparison.count_log_evidence,3)],
 ]))),
 ask:['Which event contains more possible sequences?','Does this constant change the posterior or its KL gap?'],
 expected:['The count-only event includes every ordering with 17 heads.','No. It adds the same constant to the joint, the ELBO and the evidence.'],
 point:'The whole lecture uses the ordered-sequence convention.'});
O('optional-beta-elbo','Check the sampled objective with analytic Beta moments',{nb:'optional',builds:3,
 body:stack(eq(R`\E_q[\log r]=\operatorname{digamma}(a)-\operatorname{digamma}(a+b)`),
 at(2,eq(R`\E_q[\log(1-r)]=\operatorname{digamma}(b)-\operatorname{digamma}(a+b)`)),
 at(3,p('Insert these into expected log likelihood + expected log prior − expected log q.'))),
 ask:['Which two expectations determine the coin ELBO?','Why write digamma rather than ψ here?','How was the identity checked non-circularly?'],
 expected:['Expected log r and expected log(1−r).','ψ already denotes the variational parameters.','The ELBO is computed directly from these moments, then the separately computed KL is added and compared with the evidence.']});

// ================================================================ Lecture map, filled in once every section knows its first slide
S('contents','map','Lecture 1 at a glance: five parts, twenty-four sections',{nb:'top',
 body:'',
 lines:['Five parts, twenty-four numbered sections. The notebook uses the same numbers: the ▶ Notebook link at the top right of every slide opens its section in Colab.'],
 ask:['Which part do you expect to be hardest?'],
 expected:['Usually Part 4, the ELBO derivation, so we take it one rule per line.'],
 point:'Click a section to jump to its first slide; ▶ notebook opens the matching notebook section in Colab. Run the notebook top to bottom once before class.',
 transition:'Part 1: Bayes’ rule for a neural network.'});

// The contents slide goes first, once every section knows its first slide.
slides.unshift(slides.pop());
slides[0].body=`<div class="toc toc-flow">${PARTS.map(([k,t])=>`<h3>Part ${k} · ${t}</h3><ol>${CHAPTERS.filter(([num])=>num.startsWith(k+'.')).map(([num,title])=>
 `<li><a href="#${firstOf[num]}/1">${num} ${title}</a><a class="toc-nb" href="${COLAB}#scrollTo=${secAnchor(num)}" target="_blank" rel="noopener" aria-label="notebook ${num}">▶</a></li>`).join('')}</ol>`).join('')}</div>`;
for(const [num] of CHAPTERS)if(!firstOf[num])throw new Error('Section '+num+' has no slides');
