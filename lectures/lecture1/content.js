import {m,eq,at,only,cols,code,image,table,flow,plot,palette} from '../shared/author.js';
import {panel,label,legend,fig,widget,TINT,MUTED,AX,num,lin,mount as mountWidgets} from './widgets.js';
import {normalPdf,normalKL} from './math.mjs';

const d=await fetch(new URL('./data/science.json',import.meta.url)).then(r=>r.json());
export const mount=(stage,ctx)=>mountWidgets(stage,ctx,d);
const R=String.raw;
const f=(v,n=4)=>Number(v).toFixed(n).replace('-','−');
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const n=d.normal,c=d.coin,g=d.gradients,mc=d.monte_carlo,st=d.story,b=st.bnn,ink=palette.ink;
export const meta={
 id:'lecture1',number:1,title:'KL and variational inference',
 subtitle:'Bayes for a neural network, KL from scratch, and the ELBO',minutes:75,
 notebook:'lecture1-variational-inference.ipynb',
 sections:[
  {id:'bayes',title:'Part 1 · Bayes’ rule for a neural network',minutes:16},
  {id:'kl',title:'Part 2 · How far apart are two distributions?',minutes:20},
  {id:'fit',title:'Part 3 · Find the closest q',minutes:22},
  {id:'elbo',title:'Part 4 · Back to Bayes: the ELBO',minutes:9},
  {id:'impl',title:'Part 5 · Implement it',minutes:8},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
/** The bottom bar carries one story line per build; questions and answers live in the notes. */
function S(id,section,title,o){
 const builds=o.builds||1;
 for(const k of ['lines','ask','expected'])if(!Array.isArray(o[k])||o[k].length!==builds)throw new Error(`${id}: ${k} needs ${builds} entries`);
 slides.push({id,section,title,builds,body:o.body,prompts:o.lines,
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
 return plot(series,{width:640,height:340,xMin:-11,xMax:13,yMin:0,yMax:.23,xLabel:'u',yLabel:'density',
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
function fruitBars(){
 const fr=st.fruit,W=640,H=330,base=274,sc=v=>v/.6*210;let s='';
 fr.names.forEach((name,i)=>{const cx=110+i*200;
  for(const [v,color,dx] of [[fr.q[i],palette.q,-31],[fr.p[i],palette.p,31]]){
   s+=`<rect x="${cx+dx-28}" y="${(base-sc(v)).toFixed(1)}" width="56" height="${sc(v).toFixed(1)}" rx="3" fill="${color}"/>`+label(cx+dx,base-sc(v)-10,`${Math.round(v*100)}%`,{fill:color,size:21,anchor:'middle'});}
  s+=label(cx,base+30,name,{fill:palette.ink,size:22,anchor:'middle'});});
 s+=`<line x1="20" y1="${base}" x2="${W-20}" y2="${base}" stroke="${AX}"/>`;
 s+=`<rect x="20" y="16" width="18" height="18" rx="3" fill="${palette.q}"/>`+label(46,32,'last year, q',{fill:palette.q,size:21})+
  `<rect x="220" y="16" width="18" height="18" rx="3" fill="${palette.p}"/>`+label(246,32,'this year, p',{fill:palette.p,size:21});
 return fig(W,H,s,'Favourite fruit shares: last year 50, 40, 10 percent; this year 50, 10, 40 percent.');
}
function ratioLine(logs){
 const W=560,H=160,y0=96,lo=logs?-1.7:0,hi=logs?1.7:4.4,X=v=>30+(v-lo)/(hi-lo)*500,fr=st.fruit;
 const vals=logs?fr.log_ratio:fr.ratio,colors=[MUTED,palette.loss,palette.q];
 let s=`<line x1="${X(lo)}" y1="${y0}" x2="${X(hi)}" y2="${y0}" stroke="${AX}"/>`+label(14,24,logs?'log ratio, log(p / q)':'ratio, p / q',{fill:palette.ink,size:22,weight:600});
 for(const t of logs?[-1,0,1]:[0,1,2,3,4])s+=`<line x1="${X(t)}" y1="${y0-5}" x2="${X(t)}" y2="${y0+5}" stroke="${AX}"/>`+label(X(t),y0+28,num(t,0),{size:19,anchor:'middle'});
 const one=X(logs?0:1);
 [1,2].forEach(i=>{const xv=X(vals[i]);s+=`<line x1="${one}" y1="${y0+46}" x2="${xv}" y2="${y0+46}" stroke="${colors[i]}" stroke-width="5" stroke-linecap="round"/>`;
});
 vals.forEach((v,i)=>{const lift=i===1&&!logs;s+=`<circle cx="${X(v)}" cy="${y0}" r="8" fill="${colors[i]}"/>`+label(lift?X(v)-8:X(v),lift?y0-44:y0-16,fr.names[i],{fill:colors[i],size:19,anchor:lift?'start':'middle'});});
 return fig(W,H,s,logs?'Log ratios are symmetric: minus 1.39, 0 and plus 1.39.':'Ratios are lopsided: a quarter, 1 and 4.');
}
function contributionBars(){
 const fr=st.fruit,W=560,H=330,y=v=>200-v*300,bars=[...fr.terms.map((v,i)=>[fr.names[i],v,v<0?palette.loss:v>0?palette.q:MUTED]),['KL',fr.kl,palette.ink]];
 let s=`<line x1="20" y1="${y(0)}" x2="${W-20}" y2="${y(0)}" stroke="${AX}"/>`;
 bars.forEach(([name,v,color],i)=>{const cx=85+i*130,top=Math.min(y(0),y(v)),h=Math.max(2,Math.abs(y(v)-y(0)));
  const g=`<rect x="${cx-34}" y="${top.toFixed(1)}" width="68" height="${h.toFixed(1)}" rx="3" fill="${color}"/>`+
   label(cx,v<0?y(v)+28:y(v)-10,num(v,2),{fill:color,size:21,anchor:'middle'})+label(cx,316,name,{fill:palette.ink,size:21,anchor:'middle',weight:i===3?'600':''});
  s+=i===3?`<g data-at="2">${g}</g>`:g;});
 return fig(W,H,s,'Each fruit’s contribution p times log ratio: 0, minus 0.14 and 0.55; their sum, the KL, is 0.42.');
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
  label(P.x(1.12),P.y(-.38),'touch at t = 1',{fill:palette.ink,size:20}),'The curve log t lies below its tangent line t minus 1, touching at t = 1.');
}
function twoD(){
 const W=720,H=340,S=48,sd=st.twod.analytic_sd;let s='';
 [0,370].forEach((ox,j)=>{const cx=ox+175,cy=170;
  const ell=(k,rx,ry,color,dash,rot=-45)=>`<ellipse cx="${cx}" cy="${cy}" rx="${(k*rx*S).toFixed(1)}" ry="${(k*ry*S).toFixed(1)}" transform="rotate(${rot} ${cx} ${cy})" fill="none" stroke="${color}" stroke-width="3"${dash?' stroke-dasharray="8 6"':''}/>`;
  let panelSvg=`<line x1="${cx-150}" y1="${cy}" x2="${cx+150}" y2="${cy}" stroke="${AX}"/><line x1="${cx}" y1="${cy-150}" x2="${cx}" y2="${cy+150}" stroke="${AX}"/>`+
   label(cx+150,cy+24,'u₁',{size:20,anchor:'end'})+label(cx+10,cy-112,'u₂',{size:20});
  for(const k of [1,2])panelSvg+=ell(k,Math.sqrt(1.9),Math.sqrt(.1),palette.p,false);
  if(j===0){let q='';for(const k of [1,2])q+=ell(k,sd,sd,palette.q,false,0);panelSvg+=`<g data-at="2">${q}${label(ox+10,330,`diagonal q: widths ${num(sd)}`,{fill:palette.q,size:21})}</g>`+label(ox+10,22,'target p, correlation 0.9',{fill:palette.p,size:21});}
  else{let q='';for(const k of [1,2])q+=ell(k,Math.sqrt(1.9),Math.sqrt(.1),palette.q,true);panelSvg=`<g data-at="3">${panelSvg}${q}${label(ox+10,22,'full-covariance q',{fill:palette.q,size:21})}${label(ox+10,330,'KL = 0',{fill:palette.q,size:21})}</g>`;}
  s+=panelSvg;});
 return fig(W,H,s,'A correlated Gaussian target, a diagonal q that shrinks to fit inside it, and a full-covariance q that matches it.');
}
function mixChart(){
 const a=d.plots.mixture,xs=a.x.filter(x=>x>=-4&&x<=8),py=xs.map(x=>a.p[a.x.indexOf(x)]),qy=xs.map(u=>normalPdf(u,{mu:3,sigma:1}));
 const P=panel({W:560,H:320,L:16,R:14,xMin:-4,xMax:8,yMax:.62,ticksX:[-4,0,4,8],yAxis:false,xLabel:'u'});
 return fig(560,320,P.axes+P.area(xs,py,TINT.p)+P.line(xs,py,palette.p)+P.line(xs,qy,palette.q)+legend(16,30,[['p, the PML mixture',palette.p],['q = N(3, 1²)',palette.q]]),'A two-bump mixture target and a Gaussian q.');
}
function drawGraph(){
 return flow([
 {id:'params',x:10,y:52,w:170,h:85,label:'μ, σ',sub:'parameters',role:'q'},
 {id:'q',x:250,y:42,w:170,h:105,label:'Normal',sub:'distribution q',role:'q',shape:'circle'},
 {id:'u',x:650,y:52,w:130,h:85,label:'u',sub:'draw',role:'noise'},
 {id:'loss',x:850,y:52,w:140,h:85,label:'loss',role:'loss'},
 ],[{from:'params',to:'q'},{from:'q',to:'u',label:'sample(): no path',role:'loss',dashed:true},{from:'u',to:'loss'}],
 {width:1000,height:180,label:'Parameters define q, q supplies a draw, and the draw enters the loss; sample() cuts the gradient path from u back to q.'});
}
function reparamGraph(){
 return flow([
 {id:'mu',x:10,y:20,w:130,h:70,label:'μ',role:'q'},
 {id:'sig',x:10,y:128,w:130,h:70,label:'σ',role:'q'},
 {id:'eps',x:5,y:236,w:140,h:85,label:'ε',sub:'N(0, 1)',role:'noise',shape:'circle'},
 {id:'transform',x:290,y:116,w:200,h:95,label:'μ + σε',role:'q'},
 {id:'u',x:620,y:126,w:120,h:75,label:'u',role:'noise'},
 ],[{from:'mu',to:'transform'},{from:'sig',to:'transform'},{from:'eps',to:'transform'},
 {from:'transform',to:'u'}],{width:750,height:330,label:'Standard normal noise enters the differentiable transform mu plus sigma times epsilon.'});
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
function nnDiagram(){
 const W=640,H=330,X=[70,320,570],yH=[95,235],yO=165,rIn=34,rH=42;
 const node=(cx,cy,r,text,color,size=26)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${palette.paper}" stroke="${color}" stroke-width="3"/>`+label(cx,cy+9,text,{fill:color,size,anchor:'middle'});
 const edge=(x1,y1,x2,y2,name,lx,ly)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${palette.noise}" stroke-width="3"/>`+label(lx,ly,name,{fill:palette.noise,size:26,anchor:'middle',weight:600});
 const bias=(x,y,text)=>label(x,y,text,{fill:palette.noise,size:24,anchor:'middle',weight:600});
 let s=edge(X[0]+rIn,yO,X[1]-rH,yH[0],'w₁',185,108)+edge(X[0]+rIn,yO,X[1]-rH,yH[1],'w₂',185,242)+
  edge(X[1]+rH,yH[0],X[2]-rIn,yO,'v₁',455,108)+edge(X[1]+rH,yH[1],X[2]-rIn,yO,'v₂',455,242);
 s+=node(X[0],yO,rIn,'x',palette.data)+node(X[1],yH[0],rH,'tanh',ink,22)+node(X[1],yH[1],rH,'tanh',ink,22)+node(X[2],yO,rIn,'f',palette.q);
 s+=bias(X[1],yH[0]-54,'+ b₁')+bias(X[1],yH[1]+70,'+ b₂')+bias(X[2],yO-50,'+ c');
 s+=label(X[0],yO+62,'input',{size:20,anchor:'middle'})+label(X[2],yO+62,'output',{size:20,anchor:'middle'});
 return fig(W,H,s,'A network with one input, two tanh hidden units and one output. Its seven parameters are the weights w1, w2, v1, v2 and the biases b1, b2, c.');
}
const fxPanel=(o={})=>panel({W:600,H:330,L:52,R:14,xMin:-2.5,xMax:2.5,yMin:-2.6,yMax:2.6,ticksX:[-2,-1,0,1,2],ticksY:[-2,0,2],xLabel:'x',...o});
const dataDots=(P,r=5.5)=>b.x.map((x,k)=>`<circle cx="${P.x(x).toFixed(1)}" cy="${P.y(b.y[k]).toFixed(1)}" r="${r}" fill="${palette.data}"/>`).join('');
function priorDrawsChart(){
 const P=fxPanel({yMin:-4,yMax:4,ticksY:[-4,-2,0,2,4]});
 return fig(600,330,P.axes+b.prior_draws.map(fy=>P.line(b.grid,fy,'#9CA3AF',thin(2,.85))).join('')+label(52,30,'f(x) for twenty draws of θ from the prior',{fill:MUTED,size:21}),'Twenty functions drawn from the prior.');
}
function dataChart(){
 const P=fxPanel();
 return fig(600,330,P.axes+dataDots(P)+label(52,30,'17 training points (x, y)',{fill:palette.data,size:21}),'Seventeen training points, with a gap between x = −0.4 and x = 0.5.');
}
function twoThetasChart(){
 const P=fxPanel();
 return fig(600,330,P.axes+P.line(b.grid,b.f_B,palette.noise)+P.line(b.grid,b.f_A,palette.q)+dataDots(P)+legend(52,30,[['network with θ_A',palette.q],['network with θ_B',palette.noise]]),'With theta A the network follows the data; with theta B it does not.');
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

// ---------------------------------------------------------------- Part 2 and 3 figures
function distanceChart(){
 const xs=lin(-4,4,241),py=xs.map(u=>normalPdf(u,{mu:0,sigma:1})),qy=xs.map(u=>.5*normalPdf(u,{mu:-.8,sigma:.6})+.5*normalPdf(u,{mu:.8,sigma:.6}));
 const P=panel({W:600,H:330,L:16,R:14,xMin:-4,xMax:4,yMax:.5,ticksX:[-3,-2,-1,0,1,2,3],yAxis:false,xLabel:'u'});
 return fig(600,330,P.axes+P.area(xs,py,TINT.p)+P.line(xs,py,palette.p)+P.line(xs,qy,palette.q)+legend(16,30,[['p = N(0, 1)',palette.p],['q = two bumps',palette.q]]),'A Gaussian and a two-bump mixture with the same mean and the same variance.');
}
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
  const ox=k*590,P=panel({ox,W:560,H,L:14,R:14,T:56,B:52,xMin:-8,xMax:8,yMax:.26,ticksX:[-6,-3,0,3,6],yAxis:false,xLabel:'u'});
  const u0=pp.from.mu+pp.from.sigma*eps,u1=pp.to.mu+pp.to.sigma*eps,y=P.y(0)-24,dir=Math.sign(u1-u0);
  s+=P.axes+P.line(us,us.map(u=>normalPdf(u,pp.from)),'#C4B5FD',`style="stroke-dasharray:8 6"`)+P.line(us,us.map(u=>normalPdf(u,pp.to)),palette.q)+
   `<line x1="${P.x(u0).toFixed(1)}" y1="${y}" x2="${(P.x(u1)-dir*9).toFixed(1)}" y2="${y}" stroke="${palette.noise}" stroke-width="3"/>`+
   `<polygon points="${P.x(u1).toFixed(1)},${y} ${(P.x(u1)-dir*12).toFixed(1)},${y-7} ${(P.x(u1)-dir*12).toFixed(1)},${y+7}" fill="${palette.noise}"/>`+
   `<circle cx="${P.x(u0).toFixed(1)}" cy="${P.y(0).toFixed(1)}" r="6" fill="${palette.paper}" stroke="${palette.noise}" stroke-width="2.5"/><circle cx="${P.x(u1).toFixed(1)}" cy="${P.y(0).toFixed(1)}" r="6" fill="${palette.noise}"/>`+
   label(ox+14,30,pp.title,{fill:ink,size:22,weight:600})+label(ox+546,30,`u: ${num(u0)} → ${num(u1)}`,{fill:palette.noise,size:22,anchor:'end'});
 });
 return fig(W,H,s,'With epsilon fixed at −0.75, moving mu shifts the draw u by the same amount; widening sigma moves it by epsilon times the change.');
}
function viVsHmc(){
 const h=b.hmc,v=b.vi,P=fxPanel();
 const band=(mean,sd,fill)=>P.band(b.grid,mean.map((x,k)=>x-2*sd[k]),mean.map((x,k)=>x+2*sd[k]),fill);
 return fig(600,330,P.axes+band(h.mean,h.sd,'#0F766E30')+band(v.mean,v.sd,'#8B2CDE40')+P.line(b.grid,h.mean,palette.p)+P.line(b.grid,v.mean,palette.q,`style="stroke-dasharray:8 5"`)+dataDots(P)+
  legend(52,30,[['reference (MCMC)',palette.p],['mean-field VI',palette.q,true]]),'Predictive bands from mean-field VI and from the MCMC reference.');
}

// ================================================================ Part 1 · Bayes’ rule for a neural network
const A=b.theta_A,Bt=b.theta_B,E=b.evidence,H=b.hmc,V=b.vi;
const big=(v,k=2)=>Number(v).toLocaleString('en',{minimumFractionDigits:k,maximumFractionDigits:k}).replace('-','−');
const effective=Math.round(E.runs.reduce((s,r)=>s+r.effective_draws,0)/E.runs.length);
const lt=A.log_lik_points,lp=A.log_prior_terms;
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
S('tiny-net','bayes','A tiny network: one input, two hidden units, one output',{builds:2,
 body:cols(nnDiagram(),stack(eq(R`\begin{aligned}f_\theta(x)&=v_1\tanh(w_1x+b_1)\\&\quad+v_2\tanh(w_2x+b_2)+c\end{aligned}`),
  at(2,eq(R`\ncol{\theta}=(w_1,w_2,b_1,b_2,v_1,v_2,c)\in\mathbb R^7`)),at(2,p('Seven unknown numbers: four weights and three biases.'))),'1fr 1.1fr'),
 lines:['Our running example: one input x, two tanh hidden units and one output f.','Every weight and bias is unknown, so θ is seven numbers.'],
 ask:['How many weights? How many biases?','What would a bigger network add?'],
 expected:['Four weights (w₁, w₂, v₁, v₂) and three biases (b₁, b₂, c).','Each extra hidden unit adds a weight in, a bias and a weight out. Real networks have millions of parameters.'],
 transition:'Before any data: a prior on all seven.'});
S('priors','bayes','A prior on every parameter',{builds:2,
 body:cols(stack(eq(R`\theta_j\sim\N(0,1),\quad j=1,\dots,7`),eq(R`p(\theta)=\prod_{j=1}^{7}\N(\theta_j;\,0,\,1)`)),at(2,priorDrawsChart()),'1fr 1.15fr'),
 lines:['Before seeing data, every parameter gets its own prior: θⱼ ~ N(0, 1), independently.','Each draw of θ is a whole function, so the prior is a distribution over functions. Here are twenty.'],
 ask:['Why independent N(0, 1) priors?','What do these prior functions have in common?'],
 expected:['Simplicity: before data nothing ties one weight to another, and very large weights are unlikely.','They are smooth: flat, S-shaped or gently curved, mostly between −3 and 3.'],
 transition:'Now the data, and how well each θ explains them.'});
S('data-likelihood','bayes','The data, and how well one θ explains them',{builds:2,
 body:cols(dataChart(),stack(eq(R`y_i\mid x_i,\theta\ \sim\ \N\big(f_\theta(x_i),\,0.1^2\big)`),at(2,eq(R`p(\mathcal D\mid\theta)=\prod_{i=1}^{17}\N\big(y_i;\,f_\theta(x_i),\,0.1^2\big)`))),'1.05fr 1fr'),
 lines:['Seventeen training points with a gap in the middle. Each y is the network’s output plus Gaussian noise of size 0.1.','The likelihood of θ multiplies all seventeen densities: how well this one θ explains the whole dataset.'],
 ask:['What does the 0.1 describe?','Why a product?'],
 expected:['The noise standard deviation, assumed known here.','Given θ, the noise on different points is independent.'],
 point:`Data: ${b.data}.`,
 transition:'Compute the right-hand side of Bayes’ rule for one θ.'});
S('rhs-by-hand','bayes','The right-hand side, term by term, for one θ',{builds:3,
 body:stack(eq(R`\log p(\mathcal D\mid\theta_A)=\sum_{i=1}^{17}\log\N\big(y_i;\,f_{\theta_A}(x_i),\,0.1^2\big)`),
  eq(R`=\underbrace{${num(lt[0])}}_{i=1}+\underbrace{${num(lt[1])}}_{i=2}+\dots+\underbrace{${num(lt[16])}}_{i=17}=${num(A.log_lik)}`),
  at(2,eq(R`\log p(\theta_A)=\sum_{j=1}^{7}\log\N(\theta_j;0,1)=\underbrace{${num(lp[0])}}_{w_1}+\underbrace{${num(lp[1])}}_{w_2}+\dots+\underbrace{${num(lp[6])}}_{c}=${num(A.log_prior)}`)),
  at(3,eq(R`\log p(\mathcal D)=\log\int_{\mathbb R^7}p(\mathcal D\mid\theta)\,p(\theta)\,d\theta=\ \boldsymbol{?}`))),
 lines:['The likelihood: one log density per data point, seventeen terms, added up.','The prior: one log density per parameter, seven terms, added up.','The evidence needs every θ at once: an integral over all of ℝ⁷.'],
 ask:['Why add logs instead of multiplying densities?','Which parameter costs most under the prior?','How would you compute the third term?'],
 expected:['A product of many small numbers underflows; a sum of logs is stable, and has the same maximiser.',`The largest in size: w₂ = ${num(A.theta[1])}, costing ${num(lp[1])}.`,'Students suggest a grid, or sampling. Both are next.'],
 point:`θ_A = (${A.theta.map(v=>num(v)).join(', ')}) is the most probable θ: it maximises log p(𝒟 | θ) + log p(θ). Each likelihood term is −½ log(2π · 0.1²) − (yᵢ − f(xᵢ))² / (2 · 0.1²) = 1.38 − 50 (yᵢ − f(xᵢ))².`,
 transition:'Compare θ_A with a random θ from the prior.'});
S('two-thetas','bayes','Two settings of θ, side by side',{builds:2,
 body:cols(twoThetasChart(),at(2,table(['',m(R`\theta_A`),`${m(R`\theta_B`)}, a prior draw`],[
  [m(R`\log p(\mathcal D\mid\theta)`),num(A.log_lik,1),big(Bt.log_lik,1)],
  [m(R`\log p(\theta)`),num(A.log_prior,1),num(Bt.log_prior,1)],
  ['sum of the two',num(Number(A.log_lik.toFixed(1))+Number(A.log_prior.toFixed(1)),1),big(Number(Bt.log_lik.toFixed(1))+Number(Bt.log_prior.toFixed(1)),1)]])),'1.1fr 1fr'),
 lines:['θ_A follows the data. θ_B, one random draw from the prior, does not.','Their priors are similar; their likelihoods differ by over 2,000 nats. A ratio of two posteriors does not even need p(𝒟).'],
 ask:['Which θ explains the data better, and by how much?','How many times more probable is θ_A than θ_B after seeing the data?'],
 expected:['θ_A, by about 2,170 nats of log likelihood.','About e²¹⁷⁰ times. p(𝒟) cancels in a ratio of posteriors: it is needed only to normalise.'],
 transition:'Normalising is the hard part.'});
S('evidence-integral','bayes','The evidence needs every θ, and has no formula',{builds:2,
 body:cols(gridDots(),stack(eq(R`p(\mathcal D)=\int_{\mathbb R^7}p(\mathcal D\mid\theta)\,p(\theta)\,d\theta`),
  at(2,table(['parameters','grid points'],[['1','100'],['2','10,000'],['7, our network','10<sup>14</sup>'],['10<sup>6</sup>, a small real network','10<sup>2,000,000</sup>']]))),'1.15fr 1fr'),
 lines:['θ enters through tanh, so this integral has no formula. A grid needs about 100 points for each parameter.','Our seven parameters already need 10¹⁴ grid points; a real network needs 10^2,000,000.'],
 ask:['Why is there no formula?','Could a faster computer rescue the grid?'],
 expected:['The integrand is not a Gaussian or any other form we can integrate: θ passes through tanh.','No: the count grows exponentially with the number of parameters.'],
 transition:'Sampling instead of a grid?'});
S('evidence-mc','bayes','Sampling from the prior: a few lucky draws carry everything',{builds:2,
 body:cols(evidenceMcChart(),stack(eq(R`p(\mathcal D)\approx\frac1S\sum_{s=1}^{S}p(\mathcal D\mid\theta_s)`),eq(R`\theta_s\sim p(\theta)`),
  at(2,p(`Ten draws: off by about 1,000 nats. Ten million: about ${effective} draws carry the estimate, ${num(E.estimate)}.`))),'1.05fr 1fr'),
 lines:['Average the likelihood over draws from the prior: four seeds, up to ten million draws each.','Almost every prior draw explains the data terribly. Seven parameters just about work; more parameters or more data make it hopeless.'],
 ask:['Why are the small-S estimates so low?','What happens with a million parameters?'],
 expected:['Prior draws almost never land where the likelihood is high, like θ_B; the rare good draws dominate the average.','The chance that a prior draw lands near the posterior shrinks exponentially; no feasible S is enough.'],
 point:`Effective number of draws, out of 10⁷: ${E.runs.map(r=>Math.round(r.effective_draws)).join(', ')}. The largest single draw carries ${E.runs.map(r=>Math.round(100*r.largest_share)+'%').join(', ')} of the total.`,
 transition:'Why do we want the posterior at all?'});
S('predictive','bayes','What we want in the end: predictions with error bars',{builds:2,
 body:cols(predictiveChart(),stack(eq(R`\begin{aligned}p(y^*\mid x^*,\mathcal D)&=\int p(y^*\mid x^*,\theta)\\&\qquad\times\pcol{p(\theta\mid\mathcal D)}\,d\theta\end{aligned}`),
  at(2,p('Average every network’s prediction, weighted by the posterior. The band widens where there are no data.'))),'1.05fr 1fr'),
 lines:['The posterior is a means to an end. The posterior predictive averages every network’s prediction at a new x*.','Here it comes from a long MCMC run, used only as a reference. The band widens beyond the data and in the gap.'],
 ask:['Where should the model be least sure?','How would you compute this integral?'],
 expected:['Far from the data: beyond both ends, and in the gap.','Average f_θ(x*) over samples of θ from the posterior, which we cannot draw directly. Yet.'],
 point:`Reference: Hamiltonian Monte Carlo, 4 chains of 2,000 samples, acceptance about ${num(H.acceptance.reduce((a,v)=>a+v,0)/H.acceptance.length)}. Predictive sd ${num(H.sd[0])} at x = −2.5, about ${num(Math.min(...H.sd))} in the data.`,
 transition:'How else could we approximate the posterior?'});
S('three-ways','bayes','Three ways to approximate a posterior',{builds:3,
 body:grid3(stack(threeWay('laplace'),small(`A Gaussian at the highest peak: N(${num(st.three.laplace[0],1)}, ${num(st.three.laplace[1],1)}²). It never sees the second bump.`)),
  at(2,stack(threeWay('mcmc'),small(`20,000 random-walk steps. Their mean, ${num(st.three.mcmc.mean)}, matches the true ${num(st.three.true_mean)}.`))),
  at(3,stack(threeWay('vi'),small(`The closest Gaussian, found by optimisation: N(${num(st.three.vi[0])}, ${num(st.three.vi[1])}²).`)))),
 lines:['Laplace: fit a Gaussian to the curvature at the highest peak.','MCMC: a random walk whose visits follow the posterior. Right in the long run, but slow.','Variational inference: choose the closest member of a simple family, by optimisation. That is today.'],
 ask:['What does Laplace miss here?','What does MCMC cost?','What does VI trade away?'],
 expected:['The second bump: it only looks at the peak.','Many steps, each evaluating the model on the data; hard to scale to large models and datasets.','Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.'],
 point:'Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.',
 transition:'Today: variational inference.'});
S('vi-idea','bayes','Variational inference: choose q, make it close to the posterior',{builds:3,
 body:stack(eq(R`q_\psi(\theta)=\prod_{j=1}^{7}\N(\theta_j;\,\mu_j,\,\sigma_j^2),\qquad\psi=(\mu_1,\dots,\mu_7,\ \sigma_1,\dots,\sigma_7)`),at(2,qFactors()),
  at(3,eq(R`\psi^*=\arg\min_\psi\ \text{distance}\big(q_\psi(\theta),\ \pcol{p(\theta\mid\mathcal D)}\big)`))),
 lines:['Pick a simple family: one Gaussian per parameter, each with its own mean and width. ψ is fourteen numbers.','One bell per parameter. Together they define q(θ), and drawing θ from q is easy.','Tune ψ until q is as close as possible to the posterior. Two questions: what is “close”, and how do we tune without the posterior?'],
 ask:['How many numbers does q have?','Why is drawing θ from q easy?','What do we need before we can minimise anything?'],
 expected:['Fourteen: a mean and a width for each of the seven parameters.','Each θⱼ is an independent Gaussian draw.','A number for the distance between two distributions. That is Part 2.'],
 transition:'Part 2: how far apart are two distributions?'});

// ================================================================ Part 2 · How far apart are two distributions?
S('distance-ideas','kl','How far apart are two distributions?',{builds:2,
 body:act(2,cols(distanceChart(),stack(table(['',m(R`p`),m(R`q`)],[['mean','0','0'],['variance','1','1']]),at(2,p('Same mean, same variance, different distributions. We need one number that compares the whole shapes.'))),'1.1fr 1fr')),
 lines:['Comparing means and variances is a start.','But here both match, and the shapes still differ. We want a number that compares the distributions everywhere.'],
 ask:['Are p and q the same distribution?','What would you compare instead?'],
 expected:['No: one bump against two.','The densities at every point: how many times more likely each u is under one than under the other, averaged. That leads to KL.'],
 point:'q = ½·N(−0.8, 0.6²) + ½·N(0.8, 0.6²): mean 0 and variance 0.6² + 0.8² = 1.',
 transition:'Build that number from a familiar example.'});
S('fruit','kl','Two surveys: how different are they?',{builds:2,
 body:stack(cols(fruitBars(),stack(p('Favourite fruit in a class, last year (q) and this year (p).'),at(2,p('Bananas fell from 40% to 10%. Clementines rose from 10% to 40%.'))),'1.35fr 1fr')),
 lines:['We want one number for how much the class changed.','A fourfold fall and a fourfold rise. Should they count the same?'],
 ask:['How would you score the difference between these two surveys?','Bananas ÷4, clementines ×4: same size of change?'],
 expected:['Students propose differences of percentages. Note that apples, 50% both years, should contribute nothing.','By ratio, yes: both are a factor of 4. That suggests comparing ratios.'],
 point:'p is the survey we weight by (this year); q is the reference (last year).',
 transition:'Compare by ratios.'});
S('ratios-logs','kl','Ratios say “how many times”; logs make up and down cancel',{builds:2,
 body:cols(table(['fruit','p','q',m(R`p/q`),m(R`\log(p/q)`)],st.fruit.names.map((name,i)=>[name,num(st.fruit.p[i],1),num(st.fruit.q[i],1),st.fruit.ratio[i]===.25?'¼':num(st.fruit.ratio[i],0),at(2,num(st.fruit.log_ratio[i]))])),
  stack(ratioLine(false),at(2,ratioLine(true))),'1fr 1fr'),
 lines:['Ratios p/q: 1, ¼ and 4. But on the ratio scale, ×4 lands 3 above 1 and ÷4 only 0.75 below.','Take logs: log 4 = +1.39 and log ¼ = −1.39. Equal and opposite.'],
 ask:['On the ratio scale, which change looks bigger?','After taking logs?'],
 expected:['The rise: +3 against −0.75, though both are fourfold.','Equal and opposite, ±1.39. Logs turn “times” into “plus”.'],
 transition:'Now combine the three log ratios into one number.'});
S('weight','kl','Weight by how common each fruit is now: that is KL',{builds:2,
 body:cols(contributionBars(),stack(eq(R`\KL(\pcol p\,\|\,\qcol q)=\sum_x \pcol{p(x)}\,\log\frac{\pcol{p(x)}}{\qcol{q(x)}}`),
  at(2,eq(R`=0.5(0)+0.1(-1.39)+0.4(1.39)`)),at(2,eq(R`\approx${num(st.fruit.kl)}\ \text{nats}`))),'1fr 1.1fr'),
 lines:['Average the log ratios, weighting each fruit by p(x), its share this year.','The rare banana barely counts; the common clementine dominates. Total 0.42 nats: the Kullback–Leibler divergence.'],
 ask:['Why weight by p rather than q?','What is the KL if the surveys agree?'],
 expected:['That choice defines KL(p‖q): the first argument does the weighting. Swapping gives a different number.','Zero: every ratio is 1, every log is 0.'],
 point:'Natural log gives nats. Base 2 gives bits, which the next slide uses.',
 transition:'The same number has a second meaning, in bits.'});
S('bits','kl','KL is the extra message length of using the wrong code',{builds:2,
 body:stack(table(['weather',m(R`p`),m(R`q`),'code built for p','code built for q'],st.bits.names.map((name,i)=>[name,num(st.bits.p[i]),num(st.bits.q[i]),`<code>${st.bits.right_code[i]}</code>`,`<code>${st.bits.wrong_code[i]}</code>`])),
  eq(R`\text{average length under }p:\quad\text{code for }p=${num(st.bits.entropy)}\text{ bits},\qquad\text{code for }q=${num(st.bits.cross_entropy)}\text{ bits}`),
  at(2,eq(R`${num(st.bits.cross_entropy)}-${num(st.bits.entropy)}=${num(st.bits.kl)}\text{ bits}=\KL(p\|q)\ \text{with}\ \log_2`))),
 lines:['Send each day’s weather in bits. A code built for the true frequencies p needs 1.50 bits a day.','A code built for the wrong frequencies q needs 1.75. The extra 0.25 bits a day are exactly KL(p‖q), in base 2.'],
 ask:['Why does sunny get the shortest code?','Where do the extra 0.25 bits come from?'],
 expected:['Sunny is the most common, so a short code for it saves the most bits on average.','Sunny is half of all days, and the wrong code spends 2 bits on it. KL measures that waste.'],
 transition:'Cross-entropy is this “wrong code” length, and you have minimised it many times.'});
S('used-before','kl','You have minimised KL before: in every cross-entropy loss',{builds:3,
 body:stack(eq(R`\underbrace{-\sum_x p(x)\log q(x)}_{\text{cross-entropy }H(p,q)}=\underbrace{-\sum_x p(x)\log p(x)}_{\text{entropy }H(p)}+\KL(p\|q)`),
  at(2,eq(R`\text{classifier:}\quad\min_\theta H(y,\,q_\theta)\ \Longleftrightarrow\ \min_\theta\KL(y\,\|\,q_\theta)`)),
  at(3,eq(R`\text{maximum likelihood:}\quad-\tfrac1N\textstyle\sum_i\log p_\theta(x_i)\ \approx\ H(p_{\text{data}})+\KL(p_{\text{data}}\,\|\,p_\theta)`))),
 lines:['Cross-entropy splits into entropy plus KL: the 1.75 bits were 1.50 + 0.25.','A classifier’s cross-entropy loss: the labels’ entropy is fixed, so training minimises a KL.','Maximum likelihood minimises KL(data ‖ model). Variational inference is the third use.'],
 ask:['Which term of the split depends on q?','What does your classifier’s loss minimise?','So what is new in VI?'],
 expected:['Only the KL. H(p) is fixed by p.','KL(labels ‖ predictions), plus a constant.','The direction. VI puts the approximation first, KL(q‖p). Act 1 ends with why.'],
 transition:'What does a KL of 0.19 actually feel like?'});
S('coin-evidence','kl','A KL value is how fast evidence piles up',{builds:2,
 body:cols(evidenceChart(),stack(eq(R`\KL\big(\mathrm{Bern}(0.8)\,\|\,\mathrm{Bern}(0.5)\big)\approx${num(st.coinkl.kl)}\ \text{nats}`),
  at(2,eq(R`\frac{p(\mathcal D\mid0.8)}{p(\mathcal D\mid0.5)}\approx e^{\,${num(st.coinkl.kl)}\,n}`)),at(2,small(`The reverse direction, KL(0.5‖0.8) = ${num(st.coinkl.reverse)}, is a different number.`))),'1fr 1.15fr'),
 lines:['A coin lands heads 80% of the time. Each toss favours “0.8” over “fair” by 0.19 nats on average. After 30 tosses, by what factor?','About e^(30 × 0.19) ≈ 325. KL is the rate at which evidence piles up: 6.9× after 10 tosses, 325× after 30.'],
 ask:['Predict: after 30 tosses, how many times more likely are the data under 0.8 than under 0.5?','Why does a small KL eventually matter so much?'],
 expected:['Most guess far too low. It is about 325×.','Log evidence adds up toss by toss, so the ratio grows exponentially with n.'],
 point:`PAUSE. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of ${num(st.coinkl.simulated_mean_llr_30,3)}, against 30 × KL = ${num(30*st.coinkl.kl,3)}.`,
 transition:'Sums become integrals for continuous unknowns.'});
S('densities','kl','For densities, sums become integrals, and KL ≥ 0',{builds:2,
 body:cols(stack(eq(R`\KL(\qcol q\,\|\,\pcol p)=\int \qcol{q(u)}\,\log\frac{\qcol{q(u)}}{\pcol{p(u)}}\,du`),
  at(2,eq(R`-\KL(q\|p)=\E_q\Big[\log\frac{p}{q}\Big]\le\E_q\Big[\frac{p}{q}-1\Big]`)),at(2,eq(R`=\int p\,du-1=0`))),tangentChart(),'1.25fr 1fr'),
 lines:['For a continuous unknown u the sum becomes an integral. Same recipe: weight the log ratio by the first argument.','log t never rises above its tangent t − 1. Inside the average that gives KL ≥ 0, with 0 only when q = p.'],
 ask:['Which distribution supplies the weights in KL(q‖p)?','Why can’t KL be negative?'],
 expected:['q, the first argument.','log t ≤ t − 1 for every ratio t = p/q, and the average of p/q under q is ∫p = 1.'],
 point:'Individual log ratios can be negative; only the weighted average is guaranteed nonnegative. That returns with sampling in Act 3.',
 transition:'Closed forms for the distributions we will meet.'});
S('kl-formulas','kl','Closed forms for simple distributions',{builds:2,
 body:stack(table(['distributions',m(R`\KL(p\,\|\,q)`),'example'],[
  ['Bernoulli(a), Bernoulli(b)',m(R`\displaystyle a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`),'coin, 0.8 vs 0.5: 0.19'],
  ['categorical',m(R`\displaystyle \sum_k p_k\log\frac{p_k}{q_k}`),'fruit surveys: 0.42'],
  ['Gaussians',m(R`\displaystyle \log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12`),'derived next'],
  ['diagonal Gaussians','sum of one-dimensional terms','our q against the prior']]),
  at(2,p('A mixture, or a posterior, has no closed form. Then we estimate the average by sampling.'))),
 lines:['For simple pairs, the weighted average of log ratios has a closed form. These are the ones we will use.','Anything else, such as a mixture or a posterior, has no formula; then we sample.'],
 ask:['Which row will our network’s ELBO use?','Which of these did we already compute by hand?'],
 expected:['The diagonal-Gaussian row: KL from q(θ) to the N(0, 1) prior, summed over the seven parameters.','Bernoulli (the coin, 0.19) and categorical (the fruit, 0.42).'],
 point:'The Gaussian row is KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)).',
 transition:'Derive the Gaussian row.'});
S('gaussian-derivation','kl','Deriving the Gaussian formula',{builds:3,
 body:stack(eq(R`q=\N(\mu,\sigma^2),\quad p=\N(m,s^2):\qquad\log\frac{q(u)}{p(u)}=\log\frac{s}{\sigma}+\frac{(u-m)^2}{2s^2}-\frac{(u-\mu)^2}{2\sigma^2}`),
  at(2,eq(R`\E_q[(u-m)^2]=\sigma^2+(\mu-m)^2,\qquad\E_q[(u-\mu)^2]=\sigma^2`)),
  at(3,eq(R`\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12`))),
 lines:['Write the log ratio of the two densities. The √(2π) terms cancel.','Average it under q. Two moments are all we need.','The result: a width term, a squared shift of the centres, and −½.'],
 ask:['Which constants cancel?','Why is E_q[(u − m)²] = σ² + (μ − m)²?','When is the KL zero?'],
 expected:['The common log √(2π) terms, leaving the log ratio of the widths.','Write u − m = (u − μ) + (μ − m); the cross term averages to zero under q.','Only when μ = m and σ = s.'],
 transition:'Use the formula on the PML vote.'});
S('compare-formula','kl','Use the formula: which Gaussian is closer to p?',{builds:2,
 body:cols(normalPlot(['q1','q2']),stack(eq(R`\pcol{p=\N(1,4^2)}`),table(['','substitute','KL'],[
  [`${m(R`q_1=\N(0,2^2)`)}`,m(R`\log2+\frac{4+1}{32}-\frac12`),at(2,num(n.candidates.q1.kl))],
  [`${m(R`q_2=\N(1,3^2)`)}`,m(R`\log\frac43+\frac{9+0}{32}-\frac12`),at(2,num(n.candidates.q2.kl))]])),'1.25fr 1fr'),
 lines:['The PML vote: which is closer to p, q₁ or q₂? Substitute into the formula.','KL(q₁‖p) = 0.35 nats and KL(q₂‖p) = 0.07. The formula settles the vote.'],
 ask:['Vote first: q₁ or q₂?','Which term made q₁ worse?'],
 expected:['Most pick q₂: same centre and a closer width.','Both: q₁ is off-centre, (0 − 1)² = 1, and narrower, log 2 against log 4/3.'],
 point:'PAUSE (PML L23 08:30).',
 transition:'Where does each number come from? Watch it as an area.'});
S('kl-area','kl','KL is an area you can watch',{
 body:widget('klarea','KL as the net area under its integrand, with sliders for q'),
 lines:['Purple area adds and red area subtracts; the net area is the KL. Drag q, or flip the direction.'],
 ask:['Make q narrow and centred on p. What happens in each direction?'],
 expected:['KL(q‖p) stays moderate: q only visits places where p is reasonable. KL(p‖q) explodes: p’s tails land where q is nearly zero.'],
 point:'Click q₁, then q₂, then flip to KL(p‖q). Then shrink σ in each direction.',
 transition:'What if there is no formula at all?'});
S('kl-sampling','kl','No formula needed: estimate KL by sampling',{builds:2,
 body:stack(eq(R`\KL(q\|p)=\E_{u\sim q}\big[\log q(u)-\log p(u)\big]\approx\frac1S\sum_{i=1}^{S}\big[\log q(u_i)-\log p(u_i)\big]`),
  cols(code('u = q.sample((S,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nest = (log_q - log_p).mean()\nexact = kl_divergence(q, p)'),
  at(2,table(['','KL(q₂‖p)'],[['exact formula',num(mc.exact_kl,3)],['estimate, S = 10,000',num(mc.repeats[3].estimates[0],3)]])),'1fr 1fr')),
 lines:['Draw u from q, score each draw with both log densities, and average. No formula needed.','Checked on a pair that has a formula: 0.067 from 10,000 draws, against the exact 0.069.'],
 ask:['Which distribution supplies the draws?','Should the estimate match exactly?'],
 expected:['q, the first argument: it does the weighting.','No: it is random. It is close, and gets closer with more draws.'],
 point:'q₂ = N(1, 3²) against p = N(1, 4²), seed 1000.',
 transition:'Watch the estimate settle as draws arrive.'});
S('mc-live','kl','Watch the estimate settle',{
 body:widget('mclive','A Monte Carlo estimate of KL, built up one batch of draws at a time'),
 lines:['Add draws from q and watch the running average of log q − log p settle on the exact 0.349.'],
 ask:['Press +10 a few times. Is every estimate above the exact value?'],
 expected:['No: with few draws the estimate lands on either side, and can even be negative. The spread shrinks like 1/√S.'],
 point:'q = N(0, 2²) and p = N(1, 4²), the q₁ pair. Draws are seeded, so the sequence replays identically.',
 transition:'Part 3: use the score to find the best q.'});

// ================================================================ Part 3 · Find the closest q
S('six-candidates','fit','Six candidates: which q is closest to p?',{builds:2,
 body:act(3,sixCandidates()),
 lines:['Six Gaussians against the same target p = N(1, 4²). Vote: which is closest?',`Smallest KL: N(1, 3²) at ${num(candKL[1])}, then N(2, 5²) at ${num(candKL[5])}. Could another q do better?`],
 ask:['Vote: which q is closest?','Could another (μ, σ) do better?'],
 expected:['Most pick N(1, 3²) or N(2, 5²). The formula says N(1, 3²), narrowly.','Yes: N(1, 4²), which is p itself, has KL 0. We need a way to search.'],
 point:'PAUSE. All six KLs come from the closed-form Gaussian formula.',
 transition:'Search systematically: a grid.'});
S('grid-search','fit','Search a grid of means and widths',{builds:2,
 body:cols(klGrid({bestAt:2}),stack(p(`Evaluate ${m(R`\KL(q\|p)`)} at every point of a 51 × 40 grid of means and widths: 2,040 evaluations of the formula.`),
  at(2,p(`Best grid point: q = p. But our network’s q has 14 numbers; a grid with 40 values each needs ${m(R`40^{14}\approx10^{22}`)} evaluations.`))),'1.05fr 1fr'),
 lines:['Try every (μ, σ) on a grid and colour each by its KL. Darker means further from p.','The best grid point is N(1, 4²), which is p. For the network’s fourteen numbers, a grid is hopeless.'],
 ask:['Where do you expect the minimum?','How many evaluations for fourteen parameters?'],
 expected:['At μ = 1 and σ = 4, where q = p.','40¹⁴ ≈ 2.7 × 10²², even with only 40 values per parameter.'],
 transition:'Instead of evaluating everywhere, walk downhill.'});
S('walk-downhill','fit','Instead of a grid, walk downhill',{builds:2,
 body:cols(klGrid({bestAt:1,pathAt:2}),stack(eq(R`\psi^*=\arg\min_\psi\KL(q_\psi\,\|\,p)`),eq(R`\psi\leftarrow\psi-\eta\,\nabla_\psi\KL(q_\psi\,\|\,p)`),
  at(2,p('Gradient descent from N(4, 0.5²): the teal path. About a hundred steps, and the same recipe works for millions of parameters.'))),'1.05fr 1fr'),
 lines:['Cast the search as optimisation: follow the gradient of KL with respect to q’s parameters.','From N(4, 0.5²) the optimiser walks downhill to p. It needs only gradients, which autograd provides.'],
 ask:['What does the gradient tell us at each point?','Why does this scale where the grid did not?'],
 expected:['The direction in (μ, σ) that lowers KL fastest.','Each step costs one gradient evaluation, whatever the number of parameters.'],
 point:'The path is the recorded Adam run from the next two slides, drawn on the same grid.',
 transition:'The same search in code.'});
S('fit-code','fit','The same search in code',{builds:2,
 body:cols(code('mu  = Parameter(tensor(4.0))\nrho = Parameter(tensor(log(0.5)))\nopt = Adam([mu, rho], lr=0.05)\nfor step in range(200):\n    q = Normal(mu, rho.exp())\n    loss = kl_divergence(q, p)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()'),
  stack(eq(R`\sigma=\exp(\rho)>0\ \text{ for every }\rho`),at(2,p('Rebuild q, score it, differentiate, update. The same loop fits our network in Part 5.'))),'1.25fr 1fr'),
 lines:['Store ρ = log σ, so that σ = exp(ρ) stays positive whatever an update does.','Rebuild q, score it, differentiate, update. The same five lines fit our network in Part 5.'],
 ask:['What stays fixed, and what does the optimiser change?','Why not update σ directly?'],
 expected:['p stays fixed; μ and ρ change.','A step could make σ zero or negative, which is not a valid Normal.'],
 transition:'Press play.'});
S('olive','fit','Press play: q slides onto p',{builds:2,
 body:widget('olive','q moving onto p under gradient descent'),
 lines:['Start q far away, at N(4, 0.5²). Predict: which reaches the target first, the centre or the width?','The width gets there first and overshoots a little; the centre slides in by step 100. Then KL ≈ 0.'],
 ask:['Predict: does the centre or the width arrive first?','Why can both move at once?'],
 expected:[`Students split. The width reaches 4 around step 45, overshoots to ${num(Math.max(...st.olive.frames.map(r=>r[2])))}, then settles; μ arrives by step 100.`,'Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.'],
 point:'PAUSE (PML L23 12:13). Adam, learning rate 0.05, analytic KL. Notebook section 3.',
 transition:'One dimension is easy. What happens with two correlated unknowns?'});
S('two-d','fit','In 2-D, a q that cannot tilt shrinks instead',{builds:3,
 body:cols(twoD(),stack(eq(R`\pcol{p=\N\Big(0,\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}\Big)}`),
  at(2,eq(R`\qcol{q=\N\big(0,\operatorname{diag}(s^2,s^2)\big)}`)),at(2,eq(R`s=${num(st.twod.analytic_sd)},\quad\KL=${num(st.twod.analytic_kl)}`)),
  at(3,eq(R`\qcol{q=\N(0,LL^\top)}:\ \KL=0`))),'1.15fr 1fr'),
 lines:['A correlated target: both coordinates have width 1, correlation 0.9. A diagonal q cannot tilt. Predict its widths.','It shrinks to fit inside the ellipse: widths 0.44, not 1. Mean-field VI underestimates uncertainty.','Let q tilt with a full covariance, Σ = LLᵀ, and the KL drops to 0.'],
 ask:['Predict the diagonal q’s widths: 1, more or less?','Why shrink rather than spread?','What does the full covariance cost?'],
 expected:['Less: about 0.44, which is √(1 − 0.9²).','KL(q‖p) heavily penalises q mass where p is tiny, which is off the diagonal ridge.','More parameters: d(d + 1)/2 for L instead of d. For a million weights that is far too many.'],
 point:'PAUSE. Measured fit: both widths 0.4359, KL 0.8304, matching √(1 − ρ²) and −½ log(1 − ρ²).',
 transition:'Now a target with no formula.'});
S('mixture-no-formula','fit','PyTorch has no formula for this KL. The definition still works.',{builds:2,
 body:stack(cols(mixChart(),stack(code('kl_divergence(q, mixture)'),at(2,code('NotImplementedError:\nNo KL(p || q) is implemented\nfor p type Normal and\nq type MixtureSameFamily'))),'1fr 1fr')),
 lines:['Swap the target for the PML two-bump mixture and ask PyTorch for the KL to a Gaussian q.','No formula. But KL is an average under q, and averages can be estimated by sampling.'],
 ask:['What would kl_divergence need for this pair?','Is the KL undefined, then?'],
 expected:['A closed-form expression for a Gaussian against a mixture. There isn’t one.','No. It is a perfectly good integral; we just cannot write it in closed form.'],
 point:`Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch ${d.provenance.torch}).`,
 transition:'Estimate the average by sampling.'});
S('now-optimise','fit','Now optimise it: the code runs, and q does not move',{builds:2,
 body:stack(cols(code('u = q.sample((5000,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-sample','A fit that uses sample(): q stays where it started'),'1fr 1.06fr')),
 lines:['Use the estimate as the loss and run the same loop on the mixture. Predict: does q move?',`No error and no warning, and q stays at N(6, 0.1²). After 400 steps the KL is still about ${Math.round(st.mixfit.sample.at(-1)[4])}.`],
 ask:['Predict: where is q after 400 steps?','The code runs. What could be wrong?'],
 expected:['Most expect it to slide onto the mixture. It does not move.','The gradient: it must be missing a piece.'],
 point:'PAUSE (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.',
 transition:'Look at the gradient on a case we can solve.'});
S('missing-path','fit','The gradient that would move q never reaches it',{builds:2,
 body:stack(drawGraph(),at(2,table(['',m(R`\partial\KL/\partial\mu`),m(R`\partial\KL/\partial\sigma`)],[
  ['through sample()',num(g.detached_sample_grads[0],4),num(g.detached_sample_grads[1],4)],
  ['true gradient',num(g.analytic_grads[0],3),num(g.analytic_grads[1],3)]]))),
 lines:['sample() hands back plain numbers. Autograd cannot see how u would shift if μ or σ changed.','What survives averages to zero: 0.0013 and −0.0003, where the true gradient is −0.063 and −0.375.'],
 ask:['Which arrow can the gradient not travel along?','Why isn’t the surviving gradient exactly zero?'],
 expected:['The sample arrow from q to u.','log q(u) still contains μ and σ explicitly. That part averages to zero over draws; what remains is noise.'],
 point:'Gaussian pair q = N(0, 2²), p = N(1, 4²), 50,000 shared draws. A score-function estimator is another valid fix (optional slide).',
 transition:'Give the gradient a path.'});
S('reparam','fit','Move the randomness outside: u = μ + σε',{builds:3,
 body:cols(reparamGraph(),stack(eq(R`\ncol{\epsilon}\sim\N(0,1),\qquad u=\mu+\sigma\ncol{\epsilon}`),
  at(2,eq(R`\epsilon=-0.75,\quad\mu=0,\quad\sigma=2`)),
  at(3,eq(R`u=-1.5`)),at(3,eq(R`\frac{\partial u}{\partial\mu}=1,\qquad\frac{\partial u}{\partial\sigma}=\epsilon=-0.75`))),'1fr 1fr'),
 lines:['Draw the noise ε from a fixed N(0, 1), then shift and scale it. u has the same distribution, but is now a function of μ and σ.','Pause: with ε = −0.75, μ = 0 and σ = 2, find u, ∂u/∂μ and ∂u/∂σ.','u = −1.5, ∂u/∂μ = 1 and ∂u/∂σ = ε = −0.75. The gradient now has a path. In PyTorch this is rsample().'],
 ask:['Is u still distributed as N(μ, σ²)?','Compute u and both derivatives.','Can ∂u/∂σ be negative?'],
 expected:['Yes: shifting and scaling a standard Normal gives N(μ, σ²).','u = 0 + 2(−0.75) = −1.5; ∂u/∂μ = 1; ∂u/∂σ = ε = −0.75.','Yes, whenever ε < 0: widening q pushes this draw further left.'],
 point:'PAUSE (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.',
 transition:'Picture what a fixed ε does.'});
S('reparam-picture','fit','What a fixed ε does when μ or σ changes',{builds:2,
 body:stack(reparamPicture(),at(2,eq(R`\frac{\partial u}{\partial\mu}=1,\qquad\frac{\partial u}{\partial\sigma}=\epsilon=-0.75`))),
 lines:['Keep the same noise draw, ε = −0.75. Shift μ by 1: the draw moves right by exactly 1.','Widen σ by 1: the draw moves by ε = −0.75, further out on its side. These are the derivatives the gradient needs.'],
 ask:['If μ rises by 1, how far does u move?','If σ rises by 1?'],
 expected:['By exactly 1.','By ε = −0.75: to the left, because this ε is negative.'],
 transition:'Change one word in the code.'});
S('rsample-fit','fit','Change one word: rsample() fits the mixture',{builds:2,
 body:cols(code('# one word changed\nu = q.rsample((5000,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-rsample','A fit that uses rsample(): q moves onto the mixture'),'1fr 1.06fr'),
 lines:['Same loop, same start, with q.rsample. Press play.',`q settles at N(${num(st.mixfit.rsample.at(-1)[1])}, ${num(st.mixfit.rsample.at(-1)[2])}²), KL = ${num(st.mixfit.rsample.at(-1)[4],3)}: the best single Gaussian for this mixture. No formula was needed.`],
 ask:['Predict where q settles.','Why doesn’t the KL reach 0?'],
 expected:['Over the mixture’s mass, between and around the two bumps.','No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct optimisation.'],
 point:'Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Acts 3 and 4.',
 transition:'One more question: which direction of KL?'});
S('fwdrev','fit','The price: reverse KL picks one bump',{builds:2,
 body:widget('fwdrev','Forward and reverse KL fits of one Gaussian to two bumps'),
 lines:['Fit one Gaussian to two narrow bumps, three times. Predict where each q settles.','KL(p‖q) spreads over both bumps. KL(q‖p) locks onto whichever bump is nearer its start, and reports it confidently.'],
 ask:['Predict each final q.','Which fit would make you overconfident?'],
 expected:['Forward: wide, centred between the bumps. Reverse: one bump, depending on the start.','The reverse fits: each reports one bump with a narrow spread.'],
 point:`PAUSE. Narrow bumps: forward KL ${num(st.fwdrev['0.5'].fwd.at(-1)[3])}; reverse ${num(st.fwdrev['0.5'].revl.at(-1)[3])} (= −log 0.7) from the left and ${num(st.fwdrev['0.5'].revr.at(-1)[3])} (= −log 0.3) from the right. Then switch to wide bumps: overlapping bumps give one broad reverse-KL fit.`,
 transition:'Part 4: back to Bayes.'});

// ================================================================ Part 4 · Back to Bayes
S('can-we','elbo','Back to Bayes: minimise KL to the posterior directly?',{builds:2,
 body:act(4,eq(R`\KL\big(q_\psi(\theta)\,\|\,\pcol{p(\theta\mid\mathcal D)}\big)=\E_{\theta\sim q}\big[\log q_\psi(\theta)-\log\pcol{p(\theta\mid\mathcal D)}\big]`),
  at(2,table(['the loss needs','available?'],[['draws θ ~ q','yes: rsample, as in Part 3'],[m(R`\log q_\psi(\theta)`),'yes'],[m(R`\log p(\theta\mid\mathcal D)=\log p(\mathcal D\mid\theta)+\log p(\theta)-\log\dcol{p(\mathcal D)}`),'no: needs p(𝒟)']]))),
 lines:['The plan from Part 1: make q close to the posterior. Use KL(q ‖ posterior): it averages under q, which we can sample.','Everything is available except log p(θ | 𝒟), which needs the evidence p(𝒟): the integral we cannot do.'],
 ask:['Why this direction, KL(q‖p), and not KL(p‖q)?','Which piece is missing?'],
 expected:['KL(p‖q) averages under the posterior, so it needs posterior samples: the problem itself. KL(q‖p) averages under q, which we choose.','log p(θ | 𝒟): it contains log p(𝒟).'],
 transition:'Substitute Bayes’ rule and see what survives.'});
S('bayes-sub','elbo','Substitute Bayes’ rule and see what survives',{builds:3,
 body:stack(eq(R`\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)=\E_q\big[\log q(\theta)-\log p(\theta\mid\mathcal D)\big]`),
  at(2,eq(R`=\E_q\big[\log q(\theta)-\log p(\mathcal D\mid\theta)-\log p(\theta)\big]+\dcol{\log p(\mathcal D)}`)),
  at(3,eq(R`\dcol{\log p(\mathcal D)}=\underbrace{\E_q\big[\log p(\mathcal D\mid\theta)+\log p(\theta)-\log q(\theta)\big]}_{\qcol{\ELBO(q)}}+\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)`))),
 lines:['Pause: substitute log p(θ | 𝒟) = log p(𝒟 | θ) + log p(θ) − log p(𝒟). What happens to log p(𝒟)?','It does not depend on θ, so it leaves the average as a constant.','Rearrange: log p(𝒟) = ELBO + KL. The ELBO needs only the likelihood, the prior and q.'],
 ask:['Substitute and simplify.','Why can log p(𝒟) leave the expectation?','Which side can we compute?'],
 expected:['−log p(θ | 𝒟) = −log p(𝒟 | θ) − log p(θ) + log p(𝒟).','It is a constant: it depends on neither θ nor q.','The ELBO: draws from q, the likelihood, the prior and log q.'],
 point:'PAUSE (PML L23 47:15). Students do the substitution on paper before build 2. KL ≥ 0, so the ELBO is a lower bound on log p(𝒟), and maximising it minimises the KL.',
 transition:'What does the ELBO look like for our network?'});
S('elbo-two-terms','elbo','The ELBO for our network: fit the data, stay near the prior',{builds:2,
 body:stack(eq(R`\ELBO(q)=\E_q\big[\log p(\mathcal D\mid\theta)\big]+\E_q\big[\log p(\theta)-\log q(\theta)\big]`),
  eq(R`=\underbrace{\E_q\big[\log p(\mathcal D\mid\theta)\big]}_{\text{fit the data}}-\underbrace{\KL\big(q(\theta)\,\|\,p(\theta)\big)}_{\text{stay near the prior}}`),
  at(2,p('The second term is the diagonal-Gaussian KL from Part 2: a formula, seven terms. The first is a Monte Carlo average over networks sampled with rsample.'))),
 lines:['Group the terms: the ELBO rewards fitting the data and charges for moving q away from the prior.','The KL to the prior has a formula; the data term is estimated by sampling θ from q, with the reparameterisation trick.'],
 ask:['Which term pulls q towards a narrow, well-fitting θ?','Which term needs sampling, and why?'],
 expected:['The data term. The KL to the N(0, 1) prior pulls the other way: wide, near zero.','The data term: the network is nonlinear in θ, so there is no formula for its average.'],
 transition:'Check the identity on a model where p(𝒟) is known.'});
S('elbo-slider','elbo','Check the identity where p(𝒟) is known: a coin',{
 body:widget('elboslider','Beta q sliders with the ELBO and KL split of the log evidence'),
 lines:['A coin with a Beta(5, 3) prior and 17 heads in 30 tosses has an exact posterior and an exact p(𝒟). Drag q: the ELBO and the KL trade off, and their sum never changes.'],
 ask:['Find the q with the highest ELBO. What is its KL?'],
 expected:['Beta(22, 16), the exact posterior: ELBO = log p(𝒟) = −21.40 and KL = 0.'],
 point:'Conjugate coin: posterior Beta(22, 16), log p(𝒟) = log B(22, 16) − log B(5, 3). The ELBO comes from analytic Beta moments, the KL from its own formula.',
 transition:'Part 5: implement the ELBO for the network.'});

// ================================================================ Part 5 · Implement it
S('bnn-elbo-code','impl','The ELBO for our network, in code',{builds:2,
 body:act(5,code('mu  = Parameter(0.5 * randn(7))         # 7 means\nrho = Parameter(full((7,), log(0.1)))   # 7 log-widths\nq = Normal(mu, rho.exp())               # one bell per parameter\ntheta = q.rsample((16,))                # 16 sampled networks\nlog_lik = log_likelihood(theta).mean()  # forward pass\nkl = kl_divergence(q, prior).sum()      # closed form\nloss = -(log_lik - kl)                  # minus the ELBO'),
  at(2,p('Adam, learning rate 0.02, 4,000 steps; best of five random starts. No p(𝒟) anywhere.'))),
 lines:['Fourteen parameters, sixteen sampled networks per step, and the closed-form KL to the prior.','Every line is computable: rsample, a forward pass and a formula. Never p(𝒟).'],
 ask:['Which line uses the reparameterisation trick?','Which line uses the Gaussian KL formula?'],
 expected:['theta = q.rsample(...): gradients flow back to μ and ρ.','kl = kl_divergence(q, prior).sum(): seven closed-form terms.'],
 demo:'Notebook section 8 runs this loop and records the frames for the next slide.',
 transition:'Press play.'});
S('bnn-vi','impl','Press play: q learns the network',{builds:2,
 body:widget('bnnvi','VI on the tiny network: the predictive band and the ELBO'),
 lines:['Start from random means and narrow widths. Watch q’s predictive band and the ELBO.',`The ELBO climbs from ${big(V.frames[0][1],0)} to about ${num(V.elbo,1)} and the band settles onto the data. It stays below log p(𝒟) ≈ ${num(E.estimate)}.`],
 ask:['Predict: will the ELBO reach log p(𝒟)?','What does the gap mean?'],
 expected:['It approaches, then stops below it; it can never cross.','The gap is KL(q ‖ posterior): how far the best mean-field q is from the true posterior.'],
 point:'PAUSE. Frames every 5 steps to step 200, then every 50; the ELBO for each frame uses 2,000 fixed draws.',
 transition:'What did q learn about each parameter?'});
S('bnn-q','impl','What q learned: one hidden unit switched off',{builds:2,
 body:stack(qFactors({mu:V.mu,sigma:V.sigma}),at(2,p(`v₁ ≈ ${num(V.mu[4])} with width ${num(V.sigma[4],3)}: the first hidden unit is switched off, and its w₁ and b₁ fall back to the prior.`))),
 lines:['The fitted bell for each parameter, against the N(0, 1) prior in grey.','VI switched the first hidden unit off. Mean-field VI often prunes units like this: every parameter kept away from the prior costs KL.'],
 ask:['Which bells look like the prior?','Why might the ELBO prefer to switch a unit off?'],
 expected:['w₁ and b₁: wide and centred near 0, like the prior.','The KL term charges for every parameter that moves away from the prior. Here one tanh unit already fits the data well.'],
 point:`Fitted widths: ${b.names.map((nm,j)=>`${TH[nm]} ${num(V.sigma[j],3)}`).join(', ')}.`,
 transition:'How good is q’s predictive?'});
S('bnn-vs-ref','impl','VI against the reference: the right curve, too-narrow bands',{builds:2,
 body:cols(viVsHmc(),stack(table(['predictive sd','VI','reference'],[['at x = −2.5',num(V.sd[0]),num(H.sd[0])],['at x = 2.5',num(V.sd.at(-1)),num(H.sd.at(-1))]]),
  at(2,p(`ELBO ${num(V.elbo)} against log p(𝒟) ≈ ${num(E.estimate)}: q is about ${num(V.gap,1)} nats from the posterior.`))),'1.05fr 1fr'),
 lines:['Mean-field VI gets the mean curve right.','Its bands are too narrow beyond the data, as the diagonal q in 2-D predicted. The ELBO gap says how far q is from the posterior.'],
 ask:['Where do VI and the reference disagree most?','Is a gap of 5.8 nats large?'],
 expected:['Beyond the data at both ends, where VI is overconfident.','Yes: q misses much of the posterior. Up to log 8 ≈ 2.1 nats of it is because the posterior has eight mirror-image copies (swap the two units, flip their signs) and q sits on one.'],
 transition:'Collect the tools.'});
S('five-tools','impl','Six obstacles, six tools',{
 body:table(['obstacle','tool','in our network'],[
  ['p(𝒟) is an impossible integral','approximate with q','one Gaussian per parameter'],
  ['“close” needs a number','KL(q ‖ p)','KL to the prior'],
  ['a grid cannot search 14 numbers','gradient descent','Adam on μ and ρ'],
  ['no formula for the average','Monte Carlo','16 sampled networks'],
  ['no gradient through a draw','θ = μ + σε, rsample()','reparameterisation'],
  ['KL to the posterior needs p(𝒟)','maximise the ELBO','the training loss']]),
 lines:['Every obstacle today earned a tool, and our network used them all. In Lecture 3 a VAE does the same with one q per image.'],
 ask:['Cover the middle column. Which tool removed which obstacle?'],
 expected:['Students reconstruct it row by row.'],
 transition:'Next lecture.'});
S('next-lecture','impl','Generative models create this inference problem again',{builds:2,
 body:stack(p('A hidden description can generate an observation.'),at(2,p('Next: learn representations for images, then ask whether they can generate new ones.'))),
 lines:['In an image model, a hidden code z can generate an observed image x.','Lecture 2 builds autoencoders; Lecture 3 returns to VI with an encoder that outputs q for each image.'],
 ask:['Where might a new unknown appear in an image model?','What should we ask of a good reconstruction?'],
 expected:['A latent description z can explain an observed image x.','Whether reconstructing observations also gives a useful way to generate new ones.'],
 transition:'Close the main route. Optional depth follows.'});

// ================================================================ Optional depth
const O=(id,title,o)=>S(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});
O('exit-transfer','A sensor has an unknown constant bias',{builds:3,
 body:stack(eq(R`u\sim\N(0,1),\qquad y_i\mid u\sim\N(u,0.5^2)`),
 at(2,eq(R`p(\mathcal D,u)=p(u)\prod_i p(y_i\mid u)`)),
 at(3,eq(R`\max_\psi\E_{q_\psi(u)}[\log p(\mathcal D,u)-\log q_\psi(u)]`))),
 lines:['A sensor reads a hidden bias u, with noise. Name the unknown, the data and a q.','The joint is the prior times the likelihood of each reading: available.','Maximise the ELBO over q’s parameters. The same tools apply.'],
 ask:['Identify the unknown, the observations and one possible q family.','Which available functions form the joint?','What objective would you implement?'],
 expected:['u is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.','The prior density and the likelihood of each reading.','The expectation of log joint minus log q, maximised over ψ.'],
 point:'This conjugate toy has an exact answer too; the exercise is identifying the pieces, not needing approximation.',
 transition:'Where these unknowns appear next.'});
O('mc-variability','More samples, tighter estimates',{
 body:image('figures/mc-variability.svg','Forty KL estimates for each of four sample counts, around the exact value.'),
 lines:['Forty repeats at each sample size. The spread shrinks like 1/√S; the exact value never moves.'],
 ask:['How many more draws halve the spread?'],
 expected:['Four times as many: the spread scales as 1/√S.'],
 point:'Same p and q at every size; 40 seeds each.',
 transition:'Act 4: we can estimate the loss. Now optimise it.'});
O('answer-key','A check with an answer key: the Beta coin',{builds:3,
 body:stack(only('1',coinPlot(0,'0')),only('2',coinPlot(0,'100')),only('3',coinPlot(0,'600'))),
 lines:['With the Beta(5, 3) prior we know the posterior: Beta(22, 16). Fit a Beta q from Beta(10, 10) using only the joint.','By step 100, q has moved most of the way.',`After 600 steps: Beta(${num(c.training.trace.at(-1).a,1)}, ${num(c.training.trace.at(-1).b,1)}), with KL ≈ ${num(c.training.trace.at(-1).posterior_kl,4)} to the true posterior.`],
 ask:['Where does Beta(10, 10) put its mass?','What changed by step 100?','Is the final q exactly Beta(22, 16)?'],
 expected:['Centred at 0.5, and too broad.','It shifted towards the heads proportion and narrowed.','No, but very close: KL ≈ 0.0002. The answer key was never used by the optimiser.'],
 point:'Beta.rsample is differentiable through implicit reparameterisation, not μ + σε. Adam 0.025, 512 draws, seed 2401.',
 transition:'Move q yourself and watch the total.'});
O('optional-one-step','One literal update of the mean',{builds:3,
 body:stack(eq(R`\sigma=2\text{ fixed}:\qquad L(\mu)=\text{const}+\frac{(\mu-1)^2}{32}`),
  at(2,eq(R`\frac{\partial L}{\partial\mu}=\frac{\mu-1}{16}=-0.0625\ \text{ at }\mu=0`)),
  at(3,eq(R`\mu^+=0-4(-0.0625)=0.25:\quad\KL\ ${num(n.one_step.kl_before,3)}\to${num(n.one_step.kl_after,3)}`))),
 ask:['Which term of the Gaussian KL depends on μ?','Differentiate it at μ = 0.','Take one step with η = 4. What is the best KL with σ stuck at 2?'],
 expected:['Only the squared shift, (μ − 1)²/(2·4²).','(μ − 1)/16 = −0.0625: a negative derivative, so descent moves μ right.',`μ⁺ = 0.25. Even at μ = 1 the KL is ${num(n.one_step.fixed_scale_best_kl,3)}: the width mismatch needs σ to move too.`],
 point:'η = 4 is chosen to make the arithmetic visible, not a recommended learning rate.'});
O('optional-nonnegative','KL ≥ 0 by Jensen’s inequality',{builds:3,
 body:stack(eq(R`\KL(q\|p)=-\E_q\!\left[\log\frac{p(u)}{q(u)}\right]`),
 at(2,eq(R`\geq-\log\E_q\!\left[\frac{p(u)}{q(u)}\right]`)),
 at(3,eq(R`=-\log1=0`))),
 ask:['Which function makes Jensen’s inequality useful?','Why does the inequality point this way?','What makes the final expectation one?'],
 expected:['The convex function −log.','For a convex function, its expectation is at least its value at the expectation.','Integrating q · p/q gives the integral of p, which is 1 on a common support.'],
 point:'The tangent-line argument in the main route is the same fact without naming Jensen.'});
O('optional-negative-estimate','Five draws can give a negative KL estimate',{builds:2,
 body:stack(table(['uᵢ','log q(uᵢ) − log p(uᵢ)'],mc.negative_example.draws.map((x,i)=>[f(x,3),f(mc.negative_example.terms[i],3)])),
 at(2,eq(R`\widehat{\KL}_5\approx${num(mc.negative_example.estimate,3)},\qquad\KL(q\|p)\approx${num(mc.exact_kl,3)}`))),
 ask:['Some rows are negative. Could their average be negative?','Does this contradict KL ≥ 0?'],
 expected:['Yes: this five-draw average is about −0.093.','No. A finite-sample estimate is not the exact expectation.'],
 point:'Seed 0 was chosen deliberately as the first negative example in a seed search.'});
O('optional-mixture-candidates','The two PML candidates, scored by Monte Carlo',{builds:2,
 body:stack(table(['candidate','MC estimate ± 1 s.e.','numerical integral'],d.mixture.candidates.map(r=>[
  `N(3, ${r.sigma}²)`,`${f(r.mc,2)} ± ${f(r.standard_error,2)}`,f(r.quadrature,2)])),
 at(2,p('The narrow candidate scores far better. Both sit away from the mixture’s mass, so both are poor.'))),
 ask:['How large is the difference relative to the sampling uncertainty?','What role does the numerical integral play?'],
 expected:['Far larger than the standard errors.','An independent check, possible only because this toy is one-dimensional.'],
 point:'100,000 draws each, seeds 402 and 403.'});
O('optional-rsample-check','A fixed-noise finite difference checks the path',{builds:2,
 body:stack(code('q = Normal(mu, sigma)\nu = q.rsample((S,))\nloss = (q.log_prob(u) - p.log_prob(u)).mean()'),
 at(2,table(['derivative','autograd','finite difference'],[
 ['μ',f(g.finite_difference.autograd[0],6),f(g.finite_difference.finite_difference[0],6)],
 ['σ',f(g.finite_difference.autograd[1],6),f(g.finite_difference.finite_difference[1],6)]]))),
 ask:['What is different about rsample?','What must stay fixed on both sides of the finite difference?'],
 expected:['It returns a draw with the pathwise gradient connection.','The same noise array. Fresh noise would confound the check with sampling variation.'],
 point:'Seed 88, S = 2048, h = 10⁻⁵.'});
O('optional-score-function','Pathwise gradients are one route, not the only one',{builds:2,
 body:stack(eq(R`\nabla_\psi\E_{q_\psi}[f_\psi(u)]=\E_{q_\psi}[f_\psi(u)\nabla_\psi\log q_\psi(u)+\nabla_\psi f_\psi(u)]`),
 at(2,p('A score-function estimator accounts explicitly for the changing sampling distribution.'))),
 ask:['Which term was missing from the naive sample() gradient?','Does this identity require u = μ + σε?'],
 expected:['The term accounting for the change in the probability of drawing u.','No. It works for discrete q too, usually with higher variance.'],
 point:'Regularity conditions allow differentiating under the integral.'});
O('optional-noisy-elbo','A noisy ELBO estimate can cross the evidence',{builds:2,
 body:stack(image('figures/coin-noisy-objective.svg','Sampled ELBO estimates and the analytic ELBO along the Beta coin fit.'),
 at(2,small(`${c.sampled_estimates_above_evidence} of 600 sampled estimates lie above the known evidence in this run.`))),
 ask:['Can a finite-sample ELBO estimate cross the reference line?','Which object remains a lower bound?'],
 expected:['Yes, on some draws.','The exact expected ELBO, not each sampled estimate.'],
 point:'Beta coin, seed 2401.'});
O('optional-certificate','A higher ELBO does not prove a zero gap',{builds:2,
 body:stack(eq(R`\ELBO(q_{\mathrm{new}})>\ELBO(q_{\mathrm{old}})`),
 at(2,p('The family may not contain the posterior, and optimisation may stop short.'))),
 ask:['What does a true increase in the exact ELBO establish?','Does a stable loss establish q = p(u | 𝒟)?'],
 expected:['For this fixed model and data, the posterior KL decreased.','No. The family or optimisation can leave a positive gap.'],
 point:'Recall the diagonal q in 2-D: its best possible KL was 0.83.'});
O('optional-coin-check','Check the fixed total with three Beta candidates',{builds:3,
 body:c.candidates.map((row,i)=>only(i+1,coinPlot(i,null,{height:260})+gapLine(row))).join(''),
 ask:['Add the ELBO and the gap. What is fixed?','Does the better candidate raise the ELBO by the amount its KL falls?','What happens at the exact posterior?'],
 expected:['The sum is the same log evidence, about −21.397.','Yes. The identity holds for each candidate.','The KL is zero, so the ELBO equals the log evidence.'],
 point:'ELBO computed directly from expected log joint minus expected log q; KL computed separately.'});
O('optional-reductions','A mean must say which dimension it averages',{builds:2,
 body:stack(table(['object','shape','correct reduction'],[
 ['S scalar Beta draws','[S]','mean over S'],
 ['S d-dimensional joint log densities','[S]','mean over S'],
 ['S factorised coordinate log densities','[S, d]','sum over d, then mean over S'],
 ]),at(2,p('MultivariateNormal.log_prob already sums the event dimension. Do not sum the sample axis as well.'))),
 ask:['Does every log_prob output still have an event dimension?','What goes wrong if you sum over S before averaging?'],
 expected:['No. A multivariate distribution’s log_prob already reduces its event dimensions.','It rescales the objective, often multiplying an entropy contribution by the number of samples.']});
O('optional-count-data','An ordered sequence and a count have different evidence',{builds:2,
 body:stack(eq(R`p(H=17\mid r)=\binom{30}{17}r^{17}(1-r)^{13}`),
 at(2,table(['observed object','log evidence'],[
 ['one ordered sequence',f(c.log_evidence,3)],
 ['only the head count',f(c.count_convention_comparison.count_log_evidence,3)],
 ]))),
 ask:['Which event contains more possible sequences?','Does this constant change the posterior or its KL gap?'],
 expected:['The count-only event includes every ordering with 17 heads.','No. It adds the same constant to the joint, the ELBO and the evidence.'],
 point:'The whole lecture uses the ordered-sequence convention.'});
O('optional-beta-elbo','Check the sampled objective with analytic Beta moments',{builds:3,
 body:stack(eq(R`\E_q[\log r]=\operatorname{digamma}(a)-\operatorname{digamma}(a+b)`),
 at(2,eq(R`\E_q[\log(1-r)]=\operatorname{digamma}(b)-\operatorname{digamma}(a+b)`)),
 at(3,p('Insert these into expected log likelihood + expected log prior − expected log q.'))),
 ask:['Which two expectations determine the coin ELBO?','Why write digamma rather than ψ here?','How was the identity checked non-circularly?'],
 expected:['Expected log r and expected log(1−r).','ψ already denotes the variational parameters.','The ELBO is computed directly from these moments, then the separately computed KL is added and compared with the evidence.']});
