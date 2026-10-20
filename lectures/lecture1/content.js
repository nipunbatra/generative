import {m,eq,at,only,cols,code,image,table,flow,plot,palette} from '../shared/author.js';
import {panel,label,legend,fig,widget,TINT,MUTED,AX,num,lin,mount as mountWidgets} from './widgets.js';
import {normalPdf,normalLogPdf} from './math.mjs';

const d=await fetch(new URL('./data/science.json',import.meta.url)).then(r=>r.json());
export const mount=(stage,ctx)=>mountWidgets(stage,ctx,d);
const R=String.raw;
const f=(v,n=4)=>Number(v).toFixed(n).replace('-','−');
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const n=d.normal,c=d.coin,g=d.gradients,mc=d.monte_carlo,st=d.story;
export const meta={
 id:'lecture1',number:1,title:'KL and variational inference',
 subtitle:'Push a simple distribution onto a posterior nobody can write down',minutes:70,
 notebook:'lecture1-variational-inference.ipynb',
 sections:[
  {id:'wall',title:'Act 0 · The wall',minutes:8},
  {id:'close',title:'Act 1 · Closer how?',minutes:15},
  {id:'move',title:'Act 2 · Make q move',minutes:7},
  {id:'nofo',title:'Act 3 · No formula',minutes:7},
  {id:'grad',title:'Act 4 · Gradients through a draw',minutes:10},
  {id:'elbo',title:'Act 5 · The real target',minutes:10},
  {id:'back',title:'Act 6 · Back to the wall',minutes:10},
  {id:'exit',title:'Exit',minutes:3},
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
const ACTS=['The wall','Closer how?','Make q move','No formula','Gradients','The real target','Back to the wall'];
const acts=k=>`<div class="act-strip" role="img" aria-label="Lecture map: act ${k} of 6">${ACTS.map((a,i)=>`<span class="${i<k?'done':i===k?'now':''}">${i} · ${a}</span>`).join('')}</div>`;
const act=(k,...a)=>`${acts(k)}<div class="act-body">${a.join('')}</div>`;
const grid3=(...a)=>`<div class="image-pair" style="grid-template-columns:repeat(3,minmax(0,1fr));gap:20px">${a.join('')}</div>`;

// ---------------------------------------------------------------- shared figures
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
function tosses(){
 return `<div class="coin-tosses" style="display:grid;grid-template-columns:repeat(15,1fr);gap:8px;margin:12px 0">${[...c.ordered_sequence].map(t=>`<div style="border:1px solid #d7dce1;border-radius:50%;height:46px;display:grid;place-items:center;color:${t==='H'?palette.data:palette.ink};font-size:26px">${t}</div>`).join('')}</div>`;
}
function conjugateChart(){
 const a=d.plots.coin,P=panel({W:600,H:330,L:16,R:14,xMin:0,xMax:1,yMax:5.8,ticksX:[0,.25,.5,.75,1],yAxis:false,xLabel:'r, chance of heads'});
 return fig(600,330,P.axes+P.line(a.x,a.prior,MUTED,'stroke-dasharray="8 6"')+P.area(a.x,a.posterior,TINT.p)+P.line(a.x,a.posterior,palette.p)+
  legend(16,30,[['prior Beta(5, 3)',MUTED,true],['posterior Beta(22, 16)',palette.p]]),'The Beta(5, 3) prior and the Beta(22, 16) posterior over the chance of heads.');
}
function wallChart(){
 const w=st.wall.plot.w,j=st.wall.plot.joint_e10,P=panel({W:600,H:330,L:16,R:14,xMin:-3,xMax:3.5,yMax:5.4,ticksX:[-2,-1,0,1,2,3],yAxis:false,xLabel:'w, log-odds of heads'});
 return fig(600,330,P.axes+P.area(w,j,TINT.p)+P.line(w,j,palette.p)+`<line x1="${P.x(1.42)}" y1="${P.y(3.45)}" x2="${P.x(.6)}" y2="${P.y(2.1)}" stroke="${palette.p}" stroke-width="2"/>`+label(P.x(1.5),P.y(3.55),'area = p(𝒟)',{fill:palette.p,size:25,weight:600})+
  label(16,30,'prior × likelihood, p(𝒟, w) × 10¹⁰',{fill:palette.p,size:21}),'The joint density of the data and the log-odds; its area is the evidence.');
}
function gridDots(){
 let row='',square='';
 for(let i=0;i<100;i++)row+=`M${(14+i*2.9).toFixed(1)},150h1.6v1.6h-1.6z`;
 for(let i=0;i<50;i++)for(let j=0;j<50;j++)square+=`M${(420+i*4.6).toFixed(1)},${(56+j*4.6).toFixed(1)}h2v2h-2z`;
 return fig(700,330,label(10,32,'1 unknown',{fill:palette.ink,size:24,weight:600})+`<path d="${row}" fill="${palette.ink}"/>`+
  label(10,230,'100 points',{size:22})+label(410,32,'2 unknowns',{fill:palette.ink,size:24,weight:600})+`<path d="${square}" fill="${palette.ink}"/>`+
  label(410,318,'100 × 100 = 10,000 points',{size:22}),'A grid of 100 points for one unknown, and 10,000 for two.');
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
function planFlow(){
 const steps=[['Closer how?','Act 1'],['Make q move','Act 2'],['No formula','Act 3'],['Gradient through a draw','Act 4'],['Unknown posterior','Act 5'],['Back to the coin','Act 6']];
 return flow(steps.map(([label,sub],i)=>({id:'s'+i,x:5+i*166,y:20,w:150,h:128,label,sub,role:i===5?'data':'q'})),
  steps.slice(1).map((_,i)=>({from:'s'+i,to:'s'+(i+1)})),{width:1000,height:170,label:'Six acts: five obstacles, then back to the coin.'});
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
const logSig=w=>w>=0?-Math.log1p(Math.exp(-w)):w-Math.log1p(Math.exp(w));
const logitJoint=w=>normalLogPdf(w,{mu:0,sigma:1})+17*logSig(w)+13*logSig(-w);
function trapz(xs,ys){let s=0;for(let i=1;i<xs.length;i++)s+=(xs[i]-xs[i-1])*(ys[i]+ys[i-1])/2;return s;}
const forces=[.06,st.logitvi.final.s,1.2].map(s=>{const m0=st.logitvi.final.m,us=lin(m0-10*s,m0+10*s,4001);
 const ej=trapz(us,us.map(u=>normalPdf(u,{mu:m0,sigma:s})*logitJoint(u))),h=.5*Math.log(2*Math.PI*Math.E*s*s);return {m:m0,s,ej,h,elbo:ej+h};});
function forcePanel(k){
 const fo=forces[k],w=st.wall.plot.w,post=st.wall.plot.posterior,P=panel({W:370,H:190,L:14,R:12,T:50,B:22,xMin:-3,xMax:3.5,yMax:1.4,yAxis:false});
 const name=['too narrow','balanced','too wide'][k];
 return fig(370,190,P.axes+P.area(w,post,TINT.p)+P.line(w,post,palette.p)+P.line(w,w.map(u=>normalPdf(u,{mu:fo.m,sigma:fo.s})),palette.q)+
  label(14,30,`${name}: σ = ${num(fo.s)}`,{fill:palette.q,size:22,weight:600}),`A ${name} q against the posterior.`);
}
const forceCaption=k=>{const fo=forces[k];return `<p class="caption" style="font-variant-numeric:tabular-nums">E<sub>q</sub>[log joint] ${num(fo.ej,2)}, H(q) ${num(fo.h,2)}<br><b style="color:${k===1?palette.q:palette.ink};font-size:24px">ELBO ${num(fo.elbo,2)}</b></p>`;};
function identityRows(){
 const fr=st.logitvi.frames,ev=st.logitvi.final.log_evidence_grid,rows=[0,10,600].map(s=>fr.find(r=>r[0]===s));
 const W=1150,H=280,lo=-24.6,hi=-21.2,X=v=>250+(v-lo)/(hi-lo)*860;
 let s=`<line x1="${X(ev)}" y1="40" x2="${X(ev)}" y2="${H-12}" stroke="${palette.data}" stroke-width="2.5" stroke-dasharray="8 6"/>`+
  label(X(ev),28,`log p(𝒟) = ${num(ev)}`,{fill:palette.data,size:21,anchor:'end'});
 rows.forEach((r,i)=>{const y=86+i*70,e=r[4],gap=ev-e;
  s+=label(0,y+8,i===0?'q at step 0':`step ${r[0]}`,{fill:palette.ink,size:21})+`<line x1="${X(lo)}" y1="${y}" x2="${X(hi)}" y2="${y}" stroke="${AX}" stroke-width="1"/>`+
   `<line x1="${X(e)}" y1="${y}" x2="${X(ev)}" y2="${y}" stroke="${palette.loss}" stroke-width="5"/>`+`<circle cx="${X(e).toFixed(1)}" cy="${y}" r="8" fill="${palette.q}"/>`+
   label(X(e)-14,y-14,`ELBO ${num(e,2)}`,{fill:palette.q,size:20,anchor:gap>.4?'start':'end'})+
   label(gap>.4?(X(e)+X(ev))/2:X(e)-14,gap>.4?y+30:y+30,`KL ${num(gap,gap>.01?2:4)}`,{fill:palette.loss,size:20,anchor:gap>.4?'middle':'end'});});
 return fig(W,H,s,'Three q along the fit: the ELBO rises towards the fixed log evidence and the KL gap shrinks.');
}
const qTail=(()=>{const z=st.logitvi.final.m/st.logitvi.final.s,t=1/(1+.2316419*Math.abs(z)),pd=Math.exp(-z*z/2)/Math.sqrt(2*Math.PI),
 tail=pd*t*(.31938153+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return z>0?1-tail:tail;})();
const exactAbove=(()=>{const w=st.wall.plot.w,pst=st.wall.plot.posterior;let s=0;for(let i=1;i<w.length;i++)if(w[i-1]>=0)s+=(w[i]-w[i-1])*(pst[i]+pst[i-1])/2;return s;})();
function rChart(){
 const t=st.logitvi.theta,P=panel({W:600,H:330,L:16,R:22,xMin:.2,xMax:.9,yMax:5.6,ticksX:[.2,.3,.4,.5,.6,.7,.8,.9],yAxis:false,xLabel:'r = sigmoid(w), chance of heads'});
 return fig(600,330,P.axes+`<line x1="${P.x(.5)}" y1="${P.y(0)}" x2="${P.x(.5)}" y2="${P.y(5.2)}" stroke="${MUTED}" stroke-dasharray="6 6"/>`+label(P.x(.5)+8,P.y(5.0),'fair',{size:20})+
  P.area(t.x,t.posterior,TINT.p)+P.line(t.x,t.posterior,palette.p)+P.line(t.x,t.q,palette.q,'stroke-dasharray="9 6"')+
  legend(16,30,[['exact posterior',palette.p],['q from VI',palette.q,true]]),'The VI approximation and the exact posterior over the chance of heads.');
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

// ================================================================ Act 0 · The wall
S('coin-question','wall','Is this coin fair?',{builds:2,
 body:stack(tosses(),at(2,p(`An honest answer is a distribution over ${m(R`\ncol r`)}, the coin’s chance of heads: where it probably is, and how sure we can be.`))),
 lines:['Thirty tosses: 17 heads and 13 tails. Vote: is this coin fair?','“Probably fair, but I can’t be sure” is a distribution over r. Bayes’ rule gives it.'],
 ask:['Vote: is this coin fair? How sure are you?','What would a complete answer look like?'],
 expected:['Most students hedge: 17 of 30 is a little high, but fair seems plausible. The hedge is the point.','A posterior distribution over r: a centre and a spread, not one number.'],
 point:'PAUSE 1 · the hook. Take the vote by hands and keep it on the board: Act 6 answers it.',
 transition:'For one convenient prior, Bayes’ rule answers this exactly.'});
S('conjugate-luck','wall','With a Beta prior, Bayes’ rule has a formula',{builds:2,
 body:cols(conjugateChart(),stack(eq(R`p(r\mid\mathcal D)=\frac{p(\mathcal D\mid r)\,p(r)}{\dcol{p(\mathcal D)}}`),
  eq(R`\mathrm{Beta}(5,3)\ \xrightarrow{\ 17\text{ H},\ 13\text{ T}\ }\ \pcol{\mathrm{Beta}(22,16)}`),
  at(2,eq(R`\dcol{p(\mathcal D)}=\frac{B(22,16)}{B(5,3)}`)),at(2,eq(R`\log p(\mathcal D)=${num(c.log_evidence,2)}`))),'1.1fr 1fr'),
 lines:['Start from a Beta(5, 3) prior. Add 17 heads and 13 tails: the posterior is Beta(22, 16).','Even the normaliser p(𝒟), the evidence, has a formula. That is luck: this prior and this likelihood are conjugate.'],
 ask:['Where does the posterior sit, and how wide is it?','Why can we write p(𝒟) down here?'],
 expected:['Near 22/38 ≈ 0.58, and much narrower than the prior.','Prior and likelihood share the form r^a(1−r)^b, so the integral is a Beta function. Change either and the luck runs out.'],
 point:'The evidence p(𝒟) is the integral of prior × likelihood over r. Keep it blue throughout the lecture.',
 transition:'Keep the data. Change only the prior.'});
S('no-formula','wall','Change the prior, and the formula disappears',{builds:2,
 body:cols(stack(eq(R`w\sim\N(0,1),\qquad r=\operatorname{sigmoid}(w)`),
  eq(R`\dcol{p(\mathcal D)}=\int\N(w;0,1)\,r(w)^{17}\big(1-r(w)\big)^{13}\,dw`),
  at(2,p(`On a 200,001-point grid: area ${m(R`\approx${num(Math.exp(st.wall.logit_log_evidence_grid)*1e10,1)}\times10^{-10}`)}, so ${m(R`\log p(\mathcal D)\approx${num(st.wall.logit_log_evidence_grid)}`)}.`))),wallChart(),'1.25fr 1fr'),
 lines:['Put a Normal prior on the coin’s log-odds w instead. Same coin, same data, same question.','The evidence is this shaded area, and no formula gives it. With one unknown, a grid of points still can.'],
 ask:['What changed, and what stayed the same?','Can you integrate this in closed form?'],
 expected:['Only the prior. Data, likelihood and question are unchanged.','No. A dense grid gives log p(𝒟) ≈ −21.62, but only because w is a single number.'],
 point:'Log-odds priors are what logistic regression and neural networks use, so this is the normal case, not a contrived one.',
 transition:'Why not always use a grid?'});
S('grid-explodes','wall','A grid works for one unknown, not for a million',{builds:2,
 body:cols(gridDots(),at(2,stack(p('A small network: 1,000,000 weights.'),`<p class="result">10<sup style="font-size:.5em">2,000,000</sup></p>`,small('grid points, at 100 per weight. The observable universe has about 10<sup>80</sup> atoms.'))),'1.5fr 1fr'),
 lines:['A grid needs about 100 points per unknown. Two unknowns already need 10,000.','A small neural network has a million unknowns: 10^2,000,000 grid points. We need a different idea.'],
 ask:['How many points for 3 unknowns? For 10?','Could faster computers rescue the grid?'],
 expected:['10⁶ and 10²⁰.','No. The count grows exponentially with the number of unknowns.'],
 transition:'Three families of methods get around the integral.'});
S('three-ways','wall','Three ways around an integral nobody can do',{builds:3,
 body:grid3(stack(threeWay('laplace'),small(`A Gaussian at the highest peak: N(${num(st.three.laplace[0],1)}, ${num(st.three.laplace[1],1)}²). It never sees the second bump.`)),
  at(2,stack(threeWay('mcmc'),small(`20,000 random-walk steps. Their mean, ${num(st.three.mcmc.mean)}, matches the true ${num(st.three.true_mean)}.`))),
  at(3,stack(threeWay('vi'),small(`The closest Gaussian, found by optimisation: N(${num(st.three.vi[0])}, ${num(st.three.vi[1])}²).`)))),
 lines:['Laplace: fit a Gaussian to the curvature at the highest peak.','MCMC: a random walk whose visits follow the posterior. Right in the long run, but slow.','Variational inference: choose the closest member of a simple family, by optimisation. That is today.'],
 ask:['What does Laplace miss here?','What does MCMC cost?','What does VI trade away?'],
 expected:['The second bump: it only looks at the peak.','Many steps, each evaluating the model on the data; hard to scale to large models and datasets.','Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.'],
 point:'Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.',
 transition:'The plan for VI, and what stands in its way.'});
S('plan','wall','The plan: push a simple q towards the posterior',{builds:2,
 body:stack(eq(R`q^*=\arg\min_{q\in\qcol{\mathcal Q}}\ \text{distance}\big(q,\ \pcol{p(u\mid\mathcal D)}\big)`),at(2,planFlow())),
 lines:['Pick a simple family 𝒬, such as Gaussians, and find the member closest to the posterior.','Five obstacles stand in the way. Each act removes one; then we return to this coin.'],
 ask:['What must we define before we can minimise anything?','Which obstacle looks hardest?'],
 expected:['A number for the “distance” between two distributions.','Usually the last: the posterior we are approaching is the thing we cannot evaluate.'],
 point:'u is the generic unknown from now on. For the coin, u is the log-odds w.',
 transition:'Obstacle 1: what does “closer” mean for distributions?'});

// ================================================================ Act 1 · Closer how?
S('fruit','close','Two surveys: how different are they?',{builds:2,
 body:act(1,cols(fruitBars(),stack(p('Favourite fruit in a class, last year (q) and this year (p).'),at(2,p('Bananas fell from 40% to 10%. Clementines rose from 10% to 40%.'))),'1.35fr 1fr')),
 lines:['We want one number for how much the class changed.','A fourfold fall and a fourfold rise. Should they count the same?'],
 ask:['How would you score the difference between these two surveys?','Bananas ÷4, clementines ×4: same size of change?'],
 expected:['Students propose differences of percentages. Note that apples, 50% both years, should contribute nothing.','By ratio, yes: both are a factor of 4. That suggests comparing ratios.'],
 point:'p is the survey we weight by (this year); q is the reference (last year).',
 transition:'Compare by ratios.'});
S('ratios-logs','close','Ratios say “how many times”; logs make up and down cancel',{builds:2,
 body:cols(table(['fruit','p','q',m(R`p/q`),m(R`\log(p/q)`)],st.fruit.names.map((name,i)=>[name,num(st.fruit.p[i],1),num(st.fruit.q[i],1),st.fruit.ratio[i]===.25?'¼':num(st.fruit.ratio[i],0),at(2,num(st.fruit.log_ratio[i]))])),
  stack(ratioLine(false),at(2,ratioLine(true))),'1fr 1fr'),
 lines:['Ratios p/q: 1, ¼ and 4. But on the ratio scale, ×4 lands 3 above 1 and ÷4 only 0.75 below.','Take logs: log 4 = +1.39 and log ¼ = −1.39. Equal and opposite.'],
 ask:['On the ratio scale, which change looks bigger?','After taking logs?'],
 expected:['The rise: +3 against −0.75, though both are fourfold.','Equal and opposite, ±1.39. Logs turn “times” into “plus”.'],
 transition:'Now combine the three log ratios into one number.'});
S('weight','close','Weight by how common each fruit is now: that is KL',{builds:2,
 body:cols(contributionBars(),stack(eq(R`\KL(\pcol p\,\|\,\qcol q)=\sum_x \pcol{p(x)}\,\log\frac{\pcol{p(x)}}{\qcol{q(x)}}`),
  at(2,eq(R`=0.5(0)+0.1(-1.39)+0.4(1.39)`)),at(2,eq(R`\approx${num(st.fruit.kl)}\ \text{nats}`))),'1fr 1.1fr'),
 lines:['Average the log ratios, weighting each fruit by p(x), its share this year.','The rare banana barely counts; the common clementine dominates. Total 0.42 nats: the Kullback–Leibler divergence.'],
 ask:['Why weight by p rather than q?','What is the KL if the surveys agree?'],
 expected:['That choice defines KL(p‖q): the first argument does the weighting. Swapping gives a different number.','Zero: every ratio is 1, every log is 0.'],
 point:'Natural log gives nats. Base 2 gives bits, which the next slide uses.',
 transition:'The same number has a second meaning, in bits.'});
S('bits','close','KL is the extra message length of using the wrong code',{builds:2,
 body:stack(table(['weather',m(R`p`),m(R`q`),'code built for p','code built for q'],st.bits.names.map((name,i)=>[name,num(st.bits.p[i]),num(st.bits.q[i]),`<code>${st.bits.right_code[i]}</code>`,`<code>${st.bits.wrong_code[i]}</code>`])),
  eq(R`\text{average length under }p:\quad\text{code for }p=${num(st.bits.entropy)}\text{ bits},\qquad\text{code for }q=${num(st.bits.cross_entropy)}\text{ bits}`),
  at(2,eq(R`${num(st.bits.cross_entropy)}-${num(st.bits.entropy)}=${num(st.bits.kl)}\text{ bits}=\KL(p\|q)\ \text{with}\ \log_2`))),
 lines:['Send each day’s weather in bits. A code built for the true frequencies p needs 1.50 bits a day.','A code built for the wrong frequencies q needs 1.75. The extra 0.25 bits a day are exactly KL(p‖q), in base 2.'],
 ask:['Why does sunny get the shortest code?','Where do the extra 0.25 bits come from?'],
 expected:['Sunny is the most common, so a short code for it saves the most bits on average.','Sunny is half of all days, and the wrong code spends 2 bits on it. KL measures that waste.'],
 transition:'Cross-entropy is this “wrong code” length, and you have minimised it many times.'});
S('used-before','close','You have minimised KL before: in every cross-entropy loss',{builds:3,
 body:stack(eq(R`\underbrace{-\sum_x p(x)\log q(x)}_{\text{cross-entropy }H(p,q)}=\underbrace{-\sum_x p(x)\log p(x)}_{\text{entropy }H(p)}+\KL(p\|q)`),
  at(2,eq(R`\text{classifier:}\quad\min_\theta H(y,\,q_\theta)\ \Longleftrightarrow\ \min_\theta\KL(y\,\|\,q_\theta)`)),
  at(3,eq(R`\text{maximum likelihood:}\quad-\tfrac1N\textstyle\sum_i\log p_\theta(x_i)\ \approx\ H(p_{\text{data}})+\KL(p_{\text{data}}\,\|\,p_\theta)`))),
 lines:['Cross-entropy splits into entropy plus KL: the 1.75 bits were 1.50 + 0.25.','A classifier’s cross-entropy loss: the labels’ entropy is fixed, so training minimises a KL.','Maximum likelihood minimises KL(data ‖ model). Variational inference is the third use.'],
 ask:['Which term of the split depends on q?','What does your classifier’s loss minimise?','So what is new in VI?'],
 expected:['Only the KL. H(p) is fixed by p.','KL(labels ‖ predictions), plus a constant.','The direction. VI puts the approximation first, KL(q‖p). Act 1 ends with why.'],
 transition:'What does a KL of 0.19 actually feel like?'});
S('coin-evidence','close','A KL value is how fast evidence piles up',{builds:2,
 body:cols(evidenceChart(),stack(eq(R`\KL\big(\mathrm{Bern}(0.8)\,\|\,\mathrm{Bern}(0.5)\big)\approx${num(st.coinkl.kl)}\ \text{nats}`),
  at(2,eq(R`\frac{p(\mathcal D\mid0.8)}{p(\mathcal D\mid0.5)}\approx e^{\,${num(st.coinkl.kl)}\,n}`)),at(2,small(`The reverse direction, KL(0.5‖0.8) = ${num(st.coinkl.reverse)}, is a different number.`))),'1fr 1.15fr'),
 lines:['A coin lands heads 80% of the time. Each toss favours “0.8” over “fair” by 0.19 nats on average. After 30 tosses, by what factor?','About e^(30 × 0.19) ≈ 325. KL is the rate at which evidence piles up: 6.9× after 10 tosses, 325× after 30.'],
 ask:['Predict: after 30 tosses, how many times more likely are the data under 0.8 than under 0.5?','Why does a small KL eventually matter so much?'],
 expected:['Most guess far too low. It is about 325×.','Log evidence adds up toss by toss, so the ratio grows exponentially with n.'],
 point:`PAUSE 2. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of ${num(st.coinkl.simulated_mean_llr_30,3)}, against 30 × KL = ${num(30*st.coinkl.kl,3)}.`,
 transition:'Sums become integrals for continuous unknowns.'});
S('densities','close','For densities, sums become integrals, and KL ≥ 0',{builds:2,
 body:cols(stack(eq(R`\KL(\qcol q\,\|\,\pcol p)=\int \qcol{q(u)}\,\log\frac{\qcol{q(u)}}{\pcol{p(u)}}\,du`),
  at(2,eq(R`-\KL(q\|p)=\E_q\Big[\log\frac{p}{q}\Big]\le\E_q\Big[\frac{p}{q}-1\Big]`)),at(2,eq(R`=\int p\,du-1=0`))),tangentChart(),'1.25fr 1fr'),
 lines:['For a continuous unknown u the sum becomes an integral. Same recipe: weight the log ratio by the first argument.','log t never rises above its tangent t − 1. Inside the average that gives KL ≥ 0, with 0 only when q = p.'],
 ask:['Which distribution supplies the weights in KL(q‖p)?','Why can’t KL be negative?'],
 expected:['q, the first argument.','log t ≤ t − 1 for every ratio t = p/q, and the average of p/q under q is ∫p = 1.'],
 point:'Individual log ratios can be negative; only the weighted average is guaranteed nonnegative. That returns with sampling in Act 3.',
 transition:'Back to the PML vote, now with a score.'});
S('compare','close','Which Gaussian is closer to p?',{builds:2,
 body:cols(normalPlot(['q1','q2']),stack(eq(R`\pcol{p=\N(1,4^2)}`),eq(R`\qcol{q_1=\N(0,2^2)}`),eq(R`\qcol{q_2=\N(1,3^2)}`),
  at(2,table(['',m(R`\KL(q\|p)`)],[['q₁',num(n.candidates.q1.kl)],['q₂',num(n.candidates.q2.kl)]]))),'1.45fr 1fr'),
 lines:['Vote: which is closer to p, q₁ or q₂?','KL(q₁‖p) = 0.35 nats and KL(q₂‖p) = 0.07. The vote now has a number.'],
 ask:['Vote: q₁ or q₂?','Does the score agree with your eye?'],
 expected:['Most pick q₂: same centre and a more similar width.','Yes: 0.07 < 0.35. The two-Gaussian formula is derived in the optional slides.'],
 point:'PAUSE 3 (PML L23 08:30). Normal(loc, scale) takes a standard deviation; N(μ, σ²) writes a variance.',
 transition:'Where does each number come from? Watch it as an area.'});
S('kl-area','close','KL is an area you can watch',{
 body:widget('klarea','KL as the net area under its integrand, with sliders for q'),
 lines:['Purple area adds and red area subtracts; the net area is the KL. Drag q, or flip the direction.'],
 ask:['Make q narrow and centred on p. What happens in each direction?'],
 expected:['KL(q‖p) stays moderate: q only visits places where p is reasonable. KL(p‖q) explodes: p’s tails land where q is nearly zero.'],
 point:'Click q₁, then q₂, then flip to KL(p‖q). Then shrink σ in each direction.',
 transition:'So which direction should VI use?'});
S('why-reverse','close','VI uses KL(q‖p) because it needs only what we have',{builds:2,
 body:stack(table(['divergence','needs','for a posterior p(u | 𝒟)'],[
  [m(R`\KL(p\,\|\,q)`),'samples from p','impossible: that is the problem'],
  [m(R`\KL(q\,\|\,p)`),'samples from q; log p up to a constant','possible: we choose q'],
  ['Wasserstein, TV','samples from p, or p normalised','impossible']]),
  at(2,p(`The price: ${m(R`q(u)\log\frac{q(u)}{p(u)}`)} blows up wherever q puts mass and p is near zero, so q avoids those places.`))),
 lines:['Every alternative needs samples from the posterior or its normalised density, which is exactly what we lack.','KL(q‖p) needs only what we can get. The price: q avoids any region where p is near zero.'],
 ask:['Which of these could we compute for a posterior?','What might that price look like for a posterior with two bumps?'],
 expected:['Only KL(q‖p).','q may sit on one bump and ignore the other.'],
 transition:'Watch that price on two bumps.'});
S('fwdrev','close','The price: reverse KL picks one bump',{builds:2,
 body:widget('fwdrev','Forward and reverse KL fits of one Gaussian to two bumps'),
 lines:['Fit one Gaussian to two narrow bumps, three times. Predict where each q settles.','KL(p‖q) spreads over both bumps. KL(q‖p) locks onto whichever bump is nearer its start, and reports it confidently.'],
 ask:['Predict each final q.','Which fit would make you overconfident?'],
 expected:['Forward: wide, centred between the bumps. Reverse: one bump, depending on the start.','The reverse fits: each reports one bump with a narrow spread.'],
 point:`PAUSE 4. Narrow bumps: forward KL ${num(st.fwdrev['0.5'].fwd.at(-1)[3])}; reverse ${num(st.fwdrev['0.5'].revl.at(-1)[3])} (= −log 0.7) from the left and ${num(st.fwdrev['0.5'].revr.at(-1)[3])} (= −log 0.3) from the right. Then switch to wide bumps: overlapping bumps give one broad reverse-KL fit.`,
 transition:'Act 2: a score picks a winner. Can it improve a loser?'});

// ================================================================ Act 2 · Make q move
S('score-to-search','move','A score picks a winner. Can it improve a loser?',{builds:2,
 body:act(2,cols(code('mu  = Parameter(tensor(4.0))\nrho = Parameter(tensor(log(0.5)))\nfor step in range(200):\n    q = Normal(mu, rho.exp())\n    loss = kl_divergence(q, p)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()'),
  stack(eq(R`\psi^*=\arg\min_\psi\KL(q_\psi\,\|\,p)`),at(2,eq(R`\sigma=\exp(\rho)>0\ \text{ for every }\rho`))),'1.25fr 1fr')),
 lines:['Treat the KL as a loss and run gradient descent on q’s parameters, with p fixed.','Store ρ = log σ: whatever an update does to ρ, σ = exp(ρ) stays positive.'],
 ask:['What stays fixed, and what does the optimiser change?','Why not update σ directly?'],
 expected:['p stays fixed; μ and ρ change.','A step could make σ zero or negative, which is not a valid Normal.'],
 point:'The literal one-step mean update, μ⁺ = 0.25, is in the optional slides.',
 transition:'Press play.'});
S('olive','move','Press play: q slides onto p',{builds:2,
 body:widget('olive','q moving onto p under gradient descent'),
 lines:['Start q far away, at N(4, 0.5²). Predict: which reaches the target first, the centre or the width?','The width gets there first and overshoots a little; the centre slides in by step 100. Then KL ≈ 0.'],
 ask:['Predict: does the centre or the width arrive first?','Why can both move at once?'],
 expected:[`Students split. The width reaches 4 around step 45, overshoots to ${num(Math.max(...st.olive.frames.map(r=>r[2])))}, then settles; μ arrives by step 100.`,'Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.'],
 point:'PAUSE 5 (PML L23 12:13). Adam, learning rate 0.05, analytic KL. Notebook section 3.',
 transition:'One dimension is easy. What happens with two correlated unknowns?'});
S('two-d','move','In 2-D, a q that cannot tilt shrinks instead',{builds:3,
 body:cols(twoD(),stack(eq(R`\pcol{p=\N\Big(0,\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}\Big)}`),
  at(2,eq(R`\qcol{q=\N\big(0,\operatorname{diag}(s^2,s^2)\big)}`)),at(2,eq(R`s=${num(st.twod.analytic_sd)},\quad\KL=${num(st.twod.analytic_kl)}`)),
  at(3,eq(R`\qcol{q=\N(0,LL^\top)}:\ \KL=0`))),'1.15fr 1fr'),
 lines:['A correlated target: both coordinates have width 1, correlation 0.9. A diagonal q cannot tilt. Predict its widths.','It shrinks to fit inside the ellipse: widths 0.44, not 1. Mean-field VI underestimates uncertainty.','Let q tilt with a full covariance, Σ = LLᵀ, and the KL drops to 0.'],
 ask:['Predict the diagonal q’s widths: 1, more or less?','Why shrink rather than spread?','What does the full covariance cost?'],
 expected:['Less: about 0.44, which is √(1 − 0.9²).','KL(q‖p) heavily penalises q mass where p is tiny, which is off the diagonal ridge.','More parameters: d(d + 1)/2 for L instead of d. For a million weights that is far too many.'],
 point:'PAUSE 6. Measured fit: both widths 0.4359, KL 0.8304, matching √(1 − ρ²) and −½ log(1 − ρ²).',
 transition:'Act 3: the library had a formula for every KL so far. What if it doesn’t?'});

// ================================================================ Act 3 · No formula
S('mixture-no-formula','nofo','PyTorch has no formula for this KL. The definition still works.',{builds:2,
 body:act(3,cols(mixChart(),stack(code('kl_divergence(q, mixture)'),at(2,code('NotImplementedError:\nNo KL(p || q) is implemented\nfor p type Normal and\nq type MixtureSameFamily'))),'1fr 1fr')),
 lines:['Swap the target for the PML two-bump mixture and ask PyTorch for the KL to a Gaussian q.','No formula. But KL is an average under q, and averages can be estimated by sampling.'],
 ask:['What would kl_divergence need for this pair?','Is the KL undefined, then?'],
 expected:['A closed-form expression for a Gaussian against a mixture. There isn’t one.','No. It is a perfectly good integral; we just cannot write it in closed form.'],
 point:`Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch ${d.provenance.torch}).`,
 transition:'Estimate the average by sampling.'});
S('mc-estimate','nofo','Draw from q, average the log ratio',{builds:2,
 body:stack(eq(R`\KL(q\|p)=\E_{u\sim q}\big[\log q(u)-\log p(u)\big]\approx\frac1S\sum_{i=1}^{S}\big[\log q(u_i)-\log p(u_i)\big]`),
  cols(code('u = q.sample((S,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nest = (log_q - log_p).mean()'),
  at(2,table(['','KL(q₂‖p)'],[['exact formula',num(mc.exact_kl,3)],['estimate, S = 10,000',num(mc.repeats[3].estimates[0],3)]])),'1fr 1fr')),
 lines:['Sample u from q, score each draw with both log densities, and average. No formula needed.','Check it on a pair we can solve: with 10,000 draws, 0.067 against the exact 0.069.'],
 ask:['Which distribution do the draws come from?','Should the estimate match exactly?'],
 expected:['q, the first argument: it does the weighting.','No. It is random: close, and closer with more draws.'],
 point:'Validated on q₂ = N(1, 3²) against p = N(1, 4²), seed 1000. Individual terms can be negative; see the optional slide.',
 transition:'How much does the estimate wobble?'});
S('mc-variability','nofo','More samples, tighter estimates',{
 body:image('figures/mc-variability.svg','Forty KL estimates for each of four sample counts, around the exact value.'),
 lines:['Forty repeats at each sample size. The spread shrinks like 1/√S; the exact value never moves.'],
 ask:['How many more draws halve the spread?'],
 expected:['Four times as many: the spread scales as 1/√S.'],
 point:'Same p and q at every size; 40 seeds each.',
 transition:'Act 4: we can estimate the loss. Now optimise it.'});

// ================================================================ Act 4 · Gradients through a draw
S('now-optimise','grad','Now optimise it: the code runs, and q does not move',{builds:2,
 body:act(4,cols(code('u = q.sample((5000,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-sample','A fit that uses sample(): q stays where it started'),'1fr 1.06fr')),
 lines:['Use the estimate as the loss and run the same loop on the mixture. Predict: does q move?',`No error and no warning, and q stays at N(6, 0.1²). After 400 steps the KL is still about ${Math.round(st.mixfit.sample.at(-1)[4])}.`],
 ask:['Predict: where is q after 400 steps?','The code runs. What could be wrong?'],
 expected:['Most expect it to slide onto the mixture. It does not move.','The gradient: it must be missing a piece.'],
 point:'PAUSE 7 (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.',
 transition:'Look at the gradient on a case we can solve.'});
S('missing-path','grad','The gradient that would move q never reaches it',{builds:2,
 body:stack(drawGraph(),at(2,table(['',m(R`\partial\KL/\partial\mu`),m(R`\partial\KL/\partial\sigma`)],[
  ['through sample()',num(g.detached_sample_grads[0],4),num(g.detached_sample_grads[1],4)],
  ['true gradient',num(g.analytic_grads[0],3),num(g.analytic_grads[1],3)]]))),
 lines:['sample() hands back plain numbers. Autograd cannot see how u would shift if μ or σ changed.','What survives averages to zero: 0.0013 and −0.0003, where the true gradient is −0.063 and −0.375.'],
 ask:['Which arrow can the gradient not travel along?','Why isn’t the surviving gradient exactly zero?'],
 expected:['The sample arrow from q to u.','log q(u) still contains μ and σ explicitly. That part averages to zero over draws; what remains is noise.'],
 point:'Gaussian pair q = N(0, 2²), p = N(1, 4²), 50,000 shared draws. A score-function estimator is another valid fix (optional slide).',
 transition:'Give the gradient a path.'});
S('reparam','grad','Move the randomness outside: u = μ + σε',{builds:3,
 body:cols(reparamGraph(),stack(eq(R`\ncol{\epsilon}\sim\N(0,1),\qquad u=\mu+\sigma\ncol{\epsilon}`),
  at(2,eq(R`\epsilon=-0.75,\quad\mu=0,\quad\sigma=2`)),
  at(3,eq(R`u=-1.5`)),at(3,eq(R`\frac{\partial u}{\partial\mu}=1,\qquad\frac{\partial u}{\partial\sigma}=\epsilon=-0.75`))),'1fr 1fr'),
 lines:['Draw the noise ε from a fixed N(0, 1), then shift and scale it. u has the same distribution, but is now a function of μ and σ.','Pause: with ε = −0.75, μ = 0 and σ = 2, find u, ∂u/∂μ and ∂u/∂σ.','u = −1.5, ∂u/∂μ = 1 and ∂u/∂σ = ε = −0.75. The gradient now has a path. In PyTorch this is rsample().'],
 ask:['Is u still distributed as N(μ, σ²)?','Compute u and both derivatives.','Can ∂u/∂σ be negative?'],
 expected:['Yes: shifting and scaling a standard Normal gives N(μ, σ²).','u = 0 + 2(−0.75) = −1.5; ∂u/∂μ = 1; ∂u/∂σ = ε = −0.75.','Yes, whenever ε < 0: widening q pushes this draw further left.'],
 point:'PAUSE 8 (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.',
 transition:'Change one word in the loop.'});
S('rsample-fit','grad','Change one word: rsample() fits the mixture',{builds:2,
 body:cols(code('# one word changed\nu = q.rsample((5000,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nloss = (log_q - log_p).mean()\nopt.zero_grad()\nloss.backward(); opt.step()'),
  widget('mix-rsample','A fit that uses rsample(): q moves onto the mixture'),'1fr 1.06fr'),
 lines:['Same loop, same start, with q.rsample. Press play.',`q settles at N(${num(st.mixfit.rsample.at(-1)[1])}, ${num(st.mixfit.rsample.at(-1)[2])}²), KL = ${num(st.mixfit.rsample.at(-1)[4],3)}: the best single Gaussian for this mixture. No formula was needed.`],
 ask:['Predict where q settles.','Why doesn’t the KL reach 0?'],
 expected:['Over the mixture’s mass, between and around the two bumps.','No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct optimisation.'],
 point:'Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Acts 3 and 4.',
 transition:'Act 5: every loss so far evaluated log p. For a posterior, we cannot.'});

// ================================================================ Act 5 · The real target
S('posterior-target','elbo','The real target is a posterior we cannot evaluate',{builds:2,
 body:act(5,eq(R`\KL\big(\qcol{q(u)}\,\|\,\pcol{p(u\mid\mathcal D)}\big)=\E_q\big[\log\qcol{q(u)}-\log\pcol{p(u\mid\mathcal D)}\big]`),
  at(2,eq(R`\pcol{p(u\mid\mathcal D)}=\frac{p(\mathcal D,u)}{\dcol{p(\mathcal D)}},\qquad\dcol{p(\mathcal D)}=\int p(\mathcal D,u)\,du`))),
 lines:['Every loss so far evaluated log p at the draws. Now the target is the posterior p(u | 𝒟).','But p(u | 𝒟) = p(𝒟, u) / p(𝒟), and p(𝒟) is the integral from Act 0 that nobody can do.'],
 ask:['Which pieces of this KL can we evaluate?','Which piece is missing?'],
 expected:['Draws from q, and log q at those draws.','log p(u | 𝒟): it needs the evidence p(𝒟).'],
 transition:'Substitute Bayes’ rule and see what survives.'});
S('bayes-sub','elbo','Substitute Bayes’ rule and see what survives',{builds:3,
 body:stack(eq(R`\KL\big(q\,\|\,p(u\mid\mathcal D)\big)=\E_q\big[\log q(u)-\log p(u\mid\mathcal D)\big]`),
  at(2,eq(R`=\E_q\big[\log q(u)-\log p(\mathcal D,u)\big]+\dcol{\log p(\mathcal D)}`)),
  at(3,eq(R`\dcol{\log p(\mathcal D)}=\underbrace{\E_q\big[\log p(\mathcal D,u)-\log q(u)\big]}_{\qcol{\ELBO(q)}}+\KL\big(q\,\|\,p(u\mid\mathcal D)\big)`))),
 lines:['Pause: substitute p(u | 𝒟) = p(𝒟, u) / p(𝒟). What happens to log p(𝒟)?','It does not depend on u, so it comes out of the average as a constant.','Rearrange: log p(𝒟) = ELBO + KL. The ELBO needs only the joint p(𝒟, u) = likelihood × prior.'],
 ask:['Substitute and simplify.','Why can log p(𝒟) leave the expectation?','Which term can we compute?'],
 expected:['−log p(u | 𝒟) = −log p(𝒟, u) + log p(𝒟).','It is a constant: it depends on neither u nor q.','The ELBO: draws from q, log q and the log joint.'],
 point:'PAUSE 9 (PML L23 47:15). Students do the substitution on paper before build 2.',
 transition:'What does that buy us in code?'});
S('no-evidence-needed','elbo','Maximise the ELBO; the evidence never appears',{builds:2,
 body:cols(code('w = q.rsample((256,))\nlog_q = q.log_prob(w)\nelbo = (log_joint(w) - log_q).mean()\nopt.zero_grad()\n(-elbo).backward(); opt.step()'),
  stack(table(['the loop needs','available?'],[['draws from q','yes: we chose q'],['log q(w)','yes'],['log p(𝒟, w) = log prior + log likelihood','yes']]),
  at(2,p(`${m(R`\log p(\mathcal D)`)} never appears. It does not change with q, so raising the ELBO lowers the KL.`))),'1.25fr 1fr'),
 lines:['The loop needs draws from q, log q, and the log joint: log prior plus log likelihood. All available.','log p(𝒟) is constant in q. Maximising the ELBO is minimising the KL to the posterior.'],
 ask:['Where is the evidence in this code?','What does raising the ELBO by 1 do to the KL?'],
 expected:['Nowhere.','Lowers it by exactly 1, since their sum is fixed.'],
 transition:'What does the ELBO reward?'});
S('elbo-meaning','elbo','Two forces: sit where the joint is high, stay spread out',{builds:2,
 body:stack(eq(R`\ELBO(q)=\underbrace{\E_q\big[\log p(\mathcal D,u)\big]}_{\text{sit where the joint is high}}+\underbrace{H(q)}_{\text{stay spread out}}`),
  at(2,grid3(...[0,1,2].map(k=>stack(forcePanel(k),forceCaption(k)))))),
 lines:['The first term pulls q to where prior × likelihood is high. The entropy H(q) keeps q from collapsing to a point.','Too narrow pays in entropy; too wide wanders into a low joint. The balanced q has the highest ELBO.'],
 ask:['What would maximising only the first term do?','Which q wins, and by how much?'],
 expected:['Collapse q onto the peak of the joint: a point estimate, the MAP.',`The balanced one: ELBO ${num(forces[1].elbo,2)} against ${num(forces[0].elbo,2)} and ${num(forces[2].elbo,2)}.`],
 point:'Logit coin from Act 0; q centred at the VI solution, widths 0.06, the VI width and 1.2. Expectations by quadrature.',
 transition:'Put the two pieces of the identity in one picture.'});
S('identity','elbo','One fixed total: raising the ELBO lowers the KL',{builds:2,
 body:stack(eq(R`\dcol{\log p(\mathcal D)}=\qcol{\ELBO(q)}+\lcol{\KL\big(q\,\|\,p(u\mid\mathcal D)\big)}`),identityRows(),at(2,eq(R`\ELBO(q)\le\log p(\mathcal D)`))),
 lines:['log p(𝒟) is fixed by the model and the data. Each q splits it into an ELBO and a KL gap.','As the ELBO rises the gap shrinks, never below zero: the ELBO is a lower bound on log p(𝒟).'],
 ask:['What happens to the KL when the ELBO rises by 1?','Can the ELBO pass log p(𝒟)?'],
 expected:['It falls by 1.','No: that would need a negative KL.'],
 point:'These three rows are q at steps 0, 10 and 600 of the next slide’s fit.',
 transition:'Act 6: back to the coin with no formula.'});

// ================================================================ Act 6 · Back to the wall
S('logit-vi','back','Back to the coin with no formula',{builds:2,
 body:act(6,widget('logitvi','VI on the coin with a Normal prior on the log-odds')),
 lines:['q starts at the prior, N(0, 1). Will the ELBO reach the grid’s answer, −21.62?',`Within about 30 steps q sits on the posterior. The ELBO stops ${num(st.logitvi.final.gap,4)} below log p(𝒟): that gap is KL(q ‖ posterior).`],
 ask:['Predict: will the ELBO reach the dashed line? Cross it?','The grid was used only to check. What did the optimiser use?'],
 expected:['It approaches from below and stops just short. It can never cross.','Only draws from q, log q and the log joint.'],
 point:'PAUSE 10. 256 rsample draws per step, Adam 0.05, 600 steps; exact ELBO by quadrature, for the plot only. Notebook section 8.',
 transition:'Answer the opening vote.'});
S('coin-fair','back','So, is the coin fair?',{builds:2,
 body:cols(rChart(),stack(p(`Map q back to the chance of heads, ${m(R`r=\operatorname{sigmoid}(w)`)}.`),
  at(2,table(['',m(R`P(r>0.5)`)],[['q, from VI',num(qTail)],['exact, by grid',num(exactAbove)]])))),
 lines:['Mapped back to r, q lies on top of the exact posterior.',`P(r > 0.5) ≈ ${num(qTail)}: probably biased towards heads, but a fair coin is still quite plausible. That answers the opening vote.`],
 ask:['Is the coin fair?','How did VI get this without p(𝒟)?'],
 expected:['We cannot rule it out: about a 1 in 4 chance that r < 0.5, under this model and prior.','By maximising the ELBO, which needs only the joint.'],
 point:'q in r-space includes the Jacobian 1/(r(1 − r)). The exact tail probability comes from the grid posterior.',
 transition:'Check the method on a problem with an answer key.'});
S('answer-key','back','A check with an answer key: the Beta coin',{builds:3,
 body:stack(only('1',coinPlot(0,'0')),only('2',coinPlot(0,'100')),only('3',coinPlot(0,'600'))),
 lines:['With the Beta(5, 3) prior we know the posterior: Beta(22, 16). Fit a Beta q from Beta(10, 10) using only the joint.','By step 100, q has moved most of the way.',`After 600 steps: Beta(${num(c.training.trace.at(-1).a,1)}, ${num(c.training.trace.at(-1).b,1)}), with KL ≈ ${num(c.training.trace.at(-1).posterior_kl,4)} to the true posterior.`],
 ask:['Where does Beta(10, 10) put its mass?','What changed by step 100?','Is the final q exactly Beta(22, 16)?'],
 expected:['Centred at 0.5, and too broad.','It shifted towards the heads proportion and narrowed.','No, but very close: KL ≈ 0.0002. The answer key was never used by the optimiser.'],
 point:'Beta.rsample is differentiable through implicit reparameterisation, not μ + σε. Adam 0.025, 512 draws, seed 2401.',
 transition:'Move q yourself and watch the total.'});
S('elbo-slider','back','Move q yourself: ELBO + KL never changes',{
 body:widget('elboslider','Beta q sliders with the ELBO and KL split of log evidence'),
 lines:['Drag a and b. The ELBO and the KL trade off, and their sum stays at log p(𝒟) = −21.40.'],
 ask:['Find the q with the highest ELBO. What is its KL?'],
 expected:['Beta(22, 16): ELBO = log p(𝒟) and KL = 0.'],
 point:'ELBO from analytic Beta moments; KL from its own formula; their sum is checked against the evidence in check-science.mjs.',
 transition:'Collect the five tools.'});

// ================================================================ Exit
S('five-tools','exit','Five obstacles, five tools',{
 body:table(['obstacle','tool','in the VAE (Lecture 3)'],[
  ['“closer” had no number','KL(q‖p)','the KL term in the loss'],
  ['a score cannot improve q','gradient descent on q’s parameters','training the encoder'],
  ['no formula for the KL','Monte Carlo averages','one draw per image'],
  ['no gradient through a draw','u = μ + σε, rsample()','the reparameterisation trick'],
  ['the posterior is unknown','maximise the ELBO','the VAE’s training loss']]),
 lines:['Five obstacles, five tools. In Lecture 3 a VAE uses all five at once, with one Gaussian q per image.'],
 ask:['Cover the middle column. Which tool removed which obstacle?'],
 expected:['Students reconstruct it row by row.'],
 transition:'Try the method on a new problem.'});
S('exit-transfer','exit','A sensor has an unknown constant bias',{builds:3,
 body:stack(eq(R`u\sim\N(0,1),\qquad y_i\mid u\sim\N(u,0.5^2)`),
 at(2,eq(R`p(\mathcal D,u)=p(u)\prod_i p(y_i\mid u)`)),
 at(3,eq(R`\max_\psi\E_{q_\psi(u)}[\log p(\mathcal D,u)-\log q_\psi(u)]`))),
 lines:['A sensor reads a hidden bias u, with noise. Name the unknown, the data and a q.','The joint is the prior times the likelihood of each reading: available.','Maximise the ELBO over q’s parameters. The same tools apply.'],
 ask:['Identify the unknown, the observations and one possible q family.','Which available functions form the joint?','What objective would you implement?'],
 expected:['u is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.','The prior density and the likelihood of each reading.','The expectation of log joint minus log q, maximised over ψ.'],
 point:'This conjugate toy has an exact answer too; the exercise is identifying the pieces, not needing approximation.',
 transition:'Where these unknowns appear next.'});
S('next-lecture','exit','Generative models create this inference problem again',{builds:2,
 body:stack(p('A hidden description can generate an observation.'),at(2,p('Next: learn representations for images, then ask whether they can generate new ones.'))),
 lines:['In an image model, a hidden code z can generate an observed image x.','Lecture 2 builds autoencoders; Lecture 3 returns to VI with an encoder that outputs q for each image.'],
 ask:['Where might a new unknown appear in an image model?','What should we ask of a good reconstruction?'],
 expected:['A latent description z can explain an observed image x.','Whether reconstructing observations also gives a useful way to generate new ones.'],
 transition:'Close the main route. Optional depth follows.'});

// ================================================================ Optional depth
const O=(id,title,o)=>S(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});
O('optional-one-step','One literal update of the mean',{builds:3,
 body:stack(eq(R`\sigma=2\text{ fixed}:\qquad L(\mu)=\text{const}+\frac{(\mu-1)^2}{32}`),
  at(2,eq(R`\frac{\partial L}{\partial\mu}=\frac{\mu-1}{16}=-0.0625\ \text{ at }\mu=0`)),
  at(3,eq(R`\mu^+=0-4(-0.0625)=0.25:\quad\KL\ ${num(n.one_step.kl_before,3)}\to${num(n.one_step.kl_after,3)}`))),
 ask:['Which term of the Gaussian KL depends on μ?','Differentiate it at μ = 0.','Take one step with η = 4. What is the best KL with σ stuck at 2?'],
 expected:['Only the squared shift, (μ − 1)²/(2·4²).','(μ − 1)/16 = −0.0625: a negative derivative, so descent moves μ right.',`μ⁺ = 0.25. Even at μ = 1 the KL is ${num(n.one_step.fixed_scale_best_kl,3)}: the width mismatch needs σ to move too.`],
 point:'η = 4 is chosen to make the arithmetic visible, not a recommended learning rate.'});
O('optional-normal-algebra','Where does the Gaussian KL formula come from?',{builds:3,
 body:stack(eq(R`\log\frac{q(u)}{p(u)}=\log\frac{s}{\sigma}+\frac{(u-m)^2}{2s^2}-\frac{(u-\mu)^2}{2\sigma^2}`),
 at(2,eq(R`\E_q[(u-m)^2]=\sigma^2+(\mu-m)^2,\quad\E_q[(u-\mu)^2]=\sigma^2`)),
 at(3,eq(R`\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12`))),
 ask:['Which log-normalising constants cancel?','Which two moments are needed?','Where does the final −1/2 come from?'],
 expected:['The common log√(2π) terms cancel, leaving the log scale ratio.','The target-centred and q-centred second moments.','The expectation of (u−μ)²/(2σ²) equals one half.'],
 point:'q = N(μ, σ²), p = N(m, s²). Every scale symbol is a standard deviation.'});
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
O('optional-mean-field-nets','One Gaussian per weight: Bayesian neural networks',{builds:2,
 body:stack(eq(R`q(\mathbf w)=\prod_{j}\N(w_j;\,\mu_j,\sigma_j^2)`),
 at(2,eq(R`\ELBO=\E_q\big[\log p(\mathcal D\mid\mathbf w)\big]-\KL\big(q(\mathbf w)\,\|\,p(\mathbf w)\big)`))),
 ask:['How many parameters does q have for a network with d weights?','How would you train it?'],
 expected:['2d: a mean and a width per weight. A full covariance would need about d²/2.','Exactly as today: rsample the weights, estimate the ELBO, backpropagate.'],
 point:'Mean-field: the diagonal q from Act 2, so expect underestimated uncertainty.'});
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
