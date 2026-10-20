// Live figures for Lecture 1, and the small SVG chart kit that the static slides share with them.
// Every curve comes from data/science.json (the executed notebook) or from the formulas in math.mjs.
// Print and PDF show static states; presentation mode ties each animation to a build.
import {palette,esc} from '../shared/author.js';
import {normalPdf,normalLogPdf,normalKL,betaPdf,betaKL,coinELBO,coinLogEvidence} from './math.mjs';

export const MUTED='#68717B',AX='#AAB3BC';
export const TINT={p:'#0F766E24',q:'#8B2CDE24',loss:'#BE123C2E',data:'#245EDB24',noise:'#AA4E0824',grey:'#E5E7EB'};
export const lin=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
/** Fixed decimals, a true minus sign, and no "−0.00". */
export const num=(v,n=2)=>{const s=Number(v).toFixed(n);return (/^-0(\.0*)?$/.test(s)?s.slice(1):s).replace('-','−');};
let uid=0;

/** One plotting panel, optionally offset inside a larger SVG. Data outside the axes is clipped. */
export function panel(o){
 const ox=o.ox??0,oy=o.oy??0,W=o.W??600,H=o.H??320,L=o.L??64,R=o.R??14,T=o.T??56,B=o.B??56,fs=o.fs??20;
 const {xMin,xMax}=o,yMin=o.yMin??0,yMax=o.yMax;
 const x=v=>ox+L+(v-xMin)/(xMax-xMin)*(W-L-R),y=v=>oy+H-B-(v-yMin)/(yMax-yMin)*(H-T-B);
 const id=`wclip-${++uid}`,tick=v=>String(+Number(v).toFixed(2)).replace('-','−');
 const base=yMin<0&&yMax>0?y(0):oy+H-B,tickY=o.ticksAtAxis?base+24:oy+H-B+24;
 let axes=`<defs><clipPath id="${id}"><rect x="${ox+L}" y="${oy+T-10}" width="${W-L-R}" height="${H-B-T+10}"/></clipPath></defs>`;
 axes+=`<line x1="${ox+L}" y1="${base}" x2="${ox+W-R}" y2="${base}" stroke="${AX}"/>`;
 if(o.yAxis!==false)axes+=`<line x1="${ox+L}" y1="${oy+T}" x2="${ox+L}" y2="${oy+H-B}" stroke="${AX}"/>`;
 for(const v of o.ticksX||[])axes+=`<text x="${x(v).toFixed(1)}" y="${tickY.toFixed(1)}" text-anchor="middle" fill="${MUTED}" font-size="${fs}">${(o.fmtX||tick)(v)}</text>`;
 for(const v of o.ticksY||[])axes+=`<text x="${ox+L-9}" y="${(y(v)+7).toFixed(1)}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${(o.fmtY||tick)(v)}</text>`;
 if(o.xLabel)axes+=`<text x="${ox+W-R}" y="${oy+H-9}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${esc(o.xLabel)}</text>`;
 if(o.yLabel)axes+=`<text x="${ox+L}" y="${oy+T-16}" fill="${MUTED}" font-size="${fs}">${esc(o.yLabel)}</text>`;
 const pts=(xs,ys)=>{let s='';for(let i=0;i<xs.length;i++){const v=y(ys[i]);s+=`${x(xs[i]).toFixed(1)},${(Number.isFinite(v)?Math.max(oy-600,Math.min(oy+H+600,v)):oy+H-B).toFixed(1)} `;}return s;};
 return {x,y,ox,oy,W,H,L,R,T,B,axes,
  line:(xs,ys,color,extra='')=>`<polyline clip-path="url(#${id})" points="${pts(xs,ys)}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" ${extra}/>`,
  area:(xs,ys,fill,at=Math.max(yMin,0))=>`<polygon clip-path="url(#${id})" points="${x(xs[0]).toFixed(1)},${y(at).toFixed(1)} ${pts(xs,ys)}${x(xs.at(-1)).toFixed(1)},${y(at).toFixed(1)}" fill="${fill}"/>`,
  dot:(vx,vy,color,r=6)=>`<circle cx="${x(vx).toFixed(1)}" cy="${y(vy).toFixed(1)}" r="${r}" fill="${color}" stroke="${palette.paper}" stroke-width="2"/>`,
 };
}
/** Text in an SVG; content must already be safe (no markup from data). */
export const label=(tx,ty,s,{fill=MUTED,size=22,anchor='start',weight=''}={})=>`<text x="${(+tx).toFixed(1)}" y="${(+ty).toFixed(1)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}"${weight?` font-weight="${weight}"`:''}>${s}</text>`;
export const svg=(W,H,inner,alt)=>`<svg class="plot wsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg>`;
export const fig=(W,H,inner,alt)=>`<div class="diagram-scroll">${svg(W,H,inner,alt)}</div>`;
export const widget=(name,alt)=>`<div class="widget" data-widget="${name}" role="group" aria-label="${esc(alt)}"></div>`;
const legend=(x0,y0,items,size=21)=>{let x=x0,s='';for(const [text,color,dashed] of items){s+=`<line x1="${x}" y1="${y0-7}" x2="${x+26}" y2="${y0-7}" stroke="${color}" stroke-width="3"${dashed?' stroke-dasharray="7 4"':''}/>`+label(x+34,y0,text,{fill:color,size});x+=56+text.length*size*.52;}return s;};
export {legend};

const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * A recorded optimisation, replayed frame by frame. Presentation: frame 0 until the play build,
 * which plays it (arriving there directly shows the final frame). Reading: final frame, with controls.
 * Print: frame 0 before the play build and the final frame from it, as static states.
 */
function animated(el,ctx,{count,render,step,alt,playBuild=2,duration=4200,extra=''}){
 const last=count-1;
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1-${playBuild-1}">${render(0)}</div><div data-only="${playBuild}-99">${render(last)}</div></div>`;return {rerender(){}};}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><button type="button" class="w-play">▶ Play</button><input class="w-scrub" type="range" min="0" max="${last}" step="1" value="0" aria-label="${esc(alt)}: optimiser step"><span class="w-readout w-step"></span>${extra}</div>`;
 const box=el.querySelector('.w-fig'),btn=el.querySelector('.w-play'),scrub=el.querySelector('.w-scrub'),out=el.querySelector('.w-step');
 let i=0,raf=0;
 const show=k=>{i=Math.max(0,Math.min(last,Math.round(k)));box.innerHTML=render(i);scrub.value=String(i);out.textContent=step(i);};
 const idle=k=>{cancelAnimationFrame(raf);raf=0;if(k!==undefined)show(k);btn.textContent=i>=last?'↺ Replay':'▶ Play';};
 const play=()=>{
  cancelAnimationFrame(raf);if(i>=last)show(0);
  if(reduced())return idle(last);
  const from=i,t0=performance.now();btn.textContent='❚❚ Pause';
  const tick=t=>{const k=from+(t-t0)/duration*last;show(k);if(k<last)raf=requestAnimationFrame(tick);else idle();};
  raf=requestAnimationFrame(tick);
 };
 btn.addEventListener('click',()=>raf?idle():play());
 scrub.addEventListener('input',()=>idle(Number(scrub.value)));
 scrub.addEventListener('change',()=>scrub.blur());
 const scene=el.closest('.scene');
 if(ctx.reading||!scene)idle(last);
 else{
  idle(0);let fresh=false,prev=0;
  new MutationObserver(records=>{for(const r of records){
   if(r.attributeName==='class'){if(scene.classList.contains('current'))fresh=true;else idle();continue;}
   const b=Number(scene.dataset.build||1);
   if(b<playBuild)idle(0);else if(fresh)idle(last);else if(prev===playBuild-1&&b===playBuild)play();
   fresh=false;prev=b;
  }}).observe(scene,{attributes:true,attributeFilter:['class','data-build']});
 }
 return {rerender:()=>show(i)};
}

/** A figure redrawn from sliders and buttons. Print shows the starting state without controls. */
function interactive(el,ctx,{controls,render}){
 if(ctx.printing){el.innerHTML=render();return null;}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls">${controls}</div>`;
 const box=el.querySelector('.w-fig'),draw=()=>{box.innerHTML=render();};
 for(const input of el.querySelectorAll('input[type=range]'))input.addEventListener('change',()=>input.blur());
 draw();return draw;
}
const slider=(key,text,min,max,stepSize,value)=>`<label>${text}<input type="range" min="${min}" max="${max}" step="${stepSize}" value="${value}" data-k="${key}" aria-label="${esc(text)}"><span class="w-readout" data-o="${key}">${num(value)}</span></label>`;
function bindSliders(el,state,draw,after=()=>{}){
 if(!draw)return;
 for(const input of el.querySelectorAll('[data-k]'))input.addEventListener('input',()=>{state[input.dataset.k]=Number(input.value);sync(el,state);after();draw();});
}
function sync(el,state){
 for(const input of el.querySelectorAll('[data-k]'))input.value=String(state[input.dataset.k]);
 for(const o of el.querySelectorAll('[data-o]'))o.textContent=num(state[o.dataset.o]);
}
/** A KL value for display: just the number, or with its relation sign ("= 0.35", "≈ 0"). */
const klv=v=>v<5e-4?'≈ 0':v>=10?num(v,0):num(v,v<.1?3:2),kl=v=>v<5e-4?'≈ 0':'= '+klv(v);

// ---- Act 1: KL as a net area -------------------------------------------------------------
function klarea(el,ctx){
 const P={mu:1,sigma:4},xs=lin(-15,17,401),lp=xs.map(u=>normalLogPdf(u,P)),py=lp.map(Math.exp);
 const s={mu:0,sigma:2,dir:'qp'},W=1150,H=330;
 const render=()=>{
  const q={mu:s.mu,sigma:s.sigma},lq=xs.map(u=>normalLogPdf(u,q)),qy=lq.map(Math.exp);
  const fwd=s.dir==='pq',f=xs.map((u,i)=>fwd?py[i]*(lp[i]-lq[i]):qy[i]*(lq[i]-lp[i]));
  const value=fwd?normalKL(P,q):normalKL(q,P),hi=Math.max(...f),lo=Math.min(...f);
  const top=Math.max(hi*1.12,.012),bottom=Math.min(lo*1.12,-top*.16);
  const A=panel({W:560,H,L:16,R:14,T:76,B:56,xMin:-15,xMax:17,yMax:.3,ticksX:[-10,0,10],yAxis:false,xLabel:'u'});
  const Bp=panel({ox:590,W:560,H,L:16,R:14,T:76,B:56,xMin:-15,xMax:17,yMin:bottom,yMax:top,ticksX:[-10,0,10],yAxis:false,xLabel:'u'});
  const name=fwd?'KL(p‖q)':'KL(q‖p)',w=fwd?'p':'q',o=fwd?'q':'p';
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,qy,palette.q)+
   legend(16,30,[['p = N(1, 4²)',palette.p],[`q = N(${num(s.mu)}, ${num(s.sigma)}²)`,palette.q]])+
   Bp.axes+Bp.area(xs,f.map(v=>Math.max(v,0)),'#8B2CDE47',0)+Bp.area(xs,f.map(v=>Math.min(v,0)),'#BE123C47',0)+
   Bp.line(xs,f,palette.ink,'stroke-width="2"')+
   label(606,30,`net area = ${name} ${kl(value)}`,{fill:palette.ink,size:23,weight:600})+
   label(606,60,`${w} log(${w}/${o}): purple adds, red subtracts`,{size:19}),
   `Densities p and q, and the integrand of ${name}; its net area is the KL.`);
 };
 const controls=slider('mu','μ',-3,5,.05,0)+slider('sigma','σ',.4,4.5,.05,2)+
  `<button type="button" data-q="0,2">q₁</button><button type="button" data-q="1,3">q₂</button><button type="button" data-q="1,4">q = p</button>`+
  `<button type="button" data-dir="qp" aria-pressed="true">KL(q‖p)</button><button type="button" data-dir="pq" aria-pressed="false">KL(p‖q)</button>`;
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw);
 el.querySelectorAll('[data-q]').forEach(b=>b.addEventListener('click',()=>{[s.mu,s.sigma]=b.dataset.q.split(',').map(Number);sync(el,s);draw();}));
 el.querySelectorAll('[data-dir]').forEach(b=>b.addEventListener('click',()=>{s.dir=b.dataset.dir;el.querySelectorAll('[data-dir]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
}

// ---- Act 1: forward versus reverse KL on two bumps ----------------------------------------
function fwdrev(el,ctx,d){
 const st=d.story.fwdrev,xs=st.x,W=1150,H=300,PW=370,GAP=20;
 const runs=[['fwd','KL(p‖q)'],['revl','KL(q‖p), start left'],['revr','KL(q‖p), start right']];
 let width='0.5';
 const render=i=>{let inner='';const run=st[width];
  runs.forEach(([key,title],j)=>{
   const ox=j*(PW+GAP),P=panel({ox,W:PW,H,L:14,R:12,T:62,B:52,xMin:-6,xMax:6,yMax:.95,ticksX:[-4,-2,0,2,4],yAxis:false});
   const [,m,s,v]=run[key][i];
   inner+=P.axes+P.area(xs,run.p,TINT.p)+P.line(xs,run.p,palette.p)+P.line(xs,xs.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
    label(ox+14,28,title,{fill:palette.ink,size:22,weight:600})+label(ox+PW-12,28,klv(v),{fill:palette.q,size:22,anchor:'end'});
  });
  return svg(W,H,inner,'Three Gaussian fits (purple) to a two-bump target (teal): forward KL, and reverse KL from two starts.');
 };
 const toggles=`<button type="button" data-w="0.5" aria-pressed="true">Narrow bumps</button><button type="button" data-w="1.0" aria-pressed="false">Wide bumps</button>`;
 const a=animated(el,ctx,{count:st['0.5'].fwd.length,render,step:i=>`step ${st['0.5'].fwd[i][0]}`,alt:'Forward and reverse KL fits',extra:toggles});
 el.querySelectorAll('[data-w]').forEach(b=>b.addEventListener('click',()=>{width=b.dataset.w;el.querySelectorAll('[data-w]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));a.rerender();}));
}

// ---- Act 2: q slides onto p -----------------------------------------------------------------
function olive(el,ctx,d){
 const fr=d.story.olive.frames,xs=lin(-11,13,241),py=xs.map(u=>normalPdf(u,{mu:1,sigma:4})),W=1150,H=320;
 const render=i=>{const [step,m,s,v]=fr[i],tr=fr.slice(0,i+1);
  const A=panel({W:700,H,L:58,R:14,T:62,B:56,xMin:-11,xMax:13,yMax:.25,ticksX:[-10,-5,0,5,10],ticksY:[0,.1,.2],xLabel:'u'});
  const K=panel({ox:740,W:410,H,L:52,R:26,T:62,B:56,xMin:0,xMax:200,yMax:2,ticksX:[0,100,200],ticksY:[0,1,2],xLabel:'step'});
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,xs.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
   legend(58,30,[['p = N(1, 4²)',palette.p],[`q = N(${num(m)}, ${num(s)}²)`,palette.q]])+
   K.axes+K.line(tr.map(r=>r[0]),tr.map(r=>r[3]),palette.loss)+K.dot(step,v,palette.loss)+
   label(792,30,`KL(q‖p) ${kl(v)}`,{fill:palette.loss,size:22}),
   'The Gaussian q (purple) moving onto the fixed target p (teal), with the KL falling.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'q moving onto p'});
}

// ---- Act 4: Monte Carlo KL on the mixture, with sample() or rsample() ------------------------
function mixfit(el,ctx,d,run){
 const st=d.story.mixfit,fr=st[run],xs=st.x,py=st.p,W=600,H=330;
 const render=i=>{const [,m,s,,v]=fr[i];
  const A=panel({W,H,L:16,R:14,T:62,B:56,xMin:-4,xMax:8,yMax:1.25,ticksX:[-4,-2,0,2,4,6,8],yAxis:false,xLabel:'u'});
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,xs.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
   label(A.x(-.2),A.y(.62),'p',{fill:palette.p,anchor:'middle'})+
   label(16,30,`q = N(${num(m)}, ${num(s)}²)`,{fill:palette.q,size:23})+label(W-14,30,`KL ${kl(v)}`,{fill:palette.loss,size:23,anchor:'end'}),
   'A Gaussian q (purple) and the two-bump mixture p (teal).');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:run==='sample'?'Fit with sample()':'Fit with rsample()'});
}

// ---- Act 6: VI on the coin with no formula ----------------------------------------------------
function logitvi(el,ctx,d){
 const st=d.story.logitvi,fr=st.frames,w=d.story.wall.plot.w,post=d.story.wall.plot.posterior,ev=st.final.log_evidence_grid,W=1150,H=320;
 const render=i=>{const [step,m,s,,elbo]=fr[i],tr=fr.slice(0,i+1);
  const A=panel({W:560,H,L:16,R:14,T:62,B:56,xMin:-3,xMax:3.5,yMax:1.3,ticksX:[-2,-1,0,1,2,3],yAxis:false,xLabel:'w, log-odds of heads'});
  const K=panel({ox:600,W:550,H,L:66,R:26,T:62,B:56,xMin:0,xMax:600,yMin:-24.6,yMax:-21.2,ticksX:[0,200,400,600],ticksY:[-24,-23,-22],xLabel:'step'});
  return svg(W,H,A.axes+A.area(w,post,TINT.p)+A.line(w,post,palette.p)+A.line(w,w.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
   legend(16,30,[['posterior',palette.p],[`q = N(${num(m)}, ${num(s)}²)`,palette.q]])+
   K.axes+`<line x1="${K.x(0)}" y1="${K.y(ev).toFixed(1)}" x2="${K.x(600)}" y2="${K.y(ev).toFixed(1)}" stroke="${palette.data}" stroke-width="2.5" stroke-dasharray="8 6"/>`+
   label(K.x(600),K.y(ev)+30,`log p(𝒟) = ${num(ev)}, by grid`,{fill:palette.data,size:20,anchor:'end'})+
   K.line(tr.map(r=>r[0]),tr.map(r=>r[4]),palette.q)+K.dot(step,elbo,palette.q)+
   label(666,30,`ELBO = ${num(elbo,3)}`,{fill:palette.q,size:22})+label(1136,30,`gap = KL = ${num(Math.max(ev-elbo,0),4)}`,{fill:palette.loss,size:22,anchor:'end'}),
   'q (purple) fitted to the logit-coin posterior (teal), and its ELBO rising to just below log evidence.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'VI on the logit coin'});
}

// ---- Act 6: move a Beta q yourself; ELBO + KL is fixed ----------------------------------------
function elboslider(el,ctx){
 const prior={alpha:5,beta:3},data={heads:17,tails:13},post={alpha:22,beta:16},ev=coinLogEvidence(prior,data);
 const rs=lin(.002,.998,300),py=rs.map(r=>betaPdf(r,post)),s={a:10,b:10},W=1150,H=330;
 const render=()=>{
  const q={alpha:s.a,beta:s.b},e=coinELBO(q,prior,data).elbo,k=betaKL(q,post);
  const A=panel({W:560,H,L:16,R:14,T:62,B:56,xMin:0,xMax:1,yMax:7.5,ticksX:[0,.25,.5,.75,1],yAxis:false,xLabel:'r, chance of heads'});
  const lo=-25,hi=-21,X=v=>640+(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*450,line=170,xe=X(e),xv=X(ev);
  let right=`<line x1="${X(lo)}" y1="${line}" x2="${X(hi)}" y2="${line}" stroke="${AX}"/>`;
  for(const t of [-25,-24,-23,-22,-21])right+=`<line x1="${X(t)}" y1="${line-6}" x2="${X(t)}" y2="${line+6}" stroke="${AX}"/>`+label(X(t),line+32,num(t,0),{size:19,anchor:'middle'});
  right+=`<path d="M${xe.toFixed(1)},${line-30} V${line-44} H${xv.toFixed(1)} V${line-30}" fill="none" stroke="${palette.loss}" stroke-width="2.5"/>`;
  right+=label(Math.min(Math.max((xe+xv)/2,720),1010),line-56,`gap = KL = ${num(k,3)}`,{fill:palette.loss,size:21,anchor:'middle'});
  right+=`<circle cx="${xv.toFixed(1)}" cy="${line}" r="8" fill="${palette.data}"/><circle cx="${xe.toFixed(1)}" cy="${line}" r="8" fill="${palette.q}"/>`;
  right+=label(Math.min(Math.max(xe,700),1080),line+74,`ELBO = ${num(e,3)}${e<lo?' (off the scale)':''}`,{fill:palette.q,size:21,anchor:'middle'});
  right+=label(1136,line-96,`log p(𝒟) = ${num(ev,3)}`,{fill:palette.data,size:21,anchor:'end'});
  right+=label(620,30,'One fixed total',{fill:palette.ink,size:23,weight:600});
  right+=label(620,H-16,`${num(e,3)} + ${num(k,3)} = ${num(e+k,3)}`,{fill:palette.ink,size:22});
  return svg(W,H,A.axes+A.area(rs,py,TINT.p)+A.line(rs,py,palette.p)+A.line(rs,rs.map(r=>betaPdf(r,q)),palette.q)+
   legend(16,30,[['posterior Beta(22, 16)',palette.p],[`q = Beta(${num(s.a,1)}, ${num(s.b,1)})`,palette.q]])+right,
   'A Beta q against the exact posterior, and log evidence split into the ELBO and the KL gap.');
 };
 const controls=slider('a','a',1,60,.5,10)+slider('b','b',1,60,.5,10)+
  ['10,10','30,20','22,16'].map(v=>`<button type="button" data-ab="${v}">Beta(${v.replace(',',', ')})</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw);
 el.querySelectorAll('[data-ab]').forEach(b=>b.addEventListener('click',()=>{[s.a,s.b]=b.dataset.ab.split(',').map(Number);sync(el,s);draw();}));
}

export function mount(stage,ctx,d){
 const make={klarea,fwdrev,olive,logitvi,elboslider,'mix-sample':(el,c)=>mixfit(el,c,d,'sample'),'mix-rsample':(el,c)=>mixfit(el,c,d,'rsample')};
 for(const el of stage.querySelectorAll('[data-widget]')){
  const f=make[el.dataset.widget];if(!f)throw new Error('Unknown widget '+el.dataset.widget);f(el,ctx,d);
 }
}
