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
  band:(xs,lo,hi,fill)=>`<polygon clip-path="url(#${id})" points="${pts(xs,hi)}${pts([...xs].reverse(),[...lo].reverse())}" fill="${fill}"/>`,
  dot:(vx,vy,color,r=6)=>`<circle cx="${x(vx).toFixed(1)}" cy="${y(vy).toFixed(1)}" r="${r}" fill="${color}" stroke="${palette.paper}" stroke-width="2"/>`,
 };
}
/** Text in an SVG; content must already be safe (no markup from data). */
export const label=(tx,ty,s,{fill=MUTED,size=22,anchor='start',weight=''}={})=>`<text x="${(+tx).toFixed(1)}" y="${(+ty).toFixed(1)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}"${weight?` font-weight="${weight}"`:''}>${s}</text>`;
export const svg=(W,H,inner,alt)=>`<svg class="plot wsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg>`;
export const fig=(W,H,inner,alt)=>`<div class="diagram-scroll">${svg(W,H,inner,alt)}</div>`;
export const widget=(name,alt)=>`<div class="widget" data-widget="${name}" role="group" aria-label="${esc(alt)}"></div>`;
const legend=(x0,y0,items,size=21)=>{let x=x0,s='';for(const [text,color,dashed] of items){s+=`<line x1="${x}" y1="${y0-7}" x2="${x+26}" y2="${y0-7}" stroke="${color}" stroke-width="3"${dashed?' stroke-dasharray="7 4"':''}/>`+label(x+34,y0,text,{fill:color,size});x+=56+text.replace(/<[^>]+>/g,'').length*size*.52;}return s;};
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
  const A=panel({W:560,H,L:16,R:14,T:76,B:56,xMin:-15,xMax:17,yMax:.3,ticksX:[-10,0,10],yAxis:false,xLabel:'θ'});
  const Bp=panel({ox:590,W:560,H,L:16,R:14,T:76,B:56,xMin:-15,xMax:17,yMin:bottom,yMax:top,ticksX:[-10,0,10],yAxis:false,xLabel:'θ'});
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
  const A=panel({W:700,H,L:58,R:14,T:62,B:56,xMin:-11,xMax:13,yMax:.25,ticksX:[-10,-5,0,5,10],ticksY:[0,.1,.2],xLabel:'θ'});
  const K=panel({ox:740,W:410,H,L:52,R:26,T:62,B:56,xMin:0,xMax:200,yMax:2,ticksX:[0,100,200],ticksY:[0,1,2],xLabel:'step'});
  const ghosts=[0,10,25,50,100].filter(k=>k<step).map(k=>A.line(xs,xs.map(u=>normalPdf(u,{mu:fr[k][1],sigma:fr[k][2]})),palette.q,'style="stroke-width:2;stroke-opacity:.22"')).join('');
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+ghosts+A.line(xs,xs.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
   legend(58,30,[['p = N(1, 4²)',palette.p],[`q = N(${num(m)}, ${num(s)}²)`,palette.q]])+
   K.axes+K.line(fr.map(r=>r[0]),fr.map(r=>r[3]),'#D1D5DB','style="stroke-width:2"')+K.line(tr.map(r=>r[0]),tr.map(r=>r[3]),palette.loss)+K.dot(step,v,palette.loss)+
   label(792,30,`loss = KL(q‖p) ${kl(v)}`,{fill:palette.loss,size:22}),
   'The Gaussian q (purple) moving onto the fixed target p (teal), with faint earlier positions, and the loss falling.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'q moving onto p'});
}

// ---- Act 4: Monte Carlo KL on the mixture, with sample() or rsample() ------------------------
function mixfit(el,ctx,d,run){
 const st=d.story.mixfit,fr=st[run],xs=st.x,py=st.p,W=600,H=330;
 const render=i=>{const [,m,s,,v]=fr[i];
  const A=panel({W,H,L:16,R:14,T:62,B:56,xMin:-4,xMax:8,yMax:1.25,ticksX:[-4,-2,0,2,4,6,8],yAxis:false,xLabel:'θ'});
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,xs.map(u=>normalPdf(u,{mu:m,sigma:s})),palette.q)+
   label(A.x(-.2),A.y(.62),'p',{fill:palette.p,anchor:'middle'})+
   label(16,30,`q = N(${num(m)}, ${num(s)}²)`,{fill:palette.q,size:23})+label(W-14,30,`KL ${kl(v)}`,{fill:palette.loss,size:23,anchor:'end'}),
   'A Gaussian q (purple) and the two-bump mixture p (teal).');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:run==='sample'?'Fit with sample()':'Fit with rsample()'});
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

// ---- Part 1: twenty prior draws of θ, each one function; hover to see its point in ℝ⁷ ------------
export const PARAM_NAMES=['w₁','w₂','b₁','b₂','v₁','v₂','c'];
export const supDigits=n=>String(n).replace(/\d/g,c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
/** One point θ ∈ ℝ⁷ as seven labelled bars on a −3.3 … 3.3 scale, values printed on the right. */
export function thetaBars(ox,oy,theta,color,{rowH=31,labelW=44,barW=300,size=21}={}){
 const X=v=>ox+labelW+(Math.max(-3.3,Math.min(3.3,v))+3.3)/6.6*barW,zero=X(0);
 let s=`<line x1="${zero.toFixed(1)}" y1="${oy-22}" x2="${zero.toFixed(1)}" y2="${oy+6*rowH+12}" stroke="${AX}"/>`+label(zero,oy-28,'0',{size:16,anchor:'middle'});
 theta.forEach((v,j)=>{const y=oy+j*rowH,x1=X(v);
  s+=label(ox,y+7,PARAM_NAMES[j],{fill:palette.noise,size,weight:600})+
   `<rect x="${Math.min(zero,x1).toFixed(1)}" y="${y-9}" width="${Math.max(1.5,Math.abs(x1-zero)).toFixed(1)}" height="18" rx="3" fill="${color}"/>`+
   label(ox+labelW+barW+14,y+7,num(v),{fill:color,size});});
 return s;
}
function priorfuncs(el,ctx,d){
 const b=d.story.bnn,xs=b.grid,F=b.prior_draws,T=b.prior_thetas,pp=b.prior_predictive,W=1150,H=320;
 let cur=0,build=1;
 const render=()=>{
  const P=panel({W:600,H,L:52,R:14,T:56,B:56,xMin:-2.5,xMax:2.5,yMin:-4.2,yMax:4.2,ticksX:[-2,-1,0,1,2],ticksY:[-4,-2,0,2,4],xLabel:'x'});
  let s=P.axes;
  if(build>=2)s+=P.band(xs,pp.mean.map((m,k)=>m-2*pp.sd[k]),pp.mean.map((m,k)=>m+2*pp.sd[k]),'#AA4E0820');
  F.forEach((fy,k)=>{if(k!==cur)s+=P.line(xs,fy,'#9CA3AF','style="stroke-width:2;stroke-opacity:.8"');});
  s+=P.line(xs,F[cur],palette.noise,'style="stroke-width:4.5"');
  F.forEach((fy,k)=>{s+=P.line(xs,fy,'transparent',`data-k="${k}" style="stroke-width:18;pointer-events:stroke;cursor:pointer"`);});
  s+=label(52,30,build>=2?'20 draws, and the prior predictive: mean ± 2 sd':'f(x) for 20 draws of θ from the prior',{fill:build>=2?palette.noise:MUTED,size:21});
  s+=label(640,30,`draw ${cur+1} of 20: θ⁽${supDigits(cur+1)}⁾ ∈ ℝ⁷`,{fill:palette.noise,size:22,weight:600})+thetaBars(650,92,T[cur],palette.noise);
  s+=label(640,H-12,'Hover over a line, or step through the draws.',{size:18});
  return svg(W,H,s,'Twenty functions drawn from the prior; the highlighted one is shown as its seven parameter values.');
 };
 if(ctx.printing){const first=render();build=2;el.innerHTML=`<div class="state-slot"><div data-only="1">${first}</div><div data-only="2-99">${render()}</div></div>`;return;}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><button type="button" data-step="-1">‹ previous draw</button><button type="button" data-step="1">next draw ›</button></div>`;
 const box=el.querySelector('.w-fig'),scene=el.closest('.scene'),draw=()=>{box.innerHTML=render();};
 build=ctx.reading||!scene?2:Number(scene.dataset.build||1);
 if(scene&&!ctx.reading)new MutationObserver(()=>{const nb=Number(scene.dataset.build||1);if(nb!==build){build=nb;draw();}}).observe(scene,{attributes:true,attributeFilter:['data-build']});
 box.addEventListener('pointerover',e=>{const k=e.target.closest?.('[data-k]')?.dataset.k;if(k!==undefined&&Number(k)!==cur){cur=Number(k);draw();}});
 el.querySelectorAll('[data-step]').forEach(btn=>btn.addEventListener('click',()=>{cur=(cur+Number(btn.dataset.step)+F.length)%F.length;draw();}));
 draw();
}

// ---- Part 1: training the network to find θ_A (ten random starts, the best one animated) --------
function mapfit(el,ctx,d){
 const b=d.story.bnn,m=b.map,fr=m.frames,xs=b.grid,W=1150,H=280,lg=Math.log10;
 const steps=m.objectives[0].map((_,k)=>k*m.record_every),best=m.objectives[m.best_seed];
 const render=i=>{const [step,obj,fy]=fr[i],upto=Math.round(step/m.record_every);
  const A=panel({W:600,H,L:52,R:14,T:50,B:52,xMin:-2.5,xMax:2.5,yMin:-2.6,yMax:2.6,ticksX:[-2,-1,0,1,2],ticksY:[-2,0,2],xLabel:'x'});
  const K=panel({ox:640,W:510,H,L:76,R:26,T:50,B:52,xMin:0,xMax:2000,yMin:1,yMax:4,ticksX:[0,1000,2000],ticksY:[1,2,3,4],fmtY:v=>['','10','100','1,000','10,000'][v],xLabel:'step'});
  let s=A.axes+A.line(xs,fy,palette.q)+b.x.map((x,k)=>`<circle cx="${A.x(x).toFixed(1)}" cy="${A.y(b.y[k]).toFixed(1)}" r="5" fill="${palette.data}"/>`).join('')+
   label(52,26,'the network during training',{fill:palette.q,size:20})+K.axes;
  m.objectives.forEach((o,r)=>{if(r!==m.best_seed)s+=K.line(steps,o.map(lg),'#9CA3AF','style="stroke-width:1.5;stroke-opacity:.8"');});
  s+=K.line(steps.slice(0,upto+1),best.slice(0,upto+1).map(lg),palette.q)+K.dot(step,lg(obj),palette.q)+
   label(716,26,`loss ${obj>=100?Math.round(obj).toLocaleString('en'):num(obj,1)} · grey: the other nine starts`,{fill:palette.q,size:20});
  return svg(W,H,s,'The network’s curve during training from a random start, and the loss of all ten starts.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'Training the network',duration:5000});
}

// ---- Part 2: a Monte Carlo KL estimate, one draw at a time ---------------------------------
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const gauss=r=>{let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());};
function mclive(el,ctx){
 const P={mu:1,sigma:4},Q={mu:0,sigma:2},exact=normalKL(Q,P),xs=lin(-15,17,321),py=xs.map(u=>normalPdf(u,P)),qy=xs.map(u=>normalPdf(u,Q)),W=1150,H=320;
 let rng,draws,sum,trace;
 const reset=start=>{rng=mulberry32(20261020);draws=[];sum=0;trace=[];add(start);};
 function add(k){for(let i=0;i<k;i++){const u=Q.mu+Q.sigma*gauss(rng);sum+=normalLogPdf(u,Q)-normalLogPdf(u,P);draws.push(u);
  const S=draws.length;if(S<=100||Math.floor(Math.log10(S)*80)!==Math.floor(Math.log10(S-1)*80))trace.push([Math.log10(S),sum/S]);}}
 const render=()=>{
  const S=draws.length,est=S?sum/S:NaN;
  const A=panel({W:560,H,L:16,R:14,T:62,B:56,xMin:-15,xMax:17,yMax:.3,ticksX:[-10,0,10],yAxis:false,xLabel:'θ'});
  const K=panel({ox:600,W:550,H,L:62,R:26,T:62,B:56,xMin:0,xMax:4.3,yMin:-.6,yMax:1.4,ticksX:[0,1,2,3,4],fmtX:v=>['1','10','100','1,000','10,000'][v],ticksY:[-.5,0,.5,1],xLabel:'draws S (log scale)'});
  const rug=draws.slice(-300).map(u=>`<line x1="${A.x(u).toFixed(1)}" y1="${A.y(0)-1}" x2="${A.x(u).toFixed(1)}" y2="${A.y(0)-15}" stroke="${palette.q}" stroke-opacity=".45" stroke-width="1.5"/>`).join('');
  return svg(W,H,A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,qy,palette.q)+rug+
   legend(16,30,[['p = N(1, 4²)',palette.p],['q = N(0, 2²), draws as ticks',palette.q]])+
   K.axes+`<line x1="${K.x(0)}" y1="${K.y(exact).toFixed(1)}" x2="${K.x(4.3)}" y2="${K.y(exact).toFixed(1)}" stroke="${palette.p}" stroke-width="2.5" stroke-dasharray="8 6"/>`+
   label(K.x(4.3),K.y(exact)-10,`exact ${num(exact,3)}`,{fill:palette.p,size:20,anchor:'end'})+
   (trace.length>1?K.line(trace.map(t=>t[0]),trace.map(t=>t[1]),palette.q):'')+(S?K.dot(Math.log10(S),est,palette.q):'')+
   label(662,30,S?`S = ${S.toLocaleString('en')}: estimate ${num(est,3)}`:'No draws yet',{fill:palette.q,size:22}),
   'Draws from q, and the running Monte Carlo estimate of KL(q‖p) against its exact value.');
 };
 if(ctx.printing){reset(10000);el.innerHTML=render();return;}
 reset(10);
 const controls=[1,10,100,1000].map(k=>`<button type="button" data-add="${k}">+${k.toLocaleString('en')} draw${k>1?'s':''}</button>`).join('')+`<button type="button" data-reset>Reset</button><span class="w-readout">exact KL(q‖p) = ${num(exact,3)}</span>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-add]').forEach(b=>b.addEventListener('click',()=>{if(draws.length<20000)add(Number(b.dataset.add));draw();}));
 el.querySelector('[data-reset]').addEventListener('click',()=>{reset(10);draw();});
}

// ---- Part 2: KL as an expectation, estimated by sampling grades -----------------------------------
export const GRADE_COLORS=[palette.d,palette.loss,MUTED];
function mcgrades(el,ctx,d){
 const g=d.story.grades,p=g.p,lr=g.log_ratio,exact=g.kl,W=1150,H=320;
 let rng,draws,sum,trace;
 const reset=start=>{rng=mulberry32(11);draws=[];sum=0;trace=[];add(start);};
 function add(k){for(let i=0;i<k;i++){const u=rng(),j=u<p[0]?0:u<p[0]+p[1]?1:2;draws.push(j);sum+=lr[j];
  const S=draws.length;if(S<=100||Math.floor(Math.log10(S)*80)!==Math.floor(Math.log10(S-1)*80))trace.push([Math.log10(S),sum/S]);}}
 const render=()=>{
  const S=draws.length,est=S?sum/S:NaN,counts=[0,0,0];draws.forEach(j=>counts[j]++);
  let s=label(0,26,'grades drawn from this year’s p (latest 30)',{fill:palette.p,size:21});
  draws.slice(-30).forEach((j,k)=>{const x=6+(k%10)*52,y=46+Math.floor(k/10)*52;
   s+=`<rect x="${x}" y="${y}" width="44" height="44" rx="7" fill="${GRADE_COLORS[j]}" fill-opacity=".14" stroke="${GRADE_COLORS[j]}" stroke-width="2"/>`+label(x+22,y+31,g.names[j],{fill:GRADE_COLORS[j],size:23,anchor:'middle',weight:600});});
  g.names.forEach((nm,j)=>{const y=232+j*30;
   s+=label(0,y,`${nm}: log ratio ${lr[j]>=0?'+':''}${num(lr[j])}`,{fill:GRADE_COLORS[j],size:20})+label(330,y,`${counts[j].toLocaleString('en')} draws, ${S?Math.round(100*counts[j]/S):0}%`,{fill:GRADE_COLORS[j],size:20});});
  const K=panel({ox:600,W:550,H,L:62,R:26,T:56,B:56,xMin:0,xMax:4.3,yMin:-.6,yMax:1.4,ticksX:[0,1,2,3,4],fmtX:v=>['1','10','100','1,000','10,000'][v],ticksY:[-.5,0,.5,1],xLabel:'draws S (log scale)'});
  s+=K.axes+`<line x1="${K.x(0)}" y1="${K.y(exact).toFixed(1)}" x2="${K.x(4.3)}" y2="${K.y(exact).toFixed(1)}" stroke="${palette.p}" stroke-width="2.5" stroke-dasharray="8 6"/>`+
   label(K.x(4.3),K.y(exact)-10,`exact KL ${num(exact,3)}`,{fill:palette.p,size:20,anchor:'end'})+
   (trace.length>1?K.line(trace.map(t=>t[0]),trace.map(t=>t[1]),palette.q):'')+(S?K.dot(Math.log10(S),est,palette.q):'')+
   label(662,26,S?`average of ${S.toLocaleString('en')} log ratios: ${num(est,3)}`:'No draws yet',{fill:palette.q,size:21});
  return svg(W,H,s,'Grades drawn from this year’s distribution, their counts, and the running average of their log ratios approaching the KL.');
 };
 if(ctx.printing){reset(1000);el.innerHTML=render();return;}
 reset(10);
 const controls=[1,10,100,1000].map(k=>`<button type="button" data-add="${k}">+${k.toLocaleString('en')} grade${k>1?'s':''}</button>`).join('')+`<button type="button" data-reset>Reset</button>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-add]').forEach(b=>b.addEventListener('click',()=>{if(draws.length<20000)add(Number(b.dataset.add));draw();}));
 el.querySelector('[data-reset]').addEventListener('click',()=>{reset(10);draw();});
}

// ---- Part 3: the KL over a grid of (μ, σ); hover a cell to see that q against p ----------------------
const KL_CUTS=[.01,.03,.1,.2,.4,.7,1,1.5,2],KL_RAMP=['#F7F3FF','#EBDFFD','#DBC6FA','#C7A5F4','#AE7FEA','#9258D8','#7537BE','#5A2299','#421473','#2E0C52'];
export const klColor=v=>{const k=KL_CUTS.findIndex(t=>v<t);return KL_RAMP[k<0?KL_CUTS.length:k];};
function klgrid(el,ctx){
 const P0={mu:1,sigma:4},W=1150,H=340,L=66,R=16,T=44,B=56,GW=600,nx=51,ny=40,mu0=-4.1,mu1=6.1,dl=Math.log(25)/ny,ls0=Math.log(4)-28.5*dl,ls1=ls0+ny*dl;
 const X=v=>L+(v-mu0)/(mu1-mu0)*(GW-L-R),Y=ls=>H-B-(ls-ls0)/(ls1-ls0)*(H-T-B),cw=(GW-L-R)/nx,chh=(H-T-B)/ny;
 const cell=(i,j)=>({mu:mu0+(i+.5)*(mu1-mu0)/nx,sigma:Math.exp(ls0+(j+.5)*(ls1-ls0)/ny)});
 let heat='',best={v:Infinity};
 for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){const c=cell(i,j),v=normalKL(c,P0);if(v<best.v)best={v,i,j};
  heat+=`<rect x="${(L+i*cw).toFixed(1)}" y="${(H-B-(j+1)*chh).toFixed(1)}" width="${(cw+.5).toFixed(1)}" height="${(chh+.5).toFixed(1)}" fill="${klColor(v)}"/>`;}
 heat+=`<line x1="${L}" y1="${H-B}" x2="${GW-R}" y2="${H-B}" stroke="${AX}"/><line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="${AX}"/>`;
 for(const v of [-4,-2,0,2,4,6])heat+=label(X(v),H-B+24,num(v,0),{size:20,anchor:'middle'});
 for(const v of [.5,1,2,4,8])heat+=label(L-9,Y(Math.log(v))+7,String(v),{size:20,anchor:'end'});
 heat+=label(GW-R,H-9,'mean μ',{size:20,anchor:'end'})+label(L,T-16,'width σ, log scale',{size:20})+label(GW-R,T-16,'darker = larger KL',{size:20,anchor:'end'});
 const bc=cell(best.i,best.j),bestMark=`<circle cx="${X(bc.mu).toFixed(1)}" cy="${Y(Math.log(bc.sigma)).toFixed(1)}" r="9" fill="none" stroke="${palette.loss}" stroke-width="3"/>`+
  label(X(bc.mu)+14,Y(Math.log(bc.sigma))-12,`best: N(${num(bc.mu,1)}, ${num(bc.sigma,1)}²)`,{fill:palette.loss,size:20,weight:600});
 const xs=lin(-15,17,321),py=xs.map(u=>normalPdf(u,P0));
 const side=(i,j)=>{const c=cell(i,j),v=normalKL(c,P0),qy=xs.map(u=>normalPdf(u,c)),top=Math.max(.42,Math.max(...qy)*1.08);
  const A=panel({ox:640,W:510,H,L:14,R:14,T:76,B:56,xMin:-15,xMax:17,yMax:top,ticksX:[-10,0,10],yAxis:false,xLabel:'θ'});
  return A.axes+A.area(xs,py,TINT.p)+A.line(xs,py,palette.p)+A.line(xs,qy,palette.q)+legend(654,30,[['p = N(1, 4²)',palette.p],[`q = N(${num(c.mu,1)}, ${num(c.sigma,2)}²)`,palette.q]],20)+
   label(654,62,`KL(q‖p) = ${klv(v)} nats`,{fill:palette.ink,size:22,weight:600});};
 const cursor=(i,j)=>`<rect x="${(L+i*cw-1).toFixed(1)}" y="${(H-B-(j+1)*chh-1).toFixed(1)}" width="${(cw+2).toFixed(1)}" height="${(chh+2).toFixed(1)}" fill="none" stroke="${palette.ink}" stroke-width="2.5"/>`;
 const start=[Math.round((4-mu0)/(mu1-mu0)*nx-.5),Math.round((Math.log(.5)-ls0)/(ls1-ls0)*ny-.5)];
 const alt='KL from q = N(μ, σ²) to p = N(1, 4²) over a 51 by 40 grid; the panel on the right shows the selected q against p.';
 const whole=(i,j,b)=>svg(W,H,`<g>${heat}</g><g data-cursor>${cursor(i,j)}</g><g data-best${b>=2?'':' style="display:none"'}>${bestMark}</g><g data-side>${side(i,j)}</g><rect data-hit x="${L}" y="${T}" width="${GW-L-R}" height="${H-T-B}" fill="transparent" style="cursor:crosshair"/>`,alt);
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1">${whole(...start,1)}</div><div data-only="2-99">${whole(best.i,best.j,2)}</div></div>`;return;}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><span class="w-readout">hover: one point, one q</span><button type="button" data-goto="start">q = N(4, 0.5²)</button><button type="button" data-goto="best">the best grid point</button></div>`;
 const box=el.querySelector('.w-fig'),scene=el.closest('.scene');let cur=[...start],build=ctx.reading||!scene?2:Number(scene.dataset.build||1);
 box.innerHTML=whole(...cur,build);
 const svgEl=box.querySelector('svg'),gCur=svgEl.querySelector('[data-cursor]'),gSide=svgEl.querySelector('[data-side]'),gBest=svgEl.querySelector('[data-best]');
 svgEl.setAttribute('tabindex','0');
 const show=(i,j)=>{i=Math.max(0,Math.min(nx-1,i));j=Math.max(0,Math.min(ny-1,j));if(i===cur[0]&&j===cur[1])return;cur=[i,j];gCur.innerHTML=cursor(i,j);gSide.innerHTML=side(i,j);};
 const pick=e=>{const pt=svgEl.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const m=svgEl.getScreenCTM();if(!m)return;const v=pt.matrixTransform(m.inverse());
  if(v.x<L||v.x>GW-R||v.y<T||v.y>H-B)return;show(Math.floor((v.x-L)/cw),Math.floor((H-B-v.y)/chh));};
 svgEl.addEventListener('pointermove',pick);svgEl.addEventListener('pointerdown',pick);
 svgEl.addEventListener('keydown',e=>{const k={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]}[e.key];if(!k)return;e.preventDefault();e.stopPropagation();show(cur[0]+k[0],cur[1]+k[1]);});
 el.querySelectorAll('[data-goto]').forEach(b=>b.addEventListener('click',()=>{const t=b.dataset.goto==='best'?[best.i,best.j]:start;show(...t);}));
 if(scene&&!ctx.reading)new MutationObserver(()=>{build=Number(scene.dataset.build||1);gBest.style.display=build>=2?'':'none';}).observe(scene,{attributes:true,attributeFilter:['data-build']});
}

// ---- Part 3: three covariance families fitted to one correlated target ----------------------------
/** An ellipse at Mahalanobis radius r for a 2×2 covariance, in a panel with scale k pixels per unit. */
export function covEllipse(cx,cy,S,k,r,color,extra=''){
 const [a,b,c]=S,h=Math.sqrt(((a-c)/2)**2+b*b),l1=(a+c)/2+h,l2=Math.max((a+c)/2-h,1e-9),ang=.5*Math.atan2(2*b,a-c)*180/Math.PI;
 return `<ellipse cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" rx="${(r*Math.sqrt(l1)*k).toFixed(1)}" ry="${(r*Math.sqrt(l2)*k).toFixed(1)}" transform="rotate(${(-ang).toFixed(2)} ${cx.toFixed(1)} ${cy.toFixed(1)})" fill="none" stroke="${color}" stroke-width="3" ${extra}/>`;
}
/** A 2×2 matrix in an SVG: "Σ =" and the four numbers between brackets. */
export function matrix2(x,y,name,[a,b,c],color,size=20,digits=2,nameW=1.9){
 const w=size*3.2,h=size*2.6,f=v=>num(v,digits);
 const br=(bx,dir)=>`<path d="M${(bx+dir*7).toFixed(1)},${y-h/2} h${-dir*7} v${h} h${dir*7}" fill="none" stroke="${color}" stroke-width="2"/>`;
 const x0=x+size*nameW;
 return label(x,y+size*.35,name,{fill:color,size,weight:600})+br(x0,1)+br(x0+2*w+12,-1)+
  label(x0+6+w/2,y-size*.3,f(a),{fill:color,size,anchor:'middle'})+label(x0+6+w*1.5,y-size*.3,f(b),{fill:color,size,anchor:'middle'})+
  label(x0+6+w/2,y+size*1,f(b),{fill:color,size,anchor:'middle'})+label(x0+6+w*1.5,y+size*1,f(c),{fill:color,size,anchor:'middle'});
}
const COV_TITLES={iso:['isotropic','s²I · 1 number'],diag:['diagonal','diag(s₁², s₂²) · 2 numbers'],full:['full','LLᵀ · 3 numbers']};
function cov3(el,ctx,d){
 const c=d.story.cov3,runs=c.runs,count=runs.iso.length,W=1150,H=372,PW=370,G=20,k=35,Sp=[c.target.cov[0][0],c.target.cov[0][1],c.target.cov[1][1]];
 const render=i=>{let s='';
  ['iso','diag','full'].forEach((kind,j)=>{const ox=j*(PW+G),cx=ox+PW/2,cy=176,[,m1,m2,a,b,cc,v]=runs[kind][i];
   const clip=`cclip-${kind}-${uid2++}`;
   s+=`<defs><clipPath id="${clip}"><rect x="${ox}" y="58" width="${PW}" height="238"/></clipPath></defs><g clip-path="url(#${clip})">`+
    `<line x1="${ox+8}" y1="${cy}" x2="${ox+PW-8}" y2="${cy}" stroke="${AX}"/><line x1="${cx}" y1="62" x2="${cx}" y2="292" stroke="${AX}"/>`+
    [1,2].map(r=>covEllipse(cx,cy,Sp,k,r,palette.p)).join('')+[1,2].map(r=>covEllipse(cx+m1*k,cy-m2*k,[a,b,cc],k,r,palette.q)).join('')+
    `<circle cx="${(cx+m1*k).toFixed(1)}" cy="${(cy-m2*k).toFixed(1)}" r="5" fill="${palette.q}"/></g>`+
    label(ox+6,26,COV_TITLES[kind][0],{fill:palette.ink,size:23,weight:600})+label(ox+6,52,COV_TITLES[kind][1],{size:19})+
    label(ox+PW-6,26,`KL ${klv(v)}`,{fill:palette.q,size:22,anchor:'end',weight:600})+
    label(ox+PW-10,cy+22,'θ₁',{size:18,anchor:'end'})+label(cx+8,76,'θ₂',{size:18})+
    matrix2(ox+30,336,'Σ<tspan baseline-shift="sub" font-size="70%">q</tspan><tspan dx="6">=</tspan>',[a,b,cc],palette.q,20,2,2.6);
  });
  return svg(W,H,s,'Three Gaussians with isotropic, diagonal and full covariance, each fitted to the same tilted target (teal ellipses), with the current covariance matrix of q.');
 };
 animated(el,ctx,{count,render,step:i=>`step ${runs.iso[i][0]}`,alt:'Three covariance families fitting the same target',duration:5000});
}
let uid2=0;

// ---- Part 3: a mixture of Gaussians, drawn in two steps --------------------------------------------
const COMP=[palette.d,palette.noise];
function mixsample(el,ctx,d){
 const mx=d.story.mixture_sampling,w=mx.weights,mu=mx.means,sd=mx.sds,W=1150,H=330,bins=lin(-2,3,51),bw=bins[1]-bins[0];
 const xs=lin(-2,3,251),comp=k=>xs.map(t=>w[k]*normalPdf(t,{mu:mu[k],sigma:sd[k]})),c0=comp(0),c1=comp(1),dens=xs.map((_,i)=>c0[i]+c1[i]);
 let rng,draws;
 const reset=n=>{rng=mulberry32(31);draws=[];add(n);};
 function add(n){for(let i=0;i<n;i++){const k=rng()<w[0]?0:1;draws.push([k,mu[k]+sd[k]*gauss(rng)]);}}
 const render=()=>{
  const S=draws.length,last=draws.at(-1),counts=[0,0],hist=bins.slice(1).map(()=>[0,0]);
  for(const [kk,t] of draws){counts[kk]++;const b=Math.floor((t+2)/bw);if(b>=0&&b<hist.length)hist[b][kk]++;}
  let s=label(0,26,'step 1: pick a bump',{fill:palette.ink,size:22,weight:600});
  [0,1].forEach(kk=>{const y=44+kk*86,on=last&&last[0]===kk;
   s+=`<rect x="0" y="${y}" width="370" height="70" rx="10" fill="${on?COMP[kk]+'22':palette.paper}" stroke="${COMP[kk]}" stroke-width="${on?3.5:2}"/>`+
    label(14,y+30,`bump ${kk+1}: probability ½`,{fill:COMP[kk],size:21,weight:600})+label(14,y+57,`N(${num(mu[kk],1)}, ${num(sd[kk],1)}²) · picked ${counts[kk].toLocaleString('en')}×`,{fill:COMP[kk],size:19});});
  s+=label(0,236,'step 2: draw θ from that bump',{fill:palette.ink,size:22,weight:600});
  if(last)s+=label(0,272,`latest: bump ${last[0]+1}, θ = ${num(last[1])}`,{fill:COMP[last[0]],size:22});
  s+=label(0,310,`${S.toLocaleString('en')} draws so far`,{size:19});
  const P=panel({ox:420,W:730,H,L:16,R:14,T:62,B:56,xMin:-2,xMax:3,yMax:1.1,ticksX:[-2,-1,0,1,2,3],yAxis:false,xLabel:'θ'});
  s+=P.axes;
  if(S)hist.forEach(([h0,h1],b)=>{const x0=P.x(bins[b])+1,x1=P.x(bins[b+1])-1,d0=h0/(S*bw),d1=h1/(S*bw);
   if(h0)s+=`<rect x="${x0.toFixed(1)}" y="${P.y(d0).toFixed(1)}" width="${(x1-x0).toFixed(1)}" height="${(P.y(0)-P.y(d0)).toFixed(1)}" fill="${COMP[0]}" fill-opacity=".45"/>`;
   if(h1)s+=`<rect x="${x0.toFixed(1)}" y="${P.y(d0+d1).toFixed(1)}" width="${(x1-x0).toFixed(1)}" height="${(P.y(d0)-P.y(d0+d1)).toFixed(1)}" fill="${COMP[1]}" fill-opacity=".45"/>`;});
  s+=P.line(xs,c0,COMP[0],'stroke-dasharray="7 5" stroke-width="2.5"')+P.line(xs,c1,COMP[1],'stroke-dasharray="7 5" stroke-width="2.5"')+P.line(xs,dens,palette.p);
  if(last)s+=`<line x1="${P.x(last[1]).toFixed(1)}" y1="${P.y(0)}" x2="${P.x(last[1]).toFixed(1)}" y2="${P.y(0)-22}" stroke="${COMP[last[0]]}" stroke-width="4"/>`;
  s+=legend(436,30,[['p(θ) = ½ N(−0.2, 0.4²) + ½ N(1, 0.5²)',palette.p]],20);
  return svg(W,H,s,'Draws from a two-bump mixture: first a bump is picked, then θ is drawn from it; the histogram of draws approaches the mixture density.');
 };
 if(ctx.printing){reset(1000);el.innerHTML=render();return;}
 reset(1);
 const controls=[1,10,100,1000].map(n=>`<button type="button" data-add="${n}">+${n.toLocaleString('en')} draw${n>1?'s':''}</button>`).join('')+`<button type="button" data-reset>Reset</button>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-add]').forEach(b=>b.addEventListener('click',()=>{if(draws.length<50000)add(Number(b.dataset.add));draw();}));
 el.querySelector('[data-reset]').addEventListener('click',()=>{reset(1);draw();});
}

// ---- Part 2: information theory -------------------------------------------------------------------
/** Weather icons: sunny, cloudy and rainy, centred at (cx, cy), scaled by k. */
export function weather(kind,cx,cy,k=1){
 const cloud=(x,y)=>`<g fill="#9CA3AF"><circle cx="${x-12*k}" cy="${y+3*k}" r="${11*k}"/><circle cx="${x+2*k}" cy="${y-6*k}" r="${14*k}"/><circle cx="${x+15*k}" cy="${y+4*k}" r="${10*k}"/><rect x="${x-23*k}" y="${y+3*k}" width="${38*k}" height="${11*k}" rx="${5*k}"/></g>`;
 if(kind==='sunny')return `<circle cx="${cx}" cy="${cy}" r="${12*k}" fill="#E8A317"/>`+[0,1,2,3,4,5,6,7].map(j=>{const a=j*Math.PI/4;return `<line x1="${(cx+17*k*Math.cos(a)).toFixed(1)}" y1="${(cy+17*k*Math.sin(a)).toFixed(1)}" x2="${(cx+24*k*Math.cos(a)).toFixed(1)}" y2="${(cy+24*k*Math.sin(a)).toFixed(1)}" stroke="#E8A317" stroke-width="${3*k}" stroke-linecap="round"/>`;}).join('');
 if(kind==='cloudy')return cloud(cx,cy);
 return cloud(cx,cy-6*k)+[-10,2,14].map(dx=>`<line x1="${cx+dx*k}" y1="${cy+14*k}" x2="${cx+(dx-4)*k}" y2="${cy+26*k}" stroke="${palette.data}" stroke-width="${3*k}" stroke-linecap="round"/>`).join('');
}
const log2=x=>Math.log(x)/Math.LN2,bitsOf=p=>-log2(p);
/** Surprise in bits, log₂ 1/p, as a curve with a draggable probability. */
function surprise(el,ctx){
 const W=1150,H=330,s={p:.1},ps=lin(.008,1,400);
 const render=()=>{
  const P=panel({W:700,H,L:62,R:20,T:40,B:56,xMin:0,xMax:1,yMin:0,yMax:7.5,ticksX:[0,.25,.5,.75,1],ticksY:[0,1,2,3,4,5,6,7],xLabel:'probability p(x)'});
  let g=P.axes+P.line(ps,ps.map(bitsOf),palette.loss)+label(62,24,'surprise, bits',{size:20});
  for(const [pr,t] of [[1,'0'],[.5,'1'],[.25,'2'],[.125,'3']])g+=P.dot(pr,bitsOf(pr),'#F3A9B5',5)+label(P.x(pr)+8,P.y(bitsOf(pr))-8,`${t} bit${t==='1'?'':'s'}`,{size:17});
  g+=`<line x1="${P.x(s.p).toFixed(1)}" y1="${P.y(0)}" x2="${P.x(s.p).toFixed(1)}" y2="${P.y(bitsOf(s.p)).toFixed(1)}" stroke="${palette.loss}" stroke-dasharray="5 5"/>`+P.dot(s.p,bitsOf(s.p),palette.loss,9);
  const b=bitsOf(s.p),n=-Math.log(s.p);
  g+=label(760,70,`p(x) = ${num(s.p,s.p<.1?3:2)}`,{fill:palette.ink,size:28,weight:600})+
   label(760,130,`surprise = log₂ 1/p = ${num(b,2)} bits`,{fill:palette.loss,size:24})+
   label(760,172,`= ln 1/p = ${num(n,2)} nats`,{fill:palette.loss,size:24})+
   label(760,232,b<.05?'a certain outcome: no surprise':b<1.2?'a common outcome: little surprise':b<3.5?'a rare outcome: a real surprise':'a very rare outcome: a big surprise',{size:21})+
   label(760,280,'one bit = the surprise of a fair coin flip',{size:19});
  return svg(W,H,g,`The surprise log base 2 of 1 over p, with the selected probability ${num(s.p,2)} giving ${num(b,2)} bits.`);
 };
 const controls=slider('p','p(x)',.01,1,.01,.1)+[['1',1],['½',.5],['¼',.25],['⅛',.125],['0.9',.9],['0.1',.1]].map(([t,v])=>`<button type="button" data-p="${v}">p = ${t}</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw);
 el.querySelectorAll('[data-p]').forEach(bt=>bt.addEventListener('click',()=>{s.p=Number(bt.dataset.p);sync(el,s);draw();}));
}
/** Entropy of two kinds of weather: drag P(sunny) and watch the average surprise. */
function entropy2(el,ctx){
 const W=1150,H=330,s={a:.9},xs=lin(.001,.999,300),Hb=a=>a<=0||a>=1?0:a*bitsOf(a)+(1-a)*bitsOf(1-a);
 const render=()=>{
  const a=s.a,sv=[a>0?bitsOf(a):Infinity,a<1?bitsOf(1-a):Infinity];
  let g='';
  [['sunny',a],['rainy',1-a]].forEach(([kind,pr],k)=>{const cx=90+k*200,base=250,h=pr*150;
   g+=weather(kind,cx,40,1.1)+`<rect x="${cx-34}" y="${(base-h).toFixed(1)}" width="68" height="${Math.max(1.5,h).toFixed(1)}" rx="4" fill="${palette.p}"/>`+
    label(cx,base-h-10,num(pr,3),{fill:palette.p,size:21,anchor:'middle'})+
    label(cx,base+30,Number.isFinite(sv[k])?`surprise ${num(sv[k],2)} bits`:'never happens',{fill:palette.loss,size:18,anchor:'middle'});});
  g+=`<line x1="40" y1="250" x2="360" y2="250" stroke="${AX}"/>`;
  const hv=Hb(a);
  g+=label(420,110,'H = average surprise',{fill:palette.ink,size:23,weight:600})+
   label(420,152,`= ${num(a,3)} × ${Number.isFinite(sv[0])?num(sv[0],2):'0'} + ${num(1-a,3)} × ${Number.isFinite(sv[1])?num(sv[1],2):'0'}`,{size:21})+
   label(420,200,`= ${num(hv,3)} bits`,{fill:palette.p,size:30,weight:600});
  const P=panel({ox:740,W:410,H,L:52,R:16,T:40,B:56,xMin:0,xMax:1,yMin:0,yMax:1.1,ticksX:[0,.5,1],ticksY:[0,.5,1],xLabel:'P(sunny)'});
  g+=P.axes+P.line(xs,xs.map(Hb),palette.p)+P.dot(a,hv,palette.p,8)+label(792,24,'entropy, bits',{size:19});
  return svg(W,H,g,`Two kinds of weather with P(sunny) = ${num(a,3)}: entropy ${num(hv,3)} bits.`);
 };
 const controls=slider('a','P(sunny)',0,1,.005,.9)+[['50/50',.5],['87.5/12.5',.875],['90/10',.9],['100/0',1]].map(([t,v])=>`<button type="button" data-a="${v}">${t}</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw);
 el.querySelectorAll('[data-a]').forEach(bt=>bt.addEventListener('click',()=>{s.a=Number(bt.dataset.a);sync(el,s);draw();}));
}
/**
 * Days of weather, one at a time. mode 'entropy': two kinds of weather and each day's surprise;
 * 'codes': three kinds, sent through the code for p and the code for q; 'logratio': each day's log₂ p/q.
 */
function days(el,ctx,d,mode){
 const W=1150,H=330,info=d.story.info,wb=d.story.bits;
 const kinds=mode==='entropy'?['sunny','rainy']:wb.names,probs=mode==='entropy'?info.forecast:wb.p;
 const value=mode==='entropy'?probs.map(bitsOf):mode==='codes'?[wb.right_code.map(x=>x.length),wb.wrong_code.map(x=>x.length)]:probs.map((v,i)=>log2(v/wb.q[i]));
 const exact=mode==='entropy'?info.entropy.forecast:mode==='codes'?[wb.entropy,wb.cross_entropy]:wb.kl;
 let rng,drawn,sums,trace;
 const reset=n=>{rng=mulberry32(mode==='codes'?7:mode==='entropy'?3:11);drawn=[];sums=[0,0];trace=[];add(n);};
 function add(n){for(let i=0;i<n;i++){const u=rng();let k=0,c=probs[0];while(u>c&&k<probs.length-1){k++;c+=probs[k];}drawn.push(k);
  if(mode==='codes'){sums[0]+=value[0][k];sums[1]+=value[1][k];}else sums[0]+=value[k];
  const S=drawn.length;if(S<=100||Math.floor(Math.log10(S)*80)!==Math.floor(Math.log10(S-1)*80))trace.push([Math.log10(S),sums[0]/S,sums[1]/S]);}}
 const render=()=>{
  const S=drawn.length,last=drawn.slice(-(mode==='codes'?8:30));let g='';
  if(mode==='codes'){
   g+=label(0,24,'the latest days, and the bits each code sends',{size:20});
   last.forEach((k,j)=>{const x=104+j*56;g+=weather(kinds[k],x+24,62,.62)+label(x+24,112,wb.right_code[k],{fill:palette.p,size:20,anchor:'middle',weight:600})+label(x+24,142,wb.wrong_code[k],{fill:palette.q,size:20,anchor:'middle',weight:600});});
   g+=label(0,112,'code for p',{fill:palette.p,size:18})+label(0,142,'code for q',{fill:palette.q,size:18});
   g+=label(0,200,`${S.toLocaleString('en')} days sent`,{size:20})+
    label(0,236,`code for p: ${(sums[0]).toLocaleString('en')} bits, ${num(sums[0]/S,3)} per day`,{fill:palette.p,size:21})+
    label(0,268,`code for q: ${(sums[1]).toLocaleString('en')} bits, ${num(sums[1]/S,3)} per day`,{fill:palette.q,size:21})+
    label(0,304,`wasted: ${num((sums[1]-sums[0])/S,3)} bits per day`,{fill:palette.loss,size:22,weight:600});
  }else{
   g+=label(0,24,mode==='entropy'?'days drawn from the forecast (latest 30), with their surprise':'days drawn from p (latest 30), with log₂ p/q',{size:20});
   last.forEach((k,j)=>{const x=6+(j%10)*56,y=46+Math.floor(j/10)*80;g+=weather(kinds[k],x+24,y+22,.6)+label(x+24,y+66,mode==='entropy'?num(value[k],2):(value[k]>0?'+':'')+num(value[k],0),{fill:palette.loss,size:16,anchor:'middle'});});
   g+=label(0,300,`${S.toLocaleString('en')} days`,{size:20});
  }
  const lo=mode==='entropy'?0:mode==='codes'?1:-1,hi=mode==='entropy'?1.5:mode==='codes'?2.25:1.25;
  const K=panel({ox:600,W:550,H,L:62,R:26,T:56,B:56,xMin:0,xMax:4.3,yMin:lo,yMax:hi,ticksX:[0,1,2,3,4],fmtX:v=>['1','10','100','1,000','10,000'][v],ticksY:mode==='entropy'?[0,.5,1,1.5]:mode==='codes'?[1,1.5,2]:[-1,0,.25,1],xLabel:'days (log scale)'});
  g+=K.axes;
  const hline=(v,color,text)=>`<line x1="${K.x(0)}" y1="${K.y(v).toFixed(1)}" x2="${K.x(4.3)}" y2="${K.y(v).toFixed(1)}" stroke="${color}" stroke-width="2" stroke-dasharray="8 6"/>`+label(K.x(4.3),K.y(v)-8,text,{fill:color,size:18,anchor:'end'});
  if(mode==='codes'){g+=hline(exact[0],palette.p,`H(p) = ${num(exact[0],2)}`)+hline(exact[1],palette.q,`H(p, q) = ${num(exact[1],2)}`);
   if(trace.length>1)g+=K.line(trace.map(t=>t[0]),trace.map(t=>t[1]),palette.p)+K.line(trace.map(t=>t[0]),trace.map(t=>t[2]),palette.q);
   g+=label(662,30,'bits per day so far',{size:20});}
  else{g+=hline(exact,palette.p,mode==='entropy'?`H = ${num(exact,3)} bits`:`KL = ${num(exact,2)} bits`);
   if(trace.length>1)g+=K.line(trace.map(t=>t[0]),trace.map(t=>t[1]),palette.q);if(S)g+=K.dot(Math.log10(S),sums[0]/S,palette.q);
   g+=label(662,30,S?`average so far: ${num(sums[0]/S,3)} bits`:'no days yet',{fill:palette.q,size:21});}
  return svg(W,H,g,mode==='codes'?'Days of weather sent through two codes; the running bits per day approach the entropy and the cross-entropy.':'Days of weather drawn one at a time; the running average approaches the exact expectation.');
 };
 if(ctx.printing){reset(mode==='codes'?1000:1000);el.innerHTML=render();return;}
 reset(mode==='codes'?10:10);
 const controls=[1,10,100,1000].map(k=>`<button type="button" data-add="${k}">+${k.toLocaleString('en')} day${k>1?'s':''}</button>`).join('')+`<button type="button" data-reset>Reset</button>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-add]').forEach(bt=>bt.addEventListener('click',()=>{if(drawn.length<20000)add(Number(bt.dataset.add));draw();}));
 el.querySelector('[data-reset]').addEventListener('click',()=>{reset(10);draw();});
}
/** Maximum likelihood for twenty days: move q and watch the data's average surprise split into H(p̂) + KL. */
function mlefit(el,ctx,d){
 const m=d.story.mle,ph=m.p_hat,names=['sunny','cloudy','rainy'],N=m.data.length,W=1150,H=330,s={a:.25,b:.25};
 const Hhat=ph.reduce((t,v)=>t+(v>0?v*bitsOf(v):0),0);
 const render=()=>{
  const q=[s.a,s.b,Math.max(1e-6,1-s.a-s.b)],avg=ph.reduce((t,v,i)=>t+v*bitsOf(q[i]),0),kl=avg-Hhat;
  let g=label(0,24,'data frequencies p̂ and the model q',{size:20});
  names.forEach((nm,i)=>{const cx=70+i*170,base=250,sc=v=>v*300;
   g+=weather(nm,cx,46,.75);
   for(const [v,color,dx] of [[ph[i],palette.p,-26],[q[i],palette.q,26]]){const h=sc(v);g+=`<rect x="${cx+dx-22}" y="${(base-h).toFixed(1)}" width="44" height="${Math.max(1.5,h).toFixed(1)}" rx="3" fill="${color}"/>`+label(cx+dx,base-h-8,num(v,2),{fill:color,size:18,anchor:'middle'});}});
  g+=`<line x1="10" y1="250" x2="520" y2="250" stroke="${AX}"/>`+legend(10,290,[['p̂: 10, 5, 5 out of 20',palette.p],['q, the model',palette.q]],19);
  const X=v=>620+v*190,y0=150;
  g+=label(600,40,'the data’s average surprise under q',{fill:palette.ink,size:21,weight:600})+
   `<rect x="${X(0)}" y="${y0}" width="${(X(Hhat)-X(0)).toFixed(1)}" height="54" fill="${TINT.p}" stroke="${palette.p}" stroke-width="2"/>`+
   `<rect x="${X(Hhat).toFixed(1)}" y="${y0}" width="${Math.max(0,X(avg)-X(Hhat)).toFixed(1)}" height="54" fill="${TINT.loss}" stroke="${palette.loss}" stroke-width="2"/>`+
   label(X(Hhat/2),y0+34,`H(p̂) = ${num(Hhat,2)}`,{fill:palette.p,size:19,anchor:'middle'})+(X(avg)-X(Hhat)>80?label((X(Hhat)+X(avg))/2,y0+34,`KL ${num(kl,2)}`,{fill:palette.loss,size:19,anchor:'middle'}):'')+
   label(600,100,`= ${num(avg,3)} bits per day`,{fill:palette.ink,size:26,weight:600})+
   label(600,250,`KL(p̂ ‖ q) = ${num(kl,3)} bits`,{fill:palette.loss,size:21})+
   label(600,284,`log₂ p(𝒟 | q) = ${num(-N*avg,1)}: ${kl<5e-4?'the most probable q':`${(2**(N*kl)).toLocaleString('en',{maximumFractionDigits:0})}× less probable than under p̂`}`,{size:19});
  return svg(W,H,g,`The data frequencies and the model q; the data’s average surprise ${num(avg,3)} bits splits into the entropy of the data and KL ${num(kl,3)}.`);
 };
 const controls=slider('a','q(sunny)',.02,.96,.01,.25)+slider('b','q(cloudy)',.02,.96,.01,.25)+
  [['(¼, ¼, ½)',[.25,.25]],['(⅓, ⅓, ⅓)',[1/3,1/3]],['p̂ = (½, ¼, ¼)',[.5,.25]]].map(([t,v])=>`<button type="button" data-q="${v.join(',')}">q = ${t}</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw,()=>{if(s.a+s.b>.98){const k=Object.keys(s).find(x=>x!==document.activeElement?.dataset?.k)||'b';s[k]=Math.max(.02,.98-(k==='a'?s.b:s.a));sync(el,s);}});
 el.querySelectorAll('[data-q]').forEach(bt=>bt.addEventListener('click',()=>{[s.a,s.b]=bt.dataset.q.split(',').map(Number);sync(el,s);draw();}));
}

// ---- Section 2.7: a small student learns a big teacher's distribution --------------------------
const CLASS_RGB=[[36,94,219],[190,18,60],[20,119,55]];
/** A 24 × 24 map of class probabilities, each cell coloured by its probability-weighted class colour. */
function probMap(P,probs,n){
 let s='';const cw=(P.x(P.x1)-P.x(P.x0))/n,ch=(P.y(P.y0)-P.y(P.y1))/n;
 probs.forEach((pr,k)=>{const i=k%n,j=Math.floor(k/n),rgb=[0,1,2].map(c=>Math.round(255*(.35+.65*(pr[0]*CLASS_RGB[0][c]+pr[1]*CLASS_RGB[1][c]+pr[2]*CLASS_RGB[2][c])/255)));
  s+=`<rect x="${(P.x(P.x0)+i*cw).toFixed(1)}" y="${(P.y(P.y0)-(j+1)*ch).toFixed(1)}" width="${(cw+.6).toFixed(1)}" height="${(ch+.6).toFixed(1)}" fill="rgb(${rgb.join(',')})"/>`;});
 return s;
}
function distill(el,ctx,d){
 const ds=d.story.distill,fr=ds.frames,n=ds.grid.n,W=1150,H=330,[x0,x1]=ds.grid.x,[y0,y1]=ds.grid.y;
 const mapPanel=(ox,probs,title,color)=>{const P=panel({ox,W:350,H,L:10,R:10,T:44,B:16,xMin:x0,xMax:x1,yMin:y0,yMax:y1,yAxis:false});P.x0=x0;P.x1=x1;P.y0=y0;P.y1=y1;
  return probMap(P,probs,n)+ds.points.map((pt,k)=>`<circle cx="${P.x(pt[0]).toFixed(1)}" cy="${P.y(pt[1]).toFixed(1)}" r="3" fill="rgb(${CLASS_RGB[ds.labels[k]].join(',')})" stroke="#fff" stroke-width=".8"/>`).join('')+label(ox+10,28,title,{fill:color,size:19,weight:600});};
 const render=i=>{const [step,kl,probs]=fr[i];
  const K=panel({ox:760,W:390,H,L:56,R:34,T:44,B:56,xMin:0,xMax:3000,yMin:0,yMax:Math.max(.6,fr[0][1]*1.1),ticksX:[0,1500,3000],ticksY:[0,.25,.5],xLabel:'training step'});
  const tr=ds.kl_curve.filter(r=>r[0]<=step);
  return svg(W,H,mapPanel(0,ds.teacher_grid,`teacher · ${ds.teacher_params.toLocaleString('en')} parameters`,palette.p)+mapPanel(380,probs,`student · ${ds.student_params} parameters · step ${step}`,palette.q)+
   K.axes+(tr.length>1?K.line(tr.map(r=>r[0]),tr.map(r=>r[1]),palette.loss):'')+K.dot(step,kl,palette.loss)+label(816,28,`KL(teacher ‖ student) = ${num(kl,3)}`,{fill:palette.loss,size:19,weight:600}),
   'The teacher’s class-probability map, the student’s map at this training step, and the KL from teacher to student falling.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'A student network learning a teacher’s distribution',duration:4500});
}
// ---- Section 2.7: RLHF's best policy as the KL leash loosens -----------------------------------
function rlhfbeta(el,ctx,d){
 const r=d.story.rlhf,W=1150,H=300,s={lb:0};
 const best=beta=>{const w=r.ref.map((v,i)=>v*Math.exp(r.reward[i]/beta)),z=w.reduce((a,b2)=>a+b2,0);return w.map(v=>v/z);};
 const render=()=>{
  const beta=Math.pow(10,s.lb),pi=best(beta),er=pi.reduce((a,v,i)=>a+v*r.reward[i],0),kl=pi.reduce((a,v,i)=>a+(v>0?v*Math.log(v/r.ref[i]):0),0);
  let g=label(0,24,'reply',{size:18})+label(560,24,'reward',{size:18,anchor:'end'})+label(590,24,'<tspan font-family="Times New Roman,Georgia,serif" font-style="italic">π</tspan><tspan baseline-shift="sub" font-size="70%">ref</tspan> (teal) and the best policy <tspan font-family="Times New Roman,Georgia,serif" font-style="italic">π</tspan>* (purple)',{size:18});
  r.replies.forEach((rep,i)=>{const y=46+i*46,bw=480;
   g+=label(0,y+18,rep,{fill:palette.ink,size:19})+label(560,y+18,(r.reward[i]>0?'+':'')+num(r.reward[i],1),{fill:r.reward[i]>=0?palette.d:palette.loss,size:19,anchor:'end',weight:600})+
    `<rect x="590" y="${y}" width="${Math.max(1.5,r.ref[i]*bw).toFixed(1)}" height="14" rx="3" fill="${palette.p}"/>`+label(596+r.ref[i]*bw,y+12,num(r.ref[i],2),{fill:palette.p,size:15})+
    `<rect x="590" y="${y+17}" width="${Math.max(1.5,pi[i]*bw).toFixed(1)}" height="14" rx="3" fill="${palette.q}"/>`+label(596+pi[i]*bw,y+29,num(pi[i],2),{fill:palette.q,size:15});});
  g+=label(0,292,`β = ${num(beta,beta<1?2:1)}:  expected reward ${num(er,2)}  ·  KL to the reference = ${num(kl,2)} nats`,{fill:palette.ink,size:21,weight:600});
  return svg(W,H,g,`With beta ${num(beta,2)}, the best policy over five replies, against the reference, with expected reward ${num(er,2)}.`);
 };
 const controls=`<label>log₁₀ β<input type="range" min="-1.3" max="1.3" step=".02" value="0" data-k="lb" aria-label="log10 of beta"><span class="w-readout" data-o="lb">0.00</span></label>`+
  [['β = 0.05: no leash',-1.3],['β = 1',0],['β = 20: tight leash',1.3]].map(([t,v])=>`<button type="button" data-lb="${v}">${t}</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 bindSliders(el,s,draw);
 el.querySelectorAll('[data-lb]').forEach(bt=>bt.addEventListener('click',()=>{s.lb=Number(bt.dataset.lb);sync(el,s);draw();}));
}

// ---- Part 5: variational inference for the tiny network -----------------------------------------
function bnnvi(el,ctx,d){
 const b=d.story.bnn,fr=b.vi.frames,xs=b.grid,ev=b.evidence.estimate,W=1150,H=330;
 const render=i=>{const [step,elbo,mean,sd]=fr[i],tr=fr.slice(0,i+1);
  const A=panel({W:600,H,L:52,R:14,T:62,B:56,xMin:-2.5,xMax:2.5,yMin:-2.6,yMax:2.6,ticksX:[-2,-1,0,1,2],ticksY:[-2,0,2],xLabel:'x'});
  const K=panel({ox:640,W:510,H,L:62,R:30,T:62,B:56,xMin:0,xMax:4000,yMin:-40,yMax:4,ticksX:[0,2000,4000],ticksY:[-40,-20,0],xLabel:'step'});
  const pts=b.x.map((x,k)=>`<circle cx="${A.x(x).toFixed(1)}" cy="${A.y(b.y[k]).toFixed(1)}" r="5.5" fill="${palette.data}"/>`).join('');
  return svg(W,H,A.axes+A.band(xs,mean.map((m,k)=>m-2*sd[k]),mean.map((m,k)=>m+2*sd[k]),TINT.q)+A.line(xs,mean,palette.q)+pts+
   legend(52,30,[['q’s predictive: mean ± 2 sd',palette.q]])+
   K.axes+`<line x1="${K.x(0)}" y1="${K.y(ev).toFixed(1)}" x2="${K.x(4000)}" y2="${K.y(ev).toFixed(1)}" stroke="${palette.data}" stroke-width="2.5" stroke-dasharray="8 6"/>`+
   label(K.x(4000),K.y(ev)-10,`log p(𝒟) ≈ ${num(ev)}`,{fill:palette.data,size:20,anchor:'end'})+
   K.line(tr.map(r=>r[0]),tr.map(r=>r[1]),palette.q)+(elbo>=-40?K.dot(step,elbo,palette.q):'')+
   label(702,30,`ELBO = ${elbo<-1000?Math.round(elbo).toLocaleString('en').replace('-','−'):num(elbo,2)}${elbo<-40?' (below the axis)':''}`,{fill:palette.q,size:22}),
   'The predictive band of q over training, and the ELBO rising towards the evidence.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'VI on the tiny network',duration:6500});
}

export function mount(stage,ctx,d){
 const make={klarea,fwdrev,olive,elboslider,mclive,bnnvi,priorfuncs,mapfit,mcgrades,klgrid,cov3,mixsample,surprise,entropy2,mlefit,distill,rlhfbeta,'days-entropy':(e,c)=>days(e,c,d,'entropy'),'days-codes':(e,c)=>days(e,c,d,'codes'),'days-logratio':(e,c)=>days(e,c,d,'logratio'),'mix-sample':(el,c)=>mixfit(el,c,d,'sample'),'mix-rsample':(el,c)=>mixfit(el,c,d,'rsample')};
 for(const el of stage.querySelectorAll('[data-widget]')){
  const f=make[el.dataset.widget];if(!f)throw new Error('Unknown widget '+el.dataset.widget);f(el,ctx,d);
 }
}
