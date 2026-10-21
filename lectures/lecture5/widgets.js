// Live figures for Lecture 5, and the small SVG chart kit the static slides share with them.
// Exact pixel quantities come from math.mjs; recorded runs from data/*.json and figures/*.png;
// the 2-D smiley network runs here, from data/toy-weights.json (scripts/check-science.mjs checks it
// against PyTorch). Print and PDF show fixed states; presentation mode ties animations to builds.
import {palette,esc} from '../shared/author.js';
import {LIN,COS,T,TOY,TOY_EPS,TOY_Z,npdf,pInk,x0Hat,epsStar,density,reverseKernel,forwardStep,reverseStep,randn} from './math.mjs';

export const MUTED='#68717B',AX='#AAB3BC';
export const TINT={p:'#0F766E24',q:'#8B2CDE24',loss:'#BE123C2E',data:'#245EDB24',noise:'#AA4E0824',grey:'#E5E7EB'};
export const lin=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
/** Fixed decimals, a true minus sign, and no "−0.00". */
export const num=(v,n=2)=>{const s=Number(v).toFixed(n);return (/^-0(\.0*)?$/.test(s)?s.slice(1):s).replace('-','−');};
let uid=0;

/** One plotting panel, optionally offset inside a larger SVG. Data outside the axes is clipped. */
export function panel(o){
 const ox=o.ox??0,oy=o.oy??0,W=o.W??600,H=o.H??320,L=o.L??64,R=o.R??14,Tp=o.T??56,B=o.B??56,fs=o.fs??20;
 const {xMin,xMax}=o,yMin=o.yMin??0,yMax=o.yMax;
 const x=v=>ox+L+(v-xMin)/(xMax-xMin)*(W-L-R),y=v=>oy+H-B-(v-yMin)/(yMax-yMin)*(H-Tp-B);
 const id=`wclip-${++uid}`,tick=v=>String(+Number(v).toFixed(3)).replace('-','−');
 const base=yMin<0&&yMax>0&&o.zeroAxis?y(0):oy+H-B;
 let axes=`<defs><clipPath id="${id}"><rect x="${ox+L}" y="${oy+Tp-10}" width="${W-L-R}" height="${H-B-Tp+10}"/></clipPath></defs>`;
 axes+=`<line x1="${ox+L}" y1="${base}" x2="${ox+W-R}" y2="${base}" stroke="${AX}"/>`;
 if(o.yAxis!==false)axes+=`<line x1="${ox+L}" y1="${oy+Tp}" x2="${ox+L}" y2="${oy+H-B}" stroke="${AX}"/>`;
 for(const v of o.ticksX||[])axes+=`<text x="${x(v).toFixed(1)}" y="${(oy+H-B+24).toFixed(1)}" text-anchor="middle" fill="${MUTED}" font-size="${fs}">${(o.fmtX||tick)(v)}</text>`;
 for(const v of o.ticksY||[])axes+=`<text x="${ox+L-9}" y="${(y(v)+7).toFixed(1)}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${(o.fmtY||tick)(v)}</text>`;
 if(o.xLabel)axes+=`<text x="${ox+W-R}" y="${oy+H-9}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${esc(o.xLabel)}</text>`;
 if(o.yLabel)axes+=`<text x="${ox+L}" y="${oy+Tp-16}" fill="${MUTED}" font-size="${fs}">${esc(o.yLabel)}</text>`;
 const pts=(xs,ys)=>{let s='';for(let i=0;i<xs.length;i++){const v=y(ys[i]);s+=`${x(xs[i]).toFixed(1)},${(Number.isFinite(v)?Math.max(oy-600,Math.min(oy+H+600,v)):oy+H-B).toFixed(1)} `;}return s;};
 return {x,y,ox,oy,W,H,L,R,T:Tp,B,axes,clip:id,
  line:(xs,ys,color,extra='')=>`<polyline clip-path="url(#${id})" points="${pts(xs,ys)}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" ${extra}/>`,
  area:(xs,ys,fill,at=Math.max(yMin,0))=>`<polygon clip-path="url(#${id})" points="${x(xs[0]).toFixed(1)},${y(at).toFixed(1)} ${pts(xs,ys)}${x(xs.at(-1)).toFixed(1)},${y(at).toFixed(1)}" fill="${fill}"/>`,
  dot:(vx,vy,color,r=6)=>`<circle cx="${x(vx).toFixed(1)}" cy="${y(vy).toFixed(1)}" r="${r}" fill="${color}" stroke="${palette.paper}" stroke-width="2"/>`,
  vline:(vx,color,extra='')=>`<line x1="${x(vx).toFixed(1)}" y1="${oy+Tp-6}" x2="${x(vx).toFixed(1)}" y2="${oy+H-B}" stroke="${color}" stroke-width="2.5" ${extra}/>`,
 };
}
/** Text in an SVG; content must already be safe (no markup from data). */
export const label=(tx,ty,s,{fill=MUTED,size=22,anchor='start',weight=''}={})=>`<text x="${(+tx).toFixed(1)}" y="${(+ty).toFixed(1)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}"${weight?` font-weight="${weight}"`:''}>${s}</text>`;
export const svg=(W,H,inner,alt)=>`<svg class="plot wsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg>`;
export const fig=(W,H,inner,alt)=>`<div class="diagram-scroll">${svg(W,H,inner,alt)}</div>`;
export const widget=(name,alt)=>`<div class="widget" data-widget="${name}" role="group" aria-label="${esc(alt)}"></div>`;
export const legend=(x0,y0,items,size=21)=>{let x=x0,s='';for(const [text,color,dashed] of items){s+=`<line x1="${x}" y1="${y0-7}" x2="${x+26}" y2="${y0-7}" stroke="${color}" stroke-width="3"${dashed?' stroke-dasharray="7 4"':''}/>`+label(x+34,y0,text,{fill:color,size});x+=56+text.replace(/<[^>]+>/g,'').length*size*.52;}return s;};
export const arrowHead=(id,color)=>`<marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="${color}"/></marker>`;

// ---------------------------------------------------------------- images: digits drawn from values, frames cut from strips
const grey=v=>{const g=Math.round(250-Math.max(0,Math.min(1,v))*230);return `rgb(${g},${g},${Math.min(255,g+3)})`;};
/** A 28×28 image as SVG rects (ink dark on light paper); mark = [row, col] outlines one pixel. */
export function digitSVG(vals,{size=260,mark=null,lo=-1,hi=1,x=0,y=0,label:lab=''}={}){
 let s='';
 for(let r=0;r<28;r++)for(let c=0;c<28;c++)s+=`<rect x="${c}" y="${r}" width="1.04" height="1.04" fill="${grey((vals[r*28+c]-lo)/(hi-lo))}"/>`;
 if(mark)s+=`<rect x="${mark[1]-.2}" y="${mark[0]-.2}" width="1.4" height="1.4" fill="none" stroke="${palette.noise}" stroke-width=".42"/>`;
 return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="0 0 28 28" style="overflow:hidden" shape-rendering="crispEdges" role="img" aria-label="${esc(lab||'A 28 by 28 image')}">${s}<rect x="0" y="0" width="28" height="28" fill="none" stroke="#D7DCE1" stroke-width=".12"/></svg>`;
}
/** One frame of a horizontal strip of n 28×28 frames, as an inline block. */
export const frame=(src,k,n,size=96,extra='')=>`<span class="mn" style="width:${size}px;height:${size}px;background-image:url(figures/${src});background-size:${n*100}% 100%;background-position:${n>1?(k/(n-1)*100).toFixed(4):0}% 0;${extra}"></span>`;
/** The same frame inside an SVG, cropped with a nested viewBox (safe in PDFs). */
export const svgFrame=(src,k,n,x,y,size,extra='')=>`<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="${k*28} 0 28 28" preserveAspectRatio="none" style="overflow:hidden" ${extra}><image href="figures/${src}" x="0" y="0" width="${n*28}" height="28" preserveAspectRatio="none" style="image-rendering:pixelated"/></svg><rect x="${x}" y="${y}" width="${size}" height="${size}" fill="none" stroke="#D7DCE1"/>`;
/** A grid PNG (28-pixel cells, 2-pixel gaps), shown with sharp pixels. */
export const pic=(src,width,alt,extra='')=>`<img class="pix" src="figures/${src}" alt="${esc(alt)}" style="width:${width}px;${extra}">`;

// ---------------------------------------------------------------- the 2-D smiley network, in the browser
/** ToyDenoiser's forward pass: [x, y, 16 time features (, one-hot part)] → 3 × SiLU → 2. */
export function toyNet(spec){
 const L=spec.layers.map(l=>({W:Float64Array.from(l.W.flat()),b:Float64Array.from(l.b),o:l.b.length,i:l.W[0].length}));
 const half=spec.tdim/2,freq=Array.from({length:half},(_,k)=>Math.exp(-Math.log(10000)*k/(half-1)));
 const a=new Float64Array(256),bb=new Float64Array(256);
 return (x,y,t,c=null)=>{
  let n=0;a[n++]=x;a[n++]=y;
  for(let k=0;k<half;k++)a[n++]=Math.sin(t*freq[k]);
  for(let k=0;k<half;k++)a[n++]=Math.cos(t*freq[k]);
  if(spec.n_parts)for(let k=0;k<=spec.n_parts;k++)a[n++]=k===c?1:0;
  let inp=a,out=bb;
  L.forEach((l,j)=>{
   for(let o=0;o<l.o;o++){let s=l.b[o];const row=o*l.i;for(let i=0;i<l.i;i++)s+=l.W[row+i]*inp[i];out[o]=j<L.length-1?s/(1+Math.exp(-s)):s;}
   [inp,out]=[out,inp];
  });
  return [inp[0],inp[1]];
 };
}

const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
/**
 * A recorded sequence, replayed frame by frame. Presentation: frame 0 until the play build, which plays it
 * (arriving there directly shows the final frame). Reading: final frame, with controls. Print: frame 0 before
 * the play build and the final frame from it, as static states.
 */
function animated(el,ctx,{count,render,step,alt,playBuild=2,duration=4200,extra='',start=0}){
 const last=count-1;
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1-${Math.max(1,playBuild-1)}">${render(playBuild>1?start:last)}</div><div data-only="${playBuild}-99">${render(last)}</div></div>`;return {rerender(){}};}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><button type="button" class="w-play">▶ Play</button><input class="w-scrub" type="range" min="0" max="${last}" step="1" value="0" aria-label="${esc(alt)}: frame"><span class="w-readout w-step"></span>${extra}</div>`;
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
  idle(start);let fresh=false,prev=0;
  new MutationObserver(records=>{for(const r of records){
   if(r.attributeName==='class'){if(scene.classList.contains('current'))fresh=true;else idle();continue;}
   const b=Number(scene.dataset.build||1);
   if(b<playBuild)idle(start);else if(fresh)idle(last);else if(prev===playBuild-1&&b===playBuild)play();
   fresh=false;prev=b;
  }}).observe(scene,{attributes:true,attributeFilter:['class','data-build']});
 }
 return {rerender:()=>show(i)};
}
/** A figure redrawn from sliders and buttons. Print shows printState (or the starting state) without controls. */
function interactive(el,ctx,{controls,render,print}){
 if(ctx.printing){el.innerHTML=print?print():render();return null;}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls">${controls}</div>`;
 const box=el.querySelector('.w-fig'),draw=()=>{box.innerHTML=render();};
 for(const input of el.querySelectorAll('input[type=range]'))input.addEventListener('change',()=>input.blur());
 draw();return draw;
}
const slider=(key,text,min,max,stepSize,value,fmt='')=>`<label>${text}<input type="range" min="${min}" max="${max}" step="${stepSize}" value="${value}" data-k="${key}" aria-label="${esc(text)}"><span class="w-readout" data-o="${key}"${fmt?` data-f="${fmt}"`:''}>${value}</span></label>`;
function bindSliders(el,state,draw,after=()=>{}){
 if(!draw)return;
 for(const input of el.querySelectorAll('[data-k]'))input.addEventListener('input',()=>{state[input.dataset.k]=Number(input.value);sync(el,state);after();draw();});
 sync(el,state);
}
const fmts={int:v=>String(Math.round(v)),d2:v=>num(v,2),d1:v=>num(v,1),steps:v=>String(STEPS[v]),w:v=>String(WS[v])};
function sync(el,state){
 for(const input of el.querySelectorAll('[data-k]'))input.value=String(state[input.dataset.k]);
 for(const o of el.querySelectorAll('[data-o]'))o.textContent=(fmts[o.dataset.f]||fmts.d2)(state[o.dataset.o]);
}
/** Re-render when the slide's build changes (for figures whose content depends on the build). */
function onBuild(el,ctx,cb){
 const scene=el.closest('.scene');if(!scene||ctx.reading||ctx.printing)return;
 new MutationObserver(()=>cb(Number(scene.dataset.build||1))).observe(scene,{attributes:true,attributeFilter:['data-build']});
}
const buildOf=(el,ctx)=>ctx.reading?99:Number(el.closest('.scene')?.dataset.build||1);

// ================================================================ Part 0: the opening film
function hookfilm(el,ctx,d){
 const ts=d.mn.trajectory_t,n=ts.length,seed=0,W=1150,H=360;
 const render=i=>{
  const big=svgFrame(`mnist-traj-${seed}-x.png`,i,n,0,10,330);
  let strip='';for(let k=0;k<n;k+=3)strip+=svgFrame(`mnist-traj-${seed}-x.png`,k,n,380+(k/3)*96,40,88,k<=i?'':'opacity=".25"')+label(380+(k/3)*96+44,158,`t = ${ts[k]}`,{size:17,anchor:'middle',fill:k<=i?palette.ink:MUTED});
  return svg(W,H,big+strip+label(380,240,`step ${T-ts[i]} of 1,000`,{fill:palette.q,size:30,weight:650})+
   label(380,285,ts[i]>0?'each step: predict the noise, remove a little of it':'a digit that is not in the training set',{size:23})+
   label(380,320,'our network: 814,145 parameters, trained on MNIST',{size:19}),'Our trained diffusion model turning static into a digit, step by step.');
 };
 animated(el,ctx,{count:n,render,step:i=>`t = ${ts[i]}`,alt:'From noise to a digit',duration:5200});
}

// ================================================================ Part 1: destroying is easy
/** 1-D noising of our 3's 784 pixel values: add noise only, or shrink then add. */
function explode(el,ctx,d){
 const x0=d.mn.held.x0,S=LIN,W=1150,H=340,state={t:1000};
 const run=(t,shrink)=>{const r=randn(shrink?11:12),x=x0.slice();for(let s=1;s<=t;s++){const b=S.beta[s],a=shrink?Math.sqrt(1-b):1;for(let i=0;i<x.length;i++)x[i]=a*x[i]+Math.sqrt(b)*r();}return x;};
 const hist=(P,xs,color,tag)=>{const lo=-8,hi=8,nb=64,h=new Array(nb).fill(0);for(const v of xs){const k=Math.floor((v-lo)/(hi-lo)*nb);if(k>=0&&k<nb)h[k]++;}
  const mx=Math.max(...h);let s='';h.forEach((c,k)=>{if(!c)return;const x1=P.x(lo+(hi-lo)*k/nb),x2=P.x(lo+(hi-lo)*(k+1)/nb);s+=`<rect x="${x1.toFixed(1)}" y="${P.y(c/mx).toFixed(1)}" width="${(x2-x1-1).toFixed(1)}" height="${(P.y(0)-P.y(c/mx)).toFixed(1)}" fill="${color}" fill-opacity=".75"/>`;});
  const m=xs.reduce((a,b)=>a+b,0)/xs.length,sd=Math.sqrt(xs.reduce((a,b)=>a+(b-m)**2,0)/xs.length);
  return s+label(P.ox+P.L,P.oy+28,tag,{fill:color,size:22,weight:650})+label(P.ox+P.W-P.R,P.oy+28,`spread (sd) = ${num(sd,2)}`,{fill:color,size:22,anchor:'end'});};
 const render=(b=buildOf(el,ctx))=>{
  const A=panel({W:560,H,L:16,R:14,T:50,B:52,xMin:-8,xMax:8,yMax:1.08,ticksX:[-8,-4,0,4,8],yAxis:false,xLabel:'pixel value'});
  const Bp=panel({ox:590,W:560,H,L:16,R:14,T:50,B:52,xMin:-8,xMax:8,yMax:1.08,ticksX:[-8,-4,0,4,8],yAxis:false,xLabel:'pixel value'});
  let s=A.axes+hist(A,run(state.t,false),palette.loss,'add noise only');
  if(b>=2)s+=Bp.axes+hist(Bp,run(state.t,true),palette.p,'shrink a little, then add');
  return svg(W,H,s,'Histograms of the 784 pixel values of our 3 after t noising steps, without and with the shrink.');
 };
 const controls=slider('t','steps t',0,1000,10,state.t,'int');
 const draw=interactive(el,ctx,{controls,render:()=>render(),print:()=>`<div class="state-slot"><div data-only="1">${render(1)}</div><div data-only="2-99">${render(2)}</div></div>`});
 if(!draw)return;bindSliders(el,state,draw);onBuild(el,ctx,()=>draw());
}

/** Our 3 at noise level t, computed live: x_t = sqrt(ab) x0 + sqrt(1 − ab) eps, with one fixed eps. */
function film(el,ctx,d){
 const {x0,eps}=d.mn.held,P=d.px.pixel,S=LIN,state={t:0},W=1150,H=330;
 const xt=t=>{const a=S.ab[t];return x0.map((v,i)=>Math.sqrt(a)*v+Math.sqrt(1-a)*eps[i]);};
 const bars=(t,ox)=>{const a=S.ab[t],sg=Math.sqrt(a),nz=Math.sqrt(1-a),w=420;
  return label(ox,40,`t = ${t}`,{fill:palette.ink,size:30,weight:650})+
  label(ox,100,`signal √ᾱₜ = ${num(sg,3)}`,{fill:palette.data,size:23})+`<rect x="${ox}" y="112" width="${w}" height="22" fill="${TINT.data}"/><rect x="${ox}" y="112" width="${(w*sg).toFixed(1)}" height="22" fill="${palette.data}"/>`+
  label(ox,178,`noise √(1 − ᾱₜ) = ${num(nz,3)}`,{fill:palette.noise,size:23})+`<rect x="${ox}" y="190" width="${w}" height="22" fill="${TINT.noise}"/><rect x="${ox}" y="190" width="${(w*nz).toFixed(1)}" height="22" fill="${palette.noise}"/>`+
  label(ox,262,`our pixel: ${num(Math.sqrt(a)*x0[P[0]*28+P[1]],2)} + ${num(Math.sqrt(1-a)*eps[P[0]*28+P[1]],2)} = ${num(xt(t)[P[0]*28+P[1]],2)}`,{fill:palette.noise,size:22})+
  label(ox,296,'signal part + noise part (ink is +1, paper −1)',{size:18});};
 const render=()=>svg(W,H,digitSVG(xt(state.t),{size:320,mark:P,x:60,y:5})+bars(state.t,470),`Our 3 at noise level t = ${state.t}`);
 const print=()=>{let s='';[0,100,250,500,750,1000].forEach((t,k)=>{s+=digitSVG(xt(t),{size:170,mark:P,x:12+k*190,y:20})+label(97+k*190,225,`t = ${t}`,{size:22,anchor:'middle',fill:palette.ink})+label(97+k*190,256,`√ᾱ = ${num(Math.sqrt(S.ab[t]),2)}`,{size:19,anchor:'middle',fill:palette.data});});return svg(W,280,s,'Our 3 at six noise levels.');};
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',0,1000,1,0,'int')+`<button type="button" data-go="250">t = 250</button><button type="button" data-go="500">t = 500</button><button type="button" data-go="1000">t = 1000</button>`,render,print});
 if(!draw)return;bindSliders(el,state,draw);
 el.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>{state.t=Number(b.dataset.go);sync(el,state);draw();}));
}

/** All 784 pixels of our 3 at level t, the exact density over all MNIST pixels, and N(0, 1). */
function pixhist(el,ctx,d){
 const {x0,eps}=d.mn.held,lev=d.ex.pixel_levels,S=LIN,state={t:0},W=1150,H=340;
 const dens=(x,t)=>{const a=S.ab[t],v=Math.max(1-a,1e-4);let s=0;for(let k=0;k<256;k++)if(lev[k]>1e-6)s+=lev[k]*npdf(x,Math.sqrt(a)*(k/127.5-1),v);return s;};
 const xs=lin(-4,4,321);
 const render=()=>{
  const t=state.t,a=S.ab[t],vals=x0.map((v,i)=>Math.sqrt(a)*v+Math.sqrt(1-a)*eps[i]);
  const P=panel({W:760,H,L:16,R:14,T:56,B:52,xMin:-4,xMax:4,yMax:1,ticksX:[-4,-2,-1,0,1,2,4],yAxis:false,xLabel:'pixel value'});
  const nb=80,h=new Array(nb).fill(0);for(const v of vals){const k=Math.floor((v+4)/8*nb);if(k>=0&&k<nb)h[k]++;}
  const top=Math.max(...h)/784/(8/nb);let s=P.axes;const scale=.95/Math.max(top,.5);
  h.forEach((c,k)=>{if(!c)return;const x1=P.x(-4+8*k/nb),x2=P.x(-4+8*(k+1)/nb),hh=c/784/(8/nb)*scale;s+=`<rect x="${x1.toFixed(1)}" y="${P.y(hh).toFixed(1)}" width="${(x2-x1-1).toFixed(1)}" height="${(P.y(0)-P.y(hh)).toFixed(1)}" fill="${palette.data}" fill-opacity=".55"/>`;});
  if(t>=60)s+=P.line(xs,xs.map(x=>Math.min(dens(x,t)*scale,1.2)),palette.p);
  s+=P.line(xs,xs.map(x=>npdf(x,0,1)*scale),palette.ink,'stroke-dasharray="8 6" style="stroke-width:2"');
  s+=legend(16,30,[['our 3: its 784 pixels',palette.data],['all MNIST pixels (exact)',palette.p],['N(0, 1)',palette.ink,true]]);
  s+=label(800,90,`t = ${t}`,{fill:palette.ink,size:30,weight:650})+label(800,140,`signal √ᾱ = ${num(Math.sqrt(a),3)}`,{fill:palette.data,size:22})+
   label(800,175,`noise √(1 − ᾱ) = ${num(Math.sqrt(1-a),3)}`,{fill:palette.noise,size:22})+
   label(800,235,t<200?'paper at −1, ink at +1':t<700?'the two bumps blur and merge':'only the noise term is left',{size:21,fill:palette.ink});
  return svg(W,H,s,'The pixel values of our 3 at noise level t, against N(0, 1).');
 };
 const print=()=>{state.t=1000;return render();};
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',0,1000,5,0,'int'),render,print});
 if(!draw)return;bindSliders(el,state,draw);
}

/** The pixel's density over time (exact), with forward paths simulated step by step. */
function river(el,ctx,d){
 const pi=d.px.pi,S=LIN,W=1150,H=380,L=70,R=34,Tp=40,B=60,ymin=-3.2,ymax=3.2;
 const X=t=>L+t/T*(W-L-R),Y=v=>Tp+(ymax-v)/(ymax-ymin)*(H-Tp-B);
 const cols=125,rows=80;let cells='';
 for(let i=0;i<cols;i++){const t=Math.max(1,Math.round((i+.5)/cols*T));let mx=0;const col=[];
  for(let j=0;j<rows;j++){const v=ymax-(j+.5)/rows*(ymax-ymin),p=density(v,t,pi,S);col.push(p);mx=Math.max(mx,p);}
  col.forEach((p,j)=>{const o=Math.pow(p/mx,.6);if(o>.02)cells+=`<rect x="${(X(i/cols*T)).toFixed(1)}" y="${(Tp+j/rows*(H-Tp-B)).toFixed(1)}" width="${((W-L-R)/cols+.6).toFixed(1)}" height="${((H-Tp-B)/rows+.6).toFixed(1)}" fill="${palette.p}" fill-opacity="${(o*.55).toFixed(3)}"/>`;});}
 const r=randn(5),paths=[];
 for(let k=0;k<14;k++){let x=k<5?1:-1;const p=[[0,x]];for(let t=1;t<=T;t++){x=Math.sqrt(1-S.beta[t])*x+Math.sqrt(S.beta[t])*r();if(t%5===0)p.push([t,x]);}paths.push(p);}
 const axes=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="${AX}"/><line x1="${L}" y1="${Tp}" x2="${L}" y2="${H-B}" stroke="${AX}"/>`+
  [0,250,500,750,1000].map(t=>label(X(t),H-B+26,String(t),{size:20,anchor:'middle'})).join('')+[-2,-1,0,1,2].map(v=>label(L-10,Y(v)+7,num(v,0),{size:20,anchor:'end'})).join('')+
  label(W-R,H-12,'step t',{size:20,anchor:'end'})+label(L+8,Tp-12,'pixel value',{size:20});
 const pathSVG=paths.map((p,k)=>`<polyline points="${p.map(([t,v])=>`${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${k<5?palette.data:palette.noise}" stroke-width="1.6" stroke-opacity=".85"/>`).join('');
 const render=b=>svg(W,H,cells+axes+(b>=2?pathSVG:'')+label(X(15),Y(1)-14,'ink',{fill:palette.data,size:21,weight:600})+label(X(15),Y(-1)+30,'paper',{fill:palette.noise,size:21,weight:600})+
  label(X(1000)-8,Tp+24,'one bell: N(0, 1)',{fill:palette.p,size:21,anchor:'end',weight:600}),'The density of our pixel over the thousand steps: two thin lines at ink and paper widen and merge into one bell.'+(b>=2?' Fourteen noising paths are drawn.':''));
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1" class="diagram-scroll">${render(1)}</div><div data-only="2-99" class="diagram-scroll">${render(2)}</div></div>`;return;}
 const draw=b=>{el.innerHTML=`<div class="diagram-scroll">${render(b)}</div>`;};draw(buildOf(el,ctx));onBuild(el,ctx,draw);
}

/** Step by step versus one jump: histograms of 4,000 runs each, and the exact density. */
function shortcut(el,ctx,d){
 const S=LIN,state={t:250},W=1150,H=330,N=4000;
 const render=()=>{
  const t=state.t,a=S.ab[t],r1=randn(21),r2=randn(22),step=[],jump=[];
  for(let n=0;n<N;n++){let x=1;for(let s=1;s<=t;s++)x=Math.sqrt(1-S.beta[s])*x+Math.sqrt(S.beta[s])*r1();step.push(x);jump.push(Math.sqrt(a)*1+Math.sqrt(1-a)*r2());}
  const P=panel({W:760,H,L:16,R:14,T:56,B:52,xMin:-3.5,xMax:3.5,yMax:1,ticksX:[-3,-2,-1,0,1,2,3],yAxis:false,xLabel:'x_t of the pixel'});
  const sd=Math.sqrt(Math.max(1-a,1e-6)),peak=npdf(Math.sqrt(a),Math.sqrt(a),sd*sd),sc=.92/Math.min(peak,6);
  const hist=(xs,color,off)=>{const nb=56,h=new Array(nb).fill(0);for(const v of xs){const k=Math.floor((v+3.5)/7*nb);if(k>=0&&k<nb)h[k]++;}let s='';h.forEach((c,k)=>{if(!c)return;const x1=P.x(-3.5+7*k/nb),x2=P.x(-3.5+7*(k+1)/nb),hh=Math.min(c/N/(7/nb)*sc,1.05);s+=`<rect x="${(x1+off).toFixed(1)}" y="${P.y(hh).toFixed(1)}" width="${((x2-x1)/2-1).toFixed(1)}" height="${(P.y(0)-P.y(hh)).toFixed(1)}" fill="${color}" fill-opacity=".7"/>`;});return s;};
  const half=(P.x(-3.5+7/56)-P.x(-3.5))/2,xs=lin(-3.5,3.5,281);
  let s=P.axes+hist(step,palette.noise,0)+hist(jump,palette.q,half)+P.line(xs,xs.map(x=>Math.min(npdf(x,Math.sqrt(a),sd*sd)*sc,1.1)),palette.ink,'style="stroke-width:2"');
  s+=legend(16,30,[[`${t} small steps`,palette.noise],['one jump',palette.q],[`N(${num(Math.sqrt(a),2)}, ${num(1-a,2)})`,palette.ink]]);
  const m=v=>v.reduce((x,y)=>x+y,0)/v.length,sdv=v=>{const mm=m(v);return Math.sqrt(v.reduce((x,y)=>x+(y-mm)**2,0)/v.length);};
  s+=label(800,90,`t = ${t}, 4,000 runs each`,{fill:palette.ink,size:24,weight:650})+
   label(800,140,`steps: mean ${num(m(step),3)}, sd ${num(sdv(step),3)}`,{fill:palette.noise,size:21})+
   label(800,175,`jump: mean ${num(m(jump),3)}, sd ${num(sdv(jump),3)}`,{fill:palette.q,size:21})+
   label(800,210,`exact: mean ${num(Math.sqrt(a),3)}, sd ${num(sd,3)}`,{fill:palette.ink,size:21})+
   label(800,265,`${t} noise draws per run, or one`,{size:20});
  return svg(W,H,s,'Histograms of the noisy pixel after t steps, made step by step and with one jump.');
 };
 const draw=interactive(el,ctx,{controls:slider('t','step t',1,1000,1,250,'int'),render});
 if(!draw)return;bindSliders(el,state,draw);
}

// ================================================================ Part 2: one step back
/** Bayes for one pixel: likelihoods, prior, posterior and the best guess, under sliders for t and x_t. */
function inkpaper(el,ctx,d){
 const pi=d.px.pi,S=LIN,state={t:250,x:d.px.run.xt},W=1150,H=340,xs=lin(-3,3,301);
 const render=()=>{
  const t=Math.max(1,Math.round(state.t)),a=S.ab[t],v=1-a,sg=Math.sqrt(a),x=state.x,P=pInk(x,t,pi),m=2*P-1;
  const A=panel({W:640,H,L:16,R:14,T:62,B:52,xMin:-3,xMax:3,yMax:1,ticksX:[-2,-1,0,1,2],yAxis:false,xLabel:'x_t'});
  const top=Math.max(pi,1-pi)*npdf(0,0,v),sc=.9/top;
  let s=A.axes+A.area(xs,xs.map(u=>pi*npdf(u,sg,v)*sc),TINT.data)+A.line(xs,xs.map(u=>pi*npdf(u,sg,v)*sc),palette.data)+
   A.area(xs,xs.map(u=>(1-pi)*npdf(u,-sg,v)*sc),TINT.noise)+A.line(xs,xs.map(u=>(1-pi)*npdf(u,-sg,v)*sc),palette.noise)+A.vline(x,palette.ink,'stroke-dasharray="6 5"');
  s+=legend(16,30,[[`ink: ${num(pi,2)} × N(+${num(sg,2)}, ${num(v,2)})`,palette.data],[`paper: ${num(1-pi,2)} × N(−${num(sg,2)}, ${num(v,2)})`,palette.noise]],19);
  const Bx=690,bw=420;
  s+=label(Bx,95,`t = ${t},  x_t = ${num(x,2)}`,{fill:palette.ink,size:26,weight:650})+
   label(Bx,145,`P(ink | x_t) = ${num(P,2)}`,{fill:palette.data,size:24})+`<rect x="${Bx}" y="160" width="${(bw*P).toFixed(1)}" height="26" fill="${palette.data}"/><rect x="${(Bx+bw*P).toFixed(1)}" y="160" width="${(bw*(1-P)).toFixed(1)}" height="26" fill="${palette.noise}" fill-opacity=".8"/>`+
   label(Bx+bw,145,`paper ${num(1-P,2)}`,{fill:palette.noise,size:24,anchor:'end'})+
   label(Bx,240,`best guess E[x₀ | x_t] = ${num(m,2)}`,{fill:palette.q,size:24,weight:650})+
   label(Bx,275,m>.9?'confidently ink':m<-.9?'confidently paper':'a grey: neither paper nor ink',{size:21});
  return svg(W,H,s,`Prior times likelihood for ink and paper at t = ${t}, and the posterior at x_t = ${num(x,2)}.`);
 };
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',1,1000,1,state.t,'int')+slider('x','observed x_t',-3,3,.01,state.x),render});
 if(!draw)return;bindSliders(el,state,draw);
}

/** The exact reverse density q(x_s | x_t) of the pixel for a jump from t back to s, against one Gaussian. */
function reverse(el,ctx,d){
 const pi=d.px.pi,S=LIN,t=500,state={j:500,x:0},W=1150,H=350;
 const render=()=>{
  const j=Math.max(1,Math.round(state.j)),s=t-j,K=reverseKernel(state.x,t,s,pi,S);
  const span=s===0?1.6:Math.max(Math.abs(K.means[0])+3*K.sd,Math.abs(K.means[1])+3*K.sd,K.fitSd*3.2,0.2),xs=lin(-span,span,401);
  const ys=s===0?null:xs.map(K.pdf),fy=xs.map(K.fit),top=Math.max(...(ys||[0]),...fy)*1.1;
  const P=panel({W:760,H,L:30,R:30,T:60,B:52,xMin:-span,xMax:span,yMax:s===0?1.05:top,ticksX:s===0?[-1,0,1]:[-span*.8,0,span*.8].map(v=>+v.toFixed(2)),yAxis:false,xLabel:`x at step ${s}`});
  let g=P.axes;
  if(s===0){g+=[[1,K.weights[0]],[-1,K.weights[1]]].map(([v,w])=>`<line x1="${P.x(v)}" y1="${P.y(0)}" x2="${P.x(v)}" y2="${P.y(w)}" stroke="${palette.p}" stroke-width="7"/>`+label(P.x(v),P.y(w)-12,num(w,2),{fill:palette.p,size:22,anchor:'middle'})).join('');
   const fyS=fy.map(v=>v/Math.max(...fy)*.5);g+=P.line(xs,fyS,palette.q,'stroke-dasharray="9 6"');}
  else g+=P.area(xs,ys,TINT.p)+P.line(xs,ys,palette.p)+P.line(xs,fy,palette.q,'stroke-dasharray="9 6"');
  g+=legend(16,30,[['exact step back',palette.p],['best single Gaussian',palette.q,true]]);
  const sep=s===0?Infinity:Math.abs(K.means[0]-K.means[1])/K.sd;
  g+=label(800,80,`from t = ${t} back to ${s}`,{fill:palette.ink,size:26,weight:650})+label(800,118,`a jump of ${j} step${j>1?'s':''}`,{size:22})+
   label(800,175,s===0?'two spikes: paper or ink':sep>2.5?'two bumps: a Gaussian fails':sep>1?'lopsided: a rough fit':'one bell: a Gaussian fits',{fill:s===0||sep>2.5?palette.loss:sep>1?palette.noise:palette.p,size:24,weight:650})+
   label(800,215,s===0?'':`bump centres ${num(sep,1)} widths apart`,{size:21})+label(800,270,`observed x_t = ${num(state.x,2)}`,{size:20});
  return svg(W,H,g,`The exact distribution of the pixel ${j} steps back from t = 500, and the best single Gaussian.`);
 };
 const print=()=>{let out='';for(const [jj,b] of [[500,'1'],[200,'2'],[1,'3-99']]){state.j=jj;out+=`<div data-only="${b}">${render()}</div>`;}return `<div class="state-slot">${out}</div>`;};
 const draw=interactive(el,ctx,{controls:slider('j','jump size (steps back)',1,500,1,500,'int')+slider('x','observed x_t',-2.5,2.5,.01,0)+`<button type="button" data-play>▶ Shrink the jump</button>`,render,print});
 if(!draw)return;bindSliders(el,state,draw);onBuild(el,ctx,b=>{state.j=[500,200,1][Math.min(b,3)-1];sync(el,state);draw();});
 el.querySelector('[data-play]').addEventListener('click',()=>{const t0=performance.now();const tick=now=>{const u=Math.min(1,(now-t0)/4500);state.j=Math.max(1,Math.round(500*Math.pow(1/500,u)));sync(el,state);draw();if(u<1)requestAnimationFrame(tick);};requestAnimationFrame(tick);});
}

// ================================================================ Part 3: training
function playground(el,ctx,d){
 const pg=d.toy.playground,hist=d.toy.history,data=d.toy.data.x,W=1150,H=360,n=pg.snaps.length;
 const Q=(ox,oy,s)=>({x:v=>ox+(v+1.6)/3.2*s,y:v=>oy+(1.6-v)/3.2*s});
 const render=i=>{
  const sn=pg.snaps[i],A=Q(10,36,325),B=Q(375,36,325),cid=`pg-${Math.random().toString(36).slice(2)}`;
  let g=`<defs>${arrowHead('pg-a',palette.q)}</defs>`;
  g+=data.filter((_,k)=>k%2===0).map(([x,y])=>`<circle cx="${A.x(x).toFixed(1)}" cy="${A.y(y).toFixed(1)}" r="2.2" fill="${palette.data}" fill-opacity=".35"/>`).join('');
  pg.grid.forEach(([x,y],k)=>{const [tx,ty]=sn.x0[k];let dx=tx-x,dy=ty-y;const len=Math.hypot(dx,dy),cap=.3;if(len>cap){dx*=cap/len;dy*=cap/len;}
   g+=`<line x1="${A.x(x).toFixed(1)}" y1="${A.y(y).toFixed(1)}" x2="${A.x(x+dx).toFixed(1)}" y2="${A.y(y+dy).toFixed(1)}" stroke="${palette.q}" stroke-width="1.8" marker-end="url(#pg-a)"/>`;});
  g+=label(10,26,`arrows: xₜ → the network’s x̂₀ (t = ${pg.t})`,{fill:palette.q,size:19});
  g+=`<defs><clipPath id="${cid}"><rect x="375" y="36" width="325" height="325"/></clipPath></defs><g clip-path="url(#${cid})">`+sn.samples.map(([x,y])=>`<circle cx="${B.x(x).toFixed(1)}" cy="${B.y(y).toFixed(1)}" r="2.6" fill="${palette.q}" fill-opacity=".8"/>`).join('')+'</g>';
  const inside=sn.samples.filter(([x,y])=>Math.abs(x)<1.6&&Math.abs(y)<1.6).length;
  g+=`<rect x="375" y="36" width="325" height="325" fill="none" stroke="#E5E7EB"/><rect x="10" y="36" width="325" height="325" fill="none" stroke="#E5E7EB"/>`+label(375,26,'300 samples from this network',{fill:palette.q,size:19})+(inside<150?label(537,200,`${300-inside} of 300 land off this chart`,{fill:palette.q,size:18,anchor:'middle'}):'');
  const K=panel({ox:740,W:410,H,L:58,R:10,T:36,B:48,xMin:0,xMax:4.65,yMin:.15,yMax:.6,ticksX:[1,2,3,4],fmtX:v=>['','10','100','1k','10k'][v],ticksY:[.2,.4,.6],xLabel:'step'});
  const st=hist.map(h=>Math.log10(Math.max(1,h[0]))),ls=hist.map(h=>h[2]);
  const upto=hist.findIndex(h=>h[0]>sn.step);const k2=upto<0?hist.length:upto;
  g+=K.axes+K.line(st,ls,'#C9CED6','style="stroke-width:2"')+K.line(st.slice(0,k2),ls.slice(0,k2),palette.loss)+label(798,26,`loss at step ${sn.step.toLocaleString('en')}`,{fill:palette.loss,size:19});
  return svg(W,H,g,`The smiley network after ${sn.step} training steps: its arrows, its samples and its loss.`);
 };
 animated(el,ctx,{count:n,render,step:i=>`step ${pg.snaps[i].step.toLocaleString('en')}`,alt:'Training the smiley network',duration:6000});
}

function x0film(el,ctx,d){
 const f=d.mn.film,n=f.t.length,state={i:9},W=1150,H=330;
 const render=()=>{
  const i=state.i,b=buildOf(el,ctx),r=f.rows[i],sz=230,cell=(k,src,title,color,note)=>svgFrame(src,i,n,k*285,40,sz)+label(k*285,28,title,{fill:color,size:21,weight:600})+label(k*285,300,note,{size:19});
  let g=cell(0,'mnist-film-xt.png',`x_t at t = ${f.t[i]}`,palette.ink,`√ᾱ = ${num(Math.sqrt(r.alpha_bar),3)}`)+cell(1,'mnist-film-eps.png','ε̂: the predicted noise',palette.noise,`error vs ε: ${num(r.eps_mse,3)}`)+
   cell(2,'mnist-film-x0.png','x̂₀: the network’s guess',palette.q,`error vs our 3: ${num(r.x0_mse,3)}`);
  if(b>=2)g+=cell(3,'mnist-film-alone.png','each pixel alone',palette.p,`error: ${num(r.alone_mse,3)}`);
  return svg(W,H,g,`Our 3 at t = ${f.t[i]}: the noisy input, the network's predicted noise and its guess of the clean image.`);
 };
 const print=()=>{const one=b=>{let g='';[0,4,9,13].forEach((i,k)=>{const r=f.rows[i];[['mnist-film-xt.png',0],['mnist-film-x0.png',1],['mnist-film-alone.png',2]].forEach(([src,j])=>{if(j<2||b>=2)g+=svgFrame(src,i,n,k*290,20+j*100,92);});g+=label(k*290+110,70,`t = ${f.t[i]}`,{fill:palette.ink,size:22,weight:600})+label(k*290+110,170,`x̂₀ error ${num(r.x0_mse,3)}`,{fill:palette.q,size:18})+(b>=2?label(k*290+110,270,`alone ${num(r.alone_mse,3)}`,{fill:palette.p,size:18}):'');});return svg(W,320,g,'Rows: x_t, the network guess, and each pixel alone, at four noise levels.');};return `<div class="state-slot"><div data-only="1">${one(1)}</div><div data-only="2-99">${one(2)}</div></div>`;};
 const draw=interactive(el,ctx,{controls:`<label>noise level<input type="range" min="0" max="${n-1}" step="1" value="${state.i}" data-k="i" aria-label="noise level"><span class="w-readout" data-o="i" data-f="tlev"></span></label>`,render,print});
 if(!draw)return;fmts.tlev=v=>`t = ${f.t[v]}`;bindSliders(el,state,draw);onBuild(el,ctx,()=>draw());
}

// ================================================================ Part 4: generating
function sampler2d(el,ctx,d){
 const P=d.toy.paths,frames=P.ddpm,ts=P.ddpm_t,n=ts.length,data=d.toy.data.x,net=toyNet(d.W.uncond),S=LIN,W=1150,H=375;
 const Q={x:v=>20+(v+1.7)/3.4*350,y:v=>12+(1.7-v)/3.4*350},clip=`s2d-clip-${Math.random().toString(36).slice(2)}`;
 const live=[];
 const dots=(i,shown)=>frames.slice(0,shown).map(p=>{const [x,y]=p[i];return `<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="2.6" fill="${palette.q}" fill-opacity=".85"/>`;}).join('');
 const render=i=>{
  let g=`<defs><clipPath id="${clip}"><rect x="20" y="12" width="350" height="350"/></clipPath></defs><rect x="20" y="12" width="350" height="350" fill="#FFFFFF" stroke="#E5E7EB" data-pad="1"/><g clip-path="url(#${clip})">`;
  g+=data.filter((_,k)=>k%3===0).map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="1.8" fill="${palette.data}" fill-opacity=".25"/>`).join('');
  for(const k of [3,17,42])g+=`<polyline points="${frames[k].slice(0,i+1).map(([x,y])=>`${Q.x(x).toFixed(1)},${Q.y(y).toFixed(1)}`).join(' ')}" fill="none" stroke="${palette.noise}" stroke-width="1.2" stroke-opacity=".7"/>`;
  g+=dots(i,frames.length);
  for(const p of live)g+=`<polyline points="${p.path.map(([x,y])=>`${Q.x(x).toFixed(1)},${Q.y(y).toFixed(1)}`).join(' ')}" fill="none" stroke="${palette.loss}" stroke-width="2"/>`+(p.path.length?`<circle cx="${Q.x(p.path.at(-1)[0]).toFixed(1)}" cy="${Q.y(p.path.at(-1)[1]).toFixed(1)}" r="6" fill="${palette.loss}"/>`:'');
  g+='</g>'+label(430,60,`t = ${ts[i]}`,{fill:palette.ink,size:32,weight:650})+label(430,105,`300 particles, each its own DDPM run`,{size:21,fill:palette.q})+
   label(430,140,'orange: three of their paths',{size:21,fill:palette.noise})+label(430,175,'faint blue: the training data',{size:21,fill:palette.data})+
   (ctx.printing?'':label(430,230,'Click in the square: your own particle,',{size:21,fill:palette.loss})+label(430,258,'denoised live by the network in this page.',{size:21,fill:palette.loss}));
  return svg(W,H,g,`Three hundred particles carried from noise to the smiley by the trained network; t = ${ts[i]}.`);
 };
 const A=animated(el,ctx,{count:n,render,step:i=>`t = ${ts[i]}`,alt:'Sampling the smiley',duration:6500});
 if(ctx.printing)return;
 el.querySelector('.w-fig').addEventListener('click',e=>{
  const s=e.currentTarget.querySelector('svg');if(!s)return;const r=s.getBoundingClientRect(),vx=(e.clientX-r.left)/r.width*W,vy=(e.clientY-r.top)/r.height*H;
  if(vx<20||vx>370||vy<12||vy>362)return;
  let x=(vx-20)/350*3.4-1.7,y=1.7-(vy-12)/350*3.4;const p={path:[[x,y]]};live.push(p);if(live.length>6)live.shift();
  const rn=randn(1000+live.length*7);let t=T;
  const tick=()=>{for(let k=0;k<25&&t>0;k++,t--){const [ex,ey]=net(x,y,t),b=S.beta[t],a=S.alpha[t],ab=S.ab[t],c=b/Math.sqrt(1-ab),sd=t>1?Math.sqrt(S.btil[t]):0;
    x=(x-c*ex)/Math.sqrt(a)+sd*rn();y=(y-c*ey)/Math.sqrt(a)+sd*rn();if(t%5===0)p.path.push([x,y]);}
   p.path.push([x,y]);A.rerender();if(t>0)requestAnimationFrame(tick);};
  requestAnimationFrame(tick);
 });
}

function trajectories(el,ctx,d){
 const ts=d.mn.trajectory_t,n=ts.length,W=1150,H=300;
 const render=i=>{let g='';for(let s=0;s<8;s++)g+=svgFrame(`mnist-traj-${s}-x.png`,i,n,s*143,30,116)+svgFrame(`mnist-traj-${s}-x0.png`,i,n,s*143,170,116);
  return svg(W,H,g+label(0,22,`x_t at t = ${ts[i]}`,{fill:palette.ink,size:20,weight:600})+label(0,163,'x̂₀: the network’s guess of the finished digit',{fill:palette.q,size:20,weight:600}),`Eight sampling runs at t = ${ts[i]}.`);};
 animated(el,ctx,{count:n,render,step:i=>`t = ${ts[i]}`,alt:'Eight sampling runs',duration:7000});
}

// ================================================================ Part 5: the score
/** The learned score field of the smiley at noise level t: arrows from x towards sqrt(ab)·x̂₀. */
function scorefield(el,ctx,d){
 const net=toyNet(d.W.uncond),data=d.toy.data.x,S=LIN,state={t:100},W=1150,H=348,g0=lin(-1.6,1.6,19);
 const field=(t,ox,s,cap=.22)=>{const Q={x:v=>ox+(v+1.7)/3.4*s,y:v=>15+(1.7-v)/3.4*s},ab=S.ab[t];
  let g=`<rect x="${ox}" y="15" width="${s}" height="${s}" fill="#FFFFFF" stroke="#E5E7EB"/>`+data.filter((_,k)=>k%3===0).map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="1.7" fill="${palette.data}" fill-opacity=".3"/>`).join('');
  for(const y of g0)for(const x of g0){const [ex,ey]=net(x,y,t);let dx=-Math.sqrt(1-ab)*ex,dy=-Math.sqrt(1-ab)*ey;const L=Math.hypot(dx,dy);if(L>cap){dx*=cap/L;dy*=cap/L;}
   g+=`<line x1="${Q.x(x).toFixed(1)}" y1="${Q.y(y).toFixed(1)}" x2="${Q.x(x+dx).toFixed(1)}" y2="${Q.y(y+dy).toFixed(1)}" stroke="${palette.q}" stroke-width="1.7" marker-end="url(#sf-a)"/>`;}
  return g;};
 const render=()=>svg(W,H,`<defs>${arrowHead('sf-a',palette.q)}</defs>`+field(state.t,20,325)+label(400,60,`noise level t = ${state.t}`,{fill:palette.ink,size:30,weight:650})+
  label(400,105,`each arrow: (1 − ᾱₜ) · score = −√(1 − ᾱₜ) ε̂`,{fill:palette.q,size:21})+label(400,140,'it points from x towards √ᾱₜ x̂₀',{size:21})+label(400,195,state.t<=30?'low noise: sharp arrows next to the data,':'high noise: smooth arrows everywhere,',{size:21,fill:palette.ink})+
  label(400,223,state.t<=30?'unreliable far away (no training points there)':'pointing at a blurred, shrunk smiley',{size:21,fill:palette.ink}),`Arrows of the learned score at noise level ${state.t}.`);
 const print=()=>svg(W,340,`<defs>${arrowHead('sf-a',palette.q)}</defs>`+[[10,40],[100,420],[500,800]].map(([t,ox])=>field(t,ox,300)+label(ox+150,336,`t = ${t}`,{size:22,anchor:'middle',fill:palette.ink})).join(''),'The learned score at three noise levels.');
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',1,1000,1,100,'int')+`<button type="button" data-go="10">t = 10</button><button type="button" data-go="100">t = 100</button><button type="button" data-go="500">t = 500</button>`,render,print});
 if(!draw)return;bindSliders(el,state,draw);
 el.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>{state.t=Number(b.dataset.go);sync(el,state);draw();}));
}

/** Langevin dynamics with the learned score: at one noise level, or annealed from t = 1000 down to 1. */
function langevin(el,ctx,d){
 const net=toyNet(d.W.uncond),data=d.toy.data.x,S=LIN,W=1150,H=440,N=220,state={t:10};
 const Q={x:v=>20+(v+1.7)/3.4*400,y:v=>20+(1.7-v)/3.4*400};
 let pts=[],raf=0,status='';
 const reset=()=>{const r=randn(77);pts=Array.from({length:N},()=>[r(),r()]);status='start: pure noise';};
 const eta=t=>.1*(1-S.ab[t]);
 const step=(t,eta,r)=>{const s1=Math.sqrt(1-S.ab[t]);for(const p of pts){const [ex,ey]=net(p[0],p[1],t);p[0]+=eta/2*(-ex/s1)+Math.sqrt(eta)*r();p[1]+=eta/2*(-ey/s1)+Math.sqrt(eta)*r();}};
 const render=()=>{let g=`<rect x="20" y="20" width="400" height="400" fill="#FFFFFF" stroke="#E5E7EB"/>`+data.filter((_,k)=>k%3===0).map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="1.7" fill="${palette.data}" fill-opacity=".25"/>`).join('');
  g+=pts.map(([x,y])=>`<circle cx="${Q.x(Math.max(-1.75,Math.min(1.75,x))).toFixed(1)}" cy="${Q.y(Math.max(-1.75,Math.min(1.75,y))).toFixed(1)}" r="3" fill="${palette.loss}" fill-opacity=".8"/>`).join('');
  return svg(W,H,g+label(460,60,status,{fill:palette.ink,size:28,weight:650})+label(460,110,'x ← x + (η/2) · score(x) + √η · z',{fill:palette.loss,size:22})+
   label(460,165,'One level: follow the arrows of that noise level.',{size:21})+label(460,200,'Annealed: start at t = 1000, lower the noise',{size:21})+label(460,228,'step by step, as DDPM does.',{size:21}),'Particles moved by Langevin dynamics with the learned score.');};
 reset();
 if(ctx.printing){const r=randn(3);for(let t=1000;t>=1;t-=4)step(t,eta(t),r);status='annealed, t = 1000 → 1';el.innerHTML=render();return;}
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',1,300,1,10,'int')+`<button type="button" data-run>▶ 300 steps at this level</button><button type="button" data-anneal>▶ Annealed: t = 1000 → 1</button><button type="button" data-reset>Reset</button>`,render});
 if(!draw)return;bindSliders(el,state,draw);
 const run=(levels,label_)=>{cancelAnimationFrame(raf);const r=randn(5);let k=0;const tick=()=>{for(let j=0;j<6&&k<levels.length;j++,k++){const t=levels[k];step(t,eta(t),r);}status=label_(levels[Math.min(k,levels.length-1)]);draw();if(k<levels.length)raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);};
 el.querySelector('[data-run]').addEventListener('click',()=>{reset();run(Array(300).fill(state.t),t=>`level t = ${t}`);});
 el.querySelector('[data-anneal]').addEventListener('click',()=>{reset();const lv=[];for(let t=1000;t>=1;t-=4)lv.push(t);run(lv,t=>`annealed, now t = ${t}`);});
 el.querySelector('[data-reset]').addEventListener('click',()=>{cancelAnimationFrame(raf);reset();draw();});
}

// ================================================================ Part 6: faster and steerable
function paths(el,ctx,d){
 const P=d.toy.paths,W=1150,H=385,sel=Array.from({length:16},(_,k)=>k*17%300),data=d.toy.data.x;
 const box=(ox,arr,title,color,i,n)=>{const Q={x:v=>ox+(v+1.7)/3.4*345,y:v=>34+(1.7-v)/3.4*345},cid=`pth-${ox}-${Math.random().toString(36).slice(2)}`;
  let g=`<defs><clipPath id="${cid}"><rect x="${ox}" y="34" width="345" height="345"/></clipPath></defs><rect x="${ox}" y="34" width="345" height="345" fill="#FFFFFF" stroke="#E5E7EB"/><g clip-path="url(#${cid})">`+data.filter((_,k)=>k%3===0).map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="1.6" fill="${palette.data}" fill-opacity=".25"/>`).join('');
  const upto=Math.round(i/(n-1)*(arr[0].length-1));
  for(const k of sel)g+=`<polyline points="${arr[k].slice(0,upto+1).map(([x,y])=>`${Q.x(x).toFixed(1)},${Q.y(y).toFixed(1)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.6" stroke-opacity=".8"/><circle cx="${Q.x(arr[k][upto][0]).toFixed(1)}" cy="${Q.y(arr[k][upto][1]).toFixed(1)}" r="3.2" fill="${color}"/>`;
  return g+'</g>'+label(ox,26,title,{fill:color,size:21,weight:600});};
 const n=51,render=i=>svg(W,H,box(60,P.ddpm,'DDPM: 1,000 steps with fresh noise',palette.noise,i,n)+box(640,P.ddim,'DDIM: 50 steps, no fresh noise',palette.q,i,n),'Forty sampling paths of the same network: DDPM and DDIM.');
 animated(el,ctx,{count:n,render,step:i=>`${Math.round(i/(n-1)*100)}% of the way`,alt:'DDPM and DDIM paths',duration:5000});
}

export const STEPS=[1,2,5,10,20,50,100,1000];
function ddimsteps(el,ctx,d){
 const sw=d.mn.ddim,state={k:0},W=1150,H=300;
 const render=()=>{const s=STEPS[state.k],sec=sw.find(r=>r.steps===s)?.seconds;
  return `<div class="steps-fig"><div class="row-title" style="color:${palette.q}">DDIM, ${s} step${s>1?'s':''} · ${num(sec,3)} s for 8 digits</div>${pic(`mnist-ddim-${s}.png`,1080,`DDIM samples with ${s} steps`)}<div class="row-title" style="color:${palette.noise};margin-top:16px">DDPM, 1,000 steps with fresh noise (same starting noise)</div>${pic('mnist-ddim-ddpm.png',1080,'DDPM samples')}</div>`;};
 const print=()=>{state.k=4;return render();};
 const draw=interactive(el,ctx,{controls:`<label>steps<input type="range" min="0" max="${STEPS.length-1}" step="1" value="0" data-k="k" aria-label="number of DDIM steps"><span class="w-readout" data-o="k" data-f="steps"></span></label>`,render,print});
 if(!draw)return;fmts.steps=v=>String(STEPS[v]);bindSliders(el,state,draw);
}

/** Our 3 at noise level t under the linear and the cosine schedule, with their signal curves. */
function schedules(el,ctx,d){
 const {x0,eps}=d.mn.held,state={t:600},W=1150,H=330,ts=lin(0,1000,201);
 const xt=(S,t)=>x0.map((v,i)=>Math.sqrt(S.ab[t])*v+Math.sqrt(1-S.ab[t])*eps[i]);
 const render=()=>{const t=Math.round(state.t),P=panel({ox:600,W:550,H,L:58,R:30,T:46,B:52,xMin:0,xMax:1000,yMax:1.02,ticksX:[0,250,500,750,1000],ticksY:[0,.5,1],xLabel:'t'});
  let g=digitSVG(xt(LIN,t),{size:250,x:0,y:40})+digitSVG(xt(COS,t),{size:250,x:290,y:40})+label(0,28,'linear',{fill:palette.noise,size:22,weight:600})+label(290,28,'cosine',{fill:palette.p,size:22,weight:600})+
   label(0,322,`signal √ᾱ = ${num(Math.sqrt(LIN.ab[t]),3)}`,{fill:palette.noise,size:19})+label(290,322,`signal √ᾱ = ${num(Math.sqrt(COS.ab[t]),3)}`,{fill:palette.p,size:19});
  g+=P.axes+P.line(ts,ts.map(u=>Math.sqrt(LIN.ab[Math.round(u)])),palette.noise)+P.line(ts,ts.map(u=>Math.sqrt(COS.ab[Math.round(u)])),palette.p)+P.vline(t,palette.ink,'stroke-dasharray="6 5"')+label(658,30,'signal √ᾱₜ',{size:20});
  return svg(W,H,g,`Our 3 at t = ${t} under the linear and the cosine schedule.`);};
 const draw=interactive(el,ctx,{controls:slider('t','noise level t',0,1000,1,600,'int'),render});
 if(!draw)return;bindSliders(el,state,draw);
}

export const WS=[0,1,2,3,5,7];
function guidance(el,ctx,d){
 const G=d.mn.guidance,toy=d.toy.guided,data=d.toy.data.x,state={k:3,part:'mouth'},W=1150,H=330;
 const render=(b=buildOf(el,ctx))=>{const w=WS[state.k],r=G.find(x=>x.w===w),tw=[0,1,2,3,5].reduce((a,v)=>Math.abs(v-w)<Math.abs(a-w)?v:a,0),tg=toy[`${state.part}-${tw}`];
  const Q={x:v=>820+(v+1.4)/2.8*300,y:v=>28+(1.4-v)/2.8*300};
  let side='';
  if(b>=2)side=`<svg viewBox="0 0 1150 330" class="plot wsvg" style="position:absolute;inset:0;pointer-events:none"><rect x="820" y="28" width="300" height="300" fill="#FFFFFF" stroke="#E5E7EB"/>${data.filter((_,k)=>k%3===0).map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="1.4" fill="${palette.data}" fill-opacity=".22"/>`).join('')}${tg.x.map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="2.3" fill="${palette.q}" fill-opacity=".8"/>`).join('')}${label(820,20,`smiley guided to “${state.part}”, w = ${tw}`,{fill:palette.q,size:16})}</svg>`;
  const bar=(y,lab,v,max,color,txt)=>`<div class="g-row" style="top:${y}px"><span style="color:${color}">${lab}</span><span class="g-bar"><i style="width:${(v/max*100).toFixed(1)}%;background:${color}"></i></span><b style="color:${color}">${txt}</b></div>`;
  return `<div class="guide-fig" style="height:${H}px">${pic(`mnist-cfg-${w}.png`,500,`Sixteen samples guided to a 3 with w = ${w}`,'position:absolute;left:0;top:30px')}<div class="row-title" style="position:absolute;left:0;top:0;color:${palette.q}">16 samples asked for a 3, w = ${w}</div>${bar(40,'classified as 3',r.is_three,1,palette.p,`${Math.round(r.is_three*100)}%`)}${bar(110,'variety (spread)',r.diversity,20,palette.noise,num(r.diversity,1))}${bar(180,'ink per image',r.ink,.2,palette.data,`${Math.round(r.ink*100)}%`)}${side}</div>`;};
 const draw=interactive(el,ctx,{controls:`<label>guidance w<input type="range" min="0" max="${WS.length-1}" step="1" value="3" data-k="k" aria-label="guidance scale"><span class="w-readout" data-o="k" data-f="w"></span></label><button type="button" data-part="mouth" aria-pressed="true">mouth</button><button type="button" data-part="eyes" aria-pressed="false">eyes</button>`,render:()=>render(),print:()=>`<div class="state-slot"><div data-only="1">${render(1)}</div><div data-only="2-99">${render(2)}</div></div>`});
 if(!draw)return;fmts.w=v=>String(WS[v]);bindSliders(el,state,draw);onBuild(el,ctx,()=>draw());
 el.querySelectorAll('[data-part]').forEach(b=>b.addEventListener('click',()=>{state.part=b.dataset.part;el.querySelectorAll('[data-part]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
}

// ================================================================ Part 7: does it copy?
function copy(el,ctx,d){
 const C=d.toy.copy,W=1150,H=395;
 const box=(ox,pts,color,title,sub)=>{const Q={x:v=>ox+(v+1.25)/2.5*320,y:v=>32+(1.25-v)/2.5*320};
  return `<rect x="${ox}" y="32" width="320" height="320" fill="#FFFFFF" stroke="#E5E7EB"/>`+pts.map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="2.6" fill="${color}" fill-opacity=".75"/>`).join('')+
   C.train.map(([x,y])=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="5" fill="none" stroke="${palette.ink}" stroke-width="2"/>`).join('')+label(ox,24,title,{fill:color,size:21,weight:600})+label(ox,380,sub,{size:18});};
 const render=b=>svg(W,H,box(40,C.ideal,palette.noise,'the best possible denoiser',`400 samples, mean distance to a training point ${num(C.ideal_to_train,4)}`)+(b>=2?box(600,C.net,palette.q,'our network, trained on the same 40',`mean distance to a training point ${num(C.net_to_train,3)}; to the face ${num(C.net_to_shape,3)}`):''),'Samples from the ideal denoiser of 40 training points, and from a network trained on them.');
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1" class="diagram-scroll">${render(1)}</div><div data-only="2-99" class="diagram-scroll">${render(2)}</div></div>`;return;}
 const draw=b=>{el.innerHTML=`<div class="diagram-scroll">${render(b)}</div>`;};draw(buildOf(el,ctx));onBuild(el,ctx,draw);
}

// ================================================================ calculators: every number of the forward and reverse steps
const K=(tex,display=false)=>{try{return katex.renderToString(tex,{displayMode:display,throwOnError:false});}catch{return esc(tex);}};
const tx=(v,n=3)=>{const s=Number(v).toFixed(n);return /^-0(\.0*)?$/.test(s)?s.slice(1):s;};
const par=v=>v<0?`(${tx(v)})`:tx(v);
const SCHED={toy:TOY,real:LIN};
/** ᾱ_t written as the product of the alphas, all of them for the toy, the first and last few for the real schedule. */
function abProduct(S,t){
 if(t===0)return R0`\bar\alpha_0=1`;
 const f=k=>tx(S.alpha[k],S===TOY?1:4);
 const parts=t<=5?Array.from({length:t},(_,k)=>f(k+1)):[f(1),f(2),f(3),'\\cdots',f(t)];
 const names=t<=5?Array.from({length:t},(_,k)=>`\\alpha_{${k+1}}`).join(''):`\\alpha_1\\alpha_2\\alpha_3\\cdots\\alpha_{${t}}`;
 return `\\bar\\alpha_{${t}}=${names}=${parts.join('\\times ')}=${tx(S.ab[t],4)}`;
}
const R0=String.raw;
function scheduleRows(S,t){
 const rows=S===TOY?[1,2,3,4,5]:[...new Set([1,2,3,Math.max(1,t-1),t,Math.min(S.T,t+1)])].sort((a,b)=>a-b);
 let h=`<table class="calc-table"><thead><tr><th>t</th><th>${K('\\beta_t')}</th><th>${K('\\alpha_t=1-\\beta_t')}</th><th>${K('\\bar\\alpha_t')}</th><th>${K('\\sqrt{\\bar\\alpha_t}')}</th><th>${K('\\sqrt{1-\\bar\\alpha_t}')}</th></tr></thead><tbody>`;
 let prev=0;for(const k of rows){if(S!==TOY&&k>prev+1)h+=`<tr class="gap"><td colspan="6">⋮</td></tr>`;prev=k;
  h+=`<tr class="${k===t?'now':''}"><td>${k}</td><td>${tx(S.beta[k],S===TOY?1:5)}</td><td>${tx(S.alpha[k],S===TOY?1:5)}</td><td>${tx(S.ab[k],4)}</td><td>${tx(Math.sqrt(S.ab[k]),3)}</td><td>${tx(Math.sqrt(1-S.ab[k]),3)}</td></tr>`;}
 return h+'</tbody></table>';
}
function calcForward(el,ctx){
 const state={s:'toy',t:3,x0:1,eps:-.5};
 const render=()=>{const S=SCHED[state.s],t=Math.min(state.t,S.T),a=S.ab[t],sa=Math.sqrt(a),sn=Math.sqrt(1-a),xt=sa*state.x0+sn*state.eps;
  const P=panel({ox:0,W:470,H:200,L:12,R:12,T:40,B:44,xMin:-3.2,xMax:3.2,yMax:1,ticksX:[-3,-2,-1,0,1,2,3],yAxis:false,xLabel:'pixel value',fs:17}),xs=lin(-3.2,3.2,241),v=Math.max(1-a,1e-4),pk=npdf(0,0,v);
  const plot=svg(470,200,P.axes+P.area(xs,xs.map(x=>Math.min(npdf(x,sa*state.x0,v)/pk*.85,1)),TINT.noise)+P.line(xs,xs.map(x=>Math.min(npdf(x,sa*state.x0,v)/pk*.85,1)),palette.noise,'style="stroke-width:2"')+
   P.dot(state.x0,0,palette.data,7)+P.vline(xt,palette.q)+label(P.x(state.x0),P.y(0)-14,'x₀',{fill:palette.data,size:20,anchor:'middle',weight:600})+label(P.x(xt),52,`xₜ = ${num(xt,3)}`,{fill:palette.q,size:20,anchor:'middle',weight:600})+
   label(12,22,`where xₜ can land: N(${num(sa*state.x0,2)}, ${num(1-a,3)})`,{fill:palette.noise,size:18}),'The distribution of the noisy pixel and the drawn value.');
  return `<div class="calc"><div class="calc-lines">
   <div>${K(`\\beta_{${t}}=${tx(S.beta[t],S===TOY?1:5)},\\qquad \\alpha_{${t}}=1-\\beta_{${t}}=${tx(S.alpha[t],S===TOY?1:5)}`)}</div>
   <div>${K(abProduct(S,t))}</div>
   <div>${K(`\\sqrt{\\bar\\alpha_{${t}}}=${tx(sa)},\\qquad\\sqrt{1-\\bar\\alpha_{${t}}}=${tx(sn)}`)}</div>
   <div class="calc-main">${K(`x_{${t}}=\\sqrt{\\bar\\alpha_{${t}}}\\,x_0+\\sqrt{1-\\bar\\alpha_{${t}}}\\,\\varepsilon=${tx(sa)}\\times${par(state.x0)}+${tx(sn)}\\times${par(state.eps)}=\\mathbf{${tx(xt)}}`)}</div>
   </div><div class="calc-side">${plot}${scheduleRows(S,t)}</div></div>`;};
 const print=()=>{state.t=3;return render();};
 const controls=`<button type="button" data-s="toy" aria-pressed="true">toy: T = 5</button><button type="button" data-s="real" aria-pressed="false">real: T = 1,000</button>`+slider('t','step t',1,5,1,3,'int')+
  `<button type="button" data-x0="1" aria-pressed="true">x₀ = +1 ink</button><button type="button" data-x0="-1" aria-pressed="false">x₀ = −1 paper</button>`+slider('eps','noise ε',-2.5,2.5,.05,-.5);
 const draw=interactive(el,ctx,{controls,render,print});if(!draw)return;
 bindSliders(el,state,draw);
 const tIn=el.querySelector('[data-k="t"]');
 el.querySelectorAll('[data-s]').forEach(b=>b.addEventListener('click',()=>{state.s=b.dataset.s;const S=SCHED[state.s];tIn.max=String(S.T);state.t=state.s==='toy'?3:250;sync(el,state);el.querySelectorAll('[data-s]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
 el.querySelectorAll('[data-x0]').forEach(b=>b.addEventListener('click',()=>{state.x0=Number(b.dataset.x0);el.querySelectorAll('[data-x0]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
}
function calcReverse(el,ctx,d){
 const pi=d.px.pi;let s='toy',S=TOY,t=5,x=0,trail=[],run=0,tally={ink:0,paper:0},rn=randn(4242),first=true;
 const start=()=>{S=SCHED[s];t=S.T;if(s==='toy'&&first){x=forwardRun();}else x=rn();trail=[[t,x]];};
 const forwardRun=()=>{let v=1;for(let k=1;k<=5;k++)v=forwardStep(v,k,TOY_EPS[k],TOY);return v;};
 const zFor=k=>s==='toy'&&first?TOY_Z[k]:rn();
 const stepBack=()=>{if(t<1)return;const e=epsStar(x,t,pi,S),R=reverseStep(x,t,e,zFor(t),S);last={t,x,e,...R};x=R.next;t-=1;trail.push([t,x]);if(t===0){run++;x>0?tally.ink++:tally.paper++;}};
 let last=null;
 const render=()=>{
  const cur=t>=1?(()=>{const e=epsStar(x,t,pi,S),R=reverseStep(x,t,e,0,S);return {t,x,e,...R};})():null;
  const L=last;
  let lines='';
  if(cur){lines+=`<div>${K(`t=${cur.t},\\qquad x_{${cur.t}}=${tx(cur.x)}`)}</div>`+
   `<div>${K(`\\hat\\varepsilon=\\text{network}(x_{${cur.t}},${cur.t})=${tx(cur.e)}`)}<span class="calc-note">the best possible guess for this one-pixel dataset</span></div>`+
   `<div>${K(`\\mu=\\frac{1}{\\sqrt{\\alpha_{${cur.t}}}}\\Big(x_{${cur.t}}-\\frac{\\beta_{${cur.t}}}{\\sqrt{1-\\bar\\alpha_{${cur.t}}}}\\,\\hat\\varepsilon\\Big)=\\frac{1}{${tx(Math.sqrt(S.alpha[cur.t]))}}\\Big(${tx(cur.x)}-\\frac{${tx(S.beta[cur.t],S===TOY?1:4)}}{${tx(Math.sqrt(1-S.ab[cur.t]))}}\\times${par(cur.e)}\\Big)=${tx(cur.mu)}`)}</div>`+
   `<div class="calc-main">${K(cur.t>1?`x_{${cur.t-1}}=\\mu+\\sqrt{\\beta_{${cur.t}}}\\,z=${tx(cur.mu)}+${tx(cur.sigma)}\\,z`:`x_0=\\mu=${tx(cur.mu)}\\quad\\text{(no noise at the last step)}`)}</div>`+
   `<div class="calc-aside">${K(`\\text{the network's guess of the clean pixel: }\\hat x_0=\\frac{x_{${cur.t}}-\\sqrt{1-\\bar\\alpha_{${cur.t}}}\\,\\hat\\varepsilon}{\\sqrt{\\bar\\alpha_{${cur.t}}}}=${tx(cur.x0)}`)}</div>`;}
  else lines+=`<div class="calc-main">${K(`x_0=${tx(x)}\\ \\Rightarrow\\ \\text{${x>0?'ink':'paper'}}`)}</div><div>${K(`\\text{last step: }z\\text{ was }${L?tx(L.z,2):'0'}`)}</div>`;
  if(L&&cur)lines+=`<div class="calc-prev">${K(`\\text{previous step: }x_{${L.t-1}}=${tx(L.mu)}+${tx(L.sigma)}\\times${par(L.z)}=${tx(L.next)}`)}</div>`;
  const P=panel({W:470,H:260,L:44,R:16,T:44,B:46,fs:17,xMin:S.T,xMax:0,yMin:-2.6,yMax:2.6,ticksX:S===TOY?[5,4,3,2,1,0]:[1000,750,500,250,0],ticksY:[-2,-1,0,1,2],xLabel:'t (going back)'});
  const plot=svg(470,260,P.axes+`<rect x="${P.x(S.T)}" y="${P.y(1.08)}" width="${P.x(0)-P.x(S.T)}" height="${P.y(.92)-P.y(1.08)}" fill="${TINT.data}"/><rect x="${P.x(S.T)}" y="${P.y(-.92)}" width="${P.x(0)-P.x(S.T)}" height="${P.y(-1.08)-P.y(-.92)}" fill="${TINT.noise}"/>`+
   label(P.x(0)-4,P.y(1)-6,'ink',{fill:palette.data,size:17,anchor:'end'})+label(P.x(0)-4,P.y(-1)-6,'paper',{fill:palette.noise,size:17,anchor:'end'})+
   P.line(trail.map(p=>p[0]),trail.map(p=>p[1]),palette.q)+trail.filter((_,k)=>S===TOY||k===trail.length-1).map(([tt,v])=>P.dot(tt,v,palette.q,5)).join('')+
   label(44,24,`run ${run+(t>0?1:0)} · finished runs: ink ${tally.ink}, paper ${tally.paper}`,{size:17,fill:palette.ink}),'The path of the pixel going back from noise.');
  return `<div class="calc"><div class="calc-lines">${lines}</div><div class="calc-side">${plot}</div></div>`;
 };
 start();
 if(ctx.printing){while(t>2)stepBack();el.innerHTML=render();return;}
 const controls=`<button type="button" data-s="toy" aria-pressed="true">toy: T = 5</button><button type="button" data-s="real" aria-pressed="false">real: T = 1,000</button><button type="button" data-step="1">◀ one step back</button><button type="button" data-step="10">◀◀ 10 steps</button><button type="button" data-step="all">▶▶ to the end</button><button type="button" data-new>new run</button>`;
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;
 el.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{const n=b.dataset.step==='all'?S.T:Number(b.dataset.step);for(let k=0;k<n&&t>0;k++)stepBack();draw();}));
 el.querySelector('[data-new]').addEventListener('click',()=>{first=false;last=null;start();draw();});
 el.querySelectorAll('[data-s]').forEach(b=>b.addEventListener('click',()=>{s=b.dataset.s;first=s==='toy'&&run===0;last=null;tally={ink:0,paper:0};run=0;start();el.querySelectorAll('[data-s]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
}
/** Classifier-free guidance as arrows: the network's two noise guesses at one point, and the guided one. */
function cfgvec(el,ctx,d){
 const net=toyNet(d.W.cond),data=d.toy.data.x,S=LIN,state={w:3,t:300,part:2,px:.15,py:-.05},W=1150,H=350,SC=3;
 const Q={x:v=>20+(v+1.4)/2.8*330,y:v=>10+(1.4-v)/2.8*330};
 const render=()=>{
  const t=Math.round(state.t),s1=Math.sqrt(1-S.ab[t]),eu=net(state.px,state.py,t,3),ec=net(state.px,state.py,t,state.part),w=state.w;
  const eg=[eu[0]+w*(ec[0]-eu[0]),eu[1]+w*(ec[1]-eu[1])];
  const arrow=(e,color,dash='')=>{const dx=-s1*e[0]*SC,dy=-s1*e[1]*SC;return `<line x1="${Q.x(state.px)}" y1="${Q.y(state.py)}" x2="${Q.x(state.px+dx)}" y2="${Q.y(state.py+dy)}" stroke="${color}" stroke-width="4" ${dash} marker-end="url(#cv-${color.slice(1)})"/>`;};
  let g=`<defs>${[palette.ink,palette.q,palette.loss].map(c=>arrowHead('cv-'+c.slice(1),c)).join('')}</defs><rect x="20" y="10" width="330" height="330" fill="#FFFFFF" stroke="#E5E7EB"/>`;
  g+=data.filter((_,k)=>k%2===0).map(([x,y],k)=>`<circle cx="${Q.x(x).toFixed(1)}" cy="${Q.y(y).toFixed(1)}" r="2" fill="${d.toy.data.part[k*2]===state.part?palette.q:palette.data}" fill-opacity="${d.toy.data.part[k*2]===state.part?.45:.18}"/>`).join('');
  g+=arrow(eu,palette.ink)+arrow(ec,palette.q)+arrow(eg,palette.loss)+`<circle cx="${Q.x(state.px)}" cy="${Q.y(state.py)}" r="7" fill="${palette.ink}"/>`;
  const name=['outline','eyes','mouth'][state.part];
  g+=label(390,40,`at the black point, t = ${t}`,{fill:palette.ink,size:24,weight:650})+
   label(390,80,`no label:  ε̂(∅) = (${num(eu[0],2)}, ${num(eu[1],2)})`,{fill:palette.ink,size:21})+
   label(390,112,`“${name}”:  ε̂(c) = (${num(ec[0],2)}, ${num(ec[1],2)})`,{fill:palette.q,size:21})+
   label(390,152,`guided:  ε̂(∅) + ${num(w,1)} × (ε̂(c) − ε̂(∅)) = (${num(eg[0],2)}, ${num(eg[1],2)})`,{fill:palette.loss,size:21})+
   label(390,200,w>1?'w > 1: the red arrow goes past the purple one,':w===1?'w = 1: exactly the label’s own guess.':'w < 1: between the two guesses.',{size:19,fill:palette.loss})+(w>1?label(390,226,'an extra push away from “any part of the face”.',{size:19,fill:palette.loss}):'')+
   label(390,272,'Arrows: the direction −ε̂ in which each guess moves the point,',{size:18})+label(390,296,'drawn three times longer than −√(1 − ᾱ) ε̂ to be visible.',{size:18})+
   label(390,336,'Click in the square to move the point.',{size:18});
  return svg(W,H,g,`Classifier-free guidance at one point: the unconditional, conditional and guided noise guesses as arrows (w = ${w}).`);
 };
 const controls=slider('w','guidance w',0,7,.5,3,'d1')+slider('t','noise level t',50,800,10,300,'int')+`<button type="button" data-part="2" aria-pressed="true">mouth</button><button type="button" data-part="1" aria-pressed="false">eyes</button><button type="button" data-part="0" aria-pressed="false">outline</button>`;
 const draw=interactive(el,ctx,{controls,render});if(!draw)return;bindSliders(el,state,draw);
 el.querySelectorAll('[data-part]').forEach(b=>b.addEventListener('click',()=>{state.part=Number(b.dataset.part);el.querySelectorAll('[data-part]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
 el.querySelector('.w-fig').addEventListener('click',e=>{const s=e.currentTarget.querySelector('svg'),r=s.getBoundingClientRect(),vx=(e.clientX-r.left)/r.width*W,vy=(e.clientY-r.top)/r.height*H;if(vx<20||vx>350||vy<10||vy>340)return;state.px=(vx-20)/330*2.8-1.4;state.py=1.4-(vy-10)/330*2.8;draw();});
}

export const CSS=`
.w-fig .katex svg,.calc .katex svg{height:inherit;display:block}
.toc-flow{column-count:4;font-size:16.5px;column-gap:30px}
.calc{display:grid;grid-template-columns:1.3fr 1fr;gap:20px;align-items:start;width:100%}
.calc-lines{display:flex;flex-direction:column;gap:10px;font-size:22px}.calc-lines .katex{font-size:1.05em}
.calc-main{font-size:25px;padding:6px 12px;border-left:4px solid var(--q);background:#8B2CDE0D;border-radius:4px}
.calc-note{display:block;font-size:16px;color:var(--muted);margin-top:2px}.calc-aside{font-size:21px;color:var(--muted)}.calc-prev{font-size:19px;color:var(--muted)}
.calc-side{display:flex;flex-direction:column;gap:8px}
.calc-table{border-collapse:collapse;font-size:15px;font-variant-numeric:tabular-nums;width:100%}.calc-table th,.calc-table td{padding:2px 7px;text-align:right;border-bottom:1px solid #E5E7EB}
.calc-table th{font-weight:600;color:var(--muted)}.calc-table tr.now td{background:#8B2CDE1A;color:var(--ink);font-weight:650}.calc-table tr.gap td{text-align:center;color:var(--muted);border:0;padding:0}
.eq-sm .display{font-size:25px}
.hand td,.hand th{font-size:21px;padding:6px 12px;text-align:right}.hand td:first-child,.hand th:first-child{text-align:center}
.hand tr.done td{color:var(--ink)}.hand .hl{color:var(--q);font-weight:650}`;

export const CSS_BASE=`
.mn{display:inline-block;image-rendering:pixelated;background-repeat:no-repeat;border:1px solid #D7DCE1;border-radius:4px;vertical-align:middle;background-color:#FAFAFB}
img.pix{image-rendering:pixelated;display:block;border-radius:4px;max-width:100%;height:auto}
.steps-fig .row-title,.guide-fig .row-title{font-size:20px;font-weight:600;margin:0 0 6px}
.guide-fig{position:relative;width:1150px;max-width:100%}
.g-row{position:absolute;left:530px;width:270px;display:grid;grid-template-columns:1fr;gap:4px;font-size:19px}
.g-row .g-bar{display:block;height:16px;background:#EEF0F3;border-radius:3px;overflow:hidden}.g-row .g-bar i{display:block;height:100%}
.g-row b{font-size:22px}
.thumbs{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap}.thumbs figure{margin:0;display:flex;flex-direction:column;align-items:center;gap:6px;font-size:19px;color:var(--muted)}
.big-quote{font-size:30px;line-height:1.3;font-weight:600;letter-spacing:-.5px;color:var(--ink)}.big-quote .by{display:block;font-size:22px;font-weight:400;color:var(--muted);margin-top:14px}
.algo{font-size:24px;line-height:1.5;border:1px solid var(--line);border-radius:10px;padding:16px 22px;background:#FFFFFF}.algo h4{margin:0 0 6px;font-size:22px;color:var(--p)}.algo ol{margin:0;padding-left:28px}
.tag{display:inline-block;font-size:15px;font-weight:650;letter-spacing:.6px;border-radius:999px;padding:3px 10px;border:1px solid currentColor;margin-right:6px;vertical-align:middle}
.illus img{width:100%;border-radius:10px;display:block}.illus figcaption{font-size:17px}
.qa{display:grid;grid-template-columns:auto 1fr;gap:10px 18px;font-size:24px;line-height:1.35}.qa b{color:var(--q)}
.refs{column-count:2;column-gap:36px;font-size:14.5px;line-height:1.3}.refs p{margin:0 0 6px;break-inside:avoid}
.compare td,.compare th{font-size:21px}
`;

export function mount(stage,ctx,d){
 const make={hookfilm,explode,film,pixhist,river,shortcut,inkpaper,reverse,playground,x0film,sampler2d,trajectories,scorefield,langevin,paths,ddimsteps,schedules,guidance,copy,cfgvec,'calc-forward':calcForward,'calc-reverse':calcReverse};
 for(const el of stage.querySelectorAll('[data-widget]')){
  const f=make[el.dataset.widget];if(!f)throw new Error('Unknown widget '+el.dataset.widget);f(el,ctx,d);
 }
}
