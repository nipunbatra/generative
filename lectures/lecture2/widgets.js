// Live figures for Lecture 2, and the small SVG chart kit and digit renderer that the static slides share.
// Every number, curve and image comes from data/science.json, data/digits.json or data/nets.json,
// which scripts/science.py computes from the data and the trained networks.
// Print and PDF show static states; presentation mode ties each animation to a build.
import {palette,esc} from '../shared/author.js';

export const MUTED='#68717B',AX='#AAB3BC',LINE='#E1E6EB';
export const TINT={p:'#0F766E24',q:'#8B2CDE24',loss:'#BE123C2E',data:'#245EDB24',noise:'#AA4E0824',grey:'#E5E7EB'};
/** One colour per digit class, 0–9, used by every scatter of codes. */
export const CLASS=['#245EDB','#AA4E08','#147737','#BE123C','#8B2CDE','#8C564B','#D6489B','#4B5563','#9A8A00','#0F9BAE'];
/** The two bumps of the eruption mixture: short (orange) and long (blue). */
export const BUMP=[palette.noise,palette.data];
export const lin=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
/** Fixed decimals, a true minus sign, and no "−0.00". */
export const num=(v,n=2)=>{const s=Number(v).toFixed(n);return (/^-0(\.0*)?$/.test(s)?s.slice(1):s).replace('-','−');};
export const comma=v=>Math.round(v).toLocaleString('en').replace('-','−');
export const normalPdf=(x,mu,sd)=>Math.exp(-.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
export function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export const gauss=r=>{let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());};
let uid=0;

// ---------------------------------------------------------------- chart kit (as in Lecture 1)
/** One plotting panel, optionally offset inside a larger SVG. Data outside the axes is clipped. */
export function panel(o){
 const ox=o.ox??0,oy=o.oy??0,W=o.W??600,H=o.H??320,L=o.L??64,R=o.R??14,T=o.T??56,B=o.B??56,fs=o.fs??20;
 const {xMin,xMax}=o,yMin=o.yMin??0,yMax=o.yMax;
 const x=v=>ox+L+(v-xMin)/(xMax-xMin)*(W-L-R),y=v=>oy+H-B-(v-yMin)/(yMax-yMin)*(H-T-B);
 const id=`w2clip-${++uid}`,tick=v=>String(+Number(v).toFixed(2)).replace('-','−');
 const base=yMin<0&&yMax>0&&!o.xAxisAtBottom?y(0):oy+H-B,tickY=oy+H-B+24;
 let axes=`<defs><clipPath id="${id}"><rect x="${ox+L}" y="${oy+T-10}" width="${W-L-R}" height="${H-B-T+10}"/></clipPath></defs>`;
 if(o.xAxis!==false)axes+=`<line x1="${ox+L}" y1="${base}" x2="${ox+W-R}" y2="${base}" stroke="${AX}"/>`;
 if(o.yAxis!==false)axes+=`<line x1="${ox+L}" y1="${oy+T}" x2="${ox+L}" y2="${oy+H-B}" stroke="${AX}"/>`;
 for(const v of o.ticksX||[])axes+=`<text x="${x(v).toFixed(1)}" y="${tickY.toFixed(1)}" text-anchor="middle" fill="${MUTED}" font-size="${fs}">${(o.fmtX||tick)(v)}</text>`;
 for(const v of o.ticksY||[])axes+=`<text x="${ox+L-9}" y="${(y(v)+7).toFixed(1)}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${(o.fmtY||tick)(v)}</text>`;
 if(o.xLabel)axes+=`<text x="${ox+W-R}" y="${oy+H-9}" text-anchor="end" fill="${MUTED}" font-size="${fs}">${o.xLabel}</text>`;
 if(o.yLabel)axes+=`<text x="${ox+L}" y="${oy+T-16}" fill="${MUTED}" font-size="${fs}">${o.yLabel}</text>`;
 const pts=(xs,ys)=>{let s='';for(let i=0;i<xs.length;i++){const v=y(ys[i]);s+=`${x(xs[i]).toFixed(1)},${(Number.isFinite(v)?Math.max(oy-600,Math.min(oy+H+600,v)):oy+H-B).toFixed(1)} `;}return s;};
 return {x,y,ox,oy,W,H,L,R,T,B,axes,clip:`clip-path="url(#${id})"`,
  line:(xs,ys,color,extra='')=>`<polyline clip-path="url(#${id})" points="${pts(xs,ys)}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" ${extra}/>`,
  area:(xs,ys,fill,at=Math.max(yMin,0))=>`<polygon clip-path="url(#${id})" points="${x(xs[0]).toFixed(1)},${y(at).toFixed(1)} ${pts(xs,ys)}${x(xs.at(-1)).toFixed(1)},${y(at).toFixed(1)}" fill="${fill}"/>`,
  dot:(vx,vy,color,r=6,extra='')=>`<circle cx="${x(vx).toFixed(1)}" cy="${y(vy).toFixed(1)}" r="${r}" fill="${color}" stroke="${palette.paper}" stroke-width="2" ${extra}/>`,
  bars:(edges,heights,fill,extra='')=>heights.map((h,i)=>h>0?`<rect x="${(x(edges[i])+1).toFixed(1)}" y="${y(h).toFixed(1)}" width="${Math.max(1,x(edges[i+1])-x(edges[i])-2).toFixed(1)}" height="${(y(yMin)-y(h)).toFixed(1)}" fill="${fill}" ${extra}/>`:'').join(''),
 };
}
/** Text in an SVG; content must already be safe (no markup from data). */
export const label=(tx,ty,s,{fill=MUTED,size=22,anchor='start',weight='',extra=''}={})=>`<text x="${(+tx).toFixed(1)}" y="${(+ty).toFixed(1)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}"${weight?` font-weight="${weight}"`:''} ${extra}>${s}</text>`;
export const svg=(W,H,inner,alt)=>`<svg class="plot wsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg>`;
export const fig=(W,H,inner,alt)=>`<div class="diagram-scroll">${svg(W,H,inner,alt)}</div>`;
export const widget=(name,alt,attrs={})=>`<div class="widget" data-widget="${name}" role="group" aria-label="${esc(alt)}" ${Object.entries(attrs).map(([k,v])=>`data-${k}="${esc(v)}"`).join(' ')}></div>`;
export const legend=(x0,y0,items,size=21)=>{let x=x0,s='';for(const [text,color,dashed,kind] of items){s+=kind==='box'?`<rect x="${x}" y="${y0-17}" width="24" height="16" fill="${color}" fill-opacity=".45"/>`:kind==='dot'?`<circle cx="${x+12}" cy="${y0-7}" r="6" fill="${color}"/>`:`<line x1="${x}" y1="${y0-7}" x2="${x+26}" y2="${y0-7}" stroke="${color}" stroke-width="3"${dashed?' stroke-dasharray="7 4"':''}/>`;s+=label(x+34,y0,text,{fill:color,size});x+=56+text.replace(/<[^>]+>/g,'').length*size*.52;}return s;};
/** An ellipse at k standard deviations for a covariance [a, b, c] = [[a, b], [b, c]] in data units; sx, sy: pixels per unit. */
export function covEllipse(cx,cy,[a,b,c],k,sx,sy,color,extra=''){
 // eigen-decomposition in pixel space: scale the covariance by the axis scales first (y is flipped)
 const A=a*sx*sx,B=-b*sx*sy,C=c*sy*sy,h=Math.sqrt(((A-C)/2)**2+B*B),l1=(A+C)/2+h,l2=Math.max((A+C)/2-h,1e-9),ang=.5*Math.atan2(2*B,A-C)*180/Math.PI;
 return `<ellipse cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" rx="${(k*Math.sqrt(l1)).toFixed(1)}" ry="${(k*Math.sqrt(l2)).toFixed(1)}" transform="rotate(${ang.toFixed(2)} ${cx.toFixed(1)} ${cy.toFixed(1)})" fill="none" stroke="${color}" stroke-width="3" ${extra}/>`;
}

const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * A recorded run, replayed frame by frame. Presentation: frame 0 until the play build,
 * which plays it (arriving there directly shows the final frame). Reading: final frame, with controls.
 * Print: frame 0 before the play build and the final frame from it, as static states.
 */
function animated(el,ctx,{count,render,step,alt,playBuild=2,duration=4200,extra=''}){
 const last=count-1;
 if(ctx.printing){el.innerHTML=`<div class="state-slot"><div data-only="1-${playBuild-1}">${render(0)}</div><div data-only="${playBuild}-99">${render(last)}</div></div>`;return {rerender(){}};}
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

/** A figure redrawn from buttons and sliders. Print shows the starting state without controls. */
function interactive(el,ctx,{controls,render}){
 if(ctx.printing){el.innerHTML=render();return null;}
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls">${controls}</div>`;
 const box=el.querySelector('.w-fig'),draw=()=>{box.innerHTML=render();};
 for(const input of el.querySelectorAll('input[type=range]'))input.addEventListener('change',()=>input.blur());
 draw();return draw;
}
const pressed=(el,sel,btn)=>el.querySelectorAll(sel).forEach(x=>x.setAttribute('aria-pressed',String(x===btn)));

// ---------------------------------------------------------------- digits
let DATA=null,DIGITS=null,NETS=null;
/** Hand the module its data once: science.json, digits.json and nets.json. */
export function init(d,digits,nets){DATA=d;DIGITS=digits;NETS=nets;}
const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const setCache=new Map();
/** An image set from science.json: {n, px (Uint8Array, 784 per image), signed, scale}. */
export function imgs(key){
 if(setCache.has(key))return setCache.get(key);
 const rec=key==='display'?DIGITS:DATA.img[key];if(!rec)throw new Error('Unknown image set '+key);
 const out={n:rec.n,px:b64(rec.data),signed:!!rec.signed,scale:rec.scale||1};setCache.set(key,out);return out;
}
const PAPER=[250,250,251],INKC=[20,23,31],TEAL=[15,118,110],RED=[190,18,60];
const mix=(a,b,t)=>[0,1,2].map(i=>Math.round(a[i]+(b[i]-a[i])*t));
/** Paint 784 values in [0, 1] (or signed, 0.5 = zero) into a 28 × 28 data URL: ink on paper. */
export function digitURL(values,offset=0,signed=false){
 const bytes=values instanceof Uint8Array;
 const c=document.createElement('canvas');c.width=c.height=28;const g=c.getContext('2d'),im=g.createImageData(28,28);
 for(let i=0;i<784;i++){
  const v=bytes?values[offset+i]/255:values[offset+i];let col;
  if(signed){const t=v*2-1;col=t>=0?mix(PAPER,TEAL,Math.min(1,t)):mix(PAPER,RED,Math.min(1,-t));}else col=mix(PAPER,INKC,Math.max(0,Math.min(1,v)));
  im.data.set([...col,255],i*4);
 }
 g.putImageData(im,0,0);return c.toDataURL();
}
const urlCache=new Map();
export function digitSrc(key,i){const k=key+'#'+i;if(!urlCache.has(k)){const s=imgs(key);urlCache.set(k,digitURL(s.px,i*784,s.signed));}return urlCache.get(k);}
/** One digit as an <img>, size in px. */
export const digit=(key,i,size=56,extra='')=>`<img class="dg" src="${digitSrc(key,i)}" width="${size}" height="${size}" alt="" ${extra}>`;
/**
 * A grid of digits from one image set. idx: which images (default all); cols; size; labels: one caption per image;
 * marks: per-image outline colour (or null).
 */
export function digits(key,{idx,cols,size=56,gap=6,labels,marks,title}={}){
 const s=imgs(key),ids=idx||Array.from({length:s.n},(_,i)=>i),c=cols||ids.length;
 const cells=ids.map((i,j)=>`<figure class="dg-cell">${digit(key,i,size,marks?.[j]?`style="outline:3px solid ${marks[j]};outline-offset:1px"`:'')}${labels?`<figcaption>${labels[j]}</figcaption>`:''}</figure>`).join('');
 return `<div class="dgrid" style="grid-template-columns:repeat(${c},${size}px);gap:${gap}px">${title?`<div class="dg-title" style="grid-column:1/-1">${title}</div>`:''}${cells}</div>`;
}
/** A raw 28 × 28 array (values in [0, 1]) as an <img>. */
export const rawDigit=(values,size=56,extra='')=>`<img class="dg" src="${digitURL(values)}" width="${size}" height="${size}" alt="" ${extra}>`;

// ---------------------------------------------------------------- the networks, in the browser
function halfToFloat(h){const s=(h&0x8000)?-1:1,e=(h>>10)&0x1f,f=h&0x3ff;return e===0?s*Math.pow(2,-14)*(f/1024):e===31?(f?NaN:s*Infinity):s*Math.pow(2,e-15)*(1+f/1024);}
function f16(rec){const b=b64(rec.data),u=new Uint16Array(b.buffer,b.byteOffset,b.byteLength/2),o=new Float32Array(u.length);for(let i=0;i<u.length;i++)o[i]=halfToFloat(u[i]);return o;}
function f32(rec){const b=b64(rec.data);return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);}
const netCache=new Map();
function net(name){
 if(netCache.has(name))return netCache.get(name);
 const n=NETS[name],[W1,b1,W2,b2]=n.dec.map(f16),[E1,c1,E2,c2]=n.enc.map(f16),H=b1.length;
 const out={
  decode(z){const h=new Float32Array(H);for(let j=0;j<H;j++){let s=b1[j];for(let i=0;i<z.length;i++)s+=W1[j*z.length+i]*z[i];h[j]=s>0?s:0;}
   const x=new Float32Array(784);for(let k=0;k<784;k++){let s=b2[k];const r=k*H;for(let j=0;j<H;j++)s+=W2[r+j]*h[j];x[k]=1/(1+Math.exp(-s));}return x;},
  encode(px){const h=new Float32Array(H);for(let j=0;j<H;j++){let s=c1[j];const r=j*784;for(let i=0;i<784;i++)s+=E1[r+i]*px[i];h[j]=s>0?s:0;}
   return [0,1].map(k=>{let s=c2[k];for(let j=0;j<H;j++)s+=E2[k*H+j]*h[j];return s;});},
 };
 netCache.set(name,out);return out;
}
/** The display digit i as values in [0, 1]. */
const displayPx=i=>{const s=imgs('display');return Float32Array.from(s.px.subarray(i*784,(i+1)*784),v=>v/255);};

// ---------------------------------------------------------------- Part 2: eruptions of Old Faithful
const WAIT_BINS=lin(40,100,31);
function histogram(xs,edges){const h=new Array(edges.length-1).fill(0),w=edges[1]-edges[0];for(const x of xs){const b=Math.floor((x-edges[0])/w);if(b>=0&&b<h.length)h[b]++;}return h;}
/** Waiting times: data histogram, optional model curve, optional draws (coloured by bump). Returns SVG inner. */
export function waitPanel(o){
 const F=DATA.faithful,P=panel({ox:o.ox||0,oy:o.oy||0,W:o.W||700,H:o.H||330,L:22,R:28,T:o.T??50,B:52,xMin:40,xMax:100,yMax:o.yMax||.062,ticksX:[40,50,60,70,80,90,100],yAxis:false,xLabel:o.xLabel??'minutes until the next eruption'});
 const bw=WAIT_BINS[1]-WAIT_BINS[0],xs=lin(40,100,241);let s=P.axes;
 if(o.data!==false){const h=histogram(F.waiting,WAIT_BINS).map(c=>c/(F.n*bw));s+=P.bars(WAIT_BINS,h,o.dataFill||'#C9D1D9',o.dataExtra||'');}
 if(o.draws&&o.draws.length){
  const n=o.drawsN||o.draws.length,h0=new Array(30).fill(0),h1=new Array(30).fill(0);
  for(const [k,x] of o.draws){const b=Math.floor((x-40)/bw);if(b>=0&&b<30)(k?h1:h0)[b]++;}
  h0.forEach((c,b)=>{const a=c/(n*bw),bb=h1[b]/(n*bw),x0=P.x(WAIT_BINS[b])+1,x1=P.x(WAIT_BINS[b+1])-1;
   if(c)s+=`<rect x="${x0.toFixed(1)}" y="${P.y(a).toFixed(1)}" width="${(x1-x0).toFixed(1)}" height="${(P.y(0)-P.y(a)).toFixed(1)}" fill="${o.oneColour||BUMP[0]}" fill-opacity=".55"/>`;
   if(bb)s+=`<rect x="${x0.toFixed(1)}" y="${P.y(a+bb).toFixed(1)}" width="${(x1-x0).toFixed(1)}" height="${(P.y(a)-P.y(a+bb)).toFixed(1)}" fill="${o.oneColour||BUMP[1]}" fill-opacity=".55"/>`;});
 }
 const g=F.gauss,mx=o.mix||F.mix;
 if(o.gauss)s+=P.line(xs,xs.map(x=>normalPdf(x,g.mu,g.sd)),palette.q);
 if(o.mixture){
  const c0=xs.map(x=>mx.pi[0]*normalPdf(x,mx.mu[0],mx.sd[0])),c1=xs.map(x=>mx.pi[1]*normalPdf(x,mx.mu[1],mx.sd[1]));
  if(o.components!==false)s+=P.line(xs,c0,BUMP[0],'stroke-dasharray="7 5" stroke-width="2.5"')+P.line(xs,c1,BUMP[1],'stroke-dasharray="7 5" stroke-width="2.5"');
  s+=P.line(xs,xs.map((_,i)=>c0[i]+c1[i]),palette.p);
 }
 if(o.marks)for(const [v,color,text] of o.marks)s+=`<line x1="${P.x(v).toFixed(1)}" y1="${P.y(0)}" x2="${P.x(v).toFixed(1)}" y2="${(P.y(0)-28).toFixed(1)}" stroke="${color}" stroke-width="4"/>`+(text?label(P.x(v),P.y(0)-36,text,{fill:color,size:19,anchor:'middle',weight:600}):'');
 return {s,P};
}
/** Draw eruptions from one Gaussian or from the two-bump mixture; histogram of the draws over the data. */
function faithSample(el,ctx){
 const F=DATA.faithful,g=F.gauss,mx=F.mix,model=el.dataset.model||'mix',W=1150,H=340;
 let rng,draws;
 const reset=()=>{rng=mulberry32(model==='mix'?41:42);draws=[];};
 const add=n=>{for(let i=0;i<n;i++){if(model==='mix'){const k=rng()<mx.pi[1]?1:0;draws.push([k,mx.mu[k]+mx.sd[k]*gauss(rng)]);}else draws.push([0,g.mu+g.sd*gauss(rng)]);}};
 const render=()=>{
  const last=draws.at(-1),counts=[0,0];for(const [k] of draws)counts[k]++;
  let s='';
  if(model==='mix'){
   s+=label(0,26,'step 1: pick a bump',{fill:palette.ink,size:22,weight:600});
   [0,1].forEach(k=>{const y=42+k*84,on=last&&last[0]===k;
    s+=`<rect x="0" y="${y}" width="360" height="70" rx="10" fill="${on?BUMP[k]+'22':palette.paper}" stroke="${BUMP[k]}" stroke-width="${on?3.5:2}"/>`+
     label(14,y+30,`${k?'long':'short'} wait: probability ${num(mx.pi[k],2)}`,{fill:BUMP[k],size:21,weight:600})+
     label(14,y+57,`N(${num(mx.mu[k],1)}, ${num(mx.sd[k],1)}²) · picked ${comma(counts[k])}×`,{fill:BUMP[k],size:19});});
   s+=label(0,236,'step 2: draw a time from that bump',{fill:palette.ink,size:22,weight:600});
   if(last)s+=label(0,272,`latest: ${last[0]?'long':'short'} bump, ${num(last[1],1)} minutes`,{fill:BUMP[last[0]],size:22});
  }else{
   s+=label(0,26,'one step: draw a time',{fill:palette.ink,size:22,weight:600});
   s+=`<rect x="0" y="42" width="360" height="70" rx="10" fill="${palette.paper}" stroke="${palette.q}" stroke-width="2"/>`+label(14,72,`N(${num(g.mu,1)}, ${num(g.sd,1)}²)`,{fill:palette.q,size:22,weight:600})+label(14,99,'the one fitted Gaussian',{fill:palette.q,size:19});
   if(last)s+=label(0,160,`latest: ${num(last[1],1)} minutes`,{fill:palette.q,size:22});
   const inGap=draws.filter(([,x])=>x>=62&&x<=72).length;
   if(draws.length)s+=label(0,200,`${comma(inGap)} of ${comma(draws.length)} draws land in 62–72 min`,{size:20})+label(0,228,`(the data: ${F.waiting.filter(x=>x>=62&&x<=72).length} of ${F.n})`,{size:20});
  }
  s+=label(0,312,draws.length?`${comma(draws.length)} draws so far`:'press +1 draw to start',{size:19});
  const few=draws.length<30,{s:ps,P}=waitPanel({ox:400,W:750,H,draws:few?[]:draws,gauss:model==='gauss',mixture:model==='mix',oneColour:model==='gauss'?palette.q:null,yMax:.07});
  if(few)s+=draws.map(([k,x],i)=>`<line x1="${P.x(x).toFixed(1)}" y1="${P.y(0)}" x2="${P.x(x).toFixed(1)}" y2="${(P.y(0)-(i===draws.length-1?44:26)).toFixed(1)}" stroke="${model==='mix'?BUMP[k]:palette.q}" stroke-width="${i===draws.length-1?5:3}" stroke-opacity="${i===draws.length-1?1:.6}"/>`).join('');
  s=ps+s+legend(416,30,[['the 272 real waits',`#9AA5B1`,false,'box'],['draws from the model',model==='mix'?BUMP[1]:palette.q,false,'box']],19);
  return svg(W,H,s,model==='mix'?'Draws from the two-bump mixture: a bump is picked, then a time is drawn from it; the histogram of draws approaches the data.':'Draws from one Gaussian: the draws fill the gap between the two clumps of real data.');
 };
 reset();
 if(ctx.printing){add(F.n);el.innerHTML=render();return;}
 if(false)add(1);
 const controls=[1,10,100,272].map(n=>`<button type="button" data-add="${n}">+${n} draw${n>1?'s':''}</button>`).join('')+`<button type="button" data-reset>Reset</button>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-add]').forEach(b=>b.addEventListener('click',()=>{if(draws.length<20000)add(Number(b.dataset.add));draw();}));
 el.querySelector('[data-reset]').addEventListener('click',()=>{reset();draw();});
 // Presentation: advancing to build 2 fills in 272 draws, so the story line about them is on screen.
 const scene=el.closest('.scene');
 if(ctx.reading||!scene){add(F.n);draw();return;}
 new MutationObserver(()=>{const b=Number(scene.dataset.build||1);if(b>=2&&draws.length<F.n){add(F.n-draws.length);draw();}else if(b<2&&draws.length&&scene.classList.contains('current')){reset();draw();}}).observe(scene,{attributes:true,attributeFilter:['data-build','class']});
}
/** The two-bump mixture, fitted by gradient descent: replay the recorded run. */
function faithFit(el,ctx){
 const F=DATA.faithful,fr=F.mix.frames,W=1150,H=340;
 const render=i=>{
  const [step,pl,m0,m1,s0,s1,ll]=fr[i],mx={pi:[1-pl,pl],mu:[m0,m1],sd:[s0,s1]};
  const {s}=waitPanel({W:720,H,mixture:true,mix:mx,yMax:.07});
  const steps=fr.map(f=>f[0]),lls=fr.map(f=>f[6]),Q=panel({ox:770,W:380,H,L:70,R:28,T:62,B:56,xMin:0,xMax:300,yMin:-4.1,yMax:-3.75,ticksX:[0,100,200,300],ticksY:[-4.1,-4,-3.9,-3.8],xLabel:'step'});
  return svg(W,H,s+legend(16,30,[['short bump',BUMP[0],true],['long bump',BUMP[1],true],['mixture',palette.p]],19)+
   Q.axes+Q.line(steps.slice(0,i+1),lls.slice(0,i+1),palette.loss)+Q.dot(step,ll,palette.loss)+label(770+70,30,`average log-likelihood ${num(ll,3)}`,{fill:palette.loss,size:20,weight:600}),
   'A two-bump mixture fitted by gradient descent: the bumps move apart onto the two clumps, and the average log-likelihood rises.');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'Mixture fit',duration:5200});
}

// ---------------------------------------------------------------- Part 3: PCA
let PCA=null;
function pcaBasis(){if(!PCA){const M=DATA.mnist.pca;PCA={W:f32(M.basis),mean:f32(M.mean),z3:M.z3_100};}return PCA;}
/** Reconstruct our 3 from its first d PCA numbers, live. */
function pcaSlider(el,ctx){
 const M=DATA.mnist,B=pcaBasis(),x3=imgs('example');
 const st={d:Number(el.dataset.d||10)};
 // the squared error is measured before clipping to [0, 1], as in science.py; the picture is clipped
 const recon=d=>{const x=Float32Array.from(B.mean);if(d>=784)return Float32Array.from(x3.px,v=>v/255);for(let j=0;j<d;j++){const z=B.z3[j],r=j*784;for(let k=0;k<784;k++)x[k]+=z*B.W[r+k];}return x;};
 const render=()=>{
  const d=st.d,raw=recon(d),orig=Float32Array.from(x3.px,v=>v/255);let sse=0;for(let k=0;k<784;k++)sse+=(raw[k]-orig[k])**2;
  const xh=raw.map(v=>Math.max(0,Math.min(1,v)));
  const ve=d>=784?1:M.pca.var_explained[Math.min(d,200)-1];
  return `<div class="pca-live"><figure>${rawDigit(orig,200)}<figcaption>our 3: 784 numbers</figcaption></figure><div class="arrow">→</div><div class="pca-code">${d>=784?'all 784':d} number${d>1?'s':''}<span>z₁ = ${num(B.z3[0],2)}, z₂ = ${num(B.z3[1],2)}${d>2?', …':''}</span></div><div class="arrow">→</div><figure>${rawDigit(xh,200)}<figcaption>rebuilt from ${d>=784?'all 784':'the first '+d}</figcaption></figure><div class="pca-read"><div><b>${num(sse,1)}</b>squared error</div><div><b>${num(100*ve,1)}%</b>of the variance of all 60,000 digits kept</div></div></div>`;
 };
 if(ctx.printing){el.innerHTML=render();return;}
 const controls=`<label>numbers kept d<input type="range" min="1" max="100" step="1" value="${st.d}" data-k="d" aria-label="numbers kept"><span class="w-readout" data-o="d">${st.d}</span></label>`+[1,2,5,10,20,50,100].map(v=>`<button type="button" data-set="${v}">${v}</button>`).join('')+`<button type="button" data-set="784">all 784</button>`;
 const draw=interactive(el,ctx,{controls,render});
 const sync=()=>{el.querySelector('[data-k]').value=String(Math.min(100,st.d));el.querySelector('[data-o]').textContent=st.d>=784?'784':String(st.d);};
 el.querySelector('[data-k]').addEventListener('input',e=>{st.d=Number(e.target.value);sync();draw();});
 el.querySelectorAll('[data-set]').forEach(b=>b.addEventListener('click',()=>{st.d=Number(b.dataset.set);sync();draw();}));
}
/** A mixture of K Gaussians over the first d PCA numbers: sample 16 codes, rebuild each with inverse PCA, live. */
function mogGen(el,ctx){
 const M=DATA.mnist,B=pcaBasis(),st={d:Number(el.dataset.d||10),K:Number(el.dataset.k||10)},rng=mulberry32(99);
 const dec=key=>{const g=M.gens[key];if(!g._w){g._w=f32(g.w);g._m=f32(g.means);g._L=f32(g.chol);}return g;};
 const sample=()=>{const g=dec(`d${st.d}k${st.K}`),d=st.d,out=[];
  for(let n=0;n<16;n++){let u=rng(),k=0;while(k<st.K-1&&(u-=g._w[k])>0)k++;
   const e=Array.from({length:d},()=>gauss(rng)),z=new Float32Array(d);
   for(let i=0;i<d;i++){let s=g._m[k*d+i];for(let j=0;j<=i;j++)s+=g._L[k*d*d+i*d+j]*e[j];z[i]=s;}
   const x=Float32Array.from(B.mean);for(let j=0;j<d;j++){const r=j*784;for(let p=0;p<784;p++)x[p]+=z[j]*B.W[r+p];}
   out.push(x.map(v=>Math.max(0,Math.min(1,v))));}
  return out;};
 let current=null;
 const render=()=>{
  const key=`d${st.d}k${st.K}`,g=M.gens[key];
  const cells=current?current.map(x=>rawDigit(x,62)).join(''):digits('gen_'+key,{cols:8,size:62});
  return `<div class="mog-live"><div class="dgrid" style="grid-template-columns:repeat(8,62px);gap:6px">${current?cells:''}</div>${current?'':cells}<div class="mog-read">d = ${st.d} PCA numbers · K = ${st.K} bump${st.K>1?'s':''} · ${comma(g.params)} fitted numbers · held-out average log-likelihood ${num(g.test_ll,1)}</div></div>`;
 };
 if(ctx.printing){el.innerHTML=render();return;}
 const controls=`<span>d</span>${[2,10,20,50].map(v=>`<button type="button" data-d="${v}" aria-pressed="${v===st.d}">${v}</button>`).join('')}<span>K</span>${[1,5,10,20].map(v=>`<button type="button" data-kk="${v}" aria-pressed="${v===st.K}">${v}</button>`).join('')}<button type="button" data-sample>Sample 16 new digits</button>`;
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-d]').forEach(b=>b.addEventListener('click',()=>{st.d=Number(b.dataset.d);pressed(el,'[data-d]',b);current=null;draw();}));
 el.querySelectorAll('[data-kk]').forEach(b=>b.addEventListener('click',()=>{st.K=Number(b.dataset.kk);pressed(el,'[data-kk]',b);current=null;draw();}));
 el.querySelector('[data-sample]').addEventListener('click',()=>{current=sample();draw();});
}

// ---------------------------------------------------------------- scatters of codes, with hover
/** Codes of the 1,000 display digits for a source: 'pca', 'ae2', 'vae2' or 'beta:<b>'. */
export function codesOf(src){
 if(src==='pca')return DATA.mnist.pca.display_codes2;
 if(src==='ae2')return DATA.ae.ae2_codes;
 if(src==='vae2')return DATA.vae.codes;
 if(src.startsWith('beta:'))return DATA.vae.beta[src.slice(5)].codes;
 throw new Error('Unknown code source '+src);
}
/** The scatter itself: SVG inner and the panel. lim: [xmin, xmax, ymin, ymax]. */
export function scatterSVG(src,{ox=0,oy=0,W=560,H=420,lim,prior=0,priorColor=palette.loss,r=4,opacity=.55,highlight=null,ticks,box=null,mog=null,points=true,contours=false}={}){
 const C=codesOf(src),labels=DATA.mnist.display_labels;
 const [x0,x1,y0,y1]=lim,P=panel({ox,oy,W,H,L:46,R:12,T:16,B:46,xMin:x0,xMax:x1,yMin:y0,yMax:y1,ticksX:ticks?.[0],ticksY:ticks?.[1],fs:17,xAxisAtBottom:true});
 let s=`<rect x="${ox+46}" y="${oy+16}" width="${W-58}" height="${H-62}" fill="#FFFFFF" stroke="${LINE}"/>`+P.axes;
 if(box)s+=`<rect ${P.clip} x="${P.x(box[0][0]).toFixed(1)}" y="${P.y(box[1][1]).toFixed(1)}" width="${(P.x(box[1][0])-P.x(box[0][0])).toFixed(1)}" height="${(P.y(box[0][1])-P.y(box[1][1])).toFixed(1)}" fill="none" stroke="${MUTED}" stroke-dasharray="6 5"/>`;
 if(points)s+=`<g ${P.clip}>${C.map(([a,b],i)=>`<circle data-i="${i}" cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="${r}" fill="${CLASS[labels[i]]}" fill-opacity="${opacity}"/>`).join('')}</g>`;
 if(mog)mog.means.forEach((m,k)=>{const c=mog.covs[k];s+=covEllipse(P.x(m[0]),P.y(m[1]),[c[0][0],c[0][1],c[1][1]],2,(P.x(1)-P.x(0)),(P.y(0)-P.y(1)),palette.ink,`${P.clip} stroke-width="2" stroke-opacity=".75"`);});
 if(prior){const pts=DATA.gen.prior_points.slice(0,prior);s+=`<g ${P.clip}>${pts.map(([a,b],i)=>`<circle data-p="${i}" cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="3.4" fill="${priorColor}" stroke="#fff" stroke-width=".8"/>`).join('')}</g>`;}
 if(contours){const sx=P.x(1)-P.x(0),sy=P.y(0)-P.y(1);for(const k of [1,2,3])s+=`<ellipse ${P.clip} cx="${P.x(0).toFixed(1)}" cy="${P.y(0).toFixed(1)}" rx="${(k*sx).toFixed(1)}" ry="${(k*sy).toFixed(1)}" fill="none" stroke="${palette.ink}" stroke-width="${k===1?2.5:1.8}" stroke-dasharray="${k===1?'':'6 5'}" pointer-events="none"/>`+label(P.x(0)+k*sx*.71+4,P.y(0)-k*sy*.71-4,String(k),{fill:palette.ink,size:15,weight:600,extra:'pointer-events="none"'});}
 if(highlight!==null){const [a,b]=C[highlight];s+=`<circle cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="11" fill="none" stroke="${palette.ink}" stroke-width="3"/>`;}
 return {s,P};
}
/** The SVG width that gives equal units on both axes, for a scatter of height H (plot margins as in scatterSVG). */
export const squareW=(lim,H,max=600)=>Math.min(max,Math.round((H-62)*(lim[1]-lim[0])/(lim[3]-lim[2])+58));
export const classLegend=(x,y,size=18,step=36)=>Array.from({length:10},(_,c)=>`<circle cx="${x+c*step+7}" cy="${y-6}" r="7" fill="${CLASS[c]}"/>`+label(x+c*step+18,y,String(c),{fill:CLASS[c],size,weight:600})).join('');
/** Hover a point to see its digit. Attributes: src, lim (comma list), prior (count), title. */
function scatter(el,ctx){
 const src=el.dataset.src,lim=el.dataset.lim.split(',').map(Number),prior=Number(el.dataset.prior||0),W=1150,H=Number(el.dataset.h||400);
 const ticks=el.dataset.ticks?el.dataset.ticks.split('|').map(t=>t.split(',').map(Number)):undefined;
 const contours=el.dataset.contours==='1',decodeNet=el.dataset.decode||null;
 const SW=squareW(lim,H),X0=SW+40,{s}=scatterSVG(src,{W:SW,H,lim,prior,ticks,contours});
 const title=el.dataset.title||'',side=el.dataset.side||'';
 let base=s+classLegend(X0,36)+label(X0,80,title,{fill:palette.ink,size:22,weight:600});
 (side?side.split('|'):[]).forEach((t,i)=>{base+=label(X0,112+i*26,t,{size:19});});
 if(ctx.printing){el.innerHTML=svg(W,H,base,'Codes of 1,000 test digits, coloured by digit.');return;}
 el.innerHTML=`<div class="w-fig diagram-scroll" style="position:relative">${svg(W,H,base,'Codes of 1,000 test digits, coloured by digit; hover a point to see its digit.')}<div class="hover-card" hidden></div></div>`;
 const card=el.querySelector('.hover-card'),svgEl=el.querySelector('svg');let ring=null;
 const show=html=>{card.hidden=false;card.innerHTML=html;};
 svgEl.addEventListener('mousemove',e=>{
  const t=e.target.closest('circle[data-i],circle[data-p]');
  if(!t){card.hidden=true;ring?.remove();ring=null;return;}
  if(t.dataset.p!==undefined){
   const z=DATA.gen.prior_points[Number(t.dataset.p)];
   show(`<div style="display:flex;gap:10px;align-items:end">${decodeNet?`<figure class="dg-cell"><img class="dg" src="${digitURL(net(decodeNet).decode(z))}" width="130" height="130" alt=""><figcaption>decoder output</figcaption></figure>`:''}</div><div>a random code from N(0, I)<br>z = (${num(z[0],2)}, ${num(z[1],2)})</div>`);
  }else{
   const i=Number(t.dataset.i),z=codesOf(src)[i],lab=DATA.mnist.display_labels[i];
   show(`<div style="display:flex;gap:10px;align-items:end"><figure class="dg-cell">${digit('display',i,decodeNet?100:150)}<figcaption>the digit</figcaption></figure>${decodeNet?`<figure class="dg-cell"><img class="dg" src="${digitURL(net(decodeNet).decode(z))}" width="100" height="100" alt=""><figcaption>decoded from its code</figcaption></figure>`:''}</div><div>a <b style="color:${CLASS[lab]}">${lab}</b>, code (${num(z[0],2)}, ${num(z[1],2)})</div>`);
  }
  ring?.remove();ring=document.createElementNS('http://www.w3.org/2000/svg','circle');
  for(const [k,v] of Object.entries({cx:t.getAttribute('cx'),cy:t.getAttribute('cy'),r:9,fill:'none',stroke:palette.ink,'stroke-width':3}))ring.setAttribute(k,v);
  svgEl.append(ring);
 });
 svgEl.addEventListener('mouseleave',()=>{card.hidden=true;ring?.remove();ring=null;});
}

// ---------------------------------------------------------------- the code explorer: three ways to choose a code
/**
 * Choose a 2-number code and see what the trained decoder draws, computed live in the browser.
 * Attributes: net ('ae2' | 'vae2'), lim (comma list), both ('1' to offer a switch between the networks).
 */
function explorer(el,ctx){
 const W=1150,H=310,both=el.dataset.both==='1';
 const LIM={ae2:el.dataset.limae?.split(',').map(Number)||el.dataset.lim.split(',').map(Number),vae2:el.dataset.limvae?.split(',').map(Number)||[-3.5,3.5,-3.5,3.5]};
 const st={net:el.dataset.net||'ae2',z:null,how:'',src:null,prior:false};
 const rng=mulberry32(5);
 const nice=(lo,hi)=>{const step=hi-lo>20?10:hi-lo>8?2:1,out=[];for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step)out.push(v);return out;};
 const scene=()=>{const lim=LIM[st.net];return scatterSVG(st.net,{W:squareW(lim,H),H,lim,prior:st.prior?300:0,opacity:.4,r:3.6,highlight:st.src,contours:true,ticks:[nice(lim[0],lim[1]),nice(lim[2],lim[3])]});};
 const render=()=>{
  const {s,P}=scene();let out=s;const X0=squareW(LIM[st.net],H)+40;
  if(st.z){const [a,b]=st.z,cx=P.x(a),cy=P.y(b);out+=`<g><line x1="${cx-16}" y1="${cy}" x2="${cx+16}" y2="${cy}" stroke="${palette.ink}" stroke-width="3"/><line x1="${cx}" y1="${cy-16}" x2="${cx}" y2="${cy+16}" stroke="${palette.ink}" stroke-width="3"/><circle cx="${cx}" cy="${cy}" r="6" fill="${palette.q}" stroke="#fff" stroke-width="2"/></g>`;}
  out+=classLegend(X0,36,17,34);
  const name=st.net==='ae2'?'autoencoder':'VAE';
  let right=label(X0,80,`decoder of the ${name}, run in your browser`,{fill:palette.ink,size:20,weight:600});
  if(st.z){
   const x=net(st.net).decode(st.z);
   right+=`<image href="${digitURL(x)}" x="${X0}" y="90" width="160" height="160" style="image-rendering:pixelated"/>`+`<rect x="${X0}" y="90" width="160" height="160" fill="none" stroke="${LINE}"/>`;
   right+=label(X0,278,`z = (${num(st.z[0],2)}, ${num(st.z[1],2)})`,{fill:palette.q,size:22,weight:600})+label(X0,302,st.how,{size:19});
   if(st.src!==null)right+=`<image href="${digitSrc('display',st.src)}" x="${X0+190}" y="90" width="100" height="100" style="image-rendering:pixelated"/><rect x="${X0+190}" y="90" width="100" height="100" fill="none" stroke="${LINE}"/>`+label(X0+190,214,'the digit we encoded',{size:17});
  }else right+=label(X0,130,'click anywhere on the plane, or use a button',{size:20});
  if(st.prior)right+=label(X0+190,250,'red dots: 300 draws',{fill:palette.loss,size:17})+label(X0+190,272,'from N(0, I)',{fill:palette.loss,size:17});
  right+=label(X0+330,140,'circles: N(0, I)',{size:17})+label(X0+330,162,'at radius 1, 2, 3',{size:17});
  return svg(W,H,out+right,`Codes of 1,000 test digits for the ${name}; a chosen code and the image its decoder produces.`);
 };
 const set=(z,how,src=null)=>{st.z=z;st.how=how;st.src=src;};
 if(ctx.printing){const C=codesOf(st.net);set(C[0],'the code of our 3, decoded',0);el.innerHTML=render();return;}
 set(codesOf(st.net)[0],'the code of our 3',0);
 const controls=`<button type="button" data-a="prior">Random code from N(0, I)</button><button type="button" data-a="encode">Encode a random test digit</button><button type="button" data-a="dots" aria-pressed="false">Show 300 N(0, I) draws</button>`+(both?`<button type="button" data-net="ae2" aria-pressed="${st.net==='ae2'}">autoencoder</button><button type="button" data-net="vae2" aria-pressed="${st.net==='vae2'}">VAE</button>`:'');
 const draw=interactive(el,ctx,{controls,render});
 const box=el.querySelector('.w-fig');
 box.addEventListener('click',e=>{
  const svgEl=box.querySelector('svg'),pt=svgEl.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(svgEl.getScreenCTM().inverse());
  const lim=LIM[st.net],L=46,T=16,PW=squareW(lim,H)-46-12,PH=H-16-46;
  if(p.x<L||p.x>L+PW||p.y<T||p.y>T+PH)return;
  set([lim[0]+(p.x-L)/PW*(lim[1]-lim[0]),lim[3]-(p.y-T)/PH*(lim[3]-lim[2])],'a point you clicked');draw();
 });
 box.style.cursor='crosshair';
 el.querySelector('[data-a=prior]').addEventListener('click',()=>{set([gauss(rng),gauss(rng)],'drawn from N(0, I)');draw();});
 el.querySelector('[data-a=encode]').addEventListener('click',()=>{const i=Math.floor(rng()*1000);set(codesOf(st.net)[i],`the code of a test ${DATA.mnist.display_labels[i]}`,i);draw();});
 el.querySelector('[data-a=dots]').addEventListener('click',e=>{st.prior=!st.prior;e.target.setAttribute('aria-pressed',String(st.prior));draw();});
 el.querySelectorAll('[data-net]').forEach(b=>b.addEventListener('click',()=>{st.net=b.dataset.net;pressed(el,'[data-net]',b);set(codesOf(st.net)[0],'the code of our 3',0);draw();}));
}

// ---------------------------------------------------------------- Part 4: training an autoencoder
function aeTrain(el,ctx){
 const A=DATA.ae,steps=A.ae2_frames_steps,n=steps.length,spe=A.steps_per_epoch,W=1150,H=360,C=A.ae2_step_curve,lx=v=>Math.log10(v+1);
 const SZ=74,GAP=8;
 const render=i=>{
  const stp=steps[i],ep=stp/spe,sse=A.ae2_frames_sse[i],per=A.ae2_frames_digit_sse[i];
  let s=label(0,22,'8 test digits',{fill:palette.ink,size:20,weight:600});
  for(let j=0;j<8;j++)s+=`<image href="${digitSrc('watch',j)}" x="${j*(SZ+GAP)}" y="32" width="${SZ}" height="${SZ}" style="image-rendering:pixelated"/>`;
  s+=label(0,140,stp===0?'what the untrained network outputs':`what it outputs after ${comma(stp)} step${stp===1?'':'s'}${stp>=spe?` (${num(ep,0)} epoch${ep>=2?'s':''})`:''}`,{fill:palette.q,size:20,weight:600});
  for(let j=0;j<8;j++)s+=`<image href="${digitSrc('ae2_train_frames',i*8+j)}" x="${j*(SZ+GAP)}" y="152" width="${SZ}" height="${SZ}" style="image-rendering:pixelated"/><rect x="${j*(SZ+GAP)}" y="152" width="${SZ}" height="${SZ}" fill="none" stroke="${palette.q}" stroke-opacity=".5"/>`+
   label(j*(SZ+GAP)+SZ/2,252,num(per[j],0),{fill:palette.loss,size:18,anchor:'middle',weight:600});
  s+=label(0,284,'squared error of each digit (summed over its 784 pixels)',{fill:palette.loss,size:17});
  s+=label(0,326,`average over the 10,000 test digits: ${num(sse,1)}`,{fill:palette.loss,size:22,weight:650});
  const P=panel({ox:690,W:460,H,L:62,R:16,T:44,B:56,xMin:0,xMax:lx(14070),yMin:0,yMax:200,ticksX:[0,1,2,3,4],fmtX:v=>comma(10**v),ticksY:[0,50,100,150,200],xLabel:'training step (log scale)',fs:16});
  const xs=C.step.map(lx);
  s+=P.axes+label(690+62,24,'test error while training',{fill:palette.loss,size:19,weight:600})+
   P.line(xs,C.test_sse,palette.loss+'44')+P.line(xs.filter((_,k)=>C.step[k]<=stp),C.test_sse.filter((_,k)=>C.step[k]<=stp),palette.loss)+P.dot(lx(stp),sse,palette.loss,7)+
   label(P.x(lx(stp))+(lx(stp)>3.4?-10:10),P.y(sse)-12,num(sse,1),{fill:palette.loss,size:18,weight:600,anchor:lx(stp)>3.4?'end':'start'});
  return svg(W,H,s,'Reconstructions of eight test digits as an autoencoder with a 2-number code trains, with each digit’s error and the test error so far.');
 };
 animated(el,ctx,{count:n,render,step:i=>`step ${comma(steps[i])}`,alt:'Autoencoder training',duration:7000});
}

// ---------------------------------------------------------------- Part 6: the VAE
/** The exact posterior p(z | our 3) on a grid, as an image layer: 'wide' covers [-4, 4]², 'zoom' a small window around it. */
const POST={};
function posteriorImage(view){
 if(POST[view])return POST[view];
 const g=DATA.vae.grid,v=view==='zoom'?g.zoom.post:g.post81,n=Math.round(Math.sqrt(v.length)),c=document.createElement('canvas');c.width=c.height=n;
 const x=c.getContext('2d'),im=x.createImageData(n,n);
 for(let r=0;r<n;r++)for(let k=0;k<n;k++){const t=Math.pow(v[r*n+k],.6),col=mix(PAPER,[13,148,136],t);im.data.set([...col,255],((n-1-r)*n+k)*4);}
 x.putImageData(im,0,0);POST[view]=c.toDataURL();return POST[view];
}
/** The posterior heat map with optional prior circles, the encoder's q and a variational q. Returns {s, P}. */
export function posteriorSVG({ox=0,oy=0,S=400,view='wide',q=null,enc=false,extra='',prior=true,title=''}={}){
 const g=DATA.vae.grid,z=g.zoom;
 const [x0,x1,y0,y1]=view==='zoom'?[z.cx-z.half,z.cx+z.half,z.cy-z.half,z.cy+z.half]:[-4,4,-4,4];
 const ticks=view==='zoom'?[[z.cx-z.half*.67,z.cx,z.cx+z.half*.67],[z.cy-z.half*.67,z.cy,z.cy+z.half*.67]].map(a=>a.map(v=>+v.toFixed(2))):[[-4,-2,0,2,4],[-4,-2,0,2,4]];
 const P=panel({ox,oy,W:S,H:S,L:52,R:10,T:title?34:10,B:40,xMin:x0,xMax:x1,yMin:y0,yMax:y1,ticksX:ticks[0],ticksY:ticks[1],fs:16,xAxisAtBottom:true});
 const sx=P.x(1)-P.x(0),sy=P.y(0)-P.y(1);
 let s=`<rect x="${ox+52}" y="${oy+(title?34:10)}" width="${S-62}" height="${S-(title?74:50)}" fill="#fff" stroke="${LINE}"/>`;
 s+=`<g ${P.clip}><image href="${posteriorImage(view)}" x="${P.x(x0)}" y="${P.y(y1)}" width="${P.x(x1)-P.x(x0)}" height="${P.y(y0)-P.y(y1)}" preserveAspectRatio="none" style="image-rendering:${view==='zoom'?'auto':'pixelated'}"/></g>`+P.axes;
 if(title)s+=label(ox+52,oy+22,title,{fill:palette.ink,size:18,weight:600});
 if(prior&&view==='wide')for(const k of [1,2])s+=`<circle ${P.clip} cx="${P.x(0)}" cy="${P.y(0)}" r="${(k*sx).toFixed(1)}" fill="none" stroke="${MUTED}" stroke-dasharray="5 5" stroke-width="1.6"/>`;
 const ell=(m,sd,stroke,dash,fill)=>`<ellipse ${P.clip} cx="${P.x(m[0]).toFixed(1)}" cy="${P.y(m[1]).toFixed(1)}" rx="${Math.max(2,2*sd[0]*sx).toFixed(1)}" ry="${Math.max(2,2*sd[1]*sy).toFixed(1)}" fill="${fill||'none'}" stroke="${stroke}" stroke-width="3" ${dash?'stroke-dasharray="7 5"':''}/>`;
 if(enc){const e=DATA.vae.q3;s+=ell(e.mu,e.sd,palette.q,true);}
 if(q)s+=ell([q[0],q[1]],[q[2],q[3]],palette.q,false,palette.q+'1F')+`<circle ${P.clip} cx="${P.x(q[0]).toFixed(1)}" cy="${P.y(q[1]).toFixed(1)}" r="4" fill="${palette.q}"/>`;
 return {s:s+extra,P};
}
/** Variational inference for our one 3: q = N(m, diag s²) climbs the ELBO onto the posterior. */
function vi3(el,ctx){
 const V=DATA.vae,fr=V.vi3.frames,W=1150,H=370,lastStep=fr.at(-1)[0],lx=s=>Math.log10(s+1);
 const render=i=>{
  const [step,m1,m2,s1,s2,elbo]=fr[i],trail=fr.slice(0,i+1),q=[m1,m2,s1,s2];
  const A=posteriorSVG({S:360,q,title:'all codes: prior circles at 1 and 2'}),Z=posteriorSVG({ox:380,S:360,view:'zoom',q,title:'zoomed in on the posterior'});
  let s=A.s+`<polyline ${A.P.clip} points="${trail.map(f=>`${A.P.x(f[1]).toFixed(1)},${A.P.y(f[2]).toFixed(1)}`).join(' ')}" fill="none" stroke="${palette.q}" stroke-width="2" stroke-dasharray="3 4"/>`+Z.s;
  const Q=panel({ox:770,W:380,H:360,L:66,R:24,T:62,B:44,xMin:0,xMax:lx(lastStep),yMin:-400,yMax:300,ticksX:[0,1,2,3],fmtX:v=>String(10**v),ticksY:[-400,-200,0,200],xLabel:'step (log scale)',fs:16});
  const xs=trail.map(f=>lx(f[0])),ys=trail.map(f=>Math.max(-400,f[5]));
  s+=Q.axes+`<line x1="${Q.x(0)}" y1="${Q.y(V.grid.log_px).toFixed(1)}" x2="${Q.x(lx(lastStep))}" y2="${Q.y(V.grid.log_px).toFixed(1)}" stroke="${palette.data}" stroke-dasharray="7 5" stroke-width="2"/>`+
   label(Q.x(0)+6,Q.y(V.grid.log_px)-10,`log p(x) = ${num(V.grid.log_px,2)}`,{fill:palette.data,size:16})+
   Q.line(xs,ys,palette.q)+Q.dot(lx(step),Math.max(-400,elbo),palette.q)+label(770+66,22,`ELBO estimate, step ${step}`,{fill:palette.q,size:18,weight:600})+label(770+66,46,`${num(elbo,1)}`,{fill:palette.q,size:22,weight:650});
  return svg(W,H,s,'Variational inference for one image: q moves from N(0, I) onto the posterior, and the ELBO rises toward log p(x).');
 };
 animated(el,ctx,{count:fr.length,render,step:i=>`step ${fr[i][0]}`,alt:'Variational inference for one image',duration:5200});
}
/** The KL weight β: codes, reconstructions and samples for β = 0, 0.1, 1, 4, 16. */
function betaSweep(el,ctx){
 const B=DATA.vae.beta,keys=['0','0.1','1','4','16','64','256'],st={b:el.dataset.beta||'1'},W=1150,H=380;
 const render=()=>{
  const r=B[st.b],{s}=scatterSVG('beta:'+st.b,{W:400,H:380,lim:[-4,4,-4,4],ticks:[[-4,-2,0,2,4],[-4,-2,0,2,4]],r:3.4,opacity:.5});
  let out=s+label(440,28,`β = ${st.b}`,{fill:palette.q,size:26,weight:650})+label(560,28,`KL ${num(r.test_kl,1)} · squared error ${num(r.test_sse,1)} · average σ of q ${num(r.sd_mean,3)}`,{size:19});
  out+=label(440,70,'reconstructions of 8 test digits',{fill:palette.ink,size:19,weight:600});
  for(let j=0;j<8;j++)out+=`<image href="${digitSrc(`beta${st.b}_recon_watch`,j)}" x="${440+j*76}" y="82" width="68" height="68" style="image-rendering:pixelated"/>`;
  out+=label(440,186,'16 samples: z ~ N(0, I), then decode',{fill:palette.ink,size:19,weight:600});
  for(let j=0;j<16;j++)out+=`<image href="${digitSrc(`beta${st.b}_prior_samples`,j)}" x="${440+(j%8)*76}" y="${198+Math.floor(j/8)*76}" width="68" height="68" style="image-rendering:pixelated"/>`;
  out+=label(440,H-8,r.dist.ratio==null?'every code is the same point: distances are all 0':`distance between two 0s ÷ distance from a 0 to a 1: ${num(r.dist.ratio,2)}`,{size:19});
  return svg(W,H,out,`The VAE trained with KL weight β = ${st.b}: its codes, reconstructions and samples.`);
 };
 if(ctx.printing){el.innerHTML=render();return;}
 const controls=`<span>KL weight β</span>`+keys.map(k=>`<button type="button" data-b="${k}" aria-pressed="${k===st.b}">${k}</button>`).join('');
 const draw=interactive(el,ctx,{controls,render});
 el.querySelectorAll('[data-b]').forEach(b=>b.addEventListener('click',()=>{st.b=b.dataset.b;pressed(el,'[data-b]',b);draw();}));
}

export function mount(stage,ctx){
 const make={'faith-sample':faithSample,'faith-fit':faithFit,'pca-slider':pcaSlider,'mog-gen':mogGen,scatter,explorer,'ae-train':aeTrain,vi3,beta:betaSweep};
 for(const el of stage.querySelectorAll('[data-widget]')){
  const f=make[el.dataset.widget];if(!f)throw new Error('Unknown widget '+el.dataset.widget);f(el,ctx);
 }
}
