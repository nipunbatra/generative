// Lecture 5's own live figure: choose q for our 3 and watch the two terms of the ELBO.
// It reuses Lecture 2's chart kit and runs Lecture 2's 2-number VAE decoder (lecture2/data/nets.json) in the browser.
import {palette,esc} from '../shared/author.js';
import {label,svg,num,comma,panel,digitURL,digitSrc,imgs,scatterSVG,squareW,classLegend,CLASS,MUTED,LINE} from '../lecture2/widgets.js';

let DATA=null,NETS=null,EPOCHS=null;
export function init(d,nets,epochs){DATA=d;NETS=nets;EPOCHS=epochs;}
const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
function halfToFloat(h){const s=(h&0x8000)?-1:1,e=(h>>10)&0x1f,f=h&0x3ff;return e===0?s*Math.pow(2,-14)*(f/1024):e===31?(f?NaN:s*Infinity):s*Math.pow(2,e-15)*(1+f/1024);}
function f16(rec){const b=b64(rec.data),u=new Uint16Array(b.buffer,b.byteOffset,b.byteLength/2),o=new Float32Array(u.length);for(let i=0;i<u.length;i++)o[i]=halfToFloat(u[i]);return o;}
/** The 2-number VAE's decoder, 2 → 400 → 784, with a sigmoid at the end. */
let DEC=null;
function decode(z){
 if(!DEC){const [W1,b1,W2,b2]=NETS.vae2.dec.map(f16);DEC={W1,b1,W2,b2,H:b1.length};}
 const {W1,b1,W2,b2,H}=DEC,h=new Float32Array(H);
 for(let j=0;j<H;j++){const s=b1[j]+W1[j*2]*z[0]+W1[j*2+1]*z[1];h[j]=s>0?s:0;}
 const x=new Float32Array(784);
 for(let k=0;k<784;k++){let s=b2[k];const r=k*H;for(let j=0;j<H;j++)s+=W2[r+j]*h[j];x[k]=1/(1+Math.exp(-s));}
 return x;
}
// E over ε ~ N(0, I) in two dimensions by 8 × 8 Gauss–Hermite quadrature: accurate to about 0.01 nats here.
const GH_X=[-2.930637420257244,-1.981656756695843,-1.157193712446780,-.3811869902073221,.3811869902073221,1.157193712446780,1.981656756695843,2.930637420257244];
const GH_W=[1.996040722114e-4,.01707798300741,.2078023258148919,.6611470125582413,.6611470125582413,.2078023258148919,.01707798300741,1.996040722114e-4];
const NODES=GH_X.flatMap((a,i)=>GH_X.map((b,j)=>[Math.SQRT2*a,Math.SQRT2*b,GH_W[i]*GH_W[j]/Math.PI]));
const SHOW_EPS=[[.6,-.4],[-1.1,.3],[.2,1.2],[1.4,.9],[-.5,-1.3]];
const SIGMA_X=.2;

/** The reconstruction term, the KL to N(0, I), and the ELBO, for q = N(μ, diag σ²) and our 3. σ: one number, or one per code number. */
function terms(x,mu,sd){
 const [s1,s2]=Array.isArray(sd)?sd:[sd,sd];
 let recon=0;
 for(const [e1,e2,w] of NODES){const f=decode([mu[0]+s1*e1,mu[1]+s2*e2]);let s=0;for(let k=0;k<784;k++){const d=x[k]-f[k];s+=d*d;}recon+=w*s/(2*SIGMA_X*SIGMA_X);}
 const kl=.5*(mu[0]**2+mu[1]**2+s1*s1+s2*s2-Math.log(s1*s1)-Math.log(s2*s2)-2);
 return {recon,kl,elbo:-recon-kl-DATA.vae.loglik_const};
}
const our3=()=>Float32Array.from(imgs('example').px,v=>v/255);
/** For static figures: the ELBO of q = N(μ, diag σ²) for our 3, and log p(x | z) of our 3 for one code. */
export const elboFor=(mu,sd)=>terms(our3(),mu,sd);
export function logLikFor(z){
 const x=our3(),f=decode(z);let sse=0;for(let k=0;k<784;k++){const d=x[k]-f[k];sse+=d*d;}
 return {sse,loglik:-sse/(2*SIGMA_X*SIGMA_X)-DATA.vae.loglik_const,url:digitURL(f)};
}

function elboExplorer(el,ctx){
 const V=DATA.vae,x=Float32Array.from(imgs('example').px,v=>v/255),logpx=V.grid.log_px;
 const lim=[-3.5,3.5,-3.5,3.5],PH=380,PW=squareW(lim,PH);
 // “q on the 1s”: μ = (0, 3.2), where the VAE puts its 1s (the mean code of the 1s among Lecture 2's 1,000 display digits is (−0.04, 3.16)).
 const presets={enc:[[V.q3.mu[0],V.q3.mu[1]],Math.sqrt(V.q3.sd[0]*V.q3.sd[1])],prior:[[0,0],1],one:[[0,3.2],.05]};
 const st={mu:presets.enc[0].slice(),sd:presets.enc[1]};
 const base=scatterSVG('vae2',{W:PW,H:PH,lim,ticks:[[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]],opacity:.28,r:3,contours:true});
 const P=base.P,ux=(P.x(1)-P.x(0)),uy=(P.y(0)-P.y(1));
 const X0=PW+34,W=1150,H=PH+8;
 const bar=(t)=>{
  const x0=X0,x1=W-20,lo=-600,hi=300,sx=v=>x0+(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*(x1-x0),y=PH-56;
  let s=`<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${MUTED}" stroke-width="2"/>`;
  for(const v of [-600,-300,0,300])s+=`<line x1="${sx(v)}" y1="${y-5}" x2="${sx(v)}" y2="${y+5}" stroke="${MUTED}"/>`+label(sx(v),y+24,num(v,0),{size:15,anchor:'middle'});
  const ce=sx(logpx),ee=sx(t.elbo);
  s+=`<line x1="${ce}" y1="${y-34}" x2="${ce}" y2="${y+8}" stroke="${palette.data}" stroke-width="3"/>`+label(ce,y-42,`log p(x) = ${num(logpx,1)}: the ceiling`,{fill:palette.data,size:17,anchor:'end',weight:650});
  s+=label(x0,y-14,'purple dot: the ELBO · red: the gap',{size:15})+`<line x1="${ee}" y1="${y}" x2="${ce}" y2="${y}" stroke="${palette.loss}" stroke-width="6" stroke-opacity=".75"/>`+`<circle cx="${ee}" cy="${y}" r="8" fill="${palette.q}" stroke="#fff" stroke-width="2"/>`+(t.elbo<lo?label(ee+12,y-12,'off the scale',{fill:palette.q,size:15}):'');
  return s;
 };
 const render=()=>{
  const t=terms(x,st.mu,st.sd),cx=P.x(st.mu[0]),cy=P.y(st.mu[1]);
  let s=base.s+`<ellipse cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" rx="${Math.max(3,2*st.sd*ux).toFixed(1)}" ry="${Math.max(3,2*st.sd*uy).toFixed(1)}" fill="${palette.q}22" stroke="${palette.q}" stroke-width="2.5"/>`+
   `<line x1="${cx-9}" y1="${cy}" x2="${cx+9}" y2="${cy}" stroke="${palette.q}" stroke-width="3"/><line x1="${cx}" y1="${cy-9}" x2="${cx}" y2="${cy+9}" stroke="${palette.q}" stroke-width="3"/>`+
   label(P.x(lim[0])+6,22,'click to move μ; q drawn at 2σ',{fill:palette.q,size:16,weight:600});
  const img=(href,xx,yy,sz,col)=>`<image href="${href}" x="${xx}" y="${yy}" width="${sz}" height="${sz}" style="image-rendering:pixelated"/><rect x="${xx}" y="${yy}" width="${sz}" height="${sz}" fill="none" stroke="${col}" stroke-width="${col===LINE?1:2.5}"/>`;
  s+=label(X0,22,'x: our 3',{fill:palette.p,size:17,weight:650})+img(digitSrc('example',0),X0,32,104,palette.p)+
   label(X0+124,22,'f(μ): decoded',{fill:palette.q,size:17,weight:650})+img(digitURL(decode(st.mu)),X0+124,32,104,palette.q)+
   label(X0+250,22,'five draws z from q, decoded',{size:17});
  SHOW_EPS.forEach(([a,b],i)=>{s+=img(digitURL(decode([st.mu[0]+st.sd*a,st.mu[1]+st.sd*b])),X0+250+i*70,44,62,LINE);});
  const row=(y,name,val,col,note)=>label(X0,y,name,{fill:col,size:19,weight:650})+label(X0+250,y,num(val,1),{fill:col,size:21,anchor:'end',weight:700})+label(X0+266,y,note,{size:16});
  s+=row(170,'reconstruction',t.recon,palette.noise,'E‖x − f(z)‖² / 2σ²: small if draws rebuild our 3')+
   row(200,'KL to the prior',t.kl,palette.ink,'KL(q ‖ N(0, I)): small if q sits near N(0, I)')+
   row(234,'ELBO',t.elbo,palette.q,'= 541.4 − reconstruction − KL')+
   label(X0+266,260,`gap to log p(x): ${num(Math.max(0,logpx-t.elbo),1)}, the KL from q to the posterior`,{fill:palette.loss,size:16})+bar(t);
  return svg(W,H,s,'Choose q for our 3: the reconstruction term, the KL to the prior, and the ELBO below log p(x).');
 };
 const describe=()=>`μ = (${num(st.mu[0],2)}, ${num(st.mu[1],2)}), σ = ${st.sd<.1?num(st.sd,3):num(st.sd,2)}`;
 if(ctx.printing){el.innerHTML=render();el.insertAdjacentHTML('beforeend',`<p class="caption">q = the encoder’s q for our 3: ${describe()}.</p>`);return;}
 const toSlider=sd=>Math.log10(sd),fromSlider=v=>10**v;
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><label>σ of q <input type="range" min="-2.3" max="0.3" step="0.01" value="${toSlider(st.sd)}" aria-label="spread of q"></label><span class="w-readout"></span>`+
  [['enc','the encoder’s q'],['prior','q = N(0, I)'],['one','q on the 1s']].map(([k,t])=>`<button type="button" data-preset="${k}" aria-pressed="${k==='enc'}">${esc(t)}</button>`).join('')+`</div>`;
 const box=el.querySelector('.w-fig'),slider=el.querySelector('input'),out=el.querySelector('.w-readout');
 // Coalesce bursts of slider events into one redraw; a timer, so it also runs where animation frames are throttled.
 let pending=false;
 const draw=()=>{if(pending)return;pending=true;setTimeout(()=>{pending=false;box.innerHTML=render();out.textContent=describe();},0);};
 const press=k=>el.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.preset===k)));
 slider.addEventListener('input',()=>{st.sd=fromSlider(Number(slider.value));press('');draw();});
 el.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{const [m,s_]=presets[b.dataset.preset];st.mu=m.slice();st.sd=s_;slider.value=String(toSlider(s_));press(b.dataset.preset);draw();}));
 box.addEventListener('click',e=>{const sv=box.querySelector('svg');if(!sv)return;const pt=sv.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(sv.getScreenCTM().inverse());
  if(p.x>PW)return;const a=(p.x-P.x(0))/ux,bb=(P.y(0)-p.y)/uy;if(a<lim[0]||a>lim[1]||bb<lim[2]||bb>lim[3])return;st.mu=[a,bb];press('');draw();});
 box.innerHTML=render();out.textContent=describe();
}

// ---------------------------------------------------------------- the codes while training (scripts/epochs.py)
const niceStep=span=>{const raw=span/5,p=10**Math.floor(Math.log10(raw)),m=raw/p;return (m<1.5?1:m<3.5?2:m<7.5?5:10)*p;};
const ticksFor=(lo,hi)=>{const st=niceStep(hi-lo),out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)out.push(+v.toFixed(6));return out;};
/** A square window, equal units on both axes, around a 1–99% box [[x0, y0], [x1, y1]] and the origin. */
function squareLim([[a,b],[c,e]]){
 const x0=Math.min(a,0),x1=Math.max(c,0),y0=Math.min(b,0),y1=Math.max(e,0),cx=(x0+x1)/2,cy=(y0+y1)/2,h=Math.max(x1-x0,y1-y0)/2*1.18||1;
 return [cx-h,cx+h,cy-h,cy+h];
}
function codePanel(C,lim,{ox=0,oy=0,S=430,r=3.2,title='',fs=16}={}){
 const P=panel({ox,oy,W:S,H:S,L:50,R:10,T:title?34:12,B:40,xMin:lim[0],xMax:lim[1],yMin:lim[2],yMax:lim[3],ticksX:ticksFor(lim[0],lim[1]),ticksY:ticksFor(lim[2],lim[3]),fs,xAxisAtBottom:true});
 const labs=DATA.mnist.display_labels;
 let s=`<rect x="${ox+50}" y="${oy+(title?34:12)}" width="${S-60}" height="${S-(title?74:52)}" fill="#fff" stroke="${LINE}"/>`+P.axes;
 if(lim[0]<0&&lim[1]>0)s+=`<line ${P.clip} x1="${P.x(0)}" y1="${P.y(lim[2])}" x2="${P.x(0)}" y2="${P.y(lim[3])}" stroke="${LINE}"/><line ${P.clip} x1="${P.x(lim[0])}" y1="${P.y(0)}" x2="${P.x(lim[1])}" y2="${P.y(0)}" stroke="${LINE}"/>`;
 s+=`<g ${P.clip}>${C.map(([a,b],i)=>`<circle cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="${r}" fill="${CLASS[labs[i]]}" fill-opacity=".6"/>`).join('')}</g>`;
 const [a,b]=C[0];s+=`<circle ${P.clip} cx="${P.x(a).toFixed(1)}" cy="${P.y(b).toFixed(1)}" r="${r+6}" fill="none" stroke="${palette.ink}" stroke-width="2.5"/>`;
 if(title)s+=label(ox+50,oy+22,title,{fill:palette.ink,size:fs+2,weight:650});
 return s;
}
/** How wide the cloud of codes is: the larger side of its 1st–99th percentile box. */
export const cloudW=sm=>Math.max(sm.box[1][0]-sm.box[0][0],sm.box[1][1]-sm.box[0][1]);
const EP_TEXT={all:'trained on all 60,000 digits',few:'trained on only 200 digits (20 of each)'};
function codeEpochs(el,ctx){
 const run=el.dataset.run,E=EPOCHS[run],save=E.save,last=String(save.at(-1));
 const fixed=squareLim(E.summary[last].box),b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
 const img=(href,x,y,sz,col)=>`<image href="${href}" x="${x}" y="${y}" width="${sz}" height="${sz}" style="image-rendering:pixelated"/><rect x="${x}" y="${y}" width="${sz}" height="${sz}" fill="none" stroke="${col}" stroke-width="2"/>`;
 const yMax=run==='few'?100:45,xMax=save.at(-1);
 const chart=(ox,oy,W,H,ep)=>{
  const lx=v=>Math.log10(Math.max(1,v)),P=panel({ox,oy,W,H,L:52,R:16,T:34,B:44,xMin:0,xMax:lx(xMax),yMin:0,yMax,ticksX:[1,10,100,1000,10000].filter(v=>v<=xMax).map(lx),ticksY:ticksFor(0,yMax),fmtX:v=>comma(10**v),fs:15});
  const keep=E.epoch.map((v,i)=>i).filter(i=>E.epoch[i]>=1),xs=keep.map(i=>lx(E.epoch[i]));
  let s=P.axes+label(ox+52,oy+20,'average squared error per digit, against epoch (log scale)',{size:15})+P.line(xs,keep.map(i=>E.train[i]),palette.data)+P.line(xs,keep.map(i=>E.test[i]),palette.noise);
  s+=label(ox+W-16,oy+20,`<tspan fill="${palette.data}" font-weight="650">training digits</tspan>  <tspan fill="${palette.noise}" font-weight="650">test digits</tspan>`,{size:15,anchor:'end'});
  if(ep>=1){const i=E.epoch.indexOf(ep);s+=`<line x1="${P.x(lx(ep))}" y1="${oy+34}" x2="${P.x(lx(ep))}" y2="${oy+H-44}" stroke="${palette.ink}" stroke-width="1.5" stroke-dasharray="4 4"/>`+P.dot(lx(ep),Math.min(yMax,E.train[i]),palette.data)+P.dot(lx(ep),Math.min(yMax,E.test[i]),palette.noise);}
  if(run==='few'){const bx=P.x(lx(E.best.epoch));s+=`<circle cx="${bx}" cy="${P.y(E.best.test)}" r="5" fill="none" stroke="${palette.noise}" stroke-width="2"/>`+label(bx,P.y(E.best.test)-14,`lowest test error, epoch ${comma(E.best.epoch)}`,{fill:palette.noise,size:14,anchor:'middle'});}
  return s;
 };
 const st={k:0,mode:'fixed'};
 const render=()=>{
  const ep=save[st.k],key=String(ep),C=E.codes[key],sm=E.summary[key];
  const lim=st.mode==='fixed'?fixed:squareLim(sm.box);
  let s=codePanel(C,lim,{S:396,title:`the codes at epoch ${comma(ep)}`});
  const X0=430;
  s+=label(X0,22,EP_TEXT[run],{fill:palette.ink,size:19,weight:650})+classLegend(X0+446,22,14,25);
  s+=chart(X0,30,694,222,ep);
  const y=282;
  s+=label(X0,y,'our 3',{fill:palette.p,size:16,weight:650})+img(digitSrc('example',0),X0,y+8,92,palette.p)+
   label(X0+110,y,'its rebuild',{fill:palette.q,size:16,weight:650})+img(digitURL(b64(E.recon3[key])),X0+110,y+8,92,palette.q);
  const nt=E.nearest_train_to_recon3;
  if(nt&&key===last)s+=label(X0+220,y,`a training ${nt.label}`,{fill:palette.ink,size:16,weight:650})+img(digitURL(b64(nt.image)),X0+220,y+8,92,palette.ink);
  const tx=X0+(nt&&key===last?366:250),row=(yy,t,v,c)=>label(tx,yy,t,{size:17})+label(X0+690,yy,v,{fill:c,size:19,anchor:'end',weight:700});
  s+=row(y+16,'error on the training digits',num(sm.train,1),palette.data)+row(y+42,'error on new (test) digits',num(sm.test,1),palette.noise)+
   row(y+68,'width of the code cloud (1st–99th percentile)',num(cloudW(sm),0),palette.ink)+row(y+94,'neighbours with the same digit',`${num(100*sm.knn5,0)}%`,palette.ink);
  return svg(1150,396,s,`The 2-number codes of 1,000 test digits at epoch ${ep}, ${EP_TEXT[run]}, with training and test error.`);
 };
 if(ctx.printing){
  const pick=run==='few'?[10,100,1000,10000]:[0,1,30,300],S_=284;
  let s='';pick.forEach((ep,i)=>{const sm=E.summary[String(ep)];s+=codePanel(E.codes[String(ep)],fixed,{ox:i*(S_+4),S:S_,r:2.2,title:`epoch ${comma(ep)}`,fs:14})+label(i*(S_+4)+50,S_+22,`train ${num(sm.train,1)} · test ${num(sm.test,1)}`,{size:15});});
  el.innerHTML=svg(1150,S_+32,s,`The codes ${EP_TEXT[run]} at epochs ${pick.join(', ')}, on the same axes.`)+`<p class="caption">${esc(EP_TEXT[run][0].toUpperCase()+EP_TEXT[run].slice(1))}: the same axes in every panel. Our 3 is ringed.</p>`;return;
 }
 el.innerHTML=`<div class="w-fig diagram-scroll"></div><div class="w-controls"><button type="button" data-a="play">▶ play</button><label>epoch <input type="range" min="0" max="${save.length-1}" step="1" value="0" aria-label="epoch"></label><span class="w-readout"></span>`+
  `<button type="button" data-mode="fixed" aria-pressed="true">same axes throughout</button><button type="button" data-mode="fit" aria-pressed="false">zoom to fit</button></div>`;
 const box=el.querySelector('.w-fig'),slider=el.querySelector('input'),out=el.querySelector('.w-readout'),play=el.querySelector('[data-a=play]');
 const draw=()=>{box.innerHTML=render();out.textContent=`epoch ${comma(save[st.k])}`;slider.value=String(st.k);};
 let timer=null;const stop=()=>{clearInterval(timer);timer=null;play.textContent='▶ play';};
 slider.addEventListener('input',()=>{stop();st.k=Number(slider.value);draw();});
 play.addEventListener('click',()=>{if(timer){stop();return;}if(st.k>=save.length-1)st.k=0;play.textContent='❚❚ pause';draw();timer=setInterval(()=>{if(!el.isConnected||st.k>=save.length-1){stop();return;}st.k++;draw();},1100);});
 el.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{st.mode=b.dataset.mode;el.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();}));
 draw();
}

/** Mount Lecture 5's own live figures (data-widget5); Lecture 2's are mounted by its module. */
export function mount(stage,ctx){
 const make={'elbo-explorer':elboExplorer,'code-epochs':codeEpochs};
 for(const el of stage.querySelectorAll('[data-widget5]')){const f=make[el.dataset.widget5];if(!f)throw new Error('Unknown widget '+el.dataset.widget5);f(el,ctx);}
}
export const widget5=(name,alt,attrs={})=>`<div class="widget" data-widget5="${name}" role="group" aria-label="${esc(alt)}" ${Object.entries(attrs).map(([k,v])=>`data-${k}="${esc(String(v))}"`).join(' ')}></div>`;
