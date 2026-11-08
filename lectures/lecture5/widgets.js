// Lecture 5's own live figure: choose q for our 3 and watch the two terms of the ELBO.
// It reuses Lecture 2's chart kit and runs Lecture 2's 2-number VAE decoder (lecture2/data/nets.json) in the browser.
import {palette,esc} from '../shared/author.js';
import {label,svg,num,digitURL,digitSrc,imgs,scatterSVG,squareW,MUTED,LINE} from '../lecture2/widgets.js';

let DATA=null,NETS=null;
export function init(d,nets){DATA=d;NETS=nets;}
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

/** The reconstruction term, the KL to N(0, I), and the ELBO, for q = N(μ, diag σ²) and our 3. */
function terms(x,mu,sd){
 let recon=0;
 for(const [e1,e2,w] of NODES){const f=decode([mu[0]+sd*e1,mu[1]+sd*e2]);let s=0;for(let k=0;k<784;k++){const d=x[k]-f[k];s+=d*d;}recon+=w*s/(2*SIGMA_X*SIGMA_X);}
 const kl=.5*(mu[0]**2+mu[1]**2+2*sd*sd-2*Math.log(sd*sd)-2);
 return {recon,kl,elbo:-recon-kl-DATA.vae.loglik_const};
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

/** Mount Lecture 5's own live figures (data-widget5); Lecture 2's are mounted by its module. */
export function mount(stage,ctx){
 const make={'elbo-explorer':elboExplorer};
 for(const el of stage.querySelectorAll('[data-widget5]')){const f=make[el.dataset.widget5];if(!f)throw new Error('Unknown widget '+el.dataset.widget5);f(el,ctx);}
}
export const widget5=(name,alt,attrs={})=>`<div class="widget" data-widget5="${name}" role="group" aria-label="${esc(alt)}" ${Object.entries(attrs).map(([k,v])=>`data-${k}="${esc(String(v))}"`).join(' ')}></div>`;
