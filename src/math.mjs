// All analytic UI values and lab fixtures originate here. Natural logs, float64.
export function distribution(p){if(!Array.isArray(p)||!p.length||p.some(x=>!Number.isFinite(x)||x<0)||Math.abs(p.reduce((a,b)=>a+b,0)-1)>1e-10)throw Error('Expected a normalised nonnegative distribution');return p;}
export function klDiscrete(p,q){distribution(p);distribution(q);if(p.length!==q.length)throw Error('Different supports');return p.reduce((s,x,i)=>s+(x===0?0:q[i]===0?Infinity:x*Math.log(x/q[i])),0);}
export function entropy(p){distribution(p);return -p.reduce((s,x)=>s+(x===0?0:x*Math.log(x)),0);}
export function crossEntropy(p,q){return entropy(p)+klDiscrete(p,q);}
export function finiteELBO(q0=.6){const prior=[.5,.5],likelihood=[.8,.2],q=[q0,1-q0];distribution(q);const evidence=prior.reduce((s,p,i)=>s+p*likelihood[i],0),posterior=prior.map((p,i)=>p*likelihood[i]/evidence),reconstruction=q.reduce((s,p,i)=>s+p*Math.log(likelihood[i]),0),priorKL=klDiscrete(q,prior),elbo=reconstruction-priorKL,gap=klDiscrete(q,posterior);return {prior,likelihood,q,evidence,posterior,reconstruction,priorKL,elbo,gap,logEvidence:Math.log(evidence)};}
export function reparam(mu=1,sigma=.5,epsilon=-.4,a=0){if(!(sigma>0))throw Error('sigma must be positive');const z=mu+sigma*epsilon,loss=(z-a)**2,dmu=2*(z-a),dsigma=dmu*epsilon,dlogvar=dsigma*sigma/2;return {mu,sigma,epsilon,a,z,loss,dmu,dsigma,dlogvar};}
export function gaussianKL(mu,logvar){return mu.map((v,i)=>.5*(v*v+Math.exp(logvar[i])-1-logvar[i]));}
export function twoPixel(){const x=[1,0],mu=[1,-1],sigma=[.5,1],epsilon=[.2,-.4],pi=[.75,.25],z=mu.map((m,i)=>m+sigma[i]*epsilon[i]),logvar=sigma.map(s=>2*Math.log(s)),terms=gaussianKL(mu,logvar),recon=-x.reduce((s,v,i)=>s+v*Math.log(pi[i])+(1-v)*Math.log(1-pi[i]),0),kl=terms.reduce((a,b)=>a+b,0);return {x,mu,sigma,epsilon,pi,z,logvar,terms,recon,kl,total:recon+kl};}
export const sigmoid=a=>a>=0?1/(1+Math.exp(-a)):Math.exp(a)/(1+Math.exp(a));
export function ganPair(real=.8,fake=.3){return {real,fake,D:-Math.log(real)-Math.log1p(-fake),G:-Math.log(fake)};}
export function saturation(a=Math.log(.01/.99)){const d=sigmoid(a);return {a,d,minimax:-Math.log1p(Math.exp(a)),nonSaturating:Math.log1p(Math.exp(-a)),dMinimax:-d,dNonSaturating:d-1};}
export function rng(seed=667){let s=seed>>>0;return ()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};}
export const normal=r=>Math.sqrt(-2*Math.log(Math.max(r(),1e-15)))*Math.cos(2*Math.PI*r());
export const normalPDF=(x,mu=0,s=1)=>Math.exp(-.5*((x-mu)/s)**2)/(s*Math.sqrt(2*Math.PI));
export function gaussianMLE(data=[-2.2,-2,-1.8,1.8,2,2.2]){const mu=data.reduce((a,b)=>a+b,0)/data.length,variance=data.reduce((s,x)=>s+(x-mu)**2,0)/data.length;return {mu,variance,sigma:Math.sqrt(variance)};}
export function toyMetrics(points,radius=.65){const hit=points.map(([x,y])=>[-2,2].map(c=>Math.hypot(x-c,y)<radius));return {realism:hit.filter(h=>h.some(Boolean)).length/points.length,coverage:[0,1].filter(j=>hit.filter(h=>h[j]).length>=5).length,threshold:radius,minimumCount:5};}
export function fixtures(){return {kl:klDiscrete([.8,.2],[.6,.4]),reverseKL:klDiscrete([.6,.4],[.8,.2]),entropy:entropy([.8,.2]),crossEntropy:crossEntropy([.8,.2],[.6,.4]),finite:finiteELBO(),reparam:reparam(),vae:twoPixel(),gan:ganPair(),mle:gaussianMLE(),saturation:saturation()};}
if(typeof process!=='undefined'&&process.argv[1]?.split('/').at(-1)==='math.mjs')console.log(JSON.stringify(fixtures(),null,2));
