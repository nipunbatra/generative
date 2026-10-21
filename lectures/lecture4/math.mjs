// Exact formulas for Lecture 4: the noise schedules and the one-pixel example (paper −1 or ink +1).
// scripts/pixel.py computes the same quantities in Python; scripts/check-science.mjs compares them.
export const T=1000;

/** beta, alpha, alpha_bar and the posterior variance beta-tilde, indexed 0..T (index 0: the clean image). */
export function schedule(kind='linear'){
 const beta=new Float64Array(T+1),ab=new Float64Array(T+1),alpha=new Float64Array(T+1),btil=new Float64Array(T+1);
 if(kind==='linear'){for(let t=1;t<=T;t++)beta[t]=1e-4+(0.02-1e-4)*(t-1)/(T-1);}
 else{const s=0.008,f=u=>Math.cos((u+s)/(1+s)*Math.PI/2)**2;for(let t=1;t<=T;t++)beta[t]=Math.min(0.999,1-f(t/T)/f((t-1)/T));}
 ab[0]=1;alpha[0]=1;
 for(let t=1;t<=T;t++){alpha[t]=1-beta[t];ab[t]=ab[t-1]*alpha[t];btil[t]=beta[t]*(1-ab[t-1])/(1-ab[t]);}
 return {beta,alpha,ab,btil};
}
export const LIN=schedule('linear'),COS=schedule('cosine');

/** A schedule from a list of betas (index 0 is the clean image). */
export function scheduleFrom(betas){
 const n=betas.length,beta=new Float64Array(n+1),alpha=new Float64Array(n+1),ab=new Float64Array(n+1),btil=new Float64Array(n+1);
 ab[0]=1;alpha[0]=1;
 for(let t=1;t<=n;t++){beta[t]=betas[t-1];alpha[t]=1-beta[t];ab[t]=ab[t-1]*alpha[t];btil[t]=beta[t]*(1-ab[t-1])/(1-ab[t]);}
 return {T:n,beta,alpha,ab,btil};
}
/** The toy: five big steps, small enough to follow every number by hand. Real models use 1,000 tiny steps. */
export const TOY=scheduleFrom([0.1,0.2,0.3,0.5,0.8]);
LIN.T=T;COS.T=T;
/** The toy's fixed noise draws: eps_1 … eps_5 going forward, and z_5 … z_2 going back (z_1 = 0). */
export const TOY_EPS=[null,0.3,-0.5,0.6,-1.2,0.5],TOY_Z=[0,0,-0.4,0.2,0.5,-0.3];
/** One forward step: x_t = sqrt(alpha_t) x_{t−1} + sqrt(beta_t) eps_t. */
export const forwardStep=(x,t,eps,S)=>Math.sqrt(S.alpha[t])*x+Math.sqrt(S.beta[t])*eps;
/** One reverse step from a noise guess: the guess of x0, the mean, and the new value (sigma_t = sqrt(beta_t)). */
export function reverseStep(xt,t,epsHat,z,S){
 const a=S.ab[t],x0=(xt-Math.sqrt(1-a)*epsHat)/Math.sqrt(a),c=S.beta[t]/Math.sqrt(1-a),mu=(xt-c*epsHat)/Math.sqrt(S.alpha[t]);
 const sigma=t>1?Math.sqrt(S.beta[t]):0;
 return {x0,c,mu,sigma,z:t>1?z:0,next:mu+sigma*(t>1?z:0)};
}

export const npdf=(x,m,v)=>Math.exp(-((x-m)**2)/(2*v))/Math.sqrt(2*Math.PI*v);
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));

/** Bayes for one pixel: P(ink | x_t) = sigmoid(logit pi + 2 sqrt(ab) x_t / (1 − ab)). */
export function pInk(xt,t,pi,S=LIN){const a=S.ab[t];return sigmoid(Math.log(pi/(1-pi))+2*Math.sqrt(a)*xt/(1-a));}
/** The best guess of the clean pixel under squared error: E[x0 | x_t] = P(ink)·(+1) + P(paper)·(−1). */
export const x0Hat=(xt,t,pi,S=LIN)=>2*pInk(xt,t,pi,S)-1;
/** The best possible noise prediction E[eps | x_t], from x_t = sqrt(ab) x0 + sqrt(1 − ab) eps. */
export const epsStar=(xt,t,pi,S=LIN)=>(xt-Math.sqrt(S.ab[t])*x0Hat(xt,t,pi,S))/Math.sqrt(1-S.ab[t]);
/** The score of the noisy pixel's density, by Tweedie: −E[eps | x_t] / sqrt(1 − ab). */
export const score=(xt,t,pi,S=LIN)=>-epsStar(xt,t,pi,S)/Math.sqrt(1-S.ab[t]);
/** Density of the noisy pixel: two bumps at ±sqrt(ab) with variance 1 − ab. */
export function density(x,t,pi,S=LIN){const a=S.ab[t];if(t===0)return 0;return pi*npdf(x,Math.sqrt(a),1-a)+(1-pi)*npdf(x,-Math.sqrt(a),1-a);}
/** Mean of q(x_{t−1} | x_t, x0): the step back if we knew the clean value. */
export function muTilde(xt,x0,t,S=LIN){const {beta,alpha,ab}=S;return (Math.sqrt(ab[t-1])*beta[t]*x0+Math.sqrt(alpha[t])*(1-ab[t-1])*xt)/(1-ab[t]);}
/**
 * The exact reverse density q(x_s | x_t) for a jump from t back to s < t: a mixture of the two Gaussians
 * q(x_s | x_t, x0 = ±1), weighted by P(x0 = ±1 | x_t). With s = t − 1 it is one step; with s = 0 it is two spikes.
 */
export function reverseKernel(xt,t,s,pi,S=LIN){
 const {ab}=S,ats=ab[t]/ab[s],v=s>0?(1-ab[s])*(1-ats)/(1-ab[t]):0;
 const m=k=>(Math.sqrt(ab[s])*(1-ats)*k+Math.sqrt(ats)*(1-ab[s])*xt)/(1-ab[t]);
 const P=pInk(xt,t,pi,S),mi=m(1),mp=m(-1);
 const fitMean=P*mi+(1-P)*mp,fitVar=v+P*(1-P)*(mi-mp)**2;
 return {weights:[P,1-P],means:[mi,mp],sd:Math.sqrt(v),fitMean,fitSd:Math.sqrt(fitVar),
  pdf:x=>v>0?P*npdf(x,mi,v)+(1-P)*npdf(x,mp,v):0,fit:x=>npdf(x,fitMean,fitVar)};
}
/** Var(eps | x_t) averaged over x_t: the part of the noise no network can predict. */
export function irreducible(t,pi,S=LIN){
 const a=S.ab[t],n=2401;let sum=0;
 for(let i=0;i<n;i++){const x=-6+12*i/(n-1),P=pInk(x,t,pi,S),d=density(x,t,pi,S);sum+=a/(1-a)*(1-(2*P-1)**2)*d*(12/(n-1));}
 return sum;
}
/** Seeded standard normals (mulberry32 + Box–Muller), so that every viewer sees the same draws. */
export function randn(seed){
 let a=seed>>>0;const u=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
 let spare=null;
 return ()=>{if(spare!==null){const s=spare;spare=null;return s;}let x,y,r;do{x=2*u()-1;y=2*u()-1;r=x*x+y*y;}while(!r||r>=1);const k=Math.sqrt(-2*Math.log(r)/r);spare=y*k;return x*k;};
}
