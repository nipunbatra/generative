// Independent JS formula checks against executed float64 PyTorch notebook output.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  normalKL, normalKLGrad, normalPdf, betaPdf, betaKL, coinELBO,
  coinLogEvidence, coinPosterior,
} from '../math.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const d=JSON.parse(fs.readFileSync(path.join(root,'data/science.json'),'utf8'));
const records=[];
function check(name,condition){if(!condition)throw new Error(name);records.push({name,kind:'assertion',passed:true});}
function near(name,actual,expected,tolerance){
 const error=Math.abs(actual-expected);
 if(!Number.isFinite(error)||error>tolerance)throw new Error(`${name}: ${actual} != ${expected}; error ${error}`);
 records.push({name,kind:'numerical',actual,expected,error,tolerance,passed:true});
}
const target=d.normal.target;
for(const [name,q] of Object.entries(d.normal.candidates)){
 near(`${name} forward KL`,normalKL(q,target),q.kl,2e-12);
 near(`${name} reverse KL`,normalKL(target,q),q.reverse_kl,2e-12);
 near(`${name} quadrature`,q.quadrature,q.kl,1e-9);
}
near('Normal matched KL',normalKL(target,target),0,0);
const step=d.normal.one_step;
const grad=normalKLGrad({mu:step.mu_before,sigma:step.sigma},target);
near('Normal analytic dmu',grad.dMu,step.gradient,1e-12);
near('Normal finite difference',step.gradient,step.finite_difference,1e-9);
near('Literal scalar mean update',step.mu_before-step.eta*step.gradient,step.mu_after,0);
near('Literal updated KL',normalKL({mu:step.mu_after,sigma:step.sigma},target),step.kl_after,1e-12);
check('Literal update improves KL',step.kl_after<step.kl_before);
near('Mean-only family optimum',normalKL({mu:1,sigma:2},target),step.fixed_scale_best_kl,1e-12);
for(const row of d.normal.training.trace){
 check(`Normal step${row.step} positive scale`,row.sigma>0);
 near(`Normal step${row.step} analytic KL`,normalKL(row,target),row.kl,2e-12);
}
const prior={alpha:5,beta:3},data={heads:17,tails:13},post={alpha:22,beta:16};
check('Conjugate posterior shapes',JSON.stringify(coinPosterior(prior,data))===JSON.stringify(post));
near('Ordered sequence evidence',coinLogEvidence(prior,data),d.coin.log_evidence,5e-11);
check('Sequence counts',d.coin.ordered_sequence.length===30&&[...d.coin.ordered_sequence].filter(t=>t==='H').length===17);
for(const [i,row] of d.coin.candidates.entries()){
 const q={alpha:row.a,beta:row.b};
 near(`Beta candidate${i} direct ELBO`,coinELBO(q,prior,data).elbo,row.elbo,5e-11);
 near(`Beta candidate${i} posterior KL`,betaKL(q,post),row.posterior_kl,5e-11);
 near(`Beta candidate${i} identity`,row.elbo+row.posterior_kl,d.coin.log_evidence,5e-11);
 near(`Beta candidate${i} quadrature`,d.checks.coin_quadrature[i].numerical_elbo,row.elbo,1e-8);
}
for(const row of d.coin.training.trace){
 const q={alpha:row.a,beta:row.b};
 check(`Coin step${row.step} positive shapes`,row.a>0&&row.b>0);
 near(`Coin step${row.step} direct ELBO`,coinELBO(q,prior,data).elbo,row.exact_elbo,7e-11);
 near(`Coin step${row.step} KL`,betaKL(q,post),row.posterior_kl,7e-11);
 near(`Coin step${row.step} identity`,row.exact_elbo+row.posterior_kl,d.coin.log_evidence,7e-11);
}
check('Coin fit uses joint-only objective',d.coin.training.reference_used_by_optimiser===false);
check('Coin fit close but not falsely labelled exact',d.coin.training.trace.at(-1).posterior_kl>0&&d.coin.training.trace.at(-1).posterior_kl<.01);
const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
for(const [i,row] of d.monte_carlo.repeats.entries()){
 check(`MC group${i} forty seeds`,row.estimates.length===40&&row.seeds.length===40);
 near(`MC group${i} empirical mean`,mean(row.estimates),row.empirical_mean,1e-12);
 near(`MC group${i} empirical sd`,Math.sqrt(sum(row.estimates.map(x=>(x-mean(row.estimates))**2))/39),row.empirical_std,1e-12);
}
const neg=d.monte_carlo.negative_example;
near('Negative example sample average',mean(neg.terms),neg.estimate,1e-12);
check('Negative estimate, positive true KL',neg.estimate<0&&d.monte_carlo.exact_kl>0);
for(let i=0;i<5;i++)near(`Negative sample${i} ratio`,neg.log_q[i]-neg.log_p[i],neg.terms[i],1e-12);
for(let i=0;i<2;i++)near(`Fixed noise gradient${i}`,d.gradients.finite_difference.autograd[i],
 d.gradients.finite_difference.finite_difference[i],1e-8);
check('Detached scalar sample has no path',d.gradients.sample_requires_grad===false&&d.gradients.sample_backward_error.includes('does not require grad'));
check('Naive detached and pathwise gradients differ',Math.abs(d.gradients.detached_sample_grads[1]-d.gradients.pathwise_grads[1])>.3);
const scalar=d.gradients.scalar;
near('Scalar sample',scalar.mu+scalar.sigma*scalar.epsilon,scalar.u,0);
near('Scalar dloss/dmu',2*(scalar.u-1),scalar.dloss_dmu,0);
near('Scalar dloss/dsigma',2*(scalar.u-1)*scalar.epsilon,scalar.dloss_dsigma,0);
for(const row of d.mixture.candidates){
 check(`${row.id} mixture estimate compatible with quadrature`,Math.abs(row.mc-row.quadrature)<5*row.standard_error);
}
function curve(name,x,ys,fn,tolerance){
 check(`${name} length and finite`,x.length===ys.length&&ys.every(Number.isFinite));
 near(`${name} independent density max error`,Math.max(...x.map((v,i)=>Math.abs(fn(v)-ys[i]))),0,tolerance);
}
for(const key of ['p','q1','q2','updated']){
 const q=key==='p'?target:key==='updated'?{mu:.25,sigma:2}:d.normal.candidates[key];
 curve(`Normal ${key}`,d.plots.normal.x,d.plots.normal[key],x=>normalPdf(x,q),1e-12);
}
for(let i=0;i<3;i++){
 const row=d.coin.candidates[i];
 curve(`Beta candidate${i}`,d.plots.coin.x,d.plots.coin.candidates[i],
 x=>betaPdf(x,{alpha:row.a,beta:row.b}),2e-10);
}
let rejects=0;
for(const sigma of [0,-1,Infinity,NaN]){try{normalPdf(0,{mu:0,sigma});}catch{rejects++;}}
check('Invalid Normal scales rejected',rejects===4);
rejects=0;
for(const alpha of [0,-1,Infinity,NaN]){try{betaPdf(.5,{alpha,beta:2});}catch{rejects++;}}
check('Invalid Beta shapes rejected',rejects===4);
// Story fixtures added for the rebuilt lecture: every value checked independently.
const st=d.story;
const trap=(xs,f)=>{let a=0;for(let i=1;i<xs.length;i++)a+=(xs[i]-xs[i-1])*(f(xs[i])+f(xs[i-1]))/2;return a;};
const grid=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const kl=(p,q)=>p.reduce((a,v,i)=>a+(v>0?v*Math.log(v/q[i]):0),0);
const gr=st.grades;
near('Grades KL',gr.kl,kl(gr.p,gr.q),1e-12);
near('Grades KL in bits',gr.kl_bits,gr.kl/Math.LN2,1e-12);
gr.ratio.forEach((r,i)=>near(`Grades ratio${i}`,r,gr.p[i]/gr.q[i],1e-12));
near('Plain average of the log ratios',gr.log_ratio_mean,gr.log_ratio.reduce((a,v)=>a+v,0)/3,1e-12);
near('Grades e^KL',gr.geometric_mean_ratio,Math.exp(gr.kl),1e-12);
near('Swapped grades KL(this||last)',gr.swap.kl_pq,kl(gr.swap.p,gr.swap.q),1e-12);
near('Swapped grades KL(last||this)',gr.swap.kl_qp,kl(gr.swap.q,gr.swap.p),1e-12);
check('Swapping the reference changes the KL',Math.abs(gr.swap.kl_pq-gr.swap.kl_qp)>.2);
const lg2=x=>Math.log(x)/Math.LN2,inf=st.info,cl=st.classify,ml=st.mle;
near('Surprise of sun, bits',inf.surprise_bits[0],-lg2(.9),1e-12);
near('Surprise of rain, bits',inf.surprise_bits[1],-lg2(.1),1e-12);
near('Five sunny days: surprises add',inf.five_sun.bits,5*-lg2(.9),1e-12);
near('One rainy day among five',inf.one_rain.bits,4*-lg2(.9)-lg2(.1),1e-12);
near('Forecast entropy',inf.entropy.forecast,.9*-lg2(.9)+.1*-lg2(.1),1e-12);
near('Entropy of 87.5/12.5',inf.entropy.examples['0.875'],.875*-lg2(.875)+.125*3,1e-12);
check('Entropy days: the running average approaches H',Math.abs(inf.entropy.days_running.at(-1)-inf.entropy.forecast)<.02);
const bitsW=st.bits;
check('Huffman codes for p and q',bitsW.right_code.join()==='0,10,11'&&bitsW.wrong_code.join()==='10,11,0');
near('KL in bits from codes',bitsW.kl,bitsW.p.reduce((a,v,i)=>a+v*lg2(v/bitsW.q[i]),0),1e-12);
check('1,000 simulated days: bits per day near H and H(p, q)',Math.abs(bitsW.simulation.bits_p/1000-1.5)<.05&&Math.abs(bitsW.simulation.bits_q/1000-1.75)<.05);
near('Asymmetry: KL(p||sunny town)',inf.kl.asym.fwd,bitsW.p.reduce((a,v,i)=>a+v*lg2(v/inf.kl.asym.q[i]),0),1e-12);
near('Asymmetry: KL(sunny town||p)',inf.kl.asym.rev,inf.kl.asym.q.reduce((a,v,i)=>a+v*lg2(v/bitsW.p[i]),0),1e-12);
near('Classification loss = -ln q(true class)',cl.loss_nats,-Math.log(cl.pred[cl.label]),1e-12);
near('One-hot KL equals the loss',cl.kl_onehot,cl.loss_nats,1e-12);
near('Soft label: cross-entropy = entropy + KL',cl.ce_soft,cl.entropy_soft+cl.kl_soft,1e-12);
near('Soft label entropy',cl.entropy_soft,-cl.teacher.reduce((a,v)=>a+v*Math.log(v),0),1e-12);
const counts=[0,1,2].map(k=>ml.data.filter(x=>x===k).length);
check('Twenty days: 10 sunny, 5 cloudy, 5 rainy',counts.join()==='10,5,5'&&ml.data.length===20);
near('Average surprise under q = cross-entropy with the frequencies',ml.avg_q,ml.p_hat.reduce((a,v,i)=>a-v*lg2(ml.q[i]),0),1e-12);
near('Average surprise under p-hat = entropy of the data',ml.avg_hat,ml.p_hat.reduce((a,v)=>a-v*lg2(v),0),1e-12);
near('Likelihood ratio 2^(20 KL)',ml.ratio,2**(20*(ml.avg_q-ml.avg_hat)),1e-9);
ml.fitted.forEach((v,i)=>near(`Gradient descent finds the frequency ${i}`,v,ml.p_hat[i],1e-3));
const es=st.estimators;
check('Better estimator is never negative and has lower spread',es.rows.every(r=>r.better_negative===0&&r.better_sd<r.plain_sd));
near('Weather sampling at 10,000 days is close to the KL in bits',st.weather_sampling.estimates.at(-1),bitsW.kl,.03);
const cf=st.closed_forms;
near('Closed form: Bernoulli',cf.bernoulli.by_hand,kl([cf.bernoulli.a,1-cf.bernoulli.a],[cf.bernoulli.b,1-cf.bernoulli.b]),1e-12);
near('Closed form: Bernoulli against PyTorch',cf.bernoulli.torch,cf.bernoulli.by_hand,1e-12);
near('Closed form: categorical (grades)',cf.categorical.by_hand,gr.kl,1e-12);
near('Closed form: Gaussian',cf.normal.by_hand,normalKL({mu:cf.normal.mu,sigma:cf.normal.sigma},{mu:cf.normal.m,sigma:cf.normal.s}),1e-12);
near('Closed form: diagonal Gaussian is the sum of 1-D KLs',cf.diagonal.by_hand,cf.diagonal.mu.reduce((a,m,j)=>a+normalKL({mu:m,sigma:cf.diagonal.sigma[j]},{mu:0,sigma:1}),0),1e-12);
near('Closed form: diagonal Gaussian against PyTorch',cf.diagonal.torch,cf.diagonal.by_hand,1e-12);
st.grid.candidates.forEach(([m,s,v])=>near(`Grid candidate N(${m}, ${s}^2)`,v,normalKL({mu:m,sigma:s},target),1e-12));
check('Grid search finds q = p',Math.abs(st.grid.best.mu-1)<1e-9&&Math.abs(st.grid.best.sigma-4)<1e-9);
near('Grades sampling at 10,000 draws is close',st.grades_sampling.estimates.at(-1),gr.kl,.03);
const ev1=st.bnn.evidence_1d;
check('One-unknown evidence: 20 rectangles match a fine grid to 3 decimals',Math.abs(ev1.log_area-ev1.log_area_fine)<5e-3);
const bits=st.bits;
near('Weather entropy (bits)',bits.entropy,-bits.p.reduce((a,p)=>a+p*Math.log2(p),0),1e-12);
near('Weather cross-entropy (bits)',bits.cross_entropy,-bits.p.reduce((a,p,i)=>a+p*Math.log2(bits.q[i]),0),1e-12);
near('Right code average length',bits.p.reduce((a,p,i)=>a+p*bits.right_code[i].length,0),bits.entropy,1e-12);
near('Wrong code average length',bits.p.reduce((a,p,i)=>a+p*bits.wrong_code[i].length,0),bits.cross_entropy,1e-12);
const ck=st.coinkl,bern=(a,b)=>a*Math.log(a/b)+(1-a)*Math.log((1-a)/(1-b));
near('Coin KL(0.8||0.5)',ck.kl,bern(.8,.5),1e-12);
near('Coin KL(0.5||0.8)',ck.reverse,bern(.5,.8),1e-12);
near('Evidence ratio after 10 tosses',ck.x10,Math.exp(10*bern(.8,.5)),1e-9);
near('Evidence ratio after 30 tosses',ck.x30,Math.exp(30*bern(.8,.5)),1e-7);
check('Simulated log ratio grows at the KL rate',Math.abs(ck.simulated_mean_llr_30-30*ck.kl)<.05);
for(const [step,mu,sigma,kl] of st.olive.frames){
 check(`Olive step${step} positive scale`,sigma>0);
 near(`Olive step${step} analytic KL`,normalKL({mu,sigma},target),kl,1e-10);
}
const c3=st.cov3,[s1,s2]=c3.target.sd,rho=c3.target.rho,detP=s1*s1*s2*s2*(1-rho*rho),trPrec=(1/(s1*s1)+1/(s2*s2))/(1-rho*rho);
near('Diagonal width 1 is the conditional width',c3.analytic.diag_sd[0],s1*Math.sqrt(1-rho*rho),1e-12);
near('Diagonal width 2 is the conditional width',c3.analytic.diag_sd[1],s2*Math.sqrt(1-rho*rho),1e-12);
near('Diagonal KL is -1/2 log(1 - rho^2)',c3.analytic.diag_kl,-.5*Math.log(1-rho*rho),1e-12);
near('Isotropic width: s^2 = d / tr(precision)',c3.analytic.iso_sd,Math.sqrt(2/trPrec),1e-12);
near('Isotropic KL',c3.analytic.iso_kl,.5*(Math.log(detP)-2*Math.log(2/trPrec)),1e-12);
const kl2=([m1,m2,a,b,c])=>{const det=a*c-b*b,P=[1/((1-rho*rho)*s1*s1),-rho/((1-rho*rho)*s1*s2),1/((1-rho*rho)*s2*s2)];
 return .5*(P[0]*a+2*P[1]*b+P[2]*c+P[0]*m1*m1+2*P[1]*m1*m2+P[2]*m2*m2-2+Math.log(detP)-Math.log(det));};
for(const kind of ['iso','diag','full'])for(const fr of c3.runs[kind])near(`Covariance fit ${kind} step${fr[0]} KL`,kl2(fr.slice(1,6)),fr[6],1e-9);
check('Isotropic fit stays isotropic',c3.runs.iso.every(fr=>Math.abs(fr[3]-fr[5])<1e-12&&Math.abs(fr[4])<1e-12));
check('Diagonal fit never tilts',c3.runs.diag.every(fr=>Math.abs(fr[4])<1e-12));
near('Isotropic fit reaches the formula',Math.sqrt(c3.runs.iso.at(-1)[3]),c3.analytic.iso_sd,1e-3);
near('Diagonal fit reaches the formula',Math.sqrt(c3.runs.diag.at(-1)[3]),c3.analytic.diag_sd[0],1e-3);
check('Full fit reaches KL 0',c3.runs.full.at(-1)[6]<1e-5);
check('All three start from the same random q',['diag','full'].every(k=>c3.runs[k][0][1]===c3.runs.iso[0][1]&&c3.runs[k][0][2]===c3.runs.iso[0][2]));
const ms=st.mixture_sampling;
check('Mixture draws split roughly in half',Math.abs(ms.counts[0]/ms.draws-.5)<.05&&ms.counts[0]+ms.counts[1]===ms.draws);
const vf=st.bnn.vi_full;
check('Full-covariance VI beats mean-field and stays below the evidence',vf.elbo>st.bnn.vi.elbo&&vf.elbo<st.bnn.evidence.estimate);
near('Full-covariance gap',vf.gap,st.bnn.evidence.estimate-vf.elbo,1e-12);
check('Full-covariance correlation matrix has a unit diagonal',vf.corr.every((r,i)=>Math.abs(r[i]-1)<1e-12));
const narrow=st.fwdrev['0.5'];
near('Reverse KL from the left settles on the 0.7 bump',narrow.revl.at(-1)[3],-Math.log(.7),1e-3);
near('Reverse KL from the right settles on the 0.3 bump',narrow.revr.at(-1)[3],-Math.log(.3),1e-3);
near('Forward KL matches the mixture mean',narrow.fwd.at(-1)[1],-.8,1e-3);
near('Forward KL matches the mixture spread',narrow.fwd.at(-1)[2],Math.sqrt(.25+4-.64),1e-3);
const mixPdf=u=>.5*normalPdf(u,{mu:-.2,sigma:.4})+.5*normalPdf(u,{mu:1,sigma:.5});
const mixKL=(mu,sigma)=>{const g=grid(mu-12*sigma,mu+12*sigma,20001);return trap(g,u=>{const q=normalPdf(u,{mu,sigma});return q>0?q*(Math.log(q)-Math.log(mixPdf(u))):0;});};
for(const run of ['rsample','sample']){
 const last=st.mixfit[run].at(-1);
 near(`Mixture fit (${run}) final quadrature KL`,last[4],mixKL(last[1],last[2]),1e-4);
}
check('rsample fit reaches the best Gaussian (KL 0.0562)',Math.abs(st.mixfit.rsample.at(-1)[4]-.0562)<.002);
check('sample() run stays far from the mixture',st.mixfit.sample.at(-1)[4]>40);
// The tiny Bayesian network: every displayed number recomputed with an independent forward pass.
const bn=st.bnn,lnN=(y,mu,sd)=>-.5*Math.log(2*Math.PI*sd*sd)-(y-mu)**2/(2*sd*sd);
const net=(th,x)=>th[4]*Math.tanh(th[0]*x+th[2])+th[5]*Math.tanh(th[1]*x+th[3])+th[6];
for(const key of ['theta_A','theta_B']){
 const t=bn[key],th=t.theta,pts=bn.x.map((x,i)=>lnN(bn.y[i],net(th,x),bn.noise_sd));
 pts.forEach((v,i)=>near(`${key} likelihood term ${i}`,v,t.log_lik_points[i],1e-9));
 near(`${key} log likelihood`,pts.reduce((a,v)=>a+v,0),t.log_lik,1e-8);
 near(`${key} log prior`,th.reduce((a,v)=>a+lnN(v,0,1),0),t.log_prior,1e-10);
 near(`${key} log joint`,t.log_lik+t.log_prior,t.log_joint,1e-8);
}
check('theta_A explains the data far better than theta_B',bn.theta_A.log_lik-bn.theta_B.log_lik>1000);
bn.grid.forEach((x,i)=>{near(`f_A at grid point ${i}`,net(bn.theta_A.theta,x),bn.f_A[i],1e-9);near(`f_B at grid point ${i}`,net(bn.theta_B.theta,x),bn.f_B[i],1e-9);});
bn.prior_thetas.forEach((th,k)=>bn.grid.forEach((x,i)=>near(`prior draw ${k} at grid point ${i}`,net(th,x),bn.prior_draws[k][i],1e-9)));
const pp=bn.prior_predictive;
pp.sd.forEach((v,i)=>near(`prior predictive sd ${i} = spread and noise combined`,v,Math.hypot(pp.sd_f[i],bn.noise_sd),1e-9));
check('The prior predictive is wide everywhere (sd > 1)',pp.sd.every(v=>v>1));
check('The prior predictive is wider than the posterior predictive everywhere',pp.sd.every((v,i)=>v>bn.hmc.sd[i]));
// Training runs behind θ_A: loss = squared error / (2·0.1²) + ½‖θ‖², which is −log p(𝒟 | θ) − log p(θ) without constants.
const mp=bn.map,thA=bn.theta_A.theta,sse=th=>bn.x.reduce((a,x,i)=>a+(bn.y[i]-net(th,x))**2,0);
near('Training loss of θ_A, recomputed',sse(thA)/(2*bn.noise_sd**2)+thA.reduce((a,v)=>a+v*v,0)/2,mp.objective_A,1e-9);
near('Training loss = −log joint without its constants',-bn.theta_A.log_joint-17*.5*Math.log(2*Math.PI*bn.noise_sd**2)-7*.5*Math.log(2*Math.PI),mp.objective_A,1e-8);
check('θ_A came from the start with the lowest final loss',mp.objectives.every(o=>o.at(-1)>=mp.objectives[mp.best_seed].at(-1)-1e-9));
near('The last training frame is θ_A',mp.frames.at(-1)[1],mp.objective_A,1e-3);
mp.frames.at(-1)[2].forEach((v,i)=>near(`last training frame curve ${i}`,v,bn.f_A[i],1e-4));
check('Training starts far from the data and ends near it',mp.frames[0][1]>10*mp.objective_A);
const ev=bn.evidence,finals=ev.runs.map(r=>r.estimates.at(-1));
near('Evidence estimate is the mean of four seeds',finals.reduce((a,v)=>a+v,0)/finals.length,ev.estimate,1e-12);
check('Four 10-million-draw evidence estimates agree within 0.3',Math.max(...finals)-Math.min(...finals)<.3);
check('Ten prior draws are off by over 100 nats',ev.runs.every(r=>r.estimates[0]<ev.estimate-100));
check('Fewer than 1,000 effective draws out of 10 million',ev.runs.every(r=>r.effective_draws<1000));
const hm=bn.hmc;
check('HMC acceptance between 0.6 and 0.99',hm.acceptance.every(a=>a>.6&&a<.99));
hm.sd.forEach((v,i)=>near(`HMC predictive sd ${i} = spread and noise combined`,v,Math.hypot(hm.sd_f[i],bn.noise_sd),1e-9));
check('HMC bands widen beyond the data',hm.sd[0]>1.5*Math.min(...hm.sd)&&hm.sd.at(-1)>1.3*Math.min(...hm.sd));
const vq=bn.vi;
near('VI KL to the prior, closed form',vq.mu.reduce((a,m,j)=>a+normalKL({mu:m,sigma:vq.sigma[j]},{mu:0,sigma:1}),0),vq.kl_prior,1e-9);
near('VI ELBO = expected log likelihood − KL to the prior',vq.expected_log_lik-vq.kl_prior,vq.elbo,1e-9);
near('VI gap to the evidence estimate',ev.estimate-vq.elbo,vq.gap,1e-9);
check('VI ELBO below the evidence estimate',vq.elbo<ev.estimate);
check('VI kept the best of five restarts',Math.abs(Math.max(...vq.restart_elbos)-vq.elbo)<1e-9);
check('VI bands narrower than the reference beyond the data',vq.sd[0]<hm.sd[0]&&vq.sd.at(-1)<hm.sd.at(-1));
check('VI switched the first hidden unit off',Math.abs(vq.mu[4])<.1&&vq.sigma[4]<.1&&vq.sigma[0]>.5&&vq.sigma[2]>.5);
check('VI frames start far below and end near the final ELBO',vq.frames[0][1]<-500&&Math.abs(vq.frames.at(-1)[1]-vq.elbo)<.1);
near('Three-ways target mean',st.three.true_mean,-.8,1e-12);
check('Three-ways MCMC mean near the target mean',Math.abs(st.three.mcmc.mean+.8)<.1);
check('Three-ways Laplace sits on the big bump',Math.abs(st.three.laplace[0]+2)<.01&&Math.abs(st.three.laplace[1]-1)<.01);
const numeric=records.filter(r=>r.kind==='numerical');
const report={
 status:'passed',assertions:records.length,numericalComparisons:numeric.length,
 maximumAbsoluteError:Math.max(...numeric.map(r=>r.error)),
 source:'Independent JavaScript formulae versus executed float64 PyTorch notebook; no new training in this check.',
 tolerances:{normal:2e-12,beta:7e-11,densityBeta:2e-10,finiteDifference:1e-8,quadrature:1e-8},
 checks:records,
};
fs.mkdirSync(path.join(root,'qa'),{recursive:true});
fs.writeFileSync(path.join(root,'qa/science-checks.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({status:report.status,assertions:report.assertions,
 numericalComparisons:report.numericalComparisons,maximumAbsoluteError:report.maximumAbsoluteError},null,2));
