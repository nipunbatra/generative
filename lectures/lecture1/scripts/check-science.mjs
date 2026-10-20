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
near('Conjugate evidence on the wall slide',st.wall.conj_log_evidence,coinLogEvidence(prior,data),5e-11);
const trap=(xs,f)=>{let a=0;for(let i=1;i<xs.length;i++)a+=(xs[i]-xs[i-1])*(f(xs[i])+f(xs[i-1]))/2;return a;};
const grid=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const logSig=w=>w>=0?-Math.log1p(Math.exp(-w)):w-Math.log1p(Math.exp(w));
const logitJoint=w=>Math.log(normalPdf(w,{mu:0,sigma:1}))+17*logSig(w)+13*logSig(-w);
const wGrid=grid(-10,10,40001),wMax=Math.max(...wGrid.map(logitJoint));
const logitEvidence=wMax+Math.log(trap(wGrid,w=>Math.exp(logitJoint(w)-wMax)));
near('Logit-prior coin evidence (independent quadrature)',st.wall.logit_log_evidence_grid,logitEvidence,1e-6);
check('Logit-prior MC evidence close to the grid',Math.abs(st.wall.logit_log_evidence_mc-logitEvidence)<.02);
const fr=st.fruit;
near('Fruit KL',fr.kl,fr.p.reduce((a,p,i)=>a+p*Math.log(p/fr.q[i]),0),1e-12);
fr.ratio.forEach((r,i)=>near(`Fruit ratio${i}`,r,fr.p[i]/fr.q[i],1e-12));
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
near('2-D diagonal width',st.twod.analytic_sd,Math.sqrt(1-st.twod.rho**2),1e-12);
near('2-D diagonal KL',st.twod.analytic_kl,-.5*Math.log(1-st.twod.rho**2),1e-12);
st.twod.diag_sd.forEach((v,i)=>near(`2-D fitted width${i}`,v,st.twod.analytic_sd,2e-3));
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
const lv=st.logitvi.final,qLogit={mu:lv.m,sigma:lv.s};
const elboLogit=trap(grid(lv.m-12*lv.s,lv.m+12*lv.s,20001),w=>{const q=normalPdf(w,qLogit);return q*(logitJoint(w)-Math.log(q));});
near('Logit VI exact ELBO (independent quadrature)',lv.exact_elbo,elboLogit,1e-6);
check('Logit VI ELBO sits below the evidence',lv.gap>0&&lv.gap<.01);
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
