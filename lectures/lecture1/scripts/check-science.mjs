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
