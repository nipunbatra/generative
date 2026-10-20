import {m,eq,at,only,cols,code,image,table,flow,plot,palette} from '../shared/author.js';

const d=await fetch(new URL('./data/science.json',import.meta.url)).then(r=>r.json());
const R=String.raw;
const f=(v,n=4)=>Number(v).toFixed(n).replace('-','−');
const p=s=>`<p class="lead">${s}</p>`;
const small=s=>`<p class="caption">${s}</p>`;
const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
const n=d.normal, c=d.coin, g=d.gradients, mc=d.monte_carlo;
export const meta={
 id:'lecture1',number:1,title:'KL and variational inference',
 subtitle:'Fit an approximation to an unknown posterior',minutes:70,
 notebook:'lecture1-variational-inference.ipynb',
 sections:[
  {id:'question',title:'An inference question',minutes:4},
  {id:'compare',title:'Which distribution is closer?',minutes:9},
  {id:'optimise',title:'Turn comparison into optimisation',minutes:10},
  {id:'monte-carlo',title:'Estimate an unavailable formula',minutes:7},
  {id:'gradients',title:'Differentiate through a draw',minutes:7},
  {id:'elbo',title:'An objective without the posterior',minutes:13},
  {id:'coin',title:'Fit and check the coin',minutes:13},
  {id:'retrieve',title:'Retrieve and transfer',minutes:7},
  {id:'optional',title:'Optional depth',minutes:0},
 ],
};
export const slides=[];
function add(id,section,title,builds,body,ask,expected,point,transition,extra={}){
 slides.push({id,section,title,builds,body,prompts:ask,
  notes:{ask,expected,point,transition,misconception:extra.misconception||'',demo:extra.demo||''},
  source:extra.source||'Computed teaching example · natural logarithms · KL in nats',
  optional:extra.optional||false});
}
function normalPlot(keys=['q1'],extra={}){
 const a=d.plots.normal;
 const series=[{label:'p · fixed target',role:'p',x:a.x,y:a.p}];
 keys.forEach(k=>series.push({label:k==='updated'?'q after':k==='q1'?'q₁':k==='q2'?'q₂':`q · step ${k}`,
  role:'q',x:a.x,y:a[k]||a.snapshots[k],dashed:k==='q2'}));
 return plot(series,{width:640,height:340,xMin:-11,xMax:13,yMin:0,yMax:.23,xLabel:'u',yLabel:'density',
  ticksX:[-10,-5,0,5,10],ticksY:[0,.1,.2],...extra});
}
function mixPlot(keys=['q1','q2']){
 const a=d.plots.mixture;
 return plot([{label:'p · PML mixture',role:'p',x:a.x,y:a.p},...keys.map(k=>({
  label:k==='q1'?'q₁ · scale 1':'q₂ · scale 4.5',role:'q',x:a.x,y:a[k],dashed:k==='q2'}))],
  {xMin:-12,xMax:18,yMin:0,yMax:.62,xLabel:'u',yLabel:'density',ticksX:[-10,0,10],ticksY:[0,.3,.6]});
}
function coinPlot(index=0,snapshot=null,extra={}){
 const a=d.plots.coin;
 return plot([{label:'known posterior',role:'p',x:a.x,y:a.posterior},
  {label:snapshot===null?`q · Beta(${c.candidates[index].a},${c.candidates[index].b})`:`q · step ${snapshot}`,
   role:'q',x:a.x,y:snapshot===null?a.candidates[index]:a.snapshots[snapshot],dashed:true}],
  {xMin:0,xMax:1,yMin:0,yMax:6.5,xLabel:'r · chance of heads',yLabel:'density',ticksX:[0,.5,1],ticksY:[0,3,6],...extra});
}
function tosses(){
 return `<div class="coin-tosses" style="display:grid;grid-template-columns:repeat(15,1fr);gap:8px;margin:22px 0">${[...c.ordered_sequence].map(t=>`<div style="border:1px solid #d7dce1;border-radius:50%;height:46px;display:grid;place-items:center;color:${t==='H'?palette.data:palette.ink};font-size:26px">${t}</div>`).join('')}</div>`;
}
const KL=R`\KL(q_\psi(u)\|p(u\mid\mathcal D))`;
const Q=R`\qcol{q_\psi(u)}`, J=R`\pcol{p(\mathcal D,u)}`, E=R`\dcol{\log p(\mathcal D)}`;
const EX=R`\E_{u\sim q_\psi}`;

add('coin-question','question','What would you like to know about this coin?',2,
 stack(tosses(),at(2,p(`${m(R`\ncol r`)} is the unknown chance of heads.`))),
 ['Thirty observed tosses. What is still unknown?','Do we want one guess, or uncertainty about the guess?'],
 ['The coin’s probability of heads. The observations themselves are known.','A distribution over plausible values of r describes both a centre and uncertainty.'],
 'Point to individual observed H/T values before introducing r.',
 'The observed sequence is fixed. We will ask how it changes our belief about r.',
 {source:'One chosen ordered sequence · 17 H and 13 T · recorded in science.json'});
add('data-and-unknown','question','The data are fixed; the unknown is not a data point',2,
 cols(stack(p(`Observed ${m(R`\dcol{\mathcal D}`)}`),eq(R`17\text{ heads},\quad13\text{ tails}`)),
 stack(p(`Unknown ${m(R`\ncol r`)}`),eq(R`0<r<1`),at(2,p('Inference describes plausible values of r.')))),
 ['Which object is observed? Which object will have a posterior?','Does inferring r change any of the thirty tosses?'],
 ['The ordered sequence 𝒟 is observed. The unknown r has the posterior.','No. We update our description of r while holding the data fixed.'],
 'Keep the blue observed data separate from the amber unknown.',
 'We already know Bayes’ rule. Use it to name what we want.');
add('prior-before-data','question','A prior is a distribution over the unknown',2,
 stack(eq(R`r\sim\pcol{\mathrm{Beta}(5,3)}`),at(2,eq(R`p(r\mid\mathcal D)\ \propto\ p(\mathcal D\mid r)\,p(r)`))),
 ['What does the Beta prior describe?','What two ingredients turn it into a posterior shape?'],
 ['Uncertainty about the single unknown probability r. It is not a distribution of toss outcomes.','The likelihood of the observed sequence and the prior density.'],
 'The prior’s argument is r. The likelihood treats the already observed sequence as fixed.',
 'The proportionality hides a normalising constant.');
add('bayes-recall','question','What normalises the posterior?',2,
 stack(eq(R`p(r\mid\mathcal D)=\frac{p(\mathcal D\mid r)\,p(r)}{p(\mathcal D)}`),
 at(2,p('This coin is solvable. We will use its answer to check a more general method.'))),
 ['What is the denominator doing?','Why practise an approximate method on a solvable problem?'],
 ['It makes the posterior integrate to one. It is the evidence.','A known answer lets us check the implementation and understand the approximation.'],
 'Do not launch an extended Bayes recap; these students know the rule.',
 'First learn how to compare and improve a distribution when the target is available.');

add('compare','compare','Which Gaussian is closer to the target?',2,
 cols(stack(only('1',normalPlot(['q1'])),only('2',normalPlot(['q1','q2']))),
 stack(eq(R`p=\N(1,4^2)`),eq(R`q_1=\N(0,2^2)`),at(2,eq(R`q_2=\N(1,3^2)`)),
 small('Normal(loc=μ, scale=σ). The mathematical second argument is variance.')),'1.45fr 1fr'),
 ['Describe the centre and width of q₁.','Vote for q₁ or q₂ before calculating.'],
 ['q₁ is left of p and narrower.','q₂ shares p’s centre and has a more similar width. We need a score to make “closer” precise.'],
 'Curves use identical axes and labelled solid/dashed strokes.',
 'A score should compare the whole distribution, not only the means.');
add('one-location','compare','One density comparison is not the whole story',2,
 cols(normalPlot(['q1']),stack(eq(R`u=0`),eq(R`\log q_1(0)-\log p(0)`),
 at(2,p('This compares one location. What about the rest of the curve?'))),'1.4fr 1fr'),
 ['At u=0, which density is larger?','Could this one location settle which whole distribution is closer?'],
 ['q₁ is more concentrated and has the larger density here.','No. The relationship changes across the support. We need an average.'],
 'Point at the centres on the same axes. A density is not a point probability.',
 'Choose the distribution that supplies the weights for our average.');
add('kl-average','compare','Average the log ratio under q',2,
 stack(eq(R`\KL(q\|p)=\E_{u\sim\qcol q}[\log\qcol{q(u)}-\log\pcol{p(u)}]`),
 at(2,p('The first argument supplies the weighting distribution.'))),
 ['Where does this expectation get its values of u?','What does the order q, then p tell us?'],
 ['From q. Equivalently, q weights the integral across u.','The average is under q. Reversing the arguments changes that weighting.'],
 'Connect the purple q outside the bracket to the purple density inside it.',
 'Check that swapping the arguments really can change the number.');
add('kl-direction','compare','Reversing KL changes the question',2,
 cols(stack(eq(R`\KL(q_1\|p)`),at(2,eq(R`\approx ${n.candidates.q1.kl.toFixed(4)}`))),
 stack(eq(R`\KL(p\|q_1)`),at(2,eq(R`\approx ${n.candidates.q1.reverse_kl.toFixed(4)}`)))),
 ['Are these two averages weighted in the same way?','What does the numerical difference rule out?'],
 ['No. One averages under q₁, the other under p.','KL is not a symmetric distance. Always state the direction.'],
 'Both values use the same two Normal distributions. Only their order changed.',
 'Keep the direction fixed as we fit q to p.');
add('local-versus-average','compare','Nonnegative KL does not make every term positive',2,
 stack(eq(R`\log q(u)-\log p(u)\quad\text{can be negative}`),
 at(2,eq(R`\KL(q\|p)\geq0,\qquad\KL(p\|p)=0`))),
 ['Can p(u) exceed q(u) at a particular u?','Which nonnegative statement applies to the whole average?'],
 ['Yes. Then that local log ratio is negative.','Exact KL is nonnegative. A matched distribution has zero KL.'],
 'Separate a pointwise term from its expectation. This distinction returns when sampling.',
 'For two Gaussians, the entire expectation has a closed form.',
 {misconception:'A probability density may exceed one; a log-density ratio may be negative.'});
add('gaussian-formula','compare','For two Gaussians, evaluate the expectation exactly',2,
 stack(eq(R`q=\N(\mu,\sigma^2),\qquad p=\N(m,s^2)`),
 at(2,eq(R`\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12`))),
 ['Which parameters belong to q, and which stay fixed in p?','Where do width and centre mismatch enter?'],
 ['μ and σ belong to q; m and s belong to the target.','The log scale ratio and σ² measure width effects; the squared shift measures centre mismatch.'],
 'Every scale symbol is a standard deviation; every squared scale is a variance.',
 'Substitute our two candidates before revealing the scores.');
add('gaussian-substitution','compare','Substitute first; then compare the score',2,
 cols(stack(eq(R`q_1:\quad\log2+\frac{4+1}{32}-\frac12`),
 at(2,eq(R`\KL(q_1\|p)\approx${n.candidates.q1.kl.toFixed(4)}`))),
 stack(eq(R`q_2:\quad\log\frac43+\frac{9+0}{32}-\frac12`),
 at(2,eq(R`\KL(q_2\|p)\approx${n.candidates.q2.kl.toFixed(4)}`)))),
 ['Which candidate has no squared centre-shift term?','Does the score support the earlier prediction?'],
 ['q₂, because its mean is 1, the target mean.','Yes. q₂ has the smaller KL, approximately 0.0689 versus 0.3494 nats.'],
 'The denominator is 2s²=2×16, not twice the standard deviation.',
 'The score tells us which candidate wins. Can it tell us how to improve one?');
add('score-to-search','compare','A comparison can become a search',2,
 stack(p('Keep the target fixed. Change q’s parameters.'),at(2,eq(R`\psi^*=\arg\min_\psi\KL(q_\psi\|p)`))),
 ['What should the optimiser change?','What does this argmin select?'],
 ['The parameters of q, while the target p stays fixed.','The best member of the chosen family under this directional KL objective.'],
 'ψ is a parameter collection, not a sampled value u.',
 'Before differentiating, expose the Gaussian moment that makes the formula work.');

add('gaussian-moment','optimise','A moment identity explains the squared shift',2,
 stack(eq(R`u-m=(u-\mu)+(\mu-m)`),
 at(2,eq(R`\E_q[(u-m)^2]=\E_q[(u-\mu)^2]+(\mu-m)^2`))),
 ['When expanding the square, which cross-term should vanish?','Why does the remaining random term become σ²?'],
 ['The cross-term contains E_q[u−μ], which is zero.','By definition, the variance of q is E_q[(u−μ)²]=σ².'],
 'The mean of q is μ, even though the target has mean m.',
 'Insert the variance into the KL and isolate one movable parameter.');
add('isolate-mean','optimise','Move the centre while keeping the width fixed',2,
 stack(eq(R`\mu=0,\quad\sigma=2,\qquad p=\N(1,4^2)`),
 at(2,eq(R`L(\mu)=\underbrace{\log2+\frac4{32}-\frac12}_{\text{constant}}+\frac{(\mu-1)^2}{32}`))),
 ['If the width cannot change, which way should μ move?','Which term changes when μ changes?'],
 ['Right, towards the target mean 1.','Only the squared-shift term. Everything in the bracket is constant for this experiment.'],
 'The experiment deliberately restricts q. Do not imply that σ is being learned yet.',
 'Differentiate only the term that depends on μ.');
add('gradient','optimise','A negative derivative produces a rightward descent step',3,
 stack(eq(R`\frac{\partial L}{\partial\mu}=\frac{2(\mu-1)}{32}`),
 at(2,eq(R`=\frac{\mu-1}{16}`)),at(3,eq(R`\left.\frac{\partial L}{\partial\mu}\right|_{\mu=0}=-0.0625`))),
 ['Differentiate the squared shift.','What simplifies in the derivative?','Does a negative derivative mean move left under gradient descent?'],
 ['2(μ−1)/32.','The factor of two reduces the denominator to 16.','No. Gradient descent subtracts the derivative, so a negative derivative moves μ right.'],
 'Distinguish the derivative’s sign from the update’s direction.',
 'Use one explicit learning rate and perform the arithmetic.');
add('one-update','optimise','One literal update',3,
 stack(eq(R`\mu^+=\mu-\eta\frac{\partial L}{\partial\mu}`),
 at(2,eq(R`\eta=4,\qquad\mu^+=0-4(-0.0625)`)),at(3,eq(R`\mu^+=0.25`))),
 ['What does gradient descent subtract?','Predict the new mean.','Which parameter has not changed?'],
 ['The learning rate times the derivative.','The new mean is 0.25.','The standard deviation remains 2, and p remains fixed.'],
 'η=4 is a chosen input for this inspectable step, not a general recommended learning rate.',
 'Recompute the score; a moving curve alone is not evidence of improvement.');
add('recompute-loss','optimise','The new curve earns a new loss evaluation',2,
 cols(normalPlot(['q1','updated']),stack(eq(R`\mu:0\longrightarrow0.25`),
 at(2,table(['KL before','KL after'],[[f(n.one_step.kl_before),f(n.one_step.kl_after)]])),
 small('σ=2 throughout. Same target and axes.')),'1.45fr 1fr'),
 ['What changed in the curve?','Did the computed objective improve?'],
 ['Its position changed. Its width did not.','Yes. The KL falls from approximately 0.3494 to 0.3357.'],
 'Read the before/after values only after students inspect the shift.',
 'Ask what the restricted optimisation can achieve if repeated.',
 {demo:'Notebook section 2, first code cell: run the literal update and inspect μ.grad, μ_after and both KL values.'});
add('family-restriction','optimise','Perfect mean fitting still leaves a gap',2,
 stack(eq(R`\mu=1,\quad\sigma=2\ne4`),at(2,eq(R`\min_\mu\KL(q_\mu\|p)\approx${n.one_step.fixed_scale_best_kl.toFixed(4)}>0`))),
 ['What is the best mean if σ must stay 2?','Why is the minimum KL still positive?'],
 ['μ=1.','The target standard deviation is 4. Mean-only fitting cannot repair the width mismatch.'],
 'This is a family restriction, not an optimiser failure.',
 'To fit both parameters, the optimiser needs a valid representation of scale.');
add('positive-scale','optimise','A standard deviation must remain positive',2,
 stack(eq(R`\rho\in\mathbb R,\qquad\sigma=\exp(\rho)>0`),
 at(2,code('mu = Parameter(tensor(0.))\nrho = Parameter(tensor(log(2.)))\nq = Normal(mu, rho.exp())'))),
 ['Why not update an unconstrained σ directly?','What does ρ represent here?'],
 ['An arbitrary update could make σ nonpositive, which is invalid.','Log standard deviation. It is not log variance.'],
 'The optimiser stores μ and ρ. The distribution receives μ and exp(ρ).',
 'The optimisation loop must rebuild q and clear old gradients.');
add('optimisation-loop','optimise','Rebuild, score, differentiate, update',2,
 stack(code('q = Normal(mu, rho.exp())\nloss = kl_divergence(q, p)\noptimiser.zero_grad()\nloss.backward()\noptimiser.step()'),
 at(2,p('Repeat with the updated parameters.'))),
 ['Why build q again on the next iteration?','Why clear gradients before backward?'],
 ['Its parameters have changed, so the density and computation graph must use the current values.','PyTorch accumulates gradients unless they are cleared.'],
 'This is the executable five-line core of the notebook loop.',
 'Inspect a real saved trajectory instead of waiting for training in class.',
 {demo:'Notebook section 2, second code cell: replay saved snapshots, then inspect one update. Adam optimises analytic KL with lr=0.08.'});
add('measured-gaussian-fit','optimise','Letting both parameters move removes the restriction',3,
 stack(only('1',normalPlot(['0'],{width:1000})),only('2',normalPlot(['30'],{width:1000})),only('3',normalPlot(['150'],{width:1000}))),
 ['What can this larger family now match?','At step 30, what is still visibly different?','How close is the final measured fit?'],
 ['Both the target centre and width.','Inspect the measured curve rather than assuming monotone visual improvement.','At step 150, μ≈0.9996, σ≈4.0000 and KL≈4.36×10⁻⁹; numerical agreement is excellent.'],
 'The three states are saved steps 0, 30 and 150 from actual analytic-KL optimisation.',
 'A useful score may exist even when the library has no analytic formula for it.',
 {source:'Measured deterministic optimisation · Adam lr=0.08 · 150 steps · float64'});
add('analytic-loss-trace','optimise','A loss trace answers a specific question',2,
 stack(plot([{label:'analytic KL',role:'loss',x:n.training.trace.map(r=>r.step),y:n.training.trace.map(r=>r.kl)}],
 {xMin:0,xMax:150,yMin:0,yMax:.36,xLabel:'optimiser step',yLabel:'KL (nats)',ticksX:[0,50,100,150],ticksY:[0,.15,.3]}),
 at(2,small('This trace evaluates the exact Normal formula. It is not a Monte Carlo estimate.'))),
 ['What is being measured on the vertical axis?','Would the same smooth trace be guaranteed for a sampled objective?'],
 ['Analytic directional KL at the current q parameters.','No. Monte Carlo estimates introduce sampling variation.'],
 'Keep the evidence label tied to the actual experiment: one toy, analytic loss, recorded steps.',
 'Now replace the target with the PML mixture and retain the definition of KL.');

add('mixture-target','monte-carlo','Replace the target; keep the definition',2,
 stack(eq(R`p(u)=\tfrac12\N(u;-0.2,0.4^2)+\tfrac12\N(u;1,0.5^2)`),
 at(2,p('We can evaluate this density at a proposed u.'))),
 ['What makes this a normalised density?','Does evaluating p(u) require a closed-form KL?'],
 ['Each component integrates to one and the nonnegative weights sum to one.','No. The mixture density can be evaluated directly.'],
 'These are the original PML mixture parameters, not a newly invented target.',
 'Compare the same two Gaussian candidates used in the historical notebook.',
 {source:'PML2023 notebook cells25–30 · original mixture and candidate parameters'});
add('mixture-candidates','monte-carlo','Which candidate wastes more mass far from the mixture?',2,
 stack(mixPlot(),at(2,small('Both candidates have mean 3. Their standard deviations are 1 and 4.5.'))),
 ['Where does p assign most of its density?','Which candidate draws more often from far-away regions?'],
 ['Near the two mixture components, centred at −0.2 and 1.','The very broad q₂ draws more often from far away. Predict a larger KL(q₂‖p), then check.'],
 'The common wide x-axis keeps the broad candidate visible. Do not zoom each curve independently.',
 'Ask the library for the analytic KL and inspect the actual result.');
add('unsupported-kl','monte-carlo','A missing library formula is not an undefined quantity',2,
 stack(code('kl_divergence(q, mixture)'),at(2,code('NotImplementedError:\nKL not implemented for this distribution pair'))),
 ['What would this call need to know?','What does this error establish, and what does it not establish?'],
 ['An implemented formula or algorithm for this specific distribution pair.','It establishes that this runtime does not implement the pair. It does not show that the KL is undefined.'],
 'The exact exception is saved in science.json; the slide uses a shortened readable message.',
 'Return to the expectation definition instead of abandoning the score.',
 {source:`Observed PyTorch ${d.provenance.torch} runtime behaviour · full error saved in science.json`});
add('sample-average','monte-carlo','Draw from q, score both densities, average',3,
 stack(eq(R`u_1,\ldots,u_S\sim q`),
 at(2,eq(R`\ell_i=\log q(u_i)-\log p(u_i)`)),
 at(3,eq(R`\widehat{\KL}_S=\frac1S\sum_{i=1}^{S}\ell_i`))),
 ['Which distribution supplies the draws?','What can we compute at each draw?','Which dimension is averaged?'],
 ['q, the first argument of KL(q‖p).','The log density of q minus the log density of p at the same draw.','The S independent Monte Carlo draws.'],
 'The hat denotes an estimate. The log ratio compares the same u_i in both densities.',
 'Write the estimator in three lines and validate it on a known case.');
add('mc-code-check','monte-carlo','Validate the estimator before using the unfamiliar target',2,
 cols(code('u = q.rsample((S,))\nlog_q = q.log_prob(u)\nlog_p = p.log_prob(u)\nestimate = (log_q-log_p).mean()'),
 stack(eq(R`q=\N(1,3^2),\quad p=\N(1,4^2)`),
 at(2,table(['Exact KL','One MC estimate'],[[f(mc.exact_kl),f(mc.repeats[3].estimates[0])]])),
 small('S=10,000 · seed 1000'))),
 ['What result can we check without Monte Carlo?','Should a finite estimate equal the exact answer digit for digit?'],
 ['The analytic Normal KL, approximately 0.0689.','No. The estimate varies with the sampled draws.'],
 'The notebook’s evaluation-only helper uses sample(); this differentiable training form uses rsample(), explained next.',
 'Repeat the estimator while holding both distributions fixed.',
 {demo:'Notebook section3: run the known-Gaussian repeated-seed experiment. Do not change q or p.'});
add('mc-variability','monte-carlo','More draws reduce variability, not every individual error',2,
 stack(image('figures/mc-variability.svg','Forty measured KL estimates for each of four sample counts.',
 'Same p and q · 40 seeds per sample count · exact KL is the horizontal reference'),
 at(2,small('A new seed changes the estimate. It does not change the exact KL.'))),
 ['What changed between these four groups?','Must every larger-S estimate be closer than every smaller-S estimate?'],
 ['Only the number of Monte Carlo draws. Each group also repeats 40 specified seeds.','No. Sampling variance shrinks, but individual errors need not decrease monotonically.'],
 'Point to the spread across seeds and the unchanging exact reference line.',
 'Examine one small sample whose estimate is negative.',
 {source:'Measured estimates · seeds1000–1039 · S=10,100,1000,10000 · float64'});
add('negative-estimate','monte-carlo','Five draws can give a negative KL estimate',2,
 stack(table(['uᵢ','log q(uᵢ) − log p(uᵢ)'],mc.negative_example.draws.map((x,i)=>[f(x,3),f(mc.negative_example.terms[i],3)])),
 at(2,eq(R`\widehat{\KL}_5\approx${mc.negative_example.estimate.toFixed(4)},\qquad\KL(q\|p)\approx${mc.exact_kl.toFixed(4)}`))),
 ['Some rows are negative. Could their average be negative?','Does this contradict nonnegative exact KL?'],
 ['Yes; this particular five-draw average is approximately −0.0933.','No. The finite-sample estimate is not the exact expectation.'],
 'Seed0 was deliberately selected as the first negative example in a seed search, not as a performance result.',
 'Use many draws for the mixture comparison, and label the result as an estimate.',
 {source:'Intentional counterexample · first negative seed in0–999 is0 · five q draws'});
add('mixture-results','monte-carlo','The estimated score separates these candidates clearly',2,
 stack(table(['Candidate','MC estimate ± one SE','Numerical integral'],d.mixture.candidates.map(r=>[
  `${r.id}: σ=${r.sigma}`,`${f(r.mc,3)} ± ${f(r.standard_error,3)}`,f(r.quadrature,3)])),
 at(2,p('q₁ has the smaller directional KL. The estimate is not an analytic formula.'))),
 ['How large is the difference relative to the sampling uncertainty?','What role does the numerical integral play?'],
 ['The candidate difference is far larger than the reported standard errors.','It is an independent numerical check in this one-dimensional toy, not an available high-dimensional solution in general.'],
 'Both columns are numerical evaluations; neither should be labelled exact analytical mixture KL.',
 'An estimator is useful for scoring. Can it also provide the derivative we need?',
 {source:'100,000 q draws · seeds402/403 · separate wide-domain numerical integration'});

function drawGraph(){
 return flow([
 {id:'params',x:20,y:100,w:185,h:85,label:'μ, σ',sub:'parameters',role:'q'},
 {id:'q',x:300,y:85,w:170,h:115,label:'Normal',sub:'distribution q',role:'q',shape:'circle'},
 {id:'u',x:570,y:100,w:150,h:85,label:'u',sub:'sampled value',role:'noise'},
 {id:'loss',x:820,y:100,w:150,h:85,label:'loss',role:'loss'},
 ],[{from:'params',to:'q'},{from:'q',to:'u',label:'sample'},{from:'u',to:'loss'}],
 {width:1000,height:280,label:'Computation view: parameters define q, q supplies a draw, and the draw enters a loss.'});
}
add('sample-path','gradients','A random draw is not automatically a differentiable path',2,
 stack(drawGraph(),at(2,code('u = Normal(mu, sigma).sample()\nu.requires_grad   # False'))),
 ['Which connection would the derivative need to follow?','Does this sample retain that path in PyTorch?'],
 ['From the scalar loss through u back to the distribution parameters.','No. Normal.sample() returns a draw without that pathwise autograd connection.'],
 'This is a computation view. The word sample identifies the random operation.',
 'Check the actual runtime behaviour before proposing a remedy.');
add('sample-backward','gradients','Ask the sample itself for a gradient',2,
 stack(code('u = Normal(mu, sigma).sample()\nu.backward()'),
 at(2,p('RuntimeError: this tensor does not require grad.'))),
 ['What do you predict before running backward?','What has this failure shown?'],
 ['Backward on this sampled scalar will fail because it has no autograd path.','It identifies the absent path from this sample to μ and σ; it does not prove all stochastic gradient methods are impossible.'],
 'The notebook catches this expected error so Run All still completes.',
 'A more subtle failure occurs when the loss has another explicit parameter dependence.',
 {demo:'Notebook section4: inspect sample.requires_grad and the caught sample.backward() exception.'});
add('a-gradient-is-not-enough','gradients','A backward call can run and still miss the intended path',2,
 stack(code('u = q.sample((S,))\nloss = (q.log_prob(u) - p.log_prob(u)).mean()\nloss.backward()'),
 at(2,p('log q still contains the parameters explicitly. The missing sample path has not been repaired.'))),
 ['Can log q(u) depend on μ and σ even if u is detached?','Would a nonempty gradient tensor prove this is the desired KL derivative?'],
 ['Yes. The density formula itself contains μ and σ.','No. The derivative also needs the effect of changing the sampling distribution, which this naive construction omits.'],
 'Do not claim that backward must fail for every loss containing sample().',
 'Compare the resulting derivatives with the analytic reference using identical draws.');
add('gradient-comparison','gradients','Check the derivative, not just the existence of a tensor',2,
 stack(table(['Derivative route','∂L/∂μ','∂L/∂σ'],[
 ['Naive detached samples',...g.detached_sample_grads.map(x=>f(x,4))],
 ['Pathwise samples',...g.pathwise_grads.map(x=>f(x,4))],
 ['Analytic KL',...g.analytic_grads.map(x=>f(x,4))],
 ]),at(2,small('Same 50,000 chosen noise values · seed13 · q=Normal(0,2), p=Normal(1,4)'))),
 ['Which row agrees with the derivative of the actual KL?','Is Monte Carlo agreement expected to be exact?'],
 ['The pathwise row closely agrees; the naive detached row misses the sampling-distribution effect.','No. This is a finite Monte Carlo derivative, so it has sampling error.'],
 'A properly constructed score-function estimator is another valid approach; this comparison concerns the naive detached construction.',
 'Separate the randomness from the deterministic transformation.');
function reparamGraph(){
 return flow([
 {id:'mu',x:30,y:20,w:150,h:70,label:'μ',role:'q'},
 {id:'sig',x:30,y:130,w:150,h:70,label:'σ',role:'q'},
 {id:'eps',x:25,y:240,w:160,h:85,label:'ε',sub:'N(0,1) draw',role:'noise',shape:'circle'},
 {id:'transform',x:400,y:120,w:220,h:95,label:'μ + σε',role:'q'},
 {id:'u',x:800,y:130,w:150,h:75,label:'u',role:'noise'},
 ],[{from:'mu',to:'transform'},{from:'sig',to:'transform'},{from:'eps',to:'transform',label:'fixed draw'},
 {from:'transform',to:'u'}],{width:1000,height:350,label:'Fixed standard-normal noise enters the differentiable transform mu plus sigma times epsilon.'});
}
add('reparameterise','gradients','Keep the noise separate from the parameters',2,
 stack(reparamGraph(),at(2,eq(R`\epsilon\sim\N(0,1),\qquad u=\mu+\sigma\epsilon`))),
 ['Which random distribution is independent of μ and σ?','For a fixed ε, which part is now deterministic?'],
 ['The standard Normal that supplies ε.','The mapping from μ and σ to u=μ+σε.'],
 'The noise source is separate. Do not draw ε as an output of μ or σ.',
 'Use one chosen ε to calculate the local derivatives by hand.');
add('scalar-derivatives','gradients','What happens to this same draw when a parameter changes?',3,
 stack(eq(R`\epsilon=-0.75,\quad\mu=0,\quad\sigma=2\quad\Longrightarrow\quad u=-1.5`),
 at(2,eq(R`\frac{\partial u}{\partial\mu}=1`)),at(3,eq(R`\frac{\partial u}{\partial\sigma}=\epsilon=-0.75`))),
 ['Hold ε fixed. What is u?','If μ increases by a small amount, how does u change?','Why can increasing σ move this particular u left?'],
 ['u=0+2×(−0.75)=−1.5.','It changes by the same amount, so the derivative is 1.','Because this fixed noise draw is negative; the derivative with respect to σ equals ε.'],
 'The −0.75 is a chosen inspectable input. Other noise draws can have the opposite sign.',
 'Connect those local derivatives to a scalar loss.',
 {source:'Chosen scalar input ε=−0.75 · arithmetic and autograd checked'});
add('chain-rule','gradients','The loss gradient follows the same draw',2,
 stack(eq(R`L=(u-1)^2,\qquad\frac{\partial L}{\partial u}=2(u-1)=-5`),
 at(2,eq(R`\frac{\partial L}{\partial\mu}=-5,\qquad\frac{\partial L}{\partial\sigma}=(-5)(-0.75)=3.75`))),
 ['At u=−1.5, what derivative reaches the sampled value?','How does that derivative reach μ and σ?'],
 ['The loss derivative is −5.','Multiply by ∂u/∂μ=1 and ∂u/∂σ=−0.75, respectively.'],
 'This scalar loss is chosen to expose the chain rule. It is not being labelled the KL objective.',
 'Use rsample for the Normal, then verify the path numerically.');
add('rsample-check','gradients','A fixed-noise finite difference checks the path',2,
 stack(code('q = Normal(mu, sigma)\nu = q.rsample((S,))\nloss = (q.log_prob(u) - p.log_prob(u)).mean()'),
 at(2,table(['Derivative','Autograd','Finite difference'],[
 ['μ',f(g.finite_difference.autograd[0],6),f(g.finite_difference.finite_difference[0],6)],
 ['σ',f(g.finite_difference.autograd[1],6),f(g.finite_difference.finite_difference[1],6)]]))),
 ['What is different about rsample?','What must stay fixed on both sides of the finite difference?'],
 ['It returns a draw with the pathwise gradient connection.','The same noise array. Fresh noise would confound the derivative check with sampling variation.'],
 'The notebook checks exact rsample=μ+σε equality under the same seed, then checks both derivatives.',
 'We now know how to estimate and differentiate an objective. The remaining obstacle is the posterior itself.',
 {source:'Measured check · seed88 · S=2048 · finite-difference h=10⁻⁵',
 demo:'Notebook section4, final cell: compare autograd with fixed-noise finite differences. Training normally uses fresh draws.'});

add('posterior','elbo','Now the target is an unknown posterior',2,
 stack(flow([
 {id:'u',x:150,y:60,w:140,h:110,label:'u',sub:'unknown',role:'noise',shape:'circle'},
 {id:'data',x:690,y:60,w:170,h:110,label:'𝒟',sub:'observed',role:'data'},
 ],[{from:'u',to:'data',label:'p(𝒟|u)',role:'p'}],
 {width:1000,height:240,label:'Probabilistic model: the unknown u generates observed data 𝒟.'}),
 at(2,eq(R`\text{Inference asks for }\pcol{p(u\mid\mathcal D)}`))),
 ['The model generates data from u. What is the backwards question?','Which object is our new target distribution?'],
 ['Which values of u explain the observed data?','The posterior p(u|𝒟). The observed data and model are fixed.'],
 'Name this a probabilistic model: this arrow expresses a conditional dependency, not an optimiser update.',
 'Keep the idea of fitting q, and state what operations q must support.');
add('tractable-q','elbo','Choose a family we can draw from and evaluate',2,
 stack(eq(R`\qcol{q_\psi(u)}`),at(2,eq(R`u\sim q_\psi,\qquad\log q_\psi(u)`))),
 ['What is ψ? What is u?','What operations do we need from this chosen family?'],
 ['ψ contains the free variational parameters; u is a possible value of the unknown.','Draw samples and evaluate the log density at those samples.'],
 'A Normal example used ψ=(μ,ρ); other problems can use another family.',
 'Try the same directional KL that worked for the available target.');
add('posterior-obstacle','elbo','The direct score asks for the density we do not have',2,
 stack(eq(R`${KL}=${EX}[\log ${Q}-\log\pcol{p(u\mid\mathcal D)}]`),
 at(2,p(`The unavailable term is ${m(R`\log p(u\mid\mathcal D)`)}.`))),
 ['Which pieces can we already compute?','Which piece blocks direct evaluation of this score?'],
 ['Samples from q and log q at those samples.','The normalised posterior log density, which is exactly what we are trying to approximate.'],
 'The obstacle is not a missing Gaussian formula. It is the unavailable posterior normaliser.',
 'Use Bayes’ rule to expose the joint and the evidence.');
add('bayes-joint','elbo','Replace the posterior with joint divided by evidence',2,
 stack(eq(R`\pcol{p(u\mid\mathcal D)}=\frac{\pcol{p(\mathcal D,u)}}{\dcol{p(\mathcal D)}}`),
 at(2,eq(R`p(\mathcal D,u)=p(\mathcal D\mid u)\,p(u)`))),
 ['Which factor can be built from prior and likelihood?','Which denominator may still be difficult?'],
 ['The joint p(𝒟,u).','The evidence p(𝒟), an integral over the unknown.'],
 'Keep the data argument in both the joint and evidence. Do not replace the joint with the prior alone.',
 'Take a logarithm before substituting into the KL.');
add('log-quotient','elbo','The log quotient becomes a difference',2,
 stack(eq(R`\log p(u\mid\mathcal D)=\log p(\mathcal D,u)-\log p(\mathcal D)`),
 at(2,eq(R`-\log p(u\mid\mathcal D)=-\log p(\mathcal D,u)+\log p(\mathcal D)`))),
 ['What is the log of joint divided by evidence?','Why does evidence become positive when the posterior term is subtracted?'],
 ['Log joint minus log evidence.','Subtracting that difference reverses both signs.'],
 'This is the sign-sensitive step. Give students time to predict the second line.',
 'Place exactly this replacement inside the expectation.');
add('substitute','elbo','Keep the KL; change only the posterior term',2,
 stack(eq(R`${KL}=${EX}[\log ${Q}-\log\pcol{p(u\mid\mathcal D)}]`),
 at(2,eq(R`=${EX}[\log ${Q}-\log ${J}+${E}]`))),
 ['Which expression stays the same on the left?','Which three terms now appear inside the average?'],
 ['The same posterior KL. This is an exact equality.','Log q minus log joint plus log evidence.'],
 'Point to the unchanged q term and the expanded posterior term.',
 'Ask what the expectation averages over, and whether the evidence varies with it.');
add('constant','elbo','Constant with respect to what?',3,
 stack(eq(R`${EX}[\log ${Q}-\log ${J}+${E}]`),
 at(2,p('The model and observed data are fixed. Evidence depends on neither u nor ψ.')),
 at(3,eq(R`=${EX}[\log ${Q}-\log ${J}]+${E}`))),
 ['What varies inside this expectation?','Does log evidence vary with the sampled u or the fitted ψ?','What justifies moving it outside?'],
 ['u is drawn from q.','Neither, for this fixed model and dataset.','Linearity of expectation: the average of a constant is that same constant.'],
 'Evidence can depend on generative model parameters θ; those are not being fitted in this argument.',
 'Rearrange the equality to put log evidence on the left.');
add('rearrange','elbo','Swap sides and reverse the bracket’s signs',2,
 stack(eq(R`${KL}=${EX}[\log ${Q}-\log ${J}]+${E}`),
 at(2,eq(R`${E}=${EX}[\log ${J}-\log ${Q}]+${KL}`))),
 ['After rearranging, what is the sign of the joint term?','Which term remains as a nonnegative gap?'],
 ['Positive inside the expectation.','The KL from q to the posterior.'],
 'Retain both lines so the sign change can be checked visually.',
 'Name the expectation we can estimate using the joint and q.');
add('name-elbo','elbo','A trainable objective emerges from the rearrangement',2,
 stack(eq(R`\ELBO(q_\psi)={${EX}[\log p(\mathcal D,u)-\log q_\psi(u)]}`),
 at(2,p('Evaluate log joint and log q at q’s draws. Average to estimate the expectation.'))),
 ['Which density has disappeared from the computable expression?','What do we still need to estimate in a general problem?'],
 ['The normalised posterior.','The expectation under q; its integrand is available, but the integral may require Monte Carlo.'],
 'Say estimate rather than promising every ELBO has a closed form.',
 'Use the nonnegative remaining KL to explain the name “lower bound”.');
add('bound','elbo','One fixed total, two parts',3,
 stack(eq(R`${E}=\qcol{\ELBO(q_\psi)}+\underbrace{${KL}}_{\geq0}`),
 at(2,eq(R`\ELBO(q_\psi)\leq\log p(\mathcal D)`)),
 at(3,p('Raise the ELBO; lower the posterior KL by the same amount.'))),
 ['Why can the ELBO not exceed the log evidence?','When would equality hold?','Which optimisation direction improves q?'],
 ['The remaining posterior KL is nonnegative.','When q equals the posterior almost everywhere, under the stated support and finiteness assumptions.','Maximise ELBO, equivalently minimise negative ELBO.'],
 'This gap is KL to the posterior. It is not the KL to the prior that appears in the later VAE decomposition.',
 'Translate this into a practical objective without asking for the unknown evidence.');
add('what-we-avoid','elbo','Optimise the ELBO without computing the evidence',2,
 stack(eq(R`\max_\psi\ELBO(q_\psi)\quad\Longleftrightarrow\quad\min_\psi[-\ELBO(q_\psi)]`),
 at(2,cols(p('Need: draws from q, log q, log joint.'),p('Do not need: the normalised posterior or its evidence.')))),
 ['Is the unknown evidence being set to zero?','What exactly makes it safe to omit from ψ’s optimisation?'],
 ['No. It remains an unknown fixed constant.','It does not change when ψ changes, for this fixed model and data.'],
 'The optimisation equivalence concerns ψ only. Learning θ later changes the situation.',
 'Return to the coin and implement these available pieces.');

add('coin-return','coin','Return to the same coin and the same data',2,
 stack(tosses(),eq(R`p(r)=\mathrm{Beta}(5,3)`),at(2,p('Fit q using the joint. Keep the known posterior for validation.'))),
 ['What are we holding fixed from the opening problem?','Why retain a known answer if the optimiser does not use it?'],
 ['The same sequence, prior and coin probability r.','To independently check whether the approximation and objective behave as expected.'],
 'The sequence has not changed to make an optimisation look better.',
 'First write the exact posterior reference from the sufficient statistics.');
add('coin-reference','coin','The conjugate posterior is a reference answer',2,
 stack(eq(R`p(r\mid\mathcal D)=\mathrm{Beta}(5+17,\ 3+13)`),
 at(2,eq(R`=\pcol{\mathrm{Beta}(22,16)}`))),
 ['Which prior parameter receives the heads count?','Does the optimiser need to call this posterior density?'],
 ['The first; the tails count is added to the second.','No. The posterior is used only for checking this solvable example.'],
 'Use an explicit reference label. Do not silently swap posterior.log_prob into the training objective.',
 'Construct the joint from the prior and the observed-data likelihood.');
add('coin-prior','coin','Keep the prior’s normalising constant',2,
 stack(eq(R`p(r)=\frac{r^{4}(1-r)^2}{B(5,3)}`),
 at(2,eq(R`\log p(r)=4\log r+2\log(1-r)-\log B(5,3)`))),
 ['Why are the exponents 4 and 2 rather than 5 and 3?','Can we drop this constant when checking absolute ELBO values?'],
 ['A Beta(a,b) density has exponents a−1 and b−1.','No. It matters for the numerical value of the joint, ELBO and evidence, even if independent of ψ.'],
 'Distinguish a constant irrelevant to an optimiser from a constant required for a correct absolute score.',
 'Add the contributions of all thirty observations.');
add('coin-likelihood','coin','Each observed toss contributes one log probability',2,
 stack(eq(R`\log p(\mathcal D\mid r)=17\log r+13\log(1-r)`),
 at(2,p('One ordered sequence. There is no binomial coefficient in this convention.'))),
 ['What contributes log r? What contributes log(1−r)?','Why is there no factor counting different orders?'],
 ['Each head contributes log r; each tail contributes log(1−r).','The observed object is one ordered sequence, not the event containing all sequences with that head count.'],
 'Counts are sufficient to evaluate this sequence likelihood, but the evidence convention remains sequence-based.',
 'Add prior and likelihood to obtain log joint.');
add('coin-joint','coin','The joint is available at any proposed r',2,
 stack(eq(R`\log p(\mathcal D,r)=\log p(\mathcal D\mid r)+\log p(r)`),
 at(2,code('log_likelihood = 17*log(r) + 13*log1p(-r)\nlog_joint = prior.log_prob(r) + log_likelihood'))),
 ['Which two available functions appear here?','Where is the unknown evidence in this code?'],
 ['The likelihood and prior density.','It does not appear. The joint is evaluated without the posterior normaliser.'],
 'log1p(−r) evaluates log(1−r) stably. Our Beta draws lie inside the valid support.',
 'Choose a flexible positive-parameter q and specify what the optimiser stores.');
add('coin-family','coin','Fit positive Beta shapes through unconstrained parameters',2,
 stack(eq(R`q_\psi(r)=\mathrm{Beta}(a,b)`),
 at(2,eq(R`a=\exp(\psi_a),\qquad b=\exp(\psi_b)`))),
 ['What family contains our exact posterior in this toy?','Which quantities does the optimiser store and update?'],
 ['The Beta family.','The unconstrained log-shape parameters ψ_a and ψ_b; exponentiation produces positive a and b.'],
 'r is the unknown probability. a and b are shape parameters. ψ parameterises those shapes.',
 'Draw from q, evaluate the joint-minus-q terms, then average.');
add('coin-estimator','coin','One mean over draws; the data are already summed',2,
 stack(code('a, b = psi.exp()\nq = Beta(a, b)\nr = q.rsample((512,))\nterms = log_joint(r) - q.log_prob(r)\nloss = -terms.mean()'),
 at(2,small('r and terms each have shape [512]. Mean over Monte Carlo draws.'))),
 ['Where are the thirty observation contributions combined?','Why is there no additional sum over the 512 draws before mean?'],
 ['Inside log_joint, through 17 log r+13 log(1−r).','Each term is one sample’s joint-minus-q contribution. We average that sample axis exactly once.'],
 'Beta.rsample is differentiable through an implicit reparameterisation method; it is not μ+σε.',
 'Inspect one measured step, including the actual gradients and transformed parameters.',
 {demo:'Notebook section6, training cell: inspect first_step and trace[1]. The posterior reference is absent from the loss.'});
const cs0=c.training.trace[0], cs1=c.training.trace[1], cfinal=c.training.trace.at(-1);
add('coin-first-step','coin','Inspect one actual update',2,
 stack(table(['Quantity','Before','After one Adam step'],[
 ['a',f(cs0.a,4),f(cs1.a,4)],['b',f(cs0.b,4),f(cs1.b,4)]]),
 at(2,eq(R`\nabla_\psi[-\widehat{\ELBO}]\approx(${c.first_step.gradient_psi[0].toFixed(3)},\ ${c.first_step.gradient_psi[1].toFixed(3)})`))),
 ['Which shape parameter should initially increase for more heads?','Are the printed gradients with respect to a,b or ψ?'],
 ['a should move up relative to b, shifting q’s centre towards the observed heads proportion.','They are with respect to ψ, the stored log-shape parameters. Adam then performs its own update.'],
 'Do not use a plain SGD formula to explain these Adam parameter changes. The values are directly recorded.',
 'Replay the saved fit rather than waiting for all six hundred steps.',
 {source:'Measured first update · seed2401 · Adam lr=0.025 · 512 Beta draws'});
add('coin-fit','coin','A measured fit using only the joint',3,
 stack(only('1',coinPlot(0,'0')),only('2',coinPlot(0,'100')),only('3',coinPlot(0,'600'))),
 ['Where does the initial q put its mass?','What changed by step100?','Is the final q exactly the known posterior?'],
 ['The initial Beta(10,10) is centred at 0.5 and is too broad.','The measured q has shifted and concentrated towards the reference.','No. The measured final shapes are about22.394 and16.189, with posterior KL about0.000222. It is close, not exact.'],
 'The teal reference was used for diagnostics only. The loss uses log joint and log q.',
 'Use selected candidates to check the identity independently of the training trajectory.',
 {source:'Actual CPU training · 600 Adam steps · seed2401 · saved checkpoints0/100/600'});
function gapLine(row){
 const xmin=-22.2,xmax=-21.2,W=1000,L=65,R=45,x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R);
 const a=x(row.elbo),b=x(c.log_evidence),close=b-a<110;
 return `<svg viewBox="0 0 1000 144" role="img" aria-label="ELBO and evidence on one fixed number line; their distance is the posterior KL.">
 <line x1="${L}" y1="69" x2="${W-R}" y2="69" stroke="#aab3bc"/>
 <circle cx="${a}" cy="69" r="5" fill="${palette.q}"/><circle cx="${b}" cy="69" r="5" fill="${palette.data}"/>
 <text x="${a}" y="49" text-anchor="${close?'end':'middle'}" fill="${palette.q}" font-size="24">ELBO ≈ ${f(row.elbo,3)}</text>
 <text x="${W-R}" y="25" text-anchor="end" fill="${palette.data}" font-size="24">log p(𝒟) ≈ ${f(c.log_evidence,3)}</text>
 <path d="M${a},78 V104 H${b} V78" fill="none" stroke="${palette.loss}" stroke-width="2"/>
 <text x="${(a+b)/2}" y="133" text-anchor="middle" fill="${palette.loss}" font-size="24">gap = KL ${row.posterior_kl<1e-10?'= 0':'≈ '+f(row.posterior_kl,3)}</text>
 <text x="${L}" y="96" font-size="20" fill="#68717b">−22.2</text><text x="${W-R}" y="96" text-anchor="end" font-size="20" fill="#68717b">−21.2 nats</text></svg>`;
}
add('coin-check','coin','Check the fixed total with three selected candidates',3,
 c.candidates.map((row,i)=>only(i+1,coinPlot(i,null,{height:260})+gapLine(row))).join(''),
 ['Add the ELBO and the gap. What is fixed?','Does the better candidate raise the ELBO by the amount its KL falls?','What happens at the exact posterior?'],
 ['The sum is the same log evidence, approximately−21.397.','Yes. The identity holds independently for each selected candidate.','The posterior KL is zero, so ELBO equals log evidence.'],
 'These three candidates are checking points, not asserted optimisation checkpoints. ELBO was computed directly from expected joint minus expected log q.',
 'A noisy ELBO estimate behaves differently from this exact expectation.',
 {source:'Direct analytic Beta ELBO and independent posterior KL · selected candidates, not a training trace'});
add('noisy-elbo','coin','An estimate is not itself guaranteed to be a lower bound',2,
 stack(image('figures/coin-noisy-objective.svg','Measured sampled ELBO estimates and analytic ELBO diagnostics along the coin training run.',
 'Same pre-update q at each step · fresh training draws · known log evidence reference'),
 at(2,small(`${c.sampled_estimates_above_evidence} of 600 sampled estimates lie above the known evidence in this run.`))),
 ['Can a finite-sample ELBO estimate cross the reference line?','Which object remains a mathematical lower bound?'],
 ['Yes. The noisy estimate can lie above the evidence on some draws.','The exact expected ELBO, not every finite-sample realisation of its estimator.'],
 'This is observed behaviour from the saved run, not a universal crossing frequency.',
 'Separate an improving objective from a certificate of exact inference.',
 {source:'Measured seed2401 training trace · sampled objective versus direct analytic diagnostic'});
add('certificate','coin','A higher or stable ELBO does not prove a zero gap',2,
 stack(eq(R`\ELBO(q_{\mathrm{new}})>\ELBO(q_{\mathrm{old}})`),
 at(2,p('The family may miss the posterior. Optimisation may stop short.'))),
 ['What does a true increase in the exact ELBO establish?','Does a stable loss establish q=p(u|𝒟)?'],
 ['For this fixed model and data, posterior KL decreased.','No. The family or optimisation can leave a positive gap.'],
 'Recall the mean-only Gaussian: even its perfect optimum left a width mismatch.',
 'Retrieve the method without looking at the derivation.',
 {misconception:'Improvement, convergence of a trace and exact posterior recovery are different claims.'});

add('retrieve-samples','retrieve','For KL(q‖p), where do the samples come from?',2,
 stack(eq(R`\KL(q\|p)=\boxed{\quad ?\quad}[\log q(u)-\log p(u)]`),
 at(2,eq(R`\boxed{\E_{u\sim q}}`))),
 ['Answer without looking at the earlier definition.','Why does sampling from p change the question?'],
 ['From q, the first argument.','It changes the weighting distribution and does not estimate the same directional KL.'],
 'Ask several students for the sampling distribution before revealing it.',
 'Now separate the parameters being updated from the random values being drawn.');
add('retrieve-update','retrieve','What changes during fixed-model variational inference?',2,
 stack(table(['Object','Changes during this inference fit?'],[
 ['Observed data 𝒟','?'],['Exact posterior p(u|𝒟)','?'],['Variational parameters ψ','?'],['A newly drawn u','?']]),
 at(2,p('Update ψ. Draw fresh u. The fixed data, model and exact posterior do not change.'))),
 ['Which objects change, and in what sense?','Is a new draw an optimiser parameter update?'],
 ['ψ is updated; random draws vary; data, model and exact posterior remain fixed.','No. A draw is a sampled value, not a stored parameter being fitted.'],
 'Distinguish randomness in evaluating an objective from changes in the objective’s parameters.',
 'Explain the omitted evidence using the variable it is constant with respect to.');
add('retrieve-evidence','retrieve','Why can we omit the evidence from ψ’s optimisation?',2,
 stack(eq(R`\KL(q_\psi\|p(u\mid\mathcal D))=-\ELBO(q_\psi)+\log p(\mathcal D)`),
 at(2,eq(R`\nabla_\psi\log p(\mathcal D)=0`))),
 ['Is the reason that evidence is zero, easy, or constant?','Constant with respect to which parameters?'],
 ['It is constant, not zero and not necessarily easy to evaluate.','ψ, for the fixed model and observed data used in this inference problem.'],
 'Ask students to state both the changing parameter and what is held fixed.',
 'Apply the procedure to a new inference problem.');
add('exit-transfer','retrieve','A sensor has an unknown constant bias',3,
 stack(eq(R`u\sim\N(0,1),\qquad y_i\mid u\sim\N(u,0.5^2)`),
 at(2,eq(R`p(\mathcal D,u)=p(u)\prod_i p(y_i\mid u)`)),
 at(3,eq(R`\max_\psi\E_{q_\psi(u)}[\log p(\mathcal D,u)-\log q_\psi(u)]`))),
 ['Identify the unknown, observations and one possible q family.','Which available functions form the joint?','What objective would you implement?'],
 ['u is the bias; 𝒟 contains observed sensor values; a Normal q is one sensible choice.','The prior density and the likelihood of each reading.','The expectation of log joint minus log q, maximised over ψ.'],
 'This is an implementation-identification exercise, not another numerical derivation or a claim that approximation is required for this conjugate toy.',
 'Connect this posterior-fitting problem to the models we will build next.');
add('next-lecture','retrieve','Generative models will create this inference problem again',2,
 stack(p('A hidden description can generate an observation.'),at(2,p('Next: learn representations for images, then ask whether they can generate new images.'))),
 ['Where might a new unknown appear in an image model?','What question should we ask about a good reconstruction?'],
 ['A latent description z can explain an observed image x.','Whether reconstructing observations also gives a useful way to generate new ones.'],
 'Lecture2 introduces generative modelling and ordinary autoencoders. Lecture3 returns to variational inference with an encoder and decoder.',
 'Close the main route here. Optional derivations follow for reading or extra time.',
 {source:'Main70-minute route ends here · optional depth follows'});

add('optional-normal-algebra','optional','Where does the complete Gaussian KL formula come from?',3,
 stack(eq(R`\log\frac{q(u)}{p(u)}=\log\frac{s}{\sigma}+\frac{(u-m)^2}{2s^2}-\frac{(u-\mu)^2}{2\sigma^2}`),
 at(2,eq(R`\E_q[(u-m)^2]=\sigma^2+(\mu-m)^2,\quad\E_q[(u-\mu)^2]=\sigma^2`)),
 at(3,eq(R`\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12`))),
 ['Which log-normalising constants cancel?','Which two moments are needed?','Where does the final −1/2 come from?'],
 ['The common log√(2π) terms cancel, leaving the log scale ratio.','The target-centred and q-centred second moments.','The expectation of (u−μ)²/(2σ²) equals one half.'],
 'This is an optional completion of the Gaussian algebra. The core route exposed only its decisive moment.',
 'Nonnegativity belongs to the expectation, not every local ratio.',
 {optional:true,source:'Analytic derivation for positive Normal scales'});
add('optional-nonnegative','optional','Why is exact KL nonnegative?',3,
 stack(eq(R`\KL(q\|p)=-\E_q\!\left[\log\frac{p(u)}{q(u)}\right]`),
 at(2,eq(R`\geq-\log\E_q\!\left[\frac{p(u)}{q(u)}\right]`)),
 at(3,eq(R`=-\log1=0`))),
 ['Which function makes Jensen’s inequality useful?','Why does the inequality point this way?','What makes the final expectation one?'],
 ['The convex function −log.','For a convex function, its expectation is at least its value at the expectation.','For positive densities on the same support, integrating q·p/q gives the normalisation integral of p.'],
 'State the common-support assumption for this simple proof. More general support cases require the standard extension, including infinite KL when needed.',
 'A valid alternative estimator can handle sampling derivatives without a pathwise transform.',
 {optional:true,source:'Jensen proof · positive densities on the same support and finite terms'});
add('optional-score-function','optional','Pathwise gradients are one route, not the only route',2,
 stack(eq(R`\nabla_\psi\E_{q_\psi}[f_\psi(u)]=\E_{q_\psi}[f_\psi(u)\nabla_\psi\log q_\psi(u)+\nabla_\psi f_\psi(u)]`),
 at(2,p('A score-function estimator accounts explicitly for the changing sampling distribution.'))),
 ['Which term was missing from naive detached-sample differentiation?','Does this identity require u=μ+σε?'],
 ['The term accounting for the change in the probability of drawing u.','No. It is a different route, under conditions permitting differentiation under the integral.'],
 'The displayed identity includes explicit parameter dependence of f. Do not silently discard that term for a general integrand.',
 'The event and sample dimensions still need correct reductions, whichever estimator is used.',
 {optional:true,source:'Differentiation-under-the-integral identity · regularity assumptions apply'});
add('optional-reductions','optional','A mean must say which dimension it averages',2,
 stack(table(['Object','Shape','Correct reduction'],[
 ['S scalar Beta draws','[S]','mean over S'],
 ['S d-dimensional joint log densities','[S]','mean over S'],
 ['S factorised coordinate log densities','[S,d]','sum event d, then mean S'],
 ]),at(2,p('MultivariateNormal.log_prob already sums the event. Do not sum the sample axis a second time.'))),
 ['Does every log_prob output still have an event dimension?','What goes wrong if you sum S before subtracting log q from the joint?'],
 ['No. A multivariate distribution’s log_prob typically already reduces its event dimensions.','It changes the relative weighting, often multiplying an entropy contribution by the number of samples.'],
 'This corrects a concrete reduction pitfall in the historical multivariate regression extension.',
 'Even harmless-looking likelihood constants affect absolute evidence checks.',
 {optional:true,source:'Shape contract · scalar coin and multivariate event distinction'});
add('optional-count-data','optional','An ordered sequence and a count have different evidence',2,
 stack(eq(R`p(H=17\mid r)=\binom{30}{17}r^{17}(1-r)^{13}`),
 at(2,table(['Observed object','Log evidence'],[
 ['One ordered sequence',f(c.log_evidence,5)],
 ['Only the head count',f(c.count_convention_comparison.count_log_evidence,5)],
 ]))),
 ['Which event contains more possible sequences?','Does this constant change the posterior or its KL gap?'],
 ['The count-only event includes every ordering with17 heads.','No. It adds the same log binomial coefficient to joint, ELBO and evidence; the posterior and gap remain unchanged.'],
 'The entire main lecture consistently uses the ordered-sequence convention.',
 'The Beta family also provides analytic expectations that independently check the sampled objective.',
 {optional:true,source:'Exact convention comparison · log binomial coefficient computed with log-gamma'});
add('optional-beta-elbo','optional','Check the sampled objective with analytic Beta moments',3,
 stack(eq(R`\E_q[\log r]=\operatorname{digamma}(a)-\operatorname{digamma}(a+b)`),
 at(2,eq(R`\E_q[\log(1-r)]=\operatorname{digamma}(b)-\operatorname{digamma}(a+b)`)),
 at(3,p('Insert these into expected log likelihood + expected log prior − expected log q.'))),
 ['Which two expectations determine the coin ELBO?','Why write digamma as a name rather than ψ here?','How was the identity checked non-circularly?'],
 ['Expected log r and expected log(1−r).','ψ already denotes free variational parameters.','The notebook computes ELBO directly from these expectations, then independently adds the library posterior KL and checks the evidence.'],
 'The direct calculation is not evidence minus KL disguised as an independent check. Numerical integration supplies another check.',
 'The notebook contains the complete formulas and reproducible outputs.',
 {optional:true,source:'Analytic Beta moments · direct ELBO, independent KL and quadrature checks'});
