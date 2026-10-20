# KL and variational inference: speaker notes

Planned main route: 70 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| An inference question | 4 |
| Which distribution is closer? | 9 |
| Turn comparison into optimisation | 10 |
| Estimate an unavailable formula | 7 |
| Differentiate through a draw | 7 |
| An objective without the posterior | 13 |
| Fit and check the coin | 13 |
| Retrieve and transfer | 7 |

## coin-question: What would you like to know about this coin?

**Build 1, ask:** Thirty observed tosses. What is still unknown?

**Expected:** The coin’s probability of heads. The observations themselves are known.

**Build 2, ask:** Do we want one guess, or uncertainty about the guess?

**Expected:** A distribution over plausible values of r describes both a centre and uncertainty.

**Point here:** Point to individual observed H/T values before introducing r.

**Transition:** The observed sequence is fixed. We will ask how it changes our belief about r.

**Source / evidence:** One chosen ordered sequence · 17 H and 13 T · recorded in science.json

## data-and-unknown: The data are fixed; the unknown is not a data point

**Build 1, ask:** Which object is observed? Which object will have a posterior?

**Expected:** The ordered sequence 𝒟 is observed. The unknown r has the posterior.

**Build 2, ask:** Does inferring r change any of the thirty tosses?

**Expected:** No. We update our description of r while holding the data fixed.

**Point here:** Keep the blue observed data separate from the amber unknown.

**Transition:** We already know Bayes’ rule. Use it to name what we want.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## prior-before-data: A prior is a distribution over the unknown

**Build 1, ask:** What does the Beta prior describe?

**Expected:** Uncertainty about the single unknown probability r. It is not a distribution of toss outcomes.

**Build 2, ask:** What two ingredients turn it into a posterior shape?

**Expected:** The likelihood of the observed sequence and the prior density.

**Point here:** The prior’s argument is r. The likelihood treats the already observed sequence as fixed.

**Transition:** The proportionality hides a normalising constant.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## bayes-recall: What normalises the posterior?

**Build 1, ask:** What is the denominator doing?

**Expected:** It makes the posterior integrate to one. It is the evidence.

**Build 2, ask:** Why practise an approximate method on a solvable problem?

**Expected:** A known answer lets us check the implementation and understand the approximation.

**Point here:** Do not launch an extended Bayes recap; these students know the rule.

**Transition:** First learn how to compare and improve a distribution when the target is available.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## compare: Which Gaussian is closer to the target?

**Build 1, ask:** Describe the centre and width of q₁.

**Expected:** q₁ is left of p and narrower.

**Build 2, ask:** Vote for q₁ or q₂ before calculating.

**Expected:** q₂ shares p’s centre and has a more similar width. We need a score to make “closer” precise.

**Point here:** Curves use identical axes and labelled solid/dashed strokes.

**Transition:** A score should compare the whole distribution, not only the means.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## one-location: One density comparison is not the whole story

**Build 1, ask:** At u=0, which density is larger?

**Expected:** q₁ is more concentrated and has the larger density here.

**Build 2, ask:** Could this one location settle which whole distribution is closer?

**Expected:** No. The relationship changes across the support. We need an average.

**Point here:** Point at the centres on the same axes. A density is not a point probability.

**Transition:** Choose the distribution that supplies the weights for our average.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## kl-average: Average the log ratio under q

**Build 1, ask:** Where does this expectation get its values of u?

**Expected:** From q. Equivalently, q weights the integral across u.

**Build 2, ask:** What does the order q, then p tell us?

**Expected:** The average is under q. Reversing the arguments changes that weighting.

**Point here:** Connect the purple q outside the bracket to the purple density inside it.

**Transition:** Check that swapping the arguments really can change the number.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## kl-direction: Reversing KL changes the question

**Build 1, ask:** Are these two averages weighted in the same way?

**Expected:** No. One averages under q₁, the other under p.

**Build 2, ask:** What does the numerical difference rule out?

**Expected:** KL is not a symmetric distance. Always state the direction.

**Point here:** Both values use the same two Normal distributions. Only their order changed.

**Transition:** Keep the direction fixed as we fit q to p.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## local-versus-average: Nonnegative KL does not make every term positive

**Build 1, ask:** Can p(u) exceed q(u) at a particular u?

**Expected:** Yes. Then that local log ratio is negative.

**Build 2, ask:** Which nonnegative statement applies to the whole average?

**Expected:** Exact KL is nonnegative. A matched distribution has zero KL.

**Point here:** Separate a pointwise term from its expectation. This distinction returns when sampling.

**Common misconception:** A probability density may exceed one; a log-density ratio may be negative.

**Transition:** For two Gaussians, the entire expectation has a closed form.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## gaussian-formula: For two Gaussians, evaluate the expectation exactly

**Build 1, ask:** Which parameters belong to q, and which stay fixed in p?

**Expected:** μ and σ belong to q; m and s belong to the target.

**Build 2, ask:** Where do width and centre mismatch enter?

**Expected:** The log scale ratio and σ² measure width effects; the squared shift measures centre mismatch.

**Point here:** Every scale symbol is a standard deviation; every squared scale is a variance.

**Transition:** Substitute our two candidates before revealing the scores.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## gaussian-substitution: Substitute first; then compare the score

**Build 1, ask:** Which candidate has no squared centre-shift term?

**Expected:** q₂, because its mean is 1, the target mean.

**Build 2, ask:** Does the score support the earlier prediction?

**Expected:** Yes. q₂ has the smaller KL, approximately 0.0689 versus 0.3494 nats.

**Point here:** The denominator is 2s²=2×16, not twice the standard deviation.

**Transition:** The score tells us which candidate wins. Can it tell us how to improve one?

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## score-to-search: A comparison can become a search

**Build 1, ask:** What should the optimiser change?

**Expected:** The parameters of q, while the target p stays fixed.

**Build 2, ask:** What does this argmin select?

**Expected:** The best member of the chosen family under this directional KL objective.

**Point here:** ψ is a parameter collection, not a sampled value u.

**Transition:** Before differentiating, expose the Gaussian moment that makes the formula work.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## gaussian-moment: A moment identity explains the squared shift

**Build 1, ask:** When expanding the square, which cross-term should vanish?

**Expected:** The cross-term contains E_q[u−μ], which is zero.

**Build 2, ask:** Why does the remaining random term become σ²?

**Expected:** By definition, the variance of q is E_q[(u−μ)²]=σ².

**Point here:** The mean of q is μ, even though the target has mean m.

**Transition:** Insert the variance into the KL and isolate one movable parameter.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## isolate-mean: Move the centre while keeping the width fixed

**Build 1, ask:** If the width cannot change, which way should μ move?

**Expected:** Right, towards the target mean 1.

**Build 2, ask:** Which term changes when μ changes?

**Expected:** Only the squared-shift term. Everything in the bracket is constant for this experiment.

**Point here:** The experiment deliberately restricts q. Do not imply that σ is being learned yet.

**Transition:** Differentiate only the term that depends on μ.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## gradient: A negative derivative produces a rightward descent step

**Build 1, ask:** Differentiate the squared shift.

**Expected:** 2(μ−1)/32.

**Build 2, ask:** What simplifies in the derivative?

**Expected:** The factor of two reduces the denominator to 16.

**Build 3, ask:** Does a negative derivative mean move left under gradient descent?

**Expected:** No. Gradient descent subtracts the derivative, so a negative derivative moves μ right.

**Point here:** Distinguish the derivative’s sign from the update’s direction.

**Transition:** Use one explicit learning rate and perform the arithmetic.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## one-update: One literal update

**Build 1, ask:** What does gradient descent subtract?

**Expected:** The learning rate times the derivative.

**Build 2, ask:** Predict the new mean.

**Expected:** The new mean is 0.25.

**Build 3, ask:** Which parameter has not changed?

**Expected:** The standard deviation remains 2, and p remains fixed.

**Point here:** η=4 is a chosen input for this inspectable step, not a general recommended learning rate.

**Transition:** Recompute the score; a moving curve alone is not evidence of improvement.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## recompute-loss: The new curve earns a new loss evaluation

**Build 1, ask:** What changed in the curve?

**Expected:** Its position changed. Its width did not.

**Build 2, ask:** Did the computed objective improve?

**Expected:** Yes. The KL falls from approximately 0.3494 to 0.3357.

**Point here:** Read the before/after values only after students inspect the shift.

**Notebook demonstration:** Notebook section 2, first code cell: run the literal update and inspect μ.grad, μ_after and both KL values.

**Transition:** Ask what the restricted optimisation can achieve if repeated.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## family-restriction: Perfect mean fitting still leaves a gap

**Build 1, ask:** What is the best mean if σ must stay 2?

**Expected:** μ=1.

**Build 2, ask:** Why is the minimum KL still positive?

**Expected:** The target standard deviation is 4. Mean-only fitting cannot repair the width mismatch.

**Point here:** This is a family restriction, not an optimiser failure.

**Transition:** To fit both parameters, the optimiser needs a valid representation of scale.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## positive-scale: A standard deviation must remain positive

**Build 1, ask:** Why not update an unconstrained σ directly?

**Expected:** An arbitrary update could make σ nonpositive, which is invalid.

**Build 2, ask:** What does ρ represent here?

**Expected:** Log standard deviation. It is not log variance.

**Point here:** The optimiser stores μ and ρ. The distribution receives μ and exp(ρ).

**Transition:** The optimisation loop must rebuild q and clear old gradients.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## optimisation-loop: Rebuild, score, differentiate, update

**Build 1, ask:** Why build q again on the next iteration?

**Expected:** Its parameters have changed, so the density and computation graph must use the current values.

**Build 2, ask:** Why clear gradients before backward?

**Expected:** PyTorch accumulates gradients unless they are cleared.

**Point here:** This is the executable five-line core of the notebook loop.

**Notebook demonstration:** Notebook section 2, second code cell: replay saved snapshots, then inspect one update. Adam optimises analytic KL with lr=0.08.

**Transition:** Inspect a real saved trajectory instead of waiting for training in class.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## measured-gaussian-fit: Letting both parameters move removes the restriction

**Build 1, ask:** What can this larger family now match?

**Expected:** Both the target centre and width.

**Build 2, ask:** At step 30, what is still visibly different?

**Expected:** Inspect the measured curve rather than assuming monotone visual improvement.

**Build 3, ask:** How close is the final measured fit?

**Expected:** At step 150, μ≈0.9996, σ≈4.0000 and KL≈4.36×10⁻⁹; numerical agreement is excellent.

**Point here:** The three states are saved steps 0, 30 and 150 from actual analytic-KL optimisation.

**Transition:** A useful score may exist even when the library has no analytic formula for it.

**Source / evidence:** Measured deterministic optimisation · Adam lr=0.08 · 150 steps · float64

## analytic-loss-trace: A loss trace answers a specific question

**Build 1, ask:** What is being measured on the vertical axis?

**Expected:** Analytic directional KL at the current q parameters.

**Build 2, ask:** Would the same smooth trace be guaranteed for a sampled objective?

**Expected:** No. Monte Carlo estimates introduce sampling variation.

**Point here:** Keep the evidence label tied to the actual experiment: one toy, analytic loss, recorded steps.

**Transition:** Now replace the target with the PML mixture and retain the definition of KL.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## mixture-target: Replace the target; keep the definition

**Build 1, ask:** What makes this a normalised density?

**Expected:** Each component integrates to one and the nonnegative weights sum to one.

**Build 2, ask:** Does evaluating p(u) require a closed-form KL?

**Expected:** No. The mixture density can be evaluated directly.

**Point here:** These are the original PML mixture parameters, not a newly invented target.

**Transition:** Compare the same two Gaussian candidates used in the historical notebook.

**Source / evidence:** PML2023 notebook cells25–30 · original mixture and candidate parameters

## mixture-candidates: Which candidate wastes more mass far from the mixture?

**Build 1, ask:** Where does p assign most of its density?

**Expected:** Near the two mixture components, centred at −0.2 and 1.

**Build 2, ask:** Which candidate draws more often from far-away regions?

**Expected:** The very broad q₂ draws more often from far away. Predict a larger KL(q₂‖p), then check.

**Point here:** The common wide x-axis keeps the broad candidate visible. Do not zoom each curve independently.

**Transition:** Ask the library for the analytic KL and inspect the actual result.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## unsupported-kl: A missing library formula is not an undefined quantity

**Build 1, ask:** What would this call need to know?

**Expected:** An implemented formula or algorithm for this specific distribution pair.

**Build 2, ask:** What does this error establish, and what does it not establish?

**Expected:** It establishes that this runtime does not implement the pair. It does not show that the KL is undefined.

**Point here:** The exact exception is saved in science.json; the slide uses a shortened readable message.

**Transition:** Return to the expectation definition instead of abandoning the score.

**Source / evidence:** Observed PyTorch 2.13.0 runtime behaviour · full error saved in science.json

## sample-average: Draw from q, score both densities, average

**Build 1, ask:** Which distribution supplies the draws?

**Expected:** q, the first argument of KL(q‖p).

**Build 2, ask:** What can we compute at each draw?

**Expected:** The log density of q minus the log density of p at the same draw.

**Build 3, ask:** Which dimension is averaged?

**Expected:** The S independent Monte Carlo draws.

**Point here:** The hat denotes an estimate. The log ratio compares the same u_i in both densities.

**Transition:** Write the estimator in three lines and validate it on a known case.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## mc-code-check: Validate the estimator before using the unfamiliar target

**Build 1, ask:** What result can we check without Monte Carlo?

**Expected:** The analytic Normal KL, approximately 0.0689.

**Build 2, ask:** Should a finite estimate equal the exact answer digit for digit?

**Expected:** No. The estimate varies with the sampled draws.

**Point here:** The notebook’s evaluation-only helper uses sample(); this differentiable training form uses rsample(), explained next.

**Notebook demonstration:** Notebook section3: run the known-Gaussian repeated-seed experiment. Do not change q or p.

**Transition:** Repeat the estimator while holding both distributions fixed.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## mc-variability: More draws reduce variability, not every individual error

**Build 1, ask:** What changed between these four groups?

**Expected:** Only the number of Monte Carlo draws. Each group also repeats 40 specified seeds.

**Build 2, ask:** Must every larger-S estimate be closer than every smaller-S estimate?

**Expected:** No. Sampling variance shrinks, but individual errors need not decrease monotonically.

**Point here:** Point to the spread across seeds and the unchanging exact reference line.

**Transition:** Examine one small sample whose estimate is negative.

**Source / evidence:** Measured estimates · seeds1000–1039 · S=10,100,1000,10000 · float64

## negative-estimate: Five draws can give a negative KL estimate

**Build 1, ask:** Some rows are negative. Could their average be negative?

**Expected:** Yes; this particular five-draw average is approximately −0.0933.

**Build 2, ask:** Does this contradict nonnegative exact KL?

**Expected:** No. The finite-sample estimate is not the exact expectation.

**Point here:** Seed0 was deliberately selected as the first negative example in a seed search, not as a performance result.

**Transition:** Use many draws for the mixture comparison, and label the result as an estimate.

**Source / evidence:** Intentional counterexample · first negative seed in0–999 is0 · five q draws

## mixture-results: The estimated score separates these candidates clearly

**Build 1, ask:** How large is the difference relative to the sampling uncertainty?

**Expected:** The candidate difference is far larger than the reported standard errors.

**Build 2, ask:** What role does the numerical integral play?

**Expected:** It is an independent numerical check in this one-dimensional toy, not an available high-dimensional solution in general.

**Point here:** Both columns are numerical evaluations; neither should be labelled exact analytical mixture KL.

**Transition:** An estimator is useful for scoring. Can it also provide the derivative we need?

**Source / evidence:** 100,000 q draws · seeds402/403 · separate wide-domain numerical integration

## sample-path: A random draw is not automatically a differentiable path

**Build 1, ask:** Which connection would the derivative need to follow?

**Expected:** From the scalar loss through u back to the distribution parameters.

**Build 2, ask:** Does this sample retain that path in PyTorch?

**Expected:** No. Normal.sample() returns a draw without that pathwise autograd connection.

**Point here:** This is a computation view. The word sample identifies the random operation.

**Transition:** Check the actual runtime behaviour before proposing a remedy.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## sample-backward: Ask the sample itself for a gradient

**Build 1, ask:** What do you predict before running backward?

**Expected:** Backward on this sampled scalar will fail because it has no autograd path.

**Build 2, ask:** What has this failure shown?

**Expected:** It identifies the absent path from this sample to μ and σ; it does not prove all stochastic gradient methods are impossible.

**Point here:** The notebook catches this expected error so Run All still completes.

**Notebook demonstration:** Notebook section4: inspect sample.requires_grad and the caught sample.backward() exception.

**Transition:** A more subtle failure occurs when the loss has another explicit parameter dependence.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## a-gradient-is-not-enough: A backward call can run and still miss the intended path

**Build 1, ask:** Can log q(u) depend on μ and σ even if u is detached?

**Expected:** Yes. The density formula itself contains μ and σ.

**Build 2, ask:** Would a nonempty gradient tensor prove this is the desired KL derivative?

**Expected:** No. The derivative also needs the effect of changing the sampling distribution, which this naive construction omits.

**Point here:** Do not claim that backward must fail for every loss containing sample().

**Transition:** Compare the resulting derivatives with the analytic reference using identical draws.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## gradient-comparison: Check the derivative, not just the existence of a tensor

**Build 1, ask:** Which row agrees with the derivative of the actual KL?

**Expected:** The pathwise row closely agrees; the naive detached row misses the sampling-distribution effect.

**Build 2, ask:** Is Monte Carlo agreement expected to be exact?

**Expected:** No. This is a finite Monte Carlo derivative, so it has sampling error.

**Point here:** A properly constructed score-function estimator is another valid approach; this comparison concerns the naive detached construction.

**Transition:** Separate the randomness from the deterministic transformation.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## reparameterise: Keep the noise separate from the parameters

**Build 1, ask:** Which random distribution is independent of μ and σ?

**Expected:** The standard Normal that supplies ε.

**Build 2, ask:** For a fixed ε, which part is now deterministic?

**Expected:** The mapping from μ and σ to u=μ+σε.

**Point here:** The noise source is separate. Do not draw ε as an output of μ or σ.

**Transition:** Use one chosen ε to calculate the local derivatives by hand.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## scalar-derivatives: What happens to this same draw when a parameter changes?

**Build 1, ask:** Hold ε fixed. What is u?

**Expected:** u=0+2×(−0.75)=−1.5.

**Build 2, ask:** If μ increases by a small amount, how does u change?

**Expected:** It changes by the same amount, so the derivative is 1.

**Build 3, ask:** Why can increasing σ move this particular u left?

**Expected:** Because this fixed noise draw is negative; the derivative with respect to σ equals ε.

**Point here:** The −0.75 is a chosen inspectable input. Other noise draws can have the opposite sign.

**Transition:** Connect those local derivatives to a scalar loss.

**Source / evidence:** Chosen scalar input ε=−0.75 · arithmetic and autograd checked

## chain-rule: The loss gradient follows the same draw

**Build 1, ask:** At u=−1.5, what derivative reaches the sampled value?

**Expected:** The loss derivative is −5.

**Build 2, ask:** How does that derivative reach μ and σ?

**Expected:** Multiply by ∂u/∂μ=1 and ∂u/∂σ=−0.75, respectively.

**Point here:** This scalar loss is chosen to expose the chain rule. It is not being labelled the KL objective.

**Transition:** Use rsample for the Normal, then verify the path numerically.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## rsample-check: A fixed-noise finite difference checks the path

**Build 1, ask:** What is different about rsample?

**Expected:** It returns a draw with the pathwise gradient connection.

**Build 2, ask:** What must stay fixed on both sides of the finite difference?

**Expected:** The same noise array. Fresh noise would confound the derivative check with sampling variation.

**Point here:** The notebook checks exact rsample=μ+σε equality under the same seed, then checks both derivatives.

**Notebook demonstration:** Notebook section4, final cell: compare autograd with fixed-noise finite differences. Training normally uses fresh draws.

**Transition:** We now know how to estimate and differentiate an objective. The remaining obstacle is the posterior itself.

**Source / evidence:** Measured check · seed88 · S=2048 · finite-difference h=10⁻⁵

## posterior: Now the target is an unknown posterior

**Build 1, ask:** The model generates data from u. What is the backwards question?

**Expected:** Which values of u explain the observed data?

**Build 2, ask:** Which object is our new target distribution?

**Expected:** The posterior p(u|𝒟). The observed data and model are fixed.

**Point here:** Name this a probabilistic model: this arrow expresses a conditional dependency, not an optimiser update.

**Transition:** Keep the idea of fitting q, and state what operations q must support.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## tractable-q: Choose a family we can draw from and evaluate

**Build 1, ask:** What is ψ? What is u?

**Expected:** ψ contains the free variational parameters; u is a possible value of the unknown.

**Build 2, ask:** What operations do we need from this chosen family?

**Expected:** Draw samples and evaluate the log density at those samples.

**Point here:** A Normal example used ψ=(μ,ρ); other problems can use another family.

**Transition:** Try the same directional KL that worked for the available target.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## posterior-obstacle: The direct score asks for the density we do not have

**Build 1, ask:** Which pieces can we already compute?

**Expected:** Samples from q and log q at those samples.

**Build 2, ask:** Which piece blocks direct evaluation of this score?

**Expected:** The normalised posterior log density, which is exactly what we are trying to approximate.

**Point here:** The obstacle is not a missing Gaussian formula. It is the unavailable posterior normaliser.

**Transition:** Use Bayes’ rule to expose the joint and the evidence.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## bayes-joint: Replace the posterior with joint divided by evidence

**Build 1, ask:** Which factor can be built from prior and likelihood?

**Expected:** The joint p(𝒟,u).

**Build 2, ask:** Which denominator may still be difficult?

**Expected:** The evidence p(𝒟), an integral over the unknown.

**Point here:** Keep the data argument in both the joint and evidence. Do not replace the joint with the prior alone.

**Transition:** Take a logarithm before substituting into the KL.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## log-quotient: The log quotient becomes a difference

**Build 1, ask:** What is the log of joint divided by evidence?

**Expected:** Log joint minus log evidence.

**Build 2, ask:** Why does evidence become positive when the posterior term is subtracted?

**Expected:** Subtracting that difference reverses both signs.

**Point here:** This is the sign-sensitive step. Give students time to predict the second line.

**Transition:** Place exactly this replacement inside the expectation.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## substitute: Keep the KL; change only the posterior term

**Build 1, ask:** Which expression stays the same on the left?

**Expected:** The same posterior KL. This is an exact equality.

**Build 2, ask:** Which three terms now appear inside the average?

**Expected:** Log q minus log joint plus log evidence.

**Point here:** Point to the unchanged q term and the expanded posterior term.

**Transition:** Ask what the expectation averages over, and whether the evidence varies with it.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## constant: Constant with respect to what?

**Build 1, ask:** What varies inside this expectation?

**Expected:** u is drawn from q.

**Build 2, ask:** Does log evidence vary with the sampled u or the fitted ψ?

**Expected:** Neither, for this fixed model and dataset.

**Build 3, ask:** What justifies moving it outside?

**Expected:** Linearity of expectation: the average of a constant is that same constant.

**Point here:** Evidence can depend on generative model parameters θ; those are not being fitted in this argument.

**Transition:** Rearrange the equality to put log evidence on the left.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## rearrange: Swap sides and reverse the bracket’s signs

**Build 1, ask:** After rearranging, what is the sign of the joint term?

**Expected:** Positive inside the expectation.

**Build 2, ask:** Which term remains as a nonnegative gap?

**Expected:** The KL from q to the posterior.

**Point here:** Retain both lines so the sign change can be checked visually.

**Transition:** Name the expectation we can estimate using the joint and q.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## name-elbo: A trainable objective emerges from the rearrangement

**Build 1, ask:** Which density has disappeared from the computable expression?

**Expected:** The normalised posterior.

**Build 2, ask:** What do we still need to estimate in a general problem?

**Expected:** The expectation under q; its integrand is available, but the integral may require Monte Carlo.

**Point here:** Say estimate rather than promising every ELBO has a closed form.

**Transition:** Use the nonnegative remaining KL to explain the name “lower bound”.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## bound: One fixed total, two parts

**Build 1, ask:** Why can the ELBO not exceed the log evidence?

**Expected:** The remaining posterior KL is nonnegative.

**Build 2, ask:** When would equality hold?

**Expected:** When q equals the posterior almost everywhere, under the stated support and finiteness assumptions.

**Build 3, ask:** Which optimisation direction improves q?

**Expected:** Maximise ELBO, equivalently minimise negative ELBO.

**Point here:** This gap is KL to the posterior. It is not the KL to the prior that appears in the later VAE decomposition.

**Transition:** Translate this into a practical objective without asking for the unknown evidence.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## what-we-avoid: Optimise the ELBO without computing the evidence

**Build 1, ask:** Is the unknown evidence being set to zero?

**Expected:** No. It remains an unknown fixed constant.

**Build 2, ask:** What exactly makes it safe to omit from ψ’s optimisation?

**Expected:** It does not change when ψ changes, for this fixed model and data.

**Point here:** The optimisation equivalence concerns ψ only. Learning θ later changes the situation.

**Transition:** Return to the coin and implement these available pieces.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-return: Return to the same coin and the same data

**Build 1, ask:** What are we holding fixed from the opening problem?

**Expected:** The same sequence, prior and coin probability r.

**Build 2, ask:** Why retain a known answer if the optimiser does not use it?

**Expected:** To independently check whether the approximation and objective behave as expected.

**Point here:** The sequence has not changed to make an optimisation look better.

**Transition:** First write the exact posterior reference from the sufficient statistics.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-reference: The conjugate posterior is a reference answer

**Build 1, ask:** Which prior parameter receives the heads count?

**Expected:** The first; the tails count is added to the second.

**Build 2, ask:** Does the optimiser need to call this posterior density?

**Expected:** No. The posterior is used only for checking this solvable example.

**Point here:** Use an explicit reference label. Do not silently swap posterior.log_prob into the training objective.

**Transition:** Construct the joint from the prior and the observed-data likelihood.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-prior: Keep the prior’s normalising constant

**Build 1, ask:** Why are the exponents 4 and 2 rather than 5 and 3?

**Expected:** A Beta(a,b) density has exponents a−1 and b−1.

**Build 2, ask:** Can we drop this constant when checking absolute ELBO values?

**Expected:** No. It matters for the numerical value of the joint, ELBO and evidence, even if independent of ψ.

**Point here:** Distinguish a constant irrelevant to an optimiser from a constant required for a correct absolute score.

**Transition:** Add the contributions of all thirty observations.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-likelihood: Each observed toss contributes one log probability

**Build 1, ask:** What contributes log r? What contributes log(1−r)?

**Expected:** Each head contributes log r; each tail contributes log(1−r).

**Build 2, ask:** Why is there no factor counting different orders?

**Expected:** The observed object is one ordered sequence, not the event containing all sequences with that head count.

**Point here:** Counts are sufficient to evaluate this sequence likelihood, but the evidence convention remains sequence-based.

**Transition:** Add prior and likelihood to obtain log joint.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-joint: The joint is available at any proposed r

**Build 1, ask:** Which two available functions appear here?

**Expected:** The likelihood and prior density.

**Build 2, ask:** Where is the unknown evidence in this code?

**Expected:** It does not appear. The joint is evaluated without the posterior normaliser.

**Point here:** log1p(−r) evaluates log(1−r) stably. Our Beta draws lie inside the valid support.

**Transition:** Choose a flexible positive-parameter q and specify what the optimiser stores.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-family: Fit positive Beta shapes through unconstrained parameters

**Build 1, ask:** What family contains our exact posterior in this toy?

**Expected:** The Beta family.

**Build 2, ask:** Which quantities does the optimiser store and update?

**Expected:** The unconstrained log-shape parameters ψ_a and ψ_b; exponentiation produces positive a and b.

**Point here:** r is the unknown probability. a and b are shape parameters. ψ parameterises those shapes.

**Transition:** Draw from q, evaluate the joint-minus-q terms, then average.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-estimator: One mean over draws; the data are already summed

**Build 1, ask:** Where are the thirty observation contributions combined?

**Expected:** Inside log_joint, through 17 log r+13 log(1−r).

**Build 2, ask:** Why is there no additional sum over the 512 draws before mean?

**Expected:** Each term is one sample’s joint-minus-q contribution. We average that sample axis exactly once.

**Point here:** Beta.rsample is differentiable through an implicit reparameterisation method; it is not μ+σε.

**Notebook demonstration:** Notebook section6, training cell: inspect first_step and trace[1]. The posterior reference is absent from the loss.

**Transition:** Inspect one measured step, including the actual gradients and transformed parameters.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## coin-first-step: Inspect one actual update

**Build 1, ask:** Which shape parameter should initially increase for more heads?

**Expected:** a should move up relative to b, shifting q’s centre towards the observed heads proportion.

**Build 2, ask:** Are the printed gradients with respect to a,b or ψ?

**Expected:** They are with respect to ψ, the stored log-shape parameters. Adam then performs its own update.

**Point here:** Do not use a plain SGD formula to explain these Adam parameter changes. The values are directly recorded.

**Transition:** Replay the saved fit rather than waiting for all six hundred steps.

**Source / evidence:** Measured first update · seed2401 · Adam lr=0.025 · 512 Beta draws

## coin-fit: A measured fit using only the joint

**Build 1, ask:** Where does the initial q put its mass?

**Expected:** The initial Beta(10,10) is centred at 0.5 and is too broad.

**Build 2, ask:** What changed by step100?

**Expected:** The measured q has shifted and concentrated towards the reference.

**Build 3, ask:** Is the final q exactly the known posterior?

**Expected:** No. The measured final shapes are about22.394 and16.189, with posterior KL about0.000222. It is close, not exact.

**Point here:** The teal reference was used for diagnostics only. The loss uses log joint and log q.

**Transition:** Use selected candidates to check the identity independently of the training trajectory.

**Source / evidence:** Actual CPU training · 600 Adam steps · seed2401 · saved checkpoints0/100/600

## coin-check: Check the fixed total with three selected candidates

**Build 1, ask:** Add the ELBO and the gap. What is fixed?

**Expected:** The sum is the same log evidence, approximately−21.397.

**Build 2, ask:** Does the better candidate raise the ELBO by the amount its KL falls?

**Expected:** Yes. The identity holds independently for each selected candidate.

**Build 3, ask:** What happens at the exact posterior?

**Expected:** The posterior KL is zero, so ELBO equals log evidence.

**Point here:** These three candidates are checking points, not asserted optimisation checkpoints. ELBO was computed directly from expected joint minus expected log q.

**Transition:** A noisy ELBO estimate behaves differently from this exact expectation.

**Source / evidence:** Direct analytic Beta ELBO and independent posterior KL · selected candidates, not a training trace

## noisy-elbo: An estimate is not itself guaranteed to be a lower bound

**Build 1, ask:** Can a finite-sample ELBO estimate cross the reference line?

**Expected:** Yes. The noisy estimate can lie above the evidence on some draws.

**Build 2, ask:** Which object remains a mathematical lower bound?

**Expected:** The exact expected ELBO, not every finite-sample realisation of its estimator.

**Point here:** This is observed behaviour from the saved run, not a universal crossing frequency.

**Transition:** Separate an improving objective from a certificate of exact inference.

**Source / evidence:** Measured seed2401 training trace · sampled objective versus direct analytic diagnostic

## certificate: A higher or stable ELBO does not prove a zero gap

**Build 1, ask:** What does a true increase in the exact ELBO establish?

**Expected:** For this fixed model and data, posterior KL decreased.

**Build 2, ask:** Does a stable loss establish q=p(u|𝒟)?

**Expected:** No. The family or optimisation can leave a positive gap.

**Point here:** Recall the mean-only Gaussian: even its perfect optimum left a width mismatch.

**Common misconception:** Improvement, convergence of a trace and exact posterior recovery are different claims.

**Transition:** Retrieve the method without looking at the derivation.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## retrieve-samples: For KL(q‖p), where do the samples come from?

**Build 1, ask:** Answer without looking at the earlier definition.

**Expected:** From q, the first argument.

**Build 2, ask:** Why does sampling from p change the question?

**Expected:** It changes the weighting distribution and does not estimate the same directional KL.

**Point here:** Ask several students for the sampling distribution before revealing it.

**Transition:** Now separate the parameters being updated from the random values being drawn.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## retrieve-update: What changes during fixed-model variational inference?

**Build 1, ask:** Which objects change, and in what sense?

**Expected:** ψ is updated; random draws vary; data, model and exact posterior remain fixed.

**Build 2, ask:** Is a new draw an optimiser parameter update?

**Expected:** No. A draw is a sampled value, not a stored parameter being fitted.

**Point here:** Distinguish randomness in evaluating an objective from changes in the objective’s parameters.

**Transition:** Explain the omitted evidence using the variable it is constant with respect to.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## retrieve-evidence: Why can we omit the evidence from ψ’s optimisation?

**Build 1, ask:** Is the reason that evidence is zero, easy, or constant?

**Expected:** It is constant, not zero and not necessarily easy to evaluate.

**Build 2, ask:** Constant with respect to which parameters?

**Expected:** ψ, for the fixed model and observed data used in this inference problem.

**Point here:** Ask students to state both the changing parameter and what is held fixed.

**Transition:** Apply the procedure to a new inference problem.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## exit-transfer: A sensor has an unknown constant bias

**Build 1, ask:** Identify the unknown, observations and one possible q family.

**Expected:** u is the bias; 𝒟 contains observed sensor values; a Normal q is one sensible choice.

**Build 2, ask:** Which available functions form the joint?

**Expected:** The prior density and the likelihood of each reading.

**Build 3, ask:** What objective would you implement?

**Expected:** The expectation of log joint minus log q, maximised over ψ.

**Point here:** This is an implementation-identification exercise, not another numerical derivation or a claim that approximation is required for this conjugate toy.

**Transition:** Connect this posterior-fitting problem to the models we will build next.

**Source / evidence:** Computed teaching example · natural logarithms · KL in nats

## next-lecture: Generative models will create this inference problem again

**Build 1, ask:** Where might a new unknown appear in an image model?

**Expected:** A latent description z can explain an observed image x.

**Build 2, ask:** What question should we ask about a good reconstruction?

**Expected:** Whether reconstructing observations also gives a useful way to generate new ones.

**Point here:** Lecture2 introduces generative modelling and ordinary autoencoders. Lecture3 returns to variational inference with an encoder and decoder.

**Transition:** Close the main route here. Optional derivations follow for reading or extra time.

**Source / evidence:** Main70-minute route ends here · optional depth follows

## optional-normal-algebra: Where does the complete Gaussian KL formula come from? (optional)

**Build 1, ask:** Which log-normalising constants cancel?

**Expected:** The common log√(2π) terms cancel, leaving the log scale ratio.

**Build 2, ask:** Which two moments are needed?

**Expected:** The target-centred and q-centred second moments.

**Build 3, ask:** Where does the final −1/2 come from?

**Expected:** The expectation of (u−μ)²/(2σ²) equals one half.

**Point here:** This is an optional completion of the Gaussian algebra. The core route exposed only its decisive moment.

**Transition:** Nonnegativity belongs to the expectation, not every local ratio.

**Source / evidence:** Analytic derivation for positive Normal scales

## optional-nonnegative: Why is exact KL nonnegative? (optional)

**Build 1, ask:** Which function makes Jensen’s inequality useful?

**Expected:** The convex function −log.

**Build 2, ask:** Why does the inequality point this way?

**Expected:** For a convex function, its expectation is at least its value at the expectation.

**Build 3, ask:** What makes the final expectation one?

**Expected:** For positive densities on the same support, integrating q·p/q gives the normalisation integral of p.

**Point here:** State the common-support assumption for this simple proof. More general support cases require the standard extension, including infinite KL when needed.

**Transition:** A valid alternative estimator can handle sampling derivatives without a pathwise transform.

**Source / evidence:** Jensen proof · positive densities on the same support and finite terms

## optional-score-function: Pathwise gradients are one route, not the only route (optional)

**Build 1, ask:** Which term was missing from naive detached-sample differentiation?

**Expected:** The term accounting for the change in the probability of drawing u.

**Build 2, ask:** Does this identity require u=μ+σε?

**Expected:** No. It is a different route, under conditions permitting differentiation under the integral.

**Point here:** The displayed identity includes explicit parameter dependence of f. Do not silently discard that term for a general integrand.

**Transition:** The event and sample dimensions still need correct reductions, whichever estimator is used.

**Source / evidence:** Differentiation-under-the-integral identity · regularity assumptions apply

## optional-reductions: A mean must say which dimension it averages (optional)

**Build 1, ask:** Does every log_prob output still have an event dimension?

**Expected:** No. A multivariate distribution’s log_prob typically already reduces its event dimensions.

**Build 2, ask:** What goes wrong if you sum S before subtracting log q from the joint?

**Expected:** It changes the relative weighting, often multiplying an entropy contribution by the number of samples.

**Point here:** This corrects a concrete reduction pitfall in the historical multivariate regression extension.

**Transition:** Even harmless-looking likelihood constants affect absolute evidence checks.

**Source / evidence:** Shape contract · scalar coin and multivariate event distinction

## optional-count-data: An ordered sequence and a count have different evidence (optional)

**Build 1, ask:** Which event contains more possible sequences?

**Expected:** The count-only event includes every ordering with17 heads.

**Build 2, ask:** Does this constant change the posterior or its KL gap?

**Expected:** No. It adds the same log binomial coefficient to joint, ELBO and evidence; the posterior and gap remain unchanged.

**Point here:** The entire main lecture consistently uses the ordered-sequence convention.

**Transition:** The Beta family also provides analytic expectations that independently check the sampled objective.

**Source / evidence:** Exact convention comparison · log binomial coefficient computed with log-gamma

## optional-beta-elbo: Check the sampled objective with analytic Beta moments (optional)

**Build 1, ask:** Which two expectations determine the coin ELBO?

**Expected:** Expected log r and expected log(1−r).

**Build 2, ask:** Why write digamma as a name rather than ψ here?

**Expected:** ψ already denotes free variational parameters.

**Build 3, ask:** How was the identity checked non-circularly?

**Expected:** The notebook computes ELBO directly from these expectations, then independently adds the library posterior KL and checks the evidence.

**Point here:** The direct calculation is not evidence minus KL disguised as an independent check. Numerical integration supplies another check.

**Transition:** The notebook contains the complete formulas and reproducible outputs.

**Source / evidence:** Analytic Beta moments · direct ELBO, independent KL and quadrature checks
