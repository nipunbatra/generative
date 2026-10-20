# KL and variational inference: speaker notes

Planned main route: 70 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Act 0 · The wall | 8 |
| Act 1 · Closer how? | 15 |
| Act 2 · Make q move | 7 |
| Act 3 · No formula | 7 |
| Act 4 · Gradients through a draw | 10 |
| Act 5 · The real target | 10 |
| Act 6 · Back to the wall | 10 |
| Exit | 3 |

## coin-question: Is this coin fair?

**Build 1, ask:** Vote: is this coin fair? How sure are you?

**Expected:** Most students hedge: 17 of 30 is a little high, but fair seems plausible. The hedge is the point.

**Build 2, ask:** What would a complete answer look like?

**Expected:** A posterior distribution over r: a centre and a spread, not one number.

**Point here:** PAUSE 1 · the hook. Take the vote by hands and keep it on the board: Act 6 answers it.

**Transition:** For one convenient prior, Bayes’ rule answers this exactly.

## conjugate-luck: With a Beta prior, Bayes’ rule has a formula

**Build 1, ask:** Where does the posterior sit, and how wide is it?

**Expected:** Near 22/38 ≈ 0.58, and much narrower than the prior.

**Build 2, ask:** Why can we write p(𝒟) down here?

**Expected:** Prior and likelihood share the form r^a(1−r)^b, so the integral is a Beta function. Change either and the luck runs out.

**Point here:** The evidence p(𝒟) is the integral of prior × likelihood over r. Keep it blue throughout the lecture.

**Transition:** Keep the data. Change only the prior.

## no-formula: Change the prior, and the formula disappears

**Build 1, ask:** What changed, and what stayed the same?

**Expected:** Only the prior. Data, likelihood and question are unchanged.

**Build 2, ask:** Can you integrate this in closed form?

**Expected:** No. A dense grid gives log p(𝒟) ≈ −21.62, but only because w is a single number.

**Point here:** Log-odds priors are what logistic regression and neural networks use, so this is the normal case, not a contrived one.

**Transition:** Why not always use a grid?

## grid-explodes: A grid works for one unknown, not for a million

**Build 1, ask:** How many points for 3 unknowns? For 10?

**Expected:** 10⁶ and 10²⁰.

**Build 2, ask:** Could faster computers rescue the grid?

**Expected:** No. The count grows exponentially with the number of unknowns.

**Transition:** Three families of methods get around the integral.

## three-ways: Three ways around an integral nobody can do

**Build 1, ask:** What does Laplace miss here?

**Expected:** The second bump: it only looks at the peak.

**Build 2, ask:** What does MCMC cost?

**Expected:** Many steps, each evaluating the model on the data; hard to scale to large models and datasets.

**Build 3, ask:** What does VI trade away?

**Expected:** Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.

**Point here:** Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.

**Transition:** The plan for VI, and what stands in its way.

## plan: The plan: push a simple q towards the posterior

**Build 1, ask:** What must we define before we can minimise anything?

**Expected:** A number for the “distance” between two distributions.

**Build 2, ask:** Which obstacle looks hardest?

**Expected:** Usually the last: the posterior we are approaching is the thing we cannot evaluate.

**Point here:** u is the generic unknown from now on. For the coin, u is the log-odds w.

**Transition:** Obstacle 1: what does “closer” mean for distributions?

## fruit: Two surveys: how different are they?

**Build 1, ask:** How would you score the difference between these two surveys?

**Expected:** Students propose differences of percentages. Note that apples, 50% both years, should contribute nothing.

**Build 2, ask:** Bananas ÷4, clementines ×4: same size of change?

**Expected:** By ratio, yes: both are a factor of 4. That suggests comparing ratios.

**Point here:** p is the survey we weight by (this year); q is the reference (last year).

**Transition:** Compare by ratios.

## ratios-logs: Ratios say “how many times”; logs make up and down cancel

**Build 1, ask:** On the ratio scale, which change looks bigger?

**Expected:** The rise: +3 against −0.75, though both are fourfold.

**Build 2, ask:** After taking logs?

**Expected:** Equal and opposite, ±1.39. Logs turn “times” into “plus”.

**Transition:** Now combine the three log ratios into one number.

## weight: Weight by how common each fruit is now: that is KL

**Build 1, ask:** Why weight by p rather than q?

**Expected:** That choice defines KL(p‖q): the first argument does the weighting. Swapping gives a different number.

**Build 2, ask:** What is the KL if the surveys agree?

**Expected:** Zero: every ratio is 1, every log is 0.

**Point here:** Natural log gives nats. Base 2 gives bits, which the next slide uses.

**Transition:** The same number has a second meaning, in bits.

## bits: KL is the extra message length of using the wrong code

**Build 1, ask:** Why does sunny get the shortest code?

**Expected:** Sunny is the most common, so a short code for it saves the most bits on average.

**Build 2, ask:** Where do the extra 0.25 bits come from?

**Expected:** Sunny is half of all days, and the wrong code spends 2 bits on it. KL measures that waste.

**Transition:** Cross-entropy is this “wrong code” length, and you have minimised it many times.

## used-before: You have minimised KL before: in every cross-entropy loss

**Build 1, ask:** Which term of the split depends on q?

**Expected:** Only the KL. H(p) is fixed by p.

**Build 2, ask:** What does your classifier’s loss minimise?

**Expected:** KL(labels ‖ predictions), plus a constant.

**Build 3, ask:** So what is new in VI?

**Expected:** The direction. VI puts the approximation first, KL(q‖p). Act 1 ends with why.

**Transition:** What does a KL of 0.19 actually feel like?

## coin-evidence: A KL value is how fast evidence piles up

**Build 1, ask:** Predict: after 30 tosses, how many times more likely are the data under 0.8 than under 0.5?

**Expected:** Most guess far too low. It is about 325×.

**Build 2, ask:** Why does a small KL eventually matter so much?

**Expected:** Log evidence adds up toss by toss, so the ratio grows exponentially with n.

**Point here:** PAUSE 2. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of 5.780, against 30 × KL = 5.782.

**Transition:** Sums become integrals for continuous unknowns.

## densities: For densities, sums become integrals, and KL ≥ 0

**Build 1, ask:** Which distribution supplies the weights in KL(q‖p)?

**Expected:** q, the first argument.

**Build 2, ask:** Why can’t KL be negative?

**Expected:** log t ≤ t − 1 for every ratio t = p/q, and the average of p/q under q is ∫p = 1.

**Point here:** Individual log ratios can be negative; only the weighted average is guaranteed nonnegative. That returns with sampling in Act 3.

**Transition:** Back to the PML vote, now with a score.

## compare: Which Gaussian is closer to p?

**Build 1, ask:** Vote: q₁ or q₂?

**Expected:** Most pick q₂: same centre and a more similar width.

**Build 2, ask:** Does the score agree with your eye?

**Expected:** Yes: 0.07 < 0.35. The two-Gaussian formula is derived in the optional slides.

**Point here:** PAUSE 3 (PML L23 08:30). Normal(loc, scale) takes a standard deviation; N(μ, σ²) writes a variance.

**Transition:** Where does each number come from? Watch it as an area.

## kl-area: KL is an area you can watch

**Build 1, ask:** Make q narrow and centred on p. What happens in each direction?

**Expected:** KL(q‖p) stays moderate: q only visits places where p is reasonable. KL(p‖q) explodes: p’s tails land where q is nearly zero.

**Point here:** Click q₁, then q₂, then flip to KL(p‖q). Then shrink σ in each direction.

**Transition:** So which direction should VI use?

## why-reverse: VI uses KL(q‖p) because it needs only what we have

**Build 1, ask:** Which of these could we compute for a posterior?

**Expected:** Only KL(q‖p).

**Build 2, ask:** What might that price look like for a posterior with two bumps?

**Expected:** q may sit on one bump and ignore the other.

**Transition:** Watch that price on two bumps.

## fwdrev: The price: reverse KL picks one bump

**Build 1, ask:** Predict each final q.

**Expected:** Forward: wide, centred between the bumps. Reverse: one bump, depending on the start.

**Build 2, ask:** Which fit would make you overconfident?

**Expected:** The reverse fits: each reports one bump with a narrow spread.

**Point here:** PAUSE 4. Narrow bumps: forward KL 0.72; reverse 0.36 (= −log 0.7) from the left and 1.20 (= −log 0.3) from the right. Then switch to wide bumps: overlapping bumps give one broad reverse-KL fit.

**Transition:** Act 2: a score picks a winner. Can it improve a loser?

## score-to-search: A score picks a winner. Can it improve a loser?

**Build 1, ask:** What stays fixed, and what does the optimiser change?

**Expected:** p stays fixed; μ and ρ change.

**Build 2, ask:** Why not update σ directly?

**Expected:** A step could make σ zero or negative, which is not a valid Normal.

**Point here:** The literal one-step mean update, μ⁺ = 0.25, is in the optional slides.

**Transition:** Press play.

## olive: Press play: q slides onto p

**Build 1, ask:** Predict: does the centre or the width arrive first?

**Expected:** Students split. The width reaches 4 around step 45, overshoots to 4.52, then settles; μ arrives by step 100.

**Build 2, ask:** Why can both move at once?

**Expected:** Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.

**Point here:** PAUSE 5 (PML L23 12:13). Adam, learning rate 0.05, analytic KL. Notebook section 3.

**Transition:** One dimension is easy. What happens with two correlated unknowns?

## two-d: In 2-D, a q that cannot tilt shrinks instead

**Build 1, ask:** Predict the diagonal q’s widths: 1, more or less?

**Expected:** Less: about 0.44, which is √(1 − 0.9²).

**Build 2, ask:** Why shrink rather than spread?

**Expected:** KL(q‖p) heavily penalises q mass where p is tiny, which is off the diagonal ridge.

**Build 3, ask:** What does the full covariance cost?

**Expected:** More parameters: d(d + 1)/2 for L instead of d. For a million weights that is far too many.

**Point here:** PAUSE 6. Measured fit: both widths 0.4359, KL 0.8304, matching √(1 − ρ²) and −½ log(1 − ρ²).

**Transition:** Act 3: the library had a formula for every KL so far. What if it doesn’t?

## mixture-no-formula: PyTorch has no formula for this KL. The definition still works.

**Build 1, ask:** What would kl_divergence need for this pair?

**Expected:** A closed-form expression for a Gaussian against a mixture. There isn’t one.

**Build 2, ask:** Is the KL undefined, then?

**Expected:** No. It is a perfectly good integral; we just cannot write it in closed form.

**Point here:** Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch 2.13.0).

**Transition:** Estimate the average by sampling.

## mc-estimate: Draw from q, average the log ratio

**Build 1, ask:** Which distribution do the draws come from?

**Expected:** q, the first argument: it does the weighting.

**Build 2, ask:** Should the estimate match exactly?

**Expected:** No. It is random: close, and closer with more draws.

**Point here:** Validated on q₂ = N(1, 3²) against p = N(1, 4²), seed 1000. Individual terms can be negative; see the optional slide.

**Transition:** How much does the estimate wobble?

## mc-variability: More samples, tighter estimates

**Build 1, ask:** How many more draws halve the spread?

**Expected:** Four times as many: the spread scales as 1/√S.

**Point here:** Same p and q at every size; 40 seeds each.

**Transition:** Act 4: we can estimate the loss. Now optimise it.

## now-optimise: Now optimise it: the code runs, and q does not move

**Build 1, ask:** Predict: where is q after 400 steps?

**Expected:** Most expect it to slide onto the mixture. It does not move.

**Build 2, ask:** The code runs. What could be wrong?

**Expected:** The gradient: it must be missing a piece.

**Point here:** PAUSE 7 (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.

**Transition:** Look at the gradient on a case we can solve.

## missing-path: The gradient that would move q never reaches it

**Build 1, ask:** Which arrow can the gradient not travel along?

**Expected:** The sample arrow from q to u.

**Build 2, ask:** Why isn’t the surviving gradient exactly zero?

**Expected:** log q(u) still contains μ and σ explicitly. That part averages to zero over draws; what remains is noise.

**Point here:** Gaussian pair q = N(0, 2²), p = N(1, 4²), 50,000 shared draws. A score-function estimator is another valid fix (optional slide).

**Transition:** Give the gradient a path.

## reparam: Move the randomness outside: u = μ + σε

**Build 1, ask:** Is u still distributed as N(μ, σ²)?

**Expected:** Yes: shifting and scaling a standard Normal gives N(μ, σ²).

**Build 2, ask:** Compute u and both derivatives.

**Expected:** u = 0 + 2(−0.75) = −1.5; ∂u/∂μ = 1; ∂u/∂σ = ε = −0.75.

**Build 3, ask:** Can ∂u/∂σ be negative?

**Expected:** Yes, whenever ε < 0: widening q pushes this draw further left.

**Point here:** PAUSE 8 (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.

**Transition:** Change one word in the loop.

## rsample-fit: Change one word: rsample() fits the mixture

**Build 1, ask:** Predict where q settles.

**Expected:** Over the mixture’s mass, between and around the two bumps.

**Build 2, ask:** Why doesn’t the KL reach 0?

**Expected:** No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct optimisation.

**Point here:** Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Acts 3 and 4.

**Transition:** Act 5: every loss so far evaluated log p. For a posterior, we cannot.

## posterior-target: The real target is a posterior we cannot evaluate

**Build 1, ask:** Which pieces of this KL can we evaluate?

**Expected:** Draws from q, and log q at those draws.

**Build 2, ask:** Which piece is missing?

**Expected:** log p(u | 𝒟): it needs the evidence p(𝒟).

**Transition:** Substitute Bayes’ rule and see what survives.

## bayes-sub: Substitute Bayes’ rule and see what survives

**Build 1, ask:** Substitute and simplify.

**Expected:** −log p(u | 𝒟) = −log p(𝒟, u) + log p(𝒟).

**Build 2, ask:** Why can log p(𝒟) leave the expectation?

**Expected:** It is a constant: it depends on neither u nor q.

**Build 3, ask:** Which term can we compute?

**Expected:** The ELBO: draws from q, log q and the log joint.

**Point here:** PAUSE 9 (PML L23 47:15). Students do the substitution on paper before build 2.

**Transition:** What does that buy us in code?

## no-evidence-needed: Maximise the ELBO; the evidence never appears

**Build 1, ask:** Where is the evidence in this code?

**Expected:** Nowhere.

**Build 2, ask:** What does raising the ELBO by 1 do to the KL?

**Expected:** Lowers it by exactly 1, since their sum is fixed.

**Transition:** What does the ELBO reward?

## elbo-meaning: Two forces: sit where the joint is high, stay spread out

**Build 1, ask:** What would maximising only the first term do?

**Expected:** Collapse q onto the peak of the joint: a point estimate, the MAP.

**Build 2, ask:** Which q wins, and by how much?

**Expected:** The balanced one: ELBO −21.62 against −22.89 and −25.26.

**Point here:** Logit coin from Act 0; q centred at the VI solution, widths 0.06, the VI width and 1.2. Expectations by quadrature.

**Transition:** Put the two pieces of the identity in one picture.

## identity: One fixed total: raising the ELBO lowers the KL

**Build 1, ask:** What happens to the KL when the ELBO rises by 1?

**Expected:** It falls by 1.

**Build 2, ask:** Can the ELBO pass log p(𝒟)?

**Expected:** No: that would need a negative KL.

**Point here:** These three rows are q at steps 0, 10 and 600 of the next slide’s fit.

**Transition:** Act 6: back to the coin with no formula.

## logit-vi: Back to the coin with no formula

**Build 1, ask:** Predict: will the ELBO reach the dashed line? Cross it?

**Expected:** It approaches from below and stops just short. It can never cross.

**Build 2, ask:** The grid was used only to check. What did the optimiser use?

**Expected:** Only draws from q, log q and the log joint.

**Point here:** PAUSE 10. 256 rsample draws per step, Adam 0.05, 600 steps; exact ELBO by quadrature, for the plot only. Notebook section 8.

**Transition:** Answer the opening vote.

## coin-fair: So, is the coin fair?

**Build 1, ask:** Is the coin fair?

**Expected:** We cannot rule it out: about a 1 in 4 chance that r < 0.5, under this model and prior.

**Build 2, ask:** How did VI get this without p(𝒟)?

**Expected:** By maximising the ELBO, which needs only the joint.

**Point here:** q in r-space includes the Jacobian 1/(r(1 − r)). The exact tail probability comes from the grid posterior.

**Transition:** Check the method on a problem with an answer key.

## answer-key: A check with an answer key: the Beta coin

**Build 1, ask:** Where does Beta(10, 10) put its mass?

**Expected:** Centred at 0.5, and too broad.

**Build 2, ask:** What changed by step 100?

**Expected:** It shifted towards the heads proportion and narrowed.

**Build 3, ask:** Is the final q exactly Beta(22, 16)?

**Expected:** No, but very close: KL ≈ 0.0002. The answer key was never used by the optimiser.

**Point here:** Beta.rsample is differentiable through implicit reparameterisation, not μ + σε. Adam 0.025, 512 draws, seed 2401.

**Transition:** Move q yourself and watch the total.

## elbo-slider: Move q yourself: ELBO + KL never changes

**Build 1, ask:** Find the q with the highest ELBO. What is its KL?

**Expected:** Beta(22, 16): ELBO = log p(𝒟) and KL = 0.

**Point here:** ELBO from analytic Beta moments; KL from its own formula; their sum is checked against the evidence in check-science.mjs.

**Transition:** Collect the five tools.

## five-tools: Five obstacles, five tools

**Build 1, ask:** Cover the middle column. Which tool removed which obstacle?

**Expected:** Students reconstruct it row by row.

**Transition:** Try the method on a new problem.

## exit-transfer: A sensor has an unknown constant bias

**Build 1, ask:** Identify the unknown, the observations and one possible q family.

**Expected:** u is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.

**Build 2, ask:** Which available functions form the joint?

**Expected:** The prior density and the likelihood of each reading.

**Build 3, ask:** What objective would you implement?

**Expected:** The expectation of log joint minus log q, maximised over ψ.

**Point here:** This conjugate toy has an exact answer too; the exercise is identifying the pieces, not needing approximation.

**Transition:** Where these unknowns appear next.

## next-lecture: Generative models create this inference problem again

**Build 1, ask:** Where might a new unknown appear in an image model?

**Expected:** A latent description z can explain an observed image x.

**Build 2, ask:** What should we ask of a good reconstruction?

**Expected:** Whether reconstructing observations also gives a useful way to generate new ones.

**Transition:** Close the main route. Optional depth follows.

## optional-one-step: One literal update of the mean (optional)

**Build 1, ask:** Which term of the Gaussian KL depends on μ?

**Expected:** Only the squared shift, (μ − 1)²/(2·4²).

**Build 2, ask:** Differentiate it at μ = 0.

**Expected:** (μ − 1)/16 = −0.0625: a negative derivative, so descent moves μ right.

**Build 3, ask:** Take one step with η = 4. What is the best KL with σ stuck at 2?

**Expected:** μ⁺ = 0.25. Even at μ = 1 the KL is 0.318: the width mismatch needs σ to move too.

**Point here:** η = 4 is chosen to make the arithmetic visible, not a recommended learning rate.

## optional-normal-algebra: Where does the Gaussian KL formula come from? (optional)

**Build 1, ask:** Which log-normalising constants cancel?

**Expected:** The common log√(2π) terms cancel, leaving the log scale ratio.

**Build 2, ask:** Which two moments are needed?

**Expected:** The target-centred and q-centred second moments.

**Build 3, ask:** Where does the final −1/2 come from?

**Expected:** The expectation of (u−μ)²/(2σ²) equals one half.

**Point here:** q = N(μ, σ²), p = N(m, s²). Every scale symbol is a standard deviation.

## optional-nonnegative: KL ≥ 0 by Jensen’s inequality (optional)

**Build 1, ask:** Which function makes Jensen’s inequality useful?

**Expected:** The convex function −log.

**Build 2, ask:** Why does the inequality point this way?

**Expected:** For a convex function, its expectation is at least its value at the expectation.

**Build 3, ask:** What makes the final expectation one?

**Expected:** Integrating q · p/q gives the integral of p, which is 1 on a common support.

**Point here:** The tangent-line argument in the main route is the same fact without naming Jensen.

## optional-negative-estimate: Five draws can give a negative KL estimate (optional)

**Build 1, ask:** Some rows are negative. Could their average be negative?

**Expected:** Yes: this five-draw average is about −0.093.

**Build 2, ask:** Does this contradict KL ≥ 0?

**Expected:** No. A finite-sample estimate is not the exact expectation.

**Point here:** Seed 0 was chosen deliberately as the first negative example in a seed search.

## optional-mixture-candidates: The two PML candidates, scored by Monte Carlo (optional)

**Build 1, ask:** How large is the difference relative to the sampling uncertainty?

**Expected:** Far larger than the standard errors.

**Build 2, ask:** What role does the numerical integral play?

**Expected:** An independent check, possible only because this toy is one-dimensional.

**Point here:** 100,000 draws each, seeds 402 and 403.

## optional-rsample-check: A fixed-noise finite difference checks the path (optional)

**Build 1, ask:** What is different about rsample?

**Expected:** It returns a draw with the pathwise gradient connection.

**Build 2, ask:** What must stay fixed on both sides of the finite difference?

**Expected:** The same noise array. Fresh noise would confound the check with sampling variation.

**Point here:** Seed 88, S = 2048, h = 10⁻⁵.

## optional-score-function: Pathwise gradients are one route, not the only one (optional)

**Build 1, ask:** Which term was missing from the naive sample() gradient?

**Expected:** The term accounting for the change in the probability of drawing u.

**Build 2, ask:** Does this identity require u = μ + σε?

**Expected:** No. It works for discrete q too, usually with higher variance.

**Point here:** Regularity conditions allow differentiating under the integral.

## optional-noisy-elbo: A noisy ELBO estimate can cross the evidence (optional)

**Build 1, ask:** Can a finite-sample ELBO estimate cross the reference line?

**Expected:** Yes, on some draws.

**Build 2, ask:** Which object remains a lower bound?

**Expected:** The exact expected ELBO, not each sampled estimate.

**Point here:** Beta coin, seed 2401.

## optional-certificate: A higher ELBO does not prove a zero gap (optional)

**Build 1, ask:** What does a true increase in the exact ELBO establish?

**Expected:** For this fixed model and data, the posterior KL decreased.

**Build 2, ask:** Does a stable loss establish q = p(u | 𝒟)?

**Expected:** No. The family or optimisation can leave a positive gap.

**Point here:** Recall the diagonal q in 2-D: its best possible KL was 0.83.

## optional-coin-check: Check the fixed total with three Beta candidates (optional)

**Build 1, ask:** Add the ELBO and the gap. What is fixed?

**Expected:** The sum is the same log evidence, about −21.397.

**Build 2, ask:** Does the better candidate raise the ELBO by the amount its KL falls?

**Expected:** Yes. The identity holds for each candidate.

**Build 3, ask:** What happens at the exact posterior?

**Expected:** The KL is zero, so the ELBO equals the log evidence.

**Point here:** ELBO computed directly from expected log joint minus expected log q; KL computed separately.

## optional-mean-field-nets: One Gaussian per weight: Bayesian neural networks (optional)

**Build 1, ask:** How many parameters does q have for a network with d weights?

**Expected:** 2d: a mean and a width per weight. A full covariance would need about d²/2.

**Build 2, ask:** How would you train it?

**Expected:** Exactly as today: rsample the weights, estimate the ELBO, backpropagate.

**Point here:** Mean-field: the diagonal q from Act 2, so expect underestimated uncertainty.

## optional-reductions: A mean must say which dimension it averages (optional)

**Build 1, ask:** Does every log_prob output still have an event dimension?

**Expected:** No. A multivariate distribution’s log_prob already reduces its event dimensions.

**Build 2, ask:** What goes wrong if you sum over S before averaging?

**Expected:** It rescales the objective, often multiplying an entropy contribution by the number of samples.

## optional-count-data: An ordered sequence and a count have different evidence (optional)

**Build 1, ask:** Which event contains more possible sequences?

**Expected:** The count-only event includes every ordering with 17 heads.

**Build 2, ask:** Does this constant change the posterior or its KL gap?

**Expected:** No. It adds the same constant to the joint, the ELBO and the evidence.

**Point here:** The whole lecture uses the ordered-sequence convention.

## optional-beta-elbo: Check the sampled objective with analytic Beta moments (optional)

**Build 1, ask:** Which two expectations determine the coin ELBO?

**Expected:** Expected log r and expected log(1−r).

**Build 2, ask:** Why write digamma rather than ψ here?

**Expected:** ψ already denotes the variational parameters.

**Build 3, ask:** How was the identity checked non-circularly?

**Expected:** The ELBO is computed directly from these moments, then the separately computed KL is added and compared with the evidence.
