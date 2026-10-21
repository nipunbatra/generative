# KL and variational inference: speaker notes

Planned main route: 75 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Part 1 · Bayes’ rule for a neural network | 16 |
| Part 2 · How far apart are two distributions? | 20 |
| Part 3 · Find the closest q | 22 |
| Part 4 · Back to Bayes: the ELBO | 9 |
| Part 5 · Implement it | 8 |

## bayes-rule: Bayes’ rule: update what you believe, using data

**Build 1, ask:** Which term is the belief before seeing B?

**Expected:** p(A), the prior.

**Build 2, ask:** Which term only rescales?

**Expected:** The evidence p(B): it does not depend on A; it makes the posterior sum to one.

**Transition:** In machine learning, what are A and B?

## bayes-ml: In machine learning, the unknown is the parameters

**Build 1, ask:** What plays the role of A? Of B?

**Expected:** A is θ, the parameters; B is the dataset 𝒟 of (x, y) pairs.

**Build 2, ask:** What does a distribution over θ buy us?

**Expected:** Uncertainty: predictions with error bars, by averaging over all plausible θ.

**Transition:** Make θ concrete with the smallest interesting network.

## tiny-net: A tiny network: one input, two hidden units, one output

**Build 1, ask:** How many weights? How many biases?

**Expected:** Four weights (w₁, w₂, v₁, v₂) and three biases (b₁, b₂, c).

**Build 2, ask:** What would a bigger network add?

**Expected:** Each extra hidden unit adds a weight in, a bias and a weight out. Real networks have millions of parameters.

**Transition:** Before any data: a prior on all seven.

## priors: A prior on every parameter

**Build 1, ask:** Why independent N(0, 1) priors?

**Expected:** Simplicity: before data nothing ties one weight to another, and very large weights are unlikely.

**Build 2, ask:** What do these prior functions have in common?

**Expected:** They are smooth: flat, S-shaped or gently curved, mostly between −3 and 3.

**Transition:** Now the data, and how well each θ explains them.

## data-likelihood: The data, and how well one θ explains them

**Build 1, ask:** What does the 0.1 describe?

**Expected:** The noise standard deviation, assumed known here.

**Build 2, ask:** Why a product?

**Expected:** Given θ, the noise on different points is independent.

**Point here:** Data: y = 0.8 tanh(3x) + 0.3x + N(0, 0.1²) noise, seed 7.

**Transition:** Compute the right-hand side of Bayes’ rule for one θ.

## rhs-by-hand: The right-hand side, term by term, for one θ

**Build 1, ask:** Why add logs instead of multiplying densities?

**Expected:** A product of many small numbers underflows; a sum of logs is stable, and has the same maximiser.

**Build 2, ask:** Which parameter costs most under the prior?

**Expected:** The largest in size: w₂ = −2.00, costing −2.93.

**Build 3, ask:** How would you compute the third term?

**Expected:** Students suggest a grid, or sampling. Both are next.

**Point here:** θ_A = (−0.48, −2.00, −0.59, 0.45, −0.46, −0.96, −0.16) is the most probable θ: it maximises log p(𝒟 | θ) + log p(θ). Each likelihood term is −½ log(2π · 0.1²) − (yᵢ − f(xᵢ))² / (2 · 0.1²) = 1.38 − 50 (yᵢ − f(xᵢ))².

**Transition:** Compare θ_A with a random θ from the prior.

## two-thetas: Two settings of θ, side by side

**Build 1, ask:** Which θ explains the data better, and by how much?

**Expected:** θ_A, by about 2,170 nats of log likelihood.

**Build 2, ask:** How many times more probable is θ_A than θ_B after seeing the data?

**Expected:** About e²¹⁷⁰ times. p(𝒟) cancels in a ratio of posteriors: it is needed only to normalise.

**Transition:** Normalising is the hard part.

## evidence-integral: The evidence needs every θ, and has no formula

**Build 1, ask:** Why is there no formula?

**Expected:** The integrand is not a Gaussian or any other form we can integrate: θ passes through tanh.

**Build 2, ask:** Could a faster computer rescue the grid?

**Expected:** No: the count grows exponentially with the number of parameters.

**Transition:** Sampling instead of a grid?

## evidence-mc: Sampling from the prior: a few lucky draws carry everything

**Build 1, ask:** Why are the small-S estimates so low?

**Expected:** Prior draws almost never land where the likelihood is high, like θ_B; the rare good draws dominate the average.

**Build 2, ask:** What happens with a million parameters?

**Expected:** The chance that a prior draw lands near the posterior shrinks exponentially; no feasible S is enough.

**Point here:** Effective number of draws, out of 10⁷: 77, 93, 134, 108. The largest single draw carries 8%, 4%, 4%, 4% of the total.

**Transition:** Why do we want the posterior at all?

## predictive: What we want in the end: predictions with error bars

**Build 1, ask:** Where should the model be least sure?

**Expected:** Far from the data: beyond both ends, and in the gap.

**Build 2, ask:** How would you compute this integral?

**Expected:** Average f_θ(x*) over samples of θ from the posterior, which we cannot draw directly. Yet.

**Point here:** Reference: Hamiltonian Monte Carlo, 4 chains of 2,000 samples, acceptance about 0.90. Predictive sd 0.22 at x = −2.5, about 0.11 in the data.

**Transition:** How else could we approximate the posterior?

## three-ways: Three ways to approximate a posterior

**Build 1, ask:** What does Laplace miss here?

**Expected:** The second bump: it only looks at the peak.

**Build 2, ask:** What does MCMC cost?

**Expected:** Many steps, each evaluating the model on the data; hard to scale to large models and datasets.

**Build 3, ask:** What does VI trade away?

**Expected:** Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.

**Point here:** Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.

**Transition:** Today: variational inference.

## vi-idea: Variational inference: choose q, make it close to the posterior

**Build 1, ask:** How many numbers does q have?

**Expected:** Fourteen: a mean and a width for each of the seven parameters.

**Build 2, ask:** Why is drawing θ from q easy?

**Expected:** Each θⱼ is an independent Gaussian draw.

**Build 3, ask:** What do we need before we can minimise anything?

**Expected:** A number for the distance between two distributions. That is Part 2.

**Transition:** Part 2: how far apart are two distributions?

## distance-ideas: How far apart are two distributions?

**Build 1, ask:** Are p and q the same distribution?

**Expected:** No: one bump against two.

**Build 2, ask:** What would you compare instead?

**Expected:** The densities at every point: how many times more likely each u is under one than under the other, averaged. That leads to KL.

**Point here:** q = ½·N(−0.8, 0.6²) + ½·N(0.8, 0.6²): mean 0 and variance 0.6² + 0.8² = 1.

**Transition:** Build that number from a familiar example.

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

**Point here:** PAUSE. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of 5.780, against 30 × KL = 5.782.

**Transition:** Sums become integrals for continuous unknowns.

## densities: For densities, sums become integrals, and KL ≥ 0

**Build 1, ask:** Which distribution supplies the weights in KL(q‖p)?

**Expected:** q, the first argument.

**Build 2, ask:** Why can’t KL be negative?

**Expected:** log t ≤ t − 1 for every ratio t = p/q, and the average of p/q under q is ∫p = 1.

**Point here:** Individual log ratios can be negative; only the weighted average is guaranteed nonnegative. That returns with sampling in Act 3.

**Transition:** Closed forms for the distributions we will meet.

## kl-formulas: Closed forms for simple distributions

**Build 1, ask:** Which row will our network’s ELBO use?

**Expected:** The diagonal-Gaussian row: KL from q(θ) to the N(0, 1) prior, summed over the seven parameters.

**Build 2, ask:** Which of these did we already compute by hand?

**Expected:** Bernoulli (the coin, 0.19) and categorical (the fruit, 0.42).

**Point here:** The Gaussian row is KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)).

**Transition:** Derive the Gaussian row.

## gaussian-derivation: Deriving the Gaussian formula

**Build 1, ask:** Which constants cancel?

**Expected:** The common log √(2π) terms, leaving the log ratio of the widths.

**Build 2, ask:** Why is E_q[(u − m)²] = σ² + (μ − m)²?

**Expected:** Write u − m = (u − μ) + (μ − m); the cross term averages to zero under q.

**Build 3, ask:** When is the KL zero?

**Expected:** Only when μ = m and σ = s.

**Transition:** Use the formula on the PML vote.

## compare-formula: Use the formula: which Gaussian is closer to p?

**Build 1, ask:** Vote first: q₁ or q₂?

**Expected:** Most pick q₂: same centre and a closer width.

**Build 2, ask:** Which term made q₁ worse?

**Expected:** Both: q₁ is off-centre, (0 − 1)² = 1, and narrower, log 2 against log 4/3.

**Point here:** PAUSE (PML L23 08:30).

**Transition:** Where does each number come from? Watch it as an area.

## kl-area: KL is an area you can watch

**Build 1, ask:** Make q narrow and centred on p. What happens in each direction?

**Expected:** KL(q‖p) stays moderate: q only visits places where p is reasonable. KL(p‖q) explodes: p’s tails land where q is nearly zero.

**Point here:** Click q₁, then q₂, then flip to KL(p‖q). Then shrink σ in each direction.

**Transition:** What if there is no formula at all?

## kl-sampling: No formula needed: estimate KL by sampling

**Build 1, ask:** Which distribution supplies the draws?

**Expected:** q, the first argument: it does the weighting.

**Build 2, ask:** Should the estimate match exactly?

**Expected:** No: it is random. It is close, and gets closer with more draws.

**Point here:** q₂ = N(1, 3²) against p = N(1, 4²), seed 1000.

**Transition:** Watch the estimate settle as draws arrive.

## mc-live: Watch the estimate settle

**Build 1, ask:** Press +10 a few times. Is every estimate above the exact value?

**Expected:** No: with few draws the estimate lands on either side, and can even be negative. The spread shrinks like 1/√S.

**Point here:** q = N(0, 2²) and p = N(1, 4²), the q₁ pair. Draws are seeded, so the sequence replays identically.

**Transition:** Part 3: use the score to find the best q.

## six-candidates: Six candidates: which q is closest to p?

**Build 1, ask:** Vote: which q is closest?

**Expected:** Most pick N(1, 3²) or N(2, 5²). The formula says N(1, 3²), narrowly.

**Build 2, ask:** Could another (μ, σ) do better?

**Expected:** Yes: N(1, 4²), which is p itself, has KL 0. We need a way to search.

**Point here:** PAUSE. All six KLs come from the closed-form Gaussian formula.

**Transition:** Search systematically: a grid.

## grid-search: Search a grid of means and widths

**Build 1, ask:** Where do you expect the minimum?

**Expected:** At μ = 1 and σ = 4, where q = p.

**Build 2, ask:** How many evaluations for fourteen parameters?

**Expected:** 40¹⁴ ≈ 2.7 × 10²², even with only 40 values per parameter.

**Transition:** Instead of evaluating everywhere, walk downhill.

## walk-downhill: Instead of a grid, walk downhill

**Build 1, ask:** What does the gradient tell us at each point?

**Expected:** The direction in (μ, σ) that lowers KL fastest.

**Build 2, ask:** Why does this scale where the grid did not?

**Expected:** Each step costs one gradient evaluation, whatever the number of parameters.

**Point here:** The path is the recorded Adam run from the next two slides, drawn on the same grid.

**Transition:** The same search in code.

## fit-code: The same search in code

**Build 1, ask:** What stays fixed, and what does the optimiser change?

**Expected:** p stays fixed; μ and ρ change.

**Build 2, ask:** Why not update σ directly?

**Expected:** A step could make σ zero or negative, which is not a valid Normal.

**Transition:** Press play.

## olive: Press play: q slides onto p

**Build 1, ask:** Predict: does the centre or the width arrive first?

**Expected:** Students split. The width reaches 4 around step 45, overshoots to 4.52, then settles; μ arrives by step 100.

**Build 2, ask:** Why can both move at once?

**Expected:** Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.

**Point here:** PAUSE (PML L23 12:13). Adam, learning rate 0.05, analytic KL. Notebook section 3.

**Transition:** One dimension is easy. What happens with two correlated unknowns?

## two-d: In 2-D, a q that cannot tilt shrinks instead

**Build 1, ask:** Predict the diagonal q’s widths: 1, more or less?

**Expected:** Less: about 0.44, which is √(1 − 0.9²).

**Build 2, ask:** Why shrink rather than spread?

**Expected:** KL(q‖p) heavily penalises q mass where p is tiny, which is off the diagonal ridge.

**Build 3, ask:** What does the full covariance cost?

**Expected:** More parameters: d(d + 1)/2 for L instead of d. For a million weights that is far too many.

**Point here:** PAUSE. Measured fit: both widths 0.4359, KL 0.8304, matching √(1 − ρ²) and −½ log(1 − ρ²).

**Transition:** Now a target with no formula.

## mixture-no-formula: PyTorch has no formula for this KL. The definition still works.

**Build 1, ask:** What would kl_divergence need for this pair?

**Expected:** A closed-form expression for a Gaussian against a mixture. There isn’t one.

**Build 2, ask:** Is the KL undefined, then?

**Expected:** No. It is a perfectly good integral; we just cannot write it in closed form.

**Point here:** Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch 2.13.0).

**Transition:** Estimate the average by sampling.

## now-optimise: Now optimise it: the code runs, and q does not move

**Build 1, ask:** Predict: where is q after 400 steps?

**Expected:** Most expect it to slide onto the mixture. It does not move.

**Build 2, ask:** The code runs. What could be wrong?

**Expected:** The gradient: it must be missing a piece.

**Point here:** PAUSE (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.

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

**Point here:** PAUSE (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.

**Transition:** Picture what a fixed ε does.

## reparam-picture: What a fixed ε does when μ or σ changes

**Build 1, ask:** If μ rises by 1, how far does u move?

**Expected:** By exactly 1.

**Build 2, ask:** If σ rises by 1?

**Expected:** By ε = −0.75: to the left, because this ε is negative.

**Transition:** Change one word in the code.

## rsample-fit: Change one word: rsample() fits the mixture

**Build 1, ask:** Predict where q settles.

**Expected:** Over the mixture’s mass, between and around the two bumps.

**Build 2, ask:** Why doesn’t the KL reach 0?

**Expected:** No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct optimisation.

**Point here:** Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Acts 3 and 4.

**Transition:** One more question: which direction of KL?

## fwdrev: The price: reverse KL picks one bump

**Build 1, ask:** Predict each final q.

**Expected:** Forward: wide, centred between the bumps. Reverse: one bump, depending on the start.

**Build 2, ask:** Which fit would make you overconfident?

**Expected:** The reverse fits: each reports one bump with a narrow spread.

**Point here:** PAUSE. Narrow bumps: forward KL 0.72; reverse 0.36 (= −log 0.7) from the left and 1.20 (= −log 0.3) from the right. Then switch to wide bumps: overlapping bumps give one broad reverse-KL fit.

**Transition:** Part 4: back to Bayes.

## can-we: Back to Bayes: minimise KL to the posterior directly?

**Build 1, ask:** Why this direction, KL(q‖p), and not KL(p‖q)?

**Expected:** KL(p‖q) averages under the posterior, so it needs posterior samples: the problem itself. KL(q‖p) averages under q, which we choose.

**Build 2, ask:** Which piece is missing?

**Expected:** log p(θ | 𝒟): it contains log p(𝒟).

**Transition:** Substitute Bayes’ rule and see what survives.

## bayes-sub: Substitute Bayes’ rule and see what survives

**Build 1, ask:** Substitute and simplify.

**Expected:** −log p(θ | 𝒟) = −log p(𝒟 | θ) − log p(θ) + log p(𝒟).

**Build 2, ask:** Why can log p(𝒟) leave the expectation?

**Expected:** It is a constant: it depends on neither θ nor q.

**Build 3, ask:** Which side can we compute?

**Expected:** The ELBO: draws from q, the likelihood, the prior and log q.

**Point here:** PAUSE (PML L23 47:15). Students do the substitution on paper before build 2. KL ≥ 0, so the ELBO is a lower bound on log p(𝒟), and maximising it minimises the KL.

**Transition:** What does the ELBO look like for our network?

## elbo-two-terms: The ELBO for our network: fit the data, stay near the prior

**Build 1, ask:** Which term pulls q towards a narrow, well-fitting θ?

**Expected:** The data term. The KL to the N(0, 1) prior pulls the other way: wide, near zero.

**Build 2, ask:** Which term needs sampling, and why?

**Expected:** The data term: the network is nonlinear in θ, so there is no formula for its average.

**Transition:** Check the identity on a model where p(𝒟) is known.

## elbo-slider: Check the identity where p(𝒟) is known: a coin

**Build 1, ask:** Find the q with the highest ELBO. What is its KL?

**Expected:** Beta(22, 16), the exact posterior: ELBO = log p(𝒟) = −21.40 and KL = 0.

**Point here:** Conjugate coin: posterior Beta(22, 16), log p(𝒟) = log B(22, 16) − log B(5, 3). The ELBO comes from analytic Beta moments, the KL from its own formula.

**Transition:** Part 5: implement the ELBO for the network.

## bnn-elbo-code: The ELBO for our network, in code

**Build 1, ask:** Which line uses the reparameterisation trick?

**Expected:** theta = q.rsample(...): gradients flow back to μ and ρ.

**Build 2, ask:** Which line uses the Gaussian KL formula?

**Expected:** kl = kl_divergence(q, prior).sum(): seven closed-form terms.

**Notebook demonstration:** Notebook section 8 runs this loop and records the frames for the next slide.

**Transition:** Press play.

## bnn-vi: Press play: q learns the network

**Build 1, ask:** Predict: will the ELBO reach log p(𝒟)?

**Expected:** It approaches, then stops below it; it can never cross.

**Build 2, ask:** What does the gap mean?

**Expected:** The gap is KL(q ‖ posterior): how far the best mean-field q is from the true posterior.

**Point here:** PAUSE. Frames every 5 steps to step 200, then every 50; the ELBO for each frame uses 2,000 fixed draws.

**Transition:** What did q learn about each parameter?

## bnn-q: What q learned: one hidden unit switched off

**Build 1, ask:** Which bells look like the prior?

**Expected:** w₁ and b₁: wide and centred near 0, like the prior.

**Build 2, ask:** Why might the ELBO prefer to switch a unit off?

**Expected:** The KL term charges for every parameter that moves away from the prior. Here one tanh unit already fits the data well.

**Point here:** Fitted widths: w₁ 0.894, w₂ 0.113, b₁ 0.908, b₂ 0.071, v₁ 0.035, v₂ 0.028, c 0.022.

**Transition:** How good is q’s predictive?

## bnn-vs-ref: VI against the reference: the right curve, too-narrow bands

**Build 1, ask:** Where do VI and the reference disagree most?

**Expected:** Beyond the data at both ends, where VI is overconfident.

**Build 2, ask:** Is a gap of 5.8 nats large?

**Expected:** Yes: q misses much of the posterior. Up to log 8 ≈ 2.1 nats of it is because the posterior has eight mirror-image copies (swap the two units, flip their signs) and q sits on one.

**Transition:** Collect the tools.

## five-tools: Six obstacles, six tools

**Build 1, ask:** Cover the middle column. Which tool removed which obstacle?

**Expected:** Students reconstruct it row by row.

**Transition:** Next lecture.

## next-lecture: Generative models create this inference problem again

**Build 1, ask:** Where might a new unknown appear in an image model?

**Expected:** A latent description z can explain an observed image x.

**Build 2, ask:** What should we ask of a good reconstruction?

**Expected:** Whether reconstructing observations also gives a useful way to generate new ones.

**Transition:** Close the main route. Optional depth follows.

## exit-transfer: A sensor has an unknown constant bias (optional)

**Build 1, ask:** Identify the unknown, the observations and one possible q family.

**Expected:** u is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.

**Build 2, ask:** Which available functions form the joint?

**Expected:** The prior density and the likelihood of each reading.

**Build 3, ask:** What objective would you implement?

**Expected:** The expectation of log joint minus log q, maximised over ψ.

**Point here:** This conjugate toy has an exact answer too; the exercise is identifying the pieces, not needing approximation.

**Transition:** Where these unknowns appear next.

## mc-variability: More samples, tighter estimates (optional)

**Build 1, ask:** How many more draws halve the spread?

**Expected:** Four times as many: the spread scales as 1/√S.

**Point here:** Same p and q at every size; 40 seeds each.

**Transition:** Act 4: we can estimate the loss. Now optimise it.

## answer-key: A check with an answer key: the Beta coin (optional)

**Build 1, ask:** Where does Beta(10, 10) put its mass?

**Expected:** Centred at 0.5, and too broad.

**Build 2, ask:** What changed by step 100?

**Expected:** It shifted towards the heads proportion and narrowed.

**Build 3, ask:** Is the final q exactly Beta(22, 16)?

**Expected:** No, but very close: KL ≈ 0.0002. The answer key was never used by the optimiser.

**Point here:** Beta.rsample is differentiable through implicit reparameterisation, not μ + σε. Adam 0.025, 512 draws, seed 2401.

**Transition:** Move q yourself and watch the total.

## optional-one-step: One literal update of the mean (optional)

**Build 1, ask:** Which term of the Gaussian KL depends on μ?

**Expected:** Only the squared shift, (μ − 1)²/(2·4²).

**Build 2, ask:** Differentiate it at μ = 0.

**Expected:** (μ − 1)/16 = −0.0625: a negative derivative, so descent moves μ right.

**Build 3, ask:** Take one step with η = 4. What is the best KL with σ stuck at 2?

**Expected:** μ⁺ = 0.25. Even at μ = 1 the KL is 0.318: the width mismatch needs σ to move too.

**Point here:** η = 4 is chosen to make the arithmetic visible, not a recommended learning rate.

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
