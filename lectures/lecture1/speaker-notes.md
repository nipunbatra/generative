# KL and variational inference: speaker notes

Planned main route: 140 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Lecture map | 1 |
| Part 1 · Bayes’ rule for a neural network | 22 |
| Part 2 · How far apart are two distributions? | 74 |
| Part 3 · Find the closest q | 22 |
| Part 4 · Back to Bayes: the ELBO | 13 |
| Part 5 · Implement it | 8 |

## contents: Lecture 1 at a glance: five parts, twenty-two sections

**Build 1, ask:** Which part do you expect to be hardest?

**Expected:** Usually Part 4, the ELBO derivation, so we take it one rule per line.

**Point here:** Click a section to jump to its first slide; ▶ notebook opens the matching notebook section in Colab. Run the notebook top to bottom once before class.

**Transition:** Part 1: Bayes’ rule for a neural network.

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

**Build 2, ask:** How many numbers make one draw of θ?

**Expected:** Seven, one per parameter. This draw is θ⁽¹⁾ on the next slide.

**Transition:** What does one draw of θ look like as a network?

## one-draw: One draw of θ is one whole function

**Build 1, ask:** Look at θ⁽¹⁾. Which numbers decide where its curve steps, and how high?

**Expected:** b/w sets where each tanh steps, |w| how sharply, v how high; c shifts the whole curve.

**Build 2, ask:** Why does θ⁽²⁾ give a different shape?

**Expected:** Different weights and biases: steps in other places, of other heights and signs.

**Build 3, ask:** So what does the prior on θ say about functions?

**Expected:** It is a distribution over functions: mostly smooth curves within about ±3.

**Point here:** θ⁽¹⁾ is the draw marked on the bells on the previous slide. θ⁽¹⁾, θ⁽²⁾ and θ⁽³⁾ are the first three of the twenty lines on the next slide.

**Transition:** Twenty draws at once.

## prior-functions: Twenty draws, and the prior predictive

**Build 1, ask:** Which numbers make a line steep? Which shift it up or down?

**Expected:** A large |w| makes a sharp step; c shifts the whole curve; v sets how high each step is.

**Build 2, ask:** Before any data, how sure is the model about y at x = 0?

**Expected:** Not at all: the band is about ±2.7 wide everywhere.

**Point here:** Prior over functions: the individual curves. Prior predictive: their average with the noise added, the band (4,000 draws). Section 1.5 shows the same picture after the data: the posterior predictive.

**Notebook demonstration:** Notebook 1.2: the prior-draws cell prints θ⁽¹⁾ and draws the twenty functions with the prior predictive band; change the seed for new draws.

**Transition:** Now the data, and how well each θ explains them.

## data-likelihood: The data, and how well one θ explains them

**Build 1, ask:** What does the 0.1 describe?

**Expected:** The noise standard deviation, assumed known here.

**Build 2, ask:** Why a product?

**Expected:** Given θ, the noise on different points is independent.

**Point here:** Data: y = 0.8 tanh(3x) + 0.3x + N(0, 0.1²) noise, seed 7.

**Transition:** Compute the right-hand side of Bayes’ rule for one θ.

## map-train: Finding a good θ: train the network

**Build 1, ask:** What is the loss here, in words?

**Expected:** The squared error between the curve and the points, divided by 2 · 0.1², plus half the sum of squared parameters: the prior acts as weight decay.

**Build 2, ask:** Why try ten starts?

**Expected:** Training can stop in different places. Here most starts reach 14.34; two stop a little higher.

**Point here:** Loss = Σᵢ (yᵢ − f(xᵢ))² / (2 · 0.1²) + ½ Σⱼ θⱼ², which is −log p(𝒟 | θ) − log p(θ) without its constants. Adam, learning rate 0.05, 2,000 steps per start. θ<sub>A</sub> is the MAP estimate.

**Notebook demonstration:** Switch to notebook 1.3 and run the training cell: it prints the final loss of all ten starts, then the log likelihood and log prior of θ_A and θ_B.

**Transition:** Meet θ<sub>A</sub>.

## theta-a: Meet θ<sub>A</sub>: the network with its trained numbers

**Build 1, ask:** Which weight is largest?

**Expected:** w₂ = −2.00: the second hidden unit makes the sharp step.

**Build 2, ask:** Does this curve explain the data well?

**Expected:** Yes: it passes close to every point.

**Transition:** The likelihood of θ<sub>A</sub>.

## likelihood-formula: The likelihood of θ<sub>A</sub>: formula, then value

**Build 1, ask:** Why a product?

**Expected:** Given θ, the noise on different points is independent.

**Build 2, ask:** Which part depends on θ?

**Expected:** Only the squared errors; the constant is the same for every θ.

**Build 3, ask:** Is a larger log likelihood better?

**Expected:** Yes: the curve passes closer to the points.

**Transition:** The prior of θ<sub>A</sub>.

## prior-formula: The prior of θ<sub>A</sub>, then the evidence

**Build 1, ask:** Which parameters make the prior small?

**Expected:** Large ones, positive or negative.

**Build 2, ask:** What is ½ Σ θⱼ² for θ<sub>A</sub>?

**Expected:** About 2.98: the weight-decay term from training.

**Build 3, ask:** How could we compute the evidence?

**Expected:** Add up p(𝒟 | θ) p(θ) over all θ. Next slide.

**Transition:** Compare θ<sub>A</sub> with a random θ from the prior.

## two-thetas: Two settings of θ, side by side

**Build 1, ask:** Which θ explains the data better, and by how much?

**Expected:** θ<sub>A</sub>, by about 2,170 nats of log likelihood.

**Build 2, ask:** How many times more probable is θ<sub>A</sub> than θ<sub>B</sub> after seeing the data?

**Expected:** About e²¹⁷⁰ times. p(𝒟) cancels in a ratio of posteriors: it is needed only to normalise.

**Transition:** Normalising is the hard part.

## evidence-area: The evidence is an area: add up thin rectangles

**Build 1, ask:** What is the height of each rectangle?

**Expected:** p(𝒟 | w₂) p(w₂) at its centre: the likelihood times the prior.

**Build 2, ask:** How many rectangles would two unknowns need?

**Expected:** A grid of squares: 20 × 20 = 400, or 100 × 100 if each axis needs 100.

**Point here:** A one-dimensional slice of our network’s evidence, for illustration only: the curve is computed in the browser from θ<sub>A</sub> and the data.

**Notebook demonstration:** Notebook 1.4: the first cell computes this one-unknown evidence with 20 rectangles and with 200,000; their logs agree to three decimals.

**Transition:** And with all seven unknowns?

## intractable: Intractable: why the rectangles do not scale

**Build 1, ask:** How long would 10¹⁴ evaluations take at a billion a second?

**Expected:** About 28 hours: possible, just. For 10^2,000,000: never.

**Build 2, ask:** So what does intractable mean?

**Expected:** No closed form, and no feasible numerical computation: the cost explodes with the number of unknowns.

**Point here:** Sampling θ from the prior and averaging p(𝒟 | θ) does not rescue it: almost every draw explains the data terribly. The optional slide on prior sampling shows the numbers: with ten draws the estimate is off by about 1,000 nats, and of ten million draws only about 100 matter.

**Transition:** Why do we want the posterior at all?

## predictive: What we want in the end: predictions with error bars

**Build 1, ask:** Where should the model be least sure?

**Expected:** Far from the data: beyond both ends, and in the gap.

**Build 2, ask:** How would you compute this integral?

**Expected:** Average f<sub>θ</sub>(x*) over samples of θ from the posterior, which we cannot draw directly. Yet.

**Point here:** Reference: Hamiltonian Monte Carlo, 4 chains of 2,000 samples, acceptance about 0.90. Predictive sd 0.22 at x = −2.5, about 0.11 in the data.

**Transition:** How else could we approximate the posterior?

## core-problem: The core problem of probabilistic machine learning

**Build 1, ask:** Which of the three pieces can we compute for one θ?

**Expected:** The likelihood and the prior, as we did for θ<sub>A</sub>. Not the evidence.

**Build 2, ask:** Where else will you meet this problem?

**Expected:** In every Bayesian model with many unknowns, including the VAE in Lecture 3.

**Transition:** Three ways to approach it.

## three-ways: Three ways to approximate the posterior; we take the third

**Build 1, ask:** What does Laplace miss here?

**Expected:** The second bump: it only looks at the peak.

**Build 2, ask:** What does MCMC cost?

**Expected:** Many steps, each evaluating the model on the data; hard to scale to large models and datasets.

**Build 3, ask:** What does VI trade away?

**Expected:** Exactness. The answer is limited to the family, here one Gaussian. In return it is fast and uses the same gradient tools as deep learning.

**Point here:** Target 0.7·N(−2, 1) + 0.3·N(2, 1). Every number is computed in notebook section 0.

**Transition:** Variational inference, in one slide.

## vi-idea: Variational inference: choose q, make it close to the posterior

**Build 1, ask:** How many numbers does q have?

**Expected:** Fourteen: a mean and a width for each of the seven parameters.

**Build 2, ask:** Why is drawing θ from q easy?

**Expected:** Each θⱼ is an independent Gaussian draw.

**Build 3, ask:** What do we need before we can minimise anything?

**Expected:** A number for the distance between two distributions. That is Part 2.

**Transition:** Part 2: how far apart are two distributions?

## pretend-posterior: For now, pretend we know the posterior

**Build 1, ask:** What do we gain by pretending we know p?

**Expected:** We can check every tool against the right answer before facing a real posterior.

**Build 2, ask:** What can we change about q?

**Expected:** Its mean μ and its width σ.

**Build 3, ask:** What would you compare: means, widths, something else?

**Expected:** Students suggest means and widths. The KL divergence compares the whole densities, and works for any shapes.

**Point here:** From here to the end of Part 3, p is a known stand-in for the posterior. Part 4 returns to the real one.

**Transition:** Start with the surprise of one outcome.

## surprise-forecast: How surprised would you be? A forecast of 90% sun

**Build 1, ask:** Tomorrow it rains. How surprised are you, compared with a sunny day?

**Expected:** Much more: rain had only a 10% chance.

**Build 2, ask:** Should a 100% certain outcome surprise you at all?

**Expected:** No: zero surprise. That is our first requirement.

**Transition:** What should the surprise function be?

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## surprise-candidates: What should surprise be? Three requirements

**Build 1, ask:** Why should surprises add for independent days?

**Expected:** Independent days share nothing, so the information should simply add, just as the probabilities multiply.

**Build 2, ask:** Which requirement does 1/p fail?

**Expected:** The first, and the third: it multiplies instead of adding.

**Build 3, ask:** Is the log the only choice?

**Expected:** Up to the base, yes: any continuous f with f(ab) = f(a) + f(b) is a logarithm. The base sets the unit.

**Transition:** Check it on five days.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## surprise-five: Five days: probabilities multiply, surprises add

**Build 1, ask:** Is log₂ 1/0.59 equal to 5 × 0.152?

**Expected:** Yes: both are 0.76 bits. The log turns the product into a sum.

**Build 2, ask:** Which day is the most informative?

**Expected:** The rainy one: 3.32 of the 3.93 bits.

**Transition:** The surprise curve, and its unit.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## surprise-curve: Surprise in bits: I(x) = log₂ 1/p(x)

**Build 1, ask:** How many bits for p = ⅛?

**Expected:** 3 bits: ⅛ = ½ × ½ × ½, three halvings.

**Build 2, ask:** What is one bit, in words?

**Expected:** The surprise of a fair coin flip: one yes/no answer you could not predict.

**Transition:** One outcome is surprising or not. What about a whole distribution?

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## entropy-average: Entropy: how surprised are you, on average?

**Build 1, ask:** Which outcome is more surprising? Which happens more often?

**Expected:** Rain is far more surprising; sun far more common.

**Build 2, ask:** Why weight by 0.9 and 0.1?

**Expected:** They are how often each surprise happens. Next slide: the long-run average over days.

**Build 3, ask:** Is entropy about one outcome, or the whole distribution?

**Expected:** The whole distribution: one number for how uncertain it is.

**Transition:** Why an expectation? Let days go by.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## entropy-days: Why an expectation? It is the average over many days

**Build 1, ask:** After 10 days, is the average close to 0.47?

**Expected:** Only by luck: one rainy day in ten gives 0.47, two give 0.8. It settles as days accumulate, because sun comes on 90% of them and rain on 10%.

**Point here:** The weighted sum Σ p(x) f(x) is the average of f over infinitely many draws; with S draws the plain average estimates it (Section 2.7).

**Notebook demonstration:** Notebook 2.2: the entropy cell simulates 10,000 days and prints the running average at 10, 100, 1,000 and 10,000 days.

**Transition:** How does entropy change with the forecast?

## entropy-explorer: From certain to fully uncertain: between 0 and 1 bit

**Build 1, ask:** Where is the entropy largest, and why?

**Expected:** At 50/50: the weather is least predictable, so each day surprises you most on average.

**Transition:** Three kinds of weather.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## entropy-three: Three kinds of weather: 1.5 bits

**Build 1, ask:** Why is cloudy worth 2 bits?

**Expected:** ¼ = ½ × ½: two halvings, two bits.

**Build 2, ask:** Is 1.5 bits more or less than a fair coin?

**Expected:** More: three outcomes leave more to learn each day than two equally likely ones.

**Transition:** What does 1.5 bits mean in practice?

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## entropy-questions: Entropy counts yes/no questions

**Build 1, ask:** How many questions on a sunny day? On a rainy day?

**Expected:** One; two.

**Build 2, ask:** Can any strategy average fewer than 1.5 questions?

**Expected:** No: entropy is the least possible average. This strategy reaches it because every probability is a power of ½.

**Build 3, ask:** What is the code for rainy?

**Expected:** 11: no to “sunny?”, then no to “cloudy?”.

**Point here:** Each question here is 50/50 given the earlier answers, so each answer carries exactly 1 bit.

**Transition:** Section 2.3: send the weather as bits.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## channel: Way 2: send the weather, in bits

**Build 1, ask:** What does the receiver need to recover?

**Expected:** Which kind of day it was, exactly.

**Build 2, ask:** What do we want to minimise?

**Expected:** The average number of bits sent per day.

**Transition:** First, what is a code?

**Source / evidence:** Coding story after <a href="https://www.youtube.com/watch?v=ErfnhcEV1O8">Aurélien Géron, “A Short Introduction to Entropy, Cross-Entropy and KL-Divergence”</a> (YouTube)

## uniform-code: A code: a string of bits for each kind of day

**Build 1, ask:** How many bits does the uniform code send per day?

**Expected:** 2, every day.

**Build 2, ask:** Where is it wasteful?

**Expected:** On the common sunny days.

**Transition:** Shorter strings need care.

**Source / evidence:** Coding story after <a href="https://www.youtube.com/watch?v=ErfnhcEV1O8">Aurélien Géron, “A Short Introduction to Entropy, Cross-Entropy and KL-Divergence”</a> (YouTube)

## prefix-code: Prefix codes: the bits split up without separators

**Build 1, ask:** Decode 0 10 11 0 10.

**Expected:** Sunny, cloudy, rainy, sunny, cloudy.

**Build 2, ask:** Why is 0, 01, 1 a bad code?

**Expected:** 0 is the start of 01, so the receiver cannot tell where one day ends.

**Transition:** Which prefix code is the most efficient?

**Source / evidence:** Coding story after <a href="https://www.youtube.com/watch?v=ErfnhcEV1O8">Aurélien Géron, “A Short Introduction to Entropy, Cross-Entropy and KL-Divergence”</a> (YouTube)

## huffman: Huffman’s recipe: merge the two rarest, repeat

**Build 1, ask:** Which two do we merge first?

**Expected:** The two least likely: cloudy and rainy.

**Build 2, ask:** What is the merged node’s probability?

**Expected:** ½.

**Build 3, ask:** How many merges for K outcomes?

**Expected:** K − 1: each merge removes one node.

**Build 4, ask:** Why does the likeliest outcome get the shortest codeword?

**Expected:** It is merged last, so it sits closest to the root.

**Point here:** The notebook implements Huffman’s algorithm in ten lines with a heap, and recovers 0, 10, 11 for p and 10, 11, 0 for q.

**Transition:** How good is this code?

## efficient-code: Each codeword is exactly as long as its surprise

**Build 1, ask:** How long is the codeword for an outcome with probability ⅛?

**Expected:** 3 bits: log₂ 8.

**Build 2, ask:** Can any code average fewer than 1.5 bits here?

**Expected:** No: by Shannon’s source coding theorem, H(p) is the minimum.

**Transition:** What if the code was built for the wrong weather?

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## wrong-belief: A code built on a wrong belief

**Build 1, ask:** Which of our days does their code make expensive?

**Expected:** Sunny days: half of our days now cost 2 bits instead of 1.

**Build 2, ask:** Whose probabilities set the code lengths?

**Expected:** q’s: a code built for q spends log₂ 1/q(x) bits on x.

**Transition:** Average the cost over our weather.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## cross-entropy: Cross-entropy: the average cost when the code expects q

**Build 1, ask:** Which distribution supplies the weights?

**Expected:** p, the real weather.

**Build 2, ask:** Compute it.

**Expected:** ½(2) + ¼(2) + ¼(1) = 1.75.

**Build 3, ask:** What is H(p, p)?

**Expected:** H(p): the right code, 1.5 bits.

**Transition:** Watch it happen, day by day.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## bits-days: Watch it happen: our weather through both codes

**Build 1, ask:** After 1,000 days, how many bits did the wrong code waste?

**Expected:** About 250, 0.25 per day. The notebook’s run: 1,747 bits against 1,498.

**Notebook demonstration:** Notebook 2.3: the Huffman cell sends 1,000 days through both codes and prints the bit counts and the start of each bit stream.

**Transition:** Name the wasted bits.

## kl-extra-bits: KL divergence: the extra bits we waste

**Build 1, ask:** What would the waste be if q = p?

**Expected:** Zero: the right code wastes nothing.

**Build 2, ask:** Why can the two sums be combined?

**Expected:** Same outcomes, same weights p(x).

**Build 3, ask:** Which outcome saves bits? Does the total still lose?

**Expected:** Rainy: its codeword got shorter (−1 bit, on a quarter of days). Sunny loses a bit on half the days, so the total is +0.25.

**Transition:** Two facts about KL.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## kl-properties: Never negative, and zero only for the right code

**Build 1, ask:** Can KL be negative?

**Expected:** No: that would beat the best possible code.

**Build 2, ask:** If KL(p‖q) = 0, what do we know?

**Expected:** q = p: the belief is right.

**Transition:** Does it matter which one is the truth?

## kl-asymmetry: Which one is the truth? KL is not symmetric

**Build 1, ask:** Why is their code so costly for us?

**Expected:** It spends 6.6 bits on our cloudy and rainy days, which are half of all days.

**Build 2, ask:** Is KL a distance in the everyday sense? Which direction will VI use?

**Expected:** No: a distance is the same both ways; KL is a divergence. VI uses KL(q ‖ posterior), q first; Section 3.6 shows what that does to a fitted q.

**Point here:** Sunny town q = (0.98, 0.01, 0.01): codeword lengths log₂ 1/q = 0.03, 6.64, 6.64 bits.

**Transition:** Section 2.5: where you have already met all this.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## apps-map: KL at work: one recipe, three applications

**Build 1, ask:** What do the three KLs in row ④ have in common?

**Expected:** The first argument is always the “truth” the model must match: the label, the data, the teacher.

**Transition:** 2.5.1: classification.

## app-classify: 2.5.1 Classification: the plan

**Build 1, ask:** Which loss do you use to train a classifier?

**Expected:** Cross-entropy. Step ④ shows that it is a KL.

**Transition:** ①② The problem and the model.

## classifier: ①② A classifier outputs a distribution

**Build 1, ask:** What does the classifier output?

**Expected:** A probability for each label, from a softmax.

**Build 2, ask:** How can a label be a distribution?

**Expected:** One-hot: probability 1 on the true class, 0 on the others.

**Transition:** ③ The usual loss.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## ce-surprise: ③ The usual loss: surprise at the true label

**Build 1, ask:** Which terms survive?

**Expected:** Only the true class.

**Build 2, ask:** In words, what is the loss?

**Expected:** How surprised the model is by the right answer.

**Build 3, ask:** What loss would q(cat) = 1 give?

**Expected:** 0: no surprise.

**Transition:** ④ Where is the KL?

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## ce-is-kl: ④ As KL: with a one-hot label, cross-entropy = KL

**Build 1, ask:** What is H(p) for a one-hot label?

**Expected:** 0.

**Build 2, ask:** So what does training minimise?

**Expected:** KL(label ‖ prediction): the bits the prediction wastes on the true labels.

**Build 3, ask:** Why does PyTorch compute cross-entropy rather than KL?

**Expected:** They are equal here, and in general differ by H(p), which training cannot change. Cross-entropy avoids the log of the labels.

**Notebook demonstration:** Notebook 2.5.1: F.cross_entropy prints 0.511 nats, −log 0.6, the same number as KL(one-hot ‖ q).

**Transition:** ⑤ Check: the loss as the prediction changes.

**Source / evidence:** Story after <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang, “Fantastic KL Divergence and How to (Actually) Compute It”</a> (YouTube)

## loss-curve: ⑤ Check: the loss is the surprise curve

**Build 1, ask:** Where is the curve steepest?

**Expected:** Near 0: almost-impossible true labels cost the most.

**Build 2, ask:** Why might a model learn to be confident?

**Expected:** Each step towards q(true) = 1 lowers the loss; with no other pressure it heads there.

**Transition:** 2.5.2: maximum likelihood.

## app-mle: 2.5.2 Maximum likelihood: the plan

**Build 1, ask:** Where have you used maximum likelihood before?

**Expected:** Whenever a model is fitted by its likelihood: linear regression with Gaussian noise, logistic regression, the classifier of 2.5.1.

**Transition:** ① The problem.

## mle-problem: ① The problem: learn a distribution from data

**Build 1, ask:** What is unknown here, and what is observed?

**Expected:** The distribution p<sub>data</sub> is unknown; the twenty days are observed.

**Build 2, ask:** What could the fitted q be used for?

**Expected:** Predicting tomorrow’s weather, or generating realistic new days: a generative model.

**Transition:** ② The data and the model.

## mle-data: ② The data and the model: twenty days, three numbers

**Build 1, ask:** What would you guess for tomorrow?

**Expected:** Sunny with probability ½: the frequency.

**Build 2, ask:** What should “explains the data best” mean?

**Expected:** Gives the data the highest probability: maximum likelihood. Let us see why that is the frequency.

**Transition:** ③ The usual loss: how probable are the twenty days under q?

## mle-likelihood: ③ The usual loss: how probable are the twenty days?

**Build 1, ask:** Why a product?

**Expected:** Independent days: probabilities multiply.

**Build 2, ask:** Why is it so small?

**Expected:** Any one particular sequence of 20 days is unlikely, even under the right model.

**Build 3, ask:** Why work with the log?

**Expected:** Products of many small numbers underflow; the log turns them into a sum.

**Transition:** Per day, that is a familiar number.

## mle-surprise: ③ Per day: the data’s average surprise under q

**Build 1, ask:** What is minus the log-likelihood per day, in words?

**Expected:** The average surprise of a recorded day under q.

**Build 2, ask:** Where have we seen 1.75?

**Expected:** H(p, q): the wrong code on our weather.

**Build 3, ask:** Coincidence?

**Expected:** No. Next slide: group the days by weather.

**Transition:** Group the days by weather.

## mle-group: ④ Rewrite it: group the days by weather

**Build 1, ask:** How many sunny terms are there?

**Expected:** Ten, all equal to log₂ 1/q(sunny).

**Build 2, ask:** What do 10/20, 5/20 and 5/20 form?

**Expected:** The data’s frequencies p̂ = (½, ¼, ¼): the empirical distribution.

**Build 3, ask:** So what is maximum likelihood minimising?

**Expected:** The cross-entropy H(p̂, q).

**Transition:** Split it, as in 2.4.

## mle-split: ④ As KL: the data’s surprise = H(p̂) + KL(p̂ ‖ q)

**Build 1, ask:** Which part can the model change?

**Expected:** Only KL(p̂ ‖ q).

**Build 2, ask:** What is the smallest possible average surprise?

**Expected:** H(p̂) = 1.5 bits, when KL = 0, that is, q = p̂.

**Transition:** Find the best q yourself.

## mle-fit: ⑤ Check: find the least surprising q

**Build 1, ask:** Start from (⅓, ⅓, ⅓). Which way should q(sunny) move?

**Expected:** Up, to ½: the data have more sun than q expects. At q = p̂ the data are 32 times more probable than under (¼, ¼, ½).

**Notebook demonstration:** Notebook 2.5.2: the second cell runs gradient descent on the data’s average surprise from equal logits and lands on (0.5, 0.25, 0.25) without being told.

**Transition:** ⑤ Four formulations, one answer.

## mle-same-answer: ⑤ Four formulations, one answer

**Build 1, ask:** Why do all four rows agree?

**Expected:** Each row rewrites the one above: take the log and divide by −20, group the days by weather, subtract the fixed H(p̂).

**Build 2, ask:** How many times more probable are the data under p̂?

**Expected:** 2⁵ = 32: twenty days times 0.25 bits per day.

**Transition:** The general statement.

## mle-kl: In general: maximum likelihood = minimum KL(data ‖ model)

**Build 1, ask:** Which distribution comes first in this KL?

**Expected:** The data’s, p̂.

**Build 2, ask:** What happens with more data?

**Expected:** p̂ gets closer to the true distribution of the data.

**Build 3, ask:** What is a training set’s cross-entropy loss?

**Expected:** The average surprise at the true labels: minus the average log-likelihood.

**Transition:** 2.5.3: knowledge distillation.

## app-distill: 2.5.3 Knowledge distillation: the plan

**Build 1, ask:** Why would anyone want a smaller copy of a model?

**Expected:** Speed, memory and cost: phones, browsers, millions of requests a day.

**Transition:** ① The problem.

## distill-why: ① The problem: a big teacher, a small student

**Build 1, ask:** Why not just deploy the teacher?

**Expected:** Cost and speed: memory, latency and energy grow with the number of parameters.

**Build 2, ask:** How could the small model learn to behave like the big one?

**Expected:** Copy its outputs: make the student’s distribution over labels match the teacher’s, input by input.

**Point here:** Hinton, Vinyals and Dean (2015), “Distilling the Knowledge in a Neural Network”. Smaller deployed models, such as DistilBERT, are trained this way.

**Transition:** ② The data and the model.

## distill-match: ② The data and the model: same inputs, two networks

**Build 1, ask:** Does the student need the true labels?

**Expected:** Not for this term: the teacher labels every input, even unlabelled ones. In practice a small label term is often added.

**Build 2, ask:** What should the loss measure?

**Expected:** How far the student’s distribution is from the teacher’s, averaged over the inputs, with zero when they agree.

**Transition:** ③ The usual loss.

## distill-soft: ③ The usual loss: cross-entropy with soft labels

**Build 1, ask:** What does the student learn from the 0.20 on dog?

**Expected:** That cats and dogs look alike to the teacher, so confusing them is a small error, unlike cat versus ape.

**Build 2, ask:** Why would τ = 1 hide that information?

**Expected:** The 0.02 and 0.03 barely change the loss; dividing the logits by τ evens the distribution out.

**Point here:** Hinton, Vinyals and Dean (2015) train the student on the teacher’s soft targets at a raised temperature, plus a small weight on the true labels.

**Transition:** ④ The same loss as KL.

## distill-loss: ④ The same loss as KL, term by term

**Build 1, ask:** Which part can the student change?

**Expected:** Only q_S: H(p_T) is fixed by the frozen teacher.

**Build 2, ask:** Why does the teacher come first in the KL?

**Expected:** The teacher plays the truth, like the label in 2.5.1 and the data in 2.5.2: its distribution supplies the weights.

**Build 3, ask:** What does a temperature τ > 1 do?

**Expected:** It flattens the teacher’s distribution so its small probabilities, which carry the extra information, count for more.

**Point here:** Implementations often compute the second term with PyTorch’s KL loss, F.kl_div or nn.KLDivLoss.

**Transition:** ⑤ Check with numbers.

## soft-labels: ⑤ Check with numbers: cross-entropy = entropy + KL

**Build 1, ask:** Which part can training change?

**Expected:** Only the KL; H(p) is fixed by the teacher.

**Build 2, ask:** Can the cross-entropy reach 0 here?

**Expected:** No: its minimum is H(p) = 0.72 nats, reached when the student copies the teacher.

**Transition:** Watch a real student learn.

## distill-demo: ⑤ Check: a tiny student learns a big teacher

**Build 1, ask:** Predict: can a 27-parameter network copy a 4,547-parameter one?

**Expected:** On this problem, nearly: the classes need only gently curved boundaries.

**Build 2, ask:** Why do the boundaries stay blurred?

**Expected:** The student copies the teacher’s uncertainty too: where classes overlap, the teacher is unsure, and so is the student.

**Notebook demonstration:** Notebook 2.5.3: the distillation cell trains the teacher on labels, then the student on the teacher’s probabilities, and prints both accuracies, the agreement and the KL.

**Transition:** One principle in many disguises.

## apps-same: One principle in many disguises

**Build 1, ask:** What is the same in the three applications?

**Expected:** Each minimises a KL from a known “truth” (label, data, teacher) to the model.

**Build 2, ask:** What is different about VI?

**Expected:** The order: q comes first. We cannot sample the posterior, so the weights must come from q.

**Transition:** Back to continuous distributions: closed forms.

## densities: From sums to integrals

**Build 1, ask:** What plays the role of the weights now?

**Expected:** The density p(θ), times dθ.

**Build 2, ask:** Can a density be bigger than 1?

**Expected:** Yes; but the KL is still a weighted average of log ratios, and is still never negative.

**Transition:** Closed forms for the distributions we will use.

## bern-pmf: Bernoulli: a coin with P(x = 1) = a

**Build 1, ask:** What is p(0) for Bernoulli(0.8)?

**Expected:** 0.2.

**Build 2, ask:** Write the KL.

**Expected:** Two terms: one for x = 1 and one for x = 0. Next slide.

**Transition:** The answer, step by step.

## bern-kl: KL between two Bernoullis, step by step

**Build 1, ask:** How many terms does the sum have?

**Expected:** Two, one per value of x.

**Build 2, ask:** Which term is negative here, and why?

**Expected:** The x = 0 term: p(0) = 0.2 is smaller than q(0) = 0.5, so its log ratio is negative.

**Build 3, ask:** What is the KL if a = b?

**Expected:** 0: both log ratios are 0.

**Build 4, ask:** Is 0.19 the same as KL(q‖p)?

**Expected:** No: KL(Bernoulli(0.5) ‖ Bernoulli(0.8)) = 0.22.

**Transition:** More than two outcomes.

## cat-pmf: Categorical: K outcomes with probabilities π₁, …, π<sub>K</sub>

**Build 1, ask:** What is π for this year’s grades?

**Expected:** (0.4, 0.1, 0.5).

**Build 2, ask:** Write the KL.

**Expected:** A sum of K terms, πₖ log(πₖ/ρₖ). Next slide.

**Transition:** The answer, step by step.

## cat-kl: KL between two categoricals, step by step

**Build 1, ask:** How many terms for K outcomes?

**Expected:** K.

**Build 2, ask:** Why does K = 2 give the Bernoulli formula?

**Expected:** A Bernoulli is a categorical over {1, 0} with π = (a, 1 − a).

**Build 3, ask:** Which term is largest for the grades?

**Expected:** A: 0.4 log 4 ≈ 0.55.

**Transition:** Continuous: the Gaussian.

## gauss-pdf: The Gaussian: a bell with mean μ and width σ

**Build 1, ask:** What does σ control?

**Expected:** The width: about 68% of the mass lies within σ of μ.

**Build 2, ask:** What is log N(θ; μ, σ²)?

**Expected:** −½ log(2πσ²) − (θ − μ)²/(2σ²).

**Transition:** Step 1: the log ratio.

## gauss-log-ratio: Gaussian KL, step 1: the log ratio

**Build 1, ask:** Why take logs first?

**Expected:** The densities are exponentials; logs turn them into simple squares.

**Build 2, ask:** Where does log(s/σ) come from?

**Expected:** −½ log(2πσ²) + ½ log(2πs²) = log s − log σ.

**Build 3, ask:** Which terms still depend on θ?

**Expected:** The two squared terms.

**Transition:** Step 2: average over q.

## gauss-expectation: Gaussian KL, step 2: average over q

**Build 1, ask:** What is the average of a constant?

**Expected:** The constant itself.

**Build 2, ask:** What is E_q[(θ − μ)²] called?

**Expected:** The variance of q, σ².

**Build 3, ask:** Why does the cross term vanish?

**Expected:** It is 2(μ − m)·E_q[θ − μ], and E_q[θ − μ] = 0.

**Build 4, ask:** When is the KL zero?

**Expected:** Only when μ = m and σ = s.

**Transition:** Many dimensions at once.

## diag-pdf: A diagonal Gaussian: one bell per coordinate

**Build 1, ask:** Why are the contours axis-aligned?

**Expected:** No correlation between coordinates: the covariance is diagonal.

**Build 2, ask:** What is the log of a product?

**Expected:** The sum of the logs.

**Transition:** The answer, step by step.

## diag-kl: KL between diagonal Gaussians, step by step

**Build 1, ask:** Why can each term be averaged separately?

**Expected:** Each term depends on θⱼ alone, and q makes the coordinates independent.

**Build 2, ask:** How many one-dimensional KLs for our network?

**Expected:** Seven, one per parameter.

**Build 3, ask:** What does each term reward?

**Expected:** μⱼ near 0 and σⱼ near 1: staying close to the prior.

**Transition:** Is this really what PyTorch computes?

## torch-discrete: The same formulas in PyTorch: Bernoulli and categorical

**Build 1, ask:** Where is a log(a/b) in the Bernoulli code?

**Expected:** t1: p.probs is a, and softplus(−q.logits) − softplus(−p.logits) = log a − log b.

**Build 2, ask:** Why does PyTorch use logits and softplus?

**Expected:** Numerical stability: no log of probabilities that round to 0.

**Point here:** Shown as in PyTorch 2.13, with torch.nn.functional.softplus shortened to softplus and the lines handling probabilities of exactly 0 or 1 left out.

**Transition:** And the Gaussian.

**Source / evidence:** PyTorch source: <a href="https://github.com/pytorch/pytorch/blob/main/torch/distributions/kl.py">torch/distributions/kl.py</a>

## torch-normal: The same formulas in PyTorch: Gaussian and diagonal

**Build 1, ask:** What is var_ratio in our notation?

**Expected:** σ²/s², with p the first argument (our q) and q the second (our p).

**Build 2, ask:** Which line is the diagonal-Gaussian sum?

**Expected:** _sum_rightmost: it adds the one-dimensional KLs over the coordinates.

**Point here:** In PyTorch’s code p and q are just the first and second arguments; with our VI order, its p is our q. A NotImplementedError check is left out.

**Notebook demonstration:** Switch to notebook 2.6 and run the closed-forms cell: it checks all four formulas against kl_divergence and prints PyTorch’s own source for the Normal and Independent cases.

**Transition:** Use the formula: which Gaussian is closer?

**Source / evidence:** PyTorch source: <a href="https://github.com/pytorch/pytorch/blob/main/torch/distributions/kl.py">torch/distributions/kl.py</a>

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

## kl-expectation: KL is an average: an expectation under p

**Build 1, ask:** What do the weights p(x) add up to?

**Expected:** 1: it is a weighted average.

**Build 2, ask:** Which distribution do we draw x from?

**Expected:** p, the first argument.

**Transition:** Two ways to compute an expectation.

## expectation-sampling: Two ways to compute an expectation

**Build 1, ask:** What is the weighted sum for our weather, with f = log₂ p/q?

**Expected:** ½(1) + ¼(0) + ¼(−1) = 0.25 bits.

**Build 2, ask:** When would you need the integral?

**Expected:** When θ is continuous, like a network weight.

**Build 3, ask:** Why does a plain average over samples work?

**Expected:** About half the draws are sunny, a quarter cloudy, a quarter rainy, so the plain average approaches the weighted sum.

**Transition:** Try it with the weather.

## mc-weather: Sample days and watch the average find the KL

**Build 1, ask:** After 10 days, is the estimate close?

**Expected:** Not necessarily: with few days the shares are off. It settles as they approach ½, ¼, ¼. This is a Monte Carlo estimate of KL.

**Point here:** The bits-per-day demo of 2.4 was already a Monte Carlo estimate: the wasted bits per day, averaged over simulated days.

**Notebook demonstration:** Notebook 2.8: the weather-sampling cell draws 10,000 days and prints the running average at 10, 100, 1,000 and 10,000 days.

**Transition:** Back to our Gaussians, with the formula and with samples.

## kl-sampling: No formula needed: estimate KL by sampling

**Build 1, ask:** Which distribution supplies the draws?

**Expected:** q, the first argument: it does the weighting.

**Build 2, ask:** Should the estimate match exactly?

**Expected:** No: it is random. It is close, and gets closer with more draws.

**Point here:** q₂ = N(1, 3²) against p = N(1, 4²), seed 1000.

**Notebook demonstration:** Switch to notebook 2.7, “The code on the slide, run”: run it, then set S to 100 and to 1,000,000 and rerun. Come back for the live version.

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

**Point here:** PAUSE. All six KLs come from the closed-form Gaussian formula of 2.6.

**Transition:** Search systematically: a grid.

## grid-search: Search a grid of means and widths: hover over it

**Build 1, ask:** Hover near the bottom right. Why is it so dark?

**Expected:** Narrow and off-centre: q puts all its mass where p is small, and KL(q‖p) punishes exactly that.

**Build 2, ask:** How many evaluations for fourteen parameters?

**Expected:** 40¹⁴ ≈ 2.7 × 10²², even with only 40 values per parameter.

**Notebook demonstration:** Notebook 3.1: the grid cell prints the six candidates’ KLs and draws the same heatmap with matplotlib.

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

**Notebook demonstration:** Switch to notebook 3.2, “The code on the slide, run”. Run it: the printed table shows the loss, μ and σ every 25 steps, and the three panels show the loss falling, μ and σ arriving at 1 and 4, and the learned curve landing on p. Drag the step slider in the next cell, then come back for the animation.

**Transition:** Press play.

## olive: Press play: the loss falls as q slides onto p

**Build 1, ask:** Predict: does the centre or the width arrive first?

**Expected:** Students split. The width reaches 4 around step 45, overshoots to 4.52, then settles; μ arrives by step 100.

**Build 2, ask:** Why can both move at once?

**Expected:** Each parameter has its own gradient. Adam steps both by roughly the learning rate each step.

**Point here:** PAUSE (PML L23 12:13). Adam, learning rate 0.05, analytic KL. The faint curves are q at earlier steps.

**Transition:** One dimension is easy. What changes with two correlated unknowns?

## cov-families: In two dimensions, q needs a covariance matrix: three families

**Build 1, ask:** How many numbers describe an isotropic Σ?

**Expected:** One, s², whatever the dimension.

**Build 2, ask:** What can a diagonal q not express?

**Expected:** Correlation: knowing θ₁ tells it nothing about θ₂.

**Build 3, ask:** Why parameterise Σ = LLᵀ instead of Σ directly?

**Expected:** Any lower-triangular L with a positive diagonal gives a valid covariance; a step on Σ’s entries directly could break that, like a negative σ in 1-D.

**Transition:** Fit all three to the same tilted target.

## cov-fit: Fit all three to the same tilted target

**Build 1, ask:** Predict: which q can reach KL = 0?

**Expected:** Only the full one: p is tilted, and only the full family can tilt.

**Build 2, ask:** Watch Σ under each panel. Which entries can each family change?

**Expected:** Isotropic: one number on both diagonal entries. Diagonal: the two diagonal entries. Full: all three, including the off-diagonal one.

**Point here:** Same random start (seed 33), Adam, learning rate 0.05, 600 steps of the closed-form KL. Final KL: isotropic 0.734, diagonal 0.511, full 0.

**Notebook demonstration:** Notebook 3.3: the covariance cell runs all three fits, prints each final Σ and checks it against the formulas on the next slide.

**Transition:** Read off the scores.

## two-d: A q that cannot tilt shrinks instead

**Build 1, ask:** Why are the diagonal widths 0.90 and 0.45, not 1.5 and 0.75?

**Expected:** They are p’s conditional widths, σᵢ√(1 − ρ²) = 1.5 × 0.6 and 0.75 × 0.6: how wide p is along one axis when the other coordinate is held fixed.

**Build 2, ask:** What should we expect of mean-field VI on a real posterior?

**Expected:** Error bars that are too narrow wherever the posterior correlates its parameters.

**Point here:** Under KL(q‖p) the best diagonal q has σᵢ² = 1/(Σₚ⁻¹)ᵢᵢ and the best isotropic q has s² = d / tr(Σₚ⁻¹). The notebook checks both against the fitted values.

**Transition:** Now a target with no formula at all.

## mixture-what: A mixture of Gaussians: first pick a bump, then draw from it

**Build 1, ask:** If bump 1 is picked, where does θ land?

**Expected:** Near −0.2: within about ±0.8, two widths.

**Build 2, ask:** Why multiply each bell by ½?

**Expected:** Each bump makes only half of the draws, so it contributes half of its bell.

**Build 3, ask:** Does p(θ) integrate to 1?

**Expected:** Yes: each bell integrates to 1 and the weights add to 1.

**Point here:** The PML 2023 mixture. z is a latent variable: part of how the data are made, never observed. Lecture 3’s VAE has the same structure, with a continuous z.

**Transition:** Draw from it yourself.

## mixture-sample: Draw from the mixture yourself

**Build 1, ask:** After 1,000 draws, what fraction came from each bump?

**Expected:** About half each, and the stacked histogram follows the teal curve: the weighted sum of the two bells.

**Point here:** Seeded, so the same draws replay. log p(θ) is easy to evaluate: the log of a weighted sum of two Gaussian densities.

**Notebook demonstration:** Notebook 3.4: the sampling cell does the same two steps with Categorical and Normal, 2,000 draws, and colours the histogram by bump.

**Transition:** How far is a Gaussian q from this mixture?

## mixture-no-formula: PyTorch has no formula for this KL. The definition still works.

**Build 1, ask:** How would you compute KL(q‖p) for this p?

**Expected:** Sample θ from q and average log q(θ) − log p(θ). log p(θ) of a mixture is easy to evaluate.

**Build 2, ask:** Is the KL undefined, then?

**Expected:** No. It is a perfectly good integral; we just cannot write it in closed form.

**Point here:** Mixture: ½·N(−0.2, 0.4²) + ½·N(1, 0.5²), from the PML notebook. The full error text is saved in science.json (PyTorch 2.13.0).

**Transition:** Estimate the average by sampling, and optimise it.

## now-optimise: Now optimise it: the code runs, and q does not move

**Build 1, ask:** Predict: where is q after 400 steps?

**Expected:** Most expect it to slide onto the mixture. It does not move.

**Build 2, ask:** Is this a bug?

**Expected:** Not in the usual sense: every line does what it says. Something is missing from the gradient, and 3.5 finds out what.

**Point here:** PAUSE (PML L23 34:49). Plain gradient descent, learning rate 0.05, 5,000 draws per step. With Adam the stuck run wanders aimlessly instead; it still never fits.

**Notebook demonstration:** Notebook 3.4, “Optimise it with sample()”: run it; no error, and q ends where it started.

**Transition:** Section 3.5: draw the computation as a graph.

## missing-path: Why q does not move: the gradient stops at a random node

**Build 1, ask:** Which arrow can the gradient not travel along?

**Expected:** From θ back to μ and σ: sample() returns plain numbers, with no record of how they depend on μ and σ.

**Build 2, ask:** Why isn’t the surviving gradient exactly zero?

**Expected:** log q(θ) contains μ and σ explicitly. That part averages to zero over draws; what remains is noise.

**Point here:** Gaussian pair q = N(0, 2²), p = N(1, 4²), 50,000 shared draws. A score-function estimator is another valid fix (optional slide).

**Notebook demonstration:** Notebook 3.5: the first cell shows draw.backward() failing on a sample() draw, then the tiny surviving gradient against the true one.

**Transition:** Give the gradient a path.

**Source / evidence:** Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>, which follows Kingma’s NIPS 2015 talk

## reparam: The fix: move the randomness into ε, and θ = μ + σε

**Build 1, ask:** Is θ still distributed as N(μ, σ²)?

**Expected:** Yes: shifting and scaling a standard Normal gives N(μ, σ²).

**Build 2, ask:** Compute θ and both derivatives.

**Expected:** θ = 0 + 2(−0.75) = −1.5; ∂θ/∂μ = 1; ∂θ/∂σ = ε = −0.75.

**Build 3, ask:** Can ∂θ/∂σ be negative?

**Expected:** Yes, whenever ε < 0: widening q pushes this draw further left.

**Point here:** PAUSE (PML L24 11:20). Let students compute before build 3. The fixed-noise finite-difference check is an optional slide.

**Transition:** Picture what a fixed ε does.

**Source / evidence:** Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>

## reparam-picture: What a fixed ε does when μ or σ changes

**Build 1, ask:** If μ rises by 1, how far does θ move?

**Expected:** By exactly 1.

**Build 2, ask:** If σ rises by 1?

**Expected:** By ε = −0.75: to the left, because this ε is negative.

**Transition:** The two graphs side by side.

## reparam-compare: Side by side: where the randomness lives

**Build 1, ask:** Where is the stochastic node in each graph?

**Expected:** Top: θ itself. Bottom: ε, which has no parameters.

**Build 2, ask:** What does the reparameterisation trick change: the distribution of θ, or the graph?

**Expected:** Only the graph. θ is N(μ, σ²) either way.

**Point here:** This is the reparameterisation trick of Kingma and Welling (2014), the step that makes VAEs trainable (Lecture 3).

**Transition:** Change one word in the code.

**Source / evidence:** Graph conventions after <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">Lilian Weng, “From Autoencoder to Beta-VAE”</a>

## rsample-fit: Change one word: rsample() fits the mixture

**Build 1, ask:** Predict where q settles.

**Expected:** Over the mixture’s mass, between and around the two bumps.

**Build 2, ask:** Why doesn’t the KL reach 0?

**Expected:** No Gaussian equals a two-bump mixture. 0.056 is the best the family can do; the notebook confirms it with a direct numerical integral.

**Point here:** Same seed, learning rate and draws as the stuck run. Monte Carlo plus reparameterisation: the payoff of Sections 2.7 and 3.5.

**Notebook demonstration:** Notebook 3.5, “Change one word”: run it and compare the two fits in the figure.

**Transition:** Section 3.6: which direction of KL?

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

**Transition:** Step 1: Bayes’ rule, in logs.

**Source / evidence:** A careful written derivation: <a href="https://www.gregorygundersen.com/blog/2021/04/16/variational-inference/">Gregory Gundersen, “The ELBO in Variational Inference”</a>

## elbo-step1: Step 1: Bayes’ rule, in logs

**Build 1, ask:** What is log of a product?

**Expected:** The sum of the logs.

**Build 2, ask:** Which term does not depend on θ?

**Expected:** log p(𝒟): it was integrated over every θ.

**Build 3, ask:** Is log p(𝒟) known?

**Expected:** No: it is the intractable integral. But it is a constant, and that is enough.

**Transition:** Step 2: put this into the KL.

## elbo-step2: Step 2: substitute into the KL

**Build 1, ask:** What does E_q mean, in words?

**Expected:** Draw θ from q, evaluate, average: the Monte Carlo average of 2.7.

**Build 2, ask:** What is inside the brackets now?

**Expected:** Step 1’s expression for log p(θ | 𝒟).

**Build 3, ask:** Which signs changed?

**Expected:** All three: the likelihood and prior become minus, the evidence becomes plus.

**Transition:** Step 3: the constant leaves the average.

## elbo-step3: Step 3: a constant comes out of an average

**Build 1, ask:** Why does adding c to every value add c to the average?

**Expected:** The weights of an average add up to 1, so c gets weight 1 in total.

**Build 2, ask:** Check the table.

**Expected:** (12 + 14 + 16)/3 = 14 = 4 + 10.

**Build 3, ask:** Could log p(θ) also leave the average?

**Expected:** No: it changes with θ. Only terms with no θ can leave.

**Transition:** Step 4: rearrange.

## elbo-step4: Step 4: rearrange, and name the part we can compute

**Build 1, ask:** Check the signs inside the bracket.

**Expected:** −[a + b − c] = −a − b + c: matches the line above.

**Build 2, ask:** Move the average across. What sign does it take?

**Expected:** Plus: it was subtracted on the right, so it is added on the left.

**Build 3, ask:** Which terms in the ELBO can we compute?

**Expected:** All of them, for any θ drawn from q: log p(𝒟 | θ) by a forward pass, log p(θ) and log q(θ) by formula.

**Point here:** Notation: ELBO(q) = E_q[log p(𝒟, θ) − log q(θ)], since p(𝒟 | θ) p(θ) = p(𝒟, θ), the joint.

**Transition:** What is each of the three pieces?

## elbo-pieces: One fixed number = what we can compute + what we cannot

**Build 1, ask:** Which piece does not change when we change q?

**Expected:** log p(𝒟): it involves the model and the data, not q.

**Build 2, ask:** What do we know about the KL even without computing it?

**Expected:** KL ≥ 0. Next slide: why.

**Transition:** Why is KL never negative?

## kl-nonneg: Why KL is never negative

**Build 1, ask:** Where do log t and t − 1 touch?

**Expected:** At t = 1, where both are 0.

**Build 2, ask:** Can a single log ratio log(p/q) be positive?

**Expected:** Yes, wherever p > q. Single terms can have either sign; only the q-weighted average is guaranteed ≥ 0.

**Build 3, ask:** So can the KL be negative?

**Expected:** No. And it is 0 only when p/q = 1 everywhere, that is, q = p.

**Point here:** The same fact via Jensen’s inequality is an optional slide.

**Transition:** So the ELBO is a lower bound.

## elbo-bound: The ELBO is a lower bound on the log evidence

**Build 1, ask:** Below what does the ELBO sit?

**Expected:** The log evidence log p(𝒟), for every q: that is the “lower bound”.

**Build 2, ask:** What is the gap between them?

**Expected:** The KL from q to the posterior: here 0.62 nats.

**Build 3, ask:** Is the ELBO always negative? Is log p(𝒟)?

**Expected:** No, and no. For discrete data log p(𝒟) ≤ 0; for continuous data it can be positive. Only the order is guaranteed.

**Point here:** A finite-sample ELBO estimate can cross log p(𝒟) on an unlucky draw (optional slide); the exact expectation cannot.

**Transition:** What happens to the gap as q improves?

## elbo-max: Raise the ELBO, and the KL falls by exactly as much

**Build 1, ask:** By how much did the KL fall from Beta(10, 10) to Beta(30, 20)?

**Expected:** From 0.62 to 0.05: by 0.57 nats, exactly the ELBO’s rise.

**Build 2, ask:** Why does maximising one minimise the other?

**Expected:** Their sum is the fixed number log p(𝒟).

**Transition:** What does the ELBO ask of q?

## elbo-two-terms: The ELBO in two terms: fit the data, stay near the prior

**Build 1, ask:** Which rule splits the average?

**Expected:** Linearity: E[a + b] = E[a] + E[b].

**Build 2, ask:** Why is the second average minus a KL?

**Expected:** E_q[log p − log q] = −E_q[log q − log p] = −KL(q ‖ prior).

**Build 3, ask:** Which term pulls q towards a narrow, well-fitting θ?

**Expected:** The data term. The KL to the N(0, 1) prior pulls the other way: wide, near zero.

**Transition:** Check the identity where p(𝒟) is known.

## elbo-slider: Check the identity where p(𝒟) is known: a coin

**Build 1, ask:** Find the q with the highest ELBO. What is its KL?

**Expected:** Beta(22, 16), the exact posterior: ELBO = log p(𝒟) = −21.40 and KL = 0.

**Point here:** Conjugate coin: posterior Beta(22, 16), log p(𝒟) = log B(22, 16) − log B(5, 3). The ELBO comes from analytic Beta moments, the KL from its own formula.

**Notebook demonstration:** Notebook 4.2: the coin cells compute log p(𝒟) exactly, then fit a Beta q using only the joint, and check ELBO + KL = log p(𝒟) for every candidate.

**Transition:** Part 5: implement the ELBO for the network.

## bnn-elbo-code: The ELBO for our network, in code

**Build 1, ask:** Which line uses the reparameterisation trick?

**Expected:** theta = q.rsample(...): gradients flow back to μ and ρ.

**Build 2, ask:** Which line uses the Gaussian KL formula?

**Expected:** kl = kl_divergence(q, prior).sum(): seven closed-form terms.

**Notebook demonstration:** Switch to notebook 5.1 and run the VI cell (about 20 seconds): it prints the five restarts’ ELBOs and the split ELBO = E_q[log p(𝒟 | θ)] − KL(q ‖ prior). Then come back for the animation.

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

**Transition:** Seven separate bells: what kind of q is that?

## vi-families: Mean-field VI, and full-covariance VI

**Build 1, ask:** Why is it called mean-field?

**Expected:** From physics: each factor of q responds only to the average (“mean field”) of the others. In practice: independent factors, a diagonal covariance.

**Build 2, ask:** What does full covariance cost for a million weights?

**Expected:** 5 × 10¹¹ numbers in L: far too many to store or optimise.

**Build 3, ask:** Why start the full fit from the mean-field answer?

**Expected:** Full covariance contains mean-field as a special case, so from there it can only improve; from a random start, noisy gradients on 35 numbers can stop at a worse point.

**Point here:** ELBO: mean-field −7.181, full covariance −6.321, log p(𝒟) ≈ −1.429. Full fit: Adam, learning rate 0.01, 4,000 steps, 64 draws per step.

**Notebook demonstration:** Notebook 5.1, “Mean-field against full covariance”: run it (a few seconds); it prints both ELBOs and the largest correlation, and draws both predictive bands against the reference.

**Transition:** Do the error bars improve?

## bnn-vs-ref: Against the reference: the right curve, bands still too narrow

**Build 1, ask:** Where do VI and the reference disagree most?

**Expected:** Beyond the data at both ends, where VI is overconfident.

**Build 2, ask:** Why doesn’t full covariance close the gap?

**Expected:** A Gaussian has one peak. The posterior has eight mirror-image copies of each mode, so up to log 8 ≈ 2.1 nats of the gap is that alone; the remaining shape is not Gaussian either.

**Point here:** Even tilted, one Gaussian stays about 4.9 nats below log p(𝒟). Predictive sd at x = 0: mean-field 0.130, full 0.144, reference 0.162.

**Notebook demonstration:** Notebook 5.1: the last figure shows both VI bands against the reference.

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

## grades: By hand: two runs of this course, two grade distributions (optional)

**Build 1, ask:** How would you score the difference between the two years?

**Expected:** Students propose differences of percentages. Note that C, unchanged, should contribute nothing.

**Build 2, ask:** A fourfold rise and a fourfold fall: should they count the same?

**Expected:** By ratio, yes: both are a factor of 4. That suggests comparing ratios.

**Point here:** p is this year (the distribution we weight by); q is the reference, last year.

**Transition:** Compare grade by grade.

**Source / evidence:** KL intuition after <a href="https://www.youtube.com/watch?v=q0AkK8aYbLY">ritvikmath, “The KL Divergence: Data Science Basics”</a> (YouTube)

## grades-ratios: Compare grade by grade: ratios (optional)

**Build 1, ask:** On the ratio scale, which change looks bigger?

**Expected:** The rise: +3 against −0.75, although both are fourfold.

**Build 2, ask:** What should identical years score?

**Expected:** Zero. Raw ratios give 1.

**Transition:** Find a function of the ratio that behaves.

**Source / evidence:** KL intuition after <a href="https://www.youtube.com/watch?v=q0AkK8aYbLY">ritvikmath, “The KL Divergence: Data Science Basics”</a> (YouTube)

## grades-f: What should a ratio count for? Find the function (optional)

**Build 1, ask:** Why should f(1) be 0?

**Expected:** No change in a grade should add any difference.

**Build 2, ask:** Which test does r − 1/r fail?

**Expected:** Adding up: f(2) + f(2) = 3, but f(4) = 3.75.

**Build 3, ask:** Is the logarithm the only choice?

**Expected:** Up to its base, yes: a smooth f with f(a·b) = f(a) + f(b) must be a logarithm. The base sets the unit.

**Transition:** Now combine the three log ratios.

**Source / evidence:** KL intuition after <a href="https://www.youtube.com/watch?v=q0AkK8aYbLY">ritvikmath, “The KL Divergence: Data Science Basics”</a> (YouTube)

## grades-kl: Weight each grade by this year’s share: that is KL (optional)

**Build 1, ask:** Why is the plain average 0 here?

**Expected:** The rise in A and the fall in B are both fourfold, so their logs cancel, and C did not change.

**Build 2, ask:** Why weight by this year’s shares?

**Expected:** We are describing this year’s students: grades that many of them got should count more.

**Build 3, ask:** What would identical years give?

**Expected:** 0: every ratio is 1 and every log is 0.

**Notebook demonstration:** Notebook appendix: the grades cell computes the ratios, the logs and the weighted sum, and checks it against torch.distributions.kl_divergence.

**Transition:** What unit is 0.42 in?

**Source / evidence:** KL intuition after <a href="https://www.youtube.com/watch?v=q0AkK8aYbLY">ritvikmath, “The KL Divergence: Data Science Basics”</a> (YouTube)

## nats-bits: The unit: nats, or bits (optional)

**Build 1, ask:** Why does changing the base only rescale the KL?

**Expected:** log₂ x = ln x / ln 2: every term scales by the same factor 1/ln 2 ≈ 1.44.

**Build 2, ask:** What would a KL of 0 nats mean?

**Expected:** e⁰ = 1: grades are equally probable under both years, so the two distributions match.

**Point here:** e^KL is the geometric mean of p(x)/q(x) over x drawn from p.

**Transition:** Does it matter which year is the reference?

## asymmetry: Swap the reference year: a different number (optional)

**Build 1, ask:** Which direction do you expect to be larger?

**Expected:** From this year’s view the A’s are a big change that many students have, so 0.64 is larger.

**Build 2, ask:** Is KL a distance in the everyday sense?

**Expected:** No: a distance is the same both ways. KL is a divergence.

**Point here:** The first grades example happens to give 0.42 in both directions, because it is a mirror image (A and B swap shares). That is why this slide uses a second pair.

**Transition:** Why does the reference matter?

## asymmetry-why: Why the reference matters (optional)

**Build 1, ask:** Which grade dominates each direction?

**Expected:** A from this year’s view; B from last year’s.

**Build 2, ask:** So which order will VI use, and why might it matter?

**Expected:** VI will put q first, KL(q ‖ p). Part 3 shows what that order does to the fit.

**Transition:** Sample grades to estimate it.

## mc-grades: Sample grades and watch the average find the KL (optional)

**Build 1, ask:** After 10 draws, is the estimate close?

**Expected:** Not necessarily: with few draws the shares are off, so the estimate wobbles. It settles as the counts approach 40/10/50.

**Point here:** The exact value is the weighted sum (analytic); the running average is the Monte Carlo estimate. Draws are seeded, so the sequence replays identically.

**Notebook demonstration:** Notebook appendix: the grades-sampling cell draws 10,000 grades and prints the running average at 10, 100, 1,000 and 10,000 draws.

**Transition:** Back to the optional slides.

## kl-estimators: A KL estimator that is never negative (optional)

**Build 1, ask:** Why does adding r − 1 not change the average?

**Expected:** E_q[r] = ∫ q · p/q = ∫ p = 1, so E_q[r − 1] = 0: it only removes noise.

**Build 2, ask:** When can the better estimator be worse?

**Expected:** When p has heavier tails than q: r = p/q can then have infinite variance. Here p is narrower than q, so r stays bounded.

**Point here:** q = N(0, 2²) against p = N(0.5, 1.5²), exact KL 0.157; 400 repeats per size. From Jia-Bin Huang’s video and John Schulman’s note “Approximating KL divergence”; GRPO uses this estimator for its KL penalty.

**Source / evidence:** After <a href="https://www.youtube.com/watch?v=tXE23653JrU">Jia-Bin Huang</a> and <a href="http://joschu.net/blog/kl-approx.html">John Schulman, “Approximating KL divergence”</a>

## evidence-mc: Sampling from the prior: a few lucky draws carry everything (optional)

**Build 1, ask:** Why are the small-S estimates so low?

**Expected:** Prior draws almost never land where the likelihood is high, like θ<sub>B</sub>; the rare good draws dominate the average.

**Build 2, ask:** What happens with a million parameters?

**Expected:** The chance that a prior draw lands near the posterior shrinks exponentially; no feasible S is enough.

**Point here:** Effective number of draws, out of 10⁷: 77, 93, 134, 108. The largest single draw carries 8%, 4%, 4%, 4% of the total.

**Transition:** Why do we want the posterior at all?

## coin-evidence: A KL value is how fast evidence piles up (optional)

**Build 1, ask:** Predict: after 30 tosses, how many times more likely are the data under 0.8 than under 0.5?

**Expected:** Most guess far too low. It is about 325×.

**Build 2, ask:** Why does a small KL eventually matter so much?

**Expected:** Log evidence adds up toss by toss, so the ratio grows exponentially with n.

**Point here:** PAUSE. A simulation of 20,000 sequences gives a mean log ratio after 30 tosses of 5.780, against 30 × KL = 5.782.

**Transition:** Sums become integrals for continuous unknowns.

## exit-transfer: A sensor has an unknown constant bias (optional)

**Build 1, ask:** Identify the unknown, the observations and one possible q family.

**Expected:** θ is the bias; 𝒟 contains the readings; a Normal q is a sensible choice.

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

**Point here:** The tangent-line argument of Section 4.1 is the same fact without naming Jensen.

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

**Expected:** The term accounting for the change in the probability of drawing θ.

**Build 2, ask:** Does this identity require θ = μ + σε?

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

**Build 2, ask:** Does a stable loss establish q = p(θ | 𝒟)?

**Expected:** No. The family or optimisation can leave a positive gap.

**Point here:** Recall the diagonal q in Section 3.3: its best possible KL was 0.51.

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
