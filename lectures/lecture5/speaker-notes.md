# Autoencoders and the VAE in one hour: speaker notes

Planned main route: 59 minutes. Timing includes student responses and the specified short demonstrations. Optional extensions are outside this budget.

## Pacing

| Episode | Minutes |
|---|---:|
| Lecture map | 1 |
| Part 1 · Where we start: the ELBO | 6 |
| Part 2 · Autoencoders: squeeze, then rebuild | 16 |
| Part 3 · Can an autoencoder generate? | 8 |
| Part 4 · The VAE: Lecture 1’s ELBO, one image at a time | 29 |

## contents: Lecture 5 at a glance: four parts, seventeen sections

**Build 1, ask:** Which part leans most on Lecture 1?

**Expected:** Part 4: the VAE is Lecture 1’s ELBO with a code per image.

**Point here:** Short on time: the optional slides carry the detours (PCA, the two extremes of β, the encoder against the exact posterior). Lecture 2 has the full two-session version.

**Transition:** Part 1: where we start.

## part-1: Part 1 · Where we start: the ELBO

**Build 1, ask:** What did Lecture 1 maximise, and why did that help?

**Expected:** The ELBO. log p(𝒟) is fixed, so raising the ELBO lowers the KL from q to the posterior.

**Transition:** Section 1.1.

## recap-elbo: From Lecture 1: one objective we can compute

**Build 1, ask:** Which part of log p(𝒟) could we compute?

**Expected:** The ELBO. The KL to the posterior needs p(𝒟), which we cannot compute.

**Build 2, ask:** What does each ELBO term ask of q?

**Expected:** The first: put q on θ values that explain the data. The second: do not stray far from the prior.

**Build 3, ask:** What did gradient descent change in Lecture 1?

**Expected:** ψ: the means and log-widths of q, one bell per weight of the tiny network.

**Point here:** Colours follow Lecture 1’s Part 4: q purple, likelihood orange, prior green, evidence blue, the KL to the posterior red.

**Transition:** Two tools we will need again.

## recap-tools: Two tools from Lecture 1 we will use again

**Build 1, ask:** Why not sample z directly?

**Expected:** A sample() call cuts the gradient path from the loss back to μ and σ.

**Build 2, ask:** What is the KL for μ = 2, σ = 1?

**Expected:** ½(4 + 1 − 0 − 1) = 2 nats.

**Transition:** What we are building towards.

## realfake: Real or generated? Six of these twelve digits are fake

**Build 1, ask:** Which six are fake? Write the positions.

**Expected:** Answers vary; take a quick show of hands for two or three digits.

**Build 2, ask:** Which fakes gave themselves away, and how?

**Expected:** Broken or doubled strokes and blurry edges give some away; the cleanest fakes are hard to tell.

**Point here:** Real digits: six random MNIST test images. Fakes: the first six draws z ~ N(0, I) through Lecture 2’s 16-number VAE, not chosen by hand.

**Transition:** The route to that network.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## today: Today: two networks, then Lecture 1’s objective

**Build 1, ask:** What will the autoencoder’s 2 numbers have to keep?

**Expected:** Whatever is needed to rebuild the digit: which digit, its slant, its thickness.

**Build 2, ask:** What would “generate” mean for the second network?

**Expected:** Turn numbers we invent into a new, convincing digit.

**Build 3, ask:** Which piece of this loss have you seen before?

**Expected:** The KL to N(0, I): Lecture 1’s KL to the prior, with tool ② as its formula. The squared error turns out to be the likelihood term.

**Point here:** The loss on this slide drops a constant (784 log(σ√2π)); Section 4.4 shows where every term comes from.

**Transition:** Part 2: autoencoders.

## part-2: Part 2 · Autoencoders: squeeze, then rebuild

**Build 1, ask:** Why would squeezing a digit into 2 numbers be useful?

**Expected:** The 2 numbers must keep what matters about the digit. Later we will try inventing them.

**Transition:** Section 2.1.

## mnist: MNIST: 70,000 handwritten digits, 28 × 28 pixels each

**Build 1, ask:** How many pixels does each image have?

**Expected:** 28 × 28 = 784.

**Build 2, ask:** Why keep one running example?

**Expected:** Every model can be checked on the same 3: its code, its reconstruction and its loss.

**Transition:** Squeeze those 784 numbers into 2.

**Source / evidence:** MNIST (LeCun, Cortes and Burges), grey levels scaled to [0, 1].

## ae-arch: An autoencoder: encoder, short code, decoder

**Build 1, ask:** Why squeeze through so few numbers?

**Expected:** To force it to keep only what matters about the digit.

**Build 2, ask:** What would happen with a 784-number code?

**Expected:** It could copy the input and learn nothing useful.

**Build 3, ask:** What is the target output?

**Expected:** The input itself: the network is trained to rebuild x.

**Point here:** Hinton and Salakhutdinov (Science, 2006) made deep autoencoders work; the idea goes back to the 1980s. One hidden layer of 400 units follows Kingma and Welling (2014).

**Transition:** The same in PyTorch.

## ae-code: Two small networks in PyTorch

**Build 1, ask:** How many weights does the encoder have?

**Expected:** 784 × 400 + 400 + 400 × 2 + 2 ≈ 315,000.

**Build 2, ask:** Why no activation after the encoder’s last layer?

**Expected:** The code can be any real number; a ReLU would forbid negative codes.

**Transition:** What does it minimise?

## ae-loss: The loss: how far each rebuilt pixel is from the real one

**Build 1, ask:** Which weights does the loss depend on?

**Expected:** Both φ (encoder) and θ (decoder).

**Build 2, ask:** Where are the large errors?

**Expected:** Along the strokes: the rebuilt 3 is blurrier and thinner than the real one.

**Build 3, ask:** Why sum over pixels but average over images?

**Expected:** The sum is the error of one image; averaging over images keeps the loss independent of the batch size.

**Point here:** Teal = target, purple = prediction, red = loss, as in every loss slide of the series.

**Transition:** Before training: what comes out?

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## ae-untrained: Before training: always look at what the network outputs

**Build 1, ask:** Why is every output close to 0.5?

**Expected:** Small random weights make the pre-sigmoid values near 0, and sigmoid(0) = 0.5.

**Build 2, ask:** What would it mean if training ended near 185?

**Expected:** The network learned nothing: a bug, or a learning rate that is far off.

**Transition:** Train it.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## ae-step: One training step: forward, loss, backward, update

**Build 1, ask:** How many updates in total?

**Expected:** 14,070.

**Build 2, ask:** Does the encoder get a gradient even though the loss compares only images?

**Expected:** Yes: the reconstruction depends on the code, which depends on the encoder’s weights.

**Transition:** Watch the reconstructions form.

## ae-train: Press play: reconstructions sharpen as it trains

**Build 1, ask:** What does the network learn first?

**Expected:** The average digit: a blurry blob that lowers the error everywhere.

**Build 2, ask:** Which digits stay hardest to rebuild from 2 numbers?

**Expected:** Ones that resemble other digits, like 4 and 9, or unusual shapes.

**Transition:** How much does the code size matter?

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## code-size: More numbers in the code, sharper rebuilds

**Build 1, ask:** What does the 2-number code lose?

**Expected:** Stroke thickness, slant and small details.

**Build 2, ask:** Which of these would you pick?

**Expected:** For rebuilding, the widest. For generating, wait for Part 3.

**Transition:** Where do the 2-number codes land?

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## ae-scatter: Where the codes land: 1,000 test digits at their 2 numbers

**Build 1, ask:** Which digits are still mixed?

**Expected:** 4 and 9; some 3, 5 and 8.

**Build 2, ask:** The codes run from about −5 to 22. Is that a problem?

**Expected:** Not for rebuilding. For generating, Part 3 will show it is.

**Transition:** Part 3: can the decoder generate?

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## part-3: Part 3 · Can an autoencoder generate?

**Build 1, ask:** What would a good random code have to look like?

**Expected:** Like the codes real digits get, so the decoder knows what to draw.

**Transition:** Section 3.1.

## gen-question: Keep the decoder, feed it random numbers

**Build 1, ask:** What should a random code look like, to give a good digit?

**Expected:** Like the codes real digits get.

**Build 2, ask:** Why might N(0, I) be a poor choice?

**Expected:** Nothing in training told the codes to look like N(0, I): they run from −5 to 22.

**Transition:** Feed such draws to three decoders.

## ae-random: Random codes through three autoencoders

**Build 1, ask:** Which row looks most like digits?

**Expected:** The 2-number one.

**Build 2, ask:** The 128-number model rebuilt best. Why does it generate worst?

**Expected:** Its codes for real digits are nowhere near where N(0, I) puts its draws.

**Build 3, ask:** What would you check to find out?

**Expected:** Plot where real codes are, next to the random codes.

**Transition:** Plot where the codes are.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## overlay: Put the random codes on top of the real ones

**Build 1, ask:** Hover a red dot near the origin, then one in an empty area. What does the decoder draw?

**Expected:** Near real codes: a digit; in empty places: blurs or odd strokes.

**Build 2, ask:** For the 128-number model, where would the red points fall?

**Expected:** Even further from the real codes: in 128 numbers the space is mostly empty.

**Point here:** The decoder runs in the browser from lecture2/data/nets.json.

**Transition:** Decode a whole grid of codes.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## ae-grid: Decode a whole grid of codes

**Build 1, ask:** Which parts of the grid are useful for generation?

**Expected:** Those over the clusters of real codes.

**Build 2, ask:** What does the decoder do far from any real code?

**Expected:** It extrapolates from the nearest digits: exaggerated or broken shapes.

**Transition:** What we want instead.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## wishes: What we saw, and what we want instead

**Build 1, ask:** How could we push codes towards N(0, I)?

**Expected:** Penalise codes that stray from it, for example with a KL to N(0, I).

**Build 2, ask:** How could each image own a cloud instead of a point?

**Expected:** Let the encoder output a mean and a spread, and draw the code from them.

**Build 3, ask:** Which tool from Lecture 1 will we need?

**Expected:** Variational inference and the ELBO, with both tools from Section 1.1.

**Transition:** Part 4: the VAE.

## part-4: Part 4 · The VAE: Lecture 1’s ELBO, one image at a time

**Build 1, ask:** Which wall did Lecture 1 hit?

**Expected:** The evidence p(𝒟): an integral with no formula, so the posterior was out of reach.

**Transition:** Section 4.1.

## vae-pgm: A generative model with a hidden code

**Build 1, ask:** Which circle is observed?

**Expected:** x, the image (shaded). z is never observed.

**Build 2, ask:** What does the arrow mean?

**Expected:** x is drawn given z: through the decoder, then pixel noise.

**Build 3, ask:** How many sets of decoder weights are there?

**Expected:** One, shared by all 60,000 images.

**Point here:** Graphical-model notation (Bishop, PRML chapter 8): shaded = observed, box = repeated for each example, a dot outside = a parameter. Prior green, likelihood orange, as in Lecture 1.

**Transition:** How to draw a new image from this model.

## vae-draw: To draw an image, follow the arrows: first z, then x

**Build 1, ask:** Does generating need the encoder?

**Expected:** No: only the prior and the decoder.

**Build 2, ask:** Why is this a 9 and not another digit?

**Expected:** Because of where this z sits: the decoder has learned which region of codes means which digit.

**Build 3, ask:** Why display the mean image?

**Expected:** The noise carries no information about the digit; it only makes speckle.

**Transition:** The likelihood, in numbers.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## sigma: The likelihood, in numbers: a squared error

**Build 1, ask:** What happens to the first term if σ is small?

**Expected:** Errors cost more: the model is pushed to rebuild precisely.

**Build 2, ask:** Where have you seen squared error = Gaussian likelihood before?

**Expected:** Linear regression: least squares is maximum likelihood with Gaussian noise.

**Build 3, ask:** Is σ learned here?

**Expected:** No, fixed at 0.2. It can be learned; we keep it fixed for clarity.

**Transition:** To learn θ we need p(x). Can we get it?

## vae-wall: Which codes made our 3? Bayes’ rule, and Lecture 1’s wall

**Build 1, ask:** Why do we need the posterior?

**Expected:** To know which codes explain each image, so the decoder can be trained on them; and log p(x) is what training should maximise.

**Build 2, ask:** Why can we not compute p(x)?

**Expected:** The decoder is a neural network: the integral has no closed form.

**Build 3, ask:** What did Lecture 1 do instead?

**Expected:** Chose a simple q and maximised the ELBO.

**Transition:** What the posterior looks like for our 3.

## posterior3: For our 3: which codes could have made it?

**Build 1, ask:** Why is the patch so small?

**Expected:** 784 pixels with σ = 0.2 pin the code down very precisely.

**Build 2, ask:** Why does the grid trick fail for 16 numbers?

**Expected:** A grid of 100 points per number would need 100¹⁶ points: the same count as Lecture 1’s 10¹⁴.

**Transition:** Line Lecture 1 up with today.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## cast: Same problem as Lecture 1, new names

**Build 1, ask:** Which row changed the most?

**Expected:** The approximation: one q became a q for every image, produced by a network.

**Build 2, ask:** Why not fit 60,000 separate q’s, as Lecture 1 fitted one?

**Expected:** It would take a separate optimisation for every image, and again for every new image. One encoder pass is instant.

**Point here:** Fitting one network that outputs q for any input is called amortised inference.

**Transition:** Lecture 1’s ELBO with the new names.

## vae-elbo: Lecture 1’s ELBO, for one image

**Build 1, ask:** Which term is new compared with Lecture 1?

**Expected:** None: every term has a counterpart. Only the names changed.

**Build 2, ask:** Which term rewards a good reconstruction?

**Expected:** The first: the expected log-likelihood of the image.

**Build 3, ask:** What does maximising the ELBO do to θ, and to φ?

**Expected:** θ: the images become more probable under the model. φ: q moves towards the true posterior.

**Point here:** The same derivation as Lecture 1, Section 4.1, with θ → z and 𝒟 → x; Lecture 2, Section 6.3 writes it out for one image.

**Transition:** Who computes q for each image?

## encoder-q: The encoder is q: one network gives q for every image

**Build 1, ask:** Which way does each arrow point?

**Expected:** Decoder: z to x (generate). Encoder: x to z (infer).

**Build 2, ask:** What does the encoder output for one image?

**Expected:** Two means and two log variances: q_φ(z | x) = N(μ, σ²).

**Build 3, ask:** What is variational about the VAE?

**Expected:** q is a member of a simple family (diagonal Gaussians), fitted by maximising the ELBO: Lecture 1’s variational inference.

**Point here:** Kingma and Welling (2014), figure 1: solid = generative model p_θ, dashed = variational approximation q_φ.

**Transition:** The network, with a cloud in the middle.

## vae-arch: The network: an encoder that outputs a cloud, then tool ①

**Build 1, ask:** Why output log σ² instead of σ?

**Expected:** Any real number is a valid log variance; σ = exp(½ log σ²) is then positive.

**Build 2, ask:** Why reparameterise instead of sampling z directly?

**Expected:** sample() blocks the gradient path from the loss to μ and σ.

**Build 3, ask:** At test time, which code do we use to rebuild?

**Expected:** Usually the mean μ.

**Transition:** Write the loss.

## vae-loss: The loss for one image is minus the ELBO

**Build 1, ask:** Which term pulls the codes towards N(0, I)?

**Expected:** The KL term.

**Build 2, ask:** Why is one draw enough?

**Expected:** Over many steps and images the noise averages out; the estimate is unbiased.

**Build 3, ask:** What does the KL do to a code with μ = 5?

**Expected:** Adds about ½ × 25 = 12.5 per number: strong pressure back towards the origin.

**Point here:** The constant is 784 log(σ√2π) = −541.4 for σ = 0.2. Shape of the loss: data fit + penalty, like ridge regression, but here the penalty comes out of the ELBO.

**Transition:** Put in the numbers for our 3.

## vae-worked: One step for our 3, with numbers

**Build 1, ask:** Why is z so close to μ?

**Expected:** σ is tiny (about 0.01): the code of this 3 is pinned down precisely.

**Build 2, ask:** Which term is larger, and what does that say?

**Expected:** Reconstruction, by far: fitting the pixels dominates this model.

**Build 3, ask:** What does backward() update?

**Expected:** The decoder’s θ through the reconstruction; the encoder’s φ through both μ, σ (via z) and the KL.

**Point here:** Check: L − 541.4 = −243.7, close to −ELBO = −243.4. The loss drops the constant, and one draw is noisy.

**Transition:** The whole thing in PyTorch.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## vae-code: The VAE in PyTorch: three changes to the autoencoder

**Build 1, ask:** What are the three changes?

**Expected:** Two outputs per code number, a random draw of z, and the KL term.

**Build 2, ask:** How many lines of the training loop change?

**Expected:** None: only the model and the loss.

**Transition:** Why does adding the KL make sampling work?

## vae-tug: Two terms pull in opposite directions

**Build 1, ask:** What does the reconstruction term want the clouds to do?

**Expected:** Shrink (less noise in z) and spread apart (no confusion between images).

**Build 2, ask:** What does the KL term want?

**Expected:** Move to the centre and widen to σ = 1.

**Build 3, ask:** Where does the balance land?

**Expected:** Clouds near the centre, small but not tiny, slightly overlapping.

**Build 4, ask:** Why does sampling now work?

**Expected:** The codes fill the region N(0, I) samples from, and the decoder was trained across it.

**Point here:** A sketch, not data. Section 4.7 shows the real version: β = 0 is panel one, β = 256 is panel two.

**Transition:** What the trained VAE does.

## vae-samples: Now sample: z ~ N(0, I), then decode

**Build 1, ask:** What changed between the first two rows?

**Expected:** Training made the codes fill N(0, I), so random codes land on digits.

**Build 2, ask:** Why are the VAE samples still a little blurry?

**Expected:** Each pixel is an independent Gaussian around a mean image; the mean of plausible strokes is soft.

**Build 3, ask:** Which digits are missing from the autoencoder row?

**Expected:** Most of them: random codes reach only the digits whose codes sit near the origin.

**Transition:** Where the codes are now.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## vae-scatter: The codes now fill N(0, I)

**Build 1, ask:** Compare with the autoencoder’s map. What changed?

**Expected:** Codes fill a round region around the origin with no large gaps; the scale is about 1, not 20.

**Build 2, ask:** Is the cloud exactly N(0, I)?

**Expected:** Not quite: a little wider (spread 1.7) and slightly off-centre, since the KL only pulls codes towards N(0, I).

**Transition:** Decode a grid, and walk between two digits.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## vae-grid: Decode a grid of codes: digits everywhere

**Build 1, ask:** Compare with the autoencoder’s grid. What is different?

**Expected:** No junk: the whole region N(0, I) samples from is used.

**Build 2, ask:** Why are the in-between images plausible now?

**Expected:** During training each image owned a small cloud of codes, so the decoder learned to draw something sensible everywhere in between.

**Transition:** One knob: the weight on the KL.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## beta-what: Put a weight on the KL: β

**Build 1, ask:** Which value of β gives back the real ELBO?

**Expected:** β = 1.

**Build 2, ask:** What do you predict at the two extremes?

**Expected:** β = 0: codes spread out like an autoencoder’s. Huge β: codes collapse onto the prior and stop carrying information.

**Transition:** Seven trained VAEs.

## beta: Seven VAEs, β from 0 to 256: try them

**Build 1, ask:** At which β do samples look best? When do reconstructions start to blur?

**Expected:** Around β = 1 to 4. By β = 64 reconstructions blur; at 256 the KL is 0.00, every code decodes to the same blob (posterior collapse).

**Point here:** KL per image: β = 0 gives 68.3, β = 1 gives 9.0, β = 256 gives 0.00. The optional slides take the two extremes apart.

**Transition:** Wrap-up.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## summary: Autoencoder and VAE, side by side

**Build 1, ask:** Which row explains why sampling works?

**Expected:** The loss row: the KL pulls codes into N(0, I), where random draws land.

**Build 2, ask:** Which Lecture 1 ideas did the VAE use?

**Expected:** Variational inference and the ELBO, reparameterisation, and the closed-form KL between Gaussians.

**Transition:** What comes next.

## next: Next: can we drop the pixel likelihood altogether?

**Build 1, ask:** Why does a pixel-wise Gaussian encourage soft images?

**Expected:** Squared error favours the mean of plausible images, which is blurry.

**Build 2, ask:** What else could tell a generator its images are wrong?

**Expected:** A classifier trained to tell real from generated: a learned critic.

**Transition:** End of the main route. Optional slides follow.

## opt-ae-test: Did the autoencoder memorise? Training against test error (optional)

**Build 1, ask:** What would memorising look like in these numbers?

**Expected:** Very low training error and much higher test error.

**Build 2, ask:** Why is the test error a little higher?

**Expected:** The weights were tuned to the training digits; new digits fit a little less well.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-linear-ae: Without the ReLUs and the sigmoid, an autoencoder rediscovers PCA (optional)

**Build 1, ask:** Why does a linear autoencoder end up like PCA?

**Expected:** PCA is the best linear encoder–decoder pair under squared error; training finds it, up to a rotation within the plane.

**Build 2, ask:** What do the ReLUs buy?

**Expected:** Curved maps: a better reconstruction from the same few numbers.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-std-normal: N(0, I) in two dimensions: two independent standard bells (optional)

**Build 1, ask:** What does the I in N(0, I) say?

**Expected:** Unit variance and no correlation between the numbers.

**Build 2, ask:** Where will almost all random codes land?

**Expected:** Within about 2 or 3 of the origin.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-explore-ae: Choose a code yourself: the autoencoder (optional)

**Build 1, ask:** Which of the three ways gives the best digits?

**Expected:** Encoding a real digit. Random codes and clicks far from the dots mostly give smudges.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-ae-interp: The autoencoder: walk in a straight line from a 1 to a 0 (optional)

**Build 1, ask:** Why might the 16-number path cross empty space?

**Expected:** The straight line between two codes need not stay where real codes are; in 16 dimensions most of the space between them is empty.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-clouds: Another road to the same loss: points become clouds (optional)

**Build 1, ask:** Which parts of the code plane does an autoencoder’s decoder see in training?

**Expected:** Only the exact codes of training images.

**Build 2, ask:** What does the decoder learn when clouds overlap?

**Expected:** To draw something sensible everywhere the clouds reach, including between images.

**Point here:** Lecture 2, Section 6.2 takes this road first (“Road 1”) and shows that the ELBO gives the same loss.

## opt-q3: Check the encoder: its q for our 3, against the true posterior (optional)

**Build 1, ask:** Which q is better, and by how much?

**Expected:** The per-image fit, by 0.68 nats.

**Build 2, ask:** Why is the encoder’s q not perfect?

**Expected:** One network must serve every image; it cannot be exactly right for each.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-vae-curves: While training: reconstruction falls, the KL settles (optional)

**Build 1, ask:** What would a KL of zero mean?

**Expected:** q equals the prior for every image: the code ignores the image. That is the β = 256 collapse.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-explore-vae: Choose a code yourself: the VAE against the autoencoder (optional)

**Build 1, ask:** Click in the gaps between digit groups. What do you see now?

**Expected:** Plausible in-between shapes: the space between digits is filled.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-beta-zero: β = 0: is it just an autoencoder? (optional)

**Build 1, ask:** Is a VAE with β = 0 the same as an autoencoder?

**Expected:** Different by construction, but in practice it learns σ ≈ 0 and becomes one.

**Build 2, ask:** What does the KL term do to σ?

**Expected:** It stops σ shrinking: −log σ² grows as σ → 0.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).

## opt-beta-collapse: β = 256: every code decodes to the same blur (optional)

**Build 1, ask:** Why is KL = 0 the cheapest solution here?

**Expected:** β × KL dominates the loss; reconstruction error is worth less than any information in the code.

**Build 2, ask:** How would you spot collapse during training?

**Expected:** The KL term falls to about 0 and samples all look the same.

**Source / evidence:** Networks and numbers from Lecture 2: <code>lecture2/scripts/train.py</code> and <code>science.py</code> (MNIST, fixed seeds).
