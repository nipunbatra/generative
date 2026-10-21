Here is the complete, opinionated, and detailed pedagogical plan for Lecture 5 on Diffusion Models, designed for Nipun Batra’s ES 667 Deep Learning course.

---

# Lecture 5: Diffusion Models

## 1. Core Identity
*   **Title:** Diffusion Models: Destruction as Creation
*   **Central Question:** Can we generate complex data by learning to reverse a gradual, random destruction process?
*   **Core Insight:** If we destroy an image by adding noise in many tiny, predictable steps, a neural network can learn to estimate the noise at each step, effectively reversing the arrow of time to sculpt data from pure static.
*   **Learning Outcomes:** After this lecture, a student can:
    1. Compute the closed-form forward noising step for any timestep $t$ using the reparameterisation trick.
    2. Explain visually and mathematically why the reverse step is only Gaussian if the forward step is infinitesimally small.
    3. Derive the diffusion ELBO by mapping the Markov chain to a fixed-encoder VAE.
    4. Prove how the KL divergence between two Gaussians reduces to the $\epsilon$-matching MSE loss.
    5. Implement the DDPM training loop and sampling loop in PyTorch.
    6. Contrast DDPM (stochastic) and DDIM (deterministic) sampling.
    7. Explain Classifier-Free Guidance (CFG) as an algebraic manipulation of Bayes' rule on score functions.

---

## 2. The Story Arc & Session Split

**Session 1: The Math of Destruction and Reconstruction (70 mins)**
*   **Part 1: The Core Idea & Forward Process.** *Question: How do we destroy data mathematically?* Opens with the egg-scrambling intuition. Earns the $\sqrt{1-\beta_t}$ scaling factor by watching naive addition explode. Closes with the closed-form jump to $t$.
*   **Part 2: The Reverse Process & The "100 Cats" Problem.** *Question: Why can't we just denoise in one step?* Earns the $T=1000$ requirement by showing that a 1-step posterior is an unlearnable multimodal mess, but a tiny-step posterior is a simple Gaussian.
*   **Part 3: The Diffusion ELBO.** *Question: How do we train this?* Opens with the intractable marginal. Earns the ELBO by mapping diffusion directly to Lecture 3's VAE. Earns the $x_0$-conditioning trick to make the posterior tractable.
*   **Part 4: The Objective & The Network.** *Question: What exactly is the U-Net predicting?* Derives the reduction of KL divergence to MSE on $\epsilon$. Closes with the "Ideal Denoiser" paradox to show why neural network inductive biases are necessary.

**Session 2: Generation, Speed, and Control (70 mins)**
*   **Part 5: Sampling & The Score Perspective.** *Question: How is an image born?* Opens with the DDPM algorithm. Connects predicting noise to estimating the gradient of the data distribution (score matching).
*   **Part 6: Schedules & Speedups.** *Question: Why does sampling take so long?* Earns the Cosine schedule by watching Linear destroy data too fast. Earns DDIM by removing the stochastic variance term.
*   **Part 7: Text Conditioning & Classifier-Free Guidance.** *Question: How do we tell the model *what* to generate?* Earns CFG by deriving Bayes' rule on scores and showing the fidelity/diversity tradeoff.
*   **Part 8: Latent Diffusion.** *Question: How do we scale to 1024x1024?* Closes the loop by combining Lecture 3's VAE with Lecture 5's DDPM.

---

## 3. Slide-by-Slide Storyboard & Concrete Numbers

**Running Example Numbers:** We track a single pixel of our held-out MNIST '3'.
*   Clean pixel: $x_0 = 0.8$.
*   Schedules: $\beta_1 = 0.1$, $\beta_2 = 0.2$. (Hence $\alpha_1 = 0.9, \alpha_2 = 0.8$).
*   Timestep: $t=2$. $\bar{\alpha}_2 = 0.9 \times 0.8 = 0.72$.
*   Noise draws: $\epsilon_1 = 0.3$, $\epsilon_2 = -0.5$, overall $\epsilon = -0.1$.

### Part 1: The Forward Process
*   **Slide 1.1: Destruction as Creation.** *Visual:* Gemini illustration of an egg scrambling. *Ask:* Can we unscramble it? *Answer:* No, unless we have the film and learn to reverse frame 57 to 56.
*   **Slide 1.2: The Forward Markov Chain.** *Visual:* MNIST '3' fading to static. *Math:* $q(x_t|x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, \beta_t I)$. *Ask:* If we just add noise $\beta_t=0.1$, what happens to the pixel variance after 1000 steps? *Answer:* It explodes to 100. (Adapts Deepia 03:29).
*   **Slide 1.3: Fixing the Variance.** *Visual:* Widget 1 (Exploding vs Scaled points). *Math:* $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$. *Concrete:* $x_1 = \sqrt{0.9}(0.8) + \sqrt{0.1}(0.3) = 0.758 + 0.094 = 0.852$.
*   **Slide 1.4: The Shortcut (Reparameterisation).** *Ask:* Do we need a 1000-step loop to train on $x_{1000}$? *Answer:* No. *Math:* Derive step-by-step. Substitute $x_1$ into $x_2$. Use Lecture 1 knowledge: sum of independent Gaussians. Reveal $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
*   **Slide 1.5: The Shortcut in Numbers.** *Concrete:* $x_2 = \sqrt{0.72}(0.8) + \sqrt{0.28}(-0.1) = 0.678 - 0.052 = 0.626$. *Visual:* Widget 2 (Step-by-step vs Direct Jump).

### Part 2: The Reverse Process & The "100 Cats"
*   **Slide 2.1: Reversing the Arrow.** *Math:* We want $p(x_{t-1}|x_t)$. *Ask:* Why is this hard? *Answer:* It requires knowing the marginal $p(x_t)$, which means integrating over every possible image.
*   **Slide 2.2: The "100 Cats" Problem.** *Visual:* Widget 3 (1D Bimodal Posterior). A noisy point sits at 0. *Ask:* Did this come from the left mode (a cat) or right mode (a dog)? *Answer:* We don't know. The true posterior $q(x_0|x_T)$ is highly multimodal. If we force a network to predict it in one step, it predicts the mean (a blurry mess). (Adapts MIT L2 / Ari Seff).
*   **Slide 2.3: Why Small Steps?** *Visual:* Same widget, but $\beta_t$ is tiny. The modes haven't mixed yet. *Math:* For infinitesimally small $\beta_t$, the reverse of a Gaussian is a Gaussian. *Transition:* We need a network $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$.

### Part 3: The Diffusion ELBO
*   **Slide 3.1: Diffusion is a VAE.** *Visual:* Lecture 3 VAE diagram stacked directly above the unrolled Diffusion Markov chain. (Adapts Ari Seff 06:28). *Ask:* What is the encoder? *Answer:* The fixed forward process $q(x_{1:T}|x_0)$.
*   **Slide 3.2: The ELBO Recap.** *Math:* Recall Lecture 3: $\text{ELBO} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$. *Build:* Substitute $z$ with $x_{1:T}$.
*   **Slide 3.3: The Intractable Posterior.** *Math:* The KL terms compare $p_\theta(x_{t-1}|x_t)$ to $q(x_{t-1}|x_t)$. *Ask:* Wait, didn't we say $q(x_{t-1}|x_t)$ is intractable? *Answer:* Yes. We are stuck.
*   **Slide 3.4: The $x_0$ Conditioning Trick.** *Visual:* A noisy image with question marks. Click: The clean $x_0$ appears next to it with a solid arrow. (Adapts Outlier 18:23). *Math:* By Bayes' rule, $q(x_{t-1}|x_t, x_0)$ is tractable!
*   **Slide 3.5: Completing the Square.** *Math:* Derive the true posterior mean $\tilde{\mu}_t(x_t, x_0)$ step-by-step. *Concrete:* Plug in $x_2=0.626, x_0=0.8, \beta_2=0.2, \alpha_2=0.8, \bar{\alpha}_2=0.72$. Result: $\tilde{\mu}_2 = 0.73$.

### Part 4: The Objective & The Network
*   **Slide 4.1: KL of Two Gaussians.** *Math:* We are matching $p_\theta(\mu_\theta, \sigma^2I)$ to $q(\tilde{\mu}_t, \sigma^2I)$. *Recall Lecture 1:* KL of Gaussians with same variance is just MSE of means: $||\tilde{\mu}_t - \mu_\theta||^2$.
*   **Slide 4.2: Predicting Noise.** *Ask:* The network can predict $\mu$, $x_0$, or $\epsilon$. Which is best? *Math:* Substitute $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon) / \sqrt{\bar{\alpha}_t}$ into the $\tilde{\mu}_t$ equation. Watch the algebra collapse beautifully to: $||\epsilon - \epsilon_\theta(x_t, t)||^2$.
*   **Slide 4.3: The U-Net.** *Visual:* U-Net diagram with sinusoidal $t$ embeddings injected at residual blocks. *Ask:* Why must $t$ be an input? *Answer:* The weights are shared; it needs to know the noise scale to scale its prediction.
*   **Slide 4.4: The Ideal Denoiser Paradox.** *Visual:* Generated samples that are pixel-perfect copies of the training set. (Adapts MIT L4). *Ask:* If loss=0, do we get new images? *Answer:* No, the global minimum is a lookup table. *Transition:* The U-Net's inductive bias (smoothness) is what creates *novelty*.
*   **Slide 4.5: PyTorch Training Step.** *Code:* Minimal PyTorch for `q_sample` and MSE loss. *Visual:* Widget 4 (Training loop playing live on 2D Swiss Roll).

### Part 5: Sampling & The Score Perspective
*   **Slide 5.1: The Reverse Step.** *Math:* Rearrange the $\epsilon$ prediction to get $x_{t-1}$. *Concrete:* $x_2 = 0.626$, network predicts $\epsilon = -0.1$. Calculate $x_1$ mean. Add $\sigma_t z$.
*   **Slide 5.2: The Sampling Algorithm.** *Code:* PyTorch `p_sample_loop`. *Visual:* Widget 5 (MNIST '3' emerging from noise).
*   **Slide 5.3: What is the Network Actually Learning?** *Visual:* Widget 6 (2D Vector Field pointing to data manifold). (Adapts MIT L1). *Math:* Tweedie's formula / Score matching. $\nabla_x \log p(x_t) = -\epsilon_\theta / \sqrt{1-\bar{\alpha}_t}$. *Correction:* Explicitly fix Paras's error here (he wrote $\alpha$ instead of $\bar{\alpha}_t$).

### Part 6: Schedules & Speedups
*   **Slide 6.1: The Information Cliff.** *Visual:* Widget 7 (Linear vs Cosine schedule on MNIST '3'). (Adapts Outlier 06:26). *Ask:* Why does Linear fail? *Answer:* It destroys all signal by $t=500$; the last 500 steps are wasted on pure noise. Cosine preserves signal longer.
*   **Slide 6.2: Speeding up (DDIM).** *Ask:* Why do we add $\sigma_t z$ during sampling? *Answer:* To match the marginals of the forward Markov chain. *Build:* What if we set $\sigma_t = 0$? The process becomes deterministic. We can skip steps! *Math:* The DDIM update equation.

### Part 7: Text Conditioning & Classifier-Free Guidance
*   **Slide 7.1: Conditioning.** *Math:* We want $\nabla_x \log p(x|y)$. *Visual:* Concatenating text $y$ to the U-Net.
*   **Slide 7.2: Classifier Guidance.** *Math:* Bayes' rule! $\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$. (Adapts MIT L3). *Visual:* A Gaussian (unconditional) shifted by a linear gradient (classifier).
*   **Slide 7.3: The Problem with Classifiers.** *Ask:* Can we use a standard ResNet? *Answer:* No, it must classify *noisy* images $x_t$.
*   **Slide 7.4: Classifier-Free Guidance.** *Math:* Rearrange Bayes: $\nabla_x \log p(y|x) = \nabla_x \log p(x|y) - \nabla_x \log p(x)$. Substitute back with a scalar $w$: $\epsilon_{guided} = \epsilon_{uncond} + w (\epsilon_{cond} - \epsilon_{uncond})$.
*   **Slide 7.5: Fidelity vs Diversity.** *Visual:* Widget 8 (CFG Slider on a 2D Gaussian mixture). As $w$ increases, density collapses to the sharpest modes.

### Part 8: Latent Diffusion & Wrap-Up
*   **Slide 8.1: The Pixel Bottleneck.** *Ask:* A 1024x1024x3 image has 3M dimensions. Can we run a U-Net 50 times on this? *Answer:* No.
*   **Slide 8.2: Closing the Loop (LDM).** *Visual:* Lecture 3's VAE Encoder $\to$ Lecture 5's Diffusion in Latent Space $\to$ Lecture 3's VAE Decoder.
*   **Slide 8.3: Summary.** Recap of learning outcomes.

---

## 4. Key Derivations (Step-by-Step on Slides)

1.  **Closed-form Jump ($x_t$ from $x_0$):**
    *   *Prior knowledge:* Sum of independent Gaussians $\mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$.
    *   *Step 1:* $x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1$
    *   *Step 2:* $x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\epsilon_2$
    *   *Step 3:* Substitute $x_1$ into $x_2$: $x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1) + \sqrt{1-\alpha_2}\epsilon_2$
    *   *Step 4:* Group $x_0$ and $\epsilon$ terms. Show variances add: $\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2$.
    *   *Step 5:* Generalize to $\bar{\alpha}_t$.
2.  **The Tractable Posterior $q(x_{t-1}|x_t, x_0)$:**
    *   *Prior knowledge:* Bayes' rule $P(A|B,C) = P(B|A,C)P(A|C) / P(B|C)$, Markov property.
    *   *Step 1:* $q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0) q(x_{t-1}|x_0) / q(x_t|x_0)$
    *   *Step 2:* Apply Markov: $q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1})$.
    *   *Step 3:* Expand into exponentials: $\exp(-\frac{1}{2} [ \frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}_{t-1}} - \dots ])$
    *   *Step 4:* Complete the square for $x_{t-1}$.
    *   *Step 5:* Extract $\tilde{\mu}_t$ and $\tilde{\beta}_t$.
3.  **Mean to Noise Substitution:**
    *   *Prior knowledge:* Basic algebra.
    *   *Step 1:* Start with $\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$.
    *   *Step 2:* Substitute $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$.
    *   *Step 3:* Factor out $x_t$. Watch the coefficient of $x_t$ beautifully collapse to $1/\sqrt{\alpha_t}$.
    *   *Step 4:* Factor out $\epsilon$. Reveal $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$.
4.  **Classifier-Free Guidance:**
    *   *Prior knowledge:* Bayes' rule.
    *   *Step 1:* Score of posterior: $\nabla \log p(x|y) = \nabla \log p(x) + \nabla \log p(y|x)$.
    *   *Step 2:* Isolate classifier: $\nabla \log p(y|x) = \nabla \log p(x|y) - \nabla \log p(x)$.
    *   *Step 3:* Define guided score: $s_G = \nabla \log p(x) + w \cdot \nabla \log p(y|x)$.
    *   *Step 4:* Substitute Step 2 into Step 3. Yields $s_G = (1-w)\nabla \log p(x) + w \nabla \log p(x|y)$.
    *   *Step 5:* Convert scores to $\epsilon$ predictions.

---

## 5. Live Interactive Figures / Widgets (Ranked by value)

1.  **The 100 Cats (1D Bimodal Posterior):** *Exact Formula.* Slider for $\beta_t$. Shows a bimodal $q(x_0)$ and the resulting posterior $q(x_{t-1}|x_t)$. Proves visually why large steps fail and small steps yield a Gaussian. (Addresses Ari Seff).
2.  **Exploding vs. Scaled Variance:** *Live browser compute (JS).* 100 particles in 1D. Slider for $t$. Top track: $x_t = x_{t-1} + \epsilon$ (particles fly off screen). Bottom track: $x_t = \sqrt{1-\beta}x_{t-1} + \sqrt{\beta}\epsilon$ (particles settle into standard normal bell curve).
3.  **Vector Field / Score Map:** *Recorded run.* 2D Swiss Roll. Shows the data manifold and the learned $\epsilon_\theta$ vector field pointing back to it. Hovering shows the exact vector at that point. (Addresses MIT L1/L4).
4.  **The Information Cliff (Schedules):** *Exact Formula.* Image of MNIST '3'. Radio buttons for Linear vs Cosine. Slider for $t$. Shows the image degrading, alongside a line chart of the Signal-to-Noise Ratio (SNR).
5.  **Reverse the Film (Corrected):** *Recorded run.* Replaces Paras's cheating widget. Slider for $t$ from $T \to 0$. Shows $x_t$, the network's prediction of the noise $\epsilon_\theta$, and the resulting $x_0$ estimate $\hat{x}_0$. Shows how $\hat{x}_0$ is blurry at high $t$ and sharpens at low $t$.
6.  **CFG Fidelity vs Diversity:** *Exact Formula.* 2D plot of 3 Gaussian blobs. Slider for guidance scale $w$ (1 to 10). As $w$ increases, the density visually shrinks tightly around the peaks, leaving the space between them completely empty.
7.  **VAE to Diffusion ELBO Bridge:** *Conceptual Animation.* A visual mapping that slides the nodes of a VAE graph to perfectly align with the unrolled Markov chain of diffusion.

---

## 6. PyTorch Code Slides (Minimal & Annotated)

**1. Forward Noising (`q_sample`)**
```python
def q_sample(x0, t, noise):
    # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    sqrt_alpha_bar = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_bar = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
```

**2. Training Step (`p_losses`)**
```python
def p_losses(denoise_model, x0, t):
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    predicted_noise = denoise_model(x_t, t) # U-Net
    return F.mse_loss(noise, predicted_noise) # L_simple
```

**3. DDPM Reverse Step (`p_sample`)**
```python
def p_sample(model, x_t, t, t_index):
    betas_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)
    
    # Predict noise and calculate mean
    predicted_noise = model(x_t, t)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean # Last step: no noise
    else:
        posterior_variance_t = extract(posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
```

---

## 7. The Companion Notebook

Runs entirely on CPU in Colab in under 5 minutes, except for training the MNIST U-Net which loads a checkpoint.
*   **Section 1: The Forward Process.** Implements `q_sample`. Plots a grid of a single MNIST digit degrading over $T=1000$ steps.
*   **Section 2: The 2D Toy Model.** Trains a tiny 3-layer MLP on the 2D Swiss Roll (takes 60 seconds). Plots the learned vector field (score function).
*   **Section 3: The U-Net.** Defines the PyTorch U-Net with Sinusoidal Positional Embeddings.
*   **Section 4: MNIST Training.** Loads a pre-trained checkpoint of the U-Net (1-2M params) to save time. Runs *one* live training step to prove the code works and inspect the loss.
*   **Section 5: Sampling.** Implements the `p_sample_loop`. Generates a batch of 16 MNIST digits live. Plots the $\hat{x}_0$ estimates over time to show coarse-to-fine generation.
*   **Section 6: CFG.** Modifies the sampling loop to accept class labels (0-9) and a guidance scale $w$. Generates guided digits.

---

## 8. Models and Experiments to Train (Pre-lecture Prep)

1.  **2D Toy MLP:** Train on 8-Gaussians and Swiss Roll. Record the vector field at $t=10, 50, 100$ to show how the score function smooths out at higher noise levels.
2.  **MNIST DDPM:** Train a standard U-Net on MNIST for 20 epochs. Record checkpoints.
3.  **The Memorization Experiment (Ideal Denoiser):** Train a massive U-Net on a tiny subset of MNIST (e.g., 10 images) with zero weight decay until loss is near zero. Generate samples to show it perfectly regurgitates the 10 images. (Proves MIT L4's point about inductive bias).
4.  **Schedule Comparison:** Train two identical U-Nets, one with Linear and one with Cosine schedule. Measure the FID or visually compare the diversity of samples.

---

## 9. Gemini Image Generation Prompts

*Where it helps (Conceptual Metaphors):*
1.  **Prompt:** *"A hyper-realistic, high-speed macro photograph of an egg being scrambled in a cast iron pan. The yolk is halfway broken, mixing with the egg white in complex fluid swirls. Cinematic lighting."* (Used for Slide 1.1).
2.  **Prompt:** *"A conceptual illustration of 100 different cats sitting in a dense fog. We can only see their vague silhouettes. In the center, a perfectly clear, sharp golden retriever."* (Used for the "100 Cats" 1-step denoising failure visual).

*Where it does NOT help:*
Do not use Gemini for any math, graphs, vector fields, or MNIST digits. Those must be generated via exact Matplotlib/PyTorch scripts to maintain trust and accuracy.

---

## 10. References Slide

*   **Ho et al. (2020).** *Denoising Diffusion Probabilistic Models.* (Supports the core formulation, $\epsilon$-prediction, and L_simple).
*   **Sohl-Dickstein et al. (2015).** *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* (Supports the original Markov chain concept).
*   **Song & Ermon (2019).** *Generative Modeling by Estimating Gradients of the Data Distribution.* (Supports the vector field / score matching perspective).
*   **Nichol & Dhariwal (2021).** *Improved Denoising Diffusion Probabilistic Models.* (Supports the Cosine schedule and variance learning).
*   **Song, Meng, Ermon (2021).** *Denoising Diffusion Implicit Models (DDIM).* (Supports the deterministic sampling speedup).
*   **Ho & Salimans (2022).** *Classifier-Free Diffusion Guidance.* (Supports the CFG derivation).
*   **Rombach et al. (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models.* (Supports the final Latent Diffusion slide).
*   **Educational Sources:** Lilian Weng's blog, Calvin Luo's tutorial, Ari Seff (YouTube), MIT 6.S183 (IAP 2025).

---

## 11. The 10 "Wow" Moments & How to Stage Them

1.  **The Exploding Variance:** Let students predict what happens if we just add noise. Click to reveal the particles flying off the screen to infinity. Then click to add the $\sqrt{1-\beta}$ scale and watch them magically settle into a bell curve.
2.  **The 1000-Step Shortcut:** Show the nested equation $x_{1000} = \dots$ expanding off the slide. Click to collapse it using the sum of Gaussians trick into a single, elegant 1-line formula.
3.  **The "100 Cats" Posterior:** Show a noisy point. Ask "Where did this come from?" Reveal the bimodal true posterior. Show that predicting the mean results in a ghost image (between the modes).
4.  **The $x_0$ Conditioning Trick:** The math for $q(x_{t-1}|x_t)$ is stuck. Click to add $x_0$ to the given side of the probability. Watch the intractable mess instantly become a simple, solvable Bayes' rule application.
5.  **The KL $\to$ MSE Collapse:** Show the terrifying KL divergence formula between two Gaussians. Click through the algebraic substitutions until all the nasty variance terms cancel out, leaving just $||\epsilon - \epsilon_\theta||^2$.
6.  **The Ideal Denoiser Paradox:** Ask if a loss of 0 is good. Students will say yes. Reveal the generated samples—they are exact, pixel-perfect copies of the training set. Explain that *failing* to reach the global minimum is what makes neural networks creative.
7.  **The Vector Field:** Show the 2D Swiss Roll. Turn on the vector field of the trained network. Students will literally see the gradients pointing from empty space back to the data manifold.
8.  **The Information Cliff:** Show the Linear schedule destroying the image by step 500. Ask what the network is doing for the remaining 500 steps. (Nothing). Reveal the Cosine schedule preserving the ghost of the image all the way to step 990.
9.  **The CFG Derivation:** Show the Classifier Guidance equation. Ask how to get rid of the classifier. Use Bayes' rule to subtract the unconditional score from the conditional score. The classifier mathematically vanishes from the equation!
10. **The Latent Diffusion Reveal:** Ask how to scale to 1024x1024. Bring back the VAE diagram from Lecture 3. Insert the Diffusion block into the bottleneck. The entire 5-lecture course snaps together into one unified architecture.

---

## 12. Misconceptions to Correct Explicitly

1.  **Slide 1.3:** *Misconception:* The forward process converges to a Gaussian because of the Central Limit Theorem (Paras's error). *Correction:* It converges because the schedule $\bar{\alpha}_T$ is designed to approach 0, leaving only the $\mathcal{N}(0, I)$ noise term.
2.  **Slide 4.2:** *Misconception:* $\epsilon$ is the small noise added between step $t-1$ and $t$. *Correction:* $\epsilon$ is the *total* accumulated noise from $x_0$ to $x_t$. (From MIT L2).
3.  **Slide 4.3:** *Misconception:* The U-Net takes a noisy image and just outputs a clean one. *Correction:* The U-Net weights are shared across all 1000 steps. It *must* be fed the timestep $t$ so it knows how aggressively to denoise.
4.  **Slide 5.3:** *Misconception:* The score of the noisy marginal is $-\epsilon / \sqrt{1-\alpha_t}$ (Paras's error). *Correction:* It is $-\epsilon / \sqrt{1-\bar{\alpha}_t}$. The bar matters!
5.  **Slide 5.1:** *Misconception:* We can evaluate the "Reverse the film" widget by using the known $x_0$. *Correction:* That's cheating. At generation time, $x_0$ is unknown. We must use the network's *estimate* $\hat{x}_0$.

---

## 13. What to Leave Out (and Why)

*   **Continuous-Time SDEs & ODEs:** Moves the math from discrete probability (which they know) to Itô calculus (which they don't). Mention it as an appendix/extension, but do not derive it.
*   **Flow Matching & Rectified Flows:** Too cutting-edge for a foundational lecture. It breaks the Markov chain narrative.
*   **Detailed V-Prediction:** Mentioning that Stable Diffusion 3 uses velocity prediction is fine, but deriving the SNR weighting of v-prediction vs $\epsilon$-prediction will derail the pacing.
*   **Score Distillation Sampling (SDS):** MIT L5 covers this beautifully, but it requires introducing NeRFs and differentiable rendering. Keep the focus strictly on 2D image generation.
*   **Architecture deep-dives (DiT):** They know Transformers. Stating "you can replace the U-Net with a ViT" takes 10 seconds. Don't spend 15 minutes on patchification.