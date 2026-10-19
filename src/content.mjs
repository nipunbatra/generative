// Human-edited classroom copy. One entry per reveal; equations retain their definitions.
export const content={
  "1": [
    [
      "Where could these portraits have come from?",
      ""
    ],
    [
      "These are published generated faces. Change the random input and the face changes.",
      ""
    ]
  ],
  "2": [
    [
      "How could a model produce another digit after training?",
      ""
    ],
    [
      "A decoded grid shows how the hidden coordinates change the digit.",
      ""
    ]
  ],
  "3": [
    [
      "Which properties of these two sources could be combined?",
      ""
    ],
    [
      "Take coarse styles from B and the remaining styles from A: this is style mixing.",
      ""
    ]
  ],
  "4": [
    [
      "An outline specifies shape. What does it leave unspecified?",
      ""
    ],
    [
      "The model fills in the appearance. The photograph is a separate reference.",
      ""
    ]
  ],
  "5": [
    [
      "The layout fixes where the regions belong.",
      ""
    ],
    [
      "The layout constrains the scene; the generator supplies its appearance.",
      ""
    ]
  ],
  "6": [
    [
      "What should stay recognisable when the domain changes?",
      ""
    ],
    [
      "The horse is an observed input. The zebra-domain image is a generated translation.",
      ""
    ]
  ],
  "7": [
    [
      "The visible context constrains the missing region. Does it determine every detail?",
      ""
    ],
    [
      "A plausible completion is not proof of the hidden original.",
      ""
    ]
  ],
  "8": [
    [
      "Proposed lab application; illustrative signals. Condition: a spindle-like event.",
      ""
    ],
    [
      "Different traces can fit one event description. Scientific validity still needs testing.",
      ""
    ]
  ],
  "9": [
    [
      "What does the metadata tell a satellite-image generator?",
      ""
    ],
    [
      "These are generated possibilities, not recovered measurements or forecasts.",
      ""
    ]
  ],
  "10": [
    [
      "Choose a next character from its categorical distribution.",
      ""
    ],
    [
      "A new random seed gives another sequence from the same fitted model.",
      ""
    ]
  ],
  "11": [
    [
      "A sample is one outcome. The distribution describes possible outcomes.",
      "x"
    ],
    [
      "Learn a distribution from which we can draw new outputs.",
      "x\\sim p_\\theta(x)"
    ],
    [
      "A condition narrows which outputs are plausible.",
      "x\\sim p_\\theta(x\\mid c)"
    ]
  ],
  "12": [
    [
      "The average falls between the two clusters.",
      "\\mathbb E[X]\\neq X"
    ],
    [
      "Good samples should look plausible and cover the data distribution.",
      ""
    ]
  ],
  "13": [
    [
      "Fit the centre to six observations.",
      "\\hat\\mu=\\frac1N\\sum_i x_i=0"
    ],
    [
      "For maximum likelihood, divide the squared deviations by N.",
      "\\hat\\sigma^2=\\frac1N\\sum_i(x_i-\\hat\\mu)^2"
    ],
    [
      "Even the best fit within one family can miss the shape of the data.",
      "x=\\hat\\mu+\\hat\\sigma\\epsilon"
    ]
  ],
  "14": [
    [
      "First choose a component.",
      "z\\sim\\operatorname{Categorical}(1/2,1/2)"
    ],
    [
      "Then draw from its Gaussian. Sum over the hidden choice.",
      "p(x)=\\sum_z p(z)p(x\\mid z)"
    ]
  ],
  "15": [
    [
      "Every token has a probability given its preceding context.",
      "p_\\theta(x_t\\mid x_{<t})"
    ],
    [
      "Multiply the next-token probabilities, including the probability of stopping.",
      "p_\\theta(x)=\\prod_t p_\\theta(x_t\\mid x_{<t})"
    ]
  ],
  "16": [
    [
      "An image can be ordered into a pixel sequence.",
      "x_1,x_2,\\ldots,x_{16}"
    ],
    [
      "Training uses the observed prefix. A causal mask lets us score positions in parallel.",
      "-\\log p_\\theta(x)=-\\sum_i\\log p_\\theta(x_i\\mid x_{<i})"
    ],
    [
      "Generation feeds each sampled pixel into the next prediction.",
      "x_i\\sim p_\\theta(x_i\\mid x_{<i})"
    ]
  ],
  "17": [
    [
      "Could a random vector select an entire output?",
      "z\\sim p(z)"
    ],
    [
      "Learn a map that turns latent draws into plausible outputs.",
      "x=G_\\theta(z)"
    ]
  ],
  "18": [
    [
      "An autoencoder learns to reconstruct its inputs.",
      "x\\to E_\\phi(x)\\to G_\\theta(E_\\phi(x))"
    ],
    [
      "Reconstruction alone can leave irregularly arranged codes.",
      ""
    ],
    [
      "Where should we draw a new code? Reconstruction alone has not answered that.",
      "z\\sim\\;?"
    ]
  ],
  "19": [
    [
      "Two distributions assign different probabilities to the same categories.",
      "P=(0.8,0.2),\\quad Q=(0.6,0.4)"
    ],
    [
      "An unlikely observation is more surprising.",
      "\\operatorname{surprisal}(i)=-\\log Q_i"
    ],
    [
      "Average the extra surprise, weighting each event by its probability under P.",
      "D_{\\mathrm{KL}}(P\\|Q)=\\sum_iP_i\\log\\frac{P_i}{Q_i}"
    ]
  ],
  "20": [
    [
      "Category 1: weight the log ratio by its probability under P.",
      "P_1\\log(P_1/Q_1)"
    ],
    [
      "Category 2 can make a negative contribution.",
      "P_2\\log(P_2/Q_2)"
    ],
    [
      "Individual contributions can be negative. Their sum cannot.",
      "D_{\\mathrm{KL}}(P\\|Q)\\geq0"
    ]
  ],
  "21": [
    [
      "Swap P and Q: the weights change as well as the log ratios.",
      "D_{\\mathrm{KL}}(Q\\|P)\\neq D_{\\mathrm{KL}}(P\\|Q)"
    ],
    [
      "Matching distributions incur no extra surprisal.",
      "D_{\\mathrm{KL}}(P\\|P)=0"
    ],
    [
      "A possible event given zero model probability has infinite cost.",
      "P_i>0,\\;Q_i=0\\;\\Longrightarrow\\;D_{\\mathrm{KL}}(P\\|Q)=\\infty"
    ]
  ],
  "22": [
    [
      "Entropy is the uncertainty in P. Cross-entropy uses Q to describe it.",
      "H(P)=-\\sum_iP_i\\log P_i"
    ],
    [
      "With P fixed, minimising cross-entropy also minimises KL.",
      "H(P,Q)=H(P)+D_{\\mathrm{KL}}(P\\|Q)"
    ]
  ],
  "23": [
    [
      "For a continuous latent, sum becomes integral.",
      "D_{\\mathrm{KL}}(q\\|p)=\\int q(z)\\log\\frac{q(z)}{p(z)}\\,dz"
    ],
    [
      "The encoder gives a distribution for this x. The prior does not depend on x.",
      "D_{\\mathrm{KL}}(q_\\phi(z\\mid x)\\|p(z))"
    ]
  ],
  "24": [
    [
      "Begin with a prior. No observed image or encoder is needed.",
      "z\\sim\\mathcal N(0,I)"
    ],
    [
      "Decode probabilities first, then sample the observation.",
      "p_\\theta(x,z)=p(z)p_\\theta(x\\mid z)"
    ]
  ],
  "25": [
    [
      "Every latent that could explain this image contributes to its probability.",
      "p_\\theta(x)=\\int p(z)p_\\theta(x\\mid z)\\,dz"
    ],
    [
      "With a nonlinear decoder, this integral is usually hard to compute exactly.",
      "\\max_\\theta\\sum_n\\log p_\\theta(x_n)"
    ]
  ],
  "26": [
    [
      "After observing x, which latent values explain it?",
      "p_\\theta(z\\mid x)=\\frac{p_\\theta(x,z)}{p_\\theta(x)}"
    ],
    [
      "Use a tractable encoder distribution to approximate the true posterior.",
      "q_\\phi(z\\mid x)\\approx p_\\theta(z\\mid x)"
    ]
  ],
  "27": [
    [
      "Start with the marginal probability of the observation.",
      "\\log p_\\theta(x)=\\log\\int p_\\theta(x,z)\\,dz"
    ],
    [
      "Multiply and divide by q. It must be normalised and cover the required support.",
      "=\\log\\int q_\\phi(z\\mid x)\\frac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\,dz"
    ],
    [
      "The integral is now an expectation under q.",
      "=\\log\\mathbb E_q\\!\\left[\\frac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\right]"
    ],
    [
      "Move log inside the expectation using Jensen. This is the first inequality.",
      "\\geq\\mathbb E_q\\!\\left[\\log\\frac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\right]"
    ]
  ],
  "28": [
    [
      "Factor the joint and separate the logarithms.",
      "\\mathbb E_q[\\log p_\\theta(x\\mid z)+\\log p(z)-\\log q_\\phi(z\\mid x)]"
    ],
    [
      "The bound is expected reconstruction log likelihood minus a KL to the prior.",
      "\\mathbb E_q[\\log p_\\theta(x\\mid z)]-D_{\\mathrm{KL}}(q_\\phi(z\\mid x)\\|p(z))"
    ],
    [
      "Maximise the evidence lower bound, or minimise its negative.",
      "\\log p_\\theta(x)\\geq\\mathcal E(\\theta,\\phi;x)"
    ]
  ],
  "29": [
    [
      "Expand the KL to the true posterior using Bayes’ rule.",
      "D_{\\mathrm{KL}}(q\\|p_\\theta(z\\mid x))=\\mathbb E_q[\\log q-\\log p_\\theta(x,z)]+\\log p_\\theta(x)"
    ],
    [
      "The gap is a posterior KL, not the KL to the prior.",
      "\\log p_\\theta(x)=\\mathcal E+D_{\\mathrm{KL}}(q\\|p_\\theta(z\\mid x))"
    ],
    [
      "The gap closes when q matches the true posterior almost everywhere.",
      "q=p_\\theta(z\\mid x)\\;\\Longrightarrow\\;\\mathcal E=\\log p_\\theta(x)"
    ]
  ],
  "30": [
    [
      "Two latent states let us compute evidence and posterior exactly.",
      "p(x=1)=\\sum_zp(z)p(x=1\\mid z)"
    ],
    [
      "Move q. The bound changes; the fixed model evidence does not.",
      "\\mathcal E+D_{\\mathrm{KL}}(q\\|p(z\\mid x))=\\log p(x)"
    ]
  ],
  "31": [
    [
      "Which distribution belongs at each point in the model?",
      "p(z),\\quad p_\\theta(z\\mid x),\\quad q_\\phi(z\\mid x),\\quad p_\\theta(x\\mid z)"
    ],
    [
      "The exact ELBO is a bound. One Monte Carlo estimate can sit above log evidence.",
      "\\widehat{\\mathcal E}\\neq\\mathcal E\\;\\text{in general}"
    ]
  ],
  "32": [
    [
      "The encoder controls the sampling distribution. We need its expected-loss gradient.",
      "\\nabla_\\phi\\mathbb E_{z\\sim q_\\phi}[f(z)]"
    ],
    [
      "An opaque random draw hides the path needed for ordinary backpropagation.",
      "\\mu,\\sigma\\to\\text{sampler}\\to z\\to f(z)"
    ]
  ],
  "33": [
    [
      "Draw noise from a fixed, parameter-independent distribution.",
      "\\epsilon\\sim\\mathcal N(0,1)"
    ],
    [
      "Stretch the same noise draw.",
      "\\sigma\\epsilon"
    ],
    [
      "Shift the scaled noise. The result has exactly the Gaussian distribution we wanted.",
      "z=\\mu+\\sigma\\epsilon\\sim\\mathcal N(\\mu,\\sigma^2)"
    ]
  ],
  "34": [
    [
      "Move the expectation to parameter-independent noise.",
      "\\mathbb E_q f(z)=\\mathbb E_\\epsilon f(\\mu+\\sigma\\epsilon)"
    ],
    [
      "For this draw, hold epsilon fixed and differentiate.",
      "\\frac{\\partial z}{\\partial\\mu}=1,\\qquad\\frac{\\partial z}{\\partial\\sigma}=\\epsilon"
    ],
    [
      "Apply the chain rule to each draw; average draws to estimate the expected gradient.",
      "\\frac{\\partial f}{\\partial\\mu}=f\\prime(z),\\quad\\frac{\\partial f}{\\partial\\sigma}=f\\prime(z)\\epsilon"
    ]
  ],
  "35": [
    [
      "A fixed draw gives a concrete sample and loss.",
      "\\mu=1,\\quad\\sigma=.5,\\quad\\epsilon=-.4,\\quad a=0"
    ],
    [
      "Use the same epsilon when checking the derivatives with finite differences.",
      "\\frac{\\partial f}{\\partial v}=f\\prime(z)\\frac{\\sigma\\epsilon}{2},\\quad v=\\log\\sigma^2"
    ]
  ],
  "36": [
    [
      "The encoder predicts log variance. Convert it to standard deviation before sampling.",
      "\\operatorname{std}=\\exp(\\tfrac12\\operatorname{logvar})"
    ],
    [
      "All five tensors have the same shape.",
      "z=\\mu+\\operatorname{std}\\odot\\epsilon"
    ]
  ],
  "37": [
    [
      "Flatten each image, then compute a hidden representation.",
      "[B,1,28,28]\\to[B,784]\\to[B,128]"
    ],
    [
      "Two heads give a mean and a log variance for each image.",
      "\\mu,\\operatorname{logvar},z:\\ [B,2]"
    ],
    [
      "The decoder returns logits. Sigmoid turns them into pixel probabilities.",
      "[B,2]\\to[B,128]\\to[B,784]"
    ]
  ],
  "38": [
    [
      "Once z is given, the model assigns each binary pixel a Bernoulli probability.",
      "p_\\theta(x\\mid z)=\\prod_i\\pi_i^{x_i}(1-\\pi_i)^{1-x_i}"
    ],
    [
      "Negative log likelihood is BCE summed over pixels.",
      "-\\log p_\\theta(x\\mid z)=-\\sum_i[x_i\\log\\pi_i+(1-x_i)\\log(1-\\pi_i)]"
    ],
    [
      "Fixed-variance Gaussian observations instead give squared error plus constants.",
      "-\\log p(x\\mid z)=\\frac{\\lVert x-m_\\theta(z)\\rVert^2}{2\\tau^2}+C"
    ]
  ],
  "39": [
    [
      "For one Gaussian coordinate, the KL is analytic.",
      "\\tfrac12(\\mu^2+\\sigma^2-1-\\log\\sigma^2)"
    ],
    [
      "Add the KL contributions from the independent latent coordinates.",
      "\\tfrac12\\sum_j(\\mu_j^2+\\sigma_j^2-1-\\log\\sigma_j^2)"
    ],
    [
      "At mean zero and variance one, each contribution is zero.",
      "\\tfrac12\\sum_j(\\mu_j^2+e^{v_j}-1-v_j)"
    ]
  ],
  "40": [
    [
      "A temporary two-pixel example: sample the latent first.",
      "x=(1,0),\\quad z=\\mu+\\sigma\\odot\\epsilon"
    ],
    [
      "At this z, the specified decoder probabilities are (0.75, 0.25).",
      "-\\log .75-\\log(1-.25)"
    ],
    [
      "Add the exact Gaussian KL to this draw's reconstruction cost.",
      "\\widehat{-\\mathcal E}=\\operatorname{NLL}(x\\mid z)+D_{\\mathrm{KL}}(q\\|p)"
    ]
  ],
  "41": [
    [
      "Encode, sample, decode. The graph joins both networks.",
      ""
    ],
    [
      "For each image, sum over pixels and over latent coordinates separately.",
      ""
    ],
    [
      "Average per-example totals, then update encoder and decoder.",
      "\\mathcal L=\\frac1B\\sum_b(\\operatorname{recon}_b+\\operatorname{KL}_b)"
    ]
  ],
  "42": [
    [
      "Reconstruction starts with an observed image and its encoded posterior.",
      "z\\sim q_\\phi(z\\mid x)"
    ],
    [
      "Fresh generation starts at the prior and skips the encoder.",
      "z\\sim p(z),\\quad x\\sim p_\\theta(x\\mid z)"
    ]
  ],
  "43": [
    [
      "Move through z while keeping the trained decoder fixed.",
      "z=(z_1,z_2)"
    ],
    [
      "A regular grid helps us inspect the decoder. It is not a batch of prior draws.",
      "z\\sim\\mathcal N(0,I)"
    ]
  ],
  "44": [
    [
      "The objective balances explaining x and keeping q near the prior.",
      "\\mathcal L=\\operatorname{reconstruction\\ NLL}+\\operatorname{KL}"
    ],
    [
      "The objective does not guarantee sharp samples or interpretable latent coordinates.",
      "\\beta=1:\\ \\text{vanilla negative ELBO}"
    ]
  ],
  "45": [
    [
      "A generator makes fake samples. Compare them with real data.",
      "x_{\\mathrm{fake}}=G_\\theta(z)"
    ],
    [
      "Train a classifier to recognise the source.",
      "D_\\psi(x)=\\operatorname{sigmoid}(a_\\psi(x))"
    ]
  ],
  "46": [
    [
      "The generator maps noise into the data space.",
      "z:[B,32]\\to G_\\theta\\to[B,784]"
    ],
    [
      "Real examples enter a discriminator that outputs logits.",
      "x:[B,784]\\to D_\\psi\\to[B,1]"
    ],
    [
      "Both real and generated images go through the same discriminator.",
      "G_\\theta(z)\\to D_\\psi\\to[B,1]"
    ]
  ],
  "47": [
    [
      "Real examples have target 1.",
      "-\\mathbb E_{\\mathrm{real}}\\log D(x)"
    ],
    [
      "Generated examples have target 0.",
      "-\\mathbb E_z\\log(1-D(G(z)))"
    ],
    [
      "The discriminator loss adds the two batch means.",
      "\\mathcal L_D=-\\mathbb E\\log D(x)-\\mathbb E\\log(1-D(G(z)))"
    ]
  ],
  "48": [
    [
      "Clear gradients and generate a fake batch.",
      ""
    ],
    [
      "Detach the fake batch: keep its values, cut its connection to G.",
      ""
    ],
    [
      "Step D's optimiser. G receives no new gradient in this phase.",
      "\\psi\\leftarrow\\psi-\\eta\\nabla_\\psi\\mathcal L_D"
    ]
  ],
  "49": [
    [
      "Use a fresh forward pass through the updated discriminator.",
      ""
    ],
    [
      "Freeze D's parameters while keeping the derivative with respect to its input.",
      "\\mathcal L_G=-\\mathbb E_z\\log D(G_\\theta(z))"
    ],
    [
      "Use target 1 to update G. D's parameter values stay fixed.",
      "\\theta\\leftarrow\\theta-\\eta\\nabla_\\theta\\mathcal L_G"
    ]
  ],
  "50": [
    [
      "The original game asks G to minimise what D maximises.",
      "\\min_G\\max_D\\;\\mathbb E\\log D(x)+\\mathbb E\\log(1-D(G(z)))"
    ],
    [
      "Our code uses the non-saturating generator loss, which gives different gradients.",
      "\\mathcal L_G^{\\mathrm{NS}}=-\\mathbb E\\log D(G(z))"
    ]
  ],
  "51": [
    [
      "When D confidently rejects a fake, the minimax logit gradient becomes small.",
      "\\frac{d\\log(1-\\operatorname{sigmoid}(a))}{da}=-D"
    ],
    [
      "At small D, the non-saturating loss still has a strong logit gradient.",
      "\\frac{d[-\\log\\operatorname{sigmoid}(a)]}{da}=D-1"
    ],
    [
      "At D = 0.01, compare −0.01 with −0.99. Continue the chain rule through G.",
      "\\frac{d\\mathcal L}{d\\theta}=\\frac{d\\mathcal L}{da}\\frac{da}{dx_{\\mathrm{fake}}}\\frac{dx_{\\mathrm{fake}}}{d\\theta}"
    ]
  ],
  "52": [
    [
      "D phase: detached fake, targets 1 and 0, update D.",
      ""
    ],
    [
      "G phase: fresh graph, D frozen, target 1, update G.",
      ""
    ],
    [
      "After G's update, enable D's parameter gradients for the next iteration.",
      ""
    ]
  ],
  "53": [
    [
      "Noise dimension, batch size and pixel dimension have different roles.",
      "[64,32]\\to[64,128]\\to[64,256]\\to[64,784]"
    ],
    [
      "Reshape for viewing. D still receives flattened pixels.",
      "[64,784]\\to[64,256]\\to[64,128]\\to[64,1]"
    ],
    [
      "Targets must match logits exactly.",
      "\\operatorname{shape}(y)=\\operatorname{shape}(a)=[64,1]"
    ]
  ],
  "54": [
    [
      "Training needs real examples and the discriminator.",
      ""
    ],
    [
      "To sample after training, we only need noise and G.",
      "z\\sim p(z),\\quad x=G_\\theta(z)"
    ]
  ],
  "55": [
    [
      "Plausible samples can all come from one mode.",
      ""
    ],
    [
      "Check whether samples are plausible and whether all modes are represented.",
      ""
    ]
  ],
  "56": [
    [
      "Keep the noise fixed while replaying the recorded training checkpoints.",
      ""
    ],
    [
      "Read samples, coverage and losses together. D ≈ 1/2 is not proof of success.",
      ""
    ]
  ],
  "57": [
    [
      "Give the requested condition to the generator.",
      "x_{\\mathrm{fake}}=G(z,c)"
    ],
    [
      "D needs the condition too, so it can judge whether the output fits the request.",
      "D(x,c)"
    ]
  ],
  "58": [
    [
      "Find the detach boundary in D’s update and the open path in G’s update.",
      ""
    ],
    [
      "In the lab, compare a working GAN, a missing mode and a cut generator gradient.",
      ""
    ]
  ],
  "59": [
    [
      "Compare the learned distributions and the work needed to sample.",
      ""
    ],
    [
      "The sampling path and access to likelihood depend on the model we choose.",
      ""
    ]
  ],
  "60": [
    [
      "Explain prior sampling, the two KLs, and the gradient through frozen D.",
      ""
    ],
    [
      "Next: can deliberately corrupting data give us targets for learning to generate?",
      ""
    ]
  ],
  "61": [
    [
      "Write the Gaussian log-density ratio before taking its expectation.",
      "\\log\\frac{q(z)}{p(z)}=-\\log\\sigma-\\frac{(z-\\mu)^2}{2\\sigma^2}+\\frac{z^2}{2}"
    ],
    [
      "Use E[(z−μ)²] = σ² and E[z²] = μ² + σ².",
      "D_{\\mathrm{KL}}(q\\|p)=\\tfrac12(\\mu^2+\\sigma^2-1-\\log\\sigma^2)"
    ]
  ],
  "62": [
    [
      "Bayes’ rule turns posterior KL into a difference of objectives.",
      "\\mathbb E_q\\log\\frac{q(z\\mid x)}{p(z\\mid x)}=\\mathbb E_q[\\log q-\\log p(x,z)]+\\log p(x)"
    ],
    [
      "Rearrange, retaining the conditioning on x.",
      "\\log p_\\theta(x)=\\mathcal E+D_{\\mathrm{KL}}(q_\\phi(z\\mid x)\\|p_\\theta(z\\mid x))"
    ]
  ],
  "63": [
    [
      "Assume densities under a common measure and optimise D pointwise where their sum is positive.",
      "D^*(x)=\\frac{p_{\\mathrm{data}}(x)}{p_{\\mathrm{data}}(x)+p_g(x)}"
    ],
    [
      "For the minimax value at D*, distribution matching minimises JS. This is idealised analysis.",
      "V(D^*,G)=-\\log4+2\\operatorname{JS}(p_{\\mathrm{data}}\\|p_g)"
    ]
  ]
};
export const titles=['','Where did these faces come from?','How can a hidden space make digits?','Which properties can we mix?','What does a shoe outline leave out?','Can a layout specify a landscape?','What changes from horse to zebra?','What belongs in the missing region?','Could we generate a signal segment?','Which satellite images are possible?','Have we generated something already?','What did the gallery have in common?','Can a good average be a bad sample?','What does fitting a Gaussian learn?','What if a hidden choice selects a cluster?','What distribution did our name model learn?','Could we generate pixels in sequence?','Could one random vector make an image?','Does reconstruction tell us how to sample?','How do two distributions disagree?','Can we compute KL by hand?','Why do direction and support matter?','Where is KL hiding in cross-entropy?','What changes for a continuous latent?','What is the generative story?','What does maximum likelihood require?','Which latent could explain this image?','How does log become a lower bound?','What are the two terms in the bound?','What is the gap to log evidence?','Can we check the bound exactly?','Which distribution is which?','Where is the missing gradient path?','Can we move the randomness?','Which derivatives follow the sample?','Can we check one gradient by hand?','What should the encoder output?','How do the VAE tensors fit together?','Why is reconstruction a likelihood?','Can we compute Gaussian KL exactly?','What is a complete two-pixel VAE loss?','What happens in one VAE update?','What is needed to generate a new image?','What does moving in latent space do?','What does the VAE objective guarantee?','Can we learn the training signal?','How do the two networks connect?','Which labels train the discriminator?','Where should gradients stop for D?','How can frozen D train G?','Which generator objective do we use?','Why use the non-saturating objective?','What changes in one alternating iteration?','Where does each GAN tensor go?','Which network remains at generation?','Can realistic samples miss a mode?','What should we monitor during training?','How can both players use a condition?','Can you find the broken gradient path?','How do the three recipes compare?','What can you explain now?','Appendix · Derive the Gaussian KL','Appendix · Expand the posterior gap','Appendix · The ideal discriminator'];
export const sections={A:['Applications',12,8],B:['Distributions',14,10],C:['KL',10,8],D:['ELBO',18,14],E:['Gradients',10,8],F:['VAE',18,13],G:['GAN',32,25],H:['Comparison',6,4],X:['Appendix',0,0]};
export const sources={vae:['Kingma & Welling · AEVB','https://arxiv.org/abs/1312.6114'],gan:['Goodfellow et al. · GANs','https://arxiv.org/abs/1406.2661'],stylegan:['Karras, Laine & Aila · StyleGAN · CC BY-NC 4.0','https://github.com/NVlabs/stylegan'],shoe:['Isola et al. · pix2pix · row 100_AB','https://phillipi.github.io/pix2pix/images/sketch2photo_shoes/latest_net_G_val/index.html'],spade:['Park et al. · SPADE','https://nvlabs.github.io/SPADE/'],cycle:['Zhu et al. · CycleGAN','https://junyanz.github.io/CycleGAN/'],context:['Pathak et al. · Context Encoders · model result','https://www.cs.cmu.edu/~dpathak/context_encoder/'],satellite:['Khanna et al. · DiffusionSat','https://www.samarkhanna.com/DiffusionSat/'],name:['Course name generator · reproduced categorical example','https://nipunbatra.github.io/attention/part1.html'],torch:['PyTorch · autograd mechanics','https://docs.pytorch.org/docs/2.13/notes/autograd.html']};
export function sourceFor(n){if(n===1||n===3)return sources.stylegan;if(n===2)return sources.vae;if(n===4)return sources.shoe;if(n===5)return sources.spade;if(n===6)return sources.cycle;if(n===7)return sources.context;if(n===8)return ['Authored deterministic signal illustration','notebooks/LAB.md'];if(n===9)return sources.satellite;if(n===10||n===15||n===16)return sources.name;if(n>=45&&n<=58||n===63)return sources.gan;if(n>=23&&n<=44||n===61||n===62)return sources.vae;return ['ES 667 · original teaching diagrams and exact calculations','src/math.mjs'];}
