"""One linked bibliography for slides, presenter notes and the resource page."""
RESOURCES = {
 'lil-vae':dict(title='Lilian Weng: From Autoencoder to Beta-VAE',url='https://lilianweng.github.io/posts/2018-08-12-vae/',group='Explanations',description='Probabilistic VAE framing, KL direction, reparameterization and beta-VAE. Our diagrams are newly drawn.'),
 'lil-gan':dict(title='Lilian Weng: From GAN to WGAN',url='https://lilianweng.github.io/posts/2017-08-20-gan/',group='Explanations',description='Distributions, the optimal discriminator, Jensen–Shannon divergence and training difficulties.'),
 'jaan':dict(title='Jaan Altosaar: What is a variational autoencoder?',url='https://jaan.io/what-is-variational-autoencoder-vae-tutorial',group='Explanations',description='Generation and inference, with neural-network and probabilistic perspectives.'),
 'nipun-vi':dict(title='Nipun Batra: Variational Inference, Part 2',url='https://www.youtube.com/watch?v=fQ8D0-tTCBU',group='Course videos and notebook',description='Your lecture on variational inference, the evidence bound and sample/rsample. Earlier review used selected automatic-caption sections.'),
 'nipun-vae':dict(title='Nipun Batra: Variational Autoencoders',url='https://www.youtube.com/watch?v=bo8wceOS6C0',group='Course videos and notebook',description='Your VAE lecture, including reconstruction, prior sampling and latent geometry. Earlier review used selected automatic-caption sections.'),
 'nipun-notebook':dict(title='Nipun Batra: PML VAE notebook',url='https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb',group='Course videos and notebook',description='Course experiment reference. This lecture states its own loss reductions, data protocol and checkpoint provenance.'),
 'mit':dict(title='MIT 6.S191: Deep Generative Modeling',url='https://www.youtube.com/watch?v=R8V8CbuxryI',group='Course videos and notebook',description='Application-led motivation and audience prediction before architecture.'),
 'vae-paper':dict(title='Kingma & Welling: Auto-Encoding Variational Bayes',url='https://arxiv.org/abs/1312.6114',group='Primary papers',description='ELBO identity, the computable reconstruction/prior-KL form, and reparameterized stochastic gradients.'),
 'gan-paper':dict(title='Goodfellow et al.: Generative Adversarial Networks',url='https://arxiv.org/abs/1406.2661',group='Primary papers',description='Figure 1, Section 3, Algorithm 1 and Section 4.1: induced model distributions, alternating updates, non-saturating loss, Bayes-optimal D and the JS connection.'),
 'flow-paper':dict(title='Rezende & Mohamed: Normalizing Flows',url='https://arxiv.org/abs/1505.05770',group='Primary papers',description='Invertible transformations and the density/Jacobian calculation.'),
 'gan-training-paper':dict(title='Arjovsky & Bottou: Towards Principled Methods for Training GANs',url='https://arxiv.org/abs/1701.04862',group='Further reading',description='Further reading on supports and generator gradients; no guarantee of stable training is inferred from the loss comparison.'),
 'wgan-paper':dict(title='Arjovsky, Chintala & Bottou: Wasserstein GAN',url='https://arxiv.org/abs/1701.07875',group='Further reading',description='Optional continuation from Lilian’s article; WGAN is outside this lecture’s main derivation.'),
 'tf-cvae':dict(title='TensorFlow: Convolutional Variational Autoencoder',url='https://www.tensorflow.org/tutorials/generative/cvae',group='Worked implementations',description='Latent-grid visual reference. The lecture’s live decoder and recordings use its own saved PyTorch weights.'),
 'torch-sampling':dict(title='PyTorch: Pathwise derivative and rsample',url='https://pytorch.org/docs/stable/distributions.html#pathwise-derivative',group='Worked implementations',description='Implementation reference for the checked sample/rsample distinction.'),
 'mnist':dict(title='MNIST dataset',url='https://yann.lecun.com/exdb/mnist/',group='Data and image attribution',description='Handwritten digits; threshold and downloaded-file hashes are recorded in metrics/mnist.json.'),
 'pets':dict(title='Oxford-IIIT Pet Dataset',url='https://www.robots.ox.ac.uk/~vgg/data/pets/',group='Data and image attribution',description='Recurring dog photograph, CC BY-SA 4.0; complete provenance and hashes are in SOURCES.md.'),
 'cvdf':dict(title='CVDF MNIST download mirror',url='https://storage.googleapis.com/cvdf-datasets/mnist/',group='Data and image attribution',description='Public download source used by the reproducible training script.'),
}
SLIDE_READINGS={}
def assign(ids, keys):
 for id in ids.split():SLIDE_READINGS[id]=keys
assign('start recall many-answers generation-baselines another-digit', ['mit','nipun-vae'])
assign('generation-inference evidence-sum approximate-inference three-objects', ['jaan','lil-vae','vae-paper'])
assign('kl-meaning kl-direction kl-modes elbo-gap two-kls elbo elbo-reweight elbo-jensen elbo-learning', ['lil-vae','vae-paper','nipun-vi'])
assign('opaque-sample independent-noise computation-graph sample-arithmetic gradient-arithmetic code code-implementation', ['vae-paper','lil-vae','nipun-vi'])
assign('autograd-sampling', ['torch-sampling','nipun-vi'])
assign('ae-experiment ae-sampling measured-models measured-generation latent-plane paths-and-curves beta-experiment', ['nipun-vae','nipun-notebook','tf-cvae','lil-vae'])
assign('another-signal discriminator gan-source gan-logloss gan-bayes gan-density train-discriminator train-generator alternate gan-numbers gan-js', ['gan-paper','lil-gan'])
assign('gan-gradient gan-training coverage', ['gan-paper','lil-gan','gan-training-paper'])
assign('invertible change-of-variables', ['flow-paper'])
assign('forward-noise next', ['pets'])
