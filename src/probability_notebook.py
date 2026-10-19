"""Small executable exercises for the probabilistic detours."""
def extra_cells(md,code):
 return [md('''## ELBO: change the averaging distribution, then apply Jensen

Use the exact three-state model to compare `log(E[w])` with `E[log(w)]`. The latter is the bound. Try the true posterior as q: every importance weight then equals the evidence.

[Kingma & Welling](https://arxiv.org/abs/1312.6114) · [Lilian Weng’s VAE explanation](https://lilianweng.github.io/posts/2018-08-12-vae/) · [Nipun’s variational-inference lecture](https://www.youtube.com/watch?v=fQ8D0-tTCBU)
'''),code('''toy = json.loads((ROOT/'results/pedagogy.json').read_text())['toy']
joint = np.array(toy['prior']) * toy['likelihood']
for row in toy['states']:
    q = np.array(row['q'])
    weights = joint/q
    log_average = np.log(q @ weights)
    average_log = q @ np.log(weights)
    print('weights:', weights, 'log E[w]:', log_average, 'E[log w]:', average_log)
    assert average_log <= log_average + 1e-12
    assert np.isclose(average_log, row['elbo'])
'''),md('''## GAN: Bayes-optimal source classifier and Jensen–Shannon divergence

The class prior is part of the experiment. Start with equal source priors; then change pi. A score of one half at just one outcome does not imply that the two distributions agree everywhere.

[Original GAN paper, Sections 3–4](https://arxiv.org/abs/1406.2661) · [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
'''),code('''p_data = np.array([.6,.3,.1])
p_g = np.array([.1,.3,.6])
for pi in [.1,.5,.9]:
    d = pi*p_data/(pi*p_data+(1-pi)*p_g)
    print('data-source prior:', pi, 'Bayes source probabilities:', d)
m = (p_data+p_g)/2
js = .5 * (np.sum(p_data*np.log(p_data/m)) + np.sum(p_g*np.log(p_g/m)))
d_star = p_data/(p_data+p_g)
value = p_data @ np.log(d_star) + p_g @ np.log(1-d_star)
print('JS:', js, 'minimax value at optimal D:', value)
assert np.isclose(value, -np.log(4)+2*js)
'''),md('''## Why the non-saturating generator loss?

Differentiate with respect to the discriminator logit. The full generator gradient also passes through the discriminator input and generator parameters. These derivatives alone do not guarantee stable training.
'''),code('''a = torch.tensor(-4., requires_grad=True)
minimax = torch.nn.functional.logsigmoid(-a)
non_saturating = -torch.nn.functional.logsigmoid(a)
g_minimax = torch.autograd.grad(minimax, a, retain_graph=True)[0]
g_non_saturating = torch.autograd.grad(non_saturating, a)[0]
d = torch.sigmoid(a)
print('D:', d.item(), 'minimax derivative:', g_minimax.item(), 'non-saturating derivative:', g_non_saturating.item())
assert torch.allclose(g_minimax, -d)
assert torch.allclose(g_non_saturating, d-1)
'''),md('''## Continue with the course resources

- [Jaan Altosaar’s probabilistic VAE tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial)
- [Nipun’s VAE lecture](https://www.youtube.com/watch?v=bo8wceOS6C0)
- [Nipun’s PML notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/vae.ipynb)
- [MIT 6.S191 generative-model lecture](https://www.youtube.com/watch?v=R8V8CbuxryI)
- [All lecture resources](../resources.html)
''')]
