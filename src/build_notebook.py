from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s):return {'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]}
cells=[md('''# Generative Modelling I — from randomness to data

ES667 · Nipun Batra · companion to the 80-minute lecture.

**OBSERVED DATA** is thresholded MNIST. **MEASURED** panels come from the saved local training run. **ILLUSTRATIVE** diagrams and **ANALYTICAL / EXACT** calculations are separate.

The notebook runs offline with the included checkpoints and recordings. Set `RUN_TRAINING=True` to repeat training (first run downloads MNIST). This small instructional run is not a benchmark.
'''),code('''from pathlib import Path
import json, sys, subprocess
import numpy as np
import torch
from IPython.display import display, Image, SVG
ROOT = Path.cwd()
if not (ROOT / 'metrics').exists():
    ROOT = ROOT.parent
assert (ROOT / 'metrics/mnist.json').exists(), 'Run from the project or notebooks directory.'
sys.path.insert(0, str(ROOT / 'experiments'))
from train_mnist import Model
metrics = json.loads((ROOT / 'metrics/mnist.json').read_text())
print('Saved run:', metrics['train_examples'], 'training examples;', metrics['epochs'], 'epochs; seed', metrics['seed'])
print('Preprocessing:', metrics['preprocessing'])
'''),md('''## Reproduce the run

The script trains AE-2, VAE-2 and VAE-16 using the same binary data. It saves weights, train curves, fixed held-out reconstructions, prior draws, decoder means and latent recordings. It never uses test digits during optimization.
'''),code('''RUN_TRAINING = False
if RUN_TRAINING:
    subprocess.run([sys.executable, str(ROOT / 'experiments/train_mnist.py'), '--epochs', '15', '--seed', '667'], check=True)
else:
    print('Using the included measured run. No training is claimed for this notebook execution.')
'''),md('''## One random draw, a differentiable path

With $\\mu=2$, $\\sigma=.5$, $\\epsilon=-.8$, calculate $z$. Then suppose the upstream reconstruction gradient is 3. These are reconstruction-path derivatives only; the KL term supplies additional derivatives in the complete VAE.
'''),code('''mu = torch.tensor(2., requires_grad=True)
sigma = torch.tensor(.5, requires_grad=True)
eps = torch.tensor(-.8)
z = mu + sigma * eps
(3*z).backward()
print('z:', z.item(), 'dL/dmu:', mu.grad.item(), 'dL/dsigma:', sigma.grad.item())
q = torch.distributions.Normal(1., .5)
p = torch.distributions.Normal(0., 1.)
print('KL in nats:', torch.distributions.kl_divergence(q, p).item())
'''),md('''## Load the actual decoder

The network produces Bernoulli logits. Sigmoid gives a mean image; sampling Bernoulli observations gives an actual binary draw. These are different displays.
'''),code('''checkpoint = torch.load(ROOT / 'models/vae2.pt', map_location='cpu', weights_only=True)
model = Model(checkpoint['latent_dim'], vae=True)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
z = torch.tensor([[0., 0.]])
with torch.no_grad():
    mean_image = model.dec(z).sigmoid().reshape(28,28)
print('Shape:', tuple(mean_image.shape), 'mean probability:', mean_image.mean().item())
display(Image(filename=str(ROOT / 'figures/vae2-grid.png'), width=520))
'''),md('''## Inspect held-out reconstructions and prior outputs

Top row: original binary input. Bottom row: decoder mean evaluated at the posterior mean. The prior outputs below are unselected fixed-seed binary draws.
'''),code('''for name in ['vae2', 'vae16']:
    print(name)
    display(Image(filename=str(ROOT / f'figures/{name}-recon.png'), width=760))
    display(Image(filename=str(ROOT / f'figures/{name}-prior-draws.png'), width=360))
'''),md('''## Reproduce interpolation from the saved latent path

The endpoints are the posterior means of the first two test examples. We decode nine equally spaced points. This does not imply that the coordinates have independently interpretable meanings.
'''),code('''ck = torch.load(ROOT / 'models/vae16.pt', map_location='cpu', weights_only=True)
m16 = Model(16); m16.load_state_dict(ck['state_dict']); m16.eval()
record = np.load(ROOT / 'models/vae16-recording.npz')
path = torch.tensor(record['interpolation_z'])
with torch.no_grad():
    pixels = m16.dec(path).sigmoid()
assert pixels.shape == (9, 784)
display(Image(filename=str(ROOT / 'figures/vae16-interpolation.png'), width=900))
display(SVG(filename=str(ROOT / 'figures/training-curves.svg')))
'''),md('''## Evaluate both terms

Reported VAE test values use one stochastic posterior draw per test example at seed 668. Training curves use per-epoch means. Reconstruction sums 784 binary pixel losses; KL sums latent dimensions. All quantities below are nats per example.
'''),code('''for name, v in metrics['models'].items():
    print(name, {'test_reconstruction': round(v['test_reconstruction_nats'], 3),
                 'test_KL': round(v['test_kl_nats'], 3)})
subprocess.run([sys.executable, str(ROOT / 'experiments/verify_math.py')], check=True, capture_output=True)
print('Analytical, autograd and checkpoint-integrity checks passed.')
'''),md('''## The same two-dimensional distribution

The teaching cloud balances 35 draws per component so every mode stays visible. This is a stratified visualization, not an iid categorical sequence. The separate mixture sampler recipe draws its component at random. Contours use the known analytical density; no trained GAN or flow result is implied.
'''),code('''subprocess.run([sys.executable, str(ROOT / 'experiments/distributions.py')], check=True, capture_output=True)
display(SVG(filename=str(ROOT / 'figures/distribution-comparison.svg')))
'''),md('''## Exercises

1. Predict the reconstruction-path derivative with respect to `logvar`; check it with autograd.
2. Change the clicked latent coordinate in the lecture. Explain the difference between a decoder mean and a sampled image.
3. Why does a low AE reconstruction loss not guarantee good standard-prior samples?
4. Freeze discriminator parameters in a tiny network. Check that the generator still gets gradients.
5. Explain why the inverse Jacobian contributes $-\\log 2$ for $x=2z$.

## Sources

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [GANs](https://arxiv.org/abs/1406.2661) · [Normalizing flows](https://arxiv.org/abs/1505.05770). Full data and image provenance: `SOURCES.md`.
''')]
cells.extend([md("""## Rebuild calculations and measured GAN

The shared fixture supplies the slide arithmetic. The GAN snapshots are a separate measured training run on iid draws from the same eight-Gaussian distribution. Coverage alone does not establish that the density matches: inspect the gaps as well.
"""),code("""numbers = json.loads((ROOT/'results/generative_i_teaching_numbers.json').read_text())
for name in ['mle', 'reparam', 'bernoulli', 'kl', 'gan', 'flow']:
    print(name, numbers[name])
subprocess.run([sys.executable, str(ROOT/'experiments/verify_rebuild.py')], check=True, capture_output=True)
print('Fixtures, checkpoint-derived recordings, and bridge frame hashes verified.')
"""),code("""import matplotlib.pyplot as plt
run = json.loads((ROOT/'results/toy-gan.json').read_text())
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharex=True, sharey=True)
for ax, snapshot in zip(axes, run['snapshots']):
    points = np.array(snapshot['points'])
    ax.scatter(points[:, 0], points[:, 1], s=3, color='#8B5BB5', alpha=.6)
    ax.set(xlim=(-4,4), ylim=(-4,4), aspect='equal', title='Step '+str(snapshot['step']))
plt.show()
print(run['diagnostic'])
for snapshot in run['snapshots']:
    print(snapshot['step'], snapshot['coverage_at_least_1pct'], snapshot['fraction_within_3sigma'])
""")])

cells.extend([md('''## Why not sample pixels independently?

These baselines use the training split only: replay the first eight examples, or fit one Bernoulli probability per pixel. Ask what structure each sampler captures before viewing its output.
'''),code('''display(Image(filename=str(ROOT/'figures/pedagogy-replay.png'), width=850))
display(Image(filename=str(ROOT/'figures/pedagogy-independent.png'), width=850))
'''),md('''## A small KL detour

KL(q || p) averages log(q/p) under q. Individual contributions may be negative; the total is nonnegative. Reversing the arguments changes both the log ratio and the averaging weights.
'''),code('''q = np.array([.8, .2]); p = np.array([.5, .5])
contributions = q * np.log(q/p)
print('Contributions:', contributions)
print('KL(q || p):', contributions.sum(), 'nats')
print('KL(p || q):', (p*np.log(p/q)).sum(), 'nats')
assert np.isclose(contributions.sum(), .19274475702175753)
'''),md('''## Compute an evidence bound exactly in a finite model

The generative model has three latent states. Hold its parameters fixed while improving q. This lets us compute the evidence and posterior KL gap, quantities generally unavailable exactly in the neural model.
'''),code('''prior = np.array([.2, .5, .3]); likelihood = np.array([.1, .8, .2])
evidence = prior @ likelihood
posterior = prior * likelihood / evidence
for q in [np.array([.4,.3,.3]), np.array([.15,.7,.15]), posterior]:
    elbo = np.sum(q*np.log(prior*likelihood/q))
    gap = np.sum(q*np.log(q/posterior))
    print('q:', q, 'ELBO:', elbo, 'gap:', gap)
    assert np.isclose(np.log(evidence), elbo+gap)
'''),md('''## sample() versus rsample()

PyTorch's ordinary sample does not retain this pathwise autograd connection. A reparameterized draw does. Other gradient estimators exist; stochastic computation is not inherently impossible to differentiate.
'''),code('''mu = torch.tensor(2., requires_grad=True)
std = torch.tensor(.5, requires_grad=True)
dist = torch.distributions.Normal(mu, std)
print('sample requires grad:', dist.sample().requires_grad)
torch.manual_seed(667)
z = dist.rsample()
z.backward()
print('rsample gradients:', mu.grad.item(), std.grad.item())
assert mu.grad.item() == 1.
'''),md('''## Controlled beta experiment

Three separate 2D VAEs use identical initialization, data order and noise seeds: first 12,000 training images, 12 epochs, Adam 0.001. Reconstruction sums pixel BCE; KL sums latent coordinates; both then average over examples. The same held-out images and prior vectors appear in every panel. This single-seed comparison is distinct from the lecture's original 60,000-image run.

Beta = 0 retains a stochastic encoder. Larger beta does not guarantee semantic axes or disentanglement. Test metrics use one posterior sample per example; the Gaussian KL is analytical.
'''),code('''ped = json.loads((ROOT/'metrics/pedagogy.json').read_text())
for row in ped['beta']:
    name = 'beta'+str(int(row['beta']))
    print(name, 'test reconstruction:', row['test_reconstruction_nats'], 'test KL:', row['test_kl_nats'])
    display(Image(filename=str(ROOT/f'figures/pedagogy-{name}-recon.png'), width=850))
    display(Image(filename=str(ROOT/f'figures/pedagogy-{name}-prior.png'), width=850))
RUN_BETA_TRAINING = False
if RUN_BETA_TRAINING:
    subprocess.run([sys.executable, str(ROOT/'experiments/pedagogy_evidence.py')], check=True)
''')])

from probability_notebook import extra_cells
cells.extend(extra_cells(md,code))

nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
for i,c in enumerate(nb['cells']):c['id']=f'cell-{i:02d}'
(ROOT/'notebooks/generative_i_lab.ipynb').write_text(json.dumps(nb,indent=1))
print('Wrote offline notebook with optional training cell.')
