"""Build and execute the self-contained Lecture 1 notebook without installing packages."""
from pathlib import Path
import sys
import json
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
ROOT.mkdir(parents=True, exist_ok=True)
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Fitting an approximation to an unknown posterior

**Lecture 1 · KL divergence, variational inference and the ELBO**

This notebook follows the PML 2023 Gaussian → mixture → differentiable sampling → coin sequence. The lecture uses three short demonstrations, marked **Live demo**; the saved outputs let class continue without waiting for training.

All distributions use natural logarithms. `Normal(loc=μ, scale=σ)` takes a **standard deviation**. Mathematical notation is $\mathcal N(\mu,\sigma^2)$. We use $u$ for a generic unknown, $r$ for a coin's probability of heads, $\mathcal D$ for observed data and $\psi$ for free variational parameters.

Run all cells, in order, in a Python environment with PyTorch, NumPy and matplotlib. No dataset download, GPU, external training framework or local helper import is needed. This notebook writes its measured figures and full-precision fixtures into `figures/` and `data/`.
""")
code(r"""
from pathlib import Path
import json, math, platform
import numpy as np
import torch
from torch import distributions as dist
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(20261020)
np.random.seed(20261020)
OUT = Path.cwd()
for folder in ["figures", "data", "qa"]:
    (OUT / folder).mkdir(exist_ok=True)

C = dict(p="#0F766E", q="#8B2CDE", data="#245EDB", noise="#AA4E08", loss="#BE123C")
plt.rcParams.update({
    "font.family": ["Avenir Next", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 19, "axes.titlesize": 21,
    "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#FAFAFB", "axes.facecolor": "#FAFAFB",
    "svg.fonttype": "none", "lines.linewidth": 2.5,
})
fixtures = {"provenance": {
    "seed": 20261020, "dtype": "torch.float64", "torch": torch.__version__,
    "python": platform.python_version(), "numpy": np.__version__,
    "evidence": "computed locally; analytic, quadrature and Monte Carlo outputs labelled separately",
    "source_spine": "PML 2023 variational-inference.ipynb; corrected standalone implementation",
}}
def number(x): return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.savefig(OUT / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()
def normal_kl(q, p):
    return torch.log(p.scale/q.scale) + (q.scale.square() + (q.loc-p.loc).square())/(2*p.scale.square()) - .5
print("CPU; float64; PyTorch", torch.__version__, "· fixed seeds recorded per experiment")
""")
md(r"""
## 1. Which approximation is closer?

Fix $p=\mathcal N(1,4^2)$. Compare $q_1=\mathcal N(0,2^2)$ and $q_2=\mathcal N(1,3^2)$.

**Predict first:** compare their centres and widths. A visual judgement motivates a score; it does not define one.
""")
code(r"""
p = dist.Normal(torch.tensor(1.), torch.tensor(4.))
q1 = dist.Normal(torch.tensor(0.), torch.tensor(2.))
q2 = dist.Normal(torch.tensor(1.), torch.tensor(3.))
u = torch.linspace(-11, 13, 481)
fig, ax = plt.subplots(figsize=(8.8, 4.5))
for d, label, col, style in [(p, "p: fixed target", C["p"], "-"),
                            (q1, "q₁: mean 0, scale 2", C["q"], "-"),
                            (q2, "q₂: mean 1, scale 3", C["q"], "--")]:
    ax.plot(u, d.log_prob(u).exp(), label=label, color=col, ls=style)
ax.set(xlim=(-11,13), ylim=(0,.23), xlabel="u", ylabel="density")
ax.legend(frameon=False, fontsize=12)
savefig("gaussian-candidates")
""")
md(r"""
### A local log ratio, then an average

$\log q(u)-\log p(u)$ compares the two densities at one location. The first argument in

$$\mathrm{KL}(q\|p)=\mathbb E_{u\sim q}\left[\log q(u)-\log p(u)\right]$$

tells us which distribution weights the average. A local log ratio can be negative. The exact KL is nonnegative, under the usual support and finiteness conditions.

For two Normals:

$$\mathrm{KL}(q\|p)=\log\frac{s}{\sigma}+
\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12.$$

The key moment is $\mathbb E_q[(u-m)^2]=\sigma^2+(\mu-m)^2$. The cross-term vanishes because $\mathbb E_q[u-\mu]=0$.
""")
code(r"""
normal_values = {}
for name, q in [("q1", q1), ("q2", q2)]:
    manual = normal_kl(q,p)
    library = dist.kl_divergence(q,p)
    reverse = dist.kl_divergence(p,q)
    assert torch.allclose(manual, library, atol=1e-12, rtol=0)
    check_grid=torch.linspace(number(q.loc-12*q.scale),number(q.loc+12*q.scale),100001)
    integral=torch.trapezoid(q.log_prob(check_grid).exp()*
                            (q.log_prob(check_grid)-p.log_prob(check_grid)),check_grid)
    assert abs(number(manual-integral))<1e-9
    normal_values[name] = dict(mu=number(q.loc), sigma=number(q.scale),
                             kl=number(manual), reverse_kl=number(reverse),quadrature=number(integral))
    print(f"{name}: KL(q||p) = {manual:.9f}; KL(p||q) = {reverse:.9f} nats")
assert dist.kl_divergence(p,p) == 0
print("Matched distributions: KL(p||p) =", number(dist.kl_divergence(p,p)))
fixtures["normal"] = {"target": {"mu":1., "sigma":4.}, "candidates":normal_values}
""")
md(r"""
## 2. Turn the score into an optimisation problem

**Live demo 1 · about 3 minutes.** Hold $\sigma=2$ fixed and start at $\mu=0$.

$$\frac{\partial L}{\partial\mu}=\frac{\mu-1}{16},\qquad
\mu^+=\mu-\eta\frac{\partial L}{\partial\mu}.$$

With chosen $\eta=4$, predict the new mean before running the cell. A negative derivative means gradient descent moves the mean right.
""")
code(r"""
mu = torch.tensor(0., requires_grad=True)
q = dist.Normal(mu, torch.tensor(2.))
before = dist.kl_divergence(q,p)
before.backward()
gradient = number(mu.grad)
eta = 4.
with torch.no_grad():
    mu_after = mu - eta*mu.grad
after = dist.kl_divergence(dist.Normal(mu_after, torch.tensor(2.)), p)
h = 1e-5
fd = (normal_kl(dist.Normal(mu.detach()+h,2.),p)
      - normal_kl(dist.Normal(mu.detach()-h,2.),p))/(2*h)
assert abs(gradient-number(fd)) < 1e-9
assert number(after) < number(before)
fixtures["normal"]["one_step"] = dict(mu_before=0., sigma=2., eta=eta,
    gradient=gradient, mu_after=number(mu_after), kl_before=number(before),
    kl_after=number(after), finite_difference=number(fd),
    fixed_scale_best_kl=number(dist.kl_divergence(dist.Normal(1.,2.),p)))
print(fixtures["normal"]["one_step"])
""")
md(r"""
Moving only the mean cannot match a width of 4. Its best possible KL is still positive.

To learn both parameters, optimise unconstrained $\mu,\rho$ and construct $\sigma=\exp(\rho)>0$. Here $\rho$ means **log standard deviation**, not log variance. The following saved trajectory is an actual deterministic optimisation of analytic KL.
""")
code(r"""
mu = torch.nn.Parameter(torch.tensor(0.))
rho = torch.nn.Parameter(torch.tensor(math.log(2.)))
opt = torch.optim.Adam([mu, rho], lr=.08)
normal_trace = []
for step in range(151):
    q = dist.Normal(mu, rho.exp())
    loss = dist.kl_divergence(q,p)
    normal_trace.append(dict(step=step, mu=number(mu), sigma=number(rho.exp()), kl=number(loss)))
    if step == 150: break
    opt.zero_grad()
    loss.backward()
    opt.step()
assert normal_trace[-1]["kl"] < 1e-6
assert all(row["sigma"] > 0 for row in normal_trace)
fixtures["normal"]["training"] = dict(method="Adam on analytic KL", lr=.08, steps=150,
    parameterisation="sigma=exp(rho)", trace=normal_trace)
fig, axes = plt.subplots(1,2,figsize=(10,4.2))
for step in [0,10,30,150]:
    row = normal_trace[step]
    q = dist.Normal(row["mu"],row["sigma"])
    axes[0].plot(u,q.log_prob(u).exp(),label=f"step {step}",alpha=.65 if step<150 else 1)
axes[0].plot(u,p.log_prob(u).exp(),color=C["p"],ls="--",label="target p")
axes[0].set(xlim=(-11,13),ylim=(0,.23),xlabel="u",ylabel="density")
axes[0].legend(fontsize=10,frameon=False)
axes[1].plot([r["step"] for r in normal_trace],[r["kl"] for r in normal_trace],color=C["loss"])
axes[1].set(xlabel="optimiser step",ylabel="analytic KL (nats)")
savefig("gaussian-fit")
print("Final measured parameters:", normal_trace[-1])
""")
md(r"""
## 3. What if the library has no KL formula?

Use the original PML mixture:

$$p(u)=0.5\,\mathcal N(u;-0.2,0.4^2)+0.5\,\mathcal N(u;1,0.5^2).$$

Compare $\mathcal N(3,1^2)$ and $\mathcal N(3,4.5^2)$. An unsupported library pair is a missing implementation, not a proof that the KL is undefined.
""")
code(r"""
mixture = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([.5,.5])),
    dist.Normal(torch.tensor([-.2,1.]), torch.tensor([.4,.5])))
mix_q1, mix_q2 = dist.Normal(3.,1.), dist.Normal(3.,4.5)
try:
    dist.kl_divergence(mix_q1,mixture)
    unsupported = "This runtime implements the pair."
except NotImplementedError as exc:
    unsupported = str(exc)
print(unsupported)
xx = torch.linspace(-12,18,901)
fig, ax = plt.subplots(figsize=(8.8,4.4))
ax.plot(xx,mixture.log_prob(xx).exp(),color=C["p"],label="p: the PML mixture")
ax.plot(xx,mix_q1.log_prob(xx).exp(),color=C["q"],label="q₁: scale 1")
ax.plot(xx,mix_q2.log_prob(xx).exp(),color=C["q"],ls="--",label="q₂: scale 4.5")
ax.set(xlim=(-12,18),ylim=(0,.62),xlabel="u",ylabel="density")
ax.legend(frameon=False,fontsize=12)
savefig("mixture-candidates")
""")
md(r"""
Draw $u_1,\dots,u_S$ from **q**, evaluate the log ratio at each draw, then average over the sample dimension:

$$\widehat{\mathrm{KL}}_S=\frac1S\sum_{i=1}^S[\log q(u_i)-\log p(u_i)].$$

This is a Monte Carlo estimate. Validate the estimator on a known Gaussian case before using it for the mixture.
""")
code(r"""
def mc_kl(q,p,samples,seed):
    torch.manual_seed(seed)
    draws=q.sample((samples,))
    terms=q.log_prob(draws)-p.log_prob(draws)
    return terms.mean(), terms.std(unbiased=True)/math.sqrt(samples)

mix_results=[]
for i,q in enumerate([mix_q1,mix_q2],1):
    value,se=mc_kl(q,mixture,100_000,401+i)
    # Independent numerical integral, using a wide q-dependent domain.
    grid=torch.linspace(number(q.loc-12*q.scale),number(q.loc+12*q.scale),200_001)
    integrand=q.log_prob(grid).exp()*(q.log_prob(grid)-mixture.log_prob(grid))
    numerical=torch.trapezoid(integrand,grid)
    assert abs(number(value-numerical))<5*number(se)
    mix_results.append(dict(id=f"q{i}",mu=3.,sigma=number(q.scale),mc=number(value),
        standard_error=number(se),samples=100_000,seed=401+i,quadrature=number(numerical)))
    print(f"q{i}: MC estimate {value:.5f} ± {se:.5f} (one standard error); numerical integral {numerical:.5f}")
fixtures["mixture"]={"weights":[.5,.5],"means":[-.2,1.],"scales":[.4,.5],
                     "unsupported_message":unsupported,"candidates":mix_results}
""")
md(r"""
### Live demo 2 · estimation and gradient inspection, about 3 minutes

Keep p and q₂ fixed. Vary only the number of draws and the seed. The exact KL is unchanged. Larger S reduces estimator variance; individual estimates need not move monotonically towards the answer.
""")
code(r"""
repeats=[]
for size in [10,100,1000,10000]:
    vals=[number(mc_kl(q2,p,size,1000+j)[0]) for j in range(40)]
    repeats.append(dict(samples=size,seeds=list(range(1000,1040)),estimates=vals,
                        empirical_mean=float(np.mean(vals)),empirical_std=float(np.std(vals,ddof=1))))
fig,ax=plt.subplots(figsize=(11,3.6))
for i,row in enumerate(repeats):
    jitter=np.linspace(-.18,.18,len(row["estimates"]))
    ax.scatter(i+jitter,row["estimates"],s=18,color=C["q"],alpha=.55)
ax.axhline(number(dist.kl_divergence(q2,p)),color=C["p"],label="exact KL",lw=2)
ax.axhline(0,color="#9CA3AF",lw=1)
ax.set(xticks=range(4),xticklabels=["10","100","1,000","10,000"],
       xlabel="Monte Carlo draws S",ylabel="estimated KL (nats)")
ax.legend(frameon=False,fontsize=19)
savefig("mc-variability")
fixtures["monte_carlo"]={"q":"q2","exact_kl":number(dist.kl_divergence(q2,p)),
                         "repeats":repeats}
print([(r["samples"],round(r["empirical_std"],5)) for r in repeats])
""")
md(r"""
A finite-sample KL estimate can be negative even though the true KL is nonnegative. The next cell deliberately finds the first seed giving a negative **five-draw estimate**, as a counterexample. This seed selection is not a performance comparison.
""")
code(r"""
negative_seed=next(s for s in range(1000) if number(mc_kl(q2,p,5,s)[0])<0)
torch.manual_seed(negative_seed)
draws=q2.sample((5,))
terms=q2.log_prob(draws)-p.log_prob(draws)
fixtures["monte_carlo"]["negative_example"]=dict(seed=negative_seed,samples=5,
    selection="first seed in 0..999 giving a negative estimate; intentional counterexample",
    draws=draws.tolist(),log_q=q2.log_prob(draws).tolist(),log_p=p.log_prob(draws).tolist(),
    terms=terms.tolist(),estimate=number(terms.mean()))
print("Selected seed:",negative_seed)
for x,lq,lp,t in zip(draws,q2.log_prob(draws),p.log_prob(draws),terms):
    print(f"u={x: .4f}  log q={lq: .4f}  log p={lp: .4f}  difference={t: .4f}")
print("Five-draw estimate:",number(terms.mean()),"· exact KL:",number(dist.kl_divergence(q2,p)))
""")
md(r"""
## 4. Can we optimise through random draws?

First inspect the sample itself. `Normal.sample()` returns a draw without the pathwise gradient connection to the Normal's parameters. Asking that scalar draw for a gradient fails.

However, a larger loss containing `q.log_prob(draw)` can still produce a gradient through the explicit log-density parameters. **A gradient tensor alone does not prove that the missing sample path was handled correctly.**
""")
code(r"""
torch.manual_seed(12)
mu=torch.tensor(0.,requires_grad=True)
sigma=torch.tensor(2.,requires_grad=True)
q=dist.Normal(mu,sigma)
draw=q.sample()
try:
    draw.backward()
    sample_error="No error"
except RuntimeError as exc:
    sample_error=str(exc)
print("sample.requires_grad:",draw.requires_grad)
print("sample.backward():",sample_error)

# Same chosen noise lets us compare estimators without confounding them with new draws.
torch.manual_seed(13)
eps=torch.randn(50_000)
detached=(mu+sigma*eps).detach()
wrong_loss=(q.log_prob(detached)-p.log_prob(detached)).mean()
wrong_grads=torch.autograd.grad(wrong_loss,(mu,sigma))
correct_draw=mu+sigma*eps
path_loss=(q.log_prob(correct_draw)-p.log_prob(correct_draw)).mean()
path_grads=torch.autograd.grad(path_loss,(mu,sigma))
exact_grads=((0.-1.)/16,2./16-1./2)
fixtures["gradients"]={"sample_seed":12,"sample_requires_grad":draw.requires_grad,
    "sample_backward_error":sample_error,"comparison_seed":13,"samples":50000,
    "detached_sample_grads":[number(v) for v in wrong_grads],
    "pathwise_grads":[number(v) for v in path_grads],"analytic_grads":list(exact_grads)}
print("Naive detached-sample grads:",[number(v) for v in wrong_grads])
print("Pathwise grads:",[number(v) for v in path_grads])
print("Analytic reference:",exact_grads)
""")
md(r"""
The naive detached-sample derivative above omits the effect of changing q's sampling distribution. A properly constructed **score-function estimator** is another valid route; we are not claiming that all useful gradients require reparameterisation.

### Put the randomness outside the parameters

$$\epsilon\sim\mathcal N(0,1),\quad u=\mu+\sigma\epsilon,\quad
\frac{\partial u}{\partial\mu}=1,\quad
\frac{\partial u}{\partial\sigma}=\epsilon.$$

Hold the same $\epsilon$ fixed while checking derivatives. Normal training uses fresh noise. The chosen scalar $\epsilon=-0.75$ below makes the arithmetic inspectable; it is not presented as a random sample.
""")
code(r"""
mu=torch.tensor(0.,requires_grad=True)
sigma=torch.tensor(2.,requires_grad=True)
eps_chosen=torch.tensor(-.75)
draw=mu+sigma*eps_chosen
du=torch.autograd.grad(draw,(mu,sigma),retain_graph=True)
scalar_loss=(draw-1).square()
dL=torch.autograd.grad(scalar_loss,(mu,sigma))
fixtures["gradients"]["scalar"]=dict(mu=0.,sigma=2.,epsilon=-.75,u=number(draw),
    loss=number(scalar_loss),du_dmu=number(du[0]),du_dsigma=number(du[1]),
    dloss_dmu=number(dL[0]),dloss_dsigma=number(dL[1]))
print(fixtures["gradients"]["scalar"])
""")
md(r"""
`Normal.rsample()` uses a differentiable draw. To check the derivative numerically, reuse one fixed noise array in both finite-difference evaluations.
""")
code(r"""
torch.manual_seed(88)
eps_fixed=torch.randn(2048)
def fixed_noise_loss(mu_value,sigma_value):
    q=dist.Normal(mu_value,sigma_value)
    x=mu_value+sigma_value*eps_fixed
    return (q.log_prob(x)-p.log_prob(x)).mean()
mu=torch.tensor(0.,requires_grad=True)
sigma=torch.tensor(2.,requires_grad=True)
g=torch.autograd.grad(fixed_noise_loss(mu,sigma),(mu,sigma))
h=1e-5
fd_mu=(fixed_noise_loss(mu.detach()+h,sigma.detach())-fixed_noise_loss(mu.detach()-h,sigma.detach()))/(2*h)
fd_sigma=(fixed_noise_loss(mu.detach(),sigma.detach()+h)-fixed_noise_loss(mu.detach(),sigma.detach()-h))/(2*h)
assert abs(number(g[0]-fd_mu))<1e-8
assert abs(number(g[1]-fd_sigma))<1e-8
torch.manual_seed(88)
rs=dist.Normal(mu,sigma).rsample((2048,))
assert torch.allclose(rs,mu+sigma*eps_fixed)
fixtures["gradients"]["finite_difference"]=dict(seed=88,samples=2048,h=h,
    autograd=[number(v) for v in g],finite_difference=[number(fd_mu),number(fd_sigma)])
print(fixtures["gradients"]["finite_difference"])
""")
md(r"""
## 5. The target is now an unknown posterior

For a fixed model and observed dataset:

$$\begin{aligned}
\mathrm{KL}(q_\psi(u)\|p(u\mid\mathcal D))
&=\mathbb E_q[\log q_\psi(u)-\log p(u\mid\mathcal D)]\\
&=\mathbb E_q[\log q_\psi(u)-\log p(\mathcal D,u)+\log p(\mathcal D)]\\
&=\mathbb E_q[\log q_\psi(u)-\log p(\mathcal D,u)]+\log p(\mathcal D).
\end{aligned}$$

The second line uses Bayes' rule and the log quotient. The third uses linearity: the evidence does not vary with u or ψ for this fixed model.

Rearranging:

$$\log p(\mathcal D)=
\underbrace{\mathbb E_q[\log p(\mathcal D,u)-\log q_\psi(u)]}_{\mathrm{ELBO}(q_\psi)}
+\mathrm{KL}(q_\psi\|p(u\mid\mathcal D)).$$

Maximise the ELBO, or minimise its negative. We can evaluate the **joint** without the evidence normaliser. Estimating the expectation can still require Monte Carlo. A finite-sample ELBO estimate need not lie below the evidence on every draw; the exact expectation is the lower bound.
""")
md(r"""
## 6. Fit the coin, then check a reference answer

Use a Beta(5,3) prior and **one ordered sequence** with 17 heads and 13 tails. Any order with these counts has the same sequence likelihood:

$$p(\mathcal D\mid r)=r^{17}(1-r)^{13}.$$

There is no binomial coefficient in this convention. The known posterior Beta(22,16) and known evidence are for **validation**, not inputs to the optimiser.
""")
code(r"""
tosses="HHTHTHTHHHTTHTHHTHTTHHHTTHTHTH"
assert len(tosses)==30 and tosses.count("H")==17 and tosses.count("T")==13
prior=dist.Beta(torch.tensor(5.),torch.tensor(3.))
posterior=dist.Beta(torch.tensor(22.),torch.tensor(16.))
heads,tails=17,13
def log_joint(r):
    return prior.log_prob(r)+heads*torch.log(r)+tails*torch.log1p(-r)
def log_beta(a,b):
    return torch.lgamma(a)+torch.lgamma(b)-torch.lgamma(a+b)
log_evidence=log_beta(torch.tensor(22.),torch.tensor(16.))-log_beta(torch.tensor(5.),torch.tensor(3.))
def direct_beta_elbo(a,b):
    a,b=torch.as_tensor(a),torch.as_tensor(b)
    mean_log_r=torch.digamma(a)-torch.digamma(a+b)
    mean_log_1mr=torch.digamma(b)-torch.digamma(a+b)
    expected_likelihood=heads*mean_log_r+tails*mean_log_1mr
    expected_prior=4*mean_log_r+2*mean_log_1mr-log_beta(torch.tensor(5.),torch.tensor(3.))
    expected_log_q=(a-1)*mean_log_r+(b-1)*mean_log_1mr-log_beta(a,b)
    return expected_likelihood+expected_prior-expected_log_q
coin_candidates=[]
for a,b in [(10.,10.),(30.,20.),(22.,16.)]:
    q=dist.Beta(torch.tensor(a),torch.tensor(b))
    elbo=direct_beta_elbo(a,b)
    gap=dist.kl_divergence(q,posterior)
    assert abs(number(elbo+gap-log_evidence))<1e-10
    coin_candidates.append(dict(a=a,b=b,elbo=number(elbo),posterior_kl=number(gap)))
    print(f"Beta({a:g},{b:g}): ELBO={elbo:.9f}; posterior KL={gap:.9f}; sum={elbo+gap:.9f}")
fixtures["coin"]={"ordered_sequence":tosses,"heads":heads,"tails":tails,
    "prior":[5,3],"posterior":[22,16],"log_evidence":number(log_evidence),
    "candidates":coin_candidates,"convention":"ordered sequence; no binomial coefficient"}
""")
md(r"""
### Live demo 3 · joint-only optimisation, about 3–4 minutes

Let $q_\psi(r)=\mathrm{Beta}(a,b)$ with $a=\exp(\psi_a)$ and $b=\exp(\psi_b)$. Both shape parameters remain positive.

For S scalar draws, `log_joint(r)` and `q.log_prob(r)` each have shape `[S]`. Their difference is averaged **once over S**. The 30 observations are already summed in the log likelihood. We do not multiply the entropy by 30 or sum the sample dimension before averaging.

`Beta.rsample()` is differentiable, but it is not the Gaussian formula $\mu+\sigma\epsilon$. PyTorch supplies an implicit reparameterisation gradient.
""")
code(r"""
torch.manual_seed(2401)
psi=torch.nn.Parameter(torch.log(torch.tensor([10.,10.])))
optimiser=torch.optim.Adam([psi],lr=.025)
coin_trace=[]
for step in range(601):
    a,b=psi.exp()
    q=dist.Beta(a,b)
    with torch.no_grad():
        exact_elbo=direct_beta_elbo(a,b)
        exact_gap=dist.kl_divergence(q,posterior)
        coin_trace.append(dict(step=step,a=number(a),b=number(b),
                              exact_elbo=number(exact_elbo),posterior_kl=number(exact_gap)))
    if step==600: break
    r=q.rsample((512,))
    terms=log_joint(r)-q.log_prob(r)
    assert terms.shape==(512,)
    loss=-terms.mean()
    optimiser.zero_grad()
    loss.backward()
    if step==0:
        fixtures["coin"]["first_step"]=dict(a=number(a),b=number(b),
            sampled_negative_elbo=number(loss),gradient_psi=psi.grad.tolist(),
            shape=list(terms.shape),reduction="mean over 512 independent scalar r draws")
    optimiser.step()
    coin_trace[-1]["sampled_negative_elbo"]=number(loss)
assert coin_trace[-1]["posterior_kl"]<.01
assert all(r["a"]>0 and r["b"]>0 for r in coin_trace)
fixtures["coin"]["training"]={"seed":2401,"steps":600,"samples_per_step":512,
    "learning_rate":.025,"optimiser":"Adam","parameterisation":"a,b=exp(psi)",
    "objective":"negative Monte Carlo mean of log_joint(r)-log_q(r)",
    "reference_used_by_optimiser":False,"trace":coin_trace}
print("Measured final fit:",coin_trace[-1])
print("Reference Beta(22,16) was used only for diagnostics.")
""")
code(r"""
rr=torch.linspace(.001,.999,501)
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
axes[0].plot(rr,posterior.log_prob(rr).exp(),color=C["p"],label="known posterior")
final=coin_trace[-1]
fitted=dist.Beta(final["a"],final["b"])
axes[0].plot(rr,fitted.log_prob(rr).exp(),color=C["q"],ls="--",label="measured fitted q")
axes[0].set(xlim=(0,1),ylim=(0,6.5),xlabel="r: chance of heads",ylabel="density")
axes[0].legend(frameon=False,fontsize=11)
axes[1].plot([r["step"] for r in coin_trace],[r["exact_elbo"] for r in coin_trace],
             color=C["q"],label="analytic ELBO diagnostic")
axes[1].axhline(number(log_evidence),color=C["data"],ls="--",label="log evidence reference")
axes[1].set(xlabel="optimiser step",ylabel="nats")
axes[1].legend(frameon=False,fontsize=10)
savefig("coin-fit")
""")
md(r"""
### A noisy objective is not the exact bound

The next figure compares the actual stochastic objective used during training with the analytic ELBO of the same pre-update q. Training samples change at every step. Apparent one-step movement of the noisy trace is not a certificate of improvement.

The exact posterior is in the Beta family, so exact recovery is possible in principle. A restricted family, incomplete optimisation, or local optima can prevent it in other problems. A stable training loss does not prove zero posterior KL.
""")
code(r"""
fig,ax=plt.subplots(figsize=(11,3.6))
steps=np.array([r["step"] for r in coin_trace[:-1]])
sampled=np.array([-r["sampled_negative_elbo"] for r in coin_trace[:-1]])
exact=np.array([r["exact_elbo"] for r in coin_trace[:-1]])
ax.plot(steps,sampled,color=C["q"],alpha=.35,lw=1,label="sampled ELBO")
ax.plot(steps,exact,color=C["p"],label="analytic ELBO")
ax.axhline(number(log_evidence),color=C["data"],ls="--",label="log evidence")
ax.set(xlim=(0,600),xlabel="optimiser step",ylabel="ELBO / estimate (nats)")
ax.legend(frameon=False,fontsize=19)
savefig("coin-noisy-objective")
fixtures["coin"]["sampled_estimates_above_evidence"]=int(np.sum(sampled>number(log_evidence)))
print("MC estimates above the exact evidence:",fixtures["coin"]["sampled_estimates_above_evidence"])
""")
md(r"""
## 7. Retrieve the method

Before revealing the answers, answer these in your own words:

1. For KL(q‖p), where do the samples come from?
2. What changes during VI: data, model, posterior, variational parameters, or all of them?
3. Why can we optimise without knowing the evidence?
4. Does an ELBO estimate above the evidence disprove the lower-bound identity?
5. Does a high or stable ELBO prove that q equals the posterior?

**Answers.** Draw from q. In the fixed-model inference problem, update ψ; data, model and exact posterior stay fixed. The evidence is constant in ψ. A finite-sample estimate is not the exact expectation. Improvement is not an exact-recovery certificate.

**Transfer:** for an image model $p_\theta(x,z)$ with fixed θ, specify $q_\psi(z)$ and maximise $\mathbb E_q[\log p_\theta(x,z)-\log q_\psi(z)]$. Lecture 3 will learn model parameters and share an encoder across observations.
""")
md(r"""
## Optional depth: an independent integral check

The scalar Beta example also permits numerical integration. This is an approximate quadrature check, labelled separately from the analytic formula. At endpoints the integrands have zero limits for these chosen shape parameters.
""")
code(r"""
grid=torch.linspace(0,1,100001)
interior=grid[1:-1]
quad_checks=[]
for row in coin_candidates:
    q=dist.Beta(row["a"],row["b"])
    density=q.log_prob(interior).exp()
    integrand=density*(log_joint(interior)-q.log_prob(interior))
    padded=torch.cat([torch.zeros(1),integrand,torch.zeros(1)])
    numerical=torch.trapezoid(padded,grid)
    error=abs(number(numerical)-row["elbo"])
    assert error<1e-8
    quad_checks.append(dict(a=row["a"],b=row["b"],numerical_elbo=number(numerical),error=error))
fixtures["checks"]={"coin_quadrature":quad_checks,
    "normal_analytic_vs_library":True,"normal_gradient_finite_difference":True,
    "positive_scales_and_shapes":True,"fixed_noise_gradient_check":True,
    "coin_direct_elbo_identity":True,"coin_training_final_gap":coin_trace[-1]["posterior_kl"]}
print(quad_checks)
""")
md(r"""
## Optional depth: count data versus an ordered sequence

If the observation were only the count “17 heads out of 30”, the likelihood would include $\binom{30}{17}$. This constant would add to both log evidence and ELBO. The posterior and the posterior-KL gap would be unchanged. We have used the ordered-sequence convention consistently.
""")
code(r"""
log_choose=math.lgamma(31)-math.lgamma(18)-math.lgamma(14)
print("Log binomial coefficient:",log_choose)
print("Count-data log evidence:",number(log_evidence)+log_choose)
print("Ordered-sequence log evidence:",number(log_evidence))
fixtures["coin"]["count_convention_comparison"]={"log_binomial_coefficient":log_choose,
    "count_log_evidence":number(log_evidence)+log_choose}
fixtures["plots"]={
    "normal":{"x":u.tolist(),"p":p.log_prob(u).exp().tolist(),
        "q1":q1.log_prob(u).exp().tolist(),"q2":q2.log_prob(u).exp().tolist(),
        "updated":dist.Normal(.25,2.).log_prob(u).exp().tolist(),
        "snapshots":{str(s):dist.Normal(normal_trace[s]["mu"],normal_trace[s]["sigma"]).log_prob(u).exp().tolist()
                     for s in [0,10,30,150]}},
    "mixture":{"x":xx.tolist(),"p":mixture.log_prob(xx).exp().tolist(),
        "q1":mix_q1.log_prob(xx).exp().tolist(),"q2":mix_q2.log_prob(xx).exp().tolist()},
    "coin":{"x":rr.tolist(),"prior":prior.log_prob(rr).exp().tolist(),
        "posterior":posterior.log_prob(rr).exp().tolist(),
        "candidates":[dist.Beta(row["a"],row["b"]).log_prob(rr).exp().tolist() for row in coin_candidates],
        "snapshots":{str(s):dist.Beta(coin_trace[s]["a"],coin_trace[s]["b"]).log_prob(rr).exp().tolist()
                     for s in [0,20,100,600]}}
}
(OUT/"data"/"science.json").write_text(json.dumps(fixtures,indent=2,allow_nan=False)+"\n")
(OUT/"qa"/"notebook-checks.json").write_text(json.dumps({
    "status":"passed","runtime":fixtures["provenance"],"checks":fixtures["checks"],
    "executed_code_cells":"all cells in saved notebook",
    "training":"toy CPU optimisation; measured trajectories, not schematic",
},indent=2,allow_nan=False)+"\n")
print("Saved full-precision fixtures and verification report.")
""")
md(r"""
## Sources and adaptation

- Nipun Batra, [PML 2023 VI notebook](https://github.com/nipunbatra/pml-teaching/blob/master/notebooks/variational-inference.ipynb): Gaussian, mixture, reparameterisation and coin teaching sequence.
- [PyTorch distribution documentation](https://pytorch.org/docs/stable/distributions.html): distribution parameters, `sample`, `rsample` and KL API.
- Figurnov, Mohamed and Mnih, [Implicit Reparameterization Gradients](https://arxiv.org/abs/1805.08498): differentiable sampling beyond the location–scale Gaussian form.

This is a new standalone implementation. It removes hidden training helpers, stale computation-graph references and unsafe unconstrained positive parameters from the historical notebook. Numerical results here come from the saved execution, with the configurations above.
""")

nb=nbf.v4.new_notebook(cells=cells)
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":sys.version.split()[0]}}
path=ROOT/"lecture1-variational-inference.ipynb"
nbf.write(nb,path)
(ROOT/"scripts"/"run_science.py").write_text(
    '"""Self-contained code exported from the lecture notebook; run from lecture1/."""\n'+
    "\n\n".join(c.source for c in cells if c.cell_type=="code")+"\n")
if "--no-execute" not in sys.argv:
    km=KernelManager(kernel_name="python3")
    km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    client=NotebookClient(nb,timeout=180,kernel_name="python3",km=km,
                          resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    nbf.write(nb,path)
    print(f"Executed {sum(c.cell_type=='code' for c in cells)} code cells: {path}")
else:
    print(f"Prepared notebook: {path}")
