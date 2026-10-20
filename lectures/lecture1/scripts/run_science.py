"""Self-contained code exported from the lecture notebook; run from lecture1/."""
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

heads_w, tails_w = 17, 13
def lgB(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
conj_log_evidence = lgB(22, 16) - lgB(5, 3)
w_prior = dist.Normal(torch.tensor(0.), torch.tensor(1.))
def logit_log_joint(w):
    F = torch.nn.functional
    return w_prior.log_prob(w) + heads_w * F.logsigmoid(w) + tails_w * F.logsigmoid(-w)
w_grid = torch.linspace(-10, 10, 200_001)
dw = number(w_grid[1] - w_grid[0])
logit_log_evidence = number(torch.logsumexp(logit_log_joint(w_grid), 0)) + math.log(dw)
torch.manual_seed(31)
w_draws = w_prior.sample((100_000,))
F = torch.nn.functional
w_loglik = heads_w * F.logsigmoid(w_draws) + tails_w * F.logsigmoid(-w_draws)
logit_log_evidence_mc = number(torch.logsumexp(w_loglik, 0)) - math.log(len(w_draws))
w_plot = torch.linspace(-3, 3.5, 261)
joint_plot = logit_log_joint(w_plot).exp()
posterior_w_plot = (logit_log_joint(w_plot) - logit_log_evidence).exp()
fig, ax = plt.subplots(figsize=(8.8, 4.0))
ax.fill_between(w_plot, 0, joint_plot * 1e10, color="#E5E7EB")
ax.plot(w_plot, joint_plot * 1e10, color=C["p"])
ax.set(xlabel="w, the coin's logit (r = sigmoid(w))", ylabel="p(D, w) × 1e10")
ax.set_title("The shaded area is p(D). No formula gives it.", loc="left")
savefig("wall-logit-joint")
fixtures["story"] = {"wall": dict(
    heads=heads_w, tails=tails_w, conj_log_evidence=conj_log_evidence,
    logit_prior=[0., 1.], logit_log_evidence_grid=logit_log_evidence,
    logit_log_evidence_mc=logit_log_evidence_mc, mc_draws=100_000, mc_seed=31, grid_points=200_001,
    plot=dict(w=w_plot.tolist(), joint_e10=(joint_plot * 1e10).tolist(), posterior=posterior_w_plot.tolist()))}
print(f"Beta(5, 3) prior: log p(D) = {conj_log_evidence:.4f}, from the formula B(22,16)/B(5,3)")
print(f"Normal prior on the logit: log p(D) = {logit_log_evidence:.4f} on a 200,001-point grid; "
      f"{logit_log_evidence_mc:.4f} from 100,000 prior draws")

three_target = dist.MixtureSameFamily(dist.Categorical(probs=torch.tensor([.7, .3])),
    dist.Normal(torch.tensor([-2., 2.]), torch.tensor([1., 1.])))
def grid_kl(a, b, grid):
    # KL(a || b) by the trapezoid rule on a fixed grid
    la, lb = a.log_prob(grid), b.log_prob(grid)
    return torch.trapezoid(la.exp() * (la - lb), grid)
# Laplace: the highest peak, then the curvature there
peak = torch.tensor(-1.0, requires_grad=True)
peak_opt = torch.optim.Adam([peak], lr=.05)
for _ in range(2000):
    loss = -three_target.log_prob(peak)
    peak_opt.zero_grad(); loss.backward(); peak_opt.step()
mode = peak.detach()
curvature = torch.autograd.functional.hessian(lambda t: three_target.log_prob(t), mode)
laplace = dist.Normal(mode, (-curvature).rsqrt())
# MCMC: random-walk Metropolis
torch.manual_seed(7)
state = torch.tensor(0.); state_lp = three_target.log_prob(state); chain = []; accepted = 0
for _ in range(20_000):
    proposal = state + 2.0 * torch.randn(())
    proposal_lp = three_target.log_prob(proposal)
    if torch.log(torch.rand(())) < proposal_lp - state_lp:
        state, state_lp, accepted = proposal, proposal_lp, accepted + 1
    chain.append(number(state))
chain = np.array(chain)
# VI: minimise KL(q || p) on a grid
grid3 = torch.linspace(-8, 8, 2001)
m3 = torch.nn.Parameter(torch.tensor(-.5)); rho3 = torch.nn.Parameter(torch.tensor(0.))
opt3 = torch.optim.Adam([m3, rho3], lr=.05)
for _ in range(600):
    loss = grid_kl(dist.Normal(m3, rho3.exp()), three_target, grid3)
    opt3.zero_grad(); loss.backward(); opt3.step()
vi3 = dist.Normal(m3.detach(), rho3.detach().exp())
x3 = torch.linspace(-6, 6, 241)
hist, edges = np.histogram(chain, bins=np.linspace(-6, 6, 49), density=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, title in zip(axes, ["Laplace", "MCMC", "VI"]):
    ax.fill_between(x3, 0, three_target.log_prob(x3).exp(), color="#D9F2EF")
    ax.set_title(title, loc="left")
axes[0].plot(x3, laplace.log_prob(x3).exp(), color=C["noise"])
axes[1].hist(chain, bins=edges, density=True, color=C["noise"], alpha=.5)
axes[2].plot(x3, vi3.log_prob(x3).exp(), color=C["q"])
savefig("three-ways")
fixtures["story"]["three"] = dict(weights=[.7, .3], means=[-2., 2.], scales=[1., 1.],
    laplace=[number(laplace.loc), number(laplace.scale)],
    mcmc=dict(steps=20_000, step_size=2.0, seed=7, acceptance=accepted / 20_000,
              mean=float(chain.mean()), hist=hist.tolist(), edges=edges.tolist()),
    vi=[number(vi3.loc), number(vi3.scale)], true_mean=.7 * -2 + .3 * 2,
    plot=dict(x=x3.tolist(), p=three_target.log_prob(x3).exp().tolist(),
              laplace=laplace.log_prob(x3).exp().tolist(), vi=vi3.log_prob(x3).exp().tolist()))
print("Laplace:", fixtures["story"]["three"]["laplace"], "· MCMC mean", round(float(chain.mean()), 3),
      "acceptance", accepted / 20_000, "· VI:", fixtures["story"]["three"]["vi"])

fruit_names = ["apple", "banana", "clementine"]
fruit_q = np.array([.5, .4, .1])
fruit_p = np.array([.5, .1, .4])
fruit_ratio = fruit_p / fruit_q
fruit_terms = fruit_p * np.log(fruit_ratio)
fruit_kl = float(fruit_terms.sum())
library = number(dist.kl_divergence(dist.Categorical(probs=torch.tensor(fruit_p)),
                                    dist.Categorical(probs=torch.tensor(fruit_q))))
assert abs(fruit_kl - library) < 1e-12
weather_p = np.array([.5, .25, .25])
weather_q = np.array([.25, .5, .25])
entropy_bits = float(-(weather_p * np.log2(weather_p)).sum())
cross_entropy_bits = float(-(weather_p * np.log2(weather_q)).sum())
assert abs((cross_entropy_bits - entropy_bits) - float((weather_p * np.log2(weather_p / weather_q)).sum())) < 1e-12
coin_08, coin_05 = dist.Bernoulli(torch.tensor(.8)), dist.Bernoulli(torch.tensor(.5))
kl_coin = number(dist.kl_divergence(coin_08, coin_05))
kl_coin_reverse = number(dist.kl_divergence(coin_05, coin_08))
torch.manual_seed(5)
toss_sim = (torch.rand(20_000, 30) < .8).double()
llr = (toss_sim * math.log(.8 / .5) + (1 - toss_sim) * math.log(.2 / .5)).cumsum(1)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
xs = np.arange(3)
axes[0].bar(xs - .2, fruit_q, .4, color=C["p"], label="year 1, q")
axes[0].bar(xs + .2, fruit_p, .4, color=C["q"], label="year 2, p")
axes[0].set_xticks(xs, fruit_names); axes[0].legend(frameon=False)
n = np.arange(31)
axes[1].semilogy(n, np.exp(n * kl_coin), color=C["loss"])
axes[1].set(xlabel="tosses of the 0.8 coin", ylabel="p(D | 0.8) / p(D | 0.5)")
savefig("kl-intuition")
fixtures["story"]["fruit"] = dict(names=fruit_names, q=fruit_q.tolist(), p=fruit_p.tolist(),
    ratio=fruit_ratio.tolist(), log_ratio=np.log(fruit_ratio).tolist(), terms=fruit_terms.tolist(),
    kl=fruit_kl, ratio_mean=float(fruit_ratio.mean()))
fixtures["story"]["bits"] = dict(p=weather_p.tolist(), q=weather_q.tolist(), entropy=entropy_bits,
    cross_entropy=cross_entropy_bits, kl=cross_entropy_bits - entropy_bits,
    names=["sunny", "cloudy", "rainy"], right_code=["0", "10", "11"], wrong_code=["10", "0", "11"])
fixtures["story"]["coinkl"] = dict(kl=kl_coin, reverse=kl_coin_reverse,
    x10=math.exp(10 * kl_coin), x30=math.exp(30 * kl_coin),
    simulated_mean_llr_30=float(llr[:, -1].mean()), simulated_sequences=20_000, simulation_seed=5)
print(f"Fruit: KL(p||q) = {fruit_kl:.4f} nats (library {library:.4f})")
print(f"Weather: H(p) = {entropy_bits:.2f} bits, H(p, q) = {cross_entropy_bits:.2f} bits, "
      f"KL = {cross_entropy_bits - entropy_bits:.2f} bits")
print(f"Coins: KL(0.8 || 0.5) = {kl_coin:.4f}, reverse {kl_coin_reverse:.4f}; "
      f"e^(10 KL) = {math.exp(10 * kl_coin):.1f}, e^(30 KL) = {math.exp(30 * kl_coin):.0f}; "
      f"simulated mean log ratio after 30 = {float(llr[:, -1].mean()):.3f} (30 KL = {30 * kl_coin:.3f})")

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

def bump_mixture(width):
    return dist.MixtureSameFamily(dist.Categorical(probs=torch.tensor([.7, .3])),
        dist.Normal(torch.tensor([-2., 2.]), torch.tensor([width, width])))
grid_fr = torch.linspace(-8, 8, 2001)
x_fr = torch.linspace(-6, 6, 241)
def fit_direction(target, direction, start, steps=300, lr=.05, every=2):
    m = torch.nn.Parameter(torch.tensor(start[0]))
    rho = torch.nn.Parameter(torch.tensor(math.log(start[1])))
    opt = torch.optim.Adam([m, rho], lr=lr); frames = []
    for step in range(steps + 1):
        q = dist.Normal(m, rho.exp())
        loss = grid_kl(q, target, grid_fr) if direction == "reverse" else grid_kl(target, q, grid_fr)
        if step % every == 0:
            frames.append([step, number(m), number(rho.exp()), number(loss)])
        if step == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return frames
fwdrev = {"x": x_fr.tolist(), "start": {"fwd": [.5, .6], "revl": [-.5, .6], "revr": [.5, .6]}, "steps": 300, "lr": .05}
for width in [1.0, 0.5]:
    target = bump_mixture(width)
    runs = {"fwd": fit_direction(target, "forward", (.5, .6)),
            "revl": fit_direction(target, "reverse", (-.5, .6)),
            "revr": fit_direction(target, "reverse", (.5, .6))}
    fwdrev[str(width)] = dict(p=target.log_prob(x_fr).exp().tolist(), **runs)
    print(f"bump width {width}: " + "; ".join(
        f"{k}: N({v[-1][1]:.2f}, {v[-1][2]:.2f}^2), KL {v[-1][3]:.3f}" for k, v in runs.items()))
fixtures["story"]["fwdrev"] = fwdrev
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
narrow = bump_mixture(.5)
for ax, key, title in zip(axes, ["fwd", "revl", "revr"], ["KL(p||q)", "KL(q||p), start left", "KL(q||p), start right"]):
    _, m_, s_, k_ = fwdrev["0.5"][key][-1]
    ax.fill_between(x_fr, 0, narrow.log_prob(x_fr).exp(), color="#D9F2EF")
    ax.plot(x_fr, dist.Normal(m_, s_).log_prob(x_fr).exp(), color=C["q"])
    ax.set_title(f"{title} = {k_:.3f}", loc="left")
savefig("forward-reverse")

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

m_o = torch.nn.Parameter(torch.tensor(4.)); rho_o = torch.nn.Parameter(torch.tensor(math.log(.5)))
opt_o = torch.optim.Adam([m_o, rho_o], lr=.05); olive = []
for step in range(201):
    loss = dist.kl_divergence(dist.Normal(m_o, rho_o.exp()), p)
    olive.append([step, number(m_o), number(rho_o.exp()), number(loss)])
    if step == 200:
        break
    opt_o.zero_grad(); loss.backward(); opt_o.step()
fixtures["story"]["olive"] = dict(start=[4., .5], lr=.05, steps=200, frames=olive)
print("start", olive[0], "· step 100", olive[100], "· end", olive[-1])

rho_2d = .9
p2d = dist.MultivariateNormal(torch.zeros(2), torch.tensor([[1., rho_2d], [rho_2d, 1.]]))
def fit_2d(full, steps=1500):
    mean = torch.nn.Parameter(torch.tensor([.5, -.5]))
    raw = torch.nn.Parameter(torch.zeros(3 if full else 2))
    opt = torch.optim.Adam([mean, raw], lr=.05)
    for _ in range(steps):
        if full:
            L = torch.stack([torch.stack([raw[0].exp(), torch.tensor(0.)]),
                             torch.stack([raw[1], raw[2].exp()])])
        else:
            L = torch.diag(raw.exp())
        q = dist.MultivariateNormal(mean, scale_tril=L)
        loss = dist.kl_divergence(q, p2d)
        opt.zero_grad(); loss.backward(); opt.step()
    return q, number(loss)
q_diag, kl_diag = fit_2d(False)
q_full, kl_full = fit_2d(True)
analytic_sd = math.sqrt(1 - rho_2d ** 2)
analytic_kl = .5 * (math.log(1 - rho_2d ** 2) - 2 * math.log(1 - rho_2d ** 2))
diag_sd = q_diag.covariance_matrix.diagonal().sqrt().tolist()
assert max(abs(v - analytic_sd) for v in diag_sd) < 2e-3 and abs(kl_diag - analytic_kl) < 1e-4 and kl_full < 1e-4
fixtures["story"]["twod"] = dict(rho=rho_2d, diag_sd=diag_sd, diag_kl=kl_diag, full_kl=kl_full,
    analytic_sd=analytic_sd, analytic_kl=analytic_kl)
print(f"diagonal q: widths {diag_sd[0]:.4f}, {diag_sd[1]:.4f} (analytic {analytic_sd:.4f}), KL {kl_diag:.4f} "
      f"(analytic {analytic_kl:.4f}); full q: KL {kl_full:.2e}")

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

def mixture_kl_quadrature(q):
    g = torch.linspace(-10, 12, 20_001)
    lq = q.log_prob(g)
    return number(torch.trapezoid(lq.exp() * (lq - mixture.log_prob(g)), g))
def fit_mixture(use_rsample, steps=400, draws=5000, seed=4004, dense=120, every=5):
    torch.manual_seed(seed)
    m = torch.nn.Parameter(torch.tensor(6.)); rho = torch.nn.Parameter(torch.tensor(math.log(.1)))
    opt = torch.optim.SGD([m, rho], lr=.05); frames = []
    for step in range(steps + 1):
        q = dist.Normal(m, rho.exp())
        u_s = q.rsample((draws,)) if use_rsample else q.sample((draws,))
        loss = (q.log_prob(u_s) - mixture.log_prob(u_s)).mean()
        if step <= dense or step % every == 0:  # dense frames while q moves, for the slide animation
            frames.append([step, number(m), number(rho.exp()), number(loss),
                           mixture_kl_quadrature(dist.Normal(m.detach(), rho.detach().exp()))])
        if step == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return frames
mix_rsample = fit_mixture(True)
mix_sample = fit_mixture(False)
x_mix = torch.linspace(-4, 8, 241)
fixtures["story"]["mixfit"] = dict(start=[6., .1], optimiser="SGD", lr=.05, steps=400, draws=5000, seed=4004,
    rsample=mix_rsample, sample=mix_sample, x=x_mix.tolist(), p=mixture.log_prob(x_mix).exp().tolist())
for name, frames in [("sample()", mix_sample), ("rsample()", mix_rsample)]:
    _, m_, s_, est, exact = frames[-1]
    print(f"{name}: final N({m_:.3f}, {s_:.3f}^2), MC KL {est:.3f}, quadrature KL {exact:.3f}")
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.fill_between(x_mix, 0, mixture.log_prob(x_mix).exp(), color="#D9F2EF")
for frames, style, label in [(mix_sample, "--", "sample(): stuck"), (mix_rsample, "-", "rsample(): fitted")]:
    _, m_, s_, _, _ = frames[-1]
    ax.plot(x_mix, dist.Normal(m_, s_).log_prob(x_mix).exp(), color=C["q"], ls=style, label=label)
ax.set(ylim=(0, 1.2), xlabel="u")
ax.legend(frameon=False)
savefig("mixture-fit")

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

torch.manual_seed(3301)
m_w = torch.nn.Parameter(torch.tensor(0.)); rho_w = torch.nn.Parameter(torch.tensor(0.))
opt_w = torch.optim.Adam([m_w, rho_w], lr=.05)
w_quad = torch.linspace(-8, 8, 16_001)
def logit_exact_elbo(m, s):
    q = dist.Normal(torch.tensor(m), torch.tensor(s)); lq = q.log_prob(w_quad)
    return number(torch.trapezoid(lq.exp() * (logit_log_joint(w_quad) - lq), w_quad))
logit_frames = []
for step in range(601):
    q = dist.Normal(m_w, rho_w.exp())
    w_s = q.rsample((256,))
    elbo_estimate = (logit_log_joint(w_s) - q.log_prob(w_s)).mean()
    if step <= 60 or step % 10 == 0:  # dense frames while q moves, for the slide animation
        logit_frames.append([step, number(m_w), number(rho_w.exp()), number(elbo_estimate),
                             logit_exact_elbo(number(m_w), number(rho_w.exp()))])
    if step == 600:
        break
    opt_w.zero_grad(); (-elbo_estimate).backward(); opt_w.step()
_, m_end, s_end, _, elbo_end = logit_frames[-1]
logit_gap = logit_log_evidence - elbo_end
assert logit_gap > -1e-6
theta = torch.linspace(.2, .9, 281)
w_of_theta = torch.log(theta) - torch.log1p(-theta)
jacobian = 1 / (theta * (1 - theta))
q_theta = dist.Normal(m_end, s_end).log_prob(w_of_theta).exp() * jacobian
posterior_theta = (logit_log_joint(w_of_theta) - logit_log_evidence).exp() * jacobian
fixtures["story"]["logitvi"] = dict(start=[0., 1.], lr=.05, steps=600, draws=256, seed=3301,
    frames=logit_frames, final=dict(m=m_end, s=s_end, exact_elbo=elbo_end,
    log_evidence_grid=logit_log_evidence, gap=logit_gap),
    theta=dict(x=theta.tolist(), q=q_theta.tolist(), posterior=posterior_theta.tolist()))
print(f"q(w) = N({m_end:.3f}, {s_end:.3f}^2); exact ELBO {elbo_end:.4f}; grid log p(D) {logit_log_evidence:.4f}; "
      f"gap = KL(q || posterior) = {logit_gap:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].fill_between(w_plot, 0, posterior_w_plot, color="#D9F2EF")
axes[0].plot(w_plot, dist.Normal(0., 1.).log_prob(w_plot).exp(), color=C["q"], ls=":", label="q at the start (the prior)")
axes[0].plot(w_plot, dist.Normal(m_end, s_end).log_prob(w_plot).exp(), color=C["q"], label="q after 600 steps")
axes[0].set(xlabel="w"); axes[0].legend(frameon=False)
axes[1].fill_between(theta, 0, posterior_theta, color="#D9F2EF")
axes[1].plot(theta, q_theta, color=C["q"])
axes[1].set(xlabel="r = sigmoid(w)")
savefig("logit-vi")

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
