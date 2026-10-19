"""Independent analytical and autograd checks for the lecture's scientific claims."""
from pathlib import Path
import json, math, hashlib
import torch
import numpy as np
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
checks={}
mu=torch.tensor(2.,requires_grad=True);sigma=torch.tensor(.5,requires_grad=True);eps=-.8
z=mu+sigma*eps;(3*z).backward();checks['reparameterization']={'z':z.item(),'dL_dmu':mu.grad.item(),'dL_dsigma':sigma.grad.item()}
assert abs(z.item()-1.6)<1e-6 and abs(mu.grad.item()-3)<1e-6 and abs(sigma.grad.item()+2.4)<1e-6
lv=torch.tensor(math.log(.25),requires_grad=True);z2=2+torch.exp(.5*lv)*eps;dz=torch.autograd.grad(z2,lv,retain_graph=True)[0].item();g=torch.autograd.grad(3*z2,lv)[0].item();assert abs(dz+.2)<1e-6 and abs(g+.6)<1e-6
checks['logvar']={'dz_dlogvar':dz,'dL_dlogvar':g}
q=torch.distributions.Normal(torch.tensor(1.),torch.tensor(.5));p=torch.distributions.Normal(torch.tensor(0.),torch.tensor(1.));kl=torch.distributions.kl_divergence(q,p).item();assert abs(kl-.81814718056)<1e-6
checks['kl']={'library':kl,'formula':.5*(1+.25-math.log(.25)-1),'at_prior':0.}
# Distribution identity checked with moment values and a deterministic Monte Carlo confirmation.
rng=np.random.default_rng(667);noise=rng.standard_normal(200000);shifted=2+.5*noise
checks['location_scale']={'sample_mean':float(shifted.mean()),'sample_variance':float(shifted.var()),'expected_mean':2,'expected_variance':.25};assert abs(shifted.mean()-2)<.005 and abs(shifted.var()-.25)<.005
# Frozen discriminator still yields input/G gradients; D gradients must remain None.
torch.manual_seed(667);G=nn.Linear(2,2);D=nn.Sequential(nn.Linear(2,1),nn.Sigmoid());[v.requires_grad_(False) for v in D.parameters()];loss=-D(G(torch.randn(8,2))).log().mean();loss.backward();assert G.weight.grad.abs().sum()>0 and all(v.grad is None for v in D.parameters())
checks['frozen_D']={'G_grad_norm':G.weight.grad.norm().item(),'D_gradients_absent':True}
# Invertible shear: forward/inverse recover points; Jacobian determinant one.
xy=torch.tensor([.4,-.7],requires_grad=True);a=.85
f=lambda z:torch.stack([z[0],z[1]+a*.6*torch.sin(1.25*z[0])]);out=f(xy);inverse=torch.stack([out[0],out[1]-a*.6*torch.sin(1.25*out[0])]);jac=torch.autograd.functional.jacobian(f,xy)
assert torch.allclose(inverse,xy) and abs(torch.det(jac).item()-1)<1e-6
checks['flow']={'inverse_error':float((inverse-xy).abs().max().detach()),'determinant':torch.det(jac).item(),'stretch2_log_inverse_jacobian':math.log(.5)}
checks['gaussian_interval_probability']=math.erf(1/math.sqrt(2));assert round(checks['gaussian_interval_probability'],3)==.683
checks['fixed_variance_mle']={'data':[-.5,.5,1.5],'mle_mean':.5,'max_log_likelihood':sum(-.5*(x-.5)**2-.5*math.log(2*math.pi) for x in [-.5,.5,1.5])}
manifest=json.loads((ROOT/'metrics/mnist.json').read_text())
for name,v in manifest['models'].items():assert hashlib.sha256((ROOT/f'models/{name}.pt').read_bytes()).hexdigest()==v['checkpoint_sha256']
checks['checkpoint_hashes_match']=True
fit=json.loads((ROOT/'metrics/mixture-fit.json').read_text());assert abs(sum(fit['weights'])-1)<1e-8;assert min(np.diff(fit['log_likelihood_history']))>-1e-5
checks['EM_fit']={'iterations':fit['iterations'],'likelihood_nondecreasing':True,'weights_sum':sum(fit['weights'])}
(ROOT/'metrics/analytical-checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
