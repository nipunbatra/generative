"""Executed semantic checks: shapes, loss reductions and gradient boundaries."""
from pathlib import Path
import json,subprocess,math
import torch
from torch import nn
from torch.nn import functional as F
from models import TinyVAE,make_image_gan,vae_step,sample_vae,gan_step
from train import glyphs
R=Path(__file__).resolve().parents[1]
def validate():
 torch.set_num_threads(2);torch.manual_seed(37);checks=[]
 def ok(label,condition):
  assert condition,label
  checks.append(label)
 fixtures=json.loads(subprocess.check_output(['node',str(R/'src/math.mjs')]))
 mu=torch.tensor(1.,dtype=torch.float64,requires_grad=True);v=torch.tensor(math.log(.25),dtype=torch.float64,requires_grad=True);eps=-.4;z=mu+torch.exp(v/2)*eps;loss=z*z;loss.backward()
 ok('pathwise autograd matches shared module',abs(mu.grad.item()-fixtures['reparam']['dmu'])<1e-10 and abs(v.grad.item()-fixtures['reparam']['dlogvar'])<1e-10)
 m=TinyVAE()
 for b in [3,17]:
  x=glyphs(b);logits,mu,lv=m(x,eps=torch.zeros(b,2))
  ok(f'VAE shapes B={b}',logits.shape==(b,784) and mu.shape==lv.shape==(b,2))
  recon=F.binary_cross_entropy_with_logits(logits,x.flatten(1),reduction='none').sum(1);kl=.5*(mu.square()+lv.exp()-1-lv).sum(1)
  manual=torch.stack([F.binary_cross_entropy_with_logits(logits[i],x[i].flatten(),reduction='sum')+.5*(mu[i].square()+lv[i].exp()-1-lv[i]).sum() for i in range(b)]).mean()
  ok(f'VAE reduction semantics B={b}',torch.allclose((recon+kl).mean(),manual))
  G,D=make_image_gan();fake=G(torch.randn(b,32));l=D(fake);ok(f'GAN and target shapes B={b}',fake.shape==(b,784) and l.shape==torch.ones_like(l).shape==(b,1))
 # Prove generation never calls encode.
 original=m.encode
 def forbidden(*args):raise AssertionError('generation called encoder')
 m.encode=forbidden;sample_vae(m,3);m.encode=original;ok('prior generation bypasses encoder',True)
 G,D=make_image_gan();og=torch.optim.Adam(G.parameters(),lr=.001);od=torch.optim.Adam(D.parameters(),lr=.001);real=glyphs(8).flatten(1)
 snap=lambda net:[p.detach().clone() for p in net.parameters()]
 delta=lambda old,net:max((a-p).abs().max().item() for a,p in zip(old,net.parameters()))
 a,b=snap(G),snap(D);og.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True);fake=G(torch.randn(8,32));lr=D(real);lf=D(fake.detach());ld=F.binary_cross_entropy_with_logits(lr,torch.ones_like(lr))+F.binary_cross_entropy_with_logits(lf,torch.zeros_like(lf));ld.backward();od.step()
 ok('D phase leaves G parameters and gradients unchanged',delta(a,G)==0 and all(p.grad is None for p in G.parameters()));ok('D phase updates D',delta(b,D)>0)
 phases=[{'phase':'D','deltaG':delta(a,G),'deltaD':delta(b,D),'gGrad':0.,'loss':float(ld.detach())}]
 a,b=snap(G),snap(D);od.zero_grad(set_to_none=True);og.zero_grad(set_to_none=True);D.requires_grad_(False);fake=G(torch.randn(8,32));lf=D(fake);lg=F.binary_cross_entropy_with_logits(lf,torch.ones_like(lf));lg.backward();norm=sum(p.grad.square().sum().item() for p in G.parameters())**.5;og.step()
 ok('frozen D still transmits nonzero gradient to G',norm>1e-7);ok('G phase changes only G',delta(a,G)>0 and delta(b,D)==0 and all(p.grad is None for p in D.parameters()))
 phases.append({'phase':'G','deltaG':delta(a,G),'deltaD':delta(b,D),'gGrad':norm,'loss':float(lg.detach())});D.requires_grad_(True)
 og.zero_grad(set_to_none=True);D.requires_grad_(False);broken=D(G(torch.randn(8,32)).detach());ok('broken G detach removes the gradient graph',not broken.requires_grad and all(p.grad is None for p in G.parameters()));D.requires_grad_(True)
 # Recompute the two-pixel loss from PyTorch, independently of JS arithmetic.
 target=torch.tensor([1.,0.],dtype=torch.float64);pi=torch.tensor([.75,.25],dtype=torch.float64);mu=torch.tensor([1.,-1.],dtype=torch.float64);lv=torch.tensor([math.log(.25),0.],dtype=torch.float64)
 total=F.binary_cross_entropy_with_logits(torch.logit(pi),target,reduction='sum')+.5*(mu.square()+lv.exp()-1-lv).sum();ok('two-pixel PyTorch loss agrees with shared fixtures',abs(total.item()-fixtures['vae']['total'])<1e-10)
 (R/'assets/recorded/phases.json').write_text(json.dumps({'seed':37,'phases':phases},indent=2))
 report={'passed':len(checks),'checks':checks,'torch':torch.__version__,'fixtures_source':'src/math.mjs','executed':True};(R/'qa/lab-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':validate()
