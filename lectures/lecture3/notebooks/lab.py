"""Offline AE/VAE teaching computations. Saved model weights are replayed, not retrained."""
from pathlib import Path
import copy,json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
class Encoder(nn.Module):
    def __init__(self,d,variational):
        super().__init__()
        self.body=nn.Sequential(nn.Flatten(),nn.Linear(784,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
        self.mu=nn.Linear(256,d)
        self.logvar=nn.Linear(256,d) if variational else None
    def forward(self,x):
        h=self.body(x)
        return self.mu(h),None if self.logvar is None else self.logvar(h)
class Decoder(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,784))
    def forward(self,z): return self.net(z)
class AutoEncoder(nn.Module):
    def __init__(self,d=2,variational=False):
        super().__init__();self.encoder=Encoder(d,variational);self.decoder=Decoder(d)
    def forward(self,x,eps=None):
        mu,logvar=self.encoder(x)
        if logvar is None:z=mu
        else:
            if eps is None:eps=torch.randn_like(mu)
            z=mu+torch.exp(.5*logvar)*eps
        return self.decoder(z),mu,logvar,z

def data():
    a=np.load(ROOT/'data/mnist-snapshot.npz')
    return {k:torch.from_numpy(a[k]).float() if k in ['train','test'] else torch.from_numpy(a[k]) for k in a.files}
def model(name):
    d=16 if '16' in name else 2
    m=AutoEncoder(d,name.startswith('vae'))
    m.load_state_dict(torch.load(ROOT/f'models/{name}.pt',map_location='cpu',weights_only=True));m.eval()
    return m

def terms(logits,x,mu,logvar):
    recon=F.binary_cross_entropy_with_logits(logits,x.flatten(1),reduction='none').sum(1)
    kl=torch.zeros_like(recon) if logvar is None else .5*(mu.square()+logvar.exp()-1-logvar).sum(1)
    return recon,kl

def inspected_step(m,x,seed=667,eta=1e-4):
    """A real SGD update on a copied checkpoint. Fixed epsilon is only a before/after control."""
    m=copy.deepcopy(m).double();x=x.double()
    g=torch.Generator().manual_seed(seed);d=m.encoder.mu.out_features
    eps=torch.randn(len(x),d,generator=g,dtype=torch.float64)
    logits,mu,lv,z=m(x,eps);r,k=terms(logits,x,mu,lv);loss=(r+k).mean()
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);loss.backward()
    norms={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
    with torch.no_grad():
        for p in m.parameters():p-=eta*p.grad
        logits1,mu1,lv1,_=m(x,eps);r1,k1=terms(logits1,x,mu1,lv1)
    changes={n:float((p.detach()-before[n]).norm()) for n,p in m.named_parameters()}
    assert all(np.isfinite(list(norms.values())))
    assert any(changes[n]>0 for n in changes if n.startswith('encoder'))
    assert any(changes[n]>0 for n in changes if n.startswith('decoder'))
    return {'batch_size':len(x),'eta':eta,'seed':seed,'before':float(loss.detach()),'after_same_noise':float((r1+k1).mean()),'recon_before':float(r.mean().detach()),'kl_before':float(k.mean().detach()),'gradient_norms':norms,'parameter_delta_norms':changes,'same_noise_note':'Fixed epsilon isolates this one update. Ordinary training draws fresh epsilon.'}

def fit_one_q(m,x,steps=80):
    """Fixed decoder, one image: optimise free μ/log variance, not network weights."""
    m=copy.deepcopy(m).double()
    for p in m.parameters():p.requires_grad_(False)
    x=x.double().reshape(1,-1)
    mu=torch.zeros(1,2,dtype=torch.float64,requires_grad=True)
    lv=torch.zeros(1,2,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([mu,lv],lr=.08)
    g=torch.Generator().manual_seed(812)
    check=torch.randn(256,2,generator=torch.Generator().manual_seed(813),dtype=torch.float64)
    history=[]
    for i in range(steps+1):
        if i%10==0:
            with torch.no_grad():
                z=mu+(.5*lv).exp()*check
                rec=F.binary_cross_entropy_with_logits(m.decoder(z),x.expand(len(z),-1),reduction='none').sum(1).mean()
                kl=.5*(mu.square()+lv.exp()-1-lv).sum()
                history.append({'step':i,'mu':mu.flatten().tolist(),'logvar':lv.flatten().tolist(),'negative_elbo_estimate':float(rec+kl)})
        if i==steps:break
        eps=torch.randn(64,2,generator=g,dtype=torch.float64)
        z=mu+(.5*lv).exp()*eps
        rec=F.binary_cross_entropy_with_logits(m.decoder(z),x.expand(len(z),-1),reduction='none').sum(1).mean()
        kl=.5*(mu.square()+lv.exp()-1-lv).sum()
        opt.zero_grad();(rec+kl).backward();opt.step()
    return history

def verify():
    dtype=torch.float64
    mu=torch.tensor([[.3,-1.2]],dtype=dtype,requires_grad=True)
    lv=torch.tensor([[-.4,.2]],dtype=dtype,requires_grad=True)
    std=(.5*lv).exp();eps=torch.tensor([[.62,-1.15]],dtype=dtype)
    z=mu+std*eps;objective=z.square().sum();objective.backward()
    assert torch.allclose(mu.grad,2*z)
    assert torch.allclose(lv.grad,z*std*eps)
    q=torch.distributions.Normal(mu.detach(),std.detach());p=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
    analytic=.5*(mu.detach().square()+lv.detach().exp()-1-lv.detach()).sum()
    library=torch.distributions.kl_divergence(q,p).sum()
    assert torch.allclose(analytic,library,atol=1e-12)
    x=torch.tensor([[1.,0.,1.,0.]],dtype=dtype);prob=torch.tensor([[.8,.1,.6,.3]],dtype=dtype)
    bce=F.binary_cross_entropy_with_logits(torch.logit(prob),x,reduction='sum')
    hand=-(x*prob.log()+(1-x)*(1-prob).log()).sum()
    assert torch.allclose(bce,hand,atol=1e-12)
    return {'gaussian_KL_library_error':float(abs(analytic-library)),'BCE_hand_error':float(abs(bce-hand)),'four_pixel_BCE':float(bce),'pathwise_grad_mu':mu.grad.tolist(),'pathwise_grad_logvar':lv.grad.tolist(),'checks':'KL/library; Bernoulli/hand; both reparameterisation derivatives'}


def reconstruction_path_check(m,x):
    """Exclude the KL: reconstruction alone must reach both q heads through z."""
    m=copy.deepcopy(m).double();x=x.double()
    eps=torch.randn(len(x),m.encoder.mu.out_features,generator=torch.Generator().manual_seed(675),dtype=torch.float64)
    logits,mu,logvar,z=m(x,eps)
    reconstruction=F.binary_cross_entropy_with_logits(logits,x.flatten(1),reduction='none').sum(1).mean()
    m.zero_grad(set_to_none=True);reconstruction.backward()
    norms={}
    for group in ['encoder.mu','encoder.logvar','decoder']:
        norms[group]=sum(float(p.grad.norm()) for name,p in m.named_parameters() if name.startswith(group) and p.grad is not None)
        assert norms[group]>0,group
    return {'objective':'reconstruction only; no prior KL','seed':675,'gradient_norm_sums':norms,'passed':True}
