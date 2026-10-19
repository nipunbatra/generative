"""Independent exact calculations and autograd evidence for the probability detours."""
from pathlib import Path
import json, math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, value):
    assert value, name
    checks.append(dict(name=name,ok=True))
toy=json.loads((ROOT/'results/pedagogy.json').read_text())['toy']
joint=np.array(toy['prior'])*toy['likelihood']
for i,s in enumerate(toy['states']):
    q=np.array(s['q']);w=joint/q
    check(f'Evidence reweighting state {i}',np.isclose(q@w,toy['evidence']))
    check(f'Expected log weight is ELBO state {i}',np.isclose(q@np.log(w),s['elbo']))
    check(f'Jensen inequality state {i}',q@np.log(w)<=math.log(q@w)+1e-12)
check('Posterior makes importance weight constant',np.ptp(joint/np.array(toy['posterior']))<1e-12)
p=np.array([.6,.3,.1]);g=np.array([.1,.3,.6]);bayes=[]
for pi in [.1,.5,.9]:
    d=pi*p/(pi*p+(1-pi)*g)
    derivative=pi*p/d-(1-pi)*g/(1-d)
    check(f'Bayes maximizes weighted source log likelihood pi={pi}',np.max(abs(derivative))<1e-12)
    second=-pi*p/d**2-(1-pi)*g/(1-d)**2
    check(f'Strict concavity pi={pi}',np.all(second<0))
    bayes.append(dict(pi=pi,optimal=d.tolist()))
check('Half at one bin does not imply equal distributions',p[1]==g[1] and not np.allclose(p,g))
def KL(a,b):return float(np.sum(a*np.log(a/b)))
jsrows=[]
for t in np.linspace(0,1,101):
    gen=(1-t)*g+t*p;m=(p+gen)/2;d=p/(p+gen)
    js=(KL(p,m)+KL(gen,m))/2;v=float(p@np.log(d)+gen@np.log(1-d))
    assert abs(v-(-math.log(4)+2*js))<1e-12 and -1e-12<=js<=math.log(2)+1e-12
    if t in [0,.5,1]:jsrows.append(dict(t=float(t),js=js,value=v,first_kl=KL(p,m),second_kl=KL(gen,m)))
check('JS identity and bounds on 101 model mixtures',True)
check('Matched model gives JS zero and value -log4',abs(jsrows[-1]['js'])<1e-12 and abs(jsrows[-1]['value']+math.log(4))<1e-12)
gradients=[]
for a in [-6.,-4.,0.,6.]:
    logit=torch.tensor(a,dtype=torch.float64,requires_grad=True)
    sat=torch.nn.functional.logsigmoid(-logit)
    ns=-torch.nn.functional.logsigmoid(logit)
    sat_grad=torch.autograd.grad(sat,logit,retain_graph=True)[0].item()
    ns_grad=torch.autograd.grad(ns,logit)[0].item();d=torch.sigmoid(logit).item()
    check(f'Loss-to-logit autograd a={a}',abs(sat_grad+d)<1e-12 and abs(ns_grad-(d-1))<1e-12)
    gradients.append(dict(logit=a,probability=d,minimax=sat_grad,non_saturating=ns_grad))
for mu in [-3,0,2,3]:
    x=np.linspace(-5,5,101);data=np.exp(-x*x/2)/math.sqrt(2*math.pi);model=np.exp(-(x-mu)**2/2)/math.sqrt(2*math.pi)
    expected=data/(data+model);sigmoid=1/(1+np.exp(mu*x-mu*mu/2))
    check(f'Gaussian density-ratio curve mu={mu}',np.max(abs(expected-sigmoid))<1e-12)
fixtures=dict(toy=toy,bayes=bayes,js=jsrows,gradients=gradients,provenance='Exact illustrative probabilities and independently checked autograd; not neural training outputs.')
(ROOT/'results/probability.json').write_text(json.dumps(fixtures,indent=2))
(ROOT/'output/rebuild-qa/probability-science.json').write_text(json.dumps(dict(checks=checks),indent=2))
print(f'{len(checks)} independent probability checks passed')
