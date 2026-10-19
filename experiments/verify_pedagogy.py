"""Independent arithmetic, provenance and checkpoint checks for edition 3."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from train_mnist import data,Model
R=Path(__file__).resolve().parents[1]
e=json.loads((R/'results/pedagogy.json').read_text());checks=[]
def check(name,ok):
    checks.append({'name':name,'ok':bool(ok)})
    assert ok,name
q=np.array([.8,.2]);p=np.array([.5,.5])
check('Discrete KL q to p',abs(np.sum(q*np.log(q/p))-.19274475702175753)<1e-12)
check('Discrete KL p to q',abs(np.sum(p*np.log(p/q))-.22314355131420976)<1e-12)
t=e['toy'];prior=np.array(t['prior']);lik=np.array(t['likelihood'])
check('Finite evidence',abs(prior@lik-.48)<1e-12)
for i,s in enumerate(t['states']):
    q=np.array(s['q']);post=np.array(t['posterior']);elbo=np.sum(q*np.log(prior*lik/q));gap=np.sum(q*np.log(q/post))
    check(f'ELBO identity state {i}',abs(math.log(.48)-elbo-gap)<1e-12 and abs(elbo-s['elbo'])<1e-12 and gap>=-1e-12)
check('Exact posterior closes the gap',abs(t['states'][-1]['gap'])<1e-12)
z=np.arange(-12+.0025,12,.005)
normal=lambda x,m,s:np.exp(-.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
p=.5*normal(z,-2,.55)+.5*normal(z,2,.55);vals=[]
for m,s in [(0,2.1),(-2,.65)]:
    q=normal(z,m,s);vals.append([float(np.sum(p*np.log(p/q))*.005),float(np.sum(q*np.log(q/p))*.005)])
check('Candidate fits exhibit different KL preferences',vals[0][0]<vals[1][0] and vals[0][1]>vals[1][1])
train,_,test,_,hashes=data();check('Data hashes unchanged',hashes==e['data_sha256'])
ind=torch.bernoulli(train.mean(0).expand(8,-1),generator=torch.Generator().manual_seed(667))
check('Independent pixel samples reproduce exactly',np.array_equal(ind.numpy(),e['opening']['independent']))
for row in e['beta']:
    cp=R/f'models/pedagogy-beta{int(row["beta"])}.pt'
    check(f'Checkpoint beta {row["beta"]} hash',hashlib.sha256(cp.read_bytes()).hexdigest()==row['checkpoint_sha256'])
    ck=torch.load(cp,weights_only=True,map_location='cpu');m=Model(2,True);m.load_state_dict(ck['state_dict']);m.eval()
    with torch.no_grad():
        mu,_=m.encode(test[:8]);recon=m.dec(mu).sigmoid();zs=torch.randn(8,2,generator=torch.Generator().manual_seed(669));pr=m.dec(zs).sigmoid()
    check(f'Beta {row["beta"]} recon and prior parity',np.allclose(recon.numpy(),row['recon'],atol=1e-6) and np.allclose(pr.numpy(),row['prior'],atol=1e-6))
mu=torch.tensor(2.,requires_grad=True);std=torch.tensor(.5,requires_grad=True)
check('sample has no pathwise graph',not torch.distributions.Normal(mu,std).sample().requires_grad)
(3*(mu+std*(-.8))).backward();check('Pathwise gradients',abs(mu.grad.item()-3)<1e-6 and abs(std.grad.item()+2.4)<1e-6)
mu=torch.tensor(2.,requires_grad=True);std=torch.tensor(.5,requires_grad=True);torch.manual_seed(667)
v=torch.distributions.Normal(mu,std).rsample();v.backward();check('rsample keeps pathwise graph',mu.grad.item()==1 and std.grad is not None)
report={'checks':checks,'kl_candidate_scores':vals,'note':'Independent reproduction from cached data and saved checkpoints; no training rerun.'}
(R/'output/rebuild-qa/pedagogy-science.json').write_text(json.dumps(report,indent=2))
print(f'{len(checks)} pedagogy science checks passed; KL candidates {vals}')
