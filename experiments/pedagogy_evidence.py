"""Edition 3: measured opening baselines, controlled beta sweep, exact teaching fixtures."""
from train_mnist import *

def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    train,labels,test,test_labels,hashes=data()
    rng=torch.Generator().manual_seed(667)
    marginal=train.mean(0)
    independent=torch.bernoulli(marginal.expand(8,-1),generator=rng)
    montage(train[:8],ROOT/'figures/pedagogy-replay.png',8,4)
    montage(independent,ROOT/'figures/pedagogy-independent.png',8,4)
    montage(marginal[None],ROOT/'figures/pedagogy-pixel-mean.png',1,4)
    p=np.array([.2,.5,.3]);lik=np.array([.1,.8,.2]);evidence=float(p@lik);posterior=p*lik/evidence
    qs=[np.array([.4,.3,.3]),np.array([.15,.7,.15]),posterior]
    toy=[]
    for q in qs:
        recon=float(q@np.log(lik));kl=float(np.sum(q*np.log(q/p)))
        gap=float(np.sum(q*np.log(q/posterior)));elbo=recon-kl
        assert abs(np.log(evidence)-elbo-gap)<1e-12
        toy.append(dict(q=q.tolist(),recon=recon,prior_kl=kl,gap=gap,elbo=elbo))
    # Verify the autograd comparison that appears beside the persistent graph.
    mu=torch.tensor(2.,requires_grad=True);sigma=torch.tensor(.5,requires_grad=True)
    opaque=torch.distributions.Normal(mu,sigma).sample()
    eps=torch.tensor(-.8);z=mu+sigma*eps;(3*z).backward()
    assert not opaque.requires_grad and abs(mu.grad.item()-3)<1e-6 and abs(sigma.grad.item()+2.4)<1e-6
    torch.manual_seed(667);mu2=torch.tensor(2.,requires_grad=True);s2=torch.tensor(.5,requires_grad=True)
    z2=torch.distributions.Normal(mu2,s2).rsample();z2.backward()
    assert mu2.grad.item()==1 and abs(s2.grad.item()-((z2.detach()-2)/.5).item())<1e-5
    result={'seed':667,'data_sha256':hashes,'opening':{'replay':train[:8].tolist(),'independent':independent.tolist(),'marginal':marginal.tolist(),'protocol':'First eight training examples; independent Bernoulli pixels fitted to all 60,000 binary training images, seed 667. No selection.'},'toy':{'prior':p.tolist(),'likelihood':lik.tolist(),'evidence':evidence,'posterior':posterior.tolist(),'log_evidence':float(np.log(evidence)),'states':toy},'autograd':{'sample_requires_grad':opaque.requires_grad,'mu_grad':mu.grad.item(),'sigma_grad':sigma.grad.item(),'rsample_mu_grad':mu2.grad.item(),'rsample_sigma_grad':s2.grad.item()},'beta':[]}
    # Identical initialization, data order, posterior noise and evaluation draws across beta.
    # A small controlled run, distinct from the original full-data 15-epoch checkpoints.
    subset=train[:12000];epochs=12;d=2
    for beta in [0.,1.,4.]:
        torch.manual_seed(667);m=Model(d,True);opt=torch.optim.Adam(m.parameters(),lr=.001)
        loader=torch.utils.data.DataLoader(subset,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(667))
        hist=[]
        for ep in range(epochs):
            sums=np.zeros(2)
            for x in loader:
                opt.zero_grad();logits,mu,lv=m(x)
                rec=F.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(1).mean()
                kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
                (rec+beta*kl).backward();opt.step();sums+=np.array([rec.item(),kl.item()])*len(x)
            hist.append({'epoch':ep+1,'reconstruction_nats':float(sums[0]/len(subset)),'kl_nats':float(sums[1]/len(subset))})
            print('beta',beta,'epoch',ep+1,'reconstruction',round(hist[-1]['reconstruction_nats'],3),flush=True)
        m.eval();torch.manual_seed(668)
        with torch.no_grad():
            vals=[]
            for x in test.split(256):
                logits,mu,lv=m(x);rec=F.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(1)
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1);vals.append(torch.stack([rec,kl],1))
            metrics=torch.cat(vals).mean(0).tolist();mus,_=m.encode(test)
            recon=m.dec(mus[:8]).sigmoid()
            zs=torch.randn(8,d,generator=torch.Generator().manual_seed(669));prior=m.dec(zs).sigmoid()
            draws=torch.bernoulli(prior,generator=torch.Generator().manual_seed(670))
        name=f'beta{int(beta)}';checkpoint=ROOT/f'models/pedagogy-{name}.pt'
        torch.save({'state_dict':m.state_dict(),'latent_dim':d,'beta':beta,'seed':667,'epochs':epochs,'train_examples':len(subset)},checkpoint)
        for suffix,values in [('recon',recon),('prior',prior),('draws',draws)]:montage(values,ROOT/f'figures/pedagogy-{name}-{suffix}.png',8,4)
        result['beta'].append({'beta':beta,'recon':recon.tolist(),'prior':prior.tolist(),'draws':draws.tolist(),'mu':mus[:600].tolist(),'test_reconstruction_nats':metrics[0],'test_kl_nats':metrics[1],'history':hist,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    result['beta_protocol']={'train_examples':12000,'test_examples':10000,'epochs':epochs,'latent_dim':2,'batch_size':256,'learning_rate':.001,'optimizer':'Adam','seed':667,'test_seed':668,'prior_seed':669,'pixel_seed':670,'reduction':'BCE summed over 784 pixels; KL summed over 2 latent dimensions; both averaged over examples','evaluation':'One posterior sample per test example. Reconstruction images decode posterior means. Same seeds, initialization and minibatch order across beta. Single seed; instructional comparison, not a benchmark.','architecture':'Same Model class as train_mnist.py','torch':torch.__version__}
    (ROOT/'results/pedagogy.json').write_text(json.dumps(result,separators=(',',':')))
    (ROOT/'metrics/pedagogy.json').write_text(json.dumps({'beta_protocol':result['beta_protocol'],'beta':[dict(beta=x['beta'],test_reconstruction_nats=x['test_reconstruction_nats'],test_kl_nats=x['test_kl_nats'],checkpoint_sha256=x['checkpoint_sha256'],history=x['history']) for x in result['beta']],'data_sha256':hashes,'autograd':result['autograd'],'toy':result['toy']},indent=2))
    print('Saved measured and exact pedagogy evidence',flush=True)
if __name__=='__main__':main()
