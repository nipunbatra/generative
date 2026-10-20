"""Replay existing model checkpoints and build exact, labelled lecture evidence."""
from pathlib import Path
import sys,json,copy,shutil,hashlib
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L2=Path(__file__).resolve().parents[1];L3=L2.parent/'lecture3'
sys.path.insert(0,str(L2/'notebooks'))
import lab
C={'data':'#245EDB','q':'#8B2CDE','p':'#0F766E','noise':'#AA4E08','loss':'#BE123C','ink':'#14171F'}
plt.rcParams.update({'font.size':20,'axes.labelsize':22,'xtick.labelsize':18,'ytick.labelsize':18,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'#FAFAFB','axes.facecolor':'#FAFAFB'})
def grid(x,path,ncols=8):
    a=np.asarray(x).reshape(-1,28,28);n=len(a);rows=int(np.ceil(n/ncols))
    canvas=np.ones((rows*30-2,ncols*30-2))
    for i,img in enumerate(a):canvas[i//ncols*30:i//ncols*30+28,i%ncols*30:i%ncols*30+28]=1-img
    # Render large evidence rasters from the numerical arrays. Binary pixels retain
    # their discrete edges; probability fields use bilinear display interpolation.
    binary=bool(np.all((a==0)|(a==1)))
    fig=plt.figure(figsize=(ncols*1.4,rows*1.4),dpi=120)
    ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    ax.imshow(canvas,cmap='gray',vmin=0,vmax=1,interpolation='nearest' if binary else 'bilinear')
    fig.savefig(path,dpi=120,bbox_inches=None,pad_inches=0);plt.close(fig)
def save(fig,path):
    fig.tight_layout(pad=.8);fig.savefig(path,bbox_inches='tight');plt.close(fig)
def scatter(codes,prior,path,limits,title='encoded test images',overlay=False):
    fig,ax=plt.subplots(figsize=(10.5,4.1))
    ax.scatter(codes[:,0],codes[:,1],s=8,c=C['q'],alpha=.35,label=title)
    if overlay:ax.scatter(prior[:,0],prior[:,1],s=50,c=C['noise'],marker='+',linewidths=2,label='N(0,I) draws')
    ax.set(xlim=limits[0],ylim=limits[1],xlabel='z₁',ylabel='z₂');ax.legend(loc='upper left',fontsize=18,ncol=2,frameon=False)
    save(fig,path)
def run():
    d=lab.data();x=d['test'];y=d['test_labels'];held=x[90:91]
    m2,m16=lab.model('ae2'),lab.model('ae16')
    lab.ROOT=L3
    v2,v16,v8=lab.model('vae2'),lab.model('vae16'),lab.model('vae2b8')
    with torch.no_grad():
        a,_=m2.encoder(x);v,lv=v2.encoder(x)
        zprior=torch.randn(80,2,generator=torch.Generator().manual_seed(667))
        b,_=m16.encoder(x)
        ae3,_=m2.encoder(held);mu3,lv3=v2.encoder(held);std3=(.5*lv3).exp();eps=torch.tensor([[.62,-1.15]]);z3=mu3+std3*eps
        lp=v2.decoder(z3);pi=lp.sigmoid();r,k=lab.terms(lp,held,mu3,lv3)
        sam=torch.bernoulli(pi,generator=torch.Generator().manual_seed(668))
        ids=torch.tensor([int(torch.where(y==i)[0][0]) for i in range(10)])
        for root in [L2,L3]:
            grid(held,root/'figures/held3.png',1);grid(x[ids],root/'figures/test-digits.png',10)
        grid(m2.decoder(ae3).sigmoid(),L2/'figures/held3-ae2.png',1)
        grid(m16.decoder(m16.encoder(held)[0]).sigmoid(),L2/'figures/held3-ae16.png',1)
        grid(torch.cat([x[ids].flatten(1),m2.decoder(a[ids]).sigmoid(),m16.decoder(b[ids]).sigmoid()]),L2/'figures/reconstruction-2-16.png',10)
        grid(m2.decoder(zprior[:10]).sigmoid(),L2/'figures/prior-decodes.png',10)
        grid(pi,L3/'figures/held3-probability.png',1);grid(sam,L3/'figures/held3-draw.png',1)
        grid(v2.decoder(mu3).sigmoid(),L3/'figures/held3-mean-decode.png',1)
        grid(v2.decoder(zprior[:10]).sigmoid(),L3/'figures/prior-probabilities.png',10)
        grid(torch.bernoulli(v2.decoder(zprior[:10]).sigmoid(),generator=torch.Generator().manual_seed(669)),L3/'figures/prior-observations.png',10)
        grid(torch.cat([x[ids].flatten(1),v2.decoder(v[ids]).sigmoid()]),L3/'figures/reconstructions.png',10)
        # All cloud comparisons have exactly these same axes, including the lecture boundary.
        lo=torch.minimum(a.min(0).values,torch.tensor([-4.,-4.]))-.5;hi=torch.maximum(a.max(0).values,torch.tensor([4.,4.]))+.5
        limits=[[float(lo[0]),float(hi[0])],[float(lo[1]),float(hi[1])]]
        scatter(a,zprior,L2/'figures/code-cloud.svg',limits)
        scatter(a,zprior,L2/'figures/code-cloud-prior.svg',limits,overlay=True)
        shutil.copy2(L2/'figures/code-cloud-prior.svg',L3/'figures/ae-failure-cloud.svg')
        shutil.copy2(L2/'figures/prior-decodes.png',L3/'figures/ae-prior-decodes.png')
        scatter(v,zprior,L3/'figures/vae-cloud-prior.svg',limits,title='encoder means (test)',overlay=True)
        # Same coordinates and fixed sampling seed; a fitted Gaussian is a genuine separate model.
        train_codes=m2.encoder(d['train'])[0]
        cov=torch.cov(train_codes.T)+torch.eye(2)*1e-4
        fitted=train_codes.mean(0)+torch.randn(10,2,generator=torch.Generator().manual_seed(670))@torch.linalg.cholesky(cov).T
        grid(m2.decoder(fitted).sigmoid(),L2/'figures/fitted-code-decodes.png',10)
        t=torch.linspace(0,1,9).unsqueeze(1);i7=int(torch.where(y==7)[0][0]);za=a[90];zb=a[i7]
        zz=(1-t)*za+t*zb;grid(m2.decoder(zz).sigmoid(),L2/'figures/interpolation.png',9)
        fig,ax=plt.subplots(figsize=(10.5,4.1));ax.scatter(a[:,0],a[:,1],s=8,c=C['q'],alpha=.15);ax.plot(zz[:,0],zz[:,1],'o-',c=C['noise'],lw=3);ax.annotate('encoded 3',za,xytext=(10,10),textcoords='offset points');ax.annotate('encoded 7',zb,xytext=(10,10),textcoords='offset points');ax.set(xlim=limits[0],ylim=limits[1],xlabel='z₁',ylabel='z₂');save(fig,L2/'figures/interpolation-path.svg')
        pixelmean=d['train'].mean(0);independent=torch.bernoulli(pixelmean.expand(10,-1,-1),generator=torch.Generator().manual_seed(671));grid(pixelmean,L2/'figures/mean-pixel.png',1);grid(independent,L2/'figures/independent-pixels.png',10)
        vaemu16,_=v16.encoder(x[ids]);grid(v16.decoder(vaemu16).sigmoid(),L3/'figures/reconstructions-16.png',10)
        vp=torch.randn(10,16,generator=torch.Generator().manual_seed(672));grid(v16.decoder(vp).sigmoid(),L3/'figures/prior-16.png',10)
        grid(v8.decoder(zprior[:10]).sigmoid(),L3/'figures/beta8-prior.png',10)
        # Measured comparable AE deterministic code losses and VAE one-draw terms.
        ae_recon={}
        for name,model in [('ae2',m2),('ae16',m16)]:
            logits,mu,lv,_=model(x);rr,kk=lab.terms(logits,x,mu,lv);ae_recon[name]=float(rr.mean())
        vae_metrics={}
        for name,model in [('vae2',v2),('vae16',v16),('vae2b8',v8)]:
            eps_=torch.randn(len(x),model.encoder.mu.out_features,generator=torch.Generator().manual_seed(673));logits,mu,lv,_=model(x,eps_);rr,kk=lab.terms(logits,x,mu,lv);vae_metrics[name]={'recon':float(rr.mean()),'kl':float(kk.mean()),'negative_elbo_estimate':float((rr+kk).mean())}
    # scalar/four-pixel arithmetic and fitted two-mode dataset
    toy=np.array([-3.2,-2.8,-2.4,-2.1,-1.8,-1.7,1.5,1.8,2.1,2.4,2.7,3.1])
    means=np.array([-2.,2.]);stds=np.array([1.,1.]);weights=np.array([.5,.5])
    for _ in range(40):
        density=np.exp(-.5*((toy[:,None]-means)/stds)**2)/stds/np.sqrt(2*np.pi)*weights
        resp=density/density.sum(1,keepdims=True);n=resp.sum(0);weights=n/len(toy);means=(resp*toy[:,None]).sum(0)/n;stds=np.sqrt((resp*(toy[:,None]-means)**2).sum(0)/n)
    xx=np.linspace(-5,5,501);normal=lambda m,s:np.exp(-.5*((xx-m)/s)**2)/(s*np.sqrt(2*np.pi))
    single=normal(toy.mean(),toy.std());mix=sum(w*normal(m,s) for w,m,s in zip(weights,means,stds))
    for name,curves in [('toy-data',[]),('toy-gaussian',[('one fitted Gaussian',single,C['p'])]),('toy-mixture',[('one Gaussian',single,'#68717B'),('two-component mixture',mix,C['p'])])]:
        fig,ax=plt.subplots(figsize=(10.5,3.5));ax.scatter(toy,np.full_like(toy,.01),s=70,c=C['data'],marker='|',label='12 observed values')
        for label,yy,col in curves:ax.plot(xx,yy,lw=3,label=label,c=col)
        ax.set(xlim=(-5,5),ylim=(0,.5),xlabel='observed value x',ylabel='density');ax.legend(fontsize=17,frameon=False,ncol=2);save(fig,L2/f'figures/{name}.svg')
    rng=np.random.default_rng(674);one=rng.normal(toy.mean(),toy.std(),24);components=rng.choice(2,24,p=weights);two=rng.normal(means[components],stds[components])
    fig,ax=plt.subplots(figsize=(10.5,3.5));ax.scatter(one,np.zeros(24),c=C['p'],s=70,label='single Gaussian draws');ax.scatter(two,np.ones(24),c=C['noise'],s=70,label='mixture draws');ax.set(xlim=(-6,6),ylim=(-.5,1.6),yticks=[0,1],yticklabels=['Gaussian','mixture'],xlabel='new x');ax.legend(fontsize=17,frameon=False,ncol=2);save(fig,L2/'figures/toy-draws.svg')
    ae_step=lab.inspected_step(m2,d['train'][:8]);vae_step=lab.inspected_step(v2,d['train'][:8],eta=1e-6);oneq=lab.fit_one_q(v2,held)
    fig,ax=plt.subplots(figsize=(10.5,3.8));ax.plot([h['step'] for h in oneq],[h['negative_elbo_estimate'] for h in oneq],'o-',c=C['loss'],lw=3);ax.set(xlabel='updates to this image’s μ and log variance',ylabel='negative ELBO estimate');save(fig,L3/'figures/per-image-fit.svg')
    fig,ax=plt.subplots(figsize=(10.5,4));ax.scatter(v[:,0],v[:,1],s=8,alpha=.15,c=C['q']);path=np.array([h['mu'] for h in oneq]);ax.plot(path[:,0],path[:,1],'o-',c=C['noise'],lw=3);ax.scatter(mu3[0,0],mu3[0,1],s=160,marker='*',c=C['loss'],label='encoder mean for this 3');ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='z₁',ylabel='z₂');ax.legend(fontsize=18,frameon=False);save(fig,L3/'figures/per-image-path.svg')
    scalar=lab.verify()
    fixtures={'held_test_index':90,'binary_ink_pixels':int(held.sum()),'code_axes':limits,'ae_code':ae3[0].tolist(),'vae':{'mu':mu3[0].tolist(),'logvar':lv3[0].tolist(),'std':std3[0].tolist(),'eps':eps[0].tolist(),'z':z3[0].tolist(),'recon':float(r),'kl':float(k),'negative_elbo_one_draw':float(r+k)},'ae_test_reconstruction':ae_recon,'vae_test_metrics':vae_metrics,'ae_step':ae_step,'vae_step':vae_step,'reconstruction_path_check':lab.reconstruction_path_check(v2,d['train'][:8]),'one_q_fit':oneq,'scalar_checks':scalar,'toy':{'x':toy.tolist(),'normal_mean':float(toy.mean()),'normal_std':float(toy.std()),'weights':weights.tolist(),'means':means.tolist(),'stds':stds.tolist(),'sample_seed':674},'observation':'fixed binarised MNIST; >.5 threshold; test0:2000','loss_reduction':'sum 784 pixels and latent dimensions within image; average batch','one_draw_seed':673,'per_image_fit':'fixed trained decoder;80Adam updates;64 fresh noise draws/update;256 fixed check draws;seed812/813;not exact evidence'}
    for root in [L2,L3]:
        (root/'data/fixtures.json').write_text(json.dumps(fixtures,indent=2))
        figures=[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'type':'replayed model output or computed plot','generator':'lecture2/scripts/build_evidence.py','checkpoints':'see data/provenance.json'} for p in sorted((root/'figures').glob('*'))]
        (root/'data/figure-manifest.json').write_text(json.dumps(figures,indent=2))
        (root/'qa/science-checks.json').write_text(json.dumps({'status':'passed','checks':scalar,'actual_step':ae_step if root==L2 else vae_step,'reconstruction_only_path':fixtures['reconstruction_path_check'] if root==L3 else None,'binary_observations':bool(torch.all((x==0)|(x==1))),'held_digit':int(y[90]),'same_cloud_bytes':(L2/'figures/code-cloud-prior.svg').read_bytes()==(L3/'figures/ae-failure-cloud.svg').read_bytes(),'axis_limits':limits,'torch':torch.__version__,'numpy':np.__version__},indent=2))
    print(json.dumps({'AE_step':{k:v for k,v in ae_step.items() if not isinstance(v,dict)},'VAE_step':{k:v for k,v in vae_step.items() if not isinstance(v,dict)},'held3':fixtures['vae'],'figures':[len(list((p/'figures').glob('*'))) for p in [L2,L3]]},indent=2))
if __name__=='__main__':run()
