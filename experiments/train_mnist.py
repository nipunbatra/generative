"""Reproducible CPU MLP AE/VAE teaching experiment. No pretrained weights."""
from pathlib import Path
import argparse, gzip, hashlib, json, platform, time, urllib.request
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BLUE='#3478E5'; TEAL='#178F82'; PURPLE='#8B5BB5'; AMBER='#B98224'; CORAL='#D45555'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8','svg.fonttype':'none'})
class Model(nn.Module):
    def __init__(self,d,vae=True):
        super().__init__(); self.vae=vae; self.d=d
        self.enc=nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.mu=nn.Linear(128,d); self.logvar=nn.Linear(128,d)
        self.dec=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,784))
    def encode(self,x):
        h=self.enc(x); return self.mu(h),self.logvar(h)
    def forward(self,x):
        mu,lv=self.encode(x)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu) if self.vae else mu
        return self.dec(z),mu,lv

def data():
    folder=ROOT/'experiments/data'; folder.mkdir(exist_ok=True)
    names=['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    hashes={}; arrays=[]
    for name in names:
        p=folder/name
        if not p.exists(): urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/'+name,p)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest(); raw=gzip.decompress(p.read_bytes())
        a=np.frombuffer(raw,dtype=np.uint8,offset=16 if 'images' in name else 8).copy()
        arrays.append(a.reshape(-1,784) if 'images' in name else a)
    # Fixed threshold makes the Bernoulli observation likelihood a literal binary-data model.
    return torch.tensor(arrays[0]>127,dtype=torch.float32),arrays[1],torch.tensor(arrays[2]>127,dtype=torch.float32),arrays[3],hashes

def montage(x,path,cols=8,scale=3):
    a=np.asarray(x).reshape(-1,28,28); rows=(len(a)+cols-1)//cols
    canvas=np.full((rows*32+4,cols*32+4),250,dtype=np.uint8)
    for k,d in enumerate(a):
        y=4+(k//cols)*32; z=4+(k%cols)*32
        canvas[y:y+28,z:z+28]=np.uint8(np.clip(1-d,0,1)*255)
    im=Image.fromarray(canvas); im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST).save(path)

def plot_scatter(points,labels,name):
    fig,ax=plt.subplots(figsize=(6,5)); sc=ax.scatter(points[:,0],points[:,1],c=labels,s=7,cmap='tab10',alpha=.65,vmin=-.5,vmax=9.5)
    ax.set(xlabel='latent coordinate 1',ylabel='latent coordinate 2'); cb=fig.colorbar(sc,ax=ax,ticks=range(10)); cb.set_label('held-out digit label (not used in training)')
    fig.tight_layout(); fig.savefig(ROOT/f'figures/{name}.svg'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--seed',type=int,default=667); ap.add_argument('--train-limit',type=int,default=60000); args=ap.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    train,labels,test,tl,hashes=data(); train=train[:args.train_limit]
    for d in ['figures','models','metrics']: (ROOT/d).mkdir(exist_ok=True)
    montage(test[:32],ROOT/'figures/observed.png',8); montage(test[np.where(tl==7)[0][:8]],ROOT/'figures/sevens.png',8)
    all_metrics={}; start=time.time()
    for name,d,isvae in [('ae2',2,False),('vae2',2,True),('vae16',16,True)]:
        torch.manual_seed(args.seed); np.random.seed(args.seed); m=Model(d,isvae); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
        loader=torch.utils.data.DataLoader(train,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        history=[]
        for ep in range(args.epochs):
            m.train(); sums=np.zeros(2)
            for x in loader:
                opt.zero_grad(); logits,mu,lv=m(x); recon=F.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(1).mean()
                kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean() if isvae else torch.tensor(0.)
                (recon+kl).backward(); opt.step(); sums+=np.array([recon.item(),kl.item()])*len(x)
            row={'epoch':ep+1,'reconstruction_nats':float(sums[0]/len(train)),'kl_nats':float(sums[1]/len(train))}; history.append(row)
            print(name,json.dumps(row),flush=True)
        m.eval(); torch.manual_seed(args.seed+1)
        with torch.no_grad():
            # Fixed stochastic held-out evaluation, one posterior draw per example.
            vals=[]
            for x in test.split(256):
                logits,mu,lv=m(x); rec=F.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(1)
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1) if isvae else torch.zeros(len(x))
                vals.append(torch.stack([rec,kl],1))
            evaluation=torch.cat(vals).mean(0).tolist()
            mus,lvs=m.encode(test); probs=m.dec(mus[:16]).sigmoid(); prior_z=torch.randn(64,d); prior=m.dec(prior_z).sigmoid(); prior_draws=torch.bernoulli(prior)
            montage(torch.cat([test[:8],probs[:8]]),ROOT/f'figures/{name}-recon.png',8)
            montage(prior,ROOT/f'figures/{name}-prior-means.png',8)
            montage(prior[:16],ROOT/f'figures/{name}-prior16-means.png',8)
            montage(prior_draws,ROOT/f'figures/{name}-prior-draws.png',8)
            a,b=mus[0],mus[1]; ts=torch.linspace(0,1,9); interp=(1-ts[:,None])*a+ts[:,None]*b
            montage(m.dec(interp).sigmoid(),ROOT/f'figures/{name}-interpolation.png',9,4)
            np.savez_compressed(ROOT/f'models/{name}-recording.npz',mu=mus.numpy(),logvar=lvs.numpy(),prior_z=prior_z.numpy(),prior_mean=prior.numpy(),prior_draws=prior_draws.numpy(),interpolation_z=interp.numpy(),labels=tl)
            if d==2:
                plot_scatter(mus[:2500].numpy(),tl[:2500],f'{name}-scatter')
                axis=np.linspace(-3,3,25); grid=np.array([[x,y] for y in axis[::-1] for x in axis],np.float32)
                decoded=m.dec(torch.from_numpy(grid)).sigmoid(); montage(decoded,ROOT/f'figures/{name}-grid.png',25,1)
                browser={'layers':[{'weight':layer.weight.tolist(),'bias':layer.bias.tolist()} for layer in m.dec if isinstance(layer,nn.Linear)],'source':name,'seed':args.seed,'grid_axis':axis.tolist(),'recorded_grid':decoded.numpy().tolist(),'test_latents':mus[:600].tolist(),'test_labels':tl[:600].tolist(),'posterior_example':{'mu':mus[0].tolist(),'logvar':lvs[0].tolist()},'parity':[{'z':z.tolist(),'pixels':m.dec(z).sigmoid().tolist()} for z in torch.tensor([[0.,0.],[1.,-1.],[-2.,.5]])]}
                (ROOT/f'models/{name}-browser.json').write_text(json.dumps(browser,separators=(',',':')))
        checkpoint=ROOT/f'models/{name}.pt'; torch.save({'state_dict':m.state_dict(),'latent_dim':d,'vae':isvae,'seed':args.seed,'epochs':args.epochs},checkpoint)
        all_metrics[name]={'history':history,'test_reconstruction_nats':evaluation[0],'test_kl_nats':evaluation[1],'test_negative_elbo_nats':sum(evaluation) if isvae else None,'latent_mean':mus.mean(0).tolist(),'latent_std':mus.std(0).tolist(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for n,c in [('vae2',PURPLE),('vae16',TEAL)]:
        h=all_metrics[n]['history']; axs[0].plot([v['epoch'] for v in h],[v['reconstruction_nats'] for v in h],label=n,color=c,lw=3); axs[1].plot([v['epoch'] for v in h],[v['kl_nats'] for v in h],label=n,color=c,lw=3)
    for ax,title in zip(axs,['Reconstruction term','KL term']): ax.set(xlabel='epoch',ylabel='nats / training example',title=title); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ROOT/'figures/training-curves.svg'); plt.close(fig)
    manifest={'seed':args.seed,'epochs':args.epochs,'train_examples':len(train),'test_examples':len(test),'batch_size':256,'lr':.001,'optimizer':'Adam','device':'CPU','deterministic_algorithms':True,'architecture':'784→256 ReLU→128 ReLU→mu/logvar; d→128 ReLU→256 ReLU→784 logits','preprocessing':'fixed binary threshold uint8 > 127','data_source':'https://storage.googleapis.com/cvdf-datasets/mnist/','data_sha256':hashes,'versions':{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()},'test_evaluation':'one posterior draw per held-out example, seed 668; no test data used for training','image_protocol':'reconstructions and grids are decoder Bernoulli means; -prior-draws images sample binary x; first test examples, fixed seed, no selection or retouching','seconds':time.time()-start,'models':all_metrics}
    (ROOT/'metrics/mnist.json').write_text(json.dumps(manifest,indent=2)); print('DONE',manifest['seconds'],flush=True)
if __name__=='__main__': main()
