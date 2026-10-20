"""Rebuild exact arithmetic and measured replay assets. CPU, local data only."""
from pathlib import Path
import sys,json,gzip,time,hashlib,math
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gan import ConvG,train_toy_gan
R=Path(__file__).resolve().parents[1]; SOURCE=R.parents[1]/'generative-modelling-i-gans'; torch.set_num_threads(2)
plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Avenir Next','DejaVu Sans'],'font.size':19,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAFB','axes.facecolor':'#FAFAFB'})
C={'p':'#0F766E','d':'#147737','data':'#245EDB','loss':'#BE123C','noise':'#AA4E08'}
def dump(name,x): (R/name).write_text(json.dumps(x,indent=2))
def scalar_updates():
 dtype=torch.float64
 a=torch.tensor(.5,dtype=dtype,requires_grad=True);b=torch.tensor(-.5,dtype=dtype,requires_grad=True)
 w=torch.tensor(.4,dtype=dtype,requires_grad=True);c=torch.tensor(-.2,dtype=dtype,requires_grad=True)
 z=torch.tensor(1.,dtype=dtype);real=torch.tensor(2.,dtype=dtype);fake=a*z+b
 ld=F.softplus(-(w*real+c))+F.softplus(w*fake.detach()+c)
 before={'a':a.item(),'b':b.item(),'w':w.item(),'c':c.item(),'fake':fake.item(),'real':real.item(),'z':z.item(),'Dreal':torch.sigmoid(w*real+c).item(),'Dfake':torch.sigmoid(w*fake+c).item(),'lossD':ld.item()}
 ld.backward();dg={'w':w.grad.item(),'c':c.grad.item(),'a':a.grad,'b':b.grad}
 # Independent analytic derivative and finite difference of the exact same fixed inputs.
 dw=(before['Dreal']-1)*2+before['Dfake']*before['fake'];dc=before['Dreal']-1+before['Dfake']
 assert abs(dw-dg['w'])<1e-12 and abs(dc-dg['c'])<1e-12 and a.grad is None and b.grad is None
 eta=.1
 with torch.no_grad():w-=eta*w.grad;c-=eta*c.grad
 afterd={'w':w.item(),'c':c.item(),'a':a.item(),'b':b.item(),'lossD':(F.softplus(-(w*real+c))+F.softplus(w*fake.detach()+c)).item(),'Dreal':torch.sigmoid(w*real+c).item(),'Dfake':torch.sigmoid(w*fake+c).item()}
 w.requires_grad_(False);c.requires_grad_(False);w.grad=None;c.grad=None
 lg=F.softplus(-(w*(a*z+b)+c));lg.backward();gg={'a':a.grad.item(),'b':b.grad.item()}
 def fixed_g_loss(av,bv):return np.logaddexp(0,-(afterd['w']*(av+bv)+afterd['c']))
 h=1e-6;fd=(fixed_g_loss(.5+h,-.5)-fixed_g_loss(.5-h,-.5))/(2*h)
 assert abs(fd-gg['a'])<1e-9 and w.grad is None and c.grad is None
 with torch.no_grad():a-=eta*a.grad;b-=eta*b.grad
 afterg={'a':a.item(),'b':b.item(),'w':w.item(),'c':c.item(),'fake':(a*z+b).item(),'lossG':F.softplus(-(w*(a*z+b)+c)).item()}
 assert afterd['a']==before['a'] and afterd['b']==before['b'] and afterg['w']==afterd['w'] and afterg['c']==afterd['c']
 return {'eta':eta,'before':before,'dGradient':dg,'afterD':afterd,'lossGBefore':lg.item(),'gGradient':gg,'gFiniteDifference':fd,'afterG':afterg,'gradientChecks':{'DphaseGHasNoGrad':True,'GphaseDFrozen':True,'GphaseGReceivesGrad':True},'kind':'exact chosen scalar example, float64; two actual SGD updates'}
def image_grid(a,name,ncol=8):
 a=np.clip(a,0,1);n=min(len(a),32);rows=math.ceil(n/ncol);canvas=np.ones((rows*32,ncol*32))*.96
 for i,x in enumerate(a[:n]):canvas[(i//ncol)*32:(i//ncol)*32+28,(i%ncol)*32:(i%ncol)*32+28]=x.squeeze()
 Image.fromarray(np.uint8(canvas*255)).resize((ncol*96,rows*96),Image.Resampling.NEAREST).save(R/'figures'/name)
def linefig(name,x,ys,xlim=(-5,5),ylim=(0,.65),xlabel='x',ylabel='density'):
 fig,ax=plt.subplots(figsize=(11,3.7));
 for label,y,col,style in ys:ax.plot(x,y,label=label,color=col,lw=2.8,linestyle=style)
 ax.set(xlim=xlim,ylim=ylim,xlabel=xlabel,ylabel=ylabel);ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02));fig.tight_layout();fig.savefig(R/'figures'/name);plt.close(fig)
def density(samples,grid):
 samples=np.array(samples);return np.exp(-.5*((grid[:,None]-samples[None,:])/.16)**2).mean(1)/(.16*np.sqrt(2*np.pi))
def main():
 start=time.time();R.joinpath('figures').mkdir(exist_ok=True);fixture={'scalar':scalar_updates()}
 # Preserve and identify original official split/index; no training/test overlap.
 if (R/'data/mnist-subset.npz').exists():
  subset=dict(np.load(R/'data/mnist-subset.npz'))
 else:
  data_root=R.parents[1]/'generative-modelling-ii/work/data'
  subset={}
  for split,n in [('train',512),('test',128)]:
   prefix='train' if split=='train' else 't10k'
   x=np.frombuffer(gzip.decompress((data_root/f'{prefix}-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,28,28)[:n].copy()
   y=np.frombuffer(gzip.decompress((data_root/f'{prefix}-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8)[:n].copy();subset[split]=x;subset[split+'_labels']=y
  np.savez_compressed(R/'data/mnist-subset.npz',**subset)
 Image.fromarray(subset['test'][90]).resize((280,280),Image.Resampling.NEAREST).save(R/'figures/held-digit3.png')
 torch.manual_seed(123);g=ConvG();saved=torch.load(R/'models/gan_mnist.pt',map_location='cpu',weights_only=True);g.load_state_dict(saved['G']);g.eval()
 z=torch.randn(64,100,generator=torch.Generator().manual_seed(123))
 with torch.no_grad():pred=(g(z)[:,0].numpy()+1)/2
 snapshots=np.load(R/'data/gan_mnist_snapshots.npz');err=float(np.max(np.abs(pred-snapshots['30'])))
 assert err<2e-5,err
 for t in ['0','1','5','30']:image_grid(snapshots[t],f'mnist-epoch-{t}.png')
 image_grid(pred,'mnist-replayed.png');fixture['mnist']={'checkpoint_sha256':hashlib.sha256((R/'models/gan_mnist.pt').read_bytes()).hexdigest(),'replayMaxAbsError':err,'epochs':30,'seed':1,'fixedNoiseSeed':123,'representation':'grey uint8/127.5 - 1','heldExample':{'split':'test','index':90,'label':int(subset['test_labels'][90])}}
 one=json.loads((R/'data/toy_one_player.json').read_text());runs=json.loads((R/'data/toy_gan_runs.json').read_text());collapse=json.loads((R/'data/toy_gan_collapse.json').read_text());stable=runs[4]
 grid=np.linspace(-5,5,401);target=(np.exp(-.5*((grid-2)/.5)**2)+np.exp(-.5*((grid+2)/.5)**2))/(2*.5*np.sqrt(2*np.pi))
 for name,samp in [('initial',stable['record']['0']['samples']),('stable',stable['record']['3000']['samples']),('collapse',collapse['record']['3000']['samples'])]:
  linefig(f'toy-{name}.svg',grid,[('real density',target,C['data'],'-'),('generated KDE',density(samp,grid),C['p'],'--')],ylim=(0,2.6))
 for phase,keys in [('d_only',[0,200]),('g_only',[0,300])]:
  for step in keys:
   row=one[phase][str(step)];linefig(f'{phase}-{step}.svg',one['grid'],[('D(x)',row['d'],C['d'],'-')],xlim=(-10,10),ylim=(0,1),ylabel='P(real)')
   gx=np.linspace(-30,30,1201) if phase=='g_only' else grid
   gt=(np.exp(-.5*((gx-2)/.5)**2)+np.exp(-.5*((gx+2)/.5)**2))/(2*.5*np.sqrt(2*np.pi))
   linefig(f'{phase}-samples-{step}.svg',gx,[('real density',gt,C['data'],'-'),('generated KDE',density(row['samples'],gx),C['p'],'--')],xlim=(-30,30) if phase=='g_only' else (-5,5),ylim=(0,2.6))
 fixture['toy']={'stable':{k:stable[k] for k in ['seed','steps','left','right']},'collapse':{k:collapse[k] for k in ['seed','steps','left','right']},'coverageRule':'fraction within 1.0 of each mode at −2,+2; not a complete distribution metric','oneplayer':{'seed':one['seed'],'dSteps':one['d_steps'],'gSteps':one['g_steps'],'gOutsideFive':float(np.mean(np.abs(one['g_only']['300']['samples'])>5)),'gSampleRange':[float(min(one['g_only']['300']['samples'])),float(max(one['g_only']['300']['samples']))]}}
 fig,ax=plt.subplots(figsize=(11,3.7));
 for r,label,col in [(stable,'wide G',C['p']),(collapse,'narrow G',C['loss'])]:
  loss=np.array(r['losses']);smooth=np.convolve(loss[:,1],np.ones(100)/100,mode='valid');ax.plot(np.arange(len(smooth))+100,smooth,label=label,color=col)
 ax.set(xlabel='alternating update',ylabel='G loss (100-step mean)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(R/'figures/loss-vs-coverage.svg');plt.close(fig)
 logit=np.linspace(-6,6,241);prob=1/(1+np.exp(-logit));fixture['saturation']={'logit':logit.tolist(),'minimaxGradient':(-prob).tolist(),'nonSaturatingGradient':(prob-1).tolist()}
 linefig('saturation.svg',logit,[('minimax derivative',-prob,C['noise'],'--'),('non-saturating derivative',prob-1,C['loss'],'-')],xlim=(-6,6),ylim=(-1.05,.05),xlabel='fake logit',ylabel='derivative w.r.t. logit')
 dump('data/fixtures.json',fixture);dump('qa/science-checks.json',{'passed':True,'torch':torch.__version__,'device':'cpu','elapsedSeconds':time.time()-start,'scalarFiniteDifferenceResidual':abs(fixture['scalar']['gFiniteDifference']-fixture['scalar']['gGradient']['a']),'mnistReplayMaxAbsError':err,'claim':'fresh arithmetic/gradient checks and replay of existing checkpoints; no fresh full MNIST training'})
 print(json.dumps({'passed':True,'seconds':time.time()-start,'replay_error':err,'scalar':fixture['scalar']},indent=2))
if __name__=='__main__':main()
