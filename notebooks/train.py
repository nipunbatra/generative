"""Fresh CPU demonstrations; procedural binary glyphs, not MNIST.
Run from repo root: python notebooks/train.py --epochs 65 --steps 1800
"""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw
from models import TinyVAE,vae_step,make_image_gan,gan_step
ROOT=Path(__file__).resolve().parents[1]

def glyphs(n=2048,seed=667):
 r=np.random.default_rng(seed); out=[]
 segments=['abcedf','bc','abdeg','abcdg','bcfg','acdfg','acdefg','abc','abcdefg','abcdfg']
 lines={'a':(8,5,19,5),'b':(20,6,20,13),'c':(20,14,20,22),'d':(8,23,19,23),'e':(7,14,7,22),'f':(7,6,7,13),'g':(8,14,19,14)}
 for k in range(n):
  im=Image.new('L',(28,28));d=ImageDraw.Draw(im);dx,dy=r.integers(-2,3,2);w=int(r.integers(2,4))
  for s in segments[k%10]:
   x1,y1,x2,y2=lines[s];d.line((x1+dx,y1+dy,x2+dx,y2+dy),fill=255,width=w)
  out.append(np.asarray(im).copy()/255)
 return torch.tensor(np.array(out),dtype=torch.float32)[:,None]

def grid(t,path,columns=8):
 a=t.detach().cpu().reshape(-1,28,28).numpy();canvas=Image.new('L',(columns*32,((len(a)+columns-1)//columns)*32),24)
 for i,x in enumerate(a):canvas.paste(Image.fromarray(np.uint8(np.clip(x,0,1)*255)),((i%columns)*32+2,(i//columns)*32+2))
 canvas.resize((canvas.width*3,canvas.height*3),Image.Resampling.NEAREST).save(path)

def run(epochs=65,steps=1800):
 torch.set_num_threads(4);torch.manual_seed(667);torch.use_deterministic_algorithms(True)
 out=ROOT/'assets/recorded';out.mkdir(exist_ok=True);start=time.time();x=glyphs();held=glyphs(64,668);v=TinyVAE();opt=torch.optim.Adam(v.parameters(),lr=.001)
 fixed=torch.randn(64,2,generator=torch.Generator().manual_seed(111));history=[]
 for epoch in range(epochs+1):
  if epoch:
   vals=[]
   for idx in torch.randperm(len(x)).split(128):vals.append(vae_step(v,opt,x[idx]))
   history.append(dict(epoch=epoch,**{k:float(np.mean([a[k] for a in vals])) for k in vals[0]}))
  if epoch in {0,1,10,30,epochs}:
   with torch.no_grad():grid(v.decode(fixed).sigmoid(),out/f'vae-epoch-{epoch}.png')
  if epoch%10==0:print('VAE epoch',epoch,history[-1] if history else {},flush=True)
 torch.save(v.state_dict(),ROOT/'models/vae.pt')
 with torch.no_grad():
  mu,lv=v.encode(held);rec=v.decode(mu).sigmoid();grid(held[:16],out/'held-out.png');grid(rec[:16],out/'reconstructions.png');probs=v.decode(fixed).sigmoid();grid(probs,out/'vae-prior.png');grid(torch.bernoulli(probs),out/'vae-binary.png')
  coords=torch.cartesian_prod(torch.linspace(-3,3,9),torch.linspace(-3,3,9));grid(v.decode(coords).sigmoid(),out/'vae-grid.png',9)
  (out/'vae-data.json').write_text(json.dumps({'heldOut':held[:16].flatten(1).tolist(),'mu':mu[:16].tolist(),'reconstructions':rec[:16].tolist(),'overlay':mu.tolist(),'history':history,'fixedLatents':fixed.tolist()}))
  sd=v.state_dict();weights={k:val.numpy().tolist() for k,val in sd.items() if k.startswith('decoder')};(out/'decoder.json').write_text(json.dumps(weights,separators=(',',':')))
 # The same alternation function operates on 2D vectors: flatten(1) preserves them.
 torch.manual_seed(667);G=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(.2),nn.Linear(32,32),nn.LeakyReLU(.2),nn.Linear(32,2));D=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(.2),nn.Linear(32,32),nn.LeakyReLU(.2),nn.Linear(32,1))
 og=torch.optim.Adam(G.parameters(),lr=.001,betas=(.5,.999));od=torch.optim.Adam(D.parameters(),lr=.001,betas=(.5,.999))
 fixedz=torch.randn(256,2,generator=torch.Generator().manual_seed(71));frames=[];gh=[]
 real_ref=torch.randn(256,2,generator=torch.Generator().manual_seed(17))*.25;real_ref[:,0]+=torch.where(torch.arange(256)%2==0,-2.,2.)
 axis=torch.linspace(-4,4,33);xy=torch.cartesian_prod(axis,axis)
 def record(step,loss):
  with torch.no_grad():
   fake=G(fixedz);fl=D(fake);rl=D(real_ref)
   from torch.nn import functional as F
   monitor={'D':float(F.binary_cross_entropy_with_logits(rl,torch.ones_like(rl))+F.binary_cross_entropy_with_logits(fl,torch.zeros_like(fl))),'G':float(F.binary_cross_entropy_with_logits(fl,torch.ones_like(fl)))}
   frames.append({'step':step,'samples':fake.tolist(),'Dfield':D(xy).sigmoid().flatten().tolist(),'loss':loss,'monitorLoss':monitor})
 record(0,{'D':None,'G':None})
 for step in range(1,steps+1):
  real=torch.randn(128,2)*.25;real[:,0]+=torch.where(torch.rand(128)>.5,2.,-2.)
  loss=gan_step(G,D,og,od,real,latent_dim=2)
  if step%20==0:gh.append({'step':step,**loss})
  if step in {1,10,50,100,200,400,800,1200,steps}:record(step,loss)
  if step%400==0:print('GAN step',step,loss,flush=True)
 torch.save({'G':G.state_dict(),'D':D.state_dict()},ROOT/'models/toy-gan.pt')
 (out/'gan.json').write_text(json.dumps({'seed':667,'fixedNoiseSeed':71,'real':real_ref.tolist(),'frames':frames,'history':gh,'gridAxis':axis.tolist()}))
 # Image GAN is runnable; this short run illustrates its limits as well as outputs.
 torch.manual_seed(670);IG,ID=make_image_gan();ig=torch.optim.Adam(IG.parameters(),lr=2e-4,betas=(.5,.999));id_=torch.optim.Adam(ID.parameters(),lr=2e-4,betas=(.5,.999));iz=torch.randn(32,32);ih=[]
 for epoch in range(1,21):
  vals=[gan_step(IG,ID,ig,id_,x[idx]) for idx in torch.randperm(len(x)).split(128)]
  ih.append({'epoch':epoch,**{k:float(np.mean([a[k] for a in vals])) for k in vals[0]}})
 with torch.no_grad():grid(IG(iz),out/'image-gan.png')
 torch.save({'G':IG.state_dict(),'D':ID.state_dict()},ROOT/'models/image-gan.pt')
 meta={'evidence_type':'local trained-model output','label':'recorded training run','date':'2026-10-19','seed':667,'torch':torch.__version__,'device':'cpu','threads':4,'data':'2048 procedural binary seven-segment glyphs, ten classes; held-out seed 668','not_MNIST':True,'VAE':{'epochs':epochs,'batch':128,'lr':.001,'latent':2,'reduction':'sum pixels + sum latent dimensions, then batch mean','history':history},'toyGAN':{'steps':steps,'batch':128,'lr':.001,'latent':2,'distribution':'equal mixture N((-2,0),.25^2 I), N((2,0),.25^2 I)','sampling':'256 fixed latent draws at every checkpoint'},'imageGAN':{'seed':670,'epochs':20,'history':ih,'warning':'Short teaching run, no claim of convergence'},'elapsed_seconds':time.time()-start}
 (out/'provenance.json').write_text(json.dumps(meta,indent=2));print('Finished',meta['elapsed_seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=65);p.add_argument('--steps',type=int,default=1800);a=p.parse_args();run(a.epochs,a.steps)
