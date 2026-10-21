"""Train the documented small DDPM using local MNIST only.
Full run: python scripts/train.py --mode full --steps 2000 --device mps --mnist-root PATH
Quick run checks execution on the bundled training subset; it does not claim image quality.
"""
from pathlib import Path
import argparse,gzip,json,time,os
import numpy as np
import torch
from ddpm import TinyDenoiser,make_schedule,training_loss
ROOT=Path(__file__).resolve().parents[1]
def load_local_images(folder):
 p=Path(folder)/'train-images-idx3-ubyte.gz'
 if not p.exists():raise FileNotFoundError(f'Local MNIST archive required: {p}; downloads are disabled.')
 raw=np.frombuffer(gzip.decompress(p.read_bytes()),np.uint8,offset=16).copy().reshape(-1,1,28,28)
 return torch.from_numpy(raw).float()/127.5-1

def train(mode='quick',steps=None,device='cpu',mnist_root=None,seed=667,batch_size=64):
 torch.set_num_threads(2);torch.manual_seed(seed);steps=steps or (20 if mode=='quick' else 2000)
 if mode=='full':x=load_local_images(mnist_root or os.environ.get('MNIST_ROOT',''))
 else:x=torch.from_numpy(np.load(ROOT/'data/mnist-subset.npz')['train'][:,None].copy()).float()/127.5-1
 model=TinyDenoiser().to(device);schedule=make_schedule(device=device);opt=torch.optim.Adam(model.parameters(),lr=2e-4);history=[];start=time.time()
 for step in range(1,steps+1):
  batch=x[torch.randint(len(x),(min(batch_size,len(x)),))].to(device);loss=training_loss(model,batch,schedule[2]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if step==1 or step%100==0 or step==steps:history.append({'step':step,'loss':float(loss.detach())})
 out=ROOT/'models'/f'{mode}-training.pt';torch.save({'model':model.state_dict(),'contract':'linear1000-eps-posterior-v1','steps':steps,'seed':seed},out)
 result={'mode':mode,'steps':steps,'seed':seed,'device':device,'batch_size':batch_size,'elapsed_seconds':time.time()-start,'images':len(x),'history':history,'checkpoint':str(out),'claim':'execution check only' if mode=='quick' else 'fresh full-data training; inspect samples separately'}
 (ROOT/'qa'/f'{mode}-training.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--steps',type=int);p.add_argument('--device',default='cpu');p.add_argument('--mnist-root');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args();print(json.dumps(train(**vars(a)),indent=2))
