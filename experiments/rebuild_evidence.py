"""Additional checkpoint-derived images and an unselected eight-Gaussian GAN run."""
from train_mnist import *
import base64

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(667)
 _,_,test,labels,_=data();extra={'models':{}}
 montage(test[:8],ROOT/'figures/observed-eight.png',8,4)
 montage(test[:1],ROOT/'figures/observed-seven.png',1,4)
 common_z=torch.randn(8,2,generator=torch.Generator().manual_seed(667))
 for name,d,v in [('ae2',2,False),('vae2',2,True),('vae16',16,True)]:
  p=ROOT/f'models/{name}.pt';m=Model(d,v);m.load_state_dict(torch.load(p,weights_only=True)['state_dict']);m.eval()
  with torch.no_grad():
   mu,lv=m.encode(test[:8]);recon=m.dec(mu).sigmoid();montage(recon,ROOT/f'figures/{name}-recon-eight.png',8,4)
   priorz=common_z if d==2 else torch.randn(8,d,generator=torch.Generator().manual_seed(667));pr=m.dec(priorz).sigmoid();montage(pr,ROOT/f'figures/{name}-common-prior.png',8,4)
   draws=torch.bernoulli(pr,generator=torch.Generator().manual_seed(668));montage(draws,ROOT/f'figures/{name}-common-draws.png',8,4)
   ts=torch.linspace(0,1,41);zs=(1-ts[:,None])*mu[0]+ts[:,None]*mu[1];values=m.dec(zs).sigmoid()
   extra['models'][name]={'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'recon':recon.tolist(),'prior':pr.tolist(),'prior_z':priorz.tolist(),'interpolation_z':zs.tolist(),'interpolation_pixels':base64.b64encode(np.rint(values.numpy()*255).astype('uint8').tobytes()).decode(),'first_mu':mu[0].tolist(),'first_logvar':lv[0].tolist()}
 extra['observed_seven']=test[0].tolist();(ROOT/'results/rebuild-mnist.json').write_text(json.dumps(extra,separators=(',',':')))
 # One run, fixed seed and architecture; train from the known mixture, no snapshot selection.
 torch.manual_seed(667);G=nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2));D=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 og=torch.optim.Adam(G.parameters(),lr=.0005,betas=(.5,.999));od=torch.optim.Adam(D.parameters(),lr=.0005,betas=(.5,.999))
 means=torch.tensor([[2.4*np.cos(k*np.pi/4),2.4*np.sin(k*np.pi/4)] for k in range(8)],dtype=torch.float32)
 real=lambda n:means[torch.randint(8,(n,))]+.23*torch.randn(n,2)
 fixed_z=torch.randn(1000,2,generator=torch.Generator().manual_seed(668));fixed_real=real(1000);snapshots=[];history=[]
 def snap(step):
  with torch.no_grad():
   ps=G(fixed_z);dist=torch.cdist(ps,means);v,idx=dist.min(1);valid=v<3*.23;counts=torch.bincount(idx[valid],minlength=8);cover=int((counts>=10).sum())
   snapshots.append({'step':step,'points':ps[:500].tolist(),'counts_within_3sigma':counts.tolist(),'coverage_at_least_1pct':cover,'fraction_within_3sigma':valid.float().mean().item()})
 snap(0)
 for step in range(1,2001):
  x=real(256);z=torch.randn(256,2);od.zero_grad();fake=G(z).detach();dl=F.softplus(-D(x)).mean()+F.softplus(D(fake)).mean();dl.backward();od.step()
  for p in D.parameters():p.requires_grad_(False)
  og.zero_grad();gl=F.softplus(-D(G(torch.randn(256,2)))).mean();gl.backward();og.step()
  for p in D.parameters():p.requires_grad_(True)
  if step%20==0:history.append({'step':step,'D':dl.item(),'G':gl.item()})
  if step in [100,500,1000,2000]:snap(step);print('GAN step',step,snapshots[-1]['coverage_at_least_1pct'],flush=True)
 cp=ROOT/'models/toy-gan.pt';torch.save({'G':G.state_dict(),'D':D.state_dict(),'seed':667,'steps':2000},cp)
 evidence={'seed':667,'steps':2000,'batch':256,'optimizer':'Adam','lr':.0005,'betas':[.5,.999],'architecture':'G: 2-64-64-2 ReLU; D: 2-64-64-1 LeakyReLU(0.2) logits','training_data':'iid equal eight-Gaussian mixture, radius 2.4, sigma 0.23; fresh batch per step','diagnostic':'1,000 fixed prior samples. Assign nearest center; count only distance < 3 sigma. Mode covered if count >=10 (1%). This threshold is illustrative, not a benchmark.','snapshots':snapshots,'history':history,'real_points':fixed_real[:500].tolist(),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'torch':torch.__version__}
 (ROOT/'results/toy-gan.json').write_text(json.dumps(evidence,separators=(',',':')))
 print('Checkpoint-derived images and measured GAN saved.',flush=True)
if __name__=='__main__':main()
