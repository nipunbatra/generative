"""Fresh DDPM checks and checkpoint replay. No network access or new dependencies."""
from pathlib import Path
import json,time,hashlib,gzip,math
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ddpm import TinyDenoiser,make_schedule,extract,reverse_step,sample_ddpm
R=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Avenir Next','DejaVu Sans'],'font.size':19,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAFB','axes.facecolor':'#FAFAFB'})
def dump(name,x):(R/name).write_text(json.dumps(x,indent=2,default=float))
def tile(v,size=252,limits=(-1,1)):
 # Clipping here is a display transform only. Values passed into training/sampling are never clipped.
 lo,hi=limits;a=np.clip((np.asarray(v).squeeze()-lo)/(hi-lo),0,1)
 return Image.fromarray(np.uint8(a*255)).resize((size,size),Image.Resampling.NEAREST)
def gallery(v,path,ncol=4):
 n=len(v);can=Image.new('RGB',(ncol*196,math.ceil(n/ncol)*196),'#FAFAFB')
 for i,a in enumerate(v):can.paste(tile(a,188),((i%ncol)*196,(i//ncol)*196))
 can.save(R/'figures'/path)
def main():
 start=time.time();b,a,ab,var=make_schedule(dtype=torch.float64);T=1000
 assert ab[0]==1 and var[1]==0 and ab[-1]<5e-5
 assert torch.allclose(ab[1:],torch.exp(torch.log(a[1:]).cumsum(0)),atol=1e-14)
 x0=.8;eps=-.4;t=350;xt=ab[t].sqrt().item()*x0+(1-ab[t]).sqrt().item()*eps
 theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True);pred=theta*xt;loss=(pred-eps)**2;loss.backward();grad=theta.grad.item();eta=.1;new=theta.item()-eta*grad;newloss=(new*xt-eps)**2
 fn=lambda v:(v*xt-eps)**2;h=1e-6;fd=(fn(.3+h)-fn(.3-h))/(2*h);assert abs(grad-fd)<1e-9
 ehat=.25;mu=(xt-b[t].item()*ehat/math.sqrt(1-ab[t].item()))/math.sqrt(a[t].item());xhat=(xt-math.sqrt(1-ab[t].item())*ehat)/math.sqrt(ab[t].item())
 c0=math.sqrt(ab[t-1].item())*b[t].item()/(1-ab[t].item());ct=math.sqrt(a[t].item())*(1-ab[t-1].item())/(1-ab[t].item());mu2=c0*xhat+ct*xt
 assert abs(mu-mu2)<1e-12
 xi=-.7;xp=mu+math.sqrt(var[t].item())*xi
 end1=reverse_step(torch.tensor(.4),torch.tensor(.2),1,make_schedule(),torch.tensor(999.));end2=reverse_step(torch.tensor(.4),torch.tensor(.2),1,make_schedule(),torch.tensor(-999.));assert torch.equal(end1,end2)
 rng=np.random.default_rng(667);draw=math.sqrt(ab[t])*x0+math.sqrt(1-ab[t])*rng.normal(size=200000)
 mean_err=abs(draw.mean()-math.sqrt(ab[t])*x0);variance_err=abs(draw.var()-(1-ab[t]));assert mean_err<.006 and variance_err<.012
 fixture={'schedule':{'T':1000,'betaStart':float(b[1]),'betaEnd':float(b[-1]),'alphaBarT':float(ab[-1]),'variance':'posterior variance beta_tilde','timeIndexing':'one-based, t = 1,…,1000; alpha_bar[0]=1','selected':[{'t':j,'beta':float(b[j]),'alpha':float(a[j]),'alphaBar':float(ab[j]),'signal':float(ab[j].sqrt()),'noise':float((1-ab[j]).sqrt()),'posteriorVariance':float(var[j])} for j in [1,7,100,350,600,920,1000]]},'scalar':{'x0':x0,'epsilon':eps,'t':t,'xt':xt,'thetaBefore':.3,'prediction':float(pred.detach()),'lossBefore':float(loss.detach()),'gradient':grad,'finiteDifference':fd,'eta':eta,'thetaAfter':new,'lossAfter':newloss,'kind':'chosen single-pixel calculation, float64, squared-error objective'},'reverse':{'xt':xt,'predictedNoise':ehat,'x0Estimate':xhat,'mean':mu,'posteriorFormMean':mu2,'variance':float(var[t]),'sigma':math.sqrt(var[t]),'xi':xi,'previousSample':xp,'t':t,'coefficientX0':c0,'coefficientXt':ct},'checks':{'forwardMeanError':mean_err,'forwardVarianceError':variance_err,'reverseMeanResidual':abs(mu-mu2),'finalStepIndependentOfNoise':True}}
 if (R/'data/mnist-subset.npz').exists():
  subset=dict(np.load(R/'data/mnist-subset.npz'))
 else:
  source=R.parents[1]/'generative-modelling-ii/work/data';subset={}
  for split,n in [('train',512),('test',128)]:
   prefix='train' if split=='train' else 't10k';subset[split]=np.frombuffer(gzip.decompress((source/f'{prefix}-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,28,28)[:n].copy();subset[split+'_labels']=np.frombuffer(gzip.decompress((source/f'{prefix}-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8)[:n].copy()
  np.savez_compressed(R/'data/mnist-subset.npz',**subset)
 x=torch.tensor(subset['test'][0:1,None],dtype=torch.float32)/127.5-1;sf=make_schedule();g=torch.Generator().manual_seed(667);noise=torch.randn(x.shape,generator=g)
 tile(x[0,0]).save(R/'figures/held-seven.png');tile(noise[0,0],limits=(-3,3)).save(R/'figures/injected-noise.png')
 for j in [1,7,100,350,600,920,1000]:tile(sf[2][j].sqrt()*x+(1-sf[2][j]).sqrt()*noise).save(R/'figures'/f'forward-{j}.png')
 saved=torch.load(R/'models/ddpm.pt',map_location='cpu',weights_only=True);assert saved['contract']=='linear1000-eps-posterior-v1';model=TinyDenoiser();model.load_state_dict(saved['model']);model.eval()
 xtimage=sf[2][350].sqrt()*x+(1-sf[2][350]).sqrt()*noise
 with torch.no_grad():ep=model(xtimage,torch.tensor([350]));estimate=(xtimage-(1-sf[2][350]).sqrt()*ep)/sf[2][350].sqrt()
 tile(ep,limits=(-3,3)).save(R/'figures/predicted-noise.png');tile(estimate).save(R/'figures/clean-estimate.png')
 # A new training update uses TRAIN images; test image above is used only for inspection.
 torch.manual_seed(2026);trainx=torch.tensor(subset['train'][:4,None],dtype=torch.float32)/127.5-1;ts=torch.tensor([7,100,350,920]);ee=torch.randn_like(trainx);aa=extract(sf[2],ts,trainx);xx=aa.sqrt()*trainx+(1-aa).sqrt()*ee
 opt=torch.optim.SGD(model.parameters(),lr=1e-3);opt.zero_grad();ll=torch.nn.functional.mse_loss(model(xx,ts),ee);ll.backward();gn=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt());before={k:v.clone() for k,v in model.state_dict().items()};opt.step()
 with torch.no_grad():la=float(torch.nn.functional.mse_loss(model(xx,ts),ee))
 delta=max(float((v-before[k]).abs().max()) for k,v in model.state_dict().items());assert delta>0 and math.isfinite(la)
 fixture['neuralUpdate']={'batchSize':4,'timesteps':ts.tolist(),'seed':2026,'learningRate':.001,'lossBefore':float(ll.detach()),'lossAfterSameBatch':la,'gradientNorm':gn,'maxParameterChange':delta,'split':'train indices 0:4','reduction':'mean over B×1×28×28'}
 model.load_state_dict(saved['model']);replay_start=time.time();print('Starting full 1000-step CPU checkpoint replay (8 samples)',flush=True)
 samples,frames,estimates=sample_ddpm(model,(8,1,28,28),sf,seed=667,record=True)
 np.savez_compressed(R/'data/replayed-trajectory.npz',samples=samples.numpy(),**{'state_'+k:v for k,v in frames.items()})
 old=np.load(R/'data/original-trajectory.npz');error=float(np.abs(samples.numpy()-old['samples']).max());assert np.isfinite(samples.numpy()).all()
 gallery(samples.numpy(),'samples-replayed.png')
 for j in [1000,800,600,400,200,50,1,0]:tile(frames[str(j)][0]).save(R/'figures'/f'reverse-{j}.png')
 # Local evidence retains an exact replay scope; differing device arithmetic may accumulate in the chain.
 fixture['checkpoint']={'sha256':hashlib.sha256((R/'models/ddpm.pt').read_bytes()).hexdigest(),'trainingSteps':saved['steps'],'trainingSeed':saved['seed'],'parameters':sum(p.numel() for p in model.parameters()),'sampleSeed':667,'sampleCount':8,'samplerSteps':1000,'replaySeconds':time.time()-replay_start,'maxDifferenceFromOriginalDevice':error,'freshFullTraining':False,'dataRepresentation':'uint8/127.5−1','heldExample':{'split':'test','index':0,'label':7},'display':'images clip values to [−1,1] only for viewing; neither training states nor sampler states are clipped'}
 fig,ax=plt.subplots(figsize=(11,3.6));idx=np.arange(1001);ax.plot(idx,ab.sqrt(),label='signal coefficient',color='#245EDB',lw=2.8);ax.plot(idx,(1-ab).sqrt(),label='noise coefficient',color='#AA4E08',lw=2.8);ax.set(xlabel='timestep t',ylabel='coefficient',xlim=(0,1000),ylim=(0,1.02));ax.legend(frameon=False);fig.tight_layout();fig.savefig(R/'figures/schedule.svg');plt.close(fig)
 original=json.loads((R/'data/original-training.json').read_text());fig,ax=plt.subplots(figsize=(11,3.6));ax.plot([q['step'] for q in original['history']],[q['batch_noise_mse'] for q in original['history']],color='#BE123C',lw=2.5);ax.set(xlabel='training update',ylabel='batch noise MSE');fig.tight_layout();fig.savefig(R/'figures/training-history.svg');plt.close(fig)
 dump('data/fixtures.json',fixture);dump('qa/science-checks.json',{'passed':True,'device':'cpu','torch':torch.__version__,'elapsedSeconds':time.time()-start,**fixture['checks'],'gradientResidual':abs(grad-fd),'checkpointReplay':fixture['checkpoint'],'claim':'fresh maths and neural update; full sampler replay of a saved checkpoint, not new full training'})
 print(json.dumps({'passed':True,'elapsed':time.time()-start,'scalar':fixture['scalar'],'reverse':fixture['reverse'],'neural':fixture['neuralUpdate'],'replayDifference':error},indent=2))
if __name__=='__main__':main()
