"""Independent checks of displayed fixtures, checkpoint recordings, and continuity."""
from pathlib import Path
import json,math,hashlib,base64
import numpy as np
import torch
from train_mnist import Model
from teaching_numbers import fixtures
R=Path(__file__).resolve().parents[1];n=fixtures();saved=json.loads((R/'results/generative_i_teaching_numbers.json').read_text());assert n==saved
checks={};x=torch.tensor(n['bernoulli']['x'],dtype=torch.float64);p=torch.tensor(n['bernoulli']['probabilities'],dtype=torch.float64);bce=torch.nn.functional.binary_cross_entropy(p,x,reduction='sum').item();assert abs(bce-n['bernoulli']['sum_nats'])<1e-12;checks['bernoulli_library_matches']=bce
for m in np.linspace(-1,2,121):
 ll=sum(torch.distributions.Normal(float(m),1).log_prob(torch.tensor(n['mle']['observations'])).tolist());assert ll<=n['mle']['optimal_log_likelihood']+1e-6
checks['MLE_maximum_checked_on_grid']=True
checks['GAN_loss_library']=[torch.nn.functional.binary_cross_entropy(torch.tensor([.9,.2],dtype=torch.float64),torch.tensor([1.,0.],dtype=torch.float64),reduction='sum').item(),-math.log(.2)];assert abs(checks['GAN_loss_library'][0]-n['gan']['D_loss'])<1e-12
extra=json.loads((R/'results/rebuild-mnist.json').read_text());errs={}
for name,d,vae in [('ae2',2,False),('vae2',2,True),('vae16',16,True)]:
 cp=R/f'models/{name}.pt';assert hashlib.sha256(cp.read_bytes()).hexdigest()==extra['models'][name]['checkpoint_sha256'];m=Model(d,vae);m.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);m.eval();row=extra['models'][name]
 with torch.no_grad():
  pred=m.dec(torch.tensor(row['prior_z'])).sigmoid().numpy();assert np.max(np.abs(pred-row['prior']))<1e-6
  actual=m.dec(torch.tensor(row['interpolation_z'])).sigmoid().numpy();recorded=np.frombuffer(base64.b64decode(row['interpolation_pixels']),dtype=np.uint8).reshape(41,784)/255;err=float(np.max(np.abs(actual-recorded)));assert err<=.5/255+1e-6;errs[name]=err
checks['all_checkpoint_interpolations_max_display_error']=errs
hashes=[]
for t in [0,100,300,600,900,1000]:
 ours=(R/f'figures/bridge-dog-{t:04}.png').read_bytes();digest=hashlib.sha256(ours).hexdigest();source=R/f'diffusion/figures/generative2/dog_{t:04}.png';record=next(f for f in json.loads((R/'results/bridge-provenance.json').read_text())['frames'] if f['t']==t);assert digest==record['sha256'];assert not source.exists() or ours==source.read_bytes();hashes.append(digest)
checks['bridge_frames_byte_identical']=hashes
run=json.loads((R/'results/toy-gan.json').read_text());assert hashlib.sha256((R/'models/toy-gan.pt').read_bytes()).hexdigest()==run['checkpoint_sha256'];assert [s['step'] for s in run['snapshots']]==[0,100,500,1000,2000];assert all(math.isfinite(h['G']) and math.isfinite(h['D']) for h in run['history']);checks['GAN_snapshots_and_finite_losses']=True
# A two-dimensional isotropic radius of 3 sigma has mass 1-exp(-9/2), not 99.7%.
checks['diagnostic_scope']='Coverage threshold counts modes, not goodness of fit. Generated gaps remain visible.'
(R/'output/rebuild-qa/numerical-report.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
