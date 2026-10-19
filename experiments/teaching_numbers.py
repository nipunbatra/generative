"""Single numerical source for slides and assertions; no rounded inputs."""
from pathlib import Path
import math,json
R=Path(__file__).resolve().parents[1]
def fixtures():
 xs=[-.5,.5,1.5];ll=lambda m:sum(-.5*math.log(2*math.pi)-.5*(x-m)**2 for x in xs)
 probs=[.2,.5,.3];seed=667;counts=[0]*3;sequence=[]
 for _ in range(100):
  seed=(1664525*seed+1013904223)%2**32;u=seed/2**32;k=0 if u<.2 else 1 if u<.7 else 2;counts[k]+=1;sequence.append(k)
 mu,sigma,eps,upstream=2,.5,-.8,3
 return {'categorical':{'probabilities':probs,'seed':667,'sequence':sequence,'counts':counts},'mle':{'observations':xs,'sigma':1,'initial_mu':0,'optimal_mu':sum(xs)/len(xs),'initial_log_likelihood':ll(0),'optimal_log_likelihood':ll(.5)},'mixture':{'means':[[2.4*math.cos(k*math.pi/4),2.4*math.sin(k*math.pi/4)] for k in range(8)],'weights':[.125]*8,'sigma':.23},'reparam':{'mu':mu,'sigma':sigma,'epsilon':eps,'product':sigma*eps,'z':mu+sigma*eps,'upstream':upstream,'dz_dmu':1,'dz_dsigma':eps,'grad_mu':upstream,'grad_sigma':upstream*eps,'logvar':math.log(sigma*sigma),'dz_dlogvar':.5*sigma*eps,'grad_logvar':upstream*.5*sigma*eps},'bernoulli':{'x':[1,0,1,0],'probabilities':[.8,.3,.6,.1],'observed_probabilities':[.8,.7,.6,.9],'terms':[-math.log(p) for p in [.8,.7,.6,.9]],'sum_nats':-sum(math.log(p) for p in [.8,.7,.6,.9]),'mean_nats':-sum(math.log(p) for p in [.8,.7,.6,.9])/4},'kl':{'mu':1,'sigma':.5,'variance':.25,'logvar':math.log(.25),'inside':1+.25-math.log(.25)-1,'nats':.5*(1+.25-math.log(.25)-1)},'gan':{'real_score':.9,'fake_score':.2,'D_loss':-math.log(.9)-math.log(.8),'G_loss':-math.log(.2)},'flow':{'forward_scale':2,'inverse_jacobian':.5,'log_density_correction':-math.log(2),'base_interval':[-.5,.5],'mapped_interval':[-1,1],'interval_mass':math.erf(.5/math.sqrt(2))}}
if __name__=='__main__':
 n=fixtures();q=n['reparam'];assert n['mle']['optimal_mu']==.5;assert abs(n['mle']['initial_log_likelihood']+4.1318155996)<1e-9;assert abs(n['mle']['optimal_log_likelihood']+3.7568155996)<1e-9
 for k,v in {'z':1.6,'grad_mu':3,'grad_sigma':-2.4,'grad_logvar':-.6}.items():assert abs(q[k]-v)<1e-12
 assert abs(n['kl']['nats']-.81814718056)<1e-10
 assert abs(n['bernoulli']['sum_nats']-1.1960046347)<1e-9
 assert abs(n['gan']['D_loss']-.32850406697)<1e-10;assert abs(n['gan']['G_loss']-1.60943791243)<1e-10
 assert n['flow']['forward_scale']*n['flow']['inverse_jacobian']==1
 (R/'results/generative_i_teaching_numbers.json').write_text(json.dumps(n,indent=2));print('Teaching fixtures: all exact-value assertions passed.')
