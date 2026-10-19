"""Reproduce fixed illustrative 2D points and the analytical mixture density."""
from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def uniforms(seed=667):
    while True:
        seed=(seed*1664525+1013904223)&0xffffffff
        yield (seed+.5)/4294967296
r=uniforms();eps=[]
for i in range(280):
    rad=math.sqrt(-2*math.log(next(r)));phi=2*math.pi*next(r);eps.append([rad*math.cos(phi),rad*math.sin(phi)])
eps=np.array(eps);means=np.array([[2.4*math.cos(i*math.pi/4),2.4*math.sin(i*math.pi/4)] for i in range(8)])
x=means[np.arange(280)%8]+.23*eps
# The cloud is stratified (35/component) for stable visual comparisons. Component recipe is categorical.
# Each within-component noise is an independent Gaussian draw; labels are balanced by construction.
axis=np.linspace(-3.6,3.6,220);X,Y=np.meshgrid(axis,axis);grid=np.stack([X,Y],-1)
density=sum(np.exp(-.5*(((grid-m)/.23)**2).sum(-1))/(2*math.pi*.23**2) for m in means)/8
cov=2.4**2/2+.23**2;single=np.exp(-.5*(grid**2).sum(-1)/cov)/(2*math.pi*cov)
plt.rcParams.update({'svg.fonttype':'none','font.size':14,'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8'})
fig,axs=plt.subplots(1,2,figsize=(11,4.6))
for ax,d,title,c in zip(axs,[single,density],['One Gaussian','Eight-component mixture'],['#B98224','#178F82']):
 ax.scatter(x[:,0],x[:,1],s=9,color='#3478E5',alpha=.6);ax.contour(X,Y,d,levels=5,colors=c);ax.set(title=title,xlim=(-3.6,3.6),ylim=(-3.6,3.6),xlabel='x₁',ylabel='x₂',aspect='equal')
fig.tight_layout();fig.savefig(ROOT/'figures/distribution-comparison.svg')
(ROOT/'metrics/distributions.json').write_text(json.dumps({'kind':'ILLUSTRATIVE','seed':667,'cloud_protocol':'35 independent within-component Gaussian draws for each of 8 components; balanced stratified teaching cloud, not iid mixture component counts','sampling_recipe':'separate generative recipe samples component categorically with weights 1/8','means':means.tolist(),'component_std':.23,'single_gaussian_covariance':cov,'epsilon':eps.tolist(),'points':x.tolist(),'density':'analytical known mixture, not fitted neural model'},indent=2))
print('Saved 280 fixed points, exact density specification and SVG comparison.')
# Fit an eight-component full-covariance mixture to observations (no component labels).
rng=np.random.default_rng(667);centres=[x[rng.integers(len(x))]]
for _ in range(7):
    distance=np.min(((x[:,None,:]-np.array(centres)[None,:,:])**2).sum(-1),axis=1)
    centres.append(x[rng.choice(len(x),p=distance/distance.sum())])
mu=np.array(centres);weights=np.ones(8)/8;covars=np.tile(np.eye(2)*.7,(8,1,1));hist=[]
for iteration in range(150):
    logs=[]
    for k in range(8):
        diff=x-mu[k];logs.append(np.log(weights[k])-.5*(2*np.log(2*np.pi)+np.linalg.slogdet(covars[k])[1]+np.einsum('ni,ij,nj->n',diff,np.linalg.inv(covars[k]),diff)))
    logs=np.stack(logs,1);shift=logs.max(1,keepdims=True);ll=shift[:,0]+np.log(np.exp(logs-shift).sum(1));resp=np.exp(logs-ll[:,None]);hist.append(float(ll.sum()))
    counts=resp.sum(0);weights=counts/len(x);mu=resp.T@x/counts[:,None]
    for k in range(8):
        diff=x-mu[k];covars[k]=(diff.T@(diff*resp[:,k,None]))/counts[k]+np.eye(2)*1e-4
    if len(hist)>1 and abs(hist[-1]-hist[-2])<1e-7:break
fit={'kind':'MEASURED FIT ON ILLUSTRATIVE DATA','algorithm':'full-covariance Gaussian mixture EM, K=8, k-means++ initialization, covariance floor 1e-4','seed':667,'points':280,'iterations':iteration+1,'weights':weights.tolist(),'means':mu.tolist(),'covariances':covars.tolist(),'log_likelihood_history':hist,'note':'Training-data likelihood only; no held-out performance claim; component labels are unused.'}
(ROOT/'metrics/mixture-fit.json').write_text(json.dumps(fit,indent=2));print('Fitted Gaussian mixture:',iteration+1,'iterations; total log likelihood',hist[-1])
