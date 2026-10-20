"""Write and execute both self-contained lecture notebooks using the current interpreter."""
from pathlib import Path
import sys,json,os
import nbformat as nb
from nbclient import NotebookClient
BASE=Path(__file__).resolve().parents[2]
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
setup='''from pathlib import Path
import sys, json, math, copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from IPython.display import display, Image, SVG
ROOT = Path.cwd()
if not (ROOT / "data/mnist-snapshot.npz").exists():
    raise RuntimeError("Run this notebook with its lecture folder as the working directory.")
sys.path.insert(0, str(ROOT / "notebooks"))
import lab
lab.ROOT = ROOT
torch.set_num_threads(2)
torch.manual_seed(667)
d = lab.data()
x_train, x_test = d["train"], d["test"]
held3 = x_test[90:91]
fixtures = json.loads((ROOT / "data/fixtures.json").read_text())
print("Torch", torch.__version__)
print("Snapshot:", tuple(x_train.shape), tuple(x_test.shape))
print("Binary observations:", torch.unique(x_test).tolist())
print("Held image: official test index90, label", int(d["test_labels"][90]))
assert torch.all((x_test == 0) | (x_test == 1))
assert int(d["test_labels"][90]) == 3
'''
show='''def show_grid(images, columns=10, title=""):
    images = np.asarray(images).reshape(-1,28,28)
    rows = int(np.ceil(len(images)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns*1.2,rows*1.2), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for ax, img in zip(axes.flat, images): ax.imshow(img,cmap="gray_r",vmin=0,vmax=1)
    fig.suptitle(title)
    plt.tight_layout(); plt.show()
'''
c2=[md('''# Lecture 2: from fitting a distribution to an autoencoder

65-minute slide-led lecture companion. This offline package bundles a fixed binary MNIST subset and existing trained AE weights. It **replays** these models and performs an actual copied-checkpoint update; it does not claim to retrain the recorded checkpoints.

Data: official train images0–255 and test images0–1999, thresholded once at raw/255>0.5. Held image is official test index90. No test image is used in an update or fitted-code density. Logs are natural; losses sum pixels then average images.

Run top-to-bottom in the lecture folder. The environment needs Torch, NumPy and Matplotlib; the supplied machine has `/Users/nipun/git/clip-lab/.venv/bin/python`. No network or download is used.'''),code(setup),md('## 1. Observe the same handwritten 3\nBefore looking at a reconstruction, describe what the decoder must preserve.'),code(show+'\nshow_grid(held3, 1, "Held-out binary 3")\nprint("Ink pixels:", int(held3.sum()))'),md('## 2. Fit, then sample a small distribution\nThe component labels are unobserved. The following small EM implementation fits a two-Gaussian mixture to the same twelve points used in the slides. The component choice is the latent variable.'),code('''x=np.array([-3.2,-2.8,-2.4,-2.1,-1.8,-1.7,1.5,1.8,2.1,2.4,2.7,3.1])
mean, std=x.mean(),x.std()
means=np.array([-2.,2.]);stds=np.ones(2);weights=np.ones(2)/2
for step in range(40):
    density=np.exp(-.5*((x[:,None]-means)/stds)**2)/stds/np.sqrt(2*np.pi)*weights
    responsibility=density/density.sum(1,keepdims=True)
    count=responsibility.sum(0)
    weights=count/len(x)
    means=(responsibility*x[:,None]).sum(0)/count
    stds=np.sqrt((responsibility*(x[:,None]-means)**2).sum(0)/count)
print("One Gaussian: mean/std",mean,std)
print("Mixture weights, means, stds:",weights,means,stds)
assert np.allclose(means,fixtures["toy"]["means"])
rng=np.random.default_rng(674)
single=rng.normal(mean,std,24)
component=rng.choice(2,24,p=weights)
mixture=rng.normal(means[component],stds[component])
plt.figure(figsize=(9,3));plt.scatter(single,np.zeros(24),label="one Gaussian")
plt.scatter(mixture,np.ones(24),label="mixture");plt.yticks([0,1],["Gaussian","mixture"])
plt.xlabel("new observed value x");plt.legend();plt.show()'''),md('**Predict:** if we double the number of generated points, have we fitted a new model?\n\n**Answer:** no. The fitted parameters stay fixed. Only the sampling experiment changes.'),md('## 3. Replay the AE and separate weights from activations\nThe encoder computes a deterministic code; it does not define an approximate posterior distribution. The decoder returns logits. The network class is local in `notebooks/lab.py`.'),code('''ae2,ae16=lab.model("ae2"),lab.model("ae16")
print(ae2)
with torch.no_grad():
    z=ae2.encoder(held3)[0]
    probability=ae2.decoder(z).sigmoid()
print("held3 code:",z.numpy())
show_grid(torch.cat([held3.flatten(1),probability]),2,"Observation and decoder probabilities")'''),md('## 4. A loss contribution and one actual training step\nUse stable logits-based BCE. Sum the event dimensions (pixels); average images. Neither reduction is an arbitrary formatting choice.'),code('''observed=torch.tensor([[1.,0.,1.,0.]],dtype=torch.float64)
probability=torch.tensor([[.8,.1,.6,.3]],dtype=torch.float64)
contributions=-(observed*probability.log()+(1-observed)*(1-probability).log())
from_logits=F.binary_cross_entropy_with_logits(torch.logit(probability),observed,reduction="none")
print("Pixel contributions:",contributions.tolist())
print("Sum per image:",contributions.sum(1).tolist())
assert torch.allclose(contributions,from_logits,atol=1e-12)

# Copy the model: the bundled trained checkpoint is never overwritten.
model=copy.deepcopy(ae2).double()
xb=x_train[:8].double()
logits,mu,lv,_=model(xb)
per_image=F.binary_cross_entropy_with_logits(logits,xb.flatten(1),reduction="none").sum(1)
loss=per_image.mean()
model.zero_grad(set_to_none=True);loss.backward()
eta=1e-4
before={name:p.detach().clone() for name,p in model.named_parameters()}
with torch.no_grad():
    for parameter in model.parameters(): parameter -= eta*parameter.grad
    updated=F.binary_cross_entropy_with_logits(model(xb)[0],xb.flatten(1),reduction="none").sum(1).mean()
changes={name:float((parameter.detach()-before[name]).norm()) for name,parameter in model.named_parameters()}
print("Before/after nats per image:",float(loss.detach()),float(updated))
print("Encoder changed:",any(v>0 for name,v in changes.items() if name.startswith("encoder")))
print("Decoder changed:",any(v>0 for name,v in changes.items() if name.startswith("decoder")))
assert abs(float(updated)-fixtures["ae_step"]["after_same_noise"])<1e-8'''),md('## 5. Held-out reconstruction and bottleneck evidence\nThese are reconstructions, not new samples. Evaluate identical images and reductions for both dimensions.'),code('''ids=torch.tensor([int(torch.where(d["test_labels"]==i)[0][0]) for i in range(10)])
with torch.no_grad():
    rows=[x_test[ids].flatten(1)]
    for name,model in [("ae2",ae2),("ae16",ae16)]:
        logits,mu,lv,_=model(x_test)
        loss=F.binary_cross_entropy_with_logits(logits,x_test.flatten(1),reduction="none").sum(1).mean()
        print(name,"test reconstruction nats/image:",float(loss))
        rows.append(logits[ids].sigmoid())
show_grid(torch.cat(rows),10,"Observed / AE d2 / AE d16")'''),md('## 6. Reconstruct, interpolate and sample are different experiments\nUse exactly the same decoder. The code-cloud and prior overlay must keep the same axes. A standard-normal prior was not part of AE training.'),code('''with torch.no_grad():
    codes=ae2.encoder(x_test)[0]
    prior=torch.randn(80,2,generator=torch.Generator().manual_seed(667))
    decoded=ae2.decoder(prior[:10]).sigmoid()
limits=fixtures["code_axes"]
fig,ax=plt.subplots(figsize=(9,4))
ax.scatter(codes[:,0],codes[:,1],s=7,alpha=.3,label="encoded test data",c="#8B2CDE")
ax.scatter(prior[:,0],prior[:,1],marker="+",s=50,label="N(0,I) draws",c="#AA4E08")
ax.set(xlim=limits[0],ylim=limits[1],xlabel="z1",ylabel="z2");ax.legend();plt.show()
show_grid(decoded,10,"Standard-normal codes decoded by the AE")
with torch.no_grad():
    i7=int(torch.where(d["test_labels"]==7)[0][0])
    t=torch.linspace(0,1,9).unsqueeze(1)
    line=(1-t)*codes[90]+t*codes[i7]
    interpolation=ae2.decoder(line).sigmoid()
show_grid(interpolation,9,"Interpolation between two encoded test images")'''),md('## 7. A separate density model for the codes\nFit this Gaussian only on training-image codes. It is a legitimate additional model, but a single Gaussian can miss multimodal code geometry. This tiny 256-image fit is a demonstration, not a quality benchmark.'),code('''with torch.no_grad():
    training_codes=ae2.encoder(x_train)[0]
    mean=training_codes.mean(0)
    covariance=torch.cov(training_codes.T)+1e-4*torch.eye(2)
    epsilon=torch.randn(10,2,generator=torch.Generator().manual_seed(670))
    new_codes=mean+epsilon@torch.linalg.cholesky(covariance).T
    fitted_samples=ae2.decoder(new_codes).sigmoid()
print("Fitted training-code mean:",mean.tolist())
show_grid(fitted_samples,10,"AE plus a Gaussian fitted to training codes")'''),md('## 8. Optional small fresh training run\nThis trains a new model on the bundled 256-image subset. It cannot reproduce the full recorded model quality. Set the switch to True deliberately; results remain in memory.'),code('''RUN_SMALL_TRAINING=False
if RUN_SMALL_TRAINING:
    torch.manual_seed(667)
    fresh=lab.AutoEncoder(2,False)
    optimizer=torch.optim.Adam(fresh.parameters(),lr=1e-3)
    for epoch in range(10):
        logits,mu,lv,_=fresh(x_train)
        objective=F.binary_cross_entropy_with_logits(logits,x_train.flatten(1),reduction="none").sum(1).mean()
        optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
    print("Final tiny-subset training loss:",float(objective.detach()))
else:
    print("Optional fresh training skipped; replay and actual inspected update above have run.")'''),md('## Exit questions and answers\n1. What distribution supplied the new AE codes? **We chose one; reconstruction did not specify it.**\n2. Does interpolation prove quality under a standard-normal prior? **No. It follows a selected path between encoded examples.**\n3. What changes in the VAE? **An explicit prior and likelihood create an inference problem; an encoder amortises a variational approximation.**\n\nProvenance and checkpoint hashes: `data/provenance.json`. Figures and their hashes: `data/figure-manifest.json`. Original checkpoint training histories: `data/original-training.json`.')]
c3=[md('''# Lecture 3: a variational autoencoder, from model to one update

70-minute slide-led lecture companion. The same binary MNIST observation convention and held test image90 carry over from Lecture 2. Existing trained VAE checkpoints are replayed. All displayed new arithmetic and the inspected SGD update run locally; weights on disk are not changed.

The package is offline and contains its data and checkpoints. Run top-to-bottom from this lecture folder. Requirements: Torch, NumPy, Matplotlib. `logvar` always means log variance. Pixel and latent dimensions are summed, then the batch is averaged.'''),code(setup),code(show),md('## 1. The generative story\nDraw z from N(0,I), compute Bernoulli logits, then draw binary pixels. A probability image is not the same object as a sampled observation.'),code('''vae=lab.model("vae2")
with torch.no_grad():
    z=torch.randn(10,2,generator=torch.Generator().manual_seed(667))
    probability=vae.decoder(z).sigmoid()
    observation=torch.bernoulli(probability,generator=torch.Generator().manual_seed(669))
show_grid(probability,10,"Decoder probabilities at ten prior draws")
show_grid(observation,10,"Binary observations for those same latent draws")
print("Observation support:",torch.unique(observation).tolist())'''),md('## 2. What q does the encoder assign to the held 3?\nThe encoder returns means and log variances. Its weights are shared; its outputs depend on the image. The following epsilon is a chosen reproducible teaching input, not an empirically estimated parameter.'),code('''with torch.no_grad():
    mu,logvar=vae.encoder(held3)
    std=torch.exp(.5*logvar)
    epsilon=torch.tensor([[.62,-1.15]])
    z=mu+std*epsilon
    logits=vae.decoder(z)
    recon,kl=lab.terms(logits,held3,mu,logvar)
for name,value in [("mu",mu),("logvar",logvar),("std",std),("epsilon",epsilon),("z",z)]:
    print(name,value.numpy())
print("one-draw reconstruction NLL, exact prior KL, negative ELBO estimate:",float(recon),float(kl),float(recon+kl))
show_grid(torch.cat([held3.flatten(1),logits.sigmoid()]),2,"Observation and probability image")
q=torch.distributions.Normal(mu,std)
p=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
library_kl=torch.distributions.kl_divergence(q,p).sum(1)
assert torch.allclose(kl,library_kl,atol=1e-6)'''),md('## 3. Before amortisation: fit q for one image\nFix the decoder and optimise free mean and log variance. The optimisation uses 64 fresh epsilon draws per update; a separate 256-draw fixed-noise estimate monitors progress. Neither curve is an exact evidence calculation. The procedure need not reach the encoder’s solution or the exact posterior.'),code('''history=lab.fit_one_q(vae,held3,steps=80)
for row in history[::2]:print(row)
plt.figure(figsize=(9,3));plt.plot([r["step"] for r in history],[r["negative_elbo_estimate"] for r in history],"o-")
plt.xlabel("updates to this image's free variational parameters")
plt.ylabel("negative ELBO estimate");plt.show()'''),md('**Why amortise?** Repeating this fit for every new image costs repeated optimisation. A shared encoder predicts variational parameters in one pass. It adds a shared-mapping restriction, not a guarantee of exact inference.'),md('## 4. Reparameterisation and an independent gradient check\nFix epsilon when comparing derivatives. The analytic derivatives below are for a chosen scalar loss z². Fresh training noise and fixed gradient-check noise have different purposes.'),code('''mu=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
v=torch.tensor(-.4,dtype=torch.float64,requires_grad=True)
epsilon=torch.tensor(.62,dtype=torch.float64)
std=torch.exp(.5*v)
z=mu+std*epsilon
loss=z.square();loss.backward()
analytic_mu=2*z.detach()
analytic_v=z.detach()*std.detach()*epsilon
h=1e-6
fn=lambda mm,vv:(mm+torch.exp(.5*vv)*epsilon).square()
fd_mu=(fn(mu.detach()+h,v.detach())-fn(mu.detach()-h,v.detach()))/(2*h)
fd_v=(fn(mu.detach(),v.detach()+h)-fn(mu.detach(),v.detach()-h))/(2*h)
print("autograd / analytic / finite difference for mu:",float(mu.grad),float(analytic_mu),float(fd_mu))
print("autograd / analytic / finite difference for logvar:",float(v.grad),float(analytic_v),float(fd_v))
assert abs(float(mu.grad-fd_mu))<1e-8
assert abs(float(v.grad-fd_v))<1e-8
print(lab.verify())
print(lab.reconstruction_path_check(vae,x_train[:8]))'''),md('## 5. One complete, actual training step\nThe demonstration copies a trained checkpoint and performs SGD on eight **training** images. A small stated step size keeps the local update inspectable. A fixed epsilon is used for the before/after comparison only. Ordinary training samples fresh epsilon.'),code('''model=copy.deepcopy(vae).double()
xb=x_train[:8].double()
epsilon=torch.randn(8,2,generator=torch.Generator().manual_seed(667),dtype=torch.float64)
mu,logvar=model.encoder(xb)
std=torch.exp(.5*logvar)
z=mu+std*epsilon
logits=model.decoder(z)
pixel_nll=F.binary_cross_entropy_with_logits(logits,xb.flatten(1),reduction="none")
reconstruction=pixel_nll.sum(1)
prior_kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(1)
loss=(reconstruction+prior_kl).mean()
model.zero_grad(set_to_none=True);loss.backward()
before={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
gradients={name:float(parameter.grad.norm()) for name,parameter in model.named_parameters()}
eta=1e-6
with torch.no_grad():
    for parameter in model.parameters():parameter-=eta*parameter.grad
    new_logits,new_mu,new_lv,_=model(xb,epsilon)
    rr,kk=lab.terms(new_logits,xb,new_mu,new_lv)
    after=(rr+kk).mean()
print("same-noise loss before/after:",float(loss.detach()),float(after))
for group in ["encoder.mu","encoder.logvar","decoder"]:
    delta=sum(float((parameter.detach()-before[name]).norm()) for name,parameter in model.named_parameters() if name.startswith(group))
    grad=sum(value for name,value in gradients.items() if name.startswith(group))
    print(group,"gradient-norm sum:",grad,"parameter-delta-norm sum:",delta)
    assert grad>0 and delta>0
assert abs(float(after)-fixtures["vae_step"]["after_same_noise"])<1e-8'''),md('## 6. What did the recorded model learn?\nDecode encoder means for a visual reconstruction check. This is not an estimate of the expected reconstruction term. Separately inspect prior samples.'),code('''ids=torch.tensor([int(torch.where(d["test_labels"]==i)[0][0]) for i in range(10)])
with torch.no_grad():
    means,logvars=vae.encoder(x_test)
    reconstruction=vae.decoder(means[ids]).sigmoid()
show_grid(torch.cat([x_test[ids].flatten(1),reconstruction]),10,"Observed / mean-code reconstruction")
prior=torch.randn(80,2,generator=torch.Generator().manual_seed(667))
fig,ax=plt.subplots(figsize=(9,4));ax.scatter(means[:,0],means[:,1],s=7,alpha=.3,label="encoder means",c="#8B2CDE")
ax.scatter(prior[:,0],prior[:,1],marker="+",s=50,label="same prior draws",c="#AA4E08")
limits=fixtures["code_axes"];ax.set(xlim=limits[0],ylim=limits[1],xlabel="z1",ylabel="z2");ax.legend();plt.show()'''),md('The axes are exactly the AE axes. A mean cloud omits each q covariance, so it is not a full plot of the aggregate posterior. A normal-looking cloud alone does not prove matching distributions.'),md('## 7. Compare recorded dimensions and KL weights carefully\nThe βKL=1 and βKL=8 two-dimensional checkpoints used the same architecture, seed and 40-epoch schedule. These are different objectives. Evaluate one-draw reconstruction estimates with a stated seed, plus analytic prior KL; do not call the prior KL the unknown posterior gap.'),code('''for name in ["vae2","vae16","vae2b8"]:
    model=lab.model(name)
    epsilon=torch.randn(len(x_test),model.encoder.mu.out_features,generator=torch.Generator().manual_seed(673))
    with torch.no_grad():
        logits,mu,lv,_=model(x_test,epsilon)
        recon,kl=lab.terms(logits,x_test,mu,lv)
    print(name,"reconstruction:",float(recon.mean()),"prior KL:",float(kl.mean()),"negative ELBO estimate:",float((recon+kl).mean()))'''),md('## 8. Optional small training run\nThis fresh model sees only the bundled 256-image subset and does not reproduce the recorded checkpoint quality. No on-disk weights are overwritten.'),code('''RUN_SMALL_TRAINING=False
if RUN_SMALL_TRAINING:
    torch.manual_seed(667)
    model=lab.AutoEncoder(2,True)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(20):
        logits,mu,logvar,z=model(x_train)  # fresh epsilon every iteration
        recon,kl=lab.terms(logits,x_train,mu,logvar)
        loss=(recon+kl).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    print("Tiny-subset training loss:",float(loss.detach()))
else:
    print("Optional full training skipped; per-image VI and a real joint update were executed above.")'''),md('## Exit checks\n- **Does zero prior KL imply exact posterior inference?** No. It only says q matches the prior.\n- **Does a stable ELBO certify zero posterior gap?** No; the model evidence is generally unavailable.\n- **Is βKL=0 automatically a deterministic AE?** No. The sampled latent path may remain.\n- **Does prior sampling need an encoder?** No. It uses the prior and decoder.\n\nModel provenance, hashes and original training histories are in `data/`. The saved notebook outputs were executed in the documented local environment.')]
for number,cells in [(2,c2),(3,c3)]:
 root=BASE/f'lecture{number}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Lecture Python','language':'python','name':'lecture-python'},'language_info':{'name':'python','version':sys.version.split()[0]}})
 path=root/f'lecture{number}.ipynb';nb.write(notebook,path)
 kernels=root/'qa/jupyter/kernels/lecture-python';kernels.mkdir(parents=True,exist_ok=True)
 (kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Lecture Python','language':'python'}))
 os.environ['JUPYTER_PATH']=str(root/'qa/jupyter')
 client=NotebookClient(notebook,timeout=180,kernel_name='lecture-python',resources={'metadata':{'path':str(root)}})
 client.execute();nb.write(notebook,path)
 errors=[out for c in notebook.cells if c.cell_type=='code' for out in c.get('outputs',[]) if out.output_type=='error']
 report={'status':'passed' if not errors else 'failed','executed_code_cells':sum(c.cell_type=='code' for c in notebook.cells),'output_count':sum(len(c.get('outputs',[])) for c in notebook.cells),'errors':len(errors),'interpreter':sys.executable,'notebook':path.name}
 (root/'qa/notebook-execution.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
