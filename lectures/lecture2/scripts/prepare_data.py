"""One-time packaging from existing local MNIST/checkpoints. No downloads or training."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/Users/nipun/git/generative')
sys.path.insert(0,str(SOURCE/'scripts'))
from train_models import load_mnist
raw=load_mnist()
tr,ty=raw['train']; te,ey=raw['test']
ids=np.arange(2000)
for folder,names in [(ROOT,['ae2','ae16']), (ROOT.parent/'lecture3',['vae2','vae16','vae2b8'])]:
    np.savez_compressed(folder/'data/mnist-snapshot.npz',train=tr[:256].numpy().astype('uint8'),train_labels=ty[:256].numpy(),test=te[ids].numpy().astype('uint8'),test_labels=ey[ids].numpy(),test_indices=ids)
    manifest={'data':{'source':'MNIST official train/test split, local existing copy','source_path':'/Users/nipun/git/psdv-teaching/datasets/MNIST','preprocessing':'x=(raw/255>0.5); fixed binary observations','train_indices':[0,255],'test_indices':[0,1999],'held_digit_test_index':90,'train_test_overlap':False},'models':[]}
    history=json.loads((SOURCE/'assets/models/training.json').read_text())
    (folder/'data/original-training.json').write_text(json.dumps({n:history[n] for n in names},indent=2))
    for n in names:
        src=SOURCE/f'assets/models/{n}.pt'; dst=folder/f'models/{n}.pt'; shutil.copy2(src,dst)
        manifest['models'].append({'name':n,'source':str(src),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'status':'existing trained checkpoint replayed; not freshly trained for this package'})
    manifest['snapshot_sha256']=hashlib.sha256((folder/'data/mnist-snapshot.npz').read_bytes()).hexdigest()
    (folder/'data/provenance.json').write_text(json.dumps(manifest,indent=2))
print('Packaged binary snapshot and five existing checkpoints into independent lecture folders.')
