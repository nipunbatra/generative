from pathlib import Path
from PIL import Image
import json,urllib.request
r=Path(__file__).resolve().parents[1];manifest=json.loads((r/'assets/manifest.json').read_text())
def crop(src,name,box=None,maxsize=(1600,1000),note=''):
 im=Image.open(r/src).convert('RGB')
 if box:im=im.crop(box)
 im.thumbnail(maxsize);path=f'assets/gallery/{name}.jpg';im.save(r/path,quality=93)
 manifest.append(dict(id=name,local_path=path,source_page=next((x['source_page'] for x in manifest if x['id']==name.split('-')[0]),'https://arxiv.org/abs/1312.6114' if name.startswith('vae') else 'https://arxiv.org/abs/1812.04948'),asset_url=src,authors_or_owner='See original asset entry',paper_or_dataset=name,figure_or_caption=note,license_or_usage_note='Attributed teaching crop; original copyright retained. StyleGAN material: NVIDIA CC BY-NC 4.0.',retrieved_date='2026-10-19',transformations={'crop_pixels':box,'resize_max':maxsize},evidence_type='published model output',used_in_scene_ids=[]))
crop('assets/originals/stylegan.png','stylegan-faces')
crop('work/vae-page.png','vae-samples',(230,766,403,970),note='Figure 5(a), 2D latent random samples, paper page 10')
crop('work/vae-page.png','vae-manifold',(505,150,925,610),note='Figure 4(b), inverse Gaussian CDF coordinates; decoded grid, not prior draws')
crop('work/stylegan-page.png','stylegan-mixing',(61,130,578,736),note='Figure 3 upper-left subset. Source A rows, Source B columns; coarse styles from B. Other styles from A.')
crop('assets/originals/spade.png','spade-layout',(0,0,2020,890),note='Top legend and first two layout columns with first appearance row; correspondence preserved')
crop('assets/originals/cyclegan.jpg','cyclegan-horse',(0,0,635,340),note='Top-left input horse / generated zebra pair; labels retained')
crop('assets/originals/inpainting.jpg','inpainting-pair',(0,0,1359,681),note='First held-out pair: masked input left, generated completion right')
crop('assets/originals/satellite.png','satellite-samples',None,note='Published metadata-conditioned generated imagery. No measurement or forecast claim.')
base='https://phillipi.github.io/pix2pix/images/sketch2photo_shoes/latest_net_G_val/'
for key in ['input','output','target']:
 url=base+f'images/{key}/100_AB.jpg';p=f'assets/gallery/shoe-{key}.jpg';(r/p).write_bytes(urllib.request.urlopen(url).read());manifest.append(dict(id='shoe-'+key,local_path=p,source_page=base+'index.html',asset_url=url,authors_or_owner='Isola et al.',paper_or_dataset='pix2pix / edges2shoes',figure_or_caption={'input':'Input shoe outline','output':'Generated output','target':'Ground truth reference photo'}[key],license_or_usage_note='Attributed academic comparison; dataset image rights retained by original owners.',retrieved_date='2026-10-19',transformations='None',evidence_type='published model output' if key=='output' else 'published model input or reference',used_in_scene_ids=['s04','s57']))
(r/'assets/manifest.json').write_text(json.dumps(manifest,indent=2))
