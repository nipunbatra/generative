"""Acquire attributed primary-source teaching figures. No generated substitutes."""
from pathlib import Path
import urllib.request, json, concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
items=[
('stylegan','https://raw.githubusercontent.com/NVlabs/stylegan/master/stylegan-teaser.png','https://github.com/NVlabs/stylegan','Karras, Laine and Aila / NVIDIA','StyleGAN','Published generated faces',[1,45,54,56]),
('spade','https://nvlabs.github.io/SPADE/images/teaser_high_res_uncompressed.png','https://nvlabs.github.io/SPADE/','Park et al. / NVIDIA','SPADE','Published layouts and generated landscapes',[5,57]),
('cyclegan','https://junyanz.github.io/CycleGAN/images/objects.jpg','https://junyanz.github.io/CycleGAN/','Zhu et al.','CycleGAN','Published input / domain translation pairs',[6,57,59]),
('inpainting','https://www.cs.cmu.edu/~dpathak/context_encoder/resources/result_fig.jpg','https://www.cs.cmu.edu/~dpathak/context_encoder/','Pathak et al.','Context Encoders','Published model completions, not the human-artwork teaser',[7,59]),
('satellite','https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_samples.png','https://www.samarkhanna.com/DiffusionSat/','Khanna et al.','DiffusionSat','Published metadata-conditioned samples',[9,59]),
('vae-paper','https://arxiv.org/pdf/1312.6114','https://arxiv.org/abs/1312.6114','Kingma and Welling','Auto-Encoding Variational Bayes','Figures 4 and 5',[2]),
('stylegan-paper','https://arxiv.org/pdf/1812.04948','https://github.com/NVlabs/stylegan','Karras, Laine and Aila / NVIDIA','StyleGAN','Figure 3 style mixing',[3,43,54]),
]
def get(it):
 id,url,page,authors,paper,caption,scenes=it
 suffix='.pdf' if 'paper' in id else ('.jpg' if url.endswith('.jpg') else '.png')
 path=ROOT/'assets/originals'/f'{id}{suffix}'
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  path.write_bytes(urllib.request.urlopen(req,timeout=75).read())
  return dict(id=id,local_path=str(path.relative_to(ROOT)),source_page=page,asset_url=url,authors_or_owner=authors,paper_or_dataset=paper,figure_or_caption=caption,license_or_usage_note='Attributed academic teaching excerpt. Original copyright retained; no blanket image redistribution licence asserted. See source terms.',retrieved_date='2026-10-19',transformations='Original downloaded unchanged; derived crops recorded separately.',evidence_type='published model output',used_in_scene_ids=[f's{n:02}' for n in scenes])
 except Exception as e:return {'id':id,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: rows=list(pool.map(get,items))
(ROOT/'assets/manifest.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([{k:v for k,v in x.items() if k in ('id','error','local_path')} for x in rows],indent=2))
base='https://phillipi.github.io/pix2pix/images/sketch2photo_shoes/latest_net_G_val/'
html=urllib.request.urlopen(base+'index.html').read().decode(); (ROOT/'work/shoes.html').write_text(html)
print(html[:5000])
