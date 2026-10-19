"""Refresh checksums and scene links without replacing authored provenance."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[1]
assets={a['id']:a for a in json.loads((r/'assets/manifest.json').read_text())}
for p in sorted((r/'figures').glob('*.svg')):
 sid=p.stem.split('-')[0]
 assets['diagram-'+p.stem]={
  'id':'diagram-'+p.stem,'local_path':str(p.relative_to(r)),
  'source_page':'src/figures.mjs','asset_url':None,
  'authors_or_owner':'ES 667 teaching implementation',
  'paper_or_dataset':'Original mechanism diagram; recorded local model outputs where shown',
  'figure_or_caption':'Exported from the completed scene '+sid,
  'license_or_usage_note':'Original diagram. Embedded published shoe images retain the attribution and rights recorded in the shoe asset entries.',
  'retrieved_date':'2026-10-19','transformations':'Native SVG exported from the lecture. Image hrefs are relative to this repository.',
  'evidence_type':'original teaching diagram','used_in_scene_ids':[sid]}
for a in assets.values():
 if a['id']=='image-gan':a['used_in_scene_ids']=sorted(set(a.get('used_in_scene_ids',[])+['s17']))
 if a['id'] in ['shoe-input','shoe-output','shoe-target']:a['used_in_scene_ids']=sorted(set(a.get('used_in_scene_ids',[])+['s57']))
 if a['id']=='satellite-samples':a['transformations']='Source figure retained. Scene 9 uses SVG viewBox crops (822,153,173,173) and (1006,153,173,173) from the 1600 by 897 local figure: August and January, first method, first sample row.'
 p=r/a['local_path']
 if p.exists():a['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'assets/manifest.json').write_text(json.dumps(list(assets.values()),indent=2))
scenes=json.loads((r/'src/scene-manifest.json').read_text())
for s in scenes:s['source_asset_ids']=[a['id'] for a in assets.values() if s['id'] in a.get('used_in_scene_ids',[])]
(r/'src/scene-manifest.json').write_text(json.dumps(scenes,indent=2))
print('Updated',len(assets),'unique assets and',len(scenes),'scene source lists.')
