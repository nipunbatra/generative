"""Package the current lecture and its evidence; exclude scratch and other projects."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
slides=json.loads((ROOT/'output/slide-manifest.json').read_text())
by_id={s['id']:s for s in slides};figures=[]
for p in sorted((ROOT/'figures').glob('*')):
 if not p.is_file():continue
 name=p.name
 if name.startswith('rebuild-'):
  id=name.removeprefix('rebuild-').rsplit('-',1)[0]
  kind=by_id.get(id,{}).get('provenance','Authored teaching diagram; see slide metadata')
 elif name.startswith('bridge-dog-'):kind='OBSERVED photograph / ILLUSTRATIVE forward corruption; exact Generative II frame; CC BY-SA 4.0'
 elif name.startswith('pedagogy-replay'):kind='OBSERVED DATA: first eight binary MNIST training images'
 elif name.startswith(('pedagogy-independent','pedagogy-pixel-mean')):kind='MEASURED FIT: independent Bernoulli pixels fitted to all training images; seed 667'
 elif name.startswith('pedagogy-beta'):kind='MEASURED: controlled 2D beta sweep, 12,000 training examples, 12 epochs, seed 667'
 elif name.startswith(('observed','sevens')):kind='OBSERVED DATA: thresholded MNIST test inputs'
 elif name.startswith(('ae2','vae2','vae16','training-curves','measured-models','paths-and-curves','ae-sampling')):kind='MEASURED: saved checkpoint outputs or recorded training history'
 elif name=='fitted-mixture.svg':kind='MEASURED FIT ON ILLUSTRATIVE DATA'
 elif name=='attention-decoder.svg':kind='AUTHORED COURSE REFERENCE: Beyond Attention decoder-use.svg'
 else:kind='ILLUSTRATIVE / ANALYTICAL authored teaching visual; see slide metadata'
 figures.append({'file':str(p.relative_to(ROOT)),'kind':kind,'sha256':sha(p)})
(ROOT/'metrics/figure-manifest.json').write_text(json.dumps({'source_of_truth':'output/slide-manifest.json; metrics/mnist.json; metrics/pedagogy.json; results/probability.json; results/pedagogy.json; results/rebuild-mnist.json; results/toy-gan.json; results/bridge-provenance.json; SOURCES.md','figures':figures,'photo':{'file':'images/newfoundland.png','sha256':sha(ROOT/'images/newfoundland.png'),'license':'CC BY-SA 4.0','source':'SOURCES.md'}},indent=2))
allowed=['src','experiments','figures','images','metrics','models','notebooks','results','vendor','output/pdf']
files=[]
for name in allowed:
 files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'data' not in p.relative_to(ROOT).parts and p.name!='.DS_Store')
qa=ROOT/'output/rebuild-qa';report=json.loads((qa/'browser-report.json').read_text())
files.extend(qa/rec['file'] for rec in report['records'])
files.extend(qa.glob('*.json'));files.extend(qa.glob('*.md'));files.extend(qa.glob('pdf-contact-*.jpg'));files.append(qa/'mobile.png');files.extend(qa.glob('resources-*.png'))
files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.html','.json','.txt'])
files.extend([ROOT/'.gitignore',ROOT/'output/slide-manifest.json',ROOT/'output/slide-manifest.csv'])
files=sorted(set(p for p in files if p.exists()))
main=[s for s in slides if not s['appendix']]
manifest={'lecture':'Generative Modelling I','edition':4,'date':'2026-10-19','main_screens':len(main),'main_builds':sum(s['builds'] for s in main),'duration_minutes':sum(s['seconds'] for s in main)/60,'core_duration_minutes':sum(s['seconds'] for s in main if not s.get('detour'))/60,'detour_screens':sum(bool(s.get('detour')) for s in main),'backup_screens':len(slides)-len(main),'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
mp=ROOT/'output/release-manifest.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
p=ROOT/'output/generative-i-complete.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in files:z.write(f,'generative-i/'+str(f.relative_to(ROOT)))
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 for record in manifest['files']:assert hashlib.sha256(z.read('generative-i/'+record['path'])).hexdigest()==record['sha256']
print(f'Packaged and hash-verified {len(files)} files; {p.stat().st_size/1e6:.1f} MB. Excluded cached data, local environments, scratch, old QA, and diffusion/.')
