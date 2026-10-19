"""Inspect the PDF export and create a contact sheet for every teaching build."""
from pathlib import Path
import json,subprocess,xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];out=R/'output/rebuild-qa';work=R/'work/pdf-final';work.mkdir(parents=True,exist_ok=True)
manifest=json.loads((R/'output/slide-manifest.json').read_text());main=[s for s in manifest if not s['appendix']]
counts={'generative-i':len(main),'generative-i-all-builds':sum(s['builds'] for s in main),'generative-i-backup':len(manifest)-len(main)}
report={'counts':{},'out_of_page_words':[]}
for name,expected in counts.items():
 pdf=R/'output/pdf'/f'{name}.pdf';info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);actual=int(next(l.split(':')[1] for l in info.splitlines() if l.startswith('Pages:')));assert actual==expected
 report['counts'][name]=actual
 xml=work/f'{name}.html';subprocess.run(['pdftotext','-bbox',str(pdf),str(xml)],check=True)
 root=ET.parse(xml).getroot()
 pages=[e for e in root.iter() if e.tag.endswith('page')]
 for i,page in enumerate(pages):
  w=float(page.attrib['width']);h=float(page.attrib['height'])
  for word in page:
   if not word.tag.endswith('word'):continue
   a=word.attrib
   if float(a['xMin'])< -1 or float(a['yMin'])< -1 or float(a['xMax'])>w+1 or float(a['yMax'])>h+1:report['out_of_page_words'].append({'pdf':name,'page':i+1,'word':word.text})
 # All-build and backup pages cover every state, while compact has equivalent final states.
 if name!='generative-i':subprocess.run(['pdftoppm','-scale-to','960','-jpeg','-jpegopt','quality=90',str(pdf),str(work/name)],check=True)
images=sorted(work.glob('generative-i-all-builds-*.jpg'))+sorted(work.glob('generative-i-backup-*.jpg'))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
for start in range(0,len(images),9):
 panel=Image.new('RGB',(1440,885),'#d9d9d9');d=ImageDraw.Draw(panel)
 for j,p in enumerate(images[start:start+9]):
  x=j%3*480;y=j//3*295;im=Image.open(p).convert('RGB').resize((480,270));panel.paste(im,(x,y));d.text((x+8,y+274),p.stem,fill='black',font=font)
 panel.save(out/f'pdf-contact-{start//9+1:02}.jpg',quality=92)
report['rendered_pdf_states']=len(images);report['contact_sheets']=(len(images)+8)//9
(out/'pdf-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert not report['out_of_page_words']
