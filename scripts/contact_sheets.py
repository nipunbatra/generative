from pathlib import Path
from PIL import Image,ImageDraw
import json
r=Path(__file__).resolve().parents[1];scenes=json.loads((r/'src/scene-manifest.json').read_text())
items=[(s['id'],s['buildCount']) for s in scenes]+[('a-gaussian-kl',2),('a-posterior-gap',2),('a-optimal-d',2)]
for start in range(0,len(items),12):
 group=items[start:start+12];sheet=Image.new('RGB',(1440,4*290),'white');d=ImageDraw.Draw(sheet)
 for i,(sid,b) in enumerate(group):
  p=r/'qa/screenshots'/f'{sid}-{b:02}.png';im=Image.open(p);im.thumbnail((480,270));x=i%3*480;y=i//3*290;sheet.paste(im,(x,y+20));d.text((x+10,y+3),sid,fill='black')
 sheet.save(r/'qa'/f'contact-{start//12+1}.jpg',quality=90)
