"""Check portable PDF citations and local HTML resource targets."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from html import unescape
import json,re,subprocess

ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
 def __init__(self):super().__init__();self.hrefs=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.add(a['id'])
  if tag=='a' and 'href' in a:self.hrefs.append(a['href'])
parsed={}
for name in ['index.html','resources.html']:
 p=Links();p.feed((ROOT/name).read_text());parsed[name]=p
checks=[]
for name,doc in parsed.items():
 for href in doc.hrefs:
  u=urlsplit(href)
  if u.scheme in ['https','http','mailto','data']:continue
  target=unquote(u.path) or name
  assert (ROOT/target).exists(),(name,href)
  if u.fragment and target in parsed:assert unquote(u.fragment).split('/')[0] in parsed[target].ids,(name,href)
 checks.append(dict(name=f'All local targets and fragments resolve in {name}',ok=True))
pdf_links={}
for name in ['generative-i','generative-i-all-builds','generative-i-backup']:
 raw=subprocess.check_output(['pdftohtml','-xml','-hidden','-i','-stdout',str(ROOT/'output/pdf'/f'{name}.pdf')],text=True)
 links=[unescape(x) for x in re.findall('href="([^"]+)"',raw)]
 assert not [url for url in links if url.startswith('file:')],name
 pdf_links[name]=links
 checks.append(dict(name=f'PDF citations have no machine-specific file links: {name}',ok=True))
for part in ['2017-08-20-gan','2018-08-12-vae','fQ8D0-tTCBU','bo8wceOS6C0','R8V8CbuxryI','notebooks/vae.ipynb','jaan.io']:
 assert any(part in url for url in pdf_links['generative-i-backup']),part
checks.append(dict(name='All seven user-requested article, video and notebook links survive PDF export',ok=True))
(ROOT/'output/rebuild-qa/resource-links.json').write_text(json.dumps(dict(checks=checks,pdf_annotation_counts={k:len(v) for k,v in pdf_links.items()},scope='Presence, valid local targets and portable PDF annotations; not a claim that every remote server is continuously available.'),indent=2))
print(f'{len(checks)} resource and PDF-link checks passed')
