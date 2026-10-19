"""Verify page counts, searchable text, cue coverage and PDF link annotations."""
import json
from pathlib import Path
from pypdf import PdfReader
report={}
for name,expected in [('completed-scenes',64),('all-builds',151),('recap',2)]:
    r=PdfReader('exports/'+name+'.pdf');external=internal=0
    texts=[p.extract_text() for p in r.pages]
    for p in r.pages:
        for ref in p.get('/Annots',[]):
            a=ref.get_object()
            if a.get('/A',{}).get('/URI'):external+=1
            if a.get('/Dest') or a.get('/A',{}).get('/S')=='/GoTo':internal+=1
    report[name]=dict(pages=len(r.pages),expected=expected,selectableCharacters=sum(map(len,texts)),externalLinks=external,internalLinks=internal,sayCuePages=sum('SAY' in t for t in texts))
    assert len(r.pages)==expected and report[name]['selectableCharacters']>1000
    if name!='recap':
        assert external>0 and internal>0
        assert report[name]['sayCuePages']==expected-(1 if name=='completed-scenes' else 0)
Path('qa/pdf-checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
