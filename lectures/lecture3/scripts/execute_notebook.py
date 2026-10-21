"""Execute the saved notebook with this Python environment; save outputs and QA."""
from pathlib import Path
import nbformat,time,json,base64,sys
from nbclient import NotebookClient
R=Path(__file__).resolve().parents[1]
path=R/(R.name+'.ipynb');n=nbformat.read(path,4);start=time.time()
NotebookClient(n,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(R)}}).execute()
nbformat.write(n,path);outputs=[o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[])]
figures=[o['data']['image/png'] for o in outputs if 'image/png' in o.get('data',{})]
out=R/'qa/notebook-figures';out.mkdir(exist_ok=True)
for i,s in enumerate(figures,1):(out/f'figure-{i:02}.png').write_bytes(base64.b64decode(s))
report={'passed':True,'seconds':time.time()-start,'interpreter':sys.executable,'codeCells':sum(c.cell_type=='code' for c in n.cells),'figures':len(figures),'errors':sum(o.output_type=='error' for o in outputs)}
(R/'qa/notebook-execution.json').write_text(json.dumps(report,indent=2));print(report)
