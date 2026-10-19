from pathlib import Path
import base64, csv, hashlib, html, json, re, subprocess
from story import story
from build_resources import build_resources
ROOT=Path(__file__).resolve().parents[1]
def image_uri(p):
    p=ROOT/p; mime={'.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml'}[p.suffix]
    return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
def main():
    slides=story(); mains=[s for s in slides if not s['appendix']]
    cumulative=0
    for s in mains:
        secs=s['planned_seconds']
        s.update(seconds=secs,start_seconds=cumulative,start_label=f'{cumulative//60:02d}:{cumulative%60:02d}')
        cumulative+=secs
    assert cumulative==5700
    for s in slides[len(mains):]:s.update(seconds=0,start_seconds=5700,start_label='Optional')
    m=json.loads((ROOT/'metrics/mnist.json').read_text());rows=''
    for name,vals in m['models'].items():rows+=f'<tr><td>{name}</td><td>{vals["test_reconstruction_nats"]:.2f}</td><td>{vals["test_kl_nats"]:.2f}</td></tr>'
    table='<table><thead><tr><th>Model</th><th>Test recon. nats</th><th>Test KL nats</th></tr></thead><tbody>'+rows+'</tbody></table>'
    parts=[];guide='# Teaching guide — Generative Modelling I\n\n'+str(len(mains))+' conceptual screens, '+str(sum(s['builds'] for s in mains))+' physical builds, 95 minutes. Five backup screens are excluded from the main route.\n\nControls: arrows/Space for builds; N next concept; S notes; Q discussion; O overview; P present/read. Phone opens in reading mode. Controls menu links PDF and notebook.\n\n'
    for i,s in enumerate(slides):
        body=s['body'].replace('<div id="experiment-table"></div>',table)
        body=re.sub(r'src="(figures/[^"]+)"',lambda a:'src="'+image_uri(a.group(1))+'"',body)
        reading=s.get('readings',[])
        links='<p class="reading-links"><strong>Further reading:</strong> '+ ' · '.join(f'<a href="{html.escape(r["url"],quote=True)}" target="_blank" rel="noopener">{html.escape(r["title"])}</a>' for r in reading)+'</p>' if reading else ''
        tag='h1' if i==0 else 'h2';sources=f'<a href="{s["source"]}" target="_blank" rel="noopener">Source</a>' if s['source'] else 'Authored course example'
        if reading:sources+=f' · <a href="resources.html#{reading[0]["id"]}">Reading</a>'
        parts.append(f'<section class="slide {"title" if i==0 else ""} {"appendix" if s["appendix"] else ""}" id="{s["id"]}" data-build="0" aria-labelledby="h-{s["id"]}" {"hidden" if s["appendix"] else ""}><header class="slide-head"><p class="eyebrow">{html.escape(s["section"])}</p><{tag} id="h-{s["id"]}">{html.escape(s["title"])}</{tag}></header><div class="body">{body}</div>'+ (f'<p class="takeaway">{s["idea"]}</p>' if s['idea'] else '')+f'<p class="credit"><span>{s["provenance"]} · {sources}</span><span>{i+1:02d} · ES667</span></p><details class="reading-notes"><summary>Discussion and presenter notes · {s["start_label"]}</summary><p>{html.escape(s["notes"])}</p><p><strong>Ask:</strong> {html.escape(s["question"])}</p><p class="answer"><strong>Expected answer:</strong> {html.escape(s["answer"])}</p>{links}</details></section>')
        guide+=f'## {i+1:02d}. {s["title"]}\n\nStart {s["start_label"]}; {s["seconds"]} seconds; {s["builds"]} builds.\n\n{s["notes"]}\n\n**Ask:** {s["question"]}\n\n**Expected answer:** {s["answer"]}\n\n'
        if reading:guide+='**Further reading:** '+ ' · '.join(f'[{r["title"]}]({r["url"]})' for r in reading)+'\n\n'
    body=''.join(parts)
    expressions=[html.unescape(t) for t in re.findall(r'data-math="([^"]+)"',body)]
    katex_js="const k=require('./vendor/katex/katex.js');let s='';process.stdin.on('data',x=>s+=x);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(t=>k.renderToString(t,{displayMode:true,throwOnError:true,strict:'error',output:'htmlAndMathml'})))));"
    rendered=json.loads(subprocess.run(['node','-e',katex_js],input=json.dumps(expressions),text=True,cwd=ROOT,capture_output=True,check=True).stdout); iterator=iter(rendered)
    body=re.sub(r'(<div class="math[^"]*" data-math="[^"]+">)</div>',lambda a:a.group(1)+next(iterator)+'</div>',body)
    vae=json.loads((ROOT/'models/vae2-browser.json').read_text());ae=json.loads((ROOT/'models/ae2-browser.json').read_text());grid=vae.pop('recorded_grid');pixels=bytes(max(0,min(255,round(p*255))) for row in grid for p in row);vae['recorded_pixels']=base64.b64encode(pixels).decode();ae.pop('recorded_grid')
    meta=[{k:v for k,v in s.items() if k!='body'} for s in slides]
    payload=json.dumps(dict(slides=meta,vae2=vae,ae2=ae,mixture=json.loads((ROOT/'metrics/mixture-fit.json').read_text()),photo=image_uri('images/newfoundland.png'), numbers=json.loads((ROOT/'results/generative_i_teaching_numbers.json').read_text()), evidence=json.loads((ROOT/'results/rebuild-mnist.json').read_text()), gan=json.loads((ROOT/'results/toy-gan.json').read_text()), pedagogy=json.loads((ROOT/'results/pedagogy.json').read_text()), bridge=[image_uri(f'figures/bridge-dog-{t:04}.png') for t in [0,100,300,600,900,1000]], grid=image_uri('figures/vae2-grid.png')),separators=(',',':')).replace('</','<\\/')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="ES667 Generative Modelling I: distributions, priors, latent variables, VAEs and GANs. Interactive lecture with real MNIST experiments."><link rel="icon" href="data:,"><title>Generative Modelling I — From Randomness to Data</title><link rel="stylesheet" href="vendor/katex/katex.min.css"><style>'''+(ROOT/'src/lecture.css').read_text()+(ROOT/'src/rebuild.css').read_text()+(ROOT/'src/pedagogy.css').read_text()+(ROOT/'src/probability.css').read_text()+'''</style></head><body class="present"><script>if(innerWidth<760){document.body.className='read';}</script><nav class="toolbar" aria-label="Lecture navigation"><button id="prev" aria-label="Previous build">←</button><span class="counter" aria-live="polite"></span><button id="next-btn" aria-label="Next build">→</button><button id="tools-button" aria-expanded="false" aria-controls="tools">Controls</button><div class="tools" id="tools" hidden><button id="mode">Present / read (P)</button><button id="overview">Overview (O)</button><button id="notes">Presenter notes (S)</button><button id="backup">Show backup</button><a href="output/pdf/generative-i.pdf">Conceptual-slide PDF</a><a href="output/pdf/generative-i-all-builds.pdf">Every-build PDF</a><a href="notebooks/generative_i_lab.ipynb">Companion notebook</a><a href="resources.html">Resources and further reading</a><p>Arrows / Space: next build<br>N: next screen · Q: discuss<br>Sliders and decoder grid also work offline.</p></div></nav><main class="stage">'''+body+'''</main><footer><a href="README.md">Reproduction instructions</a> · <a href="resources.html">Resources and further reading</a> · <a href="SOURCES.md">Provenance</a> · <a href="TEACHING_GUIDE.md">Teaching guide</a> · <a href="GENERATIVE_I_V4_CHECKLIST.md">Verified rebuild checklist</a><p>Illustrative diagrams, exact calculations and measured model outputs are explicitly labelled.</p></footer><dialog id="dialog"><div class="dialog-head"><h2 id="dialog-title"></h2><button id="dialog-close">Close</button></div><div id="dialog-content"></div></dialog><script id="lesson-data" type="application/json">'''+payload+'''</script><script>'''+(ROOT/'src/figures.js').read_text()+'''</script><script>'''+(ROOT/'src/rebuild.js').read_text()+'\n'+(ROOT/'src/pedagogy.js').read_text()+'\n'+(ROOT/'src/probability.js').read_text()+'\n'+(ROOT/'src/lecture.js').read_text()+'''</script></body></html>'''
    build_resources(ROOT,slides)
    guide=guide.replace('Controls: arrows/Space', 'Full route: 95 minutes. For the 80-minute core, skip the nine screens labelled in GENERATIVE_I_V4_STORYBOARD.md; existing measured-demo allocations remain unchanged. [All linked resources](resources.html).\n\nControls: arrows/Space')
    (ROOT/'index.html').write_text(doc);(ROOT/'TEACHING_GUIDE.md').write_text(guide);(ROOT/'output/slide-manifest.json').write_text(json.dumps(meta,indent=2));(ROOT/'metrics/math-manifest.json').write_text(json.dumps({'expressions':expressions,'rendered':len(rendered),'renderer':'KaTeX','main_screens':len(mains),'main_builds':sum(s['builds'] for s in mains),'duration_seconds':5700},indent=2))
    with (ROOT/'output/slide-manifest.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','title','section','builds','seconds','start_label','appendix','detour'],extrasaction='ignore');writer.writeheader();writer.writerows(meta)
    print(f'Built {len(mains)} screens / {sum(s["builds"] for s in mains)} builds + 5 backups; {len(rendered)} KaTeX expressions; {len(doc)/1e6:.1f} MB HTML')
if __name__=='__main__':main()
