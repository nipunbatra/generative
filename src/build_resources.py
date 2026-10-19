from html import escape
from resource_catalog import RESOURCES

def build_resources(root, slides):
    sections=[]
    for group in dict.fromkeys(r['group'] for r in RESOURCES.values()):
        items=[]
        for id,r in RESOURCES.items():
            if r['group']!=group:continue
            related=[s for s in slides if any(v['id']==id for v in s.get('readings',[]))]
            anchors=' · '.join(f'<a href="index.html#{s["id"]}/1">{escape(s["title"])}</a>' for s in related[:5])
            items.append(f'<article id="{id}"><h3><a href="{escape(r["url"],quote=True)}" target="_blank" rel="noopener">{escape(r["title"])}</a></h3><p>{escape(r["description"])}</p>'+ (f'<p class="related">In the lecture: {anchors}</p>' if anchors else '')+'</article>')
        sections.append(f'<section><h2>{group}</h2>'+''.join(items)+'</section>')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Generative Modelling I: resources</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#fafaf8;color:#30343b;font:18px/1.55 system-ui,sans-serif}main{max-width:1040px;margin:0 auto;padding:52px 28px 80px}header{max-width:800px;margin-bottom:50px}.eyebrow{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#3478e5}h1{font-size:clamp(32px,6vw,52px);line-height:1.1;letter-spacing:-.035em;margin:12px 0 24px}h2{font-size:27px;margin:42px 0 10px}h3{font-size:21px;margin:0 0 8px}a{color:#2963bd;text-underline-offset:3px}a:focus-visible{outline:3px solid #8b5bb5;outline-offset:4px}article{border-top:1px solid #d9dde5;padding:22px 0;scroll-margin-top:28px}article:target{border-left:4px solid #8b5bb5;padding-left:20px}p{margin:8px 0}.related{font-size:15px;color:#596573}nav{display:flex;gap:20px;flex-wrap:wrap;font-size:16px}footer{margin-top:44px;padding-top:20px;border-top:1px solid #d9dde5;font-size:15px}
    </style></head><body><main><header><p class="eyebrow">ES667 · Nipun Batra</p><h1>Generative Modelling I<br>Resources and further reading</h1><p>The lecture connects probability models to the computations we train. These references provide the explanations, derivations and experiments behind that route.</p><nav><a href="index.html">Open lecture</a><a href="output/pdf/generative-i.pdf">Concept PDF</a><a href="notebooks/generative_i_lab.ipynb">Executed lab notebook</a><a href="SOURCES.md">Full provenance</a></nav></header>'''+''.join(sections)+'''<footer><p>The lecture’s new SVG diagrams and analytical examples are original constructions. Reference illustrations informed their explanations; the blog raster figures are not included. Measured model outputs use the saved local checkpoints.</p><p>External readings and videos need an internet connection. The lecture, local decoder, saved experiments and this resource index open offline.</p></footer></main></body></html>'''
    (root/'resources.html').write_text(doc)
