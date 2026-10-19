"""Assemble index.html from deck/slides/*.html.

    python scripts/build_deck.py

* <!--svg:NAME-->      is replaced by figures/diagrams/NAME.svg (inlined so its groups can build)
* {{a.b.0}}            is replaced by a value from figures/data/numbers.json
Every number on a slide therefore comes from the same run as the figures.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = json.loads((ROOT / "figures" / "data" / "numbers.json").read_text())
_extra = ROOT / "figures" / "data" / "numbers-extra.json"   # from scripts/make_figures_extra.py
if _extra.exists():
    NUM.update(json.loads(_extra.read_text()))
_gmm = ROOT / "figures" / "data" / "numbers-gmm.json"       # from scripts/make_figures_gmm.py, as {{gmm.*}}
if _gmm.exists():
    NUM["gmm"] = json.loads(_gmm.read_text())


def fmt(v):
    """numbers.json is already rounded for display; only swap in a typographic minus."""
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1000:
        return f"{v:,}".replace("-", "−")                    # 307720 -> 307,720
    return str(v).replace("-", "−") if isinstance(v, (int, float)) else str(v)


def derived():
    d = {}
    d["kl_neglogvar"] = [f"{-v:.2f}" for v in NUM["our3"]["logvar"]]
    bad = NUM["toygan"]["bad"]
    d["toygan_bad_left_pct"] = f"{100 * bad['left']:.0f}"
    d["toygan_bad_right_pct"] = f"{100 * bad['right']:.0f}"
    tiles = []
    for v in NUM["vote"]:
        cls = "real" if v["kind"] == "real" else "fake"
        tiles.append(
            f'<div class="vt {cls}"><img src="figures/data/vote-{v["n"]:02d}.png">'
            f'<p class="tile-lab"><span>{v["n"]}</span><span class="vk" data-b="2">{"real" if v["kind"] == "real" else "fake"} · D {v["d_real"]:.2f}</span></p></div>')
    d["vote_tiles"] = "\n      ".join(tiles)
    d["vote_tiles_hook"] = "\n      ".join(
        f'<div class="vt {"real" if v["kind"] == "real" else "fake"}"><img src="figures/data/vote-{v["n"]:02d}.png">'
        f'<p class="tile-lab"><span>{v["n"]}</span><span class="vk" data-b="2">{"real" if v["kind"] == "real" else "generated"}</span></p></div>'
        for v in NUM["vote"])
    d["cvae_labels"] = "".join(f"<span>{k}</span>" for k in range(10))
    d["vote_fakes"] = ", ".join(str(v["n"]) for v in NUM["vote"] if v["kind"] != "real")
    return d


def lookup(path):
    ctx = {**NUM, **derived()}
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


MACROS = {
    # role colours
    r"\cx": r"\textcolor{##245EDB}{#1}", r"\cenc": r"\textcolor{##8B2CDE}{#1}",
    r"\cdec": r"\textcolor{##0F766E}{#1}", r"\cz": r"\textcolor{##AA4E08}{#1}",
    r"\closs": r"\textcolor{##BE123C}{#1}", r"\cdisc": r"\textcolor{##334155}{#1}",
    # one spelling for every symbol, everywhere
    r"\N": r"\mathcal{N}", r"\KL": r"\mathrm{KL}", r"\E": r"\mathbb{E}", r"\ELBO": r"\mathrm{ELBO}",
    r"\eps": r"\varepsilon",
    # highlight the part of an equation that just changed
    r"\hl": r"\colorbox{##FFF1B8}{$\displaystyle #1$}",
}

# Two teaching sessions (about 75 minutes each). index.html holds both.
SESSIONS = {
    "part1.html": ("Session 1 · Fitting, autoencoders, the probabilistic view", ["01", "02", "03", "04"]),
    "part2.html": ("Session 2 · KL, variational inference, the VAE", ["05", "06", "07", "08", "09"]),
    "index.html": ("Generative Modelling I · VAEs", ["01", "02", "03", "04", "05", "06", "07", "08", "09"]),
}


PARTS = [
    ("--z", "Distributions and fitting"), ("--enc", "Autoencoders"), ("--x", "The probabilistic view"),
    ("--loss", "KL divergence"), ("--dec", "Variational inference and the ELBO"), ("--z", "Reparameterisation"),
    ("--enc", "The VAE on MNIST"),
]


def route(current):
    """The 7-part route with finished parts ticked and the current one highlighted."""
    items = []
    for i, (col, name) in enumerate(PARTS, 1):
        state = "done" if i < current else ("now" if i == current else "todo")
        items.append(f'<li class="{state}" style="--c:var({col})"><span class="dot">{"✓" if state == "done" else i}</span>{name}</li>')
    return '<ol class="route-mini">' + "".join(items) + "</ol>"


def assemble(prefixes):
    files = [f for f in sorted((ROOT / "deck" / "slides").glob("*.html")) if f.name[:2] in prefixes]
    body = "\n".join(f.read_text() for f in files)

    def svg(m):
        s_ = (ROOT / "figures" / "diagrams" / f"{m.group(1)}.svg").read_text()
        return s_.replace('xmlns="http://www.w3.org/2000/svg" ', "")

    body = re.sub(r"<!--svg:([\w-]+)-->", svg, body)
    body = re.sub(r"<!--route:(\d)-->", lambda m: route(int(m.group(1))), body)

    def num(m):
        v = lookup(m.group(1))
        return v if isinstance(v, str) else fmt(v)

    body = re.sub(r"\{\{([\w.]+)\}\}", num, body)
    missing = re.findall(r"\{\{[^}]*\}\}", body)
    assert not missing, missing
    return body


def page(title, body):
    macros = json.dumps(MACROS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="icon" href="data:,">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · ES 667</title>
<link rel="stylesheet" href="vendor/katex/katex.min.css">
<link rel="stylesheet" href="deck/deck.css">
</head>
<body>
<main id="deck">
{body}
</main>
<script>window.MATH_MACROS = {macros};</script>
<script src="vendor/katex/katex.min.js"></script>
<script src="vendor/katex/contrib/auto-render.min.js"></script>
<script src="deck/widgets.js"></script>
<script src="deck/deck.js"></script>
<script src="deck/annotate.js"></script>
</body>
</html>
"""


def main():
    for out, (title, prefixes) in SESSIONS.items():
        body = assemble(prefixes)
        (ROOT / out).write_text(page(title, body))
        print(f"{out}: {body.count('<section class=\"slide')} slides")


if __name__ == "__main__":
    main()
