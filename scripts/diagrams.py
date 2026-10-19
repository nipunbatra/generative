"""Hand-built SVG diagrams for Generative Modelling I.

    python scripts/diagrams.py      # writes figures/diagrams/*.svg

Each diagram is plain SVG. Groups that appear on later clicks carry data-b="n";
the slide runtime reveals them. Numbers come from figures/data/numbers.json, so
the diagram for "our 3" shows what the trained model actually computed.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.loads((ROOT / "figures" / "data" / "numbers.json").read_text())

INK, INK2, INK3, LINE = "#14171F", "#4A5160", "#6B7280", "#D9DDE5"
ROLE = {
    "x": ("#245EDB", "#E4ECFF"), "enc": ("#8B2CDE", "#F1E5FC"), "dec": ("#0F766E", "#D9F2EF"),
    "z": ("#AA4E08", "#FCEFD9"), "loss": ("#BE123C", "#FDE2E7"), "disc": ("#334155", "#E2E8F0"),
    "grey": ("#9AA1AD", "#F3F4F6"), "ink": (INK, "#FFFFFF"),
}
UI = "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif"
MATH = "KaTeX_Math, 'Times New Roman', serif"
MAIN = "KaTeX_Main, 'Times New Roman', serif"
IMG = "figures/data/"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fs(size):
    """Projector-friendly label sizes: nothing below 18 px."""
    return size if size >= 20 else round(max(18, size * 1.22))


MATHY = re.compile(r"[μσεϵφθψπΣ‖∂∼~²√∫∑]|(?<![\w])-\d")


def check_plain(s):
    """Plain SVG text must not contain maths: that goes through KaTeX (math / label)."""
    if MATHY.search(str(s)):
        raise ValueError(f"maths in plain text, use d.math or d.label: {s!r}")


class SVG:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.items, self.stack = [], []
        self.markers = {}

    # ------------------------------------------------------------ structure
    def add(self, s):
        self.items.append(s)

    def begin(self, b=None, cls=None, extra=""):
        attrs = ""
        if b is not None:
            attrs += f' data-b="{b}"'
        if cls:
            attrs += f' class="{cls}"'
        self.add(f"<g{attrs}{extra}>")

    def end(self):
        self.add("</g>")

    def marker(self, color):
        mid = f"{self.name}-ah-{color.strip('#')}"
        if mid not in self.markers:
            self.markers[mid] = (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                                 f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
        return mid

    # ------------------------------------------------------------ primitives
    def rect(self, x, y, w, h, stroke, fill, r=12, sw=2.2, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def text(self, x, y, s, size=20, color=INK, weight=500, anchor="middle", italic=False):
        check_plain(s)
        size = fs(size)
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{UI}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                 f'text-anchor="{anchor}"{st}>{esc(s)}</text>')

    def label(self, x, y, html, size=20, color=INK, weight=500, anchor="middle", width=760):
        """Text that may contain $...$ maths. Rendered as HTML so KaTeX typesets it.
        (x, y) is the baseline point, as for SVG <text>."""
        size = fs(size)
        html = html.replace("<", "&lt;").replace(">", "&gt;")  # maths like x_{<i} must not become HTML tags
        x0 = {"middle": x - width / 2, "start": x, "end": x - width}[anchor]
        align = {"middle": "center", "start": "left", "end": "right"}[anchor]
        h = size * 2.2
        self.add(f'<foreignObject x="{x0:.1f}" y="{y - size * 1.25:.1f}" width="{width}" height="{h:.1f}" overflow="visible">'
                 f'<div class="fo" style="font-size:{size}px;color:{color};font-weight:{weight};text-align:{align};'
                 f'height:{size * 1.6:.1f}px;line-height:{size * 1.6:.1f}px">{html}</div></foreignObject>')

    def math(self, x, y, tex, size=24, color=INK, anchor="middle", weight=400):
        """Pure maths, typeset by KaTeX."""
        self.label(x, y, f"${tex}$", size, color, weight, anchor)

    def line(self, pts, color=INK3, sw=2.4, head=True, dash=None, start_head=False):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        m = self.marker(color)
        h = f' marker-end="url(#{m})"' if head else ""
        if start_head:
            h += f' marker-start="url(#{m})"'
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{ds}{h}/>')

    def curve(self, d, color=INK3, sw=2.4, head=True, dash=None):
        m = self.marker(color)
        h = f' marker-end="url(#{m})"' if head else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{h}/>')

    def image(self, x, y, w, h, href, role="x", border=True, sw=2.5):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{IMG}{href}" preserveAspectRatio="none" '
                 f'style="image-rendering:pixelated"/>')
        if border:
            c = ROLE[role][0]
            self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{c}" stroke-width="{sw}"/>')

    def node(self, x, y, w, h, role, title, sub=None, size=22, dash=None):
        c, t = ROLE[role]
        self.rect(x, y, w, h, c, t, dash=dash)
        cy = y + h / 2 + (size * 0.35 if not sub else -2)
        self.label(x + w / 2, cy, title, size, c, 650, width=max(w, 300)) if "$" in title else self.text(x + w / 2, cy, title, size, c, 650)
        if sub:
            self.label(x + w / 2, cy + size * 0.95, sub, 15, INK3, 500, width=max(w, 300)) if "$" in sub \
                else self.text(x + w / 2, cy + size * 0.95, sub, 15, INK3, 500)

    def trapezoid(self, x, y_mid, w, hl, hr, role, title, sub=None, dash=None, opacity=None):
        c, t = ROLE[role]
        pts = [(x, y_mid - hl / 2), (x + w, y_mid - hr / 2), (x + w, y_mid + hr / 2), (x, y_mid + hl / 2)]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity else ""
        self.add(f'<polygon points="{" ".join(f"{a},{b}" for a, b in pts)}" fill="{t}" stroke="{c}" '
                 f'stroke-width="2.4" stroke-linejoin="round"{d}{o}/>')
        self.text(x + w / 2, y_mid + (8 if not sub else -2), title, 23, c, 650)
        if sub:
            if "$" in sub:
                self.label(x + w / 2, y_mid + 24, sub, 14, INK3, 500, width=max(w, 300))
            else:
                self.text(x + w / 2, y_mid + 22, sub, 14, INK3, 500)

    def vec(self, x, y, values, role, cw=70, ch=40, vertical=True, fmt="{:.2f}", size=17):
        c, t = ROLE[role]
        for i, v in enumerate(values):
            xx, yy = (x, y + i * ch) if vertical else (x + i * cw, y)
            self.add(f'<rect x="{xx}" y="{yy}" width="{cw}" height="{ch}" fill="{t}" stroke="{c}" stroke-width="2"/>')
            s = fmt.format(v) if not isinstance(v, str) else v
            s = s.replace("-", "−")
            self.add(f'<text x="{xx + cw / 2}" y="{yy + ch / 2 + 6}" font-family="SF Mono, Menlo, monospace" '
                     f'font-size="{size}" fill="{INK}" text-anchor="middle">{esc(s)}</text>')

    def cloud(self, cx, cy, rx, ry, role="enc"):
        c, t = ROLE[role]
        for k, op in ((1.0, .14), (0.7, .22), (0.4, .35)):
            self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx * k}" ry="{ry * k}" fill="{c}" fill-opacity="{op}"/>')
        self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{c}" stroke-width="2"/>')

    def plane(self, cx, cy, r, rings=True, label=True, color=None):
        color = color or ROLE["z"][0]
        self.add(f'<line x1="{cx - r}" y1="{cy}" x2="{cx + r}" y2="{cy}" stroke="#C7CCD4" stroke-width="1.4"/>')
        self.add(f'<line x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy + r}" stroke="#C7CCD4" stroke-width="1.4"/>')
        if rings:
            for k in (0.33, 0.66, 1.0):
                self.add(f'<circle cx="{cx}" cy="{cy}" r="{r * k * 0.92}" fill="none" stroke="{color}" '
                         f'stroke-opacity=".55" stroke-width="1.4" stroke-dasharray="4 3"/>')
        if label:
            self.math(cx + r + 4, cy + 24, "z_1", 17, INK3, "end", )
            self.math(cx + 14, cy - r + 12, "z_2", 17, INK3, "start")

    def dice(self, x, y, s=54, role="z"):
        c, t = ROLE[role]
        self.rect(x, y, s, s, c, t, r=10, sw=2.4)
        for dx, dy in ((.27, .27), (.73, .27), (.5, .5), (.27, .73), (.73, .73)):
            self.add(f'<circle cx="{x + s * dx}" cy="{y + s * dy}" r="{s * .075}" fill="{c}"/>')

    def lock(self, x, y, color=INK2, s=1.0):
        self.add(f'<g transform="translate({x},{y}) scale({s})"><path d="M-9,-4 v-7 a9,9 0 0 1 18,0 v7" fill="none" '
                 f'stroke="{color}" stroke-width="3"/><rect x="-13" y="-4" width="26" height="20" rx="4" fill="{color}"/>'
                 f'<circle cx="0" cy="5" r="3" fill="#fff"/></g>')

    def cross(self, x, y, s=14, color=ROLE["loss"][0], sw=4):
        self.add(f'<path d="M{x - s},{y - s} L{x + s},{y + s} M{x + s},{y - s} L{x - s},{y + s}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"/>')

    def scissors(self, x, y, color=ROLE["loss"][0]):
        self.add(f'<g transform="translate({x},{y})" fill="none" stroke="{color}" stroke-width="2.6">'
                 f'<circle cx="-7" cy="10" r="5"/><circle cx="7" cy="10" r="5"/>'
                 f'<path d="M-4,6 L8,-12 M4,6 L-8,-12"/></g>')

    def save(self):
        defs = "<defs>" + "".join(self.markers.values()) + "</defs>"
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" class="diagram" '
               f'role="img" aria-label="{esc(self.name)}">{defs}{"".join(self.items)}</svg>')
        (OUT / f"{self.name}.svg").write_text(svg)


C = {k: v[0] for k, v in ROLE.items()}
our = NUM["our3"]


def m(v, k=2):
    """A number for KaTeX: proper minus sign, fixed decimals."""
    return f"{v:.{k}f}".replace("-", "-")


# =============================================================== hook
def common():
    d = SVG("common", 1152, 330)
    d.begin(1)
    d.dice(70, 115, 70)
    d.text(105, 225, "randomness", 19, C["z"], 650)
    d.math(105, 256, r"z \sim \N(0, I)", 20, INK2)
    d.end()
    d.begin(3)
    d.rect(10, 0, 470, 84, C["grey"], ROLE["grey"][1], dash="6 4")
    d.text(245, 28, "condition (optional): what to make", 17, INK2, 650)
    d.text(245, 62, "a label “7” · a sentence · a sketch · a place and date", 15, INK3)
    d.line([(420, 84), (440, 112)], C["grey"])
    d.end()
    d.begin(1)
    d.line([(150, 150), (330, 150)], C["z"])
    d.trapezoid(340, 150, 300, 110, 230, "dec", "learned generator", r"a neural network with weights $\theta$")
    d.line([(650, 150), (790, 150)], C["dec"])
    d.end()
    d.begin(2)
    d.image(800, 40, 220, 220, "vae-prior-one.png", "dec")
    d.label(910, 298, r"a new sample $x$", 19, C["dec"], 650)
    d.end()
    d.save()


def two_ways():
    d = SVG("two-ways", 1152, 400)
    d.begin(1)
    d.text(260, 28, "A · one pixel at a time", 22, INK, 650)
    n, s0, x0, y0 = 10, 26, 130, 60
    shape = ["..........", "...####...", "..#....#..", ".......#..", "....###...", ".......#..",
             "..#....#..", "...####...", "..........", ".........."]
    done = 34
    for k in range(n * n):
        r_, c_ = divmod(k, n)
        ink = shape[r_][c_] == "#"
        fill = (INK if ink else "#FFFFFF") if k < done else (C["z"] if k == done else "#F0F1F4")
        d.add(f'<rect x="{x0 + c_ * s0}" y="{y0 + r_ * s0}" width="{s0 - 2}" height="{s0 - 2}" rx="3" fill="{fill}" '
              f'stroke="#D9DDE5" stroke-width="1"/>')
    for r_ in range(4):
        y = y0 + r_ * s0 + s0 / 2 - 1
        d.line([(x0 - 14, y), (x0 + n * s0 + 4, y)], C["z"], 1.6, head=True, dash="3 3")
    d.text(260, 348, "784 predictions in a row, like our names", 18, INK2)
    d.math(260, 386, r"p(x) = \prod_{i} p(x_i \mid x_{<i})", 22, INK2)
    d.end()
    d.begin(2)
    d.add(f'<line x1="560" y1="40" x2="560" y2="370" stroke="{LINE}" stroke-width="2"/>')
    d.text(860, 28, "B · all pixels from one random vector", 22, INK, 650)
    d.vec(620, 140, our["z"], "z", cw=86, ch=44)
    d.math(663, 128, "z", 24, C["z"])
    d.line([(712, 185), (742, 185)], C["z"])
    d.trapezoid(746, 185, 170, 90, 200, "dec", "decoder")
    d.line([(920, 185), (950, 185)], C["dec"])
    d.image(956, 95, 180, 180, "digit3-pi.png", "dec")
    d.text(860, 348, "one pass: 2 numbers in, 784 pixels out", 18, INK2)
    d.math(860, 386, r"x = \text{decoder}(z)", 22, INK2)
    d.end()
    d.save()


def mixture_story():
    import math
    d = SVG("mixture-story", 1152, 270)
    d.begin(1)
    d.text(140, 28, "step 1 · hidden choice", 19, C["z"], 650)
    for lab, cx in (("A", 95), ("B", 185)):
        d.add(f'<circle cx="{cx}" cy="110" r="44" fill="{ROLE["z"][1]}" stroke="{C["z"]}" stroke-width="2.5"/>')
        d.text(cx, 122, lab, 34, C["z"], 700)
    d.text(140, 190, "flip a fair coin:", 17, INK2)
    d.math(140, 228, r"p(z{=}\mathrm{A}) = p(z{=}\mathrm{B}) = \tfrac12", 20, INK2)
    d.end()
    d.begin(2)
    d.line([(250, 110), (330, 110)], C["z"])
    d.text(720, 28, "step 2 · draw x from the chosen bump", 19, C["dec"], 650)
    base, x0, x1 = 190, 360, 1100

    def X(v):
        return x0 + (v + 4.5) / 9 * (x1 - x0)

    for mu, chosen in ((-2, False), (2, True)):
        pts = [f"{X(mu - 1.6 + 3.2 * k / 120):.1f},{base - 120 * math.exp(-0.5 * ((3.2 * k / 120 - 1.6) / 0.5) ** 2):.1f}"
               for k in range(121)]
        col = C["dec"] if chosen else "#C7CCD4"
        d.add(f'<polyline points="{" ".join(pts)}" fill="{ROLE["dec"][1] if chosen else "none"}" stroke="{col}" stroke-width="2.6"/>')
    d.add(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="#9AA1AD" stroke-width="1.5"/>')
    for v in (-4, -2, 0, 2, 4):
        d.math(X(v), base + 28, str(v), 15, INK3)
    d.text(X(-2), base - 132, "bump A", 16, INK3)
    d.label(X(2), base - 132, r"bump B &nbsp;← $z = \mathrm{B}$", 16, C["dec"], 650)
    d.end()
    d.begin(3)
    d.add(f'<circle cx="{X(1.62)}" cy="{base}" r="10" fill="{C["z"]}" stroke="#fff" stroke-width="2"/>')
    d.math(X(1.62), base + 62, "x = 1.62", 20, C["z"])
    d.end()
    d.save()


def latent_story():
    d = SVG("latent-story", 1152, 340)
    z = NUM["story_z"]
    d.begin(1)
    d.text(150, 26, "1 · draw a code", 19, C["z"], 650)
    d.plane(150, 160, 105)
    zx, zy = 150 + z[0] * 32, 160 - z[1] * 32
    d.add(f'<circle cx="{zx}" cy="{zy}" r="9" fill="{C["z"]}" stroke="#fff" stroke-width="2"/>')
    d.math(150, 306, rf"z \sim \N(0, I),\ \ z = ({m(z[0])},\, {m(z[1])})", 18, INK2)
    d.end()
    d.begin(2)
    d.line([(270, 160), (310, 160)], C["z"])
    d.trapezoid(316, 160, 150, 80, 170, "dec", "decoder", r"weights $\theta$")
    d.line([(470, 160), (505, 160)], C["dec"])
    d.text(610, 26, "2 · decode probabilities", 19, C["dec"], 650)
    d.image(512, 62, 196, 196, "story-pi.png", "dec")
    d.math(610, 306, r"\pi = \mathrm{sigmoid}(\text{decoder}(z))", 18, INK2)
    d.end()
    d.begin(3)
    d.line([(714, 160), (790, 160)], C["x"])
    d.text(958, 26, "3 · flip each pixel", 19, C["x"], 650)
    d.image(860, 62, 196, 196, "story-sample.png", "x")
    d.math(958, 306, r"x_i \sim \mathrm{Bernoulli}(\pi_i)", 18, INK2)
    d.end()
    d.save()


# =============================================================== autoencoders
def ae():
    d = SVG("ae", 1152, 410)
    code = NUM["our3_ae"]
    d.begin(1)
    d.image(20, 40, 200, 200, "digit3.png", "x")
    d.label(120, 278, r"$x$: 28 × 28 = 784 pixels", 17, C["x"], 650)
    d.line([(226, 140), (262, 140)], C["x"])
    d.trapezoid(266, 140, 210, 230, 84, "enc", "encoder", "784 → 512 → 256 → 2")
    d.end()
    d.begin(2)
    d.line([(480, 140), (512, 140)], C["enc"])
    d.vec(516, 98, code, "z", cw=92, ch=42, fmt="{:.1f}")
    d.math(562, 84, "z", 26, C["z"])
    d.text(562, 216, "the code:", 16, INK2)
    d.text(562, 240, "2 numbers", 16, INK2)
    d.end()
    d.begin(3)
    d.line([(612, 140), (644, 140)], C["z"])
    d.trapezoid(648, 140, 210, 84, 230, "dec", "decoder", "2 → 256 → 512 → 784")
    d.line([(862, 140), (898, 140)], C["dec"])
    d.image(904, 40, 200, 200, "digit3-ae2-recon.png", "dec")
    d.label(1004, 278, r"$\hat x$: the reconstruction", 17, C["dec"], 650)
    d.end()
    d.begin(4)
    d.curve("M1004,292 C1004,330 120,330 120,288", C["loss"], 2.6)
    d.label(562, 386, r"loss: how badly does $\hat x$ match $x$, pixel by pixel?", 19, C["loss"], 650)
    d.end()
    d.save()


def ae_generate():
    d = SVG("ae-generate", 1152, 300)
    d.begin(1)
    d.add('<g opacity=".35">')
    d.image(20, 50, 160, 160, "digit3.png", "grey")
    d.trapezoid(206, 130, 190, 190, 76, "grey", "encoder")
    d.add("</g>")
    d.cross(300, 130, 40, C["loss"], 6)
    d.text(210, 270, "throw the encoder away", 18, INK2, 650)
    d.end()
    d.begin(2)
    d.dice(454, 84, 60)
    d.math(484, 186, r"z \sim \N(0, I)", 20, C["z"])
    d.text(484, 214, "pick a random code", 16, INK2)
    d.line([(522, 114), (580, 114)], C["z"])
    d.trapezoid(586, 114, 210, 76, 200, "dec", "decoder", "same weights as before")
    d.line([(800, 114), (840, 114)], C["dec"])
    d.end()
    d.begin(3)
    d.rect(846, 24, 180, 180, C["dec"], "#fff", r=8, dash="7 5")
    d.text(936, 136, "?", 80, C["dec"], 700)
    d.text(936, 240, "a new digit?", 19, C["dec"], 650)
    d.end()
    d.save()


def cloud_vs_point():
    d = SVG("cloud-vs-point", 1152, 400)
    for cx, title, vae in ((290, "autoencoder: each image → one point", False), (870, "VAE: each image → a small cloud", True)):
        d.begin(1 if not vae else 2)
        d.text(cx, 26, title, 21, C["enc"] if vae else INK, 650)
        d.plane(cx, 210, 140, rings=vae, label=False)
        a, b = (cx - 66, 160), (cx + 62, 252)
        # the two training images, linked to their codes
        d.image(cx - 270, 70, 74, 74, "digit3.png", "x", sw=2)
        d.image(cx + 196, 270, 74, 74, "digit7.png", "x", sw=2)
        d.line([(cx - 192, 110), (a[0] - 12, a[1] - 4)], C["enc"], 1.8, dash="4 3")
        d.line([(cx + 194, 300), (b[0] + 12, b[1] + 4)], C["enc"], 1.8, dash="4 3")
        if vae:
            d.cloud(a[0], a[1], 46, 30)
            d.cloud(b[0], b[1], 40, 34)
            for dx, dy in ((-18, 8), (14, -10), (30, 6)):
                d.add(f'<circle cx="{a[0] + dx}" cy="{a[1] + dy}" r="5" fill="{C["z"]}"/>')
            d.label(cx, 384, r"training: every code in the 3's cloud must decode back to the 3", 16, INK2, width=560)
        else:
            for pt in (a, b):
                d.add(f'<circle cx="{pt[0]}" cy="{pt[1]}" r="8" fill="{C["enc"]}"/>')
            d.text(a[0] + 4, a[1] - 18, "code of the 3", 15, C["enc"], 650)
            d.text(b[0] - 4, b[1] + 30, "code of the 7", 15, C["enc"], 650)
            m = (cx + 6, 206)
            d.add(f'<circle cx="{m[0]}" cy="{m[1]}" r="9" fill="#fff" stroke="{C["z"]}" stroke-width="3"/>')
            d.text(m[0] + 66, m[1] - 4, "a random code", 15, C["z"], 650)
            d.text(m[0] + 66, m[1] + 16, "decodes to ... ?", 15, C["z"], 650)
            d.text(cx, 384, "training only ever used the two purple points", 16, INK2)
        d.end()
    d.save()


# =============================================================== VAE, one stage at a time
def vae_encode():
    d = SVG("vae-encode", 1152, 400)
    mu, lv, sd = our["mu"], our["logvar"], our["sigma"]
    d.begin(1)
    d.image(20, 90, 200, 200, "digit3.png", "x")
    d.math(120, 330, "x", 26, C["x"])
    d.line([(226, 190), (262, 190)], C["x"])
    d.trapezoid(266, 190, 220, 240, 100, "enc", "encoder", r"weights $\varphi$")
    d.line([(490, 190), (512, 190), (512, 105), (540, 105)], C["enc"])
    d.line([(512, 190), (512, 275), (540, 275)], C["enc"])
    d.vec(544, 63, mu, "enc", cw=96, ch=42)
    d.math(592, 50, r"\mu", 26, C["enc"])
    d.vec(544, 233, lv, "enc", cw=96, ch=42)
    d.math(592, 220, r"\log\sigma^2", 24, C["enc"])
    d.end()
    d.begin(2)
    d.line([(646, 275), (700, 275)], C["enc"])
    d.label(800, 220, r"$\sigma = \exp\!\big(\tfrac12 \log\sigma^2\big)$", 21, INK2)
    d.vec(704, 233, sd, "enc", cw=96, ch=42, fmt="{:.3f}")
    d.math(752, 345, r"\sigma", 26, C["enc"])
    d.end()
    d.begin(3)
    d.label(1000, 110, r"a cloud centred at $\mu$", 19, C["enc"], 650, width=320)
    d.label(1000, 140, r"with width $\sigma$:", 19, C["enc"], 650, width=320)
    d.label(1000, 180, r"$q_\varphi(z\mid x) = \N(\mu, \sigma^2)$", 22, C["enc"], 400, width=320)
    d.end()
    d.save()


def vae_sample():
    d = SVG("vae-sample", 1152, 430)
    mu, sd, eps, z = our["mu"], our["sigma"], our["eps"], our["z"]
    d.begin(1)
    d.text(230, 28, "the cloud in code space", 19, C["enc"], 650)
    d.plane(230, 220, 170)
    S = 52
    d.cloud(230 + mu[0] * S, 220 - mu[1] * S, 22, 22)
    d.add(f'<circle cx="{230 + mu[0] * S}" cy="{220 - mu[1] * S}" r="5" fill="{C["enc"]}"/>')
    d.end()
    d.begin(2)
    d.add(f'<circle cx="520" cy="110" r="34" fill="{ROLE["z"][1]}" stroke="{C["z"]}" stroke-width="2.4" stroke-dasharray="5 3"/>')
    d.math(520, 120, r"\varepsilon", 30, C["z"])
    d.math(520, 178, rf"\varepsilon = ({m(eps[0])},\, {m(eps[1])})", 20, C["z"])
    d.text(520, 210, "drawn from N(0, I) once".replace("N(0, I)", "a standard normal"), 16, INK3)
    d.end()
    d.begin(3)
    d.label(860, 96, r"$z_1 = \mu_1 + \sigma_1\varepsilon_1$", 22, INK, 400, width=520)
    d.label(860, 136, rf"$= {m(mu[0])} + {sd[0]:.3f} \times {m(eps[0])} = {m(z[0])}$", 22, INK, 400, width=520)
    d.end()
    d.begin(4)
    d.label(860, 206, r"$z_2 = \mu_2 + \sigma_2\varepsilon_2$", 22, INK, 400, width=520)
    d.label(860, 246, rf"$= {m(mu[1])} + {sd[1]:.3f} \times ({m(eps[1])}) = {m(z[1])}$", 22, INK, 400, width=520)
    d.end()
    d.begin(5)
    d.add(f'<circle cx="{230 + z[0] * S}" cy="{220 - z[1] * S}" r="7" fill="{C["z"]}" stroke="#fff" stroke-width="1.5"/>')
    d.vec(760, 300, z, "z", cw=100, ch=46, vertical=False)
    d.math(700, 330, "z =", 24, C["z"], "end")
    d.text(860, 380, "one code from inside the cloud", 18, C["z"], 650)
    d.end()
    d.save()


def vae_decode():
    d = SVG("vae-decode", 1152, 420)
    z = our["z"]
    d.begin(1)
    d.vec(20, 130, z, "z", cw=96, ch=46)
    d.math(68, 116, "z", 26, C["z"])
    d.line([(120, 176), (156, 176)], C["z"])
    d.trapezoid(160, 176, 220, 100, 240, "dec", "decoder", r"weights $\theta$")
    d.line([(384, 176), (416, 176)], C["dec"])
    d.image(420, 56, 240, 240, "digit3-pi.png", "dec")
    d.label(540, 330, r"$\pi$: 784 pixel probabilities", 18, C["dec"], 650)
    d.end()
    d.begin(2)
    d.image(720, 56, 160, 160, "digit3.png", "x")
    d.label(800, 248, r"the real $x$", 18, C["x"], 650)
    d.node(710, 280, 430, 74, "loss", rf"reconstruction: $-\log p_\theta(x\mid z) = {our['recon']:.1f}$", "BCE, summed over 784 pixels", size=20)
    d.end()
    d.begin(3)
    d.node(710, 360, 430, 58, "loss", rf"$+\ \KL\big(q_\varphi(z\mid x)\,\|\,\N(0,I)\big) = {our['kl_total']:.2f}$", None, size=20)
    d.end()
    d.save()


def vae():
    d = SVG("vae", 1152, 500)
    mu, lv, eps, z = our["mu"], our["logvar"], our["eps"], our["z"]
    d.image(0, 120, 150, 150, "digit3.png", "x")
    d.math(75, 304, "x", 26, C["x"])
    d.line([(154, 195), (178, 195)], C["x"])
    d.trapezoid(182, 195, 150, 200, 86, "enc", "encoder", r"$\varphi$")
    d.line([(334, 195), (352, 195), (352, 118), (372, 118)], C["enc"])
    d.line([(352, 195), (352, 286), (372, 286)], C["enc"])
    d.vec(376, 76, mu, "enc", cw=80, ch=42)
    d.math(416, 66, r"\mu", 24, C["enc"])
    d.vec(376, 244, lv, "enc", cw=80, ch=42)
    d.math(416, 234, r"\log\sigma^2", 22, C["enc"])
    d.label(612, 36, r"$q_\varphi(z\mid x)$: a small cloud of codes", 18, C["enc"], 650)
    d.plane(612, 190, 118)
    d.cloud(612 + mu[0] * 38, 190 - mu[1] * 38, 28, 28)
    d.line([(458, 118), (560, 210)], C["enc"], 1.8, dash="4 3")
    d.line([(458, 286), (560, 262)], C["enc"], 1.8, dash="4 3")
    d.add(f'<circle cx="612" cy="400" r="30" fill="{ROLE["z"][1]}" stroke="{C["z"]}" stroke-width="2.4"/>')
    d.math(612, 410, r"\varepsilon", 28, C["z"])
    d.math(612, 466, rf"\varepsilon = ({m(eps[0])},\, {m(eps[1])})", 18, INK2)
    d.line([(644, 394), (742, 352)], C["z"])
    d.node(746, 316, 170, 64, "z", r"$z = \mu + \sigma\odot\varepsilon$", size=20)
    d.vec(774, 236, z, "z", cw=58, ch=40, vertical=False, size=16)
    d.line([(831, 314), (831, 280)], C["z"], 2)
    d.add(f'<circle cx="{612 + z[0] * 38}" cy="{190 - z[1] * 38}" r="6" fill="{C["z"]}" stroke="#fff" stroke-width="1.5"/>')
    d.line([(916, 348), (932, 348), (932, 196), (944, 196)], C["z"])
    d.trapezoid(948, 196, 92, 64, 140, "dec", "dec.", None)
    d.line([(1042, 196), (1050, 196)], C["dec"])
    d.image(1052, 146, 100, 100, "digit3-pi.png", "dec")
    d.math(1102, 276, r"\pi", 22, C["dec"])
    d.node(930, 400, 222, 76, "loss", f"recon {our['recon']:.1f}", r"$-\log p_\theta(x\mid z)$", size=21)
    d.node(170, 380, 300, 76, "loss", rf"$\KL$ to $\N(0, I)$ = {our['kl_total']:.2f}", "cloud vs. prior", size=21)
    d.save()


def vae_generate():
    d = SVG("vae-generate", 1152, 280)
    d.begin(1)
    d.add('<g opacity=".3">')
    d.image(20, 60, 140, 140, "digit3.png", "grey")
    d.trapezoid(180, 130, 150, 160, 70, "grey", "encoder")
    d.add("</g>")
    d.text(175, 250, "not needed to generate", 17, INK3, 650)
    d.end()
    d.begin(2)
    d.plane(470, 130, 95)
    d.add(f'<circle cx="{470 + 0.9 * 30}" cy="{130 - 0.4 * 30}" r="9" fill="{C["z"]}" stroke="#fff" stroke-width="2"/>')
    d.math(470, 266, r"z \sim \N(0, I)", 20, C["z"])
    d.line([(575, 130), (610, 130)], C["z"])
    d.trapezoid(614, 130, 200, 80, 190, "dec", "decoder", r"trained weights $\theta$")
    d.line([(818, 130), (850, 130)], C["dec"])
    d.image(856, 40, 180, 180, "vae-prior-one.png", "dec")
    d.text(946, 252, "a brand-new digit", 18, C["dec"], 650)
    d.end()
    d.save()


def pgm():
    d = SVG("pgm", 1152, 330)
    d.begin(1)
    d.math(170, 198, r"p(z) = \N(0, I)", 21, C["z"])
    d.line([(272, 190), (330, 190)], C["z"])
    d.add(f'<circle cx="390" cy="190" r="56" fill="#fff" stroke="{C["z"]}" stroke-width="3"/>')
    d.math(390, 204, "z", 36, C["z"])
    d.text(390, 276, "hidden code", 17, INK3)
    d.add(f'<circle cx="760" cy="190" r="56" fill="{ROLE["x"][1]}" stroke="{C["x"]}" stroke-width="3"/>')
    d.math(760, 204, "x", 36, C["x"])
    d.text(760, 276, "observed image (shaded)", 17, INK3)
    d.line([(448, 190), (700, 190)], C["dec"], 3)
    d.math(575, 178, r"p_\theta(x\mid z)", 24, C["dec"])
    d.text(575, 220, "decoder: generate", 17, C["dec"], 650)
    d.end()
    d.begin(2)
    d.curve("M712,146 C640,70 510,70 436,142", C["enc"], 3, dash="9 6")
    d.math(575, 54, r"q_\varphi(z\mid x)", 26, C["enc"])
    d.label(575, 320, r"encoder: infer $z$ from $x$ (dashed = an approximation we learn)", 17, C["enc"], 650, width=900)
    d.end()
    d.save()


def vae_shapes():
    d = SVG("vae-shapes", 1152, 330)
    items = [("x", r"$x$", "[B, 784]"), ("enc", "Linear", "[B, 512]"), ("enc", "Linear", "[B, 256]"),
             ("enc", r"$\mu,\ \log\sigma^2$", "[B, 2] each"), ("z", r"$z$", "[B, 2]"), ("dec", "Linear", "[B, 256]"),
             ("dec", "Linear", "[B, 512]"), ("dec", "logits", "[B, 784]")]
    w, gap, x = 128, 9, 4
    for i, (role, title, shape) in enumerate(items):
        d.begin(1 if i < 4 else (2 if i == 4 else 3))
        d.node(x, 60, w, 112, role, title, shape, size=21)
        if i < len(items) - 1:
            d.line([(x + w + 1, 116), (x + w + gap - 1, 116)], INK3, 1.8)
        d.end()
        x += w + gap
    d.begin(1)
    d.label(280, 38, r"encoder $\varphi$", 19, C["enc"], 650)
    d.end()
    d.begin(3)
    d.label(890, 38, r"decoder $\theta$", 19, C["dec"], 650)
    d.end()
    d.begin(4)
    d.label(576, 238, r"B = batch size. Only the 2 in the middle is the code; $\varepsilon$ has the same shape as $\mu$.", 18, INK2, width=1100)
    d.text(576, 274, "Loss per image: BCE summed over 784 pixels + KL summed over 2 coordinates.", 18, INK2)
    d.text(576, 308, "Then average over the B images in the batch.", 18, INK2)
    d.end()
    d.save()


def pixels():
    p = NUM["patch"]
    d = SVG("pixels", 920, 360)
    s_, gap = 54, 3
    titles = [(r"\text{observed } x_i", "x"), (r"\text{decoder's } \pi_i", "dec"), (r"-\log p(x_i\mid z)", "loss")]
    for g_, (title, role) in enumerate(titles):
        x0 = 10 + g_ * 305
        d.begin(g_ + 1)
        d.math(x0 + 2.5 * (s_ + gap), 32, title, 22, C[role])
        for r_ in range(5):
            for c_ in range(5):
                xx, yy = x0 + c_ * (s_ + gap), 50 + r_ * (s_ + gap)
                if g_ == 0:
                    v = p["x"][r_][c_]
                    fill, txt, col = (INK if v else "#FFFFFF"), str(v), ("#fff" if v else INK)
                elif g_ == 1:
                    v = p["pi"][r_][c_]
                    shade = int(255 - 210 * v)
                    fill, txt, col = f"rgb({shade},{shade},{shade})", f"{v:.2f}", ("#fff" if v > 0.55 else INK)
                else:
                    v = p["loss"][r_][c_]
                    a = min(v / 1.8, 1)
                    fill, txt, col = f"rgba(190,18,60,{0.08 + 0.75 * a:.2f})", f"{v:.2f}", ("#fff" if a > 0.6 else INK)
                d.add(f'<rect x="{xx}" y="{yy}" width="{s_}" height="{s_}" rx="4" fill="{fill}" stroke="#D9DDE5"/>')
                d.add(f'<text x="{xx + s_ / 2}" y="{yy + s_ / 2 + 6}" font-family="SF Mono, Menlo, monospace" '
                      f'font-size="17" fill="{col}" text-anchor="middle">{txt}</text>')
        d.end()
    d.begin(3)
    d.text(763, 352, f"this 5 × 5 patch: {p['total']:.1f} nats", 18, C["loss"], 650)
    d.end()
    d.save()


# =============================================================== reparameterisation
def _diamond(d, cx, cy, r, role, tex, size=22):
    c, t = ROLE[role]
    d.add(f'<rect x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" rx="8" fill="{t}" stroke="{c}" stroke-width="2.4"/>')
    d.math(cx, cy + 9, tex, size, c)


def _rnd(d, cx, cy, r, tex, size=22):
    d.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{ROLE["z"][1]}" stroke="{C["z"]}" stroke-width="2.4" stroke-dasharray="5 3"/>')
    d.math(cx, cy + 9, tex, size, C["z"])


def reparam_before():
    d = SVG("reparam-before", 1152, 380)
    d.begin(1)
    _diamond(d, 300, 110, 38, "enc", r"\mu")
    _diamond(d, 300, 270, 38, "enc", r"\sigma")
    _rnd(d, 560, 190, 52, "z")
    d.math(560, 276, r"z \sim \N(\mu, \sigma^2)", 19, INK3)
    _diamond(d, 820, 190, 38, "loss", "f")
    d.text(820, 258, "loss", 16, INK3)
    d.line([(340, 120), (510, 175)], INK3)
    d.line([(340, 260), (510, 205)], INK3)
    d.line([(614, 190), (780, 190)], INK3)
    d.text(160, 115, "from the encoder", 15, INK3, 500, "end")
    d.text(160, 275, "from the encoder", 15, INK3, 500, "end")
    d.add(f'<circle cx="1000" cy="80" r="16" fill="{ROLE["z"][1]}" stroke="{C["z"]}" stroke-width="2" stroke-dasharray="4 3"/>')
    d.text(1026, 86, "random", 15, INK3, 500, "start")
    d.add(f'<rect x="984" y="112" width="32" height="32" rx="6" fill="{ROLE["enc"][1]}" stroke="{C["enc"]}" stroke-width="2"/>')
    d.text(1026, 134, "deterministic", 15, INK3, 500, "start")
    d.end()
    d.begin(2)
    d.line([(780, 214), (616, 214)], C["loss"], 2.8)
    d.line([(508, 222), (344, 272)], C["loss"], 2.8, dash="5 4")
    d.line([(508, 160), (344, 106)], C["loss"], 2.8, dash="5 4")
    d.cross(426, 134, 14)
    d.cross(426, 246, 14)
    d.label(560, 342, r"the loss knows how $z$ should move, but $z$ is a fresh random draw:", 18, C["loss"], 650, width=1000)
    d.label(560, 368, r"how does it change when $\mu$ or $\sigma$ changes a little?", 18, C["loss"], 650, width=1000)
    d.end()
    d.save()


def reparam_after():
    d = SVG("reparam-after", 1152, 380)
    d.begin(1)
    _diamond(d, 300, 80, 38, "enc", r"\mu")
    _diamond(d, 300, 200, 38, "enc", r"\sigma")
    _rnd(d, 300, 318, 40, r"\varepsilon")
    d.math(240, 326, r"\varepsilon \sim \N(0, I)", 18, C["z"], "end")
    _diamond(d, 620, 200, 48, "z", "z", 26)
    d.math(620, 290, r"z = \mu + \sigma\odot\varepsilon", 20, C["z"])
    _diamond(d, 900, 200, 38, "loss", "f")
    d.line([(340, 88), (570, 180)], INK3)
    d.line([(340, 200), (570, 200)], INK3)
    d.line([(342, 318), (620, 318), (620, 252)], INK3)
    d.line([(670, 200), (860, 200)], INK3)
    d.end()
    d.begin(2)
    d.line([(860, 226), (672, 226)], C["loss"], 3)
    d.line([(572, 166), (344, 74)], C["loss"], 3)
    d.line([(572, 236), (344, 236)], C["loss"], 3)
    d.math(480, 66, r"\partial z/\partial\mu = 1", 20, C["loss"])
    d.math(460, 274, r"\partial z/\partial\sigma = \varepsilon", 20, C["loss"])
    d.label(1010, 300, r"$\varepsilon$ is just an input,", 18, C["loss"], 650, width=300)
    d.text(1010, 326, "like a pixel value", 18, C["loss"], 650)
    d.end()
    d.save()


def reparam_numbers():
    d = SVG("reparam-numbers", 1152, 380)
    d.begin(1)
    xs = [10, 320, 630, 940]
    w = 190
    labels = [(r"\varepsilon", "-0.4", "z"), (r"\sigma\varepsilon", "-0.2", "enc"), (r"z = \mu + \sigma\varepsilon", "0.8", "z"), (r"f = z^2", "0.64", "loss")]
    ops = [r"$\times\,\sigma$ (0.5)", r"$+\,\mu$ (1)", "square"]
    for i, (lab, val, role) in enumerate(labels):
        x = xs[i]
        c, t = ROLE[role]
        d.rect(x, 60, w, 92, c, t)
        d.math(x + w / 2, 120, val, 32, c, weight=700)
        d.math(x + w / 2, 46, lab, 22, C[role])
        if i < 3:
            d.line([(x + w + 2, 106), (xs[i + 1] - 4, 106)], INK3, 2.4)
            d.label((x + w + xs[i + 1]) / 2, 94, ops[i], 17, INK2, 650, width=200)
    d.label(576, 194, r"forward: one fixed draw $\varepsilon=-0.4$, with $\mu = 1$ and $\sigma = 0.5$", 18, INK3, width=1000)
    d.end()
    d.begin(2)
    d.line([(990, 270), (842, 270)], C["loss"], 2.8)
    d.math(1000, 252, r"\frac{\partial f}{\partial z} = 2z = 1.6", 20, C["loss"], "start")
    d.rect(560, 238, 280, 64, C["loss"], ROLE["loss"][1])
    d.math(700, 280, r"\frac{\partial f}{\partial\mu} = 1.6\times 1 = 1.6", 21, C["loss"])
    d.end()
    d.begin(3)
    d.rect(200, 238, 340, 64, C["loss"], ROLE["loss"][1])
    d.math(370, 280, r"\frac{\partial f}{\partial\sigma} = 1.6\times\varepsilon = -0.64", 21, C["loss"])
    d.label(370, 340, r"the noise value multiplies the $\sigma$-gradient", 17, C["loss"], 650)
    d.end()
    d.save()


# =============================================================== GAN
def gan_base(d, step=None):
    """Shared GAN layout. step: None (overview), 'D' or 'G'."""
    gdim = step == "D"
    ddim = step == "G"
    d.dice(30, 230, 60)
    d.math(60, 322, "z", 24, C["z"])
    d.line([(96, 260), (130, 260)], C["z"])
    d.trapezoid(134, 260, 180, 80, 180, "dec", "generator G", r"weights $\theta$", opacity=".45" if gdim else None)
    if gdim:
        d.lock(224, 196, C["dec"])
    d.line([(318, 260), (350, 260)], C["dec"])
    d.image(356, 190, 140, 140, "gan-fake-one.png", "dec")
    d.text(426, 352, "fake", 18, C["dec"], 650)
    if step != "G":
        d.image(356, 10, 140, 140, "gan-real-one.png", "x")
        d.text(426, 172, "real", 18, C["x"], 650)
        d.line([(500, 80), (560, 80), (560, 160), (600, 160)], C["x"])
    d.line([(500, 260), (560, 260), (560, 200), (600, 200)], C["dec"])
    d.trapezoid(604, 180, 200, 210, 80, "disc", "discriminator D", r"weights $\psi$", opacity=".45" if ddim else None)
    d.line([(808, 180), (850, 180)], C["disc"])


def gan():
    d = SVG("gan", 1152, 380)
    d.begin(1)
    d.dice(30, 230, 60)
    d.math(60, 322, "z", 24, C["z"])
    d.line([(96, 260), (130, 260)], C["z"])
    d.trapezoid(134, 260, 180, 80, 180, "dec", "generator G", r"weights $\theta$")
    d.line([(318, 260), (350, 260)], C["dec"])
    d.image(356, 190, 140, 140, "gan-fake-one.png", "dec")
    d.text(426, 352, "fake", 18, C["dec"], 650)
    d.end()
    d.begin(2)
    d.image(356, 10, 140, 140, "gan-real-one.png", "x")
    d.text(426, 172, "real", 18, C["x"], 650)
    d.line([(500, 80), (560, 80), (560, 160), (600, 160)], C["x"])
    d.line([(500, 260), (560, 260), (560, 200), (600, 200)], C["dec"])
    d.trapezoid(604, 180, 200, 210, 80, "disc", "discriminator D", r"weights $\psi$")
    d.line([(808, 180), (850, 180)], C["disc"])
    d.node(856, 140, 270, 80, "disc", r"$D(x) = P(\text{real})$", r"one number in $(0, 1)$", size=22)
    d.end()
    d.begin(3)
    d.text(990, 272, "D wants: real → 1, fake → 0", 17, C["disc"], 650)
    d.text(990, 302, "G wants: fake → 1", 17, C["dec"], 650)
    d.end()
    d.save()


def gan_dstep():
    d = SVG("gan-dstep", 1152, 380)
    gan_base(d, "D")
    d.rect(856, 118, 290, 124, C["loss"], ROLE["loss"][1])
    d.math(1001, 156, r"-\log D(x_{\text{real}})", 19, C["loss"], weight=600)
    d.math(1001, 192, r"-\log\big(1 - D(x_{\text{fake}})\big)", 19, C["loss"], weight=600)
    d.text(1001, 226, "targets: real 1, fake 0", 15, INK3)
    d.begin(2)
    d.line([(856, 200), (806, 200)], C["loss"], 3)
    d.curve("M700,232 C680,270 640,280 600,276", C["loss"], 3)
    d.curve("M700,128 C680,90 640,84 600,90", C["loss"], 3)
    d.add(f'<line x1="530" y1="230" x2="530" y2="300" stroke="{C["loss"]}" stroke-width="3" stroke-dasharray="6 4"/>')
    d.scissors(530, 312)
    d.text(530, 362, "fake.detach()", 17, C["loss"], 650)
    d.text(224, 362, "G gets no gradient", 17, C["dec"], 650)
    d.end()
    d.save()


def gan_gstep():
    d = SVG("gan-gstep", 1152, 380)
    gan_base(d, "G")
    d.lock(704, 120, C["disc"])
    d.label(704, 92, r"$\psi$ frozen", 16, C["disc"], 650, width=200)
    d.rect(856, 140, 270, 80, C["loss"], ROLE["loss"][1])
    d.math(991, 182, r"-\log D(x_{\text{fake}})", 21, C["loss"], weight=600)
    d.text(991, 208, "target: fake → 1", 15, INK3)
    d.begin(2)
    d.line([(856, 196), (808, 196)], C["loss"], 3)
    d.curve("M806,206 C740,250 660,250 604,214", C["loss"], 3, dash="7 5")
    d.label(705, 270, r"through D, not into $\psi$", 15, C["loss"], 650, width=300)
    d.line([(560, 286), (500, 286)], C["loss"], 3)
    d.curve("M356,300 C330,320 300,316 290,300", C["loss"], 3)
    d.label(224, 364, r"G's weights $\theta$ get the gradient", 17, C["loss"], 650, width=400)
    d.end()
    d.save()


def freeze_detach():
    d = SVG("freeze-detach", 1152, 330)
    for row, (title, y, mode) in enumerate((("detach(): cut the wire", 92, "detach"),
                                             ("requires_grad_(False): lock D's weights, keep the wire", 248, "freeze"))):
        d.begin(row + 1)
        d.text(20, y - 50, title, 21, INK, 650, "start")
        d.node(20, y - 28, 150, 60, "dec", r"$G(z)$", size=22)
        d.line([(172, y + 2), (330, y + 2)], C["dec"])
        d.node(334, y - 28, 170, 60, "disc", r"$D$", size=24)
        d.line([(506, y + 2), (600, y + 2)], C["disc"])
        d.node(604, y - 28, 130, 60, "loss", "loss", size=22)
        d.line([(604, y + 22), (508, y + 22)], C["loss"], 2.6)
        if mode == "detach":
            d.line([(332, y + 22), (296, y + 22)], C["loss"], 2.6)
            d.scissors(256, y + 8)
            d.text(770, y + 2, "the gradient stops at the cut:", 19, C["loss"], 650, "start")
            d.text(770, y + 28, "G learns nothing", 19, C["loss"], 650, "start")
        else:
            d.lock(474, y - 4, C["disc"], 0.7)
            d.line([(332, y + 22), (174, y + 22)], C["loss"], 2.6)
            d.text(770, y + 2, "D's weights stay put;", 19, C["loss"], 650, "start")
            d.text(770, y + 28, "G still learns", 19, C["loss"], 650, "start")
        d.end()
    d.save()


def cgan():
    d = SVG("cgan", 1152, 330)
    d.begin(1)
    d.dice(20, 200, 56)
    d.math(48, 288, "z", 22, C["z"])
    d.node(10, 60, 100, 70, "grey", r"$c = 7$", "a label", size=21)
    d.line([(80, 228), (160, 228)], C["z"])
    d.line([(110, 95), (130, 95), (130, 196), (160, 196)], C["grey"])
    d.trapezoid(164, 212, 180, 80, 170, "dec", "generator")
    d.math(254, 270, "G(z, c)", 19, C["dec"])
    d.line([(348, 212), (384, 212)], C["dec"])
    d.image(390, 142, 140, 140, "gan-seven.png", "dec")
    d.end()
    d.begin(2)
    d.line([(534, 212), (600, 212)], C["dec"])
    d.line([(110, 80), (580, 80), (580, 172), (600, 172)], C["grey"], dash="6 4")
    d.trapezoid(604, 192, 200, 200, 80, "disc", "discriminator")
    d.math(704, 240, "D(x, c)", 19, C["disc"])
    d.line([(808, 192), (850, 192)], C["disc"])
    d.node(856, 150, 280, 84, "disc", "real AND a 7?", r"judges the pair $(x, c)$", size=21)
    d.end()
    d.save()


def recipes():
    d = SVG("recipes", 1152, 430)
    rows = [("autoregressive", "one token at a time", "exact likelihood · slow sampling", "x"),
            ("VAE", "random z, decode once", "trained on the ELBO · tends to blur", "enc"),
            ("GAN", "random z, generate once", "trained against a judge · may miss modes", "disc")]
    for i, (name, how, props, role) in enumerate(rows):
        y = 20 + i * 140
        d.begin(i + 1)
        d.text(10, y + 46, name, 25, C[role], 700, "start")
        d.text(10, y + 76, how, 17, INK3, 500, "start")
        if i == 0:
            for k, ch in enumerate("aabid-"):
                d.node(290 + k * 74, y + 16, 58, 60, "x", ch if ch != "-" else "–", size=24)
                if k < 5:
                    d.line([(349 + k * 74, y + 46), (363 + k * 74, y + 46)], INK3, 1.8)
        else:
            d.dice(290, y + 18, 56)
            d.line([(352, y + 46), (384, y + 46)], C["z"])
            d.trapezoid(388, y + 46, 190, 60, 110, "dec", "decoder" if i == 1 else "generator")
            d.line([(582, y + 46), (614, y + 46)], C["dec"])
            d.image(620, y, 92, 92, "vae-prior-one.png" if i == 1 else "gan-fake-one.png", "dec")
        a_, b_ = props.split(" · ")
        d.text(770, y + 40, a_, 19, INK2, 650, "start")
        d.text(770, y + 68, b_, 18, INK3, 500, "start")
        d.end()
    d.save()


def task():
    d = SVG("task", 1152, 380)
    rows = [("images", 20, "digit"), ("numbers", 210, "number")]
    for i, (name, y, kind) in enumerate(rows):
        d.begin(i + 1)
        d.rect(10, y, 190, 120, C["grey"], ROLE["grey"][1], dash="6 4")
        d.text(105, y + 52, "unknown process", 18, INK2, 650)
        d.text(105, y + 80, "people writing digits" if kind == "digit" else "a recipe we hide", 16, INK3)
        d.line([(204, y + 60), (272, y + 60)], C["x"])
        d.text(238, y + 46, "data", 16, C["x"], 650)
        if kind == "digit":
            for k, f in enumerate(["vote-02.png", "vote-04.png", "vote-05.png", "vote-09.png"]):
                d.image(278 + k * 84, y + 22, 76, 76, f, "x", sw=2)
        else:
            for k, v in enumerate(NUM["toy"]["first"][:4]):
                c, t = ROLE["x"]
                d.rect(278 + k * 84, y + 36, 76, 48, c, t, r=8, sw=2)
                d.math(316 + k * 84, y + 68, f"{v:.2f}", 19, INK)
        d.line([(618, y + 60), (690, y + 60)], C["dec"])
        d.text(654, y + 46, "learn", 16, C["dec"], 650)
        d.trapezoid(694, y + 60, 170, 70, 120, "dec", "model", r"$p_\theta$")
        d.line([(868, y + 60), (950, y + 60)], C["dec"])
        d.text(909, y + 46, "sample", 16, C["dec"], 650)
        if kind == "digit":
            d.image(956, y + 10, 100, 100, "vae-prior-one.png", "dec")
            d.text(1006, y + 136, "a new digit", 16, C["dec"], 650)
        else:
            c, t = ROLE["dec"]
            d.rect(956, y + 36, 100, 48, c, t, r=8, sw=2)
            d.text(1006, y + 68, "?", 22, c, 700)
            d.text(1006, y + 108, "a new number", 16, C["dec"], 650)
        d.end()
    d.save()


def route_a():
    d = SVG("route-a", 1152, 430)
    bits = NUM["digit3_bits"]
    ra = NUM["routeA"]
    R, Cc = ra["r"], ra["c"]
    s0, x0, y0 = 14, 20, 18
    for ctx_build, ctx in ((1, "row3"), (2, "window7")):
        pass
    for k in range(784):
        rr, cc = divmod(k, 28)
        before = k < R * 28 + Cc
        if k == R * 28 + Cc:
            fill = C["z"]
        elif before:
            fill = INK if bits[k] else "#FFFFFF"
        else:
            fill = "#ECEEF2"
        d.add(f'<rect x="{x0 + cc * s0}" y="{y0 + rr * s0}" width="{s0 - 1}" height="{s0 - 1}" fill="{fill}"/>')
    d.text(x0 + 14 * s0, y0 + 28 * s0 + 26, "drawn so far · next pixel (orange) · not yet drawn (grey)", 15, INK3)
    # context outlines
    d.begin("=1")
    d.add(f'<rect x="{x0 + (Cc - 3) * s0 - 1.5}" y="{y0 + R * s0 - 1.5}" width="{3 * s0 + 2}" height="{s0 + 2}" fill="none" stroke="{C["x"]}" stroke-width="3"/>')
    d.end()
    d.begin(2)
    pts = [(Cc - 3, R - 3), (Cc + 4, R - 3), (Cc + 4, R), (Cc, R), (Cc, R + 1), (Cc - 3, R + 1)]
    path = "M" + " L".join(f"{x0 + a * s0 - 1.5},{y0 + b * s0 - 1.5}" for a, b in pts) + " Z"
    d.add(f'<path d="{path}" fill="none" stroke="{C["x"]}" stroke-width="3"/>')
    d.end()
    # right side
    d.begin("=1")
    d.text(760, 40, "context: the 3 previous pixels", 21, C["x"], 650)
    d.text(760, 70, "(the name model read 3 previous characters)", 17, INK3)
    d.label(760, 130, rf"$p(x_i = \text{{ink}} \mid 3 \text{{ pixels}}) = {ra['p_row3']}$", 24, INK, width=600)
    d.end()
    d.begin(2)
    d.text(760, 40, "context: 3 rows above + 3 to the left", 21, C["x"], 650)
    d.text(760, 70, "24 pixels the model may look at", 17, INK3)
    d.label(760, 130, rf"$p(x_i = \text{{ink}} \mid 24 \text{{ pixels}}) = {ra['p_window7']}$", 24, INK, width=600)
    d.end()
    d.begin(1)
    d.trapezoid(520, 230, 170, 100, 60, "dec", "network")
    d.line([(694, 230), (740, 230)], C["dec"])
    d.text(750, 238, "one probability: ink or paper?", 18, C["dec"], 650, "start")
    d.label(760, 330, r"Repeat for all 784 pixels, one after another:", 18, INK2, 500, width=640)
    d.label(760, 372, r"$p(x) = \prod_{i=1}^{784} p(x_i \mid x_{<i})$", 26, INK, width=640)
    d.end()
    d.save()


def story_decode():
    d = SVG("story-decode", 1152, 340)
    z = NUM["story_z"]
    d.vec(40, 110, z, "z", cw=100, ch=48)
    d.math(90, 96, "z", 26, C["z"])
    d.line([(144, 158), (196, 158)], C["z"])
    d.trapezoid(200, 158, 260, 100, 250, "dec", "decoder", r"weights $\theta$")
    d.line([(464, 158), (516, 158)], C["dec"])
    d.image(522, 18, 280, 280, "story-pi.png", "dec")
    d.math(662, 330, r"\pi = \mathrm{sigmoid}(\text{decoder}_\theta(z))", 20, C["dec"])
    d.label(990, 110, r"each $\pi_i$ is between 0 and 1", 19, INK2, 500, width=320)
    d.label(990, 150, r"$\pi_i$ = probability that", 19, INK2, 500, width=320)
    d.label(990, 180, r"pixel $i$ is ink", 19, INK2, 500, width=320)
    d.save()


def main():
    for f in (common, two_ways, mixture_story, latent_story, ae, ae_generate, cloud_vs_point, vae_encode, vae_sample,
              vae_decode, vae, vae_generate, pgm, vae_shapes, pixels, task, route_a, story_decode, reparam_before, reparam_after, reparam_numbers,
              gan, gan_dstep, gan_gstep, freeze_detach, cgan, recipes):
        f()
    print("wrote", len(list(OUT.glob("*.svg"))), "diagrams")


if __name__ == "__main__":
    main()
