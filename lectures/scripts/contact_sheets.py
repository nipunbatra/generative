"""Create labelled contact sheets from the state screenshots, without altering originals."""
from pathlib import Path
import sys
import json
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parents[1]
for number in (sys.argv[1:] or range(1, 6)):
    qa = root / f"lecture{number}" / "qa"
    manifest = json.loads((qa.parent / "manifest.json").read_text())
    shots = [qa / "screenshots" / f"{i+1:03d}-{s['id']}-{b}.jpg"
             for i, s in enumerate(manifest["slides"])
             for b in range(1, s["builds"]+1)]
    out = qa / "contacts"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for start in range(0, len(shots), 12):
        sheet = Image.new("RGB", (1536, 4 * 318), "#e5e7eb")
        draw = ImageDraw.Draw(sheet)
        for j, p in enumerate(shots[start:start + 12]):
            x, y = (j % 3) * 512, (j // 3) * 318
            tile = ImageOps.contain(Image.open(p), (512, 288))
            sheet.paste(tile, (x, y + 26))
            draw.text((x + 8, y + 7), p.stem, fill="#14171f")
        sheet.save(out / f"states-{start+1:03d}-{min(start+12,len(shots)):03d}.jpg", quality=92)
    print(f"Lecture {number}: {len(shots)} states in {(len(shots)+11)//12} contact sheets")
