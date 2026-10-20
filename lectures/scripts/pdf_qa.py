"""Render both exports; compare final reveal parity and browser/print appearance."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
COMPARE_ONLY = "--compare-only" in sys.argv
POPLER = Path("/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm")
if not POPLER.exists():
    POPLER = "pdftoppm"

def check(number):
    directory = ROOT / f"lecture{number}"
    manifest = json.loads((directory / "manifest.json").read_text())
    output = directory / "qa" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    report = {"lecture": number, "checks": [], "scope": "Every PDF page rasterised; completed/final-reveal parity and browser/print comparison for every scene."}
    def record(name, passed, **data):
        report["checks"].append({"name": name, "passed": bool(passed), **data})
    renders = {}
    for name, expected in [("completed", len(manifest["slides"])), ("all-builds", manifest["totalStates"])]:
        pdf = directory / "exports" / f"{name}.pdf"
        pages = PdfReader(pdf).pages
        record(f"{name} page count", len(pages) == expected, actual=len(pages), expected=expected)
        record(f"{name} page geometry", all(abs(float(p.mediabox.width)-960)<.1 and abs(float(p.mediabox.height)-540)<.1 for p in pages))
        record(f"{name} nonempty text", all(len(p.extract_text().strip()) > 30 for p in pages))
        folder = output / name
        folder.mkdir(exist_ok=True)
        existing = list(folder.glob("page-*.png"))
        reuse = COMPARE_ONLY and len(existing) == expected and all(p.stat().st_mtime >= pdf.stat().st_mtime for p in existing)
        if not reuse:
            for old in existing:
                old.unlink()
            subprocess.run([str(POPLER), "-r", "96", "-png", str(pdf), str(folder/"page")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        renders[name] = sorted(folder.glob("page-*.png"))
        record(f"{name} rendered count", len(renders[name]) == expected)
    last = 0
    errors, print_diffs = [], []
    for i, slide in enumerate(manifest["slides"]):
        last += slide["builds"]
        complete = np.asarray(Image.open(renders["completed"][i]).convert("RGB"), dtype=float)[:665]
        reveal = np.asarray(Image.open(renders["all-builds"][last-1]).convert("RGB"), dtype=float)[:665]
        errors.append(float(np.abs(complete-reveal).mean()))
        shot = directory / "qa" / "screenshots" / f"{i+1:03d}-{slide['id']}-{slide['builds']}.jpg"
        browser = Image.open(shot).convert("RGB")
        if browser.size != (1280,720):
            browser = browser.resize((1280,720), Image.Resampling.LANCZOS)
        browser_array=np.asarray(browser,dtype=float)[:665]
        raw=float(np.abs(complete-browser_array).mean())
        # Chrome screenshots and PDF rasterisation round subpixel origins differently.
        # Permit one pixel of global registration, preserving the same 5/255 tolerance.
        comparisons=[(float(np.abs(complete[2:-2,2:-2]-browser_array[2+dy:663+dy,2+dx:1278+dx]).mean()),dx,dy)
                     for dx in [-1,0,1] for dy in [-1,0,1]]
        difference,dx,dy=min(comparisons)
        print_diffs.append({"id":slide["id"],"meanAbsolutePixelDifference":difference,"unregisteredDifference":raw,"registrationPixels":[dx,dy]})
    record("completed equals final reveal, excluding footer", max(errors) < .1, maximumMeanAbsoluteDifference=max(errors))
    record("browser and print body appearance", max(d["meanAbsolutePixelDifference"] for d in print_diffs)<5, worst=sorted(print_diffs,key=lambda d:d["meanAbsolutePixelDifference"],reverse=True)[:5], tolerance=5, registration="At most one pixel globally in each axis; accommodates subpixel rounding between screenshot and PDF rasterisation.")
    # Full completed-page contact sheets give a readable PDF review record.
    contact = output / "contacts"
    contact.mkdir(exist_ok=True)
    for start in range(0, len(renders["completed"]), 12):
        canvas = Image.new("RGB",(1536,1272),"#e5e7eb")
        draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(renders["completed"][start:start+12]):
            x,y=(j%3)*512,(j//3)*318
            canvas.paste(ImageOps.contain(Image.open(p),(512,288)),(x,y+26))
            draw.text((x+8,y+7),f"{start+j+1:03d} {manifest['slides'][start+j]['id']}",fill="#14171f")
        canvas.save(contact/f"pages-{start+1:03d}.jpg",quality=92)
    report["passed"]=all(x["passed"] for x in report["checks"])
    (directory/"qa"/"pdf-checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"lecture":number,"passed":report["passed"],"pages":len(manifest["slides"])+manifest["totalStates"],"failures":[c for c in report["checks"] if not c["passed"]]}),flush=True)
    return report["passed"]

numbers=[x for x in sys.argv[1:] if x in ["1","2","3","4","5"]] or list(range(1,6))
with ThreadPoolExecutor(max_workers=2) as executor:
    results=list(executor.map(check,numbers))
if not all(results):
    sys.exit(1)
