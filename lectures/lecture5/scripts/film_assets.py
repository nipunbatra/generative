"""Copy real-image assets from the companion film (dl-summaries/diffusion) into compact strips for the slides.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/film_assets.py          # writes figures/film/*.jpg, figures/film/sd/* and figures/film/provenance.json

Two sources, both real model outputs computed by the film's scripts:
- math/ddpm_images.py: the original DDPM checkpoint (Ho et al. 2020, LSUN cat, 256 x 256, google/ddpm-cat-256), run on
  scikit-image's CC0 photograph "Chelsea";
- math/sd_runs.py: Stable Diffusion v1.5 (DDIM with classifier-free guidance, written out in the script), for the
  guidance and step sweeps, a latent consistency adapter, the autoencoder, cross-attention maps and sampling films.
Strips are rows of square frames that the slides cut into frames. Each part is written only once its runs are complete.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/Users/nipun/git/dl-summaries/diffusion/public/img/ddpm")
SD = SRC.parent / "sd"
SD_JSON = SRC.parents[2] / "math" / "out" / "sd.json"
OUT = ROOT / "figures" / "film"
OUT.mkdir(parents=True, exist_ok=True)
# the film's six sampling runs, in the order of sd_runs.py's WALL, with their prompts from its GRID
SD_PROMPTS = {
    "fox": "a red fox sitting in fresh snow, soft morning light, wildlife photograph",
    "peacock": "a peacock with its tail feathers open in a garden, golden hour, photograph",
    "lighthouse": "a lighthouse on a rocky cliff at sunset, oil painting",
    "hummingbird": "a watercolor painting of a hummingbird and red flowers",
    "jellyfish": "glowing jellyfish in the deep ocean, underwater photograph",
}  # the sixth run (ramen) is left out: the lecture uses no egg, meat or fish examples


def save_strip(frames, path, names, src, fmt="JPEG"):
    size = frames[0].size[0]
    canvas = Image.new(frames[0].mode, (size * len(frames), size))
    for k, f in enumerate(frames):
        canvas.paste(f, (k * size, 0))
    (OUT / path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / path, **({"quality": 86, "optimize": True, "progressive": True} if fmt == "JPEG" else {"optimize": True}))
    h = hashlib.sha256()
    for n in names:
        h.update((src / n).read_bytes())
    return {"frames": names, "size": size, "sources_sha256": h.hexdigest()}


def strip(names, path, size, src=SRC):
    frames = [Image.open(src / n).convert("RGB").resize((size, size), Image.LANCZOS) for n in names]
    return save_strip(frames, path, names, src)


def spotlights(words, path, size):
    """The attention image, then one copy per word lit by that word's cross-attention map (dark where it is ignored)."""
    img = np.asarray(Image.open(SD / "attn_image.jpg").convert("RGB").resize((size, size), Image.LANCZOS)) / 255
    frames, names = [img], ["attn_image.jpg"]
    for j, w in words:
        names.append(f"attn_{j:02d}_{w}.png")
        m = np.asarray(Image.open(SD / names[-1]).convert("L").resize((size, size), Image.LANCZOS)) / 255
        frames.append(img * (0.08 + 0.92 * m[..., None] ** 0.6))
    return save_strip([Image.fromarray((f * 255).round().astype(np.uint8)) for f in frames], path, names, SD)


def sd_assets():
    """Stable Diffusion v1.5 runs from the film's math/sd_runs.py, with the settings that sd.json records."""
    run = json.loads(SD_JSON.read_text())
    prov = {"source": "dl-summaries/diffusion (the companion film), math/sd_runs.py and math/out/sd.json",
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5, fp16 on Apple MPS (autoencoder in fp32)",
            "sampler": "DDIM (eta = 0), 50 steps, classifier-free guidance eps = eps_null + w (eps_prompt - eps_null), w = 7.5 unless stated",
            "files": {}}
    F = prov["files"]
    cfg, steps, lcm = run["cfg"], run["steps"], run["lcm"]
    F["sd/guidance.jpg"] = {"prompt": cfg["prompt"], "seed": cfg["seed"], "w": cfg["w"],
                            **strip([f"cfg_{str(w).replace('.', 'p')}.jpg" for w in cfg["w"]], "sd/guidance.jpg", 256, SD)}
    F["sd/steps.jpg"] = {"prompt": steps["prompt"], "seed": steps["seed"], "ddim_steps": steps["k"],
                         **strip([f"steps_{k:02d}.jpg" for k in steps["k"]], "sd/steps.jpg", 256, SD)}
    F["sd/lcm.jpg"] = {"prompt": run["hero"]["prompt"], "seed": run["hero"]["seed"], "adapter": lcm["adapter"], "lcm_steps": lcm["steps"],
                       "lcm_w": lcm["w"], "last_frame": "plain DDIM, 4 steps, w = 7.5",
                       **strip([f"lcm_{k}.jpg" for k in lcm["steps"]] + ["lcm_ddim4.jpg"], "sd/lcm.jpg", 256, SD)}
    F["sd/vae.jpg"] = {**run["vae"], **strip(["vae_in.jpg", "vae_out.jpg"], "sd/vae.jpg", 256, SD)}
    # the four 64 x 64 latent channels, kept blocky: the film saved them 4x nearest-upsampled
    names = [f"vae_lat{c}.png" for c in range(4)]
    chans = [Image.open(SD / n).convert("L").resize((64, 64), Image.NEAREST).resize((128, 128), Image.NEAREST) for n in names]
    F["sd/vae-latent.png"] = {"display": "each channel stretched to its 1-99% range", **save_strip(chans, "sd/vae-latent.png", names, SD, "PNG")}
    att = run["attn"]
    words = [(j, w) for j, w in enumerate(att["tokens"]) if w in ("red", "fox", "sitting", "snow")]
    F["sd/attn.jpg"] = {"prompt": att["prompt"], "seed": att["seed"], "words": [w for _, w in words],
                        "maps": "16 x 16 cross-attention, conditional half, mean over heads, layers and steps, each word scaled to 0-1",
                        **spotlights(words, "sd/attn.jpg", 256)}
    F["sd/fox.jpg"] = {"prompt": att["prompt"], "seed": att["seed"], **strip(["attn_image.jpg"], "sd/fox.jpg", 256, SD)}
    for k, r in enumerate(run["cold"]):
        if r["name"] not in SD_PROMPTS:
            continue
        frames = list(range(0, 51, 2))  # the decoded latent before DDIM steps 0, 2, ..., 48, then the final image
        F[f"sd/run{k}-xt.jpg"] = {"prompt": SD_PROMPTS[r["name"]], "seed": r["seed"], "steps_shown": frames,
                                  **strip([f"cold_{k}_{i:02d}.jpg" for i in frames], f"sd/run{k}-xt.jpg", 180, SD)}
    return prov


def main():
    prov = {"source": "dl-summaries/diffusion (the companion film), math/ddpm_images.py",
            "model": "google/ddpm-cat-256: the original DDPM LSUN-cat checkpoint (Ho, Jain & Abbeel 2020), 113.7M parameters",
            "photo": "Chelsea, from scikit-image (CC0), 256 x 256 crop of the face", "files": {}}
    for name in ("chelsea.jpg", "eps.jpg", "pair_xt_0400.jpg", "pair_epshat_0400.jpg", "pair_x0hat_0400.jpg", "pair_err_0400.jpg"):
        Image.open(SRC / name).convert("RGB").save(OUT / name.replace("_", "-"), quality=90)
        prov["files"][name.replace("_", "-")] = {"from": name, "sha256": hashlib.sha256((SRC / name).read_bytes()).hexdigest()}
    ts = list(range(0, 1001, 50))
    prov["files"]["cat-jump.jpg"] = {"t": ts, **strip([f"jump_{t:04d}.jpg" for t in ts], "cat-jump.jpg", 192)}
    guess_t = [950, 800, 600, 400, 250, 100]
    prov["files"]["cat-guess-xt.jpg"] = {"t": guess_t, **strip([f"pair_xt_{t:04d}.jpg" for t in guess_t], "cat-guess-xt.jpg", 192)}
    prov["files"]["cat-guess-x0.jpg"] = {"t": guess_t, **strip([f"pair_x0hat_{t:04d}.jpg" for t in guess_t], "cat-guess-x0.jpg", 192)}
    # sampling runs: x_t every 50 steps from 1000 to 0, and the clean-image guess made at the same steps
    runs = [k for k in range(8) if (SRC / f"s{k}_xt_0000.jpg").exists()]
    if runs:
        st = list(range(1000, -1, -50))
        for k in runs:
            prov["files"][f"cat-run{k}-xt.jpg"] = {"t": st, **strip([f"s{k}_xt_{t:04d}.jpg" for t in st], f"cat-run{k}-xt.jpg", 160)}
            x0 = [f"s{k}_x0_{max(t, 10):04d}.jpg" if t > 0 else f"s{k}_xt_0000.jpg" for t in st]
            prov["files"][f"cat-run{k}-x0.jpg"] = {"t": st, **strip(x0, f"cat-run{k}-x0.jpg", 160)}
    prov["runs"] = runs
    if SD_JSON.exists() and (SD / "lcm_8.jpg").exists() and (SD / "attn_03_fox.png").exists():
        prov["sd"] = sd_assets()
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
    print("film assets:", sorted(p.name for p in OUT.iterdir()), "runs:", runs, "sd:", sorted(prov.get("sd", {}).get("files", {})))


if __name__ == "__main__":
    main()
