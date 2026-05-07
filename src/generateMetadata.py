import pandas as pd
import hashlib
import csv
from pathlib import Path
from PIL import Image


# ### METADATA 1

# # 1–20 = short, 21–40 = long, where prompt i pairs with prompt i+20.
# PROMPTS = {
#     1: "A cat lying on the table",
#     2 : "A dog running on grass",
#     3 : "A man walking on sand",
#     4 : "An astronaut flying into space",
#     5 : "A girl swimming in the sea",
#     6 : "An apple in a fruit bowl",
#     7 : "A starred night in the dark",
#     8 : "A sailboat in the stormy sea",
#     9 : "A bird perched on a tree",
#     10 : "A scene from War and Peace",
#     11 : "A butterfly landing on a flower",
#     12 : "A child playing in the snow",
#     13 : "A chef cooking in a kitchen",
#     14 : "A car driving through mountains",
#     15 : "A painter working on a canvas",
#     16 : "A wave crashing against rocks",
#     17 : "A robot building a structure",
#     18 : "A fox hiding in bushes",
#     19 : "A balloon floating in the sky",
#     20 : "A writer typing on a laptop",
#     21 : "A cat lying on the table while its owner throws it a yellow and red ball to play with",
#     22 : "A dog running across the grass towards a child waiting for him with open arms",
#     23 : "A man walking on the beach with his son on the shore of a stormy sea with a purple kite",
#     24 : "An astronaut flying in space observing Planet Earth from black space in his white suit",
#     25 : "A girl swimming in the sea with curly blond hair next to a white sailboat with a black hull",
#     26 : "An apple in a fruit bowl during a huge sandstorm in the red Sahara desert near a waterhole",
#     27 : "A starred night in the dark as a shining comet passes by with a vortex of lightning stars",
#     28 : "A sailboat in the stormy sea under a black hole with flying fairies playing classical music",
#     29 : "A bird perched on a tree with code flowing through its branches with a clock showing 12 am",
#     30 : "A scene from War and Peace reinterpreted in a futuristic landscape with robots instead of humans",
#     31 : "A butterfly landing on a flower while a photographer tries to capture the moment",
#     32 : "A child playing in the snow with red mittens while building an ice castle that glows with blue light",
#     33 : "A chef cooking in a kitchen filled with steam as magical ingredients float in the air",
#     34 : "A car driving through mountains on a winding road with colorful autumn leaves swirling around",
#     35 : "A painter working on a canvas while the subjects of the painting step into the real world",
#     36 : "A wave crashing against rocks revealing an ancient underwater city with merfolk observers",
#     37 : "A robot building a structure made of translucent crystals on the surface of Mars",
#     38 : "A fox hiding in bushes with a pocket watch around its neck as it waits for the perfect moment",
#     39 : "A balloon floating in the sky carrying a tiny house with a chimney that releases rainbow smoke",
#     40 : "A writer typing on a laptop beside a window showing different dimensions with each keystroke",
# }

# rows = []
# for pid, text in PROMPTS.items():
#     rows.append({
#         "prompt_id": pid,
#         "length":    "short" if pid <= 20 else "long",
#         "pair_id":   pid if pid <= 20 else pid - 20,
#         "prompt":    text,
#     })

# df = pd.DataFrame(rows, columns=["prompt_id", "length", "pair_id", "prompt"])
# df.to_csv("datasets/PRISM-36K/metadata/prompts.csv", index=False)


### METADATA 2


IMAGES_ROOT = Path("images/")
OUT_CSV     = Path("metadata/images.csv")

FIELDNAMES = ["filename", "model", "prompt_id", "iter", "sha256", "width", "height"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for img_path in sorted(IMAGES_ROOT.rglob("*.png")):
    model     = img_path.parent.name          # folder name  e.g. "SANA"
    stem      = img_path.stem                 # e.g. "SANA_7_42"
    parts     = stem.split("_")
    prompt_id = int(parts[-2])
    iteration = int(parts[-1])

    with Image.open(img_path) as im:
        w, h = im.size

    rows.append({
        "filename":  img_path.name,
        "model":     model,
        "prompt_id": prompt_id,
        "iter":      iteration,
        "sha256":    sha256(img_path),
        "width":     w,
        "height":    h,
    })
    print(f"\r{len(rows):>6} / 36000", end="", flush=True)

print()

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows → {OUT_CSV}")