"""Download the DogFaceNet dataset (MIT license) into data/dogfacenet/<dog_id>/<n>.jpg.

8,363 face photos of 1,393 individual dogs, already cropped and aligned to 224x224.
Source: https://huggingface.co/datasets/dimidagd/DogFaceNet_224resize
(original project: https://github.com/GuillaumeMougeot/DogFaceNet)

Uses Hugging Face's dataset-viewer API, so no parquet library is needed.

    python scripts/fetch_dogfacenet.py            # all dogs
    python scripts/fetch_dogfacenet.py --max-dogs 300
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "dimidagd/DogFaceNet_224resize"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def get(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_page(offset: int) -> list[tuple[int, str, str]]:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                "offset": offset, "length": PAGE})
    data = json.loads(get(f"{API}?{q}"))
    return [(r["row_idx"], str(r["row"]["label"]), r["row"]["image"]["src"]) for r in data["rows"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dogfacenet")
    ap.add_argument("--max-dogs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    first = json.loads(get(f"{API}?" + urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": 0, "length": 1})))
    total = first["num_rows_total"]
    print(f"{total} rows; listing ...")

    rows: list[tuple[int, str, str]] = []
    with ThreadPoolExecutor(args.workers) as ex:
        for page in ex.map(fetch_page, range(0, total, PAGE)):
            rows.extend(page)

    dogs = sorted({label for _, label, _ in rows})
    if args.max_dogs:
        dogs = sorted(random.Random(args.seed).sample(dogs, args.max_dogs))
    keep = set(dogs)
    todo = [r for r in rows if r[1] in keep]
    print(f"{len(dogs)} dogs, {len(todo)} photos to download")

    out = Path(args.out)

    def save(row):
        idx, label, src = row
        dest = out / f"dog_{label}" / f"{idx}.jpg"
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(src))

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed(ex.submit(save, r) for r in todo):
            f.result()
            done += 1
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
