"""Fill a demo database with lost-dog reports from DogFaceNet. No API calls.

    python scripts/seed_demo.py --dogs 300 --db demo.db

Each dog's gallery photos become a lost report, placed and dated with the 'base'
simulation (dates in the last 60 days) and described with the Haiku descriptions in
eval/dogfacenet_descriptions.jsonl. Contacts are fake (@example.com).

The dogs' held-out "found" photos are listed in <db>.found.txt: report one of them
as found in the app and the right lost dog should come out on top.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from petmatch.data import group_by_pet, load_from_folders, make_split  # noqa: E402
from petmatch.embedders import EMBEDDERS  # noqa: E402
from petmatch.simulate import SCENARIOS, simulate  # noqa: E402
from petmatch.store import Report, Store  # noqa: E402

NAMES = ["לוסי", "שוקו", "טופי", "רקסי", "מיקה", "בונבון", "לולה", "צ'ארלי", "בלו", "גינגר", "נלה", "בובי"]
OWNERS = ["נועה", "אבי", "מיכל", "יוסי", "דנה", "רון", "שירה", "עומר", "תמר", "איתי"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dogfacenet")
    ap.add_argument("--descriptions", default="eval/dogfacenet_descriptions.jsonl")
    ap.add_argument("--db", default="demo.db")
    ap.add_argument("--dogs", type=int, default=300)
    ap.add_argument("--embedder", default="dinov2", choices=sorted(EMBEDDERS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if Path(args.db).exists():
        raise SystemExit(f"{args.db} exists; delete it first to reseed")
    split = make_split(load_from_folders(args.data), seed=args.seed)
    gallery = group_by_pet(split.gallery)
    queries = {q.pet_id: q for q in split.queries}
    desc = {(r["side"], r["pet_id"]): r["description"]
            for r in map(json.loads, Path(args.descriptions).read_text(encoding="utf-8").splitlines())}

    rng = random.Random(args.seed)
    dogs = sorted(rng.sample(sorted(gallery), min(args.dogs, len(gallery))))
    params = replace(SCENARIOS["base"], window_days=60)  # every date within the last 60 days
    lost, _ = simulate(dogs, params, seed=args.seed, start=date.today() - timedelta(days=60))
    store, embedder = Store(args.db), EMBEDDERS[args.embedder]()

    for n, pid in enumerate(dogs):
        photos = [p.path for p in gallery[pid]]
        d = desc[("lost", pid)]
        place, day = lost[pid]
        report = Report(side="lost", species=d["species"], colors=d["primary_colors"], event_date=day,
                        description=d, pet_name=rng.choice(NAMES), lat=place.lat, lon=place.lon,
                        place_precision="exact", area_text=place.city, contact_name=rng.choice(OWNERS),
                        contact_email=f"demo-owner-{n}@example.com", source="demo")
        store.add_report(report, list(zip(map(str, photos), embedder.embed(photos))))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(dogs)}")

    found_list = Path(args.db + ".found.txt")
    found_list.write_text("\n".join(f"{queries[p].path}\t{lost[p][0].city}\t{lost[p][0].lat:.5f},{lost[p][0].lon:.5f}"
                                    for p in dogs) + "\n", encoding="utf-8")
    store.close()  # writes everything into the .db file, so it can be copied on its own
    print(f"Seeded {len(dogs)} lost reports into {args.db}; test photos in {found_list}")


if __name__ == "__main__":
    main()
