"""Benchmark image models on "does the right pet appear in the top 5?".

Examples:
    python run_benchmark.py --data data/my_pets
    python run_benchmark.py --csv data/petface_dogs.csv --max-pets 300 --models clip dinov2 histogram

Outputs:
    - a results table in the terminal
    - results/benchmark_log.csv   (one row per model per run - your eval history)
    - results/misses_<model>.csv  (every query whose pet was NOT in the top 5, to inspect by eye)
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from petmatch.data import load_from_csv, load_from_folders, make_split
from petmatch.embedders import EMBEDDERS
from petmatch.index import VectorIndex
from petmatch.metrics import first_hit_ranks, mean_reciprocal_rank, recall_at_k

RESULTS = Path("results")
SEARCH_DEPTH = 50  # gallery photos retrieved per query before de-duplicating by pet


def run_model(name: str, split, dataset_label: str, seed: int) -> dict:
    embedder = EMBEDDERS[name]()
    t0 = time.perf_counter()
    gallery_vecs = embedder.embed([p.path for p in split.gallery])
    query_vecs = embedder.embed([p.path for p in split.queries])
    embed_s = time.perf_counter() - t0

    index = VectorIndex()
    index.add(gallery_vecs, [p.pet_id for p in split.gallery])
    ranked_ids, scores = index.search(query_vecs, k=SEARCH_DEPTH)
    ranks = first_hit_ranks(ranked_ids, [q.pet_id for q in split.queries])

    write_misses(name, split, ranked_ids, scores, ranks)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_label,
        "seed": seed,
        "model": embedder.name,
        "queries": len(split.queries),
        "gallery_photos": len(split.gallery),
        "gallery_pets": len({p.pet_id for p in split.gallery}),
        "recall@1": round(recall_at_k(ranks, 1), 4),
        "recall@5": round(recall_at_k(ranks, 5), 4),
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "embed_seconds": round(embed_s, 1),
    }


def write_misses(name, split, ranked_ids, scores, ranks) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"misses_{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_photo", "true_pet", "rank_of_true_pet(0=not_found)",
                    "top1_pet", "top1_score"])
        for q, ids, sc, r in zip(split.queries, ranked_ids, scores, ranks):
            if r == 0 or r > 5:
                w.writerow([q.path, q.pet_id, int(r), ids[0], f"{sc[0]:.3f}"])


def append_log(rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "benchmark_log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    return log


def print_table(rows: list[dict]) -> None:
    print(f"\n{'model':<18}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}{'embed s':>10}")
    for r in sorted(rows, key=lambda r: -r["recall@5"]):
        print(f"{r['model']:<18}{r['recall@1']:>10.1%}{r['recall@5']:>10.1%}"
              f"{r['mrr']:>8.3f}{r['embed_seconds']:>10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="folder with one sub-folder of photos per pet")
    src.add_argument("--csv", help="CSV with columns path,pet_id")
    ap.add_argument("--models", nargs="+", default=["histogram", "clip", "dinov2"],
                    choices=sorted(EMBEDDERS))
    ap.add_argument("--max-pets", type=int, default=None, help="sample this many pets (faster runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    photos = load_from_folders(args.data) if args.data else load_from_csv(args.csv)
    split = make_split(photos, max_pets=args.max_pets, seed=args.seed)
    label = Path(args.data or args.csv).name + (f" (max {args.max_pets} pets)" if args.max_pets else "")
    print(f"Dataset: {label} -> {split.summary()}")

    rows = []
    for m in args.models:
        print(f"\nRunning {m} ...")
        rows.append(run_model(m, split, label, args.seed))
    print_table(rows)
    log = append_log(rows)
    print(f"\nLogged to {log}; misses per model in {RESULTS}/misses_<model>.csv")


if __name__ == "__main__":
    main()
