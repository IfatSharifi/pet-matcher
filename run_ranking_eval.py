"""Does adding colors, place and date to image similarity rank the right pet higher?

    python run_ranking_eval.py                         # all simulation scenarios
    python run_ranking_eval.py --scenarios base

For each scenario (assumptions about where and when pets are found, see petmatch/simulate.py):
  - weights are tuned on half the dogs and scored on the other half (no peeking)
  - four configurations are compared: image only, +colors, +place & date, all
Needs: DINOv2 embeddings (run_benchmark.py caches them) and eval/dogfacenet_descriptions.jsonl
(scripts/describe_reports.py). No API calls.

Outputs: a table per scenario, results/ranking_log.csv, and
results/ranking_examples.md - found reports with their top matches and Hebrew match reasons.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from petmatch.data import group_by_pet, load_from_folders, make_split
from petmatch.embedders import DinoV2Embedder
from petmatch.ranking import (Features, Weights, best_photo_similarity, color_matrix, haversine_matrix,
                              mrr, reason_he, score, species_compatible, true_ranks, tune)
from petmatch.simulate import SCENARIOS, simulate

RESULTS = Path("results")
GRID = {"color": [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        "dist": [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8],
        "date": [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5]}
CONFIGS = {  # name: (weights to tune, hard rules on, image weight)
    "image only": ([], False, 1.0),
    "+ colors": (["color"], False, 1.0),
    "+ place & date": (["dist", "date"], True, 1.0),
    "all": (["color", "dist", "date"], True, 1.0),
    # found report with no photo (text-only post): appearance comes only from the text's colors
    "no photo: place & date": (["dist", "date"], True, 0.0),
    "no photo: + colors": (["color", "dist", "date"], True, 0.0),
}


def load_descriptions(path: Path) -> dict[tuple[str, str], dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["side"], r["pet_id"]): r["description"] for r in rows}


def metrics(ranks: np.ndarray) -> dict:
    return {"recall@1": float(np.mean(ranks <= 1)), "recall@5": float(np.mean(ranks <= 5)),
            "mrr": mrr(ranks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dogfacenet")
    ap.add_argument("--descriptions", default="eval/dogfacenet_descriptions.jsonl")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- reports: one found photo per dog (query), its other photos = the lost report
    split = make_split(load_from_folders(args.data), seed=args.seed)
    lost_ids = sorted(group_by_pet(split.gallery))
    col = {pid: j for j, pid in enumerate(lost_ids)}
    found_ids = [q.pet_id for q in split.queries]
    truth = np.array([col[pid] for pid in found_ids])

    emb = DinoV2Embedder()
    photo_vecs = emb.embed([p.path for p in split.gallery])
    photo_report = np.array([col[p.pet_id] for p in split.gallery])
    query_vecs = emb.embed([q.path for q in split.queries])
    image = best_photo_similarity(query_vecs, photo_vecs, photo_report, len(lost_ids))

    desc = load_descriptions(Path(args.descriptions))
    missing = [k for k in [("found", p) for p in found_ids] + [("lost", p) for p in lost_ids] if k not in desc]
    if missing:
        raise SystemExit(f"{len(missing)} reports have no description yet; run scripts/describe_reports.py")
    f_colors = [desc[("found", p)]["primary_colors"] for p in found_ids]
    l_colors = [desc[("lost", p)]["primary_colors"] for p in lost_ids]
    color = color_matrix(f_colors, l_colors)
    f_species = [desc[("found", p)]["species"] for p in found_ids]
    l_species = [desc[("lost", p)]["species"] for p in lost_ids]
    species_ok = np.array([[species_compatible(fs, ls) for ls in l_species] for fs in f_species])

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(found_ids))
    train, test = np.sort(order[: len(order) // 2]), np.sort(order[len(order) // 2:])
    print(f"{len(found_ids)} found reports vs {len(lost_ids)} lost reports; "
          f"tune on {len(train)}, test on {len(test)}")

    RESULTS.mkdir(exist_ok=True)
    log_rows, examples = [], []
    for name in args.scenarios:
        params = SCENARIOS[name]
        lost_sim, found_sim = simulate(lost_ids, params, seed=args.seed)
        lp = [lost_sim[p][0] for p in lost_ids]
        fp = [found_sim[p][0] for p in found_ids]
        km = haversine_matrix([p.lat for p in fp], [p.lon for p in fp], [p.lat for p in lp], [p.lon for p in lp])
        ld = np.array([lost_sim[p][1].toordinal() for p in lost_ids])
        fd = np.array([found_sim[p][1].toordinal() for p in found_ids])
        days = fd[:, None] - ld[None, :]
        allowed = species_ok & (days >= -1)
        feats = Features(image, color, km.astype(np.float32), days, allowed)
        f_train, f_test = feats.subset(train), feats.subset(test)

        print(f"\nScenario '{name}': found median {params.found_median_km} km, "
              f"{params.found_far_share:.0%} far, mean {params.found_mean_days} days")
        print(f"  {'config':<24}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}   weights")
        tuned = {}
        for cname, (free, rules, w_img) in CONFIGS.items():
            grid = {k: GRID[k] for k in free}
            if w_img == 0:  # without the image term, scale is free: start colors at 1
                grid = {**grid, "color": [0, 0.5, 1.0, 2.0]} if "color" in free else grid
            w = tune(f_train, truth[train], Weights(image=w_img, use_rules=rules), grid)
            tuned[cname] = w
            r = true_ranks(score(f_test, w), truth[test])
            m = metrics(r)
            wtxt = ", ".join(f"{k}={getattr(w, k)}" for k in free) or "-"
            at_edge = [k for k in free if getattr(w, k) == max(grid[k]) and max(grid[k]) > 0]
            if at_edge:
                wtxt += f"  (!) at grid max: {at_edge} - widen GRID"
            print(f"  {cname:<24}{m['recall@1']:>10.1%}{m['recall@5']:>10.1%}{m['mrr']:>8.3f}   {wtxt}")
            log_rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "scenario": name,
                             "config": cname, "seed": args.seed, "test_queries": len(test),
                             **{k: round(v, 4) for k, v in m.items()},
                             "w_color": w.color, "w_dist": w.dist, "w_date": w.date})

        # which image-only misses did the full ranking fix, by the found dog's main color?
        r_img = true_ranks(score(f_test, tuned["image only"]), truth[test])
        r_all = true_ranks(score(f_test, tuned["all"]), truth[test])
        main = [(f_colors[i] or ["?"])[0] for i in test]
        miss = r_img > 1
        fixed = miss & (r_all == 1)
        black = np.array([c == "black" for c in main])
        print(f"  image-only top-1 misses: {miss.sum()}, fixed by 'all': {fixed.sum()} "
              f"(black dogs: {(miss & black).sum()} missed, {(fixed & black).sum()} fixed); "
              f"newly broken: {((r_img == 1) & (r_all > 1)).sum()}")

        if name == "base":
            s_all = score(feats, tuned["all"])
            for qi in [int(i) for i in test[fixed][:4]] + [int(i) for i in test[r_all > 5][:2]]:
                top = np.argsort(-s_all[qi])[:3]
                lines = [f"### Found report: dog {found_ids[qi]} (true owner: {lost_ids[truth[qi]]})", ""]
                for rank, j in enumerate(top, 1):
                    mark = "✔" if j == truth[qi] else " "
                    reason = reason_he(f_colors[qi], l_colors[j], float(km[qi, j]), int(days[qi, j]),
                                       float(image[qi, j]))
                    lines.append(f"{rank}. {mark} {lost_ids[j]} - {reason}")
                examples.append("\n".join(lines))

    with (RESULTS / "ranking_log.csv").open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        if f.tell() == 0:
            w.writeheader()
        w.writerows(log_rows)
    if examples:
        (RESULTS / "ranking_examples.md").write_text(
            "# Ranking examples (scenario 'base', simulated places and dates)\n\n" + "\n\n".join(examples) + "\n",
            encoding="utf-8")
    print(f"\nLogged to {RESULTS / 'ranking_log.csv'}; examples in {RESULTS / 'ranking_examples.md'}")


if __name__ == "__main__":
    main()
