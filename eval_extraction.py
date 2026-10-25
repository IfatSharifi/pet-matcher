"""Evaluate Claude's vision extraction.

  python eval_extraction.py screenshots                 # field accuracy on eval/screenshots
  python eval_extraction.py describe --dogs 30          # description consistency on DogFaceNet
  python eval_extraction.py screenshots --model claude-sonnet-5-5

Needs ANTHROPIC_API_KEY. Responses are cached in .cache/claude, so re-runs are free.
Outputs:
  results/extraction_log.csv           one row per run (accuracy per field, cost, latency)
  results/screenshot_errors_<model>.csv every wrong field with truth vs prediction
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from petmatch.claude_client import ClaudeClient, ClaudeError
from petmatch.extract import PROMPT_VERSION, describe_pet, read_screenshot
from petmatch.schemas import SCREENSHOT_SCHEMA, validate
from petmatch.scoring import SCREENSHOT_FIELDS, consistency, field_accuracy, score_screenshot

RESULTS = Path("results")


def run_parallel(fn, items, workers):
    def safe(item):
        try:
            return fn(item), None
        except (ClaudeError, json.JSONDecodeError) as e:
            return None, str(e)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(safe, items))


def eval_screenshots(client: ClaudeClient, folder: Path, workers: int) -> dict:
    labels = [json.loads(line) for line in
              (folder / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = run_parallel(lambda l: read_screenshot(client, folder / l["file"]), labels, workers)

    scores, errors, failures, schema_problems = [], [], 0, 0
    for truth, (pred, err) in zip(labels, outputs):
        if pred is None:
            failures += 1
            scores.append({f: False for f in SCREENSHOT_FIELDS})
            errors.append([truth["file"], "CALL FAILED", "", err])
            continue
        schema_problems += bool(validate(pred, SCREENSHOT_SCHEMA))
        s = score_screenshot(truth, pred)
        scores.append(s)
        pred_view = {**pred, "colors": pred.get("primary_colors")}
        for f, ok in s.items():
            if ok is False:
                errors.append([truth["file"], f, json.dumps(truth.get(f), ensure_ascii=False),
                               json.dumps(pred_view.get(f), ensure_ascii=False)])

    RESULTS.mkdir(exist_ok=True)
    err_file = RESULTS / f"screenshot_errors_{client.model}.csv"
    with err_file.open("w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel shows Hebrew
        w = csv.writer(f)
        w.writerow(["file", "field", "truth", "predicted"])
        w.writerows(errors)

    if failures == len(labels):
        raise SystemExit(f"Every call failed. First error: {outputs[0][1]}")
    acc = field_accuracy(scores)
    print(f"\nScreenshot extraction, {len(labels)} posts, model {client.model}")
    for f in SCREENSHOT_FIELDS + ["all_fields"]:
        n = sum(1 for s in scores if s.get(f) is not None) if f != "all_fields" else len(scores)
        print(f"  {f:<15}{acc[f]:>7.1%}   (n={n})")
    print(f"  failed calls: {failures}, schema problems: {schema_problems}; errors in {err_file}")
    return {"n": len(labels), "failed_calls": failures, **{f"acc_{k}": round(v, 4) for k, v in acc.items()}}


def eval_describe(client: ClaudeClient, data: Path, n_dogs: int, per_dog: int, seed: int,
                  workers: int) -> dict:
    rng = random.Random(seed)
    dogs = [d for d in sorted(data.iterdir()) if d.is_dir() and len(list(d.glob("*.jpg"))) >= per_dog]
    dogs = rng.sample(dogs, min(n_dogs, len(dogs)))
    jobs = [(d.name, p) for d in dogs for p in rng.sample(sorted(d.glob("*.jpg")), per_dog)]
    outputs = run_parallel(lambda j: describe_pet(client, [j[1]]), jobs, workers)

    by_dog: dict[str, list[dict]] = {}
    failures = 0
    for (dog, _), (pred, err) in zip(jobs, outputs):
        if pred is None:
            failures += 1
        else:
            by_dog.setdefault(dog, []).append(pred)
    if failures == len(jobs):
        raise SystemExit(f"Every call failed. First error: {outputs[0][1]}")
    per = [consistency(v) for v in by_dog.values() if len(v) >= 2]
    keys = per[0].keys() if per else []
    agg = {k: statistics.mean(p[k] for p in per) for k in keys}
    preds = [p for v in by_dog.values() for p in v]
    agg["species_is_dog"] = sum(p["species"] == "dog" for p in preds) / max(len(preds), 1)

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"descriptions_{client.model}.jsonl").open("w", encoding="utf-8") as f:
        for (dog, path), (pred, err) in zip(jobs, outputs):
            f.write(json.dumps({"dog": dog, "photo": str(path), "description": pred, "error": err},
                               ensure_ascii=False) + "\n")

    print(f"\nDescription consistency, {len(by_dog)} dogs x {per_dog} photos, model {client.model}")
    for k, v in agg.items():
        print(f"  {k:<15}{v:>7.1%}")
    print(f"  failed calls: {failures}")
    return {"n": len(jobs), "failed_calls": failures, **{f"agree_{k}": round(v, 4) for k, v in agg.items()}}


def log(row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "extraction_log.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union of columns across tasks
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["screenshots", "describe"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--screenshots", default="eval/screenshots")
    ap.add_argument("--data", default="data/dogfacenet")
    ap.add_argument("--dogs", type=int, default=30)
    ap.add_argument("--per-dog", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    client = ClaudeClient(model=args.model)
    if not client.api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. In PowerShell: $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")
    if args.task == "screenshots":
        result = eval_screenshots(client, Path(args.screenshots), args.workers)
    else:
        result = eval_describe(client, Path(args.data), args.dogs, args.per_dog, args.seed, args.workers)

    u = client.usage
    fresh = u.calls - u.cached_calls
    avg_s = u.seconds / fresh if fresh else 0.0
    print(f"  cost ${u.cost_usd:.4f} for {fresh} new calls ({u.cached_calls} cached), "
          f"avg {avg_s:.1f}s per call, {u.input_tokens} in / {u.output_tokens} out tokens")
    log({"timestamp": datetime.now().isoformat(timespec="seconds"), "task": args.task,
         "model": args.model, "prompt": PROMPT_VERSION, **result,
         "cost_usd": round(u.cost_usd, 5), "avg_call_s": round(avg_s, 2)})


if __name__ == "__main__":
    main()
