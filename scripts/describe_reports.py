"""Describe every lost and found report in a dataset with Claude (for the ranking eval).

Uses the same split as the benchmark: per dog, one photo is the found report and the
rest form the lost report. The lost report is described from up to 3 of its photos in
one call (as the app will do), the found report from its single photo.

    python scripts/describe_reports.py --data data/dogfacenet --out eval/dogfacenet_descriptions.jsonl

Resumable: responses are cached, and rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from petmatch.claude_client import ClaudeClient, ClaudeError  # noqa: E402
from petmatch.data import group_by_pet, load_from_folders, make_split  # noqa: E402
from petmatch.extract import PROMPT_VERSION, describe_pet  # noqa: E402

MAX_LOST_PHOTOS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dogfacenet")
    ap.add_argument("--out", default="eval/dogfacenet_descriptions.jsonl")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    split = make_split(load_from_folders(args.data), seed=args.seed)
    gallery = group_by_pet(split.gallery)
    jobs = []
    for q in split.queries:
        jobs.append(("found", q.pet_id, [q.path]))
    for pid, photos in sorted(gallery.items()):
        jobs.append(("lost", pid, [p.path for p in photos[:MAX_LOST_PHOTOS]]))

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["side"], r["pet_id"]))
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} reports, {len(todo)} to describe with {args.model}")

    client = ClaudeClient(model=args.model)
    if not client.api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, failures = time.time(), 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(describe_pet, client, photos): (side, pid, photos)
                   for side, pid, photos in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            side, pid, photos = futures[fut]
            try:
                desc = fut.result()
            except (ClaudeError, json.JSONDecodeError) as e:
                failures += 1
                print(f"  failed {side} {pid}: {str(e)[:120]}")
                continue
            f.write(json.dumps({"side": side, "pet_id": pid, "photos": [str(p) for p in photos],
                                "model": args.model, "prompt": PROMPT_VERSION, "description": desc},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 200 == 0 or i == len(todo):
                u = client.usage
                print(f"  {i}/{len(todo)}  ${u.cost_usd:.3f}  {time.time() - t0:.0f}s")
    print(f"Done: {failures} failures, ${client.usage.cost_usd:.3f}, "
          f"{client.usage.cached_calls} cached calls. Re-run to retry failures.")


if __name__ == "__main__":
    main()
