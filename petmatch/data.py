"""Load a pet re-identification dataset and split it into "found" queries and a "lost" gallery.

Two input layouts are supported:

1. Folder per pet (easiest for your own photos):
       data/
         rex/      photo1.jpg  photo2.jpg ...
         mitzi/    a.png  b.png ...
   Every sub-folder name is treated as one individual animal.

2. A CSV with two columns, `path,pet_id` (paths relative to the CSV's folder or absolute).
   Use this for public datasets such as PetFace, whatever their folder layout is.

The split mirrors the real app: for each pet with 2+ photos, ONE photo plays the
"found pet" report (query) and the remaining photos are the owner's "lost pet"
report (gallery). Pets with a single photo go to the gallery only, as distractors -
just like real lost reports that nobody has found yet.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Photo:
    path: Path
    pet_id: str


@dataclass
class Split:
    queries: list[Photo]   # "found" reports
    gallery: list[Photo]   # "lost" reports (incl. distractors)

    def summary(self) -> str:
        gallery_ids = {p.pet_id for p in self.gallery}
        return (f"{len(self.queries)} found-report queries | "
                f"{len(self.gallery)} gallery photos of {len(gallery_ids)} pets")


def load_from_folders(root: str | Path) -> list[Photo]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")
    photos = []
    for pet_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(pet_dir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS:
                photos.append(Photo(img, pet_dir.name))
    if not photos:
        raise ValueError(f"No images found under {root} (expected one sub-folder per pet)")
    return photos


def load_from_csv(csv_path: str | Path) -> list[Photo]:
    csv_path = Path(csv_path)
    base = csv_path.parent
    photos = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"path", "pet_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV must have columns path,pet_id (missing: {missing})")
        for row in reader:
            p = Path(row["path"])
            photos.append(Photo(p if p.is_absolute() else base / p, str(row["pet_id"])))
    if not photos:
        raise ValueError(f"CSV is empty: {csv_path}")
    return photos


def group_by_pet(photos: list[Photo]) -> dict[str, list[Photo]]:
    groups: dict[str, list[Photo]] = {}
    for p in photos:
        groups.setdefault(p.pet_id, []).append(p)
    return groups


def make_split(photos: list[Photo], max_pets: int | None = None, seed: int = 0) -> Split:
    """One query per pet that has 2+ photos; everything else goes to the gallery."""
    rng = random.Random(seed)
    groups = group_by_pet(photos)
    pet_ids = sorted(groups)
    rng.shuffle(pet_ids)
    if max_pets:
        pet_ids = pet_ids[:max_pets]

    queries, gallery = [], []
    for pid in sorted(pet_ids):
        imgs = sorted(groups[pid], key=lambda p: str(p.path))
        if len(imgs) >= 2:
            q = rng.choice(imgs)
            queries.append(q)
            gallery.extend(i for i in imgs if i is not q)
        else:
            gallery.extend(imgs)
    if not queries:
        raise ValueError("No pet has 2+ photos, so there is nothing to match. "
                         "Each pet needs at least two photos.")
    return Split(queries, gallery)
