"""Image -> vector models. Each embedder turns a list of image paths into an (N, D) array.

Models download from Hugging Face on first use (a few hundred MB) and are cached.
Embeddings are cached on disk too, so re-running the benchmark is fast.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image

CACHE_DIR = Path(".cache/embeddings")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


_LOCK = threading.Lock()  # web servers embed from several threads; one model run at a time


class Embedder:
    name = "base"
    use_cache = True  # the web app turns this off: it stores vectors in its database

    def embed(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Embed images, reusing any photo already embedded by this model (per-photo cache)."""
        with _LOCK:
            if not self.use_cache:
                return np.vstack([self._embed_batch([_open_rgb(p) for p in paths[i:i + batch_size]])
                                  for i in range(0, len(paths), batch_size)]).astype(np.float32)
            return self._embed_cached(paths, batch_size)

    def _embed_cached(self, paths: list[Path], batch_size: int) -> np.ndarray:
        cache = self._load_cache()
        keys = [self._key(p) for p in paths]
        todo = [p for p, k in zip(paths, keys) if k not in cache]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            vecs = self._embed_batch([_open_rgb(p) for p in chunk]).astype(np.float32)
            for p, v in zip(chunk, vecs):
                cache[self._key(p)] = v
            if len(todo) > batch_size:  # progress only for bulk runs, not one upload at a time
                print(f"  {self.name}: {min(i + batch_size, len(todo))}/{len(todo)} new images", end="\r")
        if todo:
            if len(todo) > batch_size:
                print()
            self._save_cache(cache)
        return np.stack([cache[k] for k in keys])

    @staticmethod
    def _key(p: Path) -> str:
        # path + modification time: an edited photo gets re-embedded
        return hashlib.sha1(f"{p.resolve()}|{p.stat().st_mtime_ns}".encode()).hexdigest()

    def _cache_path(self) -> Path:
        return CACHE_DIR / f"{self.name}.npz"

    def _load_cache(self) -> dict[str, np.ndarray]:
        f = self._cache_path()
        if not f.exists():
            return {}
        with np.load(f) as z:
            return dict(zip(z["keys"].tolist(), z["vectors"]))

    def _save_cache(self, cache: dict[str, np.ndarray]) -> None:
        f = self._cache_path()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
        tmp.replace(f)

    def _embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError


class ClipEmbedder(Embedder):
    name = "clip-vit-b32"
    model_id = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).eval()

    def _embed_batch(self, images):
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
            # Newer transformers versions may return an output object instead of a tensor.
            if not self.torch.is_tensor(out):
                emb = getattr(out, "image_embeds", None)
                out = emb if emb is not None else out.pooler_output
        return out.cpu().numpy()


class DinoV2Embedder(Embedder):
    name = "dinov2-small"
    model_id = "facebook/dinov2-small"

    def __init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval()

    def _embed_batch(self, images):
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            out = self.model(**inputs)
        # pooler_output = the normalized [CLS] token, a whole-image summary.
        return out.pooler_output.cpu().numpy()


class ColorHistogramEmbedder(Embedder):
    """No-download baseline: a coarse color histogram. Any real model must beat it.

    Also used by the tests, so the pipeline can be checked without torch.
    """
    name = "color-histogram"

    def _embed_batch(self, images):
        feats = []
        for im in images:
            arr = np.asarray(im.resize((64, 64)), dtype=np.int32) // 64  # 4 levels per channel
            codes = arr[..., 0] * 16 + arr[..., 1] * 4 + arr[..., 2]
            hist = np.bincount(codes.ravel(), minlength=64).astype(np.float32)
            feats.append(hist / hist.sum())
        return np.stack(feats)


EMBEDDERS = {
    "clip": ClipEmbedder,
    "dinov2": DinoV2Embedder,
    "histogram": ColorHistogramEmbedder,
}
