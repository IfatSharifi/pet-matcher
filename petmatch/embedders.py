"""Image -> vector models. Each embedder turns a list of image paths into an (N, D) array.

Models download from Hugging Face on first use (a few hundred MB) and are cached.
Embeddings are cached on disk too, so re-running the benchmark is fast.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
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


class DinoV2OnnxEmbedder(Embedder):
    """The same DINOv2-small, run with ONNX Runtime instead of PyTorch: no torch or transformers.

    The server uses this one. Peak memory is about 350 MB instead of about 900 MB, which fits
    a 512 MB host, and the vectors match the PyTorch model (cosine 1.00000 on 200 DogFaceNet
    photos), so the benchmark results carry over unchanged.

    Model: onnx-community/dinov2-small (an ONNX export of facebook/dinov2-small). It returns the
    token sequence; the first token is what the PyTorch model calls pooler_output.
    """
    name = "dinov2-small-onnx"
    url = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_dir: str | Path | None = None) -> None:
        import onnxruntime as ort
        path = self.download(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))  # 0 = all cores
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def download(cls, model_dir: str | Path | None = None) -> Path:
        """Fetch the model once (88 MB) into PETMATCH_MODELS (default .models/)."""
        folder = Path(model_dir or os.environ.get("PETMATCH_MODELS", ".models"))
        path = folder / "dinov2-small.onnx"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(cls.url, tmp)
            tmp.replace(path)  # never leave a half-downloaded model behind
        return path

    @classmethod
    def preprocess(cls, im: Image.Image) -> np.ndarray:
        """DINOv2's own preprocessing in numpy: shortest side to 256 (bicubic), center 224 crop,
        scale to 0..1, normalize with ImageNet mean/std, channels first."""
        w, h = im.size
        s = 256 / min(w, h)
        im = im.resize((max(round(w * s), 256), max(round(h * s), 256)), Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        a = np.asarray(im.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
        return ((a - cls.MEAN) / cls.STD).transpose(2, 0, 1)

    def _embed_batch(self, images):
        batch = np.stack([self.preprocess(im) for im in images])
        tokens = self.session.run(None, {self.input_name: batch})[0]
        return tokens[:, 0, :]


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
    "dinov2": DinoV2Embedder,          # PyTorch; research and benchmarks
    "dinov2-onnx": DinoV2OnnxEmbedder,  # same model, no PyTorch; the app's default
    "histogram": ColorHistogramEmbedder,
}
