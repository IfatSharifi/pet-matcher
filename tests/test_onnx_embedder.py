"""ONNX DINOv2 embedder: preprocessing always; model equivalence when the model file is present."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from petmatch.embedders import DinoV2OnnxEmbedder

MODEL = Path(os.environ.get("PETMATCH_MODELS", ".models")) / "dinov2-small.onnx"


class TestPreprocess(unittest.TestCase):
    def test_shape_crop_and_normalization(self):
        # wide image: left half black, right half white, so the center crop shows both
        arr = np.zeros((300, 600, 3), np.uint8)
        arr[:, 300:] = 255
        x = DinoV2OnnxEmbedder.preprocess(Image.fromarray(arr))
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertEqual(x.dtype, np.float32)
        black = (0 - DinoV2OnnxEmbedder.MEAN) / DinoV2OnnxEmbedder.STD
        white = (1 - DinoV2OnnxEmbedder.MEAN) / DinoV2OnnxEmbedder.STD
        np.testing.assert_allclose(x[:, 112, 5], black, atol=1e-3)     # left edge of the crop
        np.testing.assert_allclose(x[:, 112, 218], white, atol=1e-3)   # right edge

    def test_small_and_tall_images(self):
        for size in [(50, 80), (224, 224), (180, 1000)]:
            x = DinoV2OnnxEmbedder.preprocess(Image.new("RGB", size, (120, 80, 40)))
            self.assertEqual(x.shape, (3, 224, 224), size)


@unittest.skipUnless(MODEL.exists(), "ONNX model not downloaded (run the app or build once)")
class TestOnnxModel(unittest.TestCase):
    def test_embeds_and_separates_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            rng = np.random.default_rng(0)
            for i, base in enumerate([(30, 30, 30), (200, 160, 110), (30, 30, 30)]):
                p = Path(tmp) / f"{i}.png"
                Image.fromarray(np.clip(np.array(base) + rng.integers(-15, 16, (240, 240, 3)), 0, 255)
                                .astype(np.uint8)).save(p)
                paths.append(p)
            emb = DinoV2OnnxEmbedder(MODEL.parent)
            emb.use_cache = False
            v = emb.embed(paths)
            self.assertEqual(v.shape, (3, 384))
            n = v / np.linalg.norm(v, axis=1, keepdims=True)
            self.assertGreater(n[0] @ n[2], n[0] @ n[1])  # the two dark images are closer to each other

    @unittest.skipUnless(os.environ.get("PETMATCH_COMPARE_TORCH") == "1", "set PETMATCH_COMPARE_TORCH=1 (needs torch)")
    def test_matches_pytorch(self):
        from petmatch.embedders import DinoV2Embedder
        photos = sorted(Path("data/dogfacenet").rglob("*.jpg"))[::400][:20]
        if len(photos) < 5:
            self.skipTest("DogFaceNet not downloaded")
        a, b = DinoV2Embedder(), DinoV2OnnxEmbedder(MODEL.parent)
        a.use_cache = b.use_cache = False
        x, y = a.embed(photos), b.embed(photos)
        cos = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        self.assertGreater(cos.min(), 0.9999)


if __name__ == "__main__":
    unittest.main()
