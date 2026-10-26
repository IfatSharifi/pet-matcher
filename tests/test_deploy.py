"""Server behaviour: rate limit, health check, demo banner, Claude outages, thread-safe embedding."""
import os
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

os.environ["PETMATCH_NO_AUTOSTART"] = "1"

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

import numpy as np
from PIL import Image

import petmatch.embedders as E
from petmatch.claude_client import ClaudeError
from tests.test_app import jpeg


class BrokenClaude:
    def extract(self, *a, **k):
        raise ClaudeError("HTTP 529: overloaded")


@unittest.skipIf(TestClient is None, "fastapi not installed")
class DeployTest(unittest.TestCase):
    def make(self, claude, limiter=None, demo=False):
        from app import RateLimiter, create_app
        from petmatch.embedders import ColorHistogramEmbedder
        from petmatch.notify import ConsoleNotifier
        from petmatch.service import MatchService
        from petmatch.store import Store
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.old_cache, E.CACHE_DIR = E.CACHE_DIR, d / "cache"
        emb = ColorHistogramEmbedder()
        emb.use_cache = False
        self.svc = MatchService(Store(d / "x.db"), emb, claude, ConsoleNotifier(quiet=True))
        self.client = TestClient(create_app(self.svc, d / "up", limiter=limiter, demo=demo))
        self.RateLimiter = RateLimiter

    def tearDown(self):
        self.svc.store.close()
        E.CACHE_DIR = self.old_cache
        self.tmp.cleanup()

    def post_lost(self, n=0):
        return self.client.post("/report/lost", data={"event_date": date.today().isoformat(), "contact_phone": "050",
                                                      "species": "dog"},
                                files=[("photos", ("d.jpg", jpeg((20, 20, 20), seed=n), "image/jpeg"))],
                                follow_redirects=False)


class TestServer(DeployTest):
    def test_health_and_demo_banner(self):
        from tests.test_service import FakeClaude
        self.make(FakeClaude(), demo=True)
        self.assertEqual(self.client.get("/health").json()["ok"], True)
        self.assertIn("זהו אתר הדגמה", self.client.get("/").text)

    def test_no_banner_by_default(self):
        from tests.test_service import FakeClaude
        self.make(FakeClaude())
        self.assertNotIn("זהו אתר הדגמה", self.client.get("/").text)

    def test_claude_outage_still_saves_the_report(self):
        self.make(BrokenClaude())
        r = self.post_lost()
        self.assertEqual(r.status_code, 303)
        self.assertEqual(len(self.svc.store.open_reports("lost")), 1)
        shot = self.client.post("/import", files=[("screenshot", ("s.jpg", jpeg((240, 240, 240)), "image/jpeg"))])
        self.assertEqual(shot.status_code, 400)
        self.assertIn("לא הצלחנו לקרוא", shot.text)

    def test_rate_limit_skips_claude_but_keeps_reports(self):
        from tests.test_service import FakeClaude
        claude = FakeClaude()
        from app import RateLimiter
        self.make(claude, limiter=RateLimiter(per_ip_per_hour=2, per_day=100))
        for n in range(4):
            self.assertEqual(self.post_lost(n).status_code, 303)
        self.assertEqual(claude.calls, 2)                       # only the first two were described
        self.assertEqual(len(self.svc.store.open_reports("lost")), 4)
        shot = self.client.post("/import", files=[("screenshot", ("s.jpg", jpeg((240, 240, 240)), "image/jpeg"))])
        self.assertIn("נסו שוב מאוחר יותר", shot.text)


@unittest.skipIf(TestClient is None, "fastapi not installed")
class TestRateLimiter(unittest.TestCase):
    def test_daily_cap(self):
        from app import RateLimiter
        lim = RateLimiter(per_ip_per_hour=100, per_day=3)
        self.assertEqual([lim.allow(f"10.0.0.{i}") for i in range(5)], [True, True, True, False, False])


class TestEmbedderThreads(unittest.TestCase):
    def test_parallel_embeds_with_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, E.CACHE_DIR = E.CACHE_DIR, Path(tmp) / "c"
            try:
                paths = []
                for i in range(24):
                    p = Path(tmp) / f"{i}.png"
                    Image.fromarray(np.full((32, 32, 3), i * 10, np.uint8)).save(p)
                    paths.append(p)
                emb, errors = E.ColorHistogramEmbedder(), []

                def work(chunk):
                    try:
                        emb.embed(chunk)
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
                threads = [threading.Thread(target=work, args=(paths[i::6],)) for i in range(6)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(errors, [])
                self.assertEqual(emb.embed(paths).shape[0], 24)
            finally:
                E.CACHE_DIR = old


if __name__ == "__main__":
    unittest.main()
