"""Web tests with FastAPI's TestClient. Skipped if FastAPI isn't installed."""
import io
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

os.environ["PETMATCH_NO_AUTOSTART"] = "1"

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

import numpy as np
from PIL import Image

import petmatch.embedders as E

TODAY = date.today()


def jpeg(rgb, seed=0, gps=False) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(rgb) + rng.integers(-10, 11, (64, 64, 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if gps:
        exif = Image.Exif()
        exif[0x8825] = {1: "N", 2: (32.0, 4.0, 5.0), 3: "E", 4: (34.0, 49.0, 30.0)}  # GPS IFD
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


@unittest.skipIf(TestClient is None, "fastapi not installed")
class AppTest(unittest.TestCase):
    def setUp(self):
        from app import create_app
        from petmatch.embedders import ColorHistogramEmbedder
        from petmatch.notify import ConsoleNotifier
        from petmatch.service import MatchService
        from petmatch.store import Store
        from tests.test_service import FakeClaude

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.old_cache, E.CACHE_DIR = E.CACHE_DIR, self.dir / "cache"
        self.mail = ConsoleNotifier(quiet=True)
        self.svc = MatchService(Store(self.dir / "a.db"), ColorHistogramEmbedder(), FakeClaude(), self.mail)
        self.uploads = self.dir / "uploads"
        self.client = TestClient(create_app(self.svc, self.uploads))

    def tearDown(self):
        self.svc.store.close()
        E.CACHE_DIR = self.old_cache
        self.tmp.cleanup()

    def post_report(self, side, rgb=(20, 20, 20), days_ago=1, lat="32.068", lon="34.824", **fields):
        data = {"event_date": (TODAY - timedelta(days=days_ago)).isoformat(), "lat": lat, "lon": lon,
                "species": "dog", **fields}
        files = [("photos", ("dog.jpg", jpeg(rgb, seed=days_ago), "image/jpeg"))] if rgb else []
        return self.client.post(f"/report/{side}", data=data, files=files, follow_redirects=False)


class TestPages(AppTest):
    def test_home_and_forms_render_in_hebrew_rtl(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn('dir="rtl"', home.text)
        self.assertIn("איבדתי", home.text)
        form = self.client.get("/report/found")
        self.assertIn("דיווח על חיה שנמצאה", form.text)
        self.assertIn("שחור", form.text)  # color swatches in Hebrew
        self.assertEqual(self.client.get("/report/adopted").status_code, 404)

    def test_full_flow_lost_then_found(self):
        r = self.post_report("lost", days_ago=3, contact_email="owner@example.com", pet_name="שוקו")
        self.assertEqual(r.status_code, 303)
        owner_url = r.headers["location"]
        r = self.post_report("found", lat="32.072", lon="34.830", contact_phone="0505551234",
                             contact_name="מורן")
        finder_url = r.headers["location"]

        finder = self.client.get(finder_url)
        self.assertIn("שוקו", finder.text)
        self.assertIn("התאמה חזקה", finder.text)
        self.assertNotIn("owner@example.com", finder.text)       # owners stay private
        self.assertEqual(finder.text.count("החיה חזרה לבעלים"), 1)  # one close button, not per card

        owner = self.client.get(owner_url)
        self.assertIn("0505551234", owner.text)                   # owner sees the finder's phone
        self.assertEqual(len(self.mail.sent), 1)
        self.assertIn("?t=", self.mail.sent[0]["html"])           # email links to the private page

        # a photo of a match is served
        lost_id = int(owner_url.split("/")[2].split("?")[0])
        self.assertEqual(self.client.get(f"/photos/{lost_id}/0").headers["content-type"], "image/jpeg")

        # owner confirms: both reports close
        found_id = int(finder_url.split("/")[2].split("?")[0])
        path, query = owner_url.split("?")
        token = query.split("&")[0]                               # "t=..."
        done = self.client.post(f"{path}/reunited?{token}", data={"other_id": str(found_id)})
        self.assertIn("היא בבית", done.text)
        self.assertEqual(self.svc.store.get(found_id).status, "reunited")

    def test_private_link_needs_token(self):
        r = self.post_report("lost", contact_email="o@example.com")
        rid = r.headers["location"].split("/")[2].split("?")[0]
        self.assertEqual(self.client.get(f"/matches/{rid}").status_code, 404)
        self.assertEqual(self.client.get(f"/matches/{rid}?t=wrong").status_code, 404)
        self.assertEqual(self.client.post(f"/matches/{rid}/reunited?t=wrong").status_code, 404)

    def test_error_in_hebrew_and_no_orphaned_photos(self):
        r = self.post_report("lost")  # no phone, no email
        self.assertEqual(r.status_code, 400)
        self.assertIn("צריך טלפון או אימייל", r.text)
        self.assertIn("התמונות לא נשמרו", r.text)
        self.assertEqual(list(self.uploads.glob("*.jpg")), [])

    def test_not_an_image(self):
        r = self.client.post("/report/lost", data={"event_date": TODAY.isoformat(), "contact_phone": "050"},
                             files=[("photos", ("x.jpg", b"not an image", "image/jpeg"))])
        self.assertEqual(r.status_code, 400)
        self.assertIn("אינו תמונה", r.text)

    def test_gps_metadata_is_stripped(self):
        self.assertIn(0x8825, Image.open(io.BytesIO(jpeg((20, 20, 20), gps=True))).getexif())
        self.post_report("lost", contact_email="o@example.com")
        saved = list(self.uploads.glob("*.jpg"))
        self.assertEqual(len(saved), 1)
        r = self.client.post("/report/lost", data={"event_date": TODAY.isoformat(), "contact_phone": "050"},
                             files=[("photos", ("g.jpg", jpeg((20, 20, 20), gps=True), "image/jpeg"))],
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        newest = max(self.uploads.glob("*.jpg"), key=lambda p: p.stat().st_mtime_ns)
        with Image.open(newest) as im:
            self.assertNotIn(0x8825, im.getexif())

    def test_screenshot_import_prefills_and_keeps_nothing(self):
        r = self.client.post("/import", files=[("screenshot", ("s.jpg", jpeg((240, 240, 240)), "image/jpeg"))])
        self.assertEqual(r.status_code, 200)
        self.assertIn("מילאנו את הפרטים מצילום המסך", r.text)
        self.assertIn('value="0585554552"', r.text)             # phone repaired and prefilled
        self.assertIn("רמת גן, ליד הפארק", r.text)
        self.assertEqual(list((self.uploads / "screenshots").glob("*")), [])
        self.assertEqual(self.svc.store.open_reports("found"), [])


class TestApi(AppTest):
    def test_api_create_and_matches(self):
        def create(side, **f):
            return self.client.post("/api/reports", data={"side": side, "event_date": TODAY.isoformat(),
                                                          "lat": "32.07", "lon": "34.82", **f},
                                    files=[("photos", ("d.jpg", jpeg((20, 20, 20)), "image/jpeg"))]).json()
        lost = create("lost", contact_email="o@example.com")
        found = create("found", contact_phone="050")
        self.assertEqual(found["matches"][0]["id"], lost["id"])
        self.assertNotIn("contact_email", found["matches"][0])
        ok = self.client.get(f"/api/reports/{lost['id']}/matches", params={"token": lost["token"]})
        self.assertEqual(ok.json()["matches"][0]["id"], found["id"])
        bad = self.client.get(f"/api/reports/{lost['id']}/matches", params={"token": "nope"})
        self.assertEqual(bad.status_code, 404)
        err = self.client.post("/api/reports", data={"side": "lost", "event_date": TODAY.isoformat(),
                                                     "colors": "black"})
        self.assertEqual(err.json()["error"], "contact")


if __name__ == "__main__":
    unittest.main()
