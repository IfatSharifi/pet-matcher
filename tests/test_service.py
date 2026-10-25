"""Service tests: store + matching + alerts, with a fake Claude and the no-download embedder."""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

import petmatch.embedders as E
from petmatch.embedders import ColorHistogramEmbedder
from petmatch.notify import ConsoleNotifier
from petmatch.service import MatchService, ValidationError, geocode
from petmatch.store import Store

TODAY = date.today()
RAMAT_GAN = (32.068, 34.824)
NEAR = (32.072, 34.830)        # ~0.7 km away
HAIFA = (32.794, 34.990)


class FakeClaude:
    """Stands in for ClaudeClient.extract: answers per photo color, or a fixed screenshot draft."""
    def __init__(self):
        self.calls = 0

    def extract(self, images, prompt, schema, system="", max_edge=768, max_tokens=1024):
        self.calls += 1
        if "report_type" in schema["properties"]:
            return {"report_type": "found", "species": "dog", "pet_name": None, "primary_colors": ["black"],
                    "area": "רמת גן, ליד הפארק", "event_date": TODAY.isoformat(),
                    "contact_phone": "4552555058", "description_he": "כלב שחור", "has_pet_photo": True}
        with Image.open(images[0]) as im:
            r, g, b = im.convert("RGB").getpixel((0, 0))
        color = "black" if max(r, g, b) < 80 else "tan"
        return {"species": "dog", "breed_guess": None, "size": "medium", "primary_colors": [color],
                "distinctive_markings": [], "collar": "no", "collar_color": None, "photo_quality": "good"}


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.old_cache = E.CACHE_DIR
        E.CACHE_DIR = self.dir / "cache"
        self.mail = ConsoleNotifier(quiet=True)
        self.claude = FakeClaude()
        self.svc = MatchService(Store(self.dir / "t.db"), ColorHistogramEmbedder(), self.claude, self.mail)
        self.n = 0

    def tearDown(self):
        self.svc.store.db.close()
        E.CACHE_DIR = self.old_cache
        self.tmp.cleanup()

    def photo(self, rgb, noise=10):
        self.n += 1
        rng = np.random.default_rng(self.n)
        arr = np.clip(np.array(rgb) + rng.integers(-noise, noise + 1, (64, 64, 3)), 0, 255).astype(np.uint8)
        p = self.dir / f"p{self.n}.png"
        Image.fromarray(arr).save(p)
        return p

    def lost(self, rgb=(20, 20, 20), at=RAMAT_GAN, days_ago=3, **kw):
        return self.svc.create_report("lost", TODAY - timedelta(days=days_ago), photos=[self.photo(rgb)],
                                      lat=at[0], lon=at[1], contact_email="owner@example.com",
                                      pet_name="שוקו", **kw)

    def found(self, rgb=(20, 20, 20), at=NEAR, days_ago=1, **kw):
        return self.svc.create_report("found", TODAY - timedelta(days=days_ago), photos=[self.photo(rgb)],
                                      lat=at[0], lon=at[1], contact_phone="0505551234",
                                      contact_email="finder@example.com", **kw)


class TestFlow(ServiceTest):
    def test_found_report_alerts_lost_owner_with_finder_contact(self):
        lid = self.lost()
        self.lost(rgb=(200, 160, 110), at=HAIFA)          # a tan dog far away
        fid = self.found()
        top = self.svc.matches(fid)[0]
        self.assertEqual(top.report.id, lid)
        self.assertTrue(top.used_photos)
        self.assertIn("אותו צבע: שחור", top.reason)
        self.assertEqual(len(self.mail.sent), 1)
        email = self.mail.sent[0]
        self.assertEqual(email["to"], "owner@example.com")       # only the owner is emailed
        self.assertIn("0505551234", email["html"])               # ...with the finder's phone
        self.assertIn('dir="rtl"', email["html"])
        self.assertIn("שוקו", email["subject"])

    def test_no_duplicate_alert(self):
        lid = self.lost()
        fid = self.found()
        self.svc._alert_new_matches(fid)
        self.svc._alert_new_matches(lid)
        self.assertEqual(len(self.mail.sent), 1)

    def test_public_view_hides_contact_and_blurs_place(self):
        lid = self.lost()
        fid = self.found()
        pub = self.svc.matches(fid)[0].public()
        self.assertEqual(pub["id"], lid)
        self.assertNotIn("contact_email", pub)
        self.assertNotIn("contact_phone", pub)
        self.assertEqual(pub["approx_lat"], round(RAMAT_GAN[0], 2))

    def test_found_before_lost_and_dog_vs_cat_are_excluded(self):
        self.svc.create_report("lost", TODAY, photos=[self.photo((20, 20, 20))], lat=RAMAT_GAN[0],
                               lon=RAMAT_GAN[1], contact_email="o@x.com")
        cat = self.svc.create_report("lost", TODAY - timedelta(days=5), photos=[self.photo((20, 20, 20))],
                                     lat=RAMAT_GAN[0], lon=RAMAT_GAN[1], contact_email="c@x.com", species="cat")
        fid = self.found(days_ago=3)   # found 3 days before the first dog was lost
        ids = [m.report.id for m in self.svc.matches(fid)]
        self.assertEqual(ids, [])
        self.assertIsNotNone(self.svc.store.get(cat))

    def test_reunited_reports_leave_the_pool(self):
        lid = self.lost()
        fid = self.found()
        self.svc.mark_reunited(lid, fid)
        later = self.found(days_ago=0)
        self.assertEqual(self.svc.matches(later), [])

    def test_photo_less_report_matches_by_color_but_never_emails(self):
        self.lost()
        self.lost(rgb=(200, 160, 110))
        fid = self.svc.create_report("found", TODAY, colors=["black"], area_text="רמת גן, רחוב ביאליק",
                                     contact_phone="0505550000")
        top = self.svc.matches(fid)
        self.assertEqual(top[0].report.colors, ["black"])
        self.assertFalse(top[0].used_photos)
        self.assertEqual(self.mail.sent, [])
        self.assertEqual(self.svc.store.get(fid).place_precision, "city")

    def test_unknown_place_still_matches(self):
        lid = self.lost()
        fid = self.svc.create_report("found", TODAY, photos=[self.photo((20, 20, 20))], contact_phone="050")
        m = self.svc.matches(fid)[0]
        self.assertEqual(m.report.id, lid)
        self.assertIn("מיקום לא ידוע", m.reason)

    def test_claude_called_once_per_report(self):
        self.lost()
        self.assertEqual(self.claude.calls, 1)
        self.svc.create_report("lost", TODAY, photos=[self.photo((20, 20, 20))], contact_phone="050",
                               description={"species": "dog", "primary_colors": ["black"]})
        self.assertEqual(self.claude.calls, 1)  # caller supplied a description: no new call


class TestScreenshotAndValidation(ServiceTest):
    def test_screenshot_draft(self):
        shot = self.photo((240, 240, 240))
        draft = self.svc.import_screenshot(shot)
        self.assertEqual(draft["side"], "found")
        self.assertEqual(draft["contact_phone"], "0585554552")   # RTL-reversed groups repaired
        self.assertEqual(draft["geocoded"]["city"], "רמת גן")
        self.assertEqual(self.svc.store.open_reports("found"), [])  # nothing saved until confirmed

    def test_validation(self):
        with self.assertRaises(ValidationError):
            self.svc.create_report("lost", TODAY, colors=["black"])                       # no contact
        with self.assertRaises(ValidationError):
            self.svc.create_report("lost", TODAY + timedelta(days=1), colors=["black"], contact_phone="050")
        with self.assertRaises(ValidationError):
            self.svc.create_report("lost", TODAY, colors=["purple"], contact_phone="050")
        with self.assertRaises(ValidationError):
            self.svc.create_report("found", TODAY, contact_phone="050")                   # no photo, no color
        with self.assertRaises(ValidationError):
            self.svc.create_report("adopted", TODAY, colors=["black"], contact_phone="050")

    def test_geocode(self):
        self.assertEqual(geocode("באזור רמת גן, ליד הפארק")[2], "רמת גן")
        self.assertEqual(geocode("ראש העין, נווה אפק")[2], "ראש העין")
        self.assertIsNone(geocode("איפשהו בצפון"))
        self.assertIsNone(geocode(None))


if __name__ == "__main__":
    unittest.main()
