"""Extraction tests - no API key, no network. Run: python -m unittest discover tests"""
import base64
import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from PIL import Image

import petmatch.claude_client as cc
from petmatch.claude_client import ClaudeClient, ClaudeError
from petmatch.extract import read_screenshot
from petmatch.schemas import PET_DESCRIPTION_SCHEMA, SCREENSHOT_SCHEMA, check_schema, validate
from petmatch.scoring import consistency, digits, field_accuracy, score_screenshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_screenshots  # noqa: E402

TRUTH = {"report_type": "lost", "species": "dog", "pet_name": "לוסי", "colors": ["black", "white"],
         "city": "רמת גן", "area": "רמת גן, ליד הפארק הלאומי", "event_date": "2026-10-11",
         "contact_phone": "0505551234", "has_pet_photo": False}
GOOD = {"report_type": "lost", "species": "dog", "pet_name": "לוסי", "primary_colors": ["white", "black"],
        "area": "רמת גן, ליד הפארק", "event_date": "2026-10-11", "contact_phone": "050-555-1234",
        "description_he": "כלבה שחורה לבנה", "has_pet_photo": False}


class TestPhoneFix(unittest.TestCase):
    def test_reversed_groups_are_restored(self):
        from petmatch.extract import fix_phone
        self.assertEqual(fix_phone("4552555058"), "0585554552")   # seen in the Haiku v1 run
        self.assertEqual(fix_phone("4552 555 058"), "0585554552")
        self.assertEqual(fix_phone("058-555-4552"), "0585554552")  # already right: untouched
        self.assertEqual(fix_phone("+972 58-555-4552"), "0585554552")
        self.assertEqual(fix_phone("1234567890"), "1234567890")    # not a reversed mobile: untouched
        self.assertIsNone(fix_phone(None))


class TestSchemas(unittest.TestCase):
    def test_schemas_follow_api_rules(self):
        self.assertEqual(check_schema(PET_DESCRIPTION_SCHEMA), [])
        self.assertEqual(check_schema(SCREENSHOT_SCHEMA), [])

    def test_checker_catches_problems(self):
        bad = {"type": "object", "properties": {"x": {"type": "string", "maxLength": 3}}}
        self.assertEqual(len(check_schema(bad)), 2)  # no additionalProperties + maxLength

    def test_validate(self):
        self.assertEqual(validate(GOOD, SCREENSHOT_SCHEMA), [])
        bad = {**GOOD, "species": "ferret"}
        del bad["area"]
        self.assertEqual(len(validate(bad, SCREENSHOT_SCHEMA)), 2)


class TestScoring(unittest.TestCase):
    def test_perfect_prediction(self):
        s = score_screenshot(TRUTH, GOOD)
        self.assertTrue(all(s.values()), s)

    def test_phone_formats(self):
        for shown in ["050-555-1234", "050 555 1234", "+972 50-555-1234", "0505551234"]:
            self.assertEqual(digits(shown), "0505551234", shown)

    def test_wrong_fields_and_unscored_colors(self):
        pred = {**GOOD, "report_type": "found", "area": "תל אביב", "primary_colors": ["black"]}
        s = score_screenshot(TRUTH, pred)
        self.assertFalse(s["report_type"])
        self.assertFalse(s["area"])
        self.assertFalse(s["colors"])  # missing white
        s2 = score_screenshot({**TRUTH, "colors": None}, pred)
        self.assertIsNone(s2["colors"])

    def test_null_name_must_stay_null(self):
        self.assertFalse(score_screenshot({**TRUTH, "pet_name": None}, GOOD)["pet_name"])

    def test_field_accuracy(self):
        acc = field_accuracy([score_screenshot(TRUTH, GOOD),
                              score_screenshot(TRUTH, {**GOOD, "species": "cat"})])
        self.assertEqual(acc["species"], 0.5)
        self.assertEqual(acc["all_fields"], 0.5)

    def test_consistency(self):
        d = {"species": "dog", "size": "medium", "collar": "yes", "primary_colors": ["black", "white"]}
        self.assertEqual(consistency([d, d, d])["color_jaccard"], 1.0)
        c = consistency([d, {**d, "primary_colors": ["black"]}, {**d, "primary_colors": ["brown"]}])
        self.assertAlmostEqual(c["main_color"], 2 / 3)
        self.assertAlmostEqual(c["color_jaccard"], (0.5 + 0 + 0) / 3)


class FakeAPI:
    """Stands in for the HTTP call; records request bodies."""
    def __init__(self, result, stop_reason="end_turn"):
        self.bodies, self.result, self.stop = [], result, stop_reason

    def __call__(self, body):
        self.bodies.append(body)
        return {"content": [{"type": "text", "text": json.dumps(self.result, ensure_ascii=False)}],
                "stop_reason": self.stop, "usage": {"input_tokens": 1000, "output_tokens": 200}}


class TestClient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cache = cc.CACHE_DIR
        cc.CACHE_DIR = Path(self.tmp.name) / "cache"
        self.img = Path(self.tmp.name) / "big.png"
        Image.new("RGB", (2000, 1000), (90, 60, 30)).save(self.img)

    def tearDown(self):
        cc.CACHE_DIR = self.old_cache
        self.tmp.cleanup()

    def test_request_shape_resize_cache_and_cost(self):
        client = ClaudeClient(model="claude-haiku-5-5", api_key="test")
        client._post = FakeAPI(GOOD)
        out = read_screenshot(client, self.img)
        self.assertEqual(out, {**GOOD, "contact_phone": "0505551234"})  # phone normalized
        body = client._post.bodies[0]
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertIs(body["output_config"]["format"]["schema"], SCREENSHOT_SCHEMA)
        img_block = body["messages"][0]["content"][0]
        self.assertEqual(img_block["source"]["media_type"], "image/jpeg")
        sent = Image.open(io.BytesIO(base64.b64decode(img_block["source"]["data"])))
        self.assertEqual(max(sent.size), 1568)  # downscaled for screenshots, aspect kept
        self.assertEqual(sent.size[1], 784)
        # second identical call: served from cache, no new request, no new cost
        read_screenshot(client, self.img)
        self.assertEqual(len(client._post.bodies), 1)
        self.assertEqual(client.usage.cached_calls, 1)
        self.assertAlmostEqual(client.usage.cost_usd, 1000 / 1e6 * 0.10 + 200 / 1e6 * 0.50)

    def test_truncated_output_is_an_error(self):
        client = ClaudeClient(api_key="test", use_cache=False)
        client._post = FakeAPI(GOOD, stop_reason="max_tokens")
        with self.assertRaises(ClaudeError):
            read_screenshot(client, self.img)

    def test_missing_key(self):
        client = ClaudeClient(api_key=None, use_cache=False)
        client.api_key = None
        with self.assertRaises(ClaudeError):
            client._post({})


class TestEvalRunner(unittest.TestCase):
    def test_end_to_end_with_fake_model(self):
        import eval_extraction as ev
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shots"
            folder.mkdir()
            labels = [{**TRUTH, "id": f"p{i}", "file": f"p{i}.jpg"} for i in range(4)]
            (folder / "labels.jsonl").write_text(
                "\n".join(json.dumps(l, ensure_ascii=False) for l in labels), encoding="utf-8")

            def fake_read(client, path):  # model gets p3's phone wrong
                return {**GOOD, "contact_phone": "000"} if path.name == "p3.jpg" else GOOD

            old_read, old_results = ev.read_screenshot, ev.RESULTS
            ev.read_screenshot, ev.RESULTS = fake_read, Path(tmp) / "results"
            try:
                res = ev.eval_screenshots(ClaudeClient(api_key="x"), folder, workers=2)
            finally:
                ev.read_screenshot, ev.RESULTS = old_read, old_results
            self.assertEqual(res["acc_contact_phone"], 0.75)
            self.assertEqual(res["acc_species"], 1.0)
            err_csv = Path(tmp) / "results/screenshot_errors_claude-haiku-5-5.csv"
            rows = list(csv.reader(err_csv.read_text(encoding="utf-8-sig").splitlines()))
            self.assertEqual(rows[1][:2], ["p3.jpg", "contact_phone"])


class TestScreenshotGenerator(unittest.TestCase):
    def test_answer_key_matches_post_text(self):
        rng = random.Random(1)
        for _ in range(200):
            text, label = make_screenshots.write_post(rng, date(2026, 10, 20))
            if label["contact_phone"]:
                self.assertIn(label["contact_phone"], re.sub(r"\D", "", text))
            else:
                self.assertIn("פרטים בהודעה פרטית", text)
            self.assertIn(label["city"], text)
            if label["pet_name"]:
                self.assertIn(label["pet_name"], text)
            self.assertLessEqual(label["event_date"], "2026-10-20")
            if label["has_pet_photo"]:
                self.assertIsNone(label["colors"])


if __name__ == "__main__":
    unittest.main()
