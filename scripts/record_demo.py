"""Record a captioned demo video of the app (phone-sized, ~75 seconds).

    pip install playwright && playwright install chromium     # one time
    uvicorn app:app                                            # app running on the demo database
    python scripts/record_demo.py --db demo.db                 # -> docs/demo.mp4 (needs ffmpeg)

Frames come from Chrome's own screencast at full phone resolution (780x1688), which is
sharper than Playwright's built-in recorder.

The walkthrough: home -> a finder reports a found dog -> ranked matches with Hebrew reasons ->
the owner's private page -> a group-post screenshot turned into a filled report -> results.
Uses the dog on line --line of <db>.found.txt and eval/screenshots/post_006.jpg.
"""
from __future__ import annotations

import argparse
import base64
import shutil
import sqlite3
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

CAPTION_JS = """
(text) => {
  let bar = document.getElementById('__cap');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = '__cap';
    bar.dir = 'ltr';
    bar.style.cssText = 'position:fixed;left:10px;right:10px;bottom:14px;z-index:99999;padding:12px 14px;' +
      'background:rgba(27,42,58,.94);color:#fff;font:600 16px/1.35 Arial,sans-serif;border-radius:4px;' +
      'box-shadow:0 6px 18px rgba(0,0,0,.25);transition:opacity .25s';
    document.body.appendChild(bar);
  }
  bar.style.opacity = text ? 1 : 0;
  bar.textContent = text || '';
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--db", default="demo.db")
    ap.add_argument("--line", type=int, default=6, help="which dog in <db>.found.txt (1-based)")
    ap.add_argument("--screenshot", default="eval/screenshots/post_006.jpg")
    ap.add_argument("--out", default="docs/demo.mp4")
    ap.add_argument("--hide-map", action="store_true", help="for machines that can't load map tiles")
    ap.add_argument("--chromium", default=None, help="path to a Chromium binary (optional)")
    args = ap.parse_args()

    photo, city, latlon = Path(args.db + ".found.txt").read_text(encoding="utf-8").splitlines()[args.line - 1].split("\t")
    lat, lon = (float(x) for x in latlon.split(","))
    dog = Path(photo).parent.name
    db = sqlite3.connect(args.db)
    row = db.execute("SELECT r.id, r.access_token, r.pet_name FROM reports r JOIN photos p ON p.report_id = r.id "
                     "WHERE r.side = 'lost' AND p.path LIKE ? LIMIT 1", (f"%/{dog}/%",)).fetchone()
    if row is None:
        raise SystemExit(f"{dog} is not in {args.db}; reseed with scripts/seed_demo.py")
    owner_id, owner_token, pet_name = row
    base = args.base_url.rstrip("/")
    out = Path(args.out)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed to assemble the video: https://ffmpeg.org/download.html")

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, locale="he-IL")
        if args.hide_map:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded', () => {"
                                "const s = document.createElement('style');"
                                "s.textContent = '#map, #locate {display:none !important}';"
                                "document.head.appendChild(s); });")
        page = ctx.new_page()
        frames: list[tuple[float, bytes]] = []
        cdp = ctx.new_cdp_session(page)

        def on_frame(params: dict) -> None:
            frames.append((params["metadata"]["timestamp"], base64.b64decode(params["data"])))
            cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 780, "maxHeight": 1688})

        def cap(text: str, hold: float = 0) -> None:
            page.evaluate(CAPTION_JS, text)
            if hold:
                page.wait_for_timeout(int(hold * 1000))

        def scroll_to(selector: str, hold: float = 1.2) -> None:
            page.locator(selector).first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy({top: -60})")
            page.wait_for_timeout(int(hold * 1000))

        def glide(px: int, steps: int = 12, pause_ms: int = 110) -> None:
            for _ in range(steps):
                page.mouse.wheel(0, px / steps)
                page.wait_for_timeout(pause_ms)

        # 1. home
        page.goto(base + "/")
        cap("A Hebrew lost & found board for pets. AI matches found pets to lost reports by face, place and date.", 4.5)

        # 2. a finder reports the dog
        cap("A neighbor finds a dog and reports it.", 1.5)
        page.locator("a.flyer.found").click()
        page.wait_for_load_state("networkidle")
        cap("They add a photo...", 1.0)
        page.set_input_files("#photos", photo)
        page.wait_for_timeout(1200)
        pin = (lat - 0.003, lon + 0.004)  # about 500 m from where the dog was lost
        if args.hide_map:
            page.evaluate(f"document.getElementById('lat').value = '{pin[0]:.5f}';"
                          f"document.getElementById('lon').value = '{pin[1]:.5f}';")
        else:
            scroll_to("#map", 0.4)
            cap("...pin where they found it...", 0.8)
            page.evaluate(f"window.setPin({pin[0]:.5f}, {pin[1]:.5f})")
            page.wait_for_timeout(2200)  # let the tiles load
        scroll_to("#area_text", 0.6)
        cap("...the area..." if not args.hide_map else "...where they found it...", 0.3)
        page.locator("#area_text").type(city, delay=90)
        page.wait_for_timeout(700)
        scroll_to("#contact_phone", 0.5)
        cap("...and a phone number. No colors or breed to fill in: Claude reads the photo.", 0.3)
        page.locator("#contact_phone").type("050-555-0147", delay=70)
        page.wait_for_timeout(1500)
        page.locator("button[type=submit]").click()
        page.wait_for_load_state("networkidle")

        # 3. ranked matches
        page.evaluate("window.scrollTo(0, 0)")
        cap("Every open lost report is ranked by face similarity (DINOv2), distance and date.", 4.0)
        scroll_to(".match", 0.5)
        cap("Each match explains itself in Hebrew: matching colors, distance, and days since the pet went missing.", 4.5)
        glide(1400, steps=20, pause_ms=120)
        cap("", 0.6)

        # 4. the owner's side
        page.goto(f"{base}/matches/{owner_id}?t={owner_token}")
        cap("The owner sees the match on their private page. Strong matches also trigger an email.", 3.5)
        scroll_to(".tear", 0.4)
        cap("Owners see the finder's phone. Finders never see the owner's, which blocks 'pay me to return your dog' scams.", 5.0)

        # 5. screenshot import
        page.goto(base + "/import")
        cap("Most lost-pet posts in Israel live in Facebook and WhatsApp groups.", 3.0)
        cap("Upload a screenshot of a post...", 0.8)
        page.set_input_files("input[name=screenshot]", args.screenshot)
        page.wait_for_timeout(600)
        page.locator("button[type=submit]").click()
        page.wait_for_load_state("networkidle")
        cap("...and Claude fills in the report for you to check.", 3.0)
        scroll_to("#area_text", 0.3)
        cap("Area and date ('this evening') are read from the Hebrew post.", 3.5)
        scroll_to("#contact_phone", 0.3)
        cap("Right-to-left text displays '058 555 4552' as '4552 555 058'. The app spots that and fixes it.", 5.0)

        # 6. results
        page.goto(base + "/")
        cap("Measured on 1,393 dogs: the right dog is ranked first 86% of the time from the photo alone, "
            "and 96-98% with place and date (simulated).", 5.5)
        cap("Hebrew screenshots: 92.5% read with every field correct, for $0.0005 each.", 4.5)
        cap("", 0.5)

        cdp.send("Page.stopScreencast")
        ctx.close()
        browser.close()
    write_video(frames, out)


def write_video(frames: list[tuple[float, bytes]], out: Path) -> None:
    """Chrome sends a frame only when the screen changes; hold each one until the next."""
    if len(frames) < 2:
        raise SystemExit("no frames captured")
    tmp = out.parent / "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    lines = []
    for i, (ts, jpg) in enumerate(frames):
        f = tmp / f"{i:05d}.jpg"
        f.write_bytes(jpg)
        dur = frames[i + 1][0] - ts if i + 1 < len(frames) else 1.0
        lines += [f"file '{f.resolve().as_posix()}'", f"duration {max(dur, 0.001):.4f}"]
    lines.append(f"file '{(tmp / f'{len(frames) - 1:05d}.jpg').resolve().as_posix()}'")  # concat needs the last twice
    (tmp / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-vf", "fps=30,scale=780:1688:force_original_aspect_ratio=decrease,"
                           "pad=780:1688:(ow-iw)/2:(oh-ih)/2:color=white",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
