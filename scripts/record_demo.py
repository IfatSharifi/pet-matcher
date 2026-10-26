"""Record the narrated demo video: Hebrew captions, Hebrew voiceover, phone format (780x1688).

    pip install playwright && playwright install chromium        # one time; also needs ffmpeg
    python scripts/make_voiceover.py                              # Hebrew voice -> build/voice/
    uvicorn app:app                                               # app running on the demo database
    python scripts/record_demo.py --db demo.db --voice build/voice/manifest.json   # -> docs/demo.mp4

Without --voice the video is silent and each caption stays up for its reading time.

The walkthrough: home -> a finder reports a found dog -> ranked matches with Hebrew reasons ->
the owner's private page -> a group-post screenshot turned into a filled report -> results.
Captions and voice come from docs/demo_lines_he.json; each line starts when its caption appears,
the on-screen actions run while it is spoken, and the next line waits for it to finish.
Frames come from Chrome's own screencast at full phone resolution.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CAPTION_JS = """
(text) => {
  let bar = document.getElementById('__cap');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = '__cap';
    bar.dir = 'rtl';
    bar.lang = 'he';
    bar.style.cssText = 'position:fixed;left:10px;right:10px;bottom:14px;z-index:99999;padding:12px 16px;' +
      'background:rgba(27,42,58,.95);color:#fff;font:600 17px/1.45 Assistant,Arial,sans-serif;' +
      'text-align:right;border-radius:4px;box-shadow:0 6px 18px rgba(0,0,0,.25)';
    document.body.appendChild(bar);
  }
  bar.style.display = text ? 'block' : 'none';
  bar.textContent = text || '';
}
"""
READ_CHARS_PER_S = 14  # caption hold without a voiceover
PAUSE_S = 0.35         # breath between lines


class Narrator:
    """Shows a line's caption, logs when it started, and makes the next line wait for it."""

    def __init__(self, page, lines: list[dict]) -> None:
        self.page, self.lines = page, {l["id"]: l for l in lines}
        self.text, self.busy_until = "", 0.0
        self.log: list[tuple[str, float]] = []

    def say(self, key: str) -> None:
        self.wait()
        line = self.lines[key]
        self.text = line["show"]
        self.refresh()
        now = time.time()
        self.log.append((key, now))
        seconds = line.get("seconds") or len(line["show"]) / READ_CHARS_PER_S + 0.8
        self.busy_until = now + seconds + PAUSE_S

    def wait(self) -> None:
        """Let the current line finish (keeps Playwright's event loop, and the screencast, running)."""
        left = self.busy_until - time.time()
        if left > 0:
            self.page.wait_for_timeout(int(left * 1000))

    def refresh(self) -> None:
        """Re-draw the caption after a page load wiped it."""
        self.page.evaluate(CAPTION_JS, self.text)

    def clear(self, hold: float = 0.6) -> None:
        self.wait()
        self.text = ""
        self.refresh()
        self.page.wait_for_timeout(int(hold * 1000))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--db", default="demo.db")
    ap.add_argument("--line", type=int, default=6, help="which dog in <db>.found.txt (1-based)")
    ap.add_argument("--screenshot", default="eval/screenshots/post_006.jpg")
    ap.add_argument("--lines", default="docs/demo_lines_he.json")
    ap.add_argument("--voice", default=None, help="manifest.json from make_voiceover.py")
    ap.add_argument("--out", default="docs/demo.mp4")
    ap.add_argument("--hide-map", action="store_true", help="for machines that can't load map tiles")
    ap.add_argument("--chromium", default=None, help="path to a Chromium binary (optional)")
    args = ap.parse_args()

    photo, city, latlon = Path(args.db + ".found.txt").read_text(encoding="utf-8").splitlines()[args.line - 1].split("\t")
    lat, lon = (float(x) for x in latlon.split(","))
    dog = Path(photo).parent.name
    db = sqlite3.connect(args.db)
    row = db.execute("SELECT r.id, r.access_token FROM reports r JOIN photos p ON p.report_id = r.id "
                     "WHERE r.side = 'lost' AND p.path LIKE ? LIMIT 1", (f"%/{dog}/%",)).fetchone()
    if row is None:
        raise SystemExit(f"{dog} is not in {args.db}; reseed with scripts/seed_demo.py")
    owner_id, owner_token = row
    base = args.base_url.rstrip("/")
    out = Path(args.out)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed to assemble the video: https://ffmpeg.org/download.html")

    if args.voice:
        voice_dir = Path(args.voice).parent
        lines = json.loads(Path(args.voice).read_text(encoding="utf-8"))
    else:
        voice_dir, lines = None, json.loads(Path(args.lines).read_text(encoding="utf-8"))

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
        clock = {}  # wall-clock time minus screencast time, from the first frame
        cdp = ctx.new_cdp_session(page)

        def on_frame(params: dict) -> None:
            ts = params["metadata"]["timestamp"]
            clock.setdefault("offset", time.time() - ts)
            frames.append((ts, base64.b64decode(params["data"])))
            cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 780, "maxHeight": 1688})
        n = Narrator(page, lines)

        def scroll_to(selector: str, hold: float = 0.5) -> None:
            page.locator(selector).first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy({top: -60})")
            page.wait_for_timeout(int(hold * 1000))

        def glide(px: int, steps: int = 20, pause_ms: int = 120) -> None:
            for _ in range(steps):
                page.mouse.wheel(0, px / steps)
                page.wait_for_timeout(pause_ms)

        def load(url: str) -> None:
            page.goto(url)
            page.wait_for_load_state("networkidle")
            n.refresh()

        # 1. home
        load(base + "/")
        page.wait_for_timeout(600)
        n.say("intro")

        # 2. a finder reports the dog
        n.say("photo")
        page.locator("a.flyer.found").click()
        page.wait_for_load_state("networkidle")
        n.refresh()
        page.wait_for_timeout(500)
        page.set_input_files("#photos", photo)
        pin = (lat - 0.003, lon + 0.004)  # about 500 m from where the dog was lost
        n.say("pin")
        if args.hide_map:
            page.evaluate(f"document.getElementById('lat').value = '{pin[0]:.5f}';"
                          f"document.getElementById('lon').value = '{pin[1]:.5f}';")
        else:
            scroll_to("#map", 0.4)
            page.evaluate(f"window.setPin({pin[0]:.5f}, {pin[1]:.5f})")
            page.wait_for_timeout(1500)  # let the tiles load
        scroll_to("#area_text", 0.3)
        page.locator("#area_text").type(city, delay=80)
        n.say("phone")
        scroll_to("#contact_phone", 0.3)
        page.locator("#contact_phone").type("050-555-0147", delay=70)
        n.wait()
        page.locator("button[type=submit]").click()
        page.wait_for_load_state("networkidle")
        n.text = ""
        n.refresh()

        # 3. ranked matches
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(400)
        n.say("ranked")
        page.wait_for_timeout(1800)
        scroll_to(".match", 0.3)
        n.say("reasons")
        page.wait_for_timeout(1500)
        glide(1300)

        # 4. the owner's side
        n.wait()
        load(f"{base}/matches/{owner_id}?t={owner_token}")
        n.say("owner")
        n.say("privacy")
        scroll_to(".tear", 0.3)

        # 5. screenshot import
        n.wait()
        load(base + "/import")
        n.say("import")
        page.set_input_files("input[name=screenshot]", args.screenshot)
        n.wait()
        page.locator("button[type=submit]").click()
        page.wait_for_load_state("networkidle")
        n.text = ""
        n.refresh()
        page.wait_for_timeout(300)
        n.say("filled")
        scroll_to("#area_text", 0.3)
        n.say("rtl")
        scroll_to("#contact_phone", 0.3)

        # 6. results
        n.wait()
        load(base + "/")
        n.say("results1")
        n.say("results2")
        n.say("results3")
        n.clear(0.8)

        cdp.send("Page.stopScreencast")
        ctx.close()
        browser.close()

    silent = out.with_name(out.stem + ".silent.mp4") if voice_dir else out
    write_video(frames, silent)
    if voice_dir:
        start = frames[0][0] + clock["offset"]
        cues = [(key, t - start) for key, t in n.log]
        add_voice(silent, cues, {l["id"]: voice_dir / l["wav"] for l in lines}, out)
        silent.unlink()
        (out.with_suffix(".cues.json")).write_text(
            json.dumps([{"id": k, "start_s": round(s, 2)} for k, s in cues], indent=1), encoding="utf-8")


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


def add_voice(video: Path, cues: list[tuple[str, float]], wavs: dict[str, Path], out: Path) -> None:
    """Place each line's audio at the moment its caption appeared, then mux with the video."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    parts = []
    for i, (key, at) in enumerate(cues, 1):
        cmd += ["-i", str(wavs[key])]
        ms = max(int(at * 1000), 0)
        parts.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(cues) + 1))
    parts.append(f"{mix}amix=inputs={len(cues)}:normalize=0:dropout_transition=0,"
                 "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    print(f"Saved {out} with {len(cues)} voiced lines")


if __name__ == "__main__":
    main()
