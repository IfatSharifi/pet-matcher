"""Web app: Hebrew pages + JSON API on top of petmatch.service.

Run locally:
    uvicorn app:app --reload
Then open http://localhost:8000

Environment:
    ANTHROPIC_API_KEY   describe photos and read screenshots (optional; without it, people pick colors)
    RESEND_API_KEY      send real alert emails (optional; without it, emails print to the console)
    PETMATCH_DB         SQLite file (default petmatch.db)
    PETMATCH_UPLOADS    photo folder (default uploads)
    PETMATCH_EMBEDDER   dinov2-onnx (default, no PyTorch), dinov2 (PyTorch), or histogram (tests)
    PETMATCH_BASE_URL   used in email links (default: Render's RENDER_EXTERNAL_URL, else localhost)
    PETMATCH_DEMO       1 = show the "demo site" banner
    PETMATCH_CLAUDE_PER_HOUR / PETMATCH_CLAUDE_PER_DAY   Claude call limits (default 20 per visitor, 300 per day)
"""
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from petmatch.ranking import HE_COLORS
from petmatch.schemas import COLORS
from petmatch.service import ALERT_THRESHOLD, MatchService, ValidationError

ROOT = Path(__file__).parent
MAX_PHOTOS = 5
MAX_BYTES = 10 * 1024 * 1024

SWATCHES = {"black": "#1d1d1d", "white": "#ffffff", "brown": "#6b4226", "tan": "#c69c6d", "golden": "#d9a441",
            "cream": "#efe2c4", "gray": "#8d939b", "orange": "#e07b2a", "red": "#9c3b22",
            "brindle": "repeating-linear-gradient(45deg,#6b4226 0 4px,#2a1a10 4px 7px)",
            "merle": "radial-gradient(circle at 30% 30%,#2b2b2b 0 20%,transparent 21%),#8d939b",
            "spotted": "radial-gradient(circle at 30% 40%,#1d1d1d 0 18%,transparent 19%),#ffffff"}
SPECIES_HE = {"dog": "כלב", "cat": "חתול", "other": "אחר"}

ERRORS_HE = {
    "contact": "צריך טלפון או אימייל, כדי שהצד השני יוכל ליצור קשר.",
    "future_date": "התאריך לא יכול להיות בעתיד.",
    "need_color": "בלי תמונה, צריך לבחור לפחות צבע אחד כדי למצוא התאמות.",
    "colors": "אחד הצבעים לא מוכר. בחרו מהרשימה.",
    "species": "בחרו כלב, חתול או אחר.",
    "side": "סוג הדיווח לא תקין.",
    "bad_image": "אחד הקבצים אינו תמונה. אפשר להעלות JPG, PNG או WebP.",
    "too_big": "תמונה גדולה מ-10MB. נסו תמונה קטנה יותר.",
    "too_many": f"אפשר להעלות עד {MAX_PHOTOS} תמונות.",
    "bad_date": "התאריך לא תקין.",
    "needs_claude": "קריאת צילומי מסך לא פעילה כרגע. מלאו את הפרטים ידנית.",
    "claude_failed": "לא הצלחנו לקרוא את צילום המסך. נסו שוב, או מלאו את הפרטים ידנית.",
    "rate_limited": "קראנו הרבה צילומי מסך מהמכשיר הזה בשעה האחרונה. נסו שוב מאוחר יותר, או מלאו את הפרטים ידנית.",
    "no_report": "הדיווח לא נמצא.",
}


class RateLimiter:
    """Caps Claude calls per visitor per hour and for the whole site per day.

    A public demo spends the owner's API credit on every photo and screenshot;
    this keeps one visitor (or a bot) from running up the bill. In memory: resets on restart.
    """
    def __init__(self, per_ip_per_hour: int, per_day: int) -> None:
        self.per_ip, self.per_day = per_ip_per_hour, per_day
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.today_count = date.today(), 0
        self.lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if date.today() != self.day:
                self.day, self.today_count = date.today(), 0
            q = self.hits[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_ip or self.today_count >= self.per_day:
                return False
            q.append(now)
            self.today_count += 1
            return True


def save_image(upload: UploadFile, folder: Path) -> Path:
    """Validate, fix rotation, shrink, and re-encode as JPEG. Re-encoding drops EXIF,
    including the GPS position phones embed in photos - it never reaches our disk."""
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValidationError("too_big", "image over 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600))
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / f"{uuid.uuid4().hex}.jpg"
            im.save(out, "JPEG", quality=88)
            return out
    except (UnidentifiedImageError, OSError):
        raise ValidationError("bad_image", "not an image") from None


def parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValidationError("bad_date", "bad date") from None


def opt_float(s: str | None) -> float | None:
    try:
        return float(s) if s not in (None, "") else None
    except ValueError:
        return None


def create_app(service: MatchService, upload_dir: str | Path = "uploads", limiter: RateLimiter | None = None,
               demo: bool = False) -> FastAPI:
    app = FastAPI(title="Pet Matcher", docs_url="/api/docs")
    limiter = limiter or RateLimiter(per_ip_per_hour=10**9, per_day=10**9)

    def ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"
    uploads = Path(upload_dir)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=ROOT / "templates")
    templates.env.globals.update(DEMO=demo, COLORS=COLORS, HE_COLORS=HE_COLORS, SWATCHES=SWATCHES, SPECIES_HE=SPECIES_HE,
                                 ALERT_THRESHOLD=ALERT_THRESHOLD, today=lambda: date.today().isoformat())
    templates.env.filters["he_colors"] = lambda cs: ", ".join(HE_COLORS.get(c, c) for c in cs)

    def page(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx, status_code=status)

    async def save_photos(files: list[UploadFile]) -> list[Path]:
        files = [f for f in files if f and f.filename]
        if len(files) > MAX_PHOTOS:
            raise ValidationError("too_many", "too many photos")
        return [save_image(f, uploads) for f in files]

    # ------------------------------------------------------------------ pages
    @app.get("/health")
    def health():
        return {"ok": True, "reports_open": len(service.store.open_reports("lost")) + len(service.store.open_reports("found"))}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return page(request, "home.html")

    @app.get("/report/{side}", response_class=HTMLResponse)
    def report_form(request: Request, side: str):
        if side not in ("lost", "found"):
            raise HTTPException(404)
        return page(request, "report.html", side=side, form={}, error=None)

    @app.post("/report/{side}", response_class=HTMLResponse)
    async def report_submit(request: Request, side: str, photos: list[UploadFile] = File(default=[])):
        raw = await request.form()
        form, colors = dict(raw), raw.getlist("colors")
        paths: list[Path] = []
        try:
            paths = await save_photos(photos)
            rid = service.create_report(
                side, parse_date(form.get("event_date", "")), photos=paths,
                lat=opt_float(form.get("lat")), lon=opt_float(form.get("lon")),
                area_text=form.get("area_text") or None, species=form.get("species") or None,
                colors=colors or None, pet_name=form.get("pet_name") or None,
                contact_name=form.get("contact_name") or None, contact_phone=form.get("contact_phone") or None,
                contact_email=form.get("contact_email") or None, source=form.get("source") or "form",
                describe=bool(paths) and limiter.allow(ip(request)))
        except ValidationError as e:
            for p in paths:  # the report wasn't saved: don't keep its photos either
                p.unlink(missing_ok=True)
            form["colors"] = colors
            message = ERRORS_HE.get(e.code, str(e))
            if any(f.filename for f in photos):
                message += " התמונות לא נשמרו, צריך לבחור אותן שוב."
            return page(request, "report.html", status=400, side=side, form=form, error=message)
        r = service.store.get(rid)
        return RedirectResponse(f"/matches/{rid}?t={r.access_token}&new=1", status_code=303)

    @app.get("/import", response_class=HTMLResponse)
    def import_form(request: Request):
        return page(request, "import.html", error=None)

    @app.post("/import", response_class=HTMLResponse)
    async def import_submit(request: Request, screenshot: UploadFile = File(...)):
        try:
            if not limiter.allow(ip(request)):
                raise ValidationError("rate_limited", "too many screenshot reads")
            path = save_image(screenshot, uploads / "screenshots")
            try:
                draft = service.import_screenshot(path)
            finally:
                path.unlink(missing_ok=True)  # the screenshot has strangers' details: don't keep it
        except ValidationError as e:
            return page(request, "import.html", status=400, error=ERRORS_HE.get(e.code, str(e)))
        form = {"area_text": draft.get("area") or "", "event_date": draft.get("event_date") or "",
                "species": draft.get("species") if draft.get("species") in SPECIES_HE else "",
                "colors": draft.get("primary_colors", []), "pet_name": draft.get("pet_name") or "",
                "contact_phone": draft.get("contact_phone") or "", "source": "screenshot"}
        return page(request, "report.html", side=draft.get("side") or "found", form=form, error=None,
                    from_screenshot=True, wanted_photo=draft.get("has_pet_photo"))

    @app.get("/matches/{rid}", response_class=HTMLResponse)
    def matches_page(request: Request, rid: int, t: str = "", new: int = 0):
        if not service.store.check_token(rid, t):
            raise HTTPException(404)
        me = service.store.get(rid)
        return page(request, "matches.html", me=me, token=t, new=bool(new), matches=service.matches(rid))

    @app.post("/matches/{rid}/reunited", response_class=HTMLResponse)
    async def reunited(request: Request, rid: int, t: str = ""):
        if not service.store.check_token(rid, t):
            raise HTTPException(404)
        other = str((await request.form()).get("other_id") or "")
        service.mark_reunited(rid, int(other) if other.isdigit() else None)
        return page(request, "reunited.html", me=service.store.get(rid))

    @app.get("/photos/{rid}/{n}")
    def photo(rid: int, n: int):
        path = service.store.photo_path(rid, n)
        if not path or not Path(path).exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------------ JSON API
    @app.post("/api/reports")
    async def api_create(request: Request, side: str = Form(...), event_date: str = Form(...), lat: str = Form(""),
                         lon: str = Form(""), area_text: str = Form(""), species: str = Form(""),
                         colors: str = Form(""), pet_name: str = Form(""), contact_name: str = Form(""),
                         contact_phone: str = Form(""), contact_email: str = Form(""),
                         photos: list[UploadFile] = File(default=[])):
        paths: list[Path] = []
        try:
            paths = await save_photos(photos)
            rid = service.create_report(
                side, parse_date(event_date), photos=paths, lat=opt_float(lat), lon=opt_float(lon),
                area_text=area_text or None, species=species or None,
                colors=[c.strip() for c in colors.split(",") if c.strip()] or None, pet_name=pet_name or None,
                contact_name=contact_name or None, contact_phone=contact_phone or None,
                contact_email=contact_email or None, describe=bool(paths) and limiter.allow(ip(request)))
        except ValidationError as e:
            for p in paths:
                p.unlink(missing_ok=True)
            return JSONResponse({"error": e.code, "detail": str(e)}, status_code=400)
        r = service.store.get(rid)
        return {"id": rid, "token": r.access_token, "report": r.public(),
                "matches": [m.public() for m in service.matches(rid)]}

    @app.get("/api/reports/{rid}/matches")
    def api_matches(rid: int, token: str, k: int = 5):
        if not service.store.check_token(rid, token):
            raise HTTPException(404)
        return {"matches": [m.public() for m in service.matches(rid, k=min(k, 20))]}

    @app.post("/api/reports/{rid}/reunited")
    def api_reunited(rid: int, token: str, other_id: int | None = None):
        if not service.store.check_token(rid, token):
            raise HTTPException(404)
        service.mark_reunited(rid, other_id)
        return {"ok": True}

    @app.post("/api/screenshots")
    async def api_screenshot(request: Request, screenshot: UploadFile = File(...)):
        try:
            if not limiter.allow(ip(request)):
                raise ValidationError("rate_limited", "too many screenshot reads")
            path = save_image(screenshot, uploads / "screenshots")
            try:
                return service.import_screenshot(path)
            finally:
                path.unlink(missing_ok=True)
        except ValidationError as e:
            return JSONResponse({"error": e.code, "detail": str(e)}, status_code=400)

    return app


def build_service_from_env() -> MatchService:
    from petmatch.claude_client import ClaudeClient
    from petmatch.embedders import EMBEDDERS
    from petmatch.notify import ConsoleNotifier, ResendNotifier
    from petmatch.store import Store

    # no response cache on a server: it would keep strangers' screenshot details on disk
    claude = ClaudeClient(use_cache=False) if os.environ.get("ANTHROPIC_API_KEY") else None
    notifier = ResendNotifier() if os.environ.get("RESEND_API_KEY") else ConsoleNotifier()
    embedder = EMBEDDERS[os.environ.get("PETMATCH_EMBEDDER", "dinov2-onnx")]()
    embedder.use_cache = False  # vectors live in the database
    return MatchService(Store(os.environ.get("PETMATCH_DB", "petmatch.db")), embedder, claude, notifier,
                        base_url=os.environ.get("PETMATCH_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
                        or "http://localhost:8000")


if os.environ.get("PETMATCH_NO_AUTOSTART") != "1":
    app = create_app(build_service_from_env(), os.environ.get("PETMATCH_UPLOADS", "uploads"),
                     limiter=RateLimiter(int(os.environ.get("PETMATCH_CLAUDE_PER_HOUR", 20)),
                                         int(os.environ.get("PETMATCH_CLAUDE_PER_DAY", 300))),
                     demo=os.environ.get("PETMATCH_DEMO") == "1")
