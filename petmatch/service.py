"""The app's core, independent of any web framework.

    svc = MatchService(Store("petmatch.db"), DinoV2Embedder(), ClaudeClient(), ConsoleNotifier())
    rid = svc.create_report("found", photos=[...], event_date=date.today(), lat=32.07, lon=34.82, ...)
    svc.matches(rid)        # ranked candidates with Hebrew reasons
    svc.import_screenshot("post.jpg")   # draft report from a Facebook/WhatsApp screenshot

Weights and the alert threshold come from run_ranking_eval.py (scenario 'base').
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .claude_client import ClaudeClient, ClaudeError
from .extract import describe_pet, read_screenshot
from .notify import Notifier, match_email
from .ranking import Features, Weights, color_matrix, haversine_matrix, reason_he, score, species_compatible
from .schemas import COLORS, SPECIES
from .simulate import CITIES
from .store import Report, Store

log = logging.getLogger(__name__)

# Tuned in run_ranking_eval.py ('base'): both sides have photos / at least one has none.
PHOTO_WEIGHTS = Weights(image=1.0, color=0.0, dist=0.08, date=0.05, use_rules=True)
NO_PHOTO_WEIGHTS = Weights(image=0.0, color=0.5, dist=0.12, date=0.05, use_rules=True)
# 99% precision needed 0.457-0.650 across scenarios; start at the strict end, recalibrate on real data.
# Photo-less pairs never reach 95% precision, so they never trigger an email.
ALERT_THRESHOLD = 0.65
UNKNOWN_KM = 5.0          # neutral distance when a report has no place
MAX_DESCRIBE_PHOTOS = 3


class ValidationError(ValueError):
    """A request the service refuses. `code` lets a UI show its own wording (e.g. Hebrew)."""
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def geocode(area_text: str | None) -> tuple[float, float, str] | None:
    """City-level lookup from Hebrew text ('רמת גן, ליד הפארק' -> Ramat Gan center)."""
    if not area_text:
        return None
    hits = [c for c in CITIES if c[0] in area_text]
    if not hits:
        return None
    city = max(hits, key=lambda c: len(c[0]))  # longest name wins ('רמת גן' over 'גן')
    return city[1], city[2], city[0]


@dataclass
class Match:
    report: Report
    score: float
    km: float | None
    days: int
    reason: str
    used_photos: bool

    def public(self) -> dict:
        return {**self.report.public(), "score": round(self.score, 3), "reason": self.reason,
                "km": None if self.km is None else round(self.km, 1), "days_apart": self.days,
                "photo_match": self.used_photos}


class MatchService:
    def __init__(self, store: Store, embedder, claude: ClaudeClient | None = None,
                 notifier: Notifier | None = None, base_url: str = "http://localhost:8000") -> None:
        self.store, self.embedder, self.claude, self.notifier = store, embedder, claude, notifier
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------ reports
    def create_report(self, side: str, event_date: date, photos: list[str | Path] = (),
                      lat: float | None = None, lon: float | None = None, area_text: str | None = None,
                      species: str | None = None, colors: list[str] | None = None,
                      pet_name: str | None = None, contact_name: str | None = None,
                      contact_phone: str | None = None, contact_email: str | None = None,
                      source: str = "form", description: dict | None = None, describe: bool = True) -> int:
        """Save a report and alert matching owners. `describe=False` skips the Claude call
        (used when a visitor hits the rate limit); photos still match by image."""
        if side not in ("lost", "found"):
            raise ValidationError("side", "side must be 'lost' or 'found'")
        if not (contact_phone or contact_email):
            raise ValidationError("contact", "a phone number or email is needed so the other side can reach you")
        if event_date > date.today():
            raise ValidationError("future_date", "the date can't be in the future")
        photos = [Path(p) for p in photos]

        desc = dict(description or {})
        if photos and self.claude and describe and not desc:
            try:
                desc = describe_pet(self.claude, photos[:MAX_DESCRIBE_PHOTOS])
            except (ClaudeError, ValueError) as e:  # outage or bad key must not lose the report
                log.warning("describe failed, saving without description: %s", e)
        species = species or desc.get("species") or "unknown"
        colors = colors if colors is not None else desc.get("primary_colors", [])
        if species not in SPECIES:
            raise ValidationError("species", f"species must be one of {SPECIES}")
        bad = [c for c in colors if c not in COLORS]
        if bad:
            raise ValidationError("colors", f"unknown colors {bad}; use {COLORS}")
        if not photos and not colors:
            raise ValidationError("need_color", "without a photo, at least one color is needed to match")

        precision = None
        if lat is not None and lon is not None:
            precision = "exact"
        else:
            g = geocode(area_text)
            if g:
                lat, lon, _ = g
                precision = "city"

        vectors = self.embedder.embed(photos) if photos else []
        report = Report(side=side, species=species, colors=colors, event_date=event_date, description=desc,
                        pet_name=pet_name, lat=lat, lon=lon, place_precision=precision, area_text=area_text,
                        contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email,
                        source=source)
        rid = self.store.add_report(report, [(str(p), v) for p, v in zip(photos, vectors)])
        self._alert_new_matches(rid)
        return rid

    def import_screenshot(self, screenshot: str | Path) -> dict:
        """Read a lost/found post screenshot into a draft. Nothing is saved: the user confirms first."""
        if not self.claude:
            raise ValidationError("needs_claude", "screenshot import needs a Claude API key")
        try:
            draft = read_screenshot(self.claude, Path(screenshot))
        except (ClaudeError, ValueError) as e:
            log.warning("screenshot read failed: %s", e)
            raise ValidationError("claude_failed", "could not read the screenshot") from None
        g = geocode(draft.get("area"))
        draft["geocoded"] = {"lat": g[0], "lon": g[1], "city": g[2]} if g else None
        draft["side"] = {"lost": "lost", "found": "found", "sighting": "found"}.get(draft["report_type"])
        return draft

    def mark_reunited(self, rid: int, other_id: int | None = None) -> None:
        for r in (rid, other_id):
            if r is not None:
                if self.store.get(r) is None:
                    raise ValidationError("no_report", f"no report {r}")
                self.store.set_status(r, "reunited")

    # ------------------------------------------------------------------ matching
    def matches(self, rid: int, k: int = 5) -> list[Match]:
        me = self.store.get(rid)
        if me is None:
            raise ValidationError("no_report", f"no report {rid}")
        others = self.store.open_reports("found" if me.side == "lost" else "lost")
        if not others:
            return []
        scores, km, days, img = self._score(me, others)
        order = [int(i) for i in np.argsort(-scores) if scores[i] > -1e8][:k]
        out = []
        for i in order:
            o = others[i]
            lost, found = (me, o) if me.side == "lost" else (o, me)
            used = img[i] is not None
            dist = None if km[i] is None else float(km[i])
            reason = reason_he(found.colors, lost.colors, dist if dist is not None else UNKNOWN_KM, int(days[i]),
                               float(img[i]) if used else 0.0)
            if dist is None:
                reason = reason.replace(f'{UNKNOWN_KM:.1f} ק"מ מהמקום שבו אבד', "מיקום לא ידוע")
            out.append(Match(o, float(scores[i]), dist, int(days[i]), reason, used))
        return out

    def _score(self, me: Report, others: list[Report]):
        n = len(others)
        # image: best photo pair, only where both sides have photos
        my_vecs, _ = self.store.photo_vectors([me.id])
        o_vecs, o_owner = self.store.photo_vectors([o.id for o in others])
        img: list[float | None] = [None] * n
        if len(my_vecs) and len(o_vecs):
            a = my_vecs / np.linalg.norm(my_vecs, axis=1, keepdims=True)
            b = o_vecs / np.linalg.norm(o_vecs, axis=1, keepdims=True)
            sims = (a @ b.T).max(axis=0)  # best of my photos, per candidate photo
            for j, o in enumerate(others):
                mine = sims[o_owner == o.id]
                if len(mine):
                    img[j] = float(mine.max())

        color = color_matrix([me.colors], [o.colors for o in others])[0]
        km: list[float | None] = [None] * n
        if me.lat is not None:
            idx = [j for j, o in enumerate(others) if o.lat is not None]
            if idx:
                d = haversine_matrix([me.lat], [me.lon], [others[j].lat for j in idx], [others[j].lon for j in idx])[0]
                for j, v in zip(idx, d):
                    km[j] = float(v)
        sign = 1 if me.side == "found" else -1  # days = found date - lost date
        days = np.array([sign * (me.event_date - o.event_date).days for o in others])
        allowed = np.array([species_compatible(me.species, o.species) for o in others]) & (days >= -1)

        km_arr = np.array([UNKNOWN_KM if v is None else v for v in km], dtype=np.float32)
        img_arr = np.array([0.0 if v is None else v for v in img], dtype=np.float32)
        f = Features(img_arr[None, :], color[None, :], km_arr[None, :], days[None, :], allowed[None, :])
        with_photo = score(f, PHOTO_WEIGHTS)[0]
        without = score(f, NO_PHOTO_WEIGHTS)[0]
        has = np.array([v is not None for v in img])
        return np.where(has, with_photo, without), km, days, img

    # ------------------------------------------------------------------ alerts
    def _alert_new_matches(self, rid: int) -> None:
        """Email the lost-report owner when a strong photo match appears. Finders stay private."""
        if not self.notifier:
            return
        best = self.matches(rid, k=1)
        if not best or not best[0].used_photos or best[0].score < ALERT_THRESHOLD:
            return
        me = self.store.get(rid)
        m = best[0]
        lost, found = (me, m.report) if me.side == "lost" else (m.report, me)
        if not lost.contact_email:
            return
        if not self.store.record_alert(lost.id, found.id, m.score, lost.contact_email):
            return  # already told them about this pair
        link = f"{self.base_url}/matches/{lost.id}?t={lost.access_token}"
        subject, html = match_email(lost, found, m.reason, link)
        try:
            self.notifier.send(lost.contact_email, subject, html)
        except Exception as e:  # an email failure must never lose the report
            log.warning("alert email failed for lost=%s found=%s: %s", lost.id, found.id, e)
