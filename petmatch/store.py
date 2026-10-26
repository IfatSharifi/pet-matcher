"""SQLite storage for reports, photos (with embeddings) and sent alerts.

SQLite needs no server and is fine for thousands of reports; matching loads the open
reports into numpy and scores them in memory. Moving to Postgres + pgvector later means
re-implementing this class only - the service talks to Store, not to SQL.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    side          TEXT NOT NULL CHECK (side IN ('lost', 'found')),
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'reunited', 'closed')),
    species       TEXT NOT NULL,
    colors        TEXT NOT NULL,          -- JSON list from the fixed vocabulary
    description   TEXT NOT NULL,          -- JSON: Claude's full description, or what the user typed
    pet_name      TEXT,
    lat           REAL,
    lon           REAL,
    place_precision TEXT,                 -- 'exact' (map pin) or 'city' (only a city name)
    area_text     TEXT,                   -- as the user or the post wrote it, Hebrew
    event_date    TEXT NOT NULL,          -- YYYY-MM-DD lost / found
    contact_name  TEXT,
    contact_phone TEXT,
    contact_email TEXT,
    source        TEXT NOT NULL DEFAULT 'form',   -- 'form' or 'screenshot'
    access_token  TEXT NOT NULL,          -- secret in the reporter's private link; guards contacts
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER NOT NULL REFERENCES reports(id),
    path       TEXT NOT NULL,
    embedding  BLOB NOT NULL               -- float32 vector
);
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lost_id       INTEGER NOT NULL REFERENCES reports(id),
    found_id      INTEGER NOT NULL REFERENCES reports(id),
    score         REAL NOT NULL,
    sent_to       TEXT NOT NULL,
    sent_at       TEXT NOT NULL,
    UNIQUE (lost_id, found_id)             -- never alert twice about the same pair
);
CREATE INDEX IF NOT EXISTS idx_reports_open ON reports(side, status);
CREATE INDEX IF NOT EXISTS idx_photos_report ON photos(report_id);
"""


@dataclass
class Report:
    side: str
    species: str
    colors: list[str]
    event_date: date
    description: dict = field(default_factory=dict)
    pet_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    place_precision: str | None = None
    area_text: str | None = None
    contact_name: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    source: str = "form"
    status: str = "open"
    id: int | None = None
    created_at: str | None = None
    has_photo: bool = False
    access_token: str | None = None

    def public(self) -> dict:
        """What other users may see: no contact details, place rounded to ~1 km."""
        return {
            "id": self.id, "side": self.side, "status": self.status, "species": self.species,
            "colors": self.colors, "pet_name": self.pet_name, "area_text": self.area_text,
            "event_date": self.event_date.isoformat(), "has_photo": self.has_photo,
            "breed_guess": self.description.get("breed_guess"),
            "markings": self.description.get("distinctive_markings", []),
            "approx_lat": round(self.lat, 2) if self.lat is not None else None,
            "approx_lon": round(self.lon, 2) if self.lon is not None else None,
        }


class Store:
    def __init__(self, path: str | Path = "petmatch.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()  # one writer at a time; web servers call from threads
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self.db.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer, or each other
        self.db.executescript(SCHEMA)

    @property
    def db(self) -> sqlite3.Connection:
        """One connection per thread. A single shared connection breaks when the web server
        answers several requests at once (e.g. the 5 photos on a matches page)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)  # closed from close()
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def close(self) -> None:
        """Fold the write-ahead log into the main file and close every thread's connection."""
        with self._conns_lock:
            for i, conn in enumerate(self._conns):
                if i == 0:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            self._conns.clear()
        self._local = threading.local()

    # ---- reports
    def add_report(self, r: Report, photos: list[tuple[str, np.ndarray]] = ()) -> int:
        r.access_token = r.access_token or secrets.token_urlsafe(16)
        with self._lock, self.db:
            cur = self.db.execute(
                """INSERT INTO reports (side, status, species, colors, description, pet_name, lat, lon,
                   place_precision, area_text, event_date, contact_name, contact_phone, contact_email,
                   source, access_token, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r.side, r.status, r.species, json.dumps(r.colors), json.dumps(r.description, ensure_ascii=False),
                 r.pet_name, r.lat, r.lon, r.place_precision, r.area_text, r.event_date.isoformat(),
                 r.contact_name, r.contact_phone, r.contact_email, r.source, r.access_token,
                 datetime.now().isoformat(timespec="seconds")))
            rid = cur.lastrowid
            for path, vec in photos:
                self.db.execute("INSERT INTO photos (report_id, path, embedding) VALUES (?,?,?)",
                                (rid, path, np.asarray(vec, dtype=np.float32).tobytes()))
        return rid

    def _row_to_report(self, row: sqlite3.Row, has_photo: bool) -> Report:
        return Report(
            id=row["id"], side=row["side"], status=row["status"], species=row["species"],
            colors=json.loads(row["colors"]), description=json.loads(row["description"]),
            pet_name=row["pet_name"], lat=row["lat"], lon=row["lon"], place_precision=row["place_precision"],
            area_text=row["area_text"], event_date=date.fromisoformat(row["event_date"]),
            contact_name=row["contact_name"], contact_phone=row["contact_phone"],
            contact_email=row["contact_email"], source=row["source"], created_at=row["created_at"],
            has_photo=has_photo, access_token=row["access_token"])

    def check_token(self, rid: int, token: str | None) -> bool:
        row = self.db.execute("SELECT access_token FROM reports WHERE id = ?", (rid,)).fetchone()
        return bool(row and token and secrets.compare_digest(row["access_token"], token))

    def get(self, rid: int) -> Report | None:
        row = self.db.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        if row is None:
            return None
        n = self.db.execute("SELECT COUNT(*) FROM photos WHERE report_id = ?", (rid,)).fetchone()[0]
        return self._row_to_report(row, n > 0)

    def open_reports(self, side: str) -> list[Report]:
        rows = self.db.execute(
            """SELECT r.*, EXISTS(SELECT 1 FROM photos p WHERE p.report_id = r.id) AS has_photo
               FROM reports r WHERE side = ? AND status = 'open' ORDER BY id""", (side,)).fetchall()
        return [self._row_to_report(row, bool(row["has_photo"])) for row in rows]

    def set_status(self, rid: int, status: str) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE reports SET status = ? WHERE id = ?", (status, rid))

    # ---- photos
    def photo_vectors(self, report_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(vectors, owning report id per vector) for the given reports."""
        if not report_ids:
            return np.zeros((0, 0), np.float32), np.zeros(0, int)
        marks = ",".join("?" * len(report_ids))
        rows = self.db.execute(f"SELECT report_id, embedding FROM photos WHERE report_id IN ({marks}) ORDER BY id",
                               report_ids).fetchall()
        if not rows:
            return np.zeros((0, 0), np.float32), np.zeros(0, int)
        vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return vecs, np.array([r["report_id"] for r in rows])

    def photo_paths(self, rid: int) -> list[str]:
        return [r["path"] for r in self.db.execute("SELECT path FROM photos WHERE report_id = ? ORDER BY id", (rid,))]

    def photo_path(self, rid: int, n: int) -> str | None:
        paths = self.photo_paths(rid)
        return paths[n] if 0 <= n < len(paths) else None

    # ---- alerts
    def record_alert(self, lost_id: int, found_id: int, score: float, sent_to: str) -> bool:
        """False if this pair was already alerted."""
        try:
            with self._lock, self.db:
                self.db.execute("INSERT INTO alerts (lost_id, found_id, score, sent_to, sent_at) VALUES (?,?,?,?,?)",
                                (lost_id, found_id, score, sent_to, datetime.now().isoformat(timespec="seconds")))
            return True
        except sqlite3.IntegrityError:
            return False

    def alerts(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM alerts ORDER BY id")]
