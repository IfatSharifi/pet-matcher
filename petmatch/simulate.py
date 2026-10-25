"""Simulated places and dates for lost/found reports.

DogFaceNet has photos only. To test whether place and date improve ranking, every
lost report gets a place and date, and its matching found report is placed and dated
relative to it. All assumptions are parameters, so the eval can show how much the
result depends on them. These are assumptions, not measurements: replace them with
real report data once the app has some.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta

# (city, lat, lon, relative report volume) - volume roughly follows population.
CITIES = [
    ("תל אביב", 32.080, 34.781, 10), ("ירושלים", 31.778, 35.214, 9), ("חיפה", 32.794, 34.990, 6),
    ("ראשון לציון", 31.964, 34.804, 5), ("פתח תקווה", 32.087, 34.887, 5), ("אשדוד", 31.801, 34.643, 5),
    ("נתניה", 32.321, 34.853, 5), ("באר שבע", 31.252, 34.791, 5), ("חולון", 32.011, 34.774, 4),
    ("רמת גן", 32.068, 34.824, 4), ("רחובות", 31.894, 34.809, 3), ("כפר סבא", 32.175, 34.907, 2),
    ("הרצליה", 32.166, 34.843, 2), ("מודיעין", 31.899, 35.010, 2), ("ראש העין", 32.096, 34.957, 1),
]


@dataclass(frozen=True)
class SimParams:
    name: str = "base"
    city_spread_km: float = 2.5        # how far lost places scatter around a city center
    found_median_km: float = 1.5       # typical distance between where lost and where found
    found_far_share: float = 0.10      # share of pets found far away (wandered, picked up by car)
    found_far_km: float = 15.0
    window_days: int = 90              # lost reports span this period
    found_mean_days: float = 3.0       # typical days between lost and found
    city_only_share: float = 0.15      # found reports that only name a city (no street)


SCENARIOS = {
    "base": SimParams(),
    "wander": SimParams(name="wander", found_median_km=5.0, found_far_share=0.25, found_mean_days=7.0),
    "dense": SimParams(name="dense", city_spread_km=1.0, window_days=30),
}


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float
    city: str


def _offset(lat: float, lon: float, km: float, bearing: float) -> tuple[float, float]:
    dlat = km / 111.0 * math.cos(bearing)
    dlon = km / (111.0 * math.cos(math.radians(lat))) * math.sin(bearing)
    return lat + dlat, lon + dlon


def haversine_km(a: Place, b: Place) -> float:
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def simulate(pet_ids: list[str], params: SimParams, seed: int = 0,
             start: date = date(2026, 7, 1)) -> tuple[dict, dict]:
    """Return ({pet_id: (Place, date)} for lost reports, same for found reports)."""
    rng = random.Random(f"{seed}-{params.name}")
    weights = [c[3] for c in CITIES]
    lost, found = {}, {}
    for pid in pet_ids:
        city, clat, clon, _ = rng.choices(CITIES, weights=weights)[0]
        lat, lon = _offset(clat, clon, abs(rng.gauss(0, params.city_spread_km)), rng.uniform(0, 2 * math.pi))
        lost_day = start + timedelta(days=rng.randrange(params.window_days))
        lost[pid] = (Place(lat, lon, city), lost_day)

        if rng.random() < params.found_far_share:
            km = rng.uniform(3, params.found_far_km)
        else:  # lognormal around the median: mostly close, a long tail
            km = params.found_median_km * math.exp(rng.gauss(0, 0.8))
        flat, flon = _offset(lat, lon, km, rng.uniform(0, 2 * math.pi))
        found_city = min(CITIES, key=lambda c: (c[1] - flat) ** 2 + (c[2] - flon) ** 2)
        if rng.random() < params.city_only_share:  # post says only "Holon": geocodes to the center
            flat, flon = found_city[1], found_city[2]
        days = min(int(rng.expovariate(1 / params.found_mean_days)), 30)
        found[pid] = (Place(flat, flon, found_city[0]), lost_day + timedelta(days=days))
    return lost, found
