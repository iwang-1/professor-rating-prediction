"""Data collection and caching for the PlanetTerp professor-rating study.

`fetch_snapshot` pulls every professor (and their written reviews) from the public PlanetTerp API and
writes them to a gzipped CSV, so modeling runs are reproducible and do not hit the API. `load_snapshot`
reads that file back.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://planetterp.com/api/v1"
PAGE_SIZE = 100


@dataclass(frozen=True)
class Professor:
    name: str
    slug: str
    average_rating: float
    reviews: tuple[str, ...]


def _get(session: requests.Session, path: str, params: dict, retries: int = 4) -> object:
    """GET a JSON endpoint, retrying transient failures with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = session.get(f"{API_BASE}/{path}", params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _clean_reviews(raw_reviews: list[dict]) -> tuple[str, ...]:
    """Keep the non-empty review texts, stripped."""
    texts = (str(r.get("review") or "").strip() for r in raw_reviews)
    return tuple(t for t in texts if t)


def _fetch_one(session: requests.Session, name: str, slug: str, rating: float) -> Professor:
    js = _get(session, "professor", {"name": name, "reviews": "true"})
    reviews = _clean_reviews(js.get("reviews", []) if isinstance(js, dict) else [])
    return Professor(name=name, slug=slug, average_rating=float(rating), reviews=reviews)


def fetch_snapshot(out_path: Path, workers: int = 20) -> int:
    """Fetch all rated professors with their reviews and write them to `out_path` (gzipped CSV).

    Returns the number of professors written. Professors without an average rating are skipped, since
    there is nothing to predict for them; professors with a rating but no written reviews are kept with
    an empty review list so the filtering step downstream is explicit.
    """
    session = requests.Session()
    rated: list[tuple[str, str, float]] = []
    offset = 0
    while True:
        page = _get(session, "professors", {"limit": PAGE_SIZE, "offset": offset})
        if not page:
            break
        for prof in page:
            if prof.get("type") == "professor" and prof.get("average_rating") is not None:
                rated.append((prof["name"], prof["slug"], prof["average_rating"]))
        offset += PAGE_SIZE

    with ThreadPoolExecutor(max_workers=workers) as pool:
        profs = list(pool.map(lambda p: _fetch_one(session, *p), rated))

    rows = [
        {
            "name": p.name,
            "slug": p.slug,
            "average_rating": p.average_rating,
            "reviews_json": json.dumps(list(p.reviews), ensure_ascii=False),
        }
        for p in profs
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).sort_values("slug").to_csv(f, index=False)
    return len(rows)


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot written by `fetch_snapshot`. Adds a `reviews` column holding a list of strings."""
    df = pd.read_csv(path, compression="gzip")
    df["reviews"] = df["reviews_json"].map(json.loads)
    return df.drop(columns=["reviews_json"])
