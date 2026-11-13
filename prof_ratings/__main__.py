"""Command line entry point.

    python -m prof_ratings fetch      # download a fresh snapshot from the PlanetTerp API
    python -m prof_ratings evaluate   # fit and score every model on the committed snapshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "planetterp_snapshot.csv.gz"
RESULTS = ROOT / "results" / "model_comparison.csv"
FIGURE = ROOT / "docs" / "model_comparison.png"


def _fetch(args: argparse.Namespace) -> None:
    from .data import fetch_snapshot

    n = fetch_snapshot(args.snapshot, workers=args.workers)
    print(f"wrote {n} rated professors to {args.snapshot}")


def _evaluate(args: argparse.Namespace) -> None:
    from .data import load_snapshot
    from .model import build_features, evaluate
    from .plots import plot_comparison

    raw = load_snapshot(args.snapshot)
    reviewed = int((raw["reviews"].map(len) > 0).sum())
    features = build_features(raw, min_reviews=args.min_reviews)
    print(
        f"snapshot: {len(raw)} rated professors, {reviewed} with written reviews, "
        f"{len(features)} with >= {args.min_reviews} reviews (modeling sample)"
    )
    table = evaluate(features, folds=args.folds)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.4f")
    plot_comparison(table, FIGURE)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prof_ratings")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download a fresh snapshot from the PlanetTerp API")
    fetch.add_argument("--workers", type=int, default=20)
    fetch.set_defaults(func=_fetch)
    ev = sub.add_parser("evaluate", help="fit and score every model on the snapshot")
    ev.add_argument("--min-reviews", type=int, default=5)
    ev.add_argument("--folds", type=int, default=10)
    ev.set_defaults(func=_evaluate)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
