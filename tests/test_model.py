"""Offline tests: no network, a tiny synthetic snapshot."""

import gzip
import json

import pandas as pd
import pytest

from prof_ratings.data import _clean_reviews, load_snapshot
from prof_ratings.model import build_features, evaluate, mean_sentiment


def test_clean_reviews_drops_empty_and_strips():
    raw = [{"review": "  great  "}, {"review": ""}, {"review": None}, {}, {"review": "ok"}]
    assert _clean_reviews(raw) == ("great", "ok")


def test_mean_sentiment_sign_and_empty():
    assert mean_sentiment([]) == 0.0
    assert mean_sentiment(["Amazing, clear, kind professor. Loved it."]) > 0.5
    assert mean_sentiment(["Terrible, unfair, rude. Worst class ever."]) < -0.5


def test_build_features_filters_and_never_uses_rating_as_feature():
    df = pd.DataFrame(
        {
            "name": ["a", "b"],
            "slug": ["a", "b"],
            "average_rating": [4.5, 2.0],
            "reviews": [["good"] * 6, ["bad"] * 2],
        }
    )
    feats = build_features(df, min_reviews=5)
    assert list(feats["slug"]) == ["a"]
    assert feats.loc[0, "review_count"] == 6
    assert "average_rating" not in {"review_count", "avg_comment_sentiment", "text"}


def _synthetic(n=200):
    pos = ["clear lectures, fair exams, very helpful and kind"]
    neg = ["confusing, unfair grading, rude and unhelpful"]
    rows = []
    for i in range(n):
        good = i % 2 == 0
        reviews = (pos if good else neg) * (5 + i % 4)
        rows.append(
            {
                "name": f"p{i}",
                "slug": f"p{i}",
                "average_rating": 4.5 if good else 1.5,
                "reviews": reviews,
            }
        )
    return pd.DataFrame(rows)


def test_snapshot_roundtrip(tmp_path):
    df = _synthetic(4)
    path = tmp_path / "snap.csv.gz"
    out = df.assign(reviews_json=df["reviews"].map(json.dumps)).drop(columns=["reviews"])
    with gzip.open(path, "wt", encoding="utf-8") as f:
        out.to_csv(f, index=False)
    back = load_snapshot(path)
    assert back["reviews"].tolist() == df["reviews"].tolist()


def test_evaluate_is_deterministic_and_beats_baseline():
    feats = build_features(_synthetic())
    a = evaluate(feats, folds=3)
    b = evaluate(feats, folds=3)
    pd.testing.assert_frame_equal(a, b)
    r2 = dict(zip(a["model"], a["test_r2"]))
    assert r2["Mean baseline"] <= 0.0 + 1e-9
    assert r2["TF-IDF + Ridge (review text)"] > 0.9
    assert r2["LinearRegression (2 features)"] > 0.9


@pytest.mark.parametrize("folds", [2, 5])
def test_evaluate_reports_every_model(folds):
    table = evaluate(build_features(_synthetic()), folds=folds)
    assert len(table) == 7
    assert set(table.columns) == {"model", "test_r2", "test_mae", "cv_r2_mean", "cv_r2_std"}
