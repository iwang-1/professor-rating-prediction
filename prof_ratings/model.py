"""Features and model evaluation for the professor-rating study.

Every model predicts a professor's average star rating from the text of their written reviews only; the
star ratings are never used as features. All randomness is seeded, so a run on the same snapshot gives
the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
MIN_REVIEWS = 5
NUMERIC_FEATURES = ["review_count", "avg_comment_sentiment"]

_sia = SentimentIntensityAnalyzer()


def mean_sentiment(reviews: list[str]) -> float:
    """Mean VADER compound score over the review texts (0.0 for no reviews)."""
    if not reviews:
        return 0.0
    return float(np.mean([_sia.polarity_scores(t)["compound"] for t in reviews]))


def build_features(df: pd.DataFrame, min_reviews: int = MIN_REVIEWS) -> pd.DataFrame:
    """Add the two numeric features and the concatenated review text; keep professors with enough reviews.

    `min_reviews` drops professors whose rating rests on very few reviews (low-information outliers).
    """
    out = df.copy()
    out["review_count"] = out["reviews"].map(len)
    out = out[out["review_count"] >= min_reviews].copy()
    out["avg_comment_sentiment"] = out["reviews"].map(mean_sentiment)
    out["text"] = out["reviews"].map(lambda rs: "\n".join(rs))
    return out.reset_index(drop=True)


def _text_column(frame: pd.DataFrame) -> pd.Series:
    return frame["text"]


def _numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[NUMERIC_FEATURES]


def models() -> dict[str, object]:
    """The compared models. Each takes the feature frame from `build_features` as X."""
    numeric = FunctionTransformer(_numeric_columns)
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "LinearRegression (2 features)": make_pipeline(numeric, LinearRegression()),
        "RandomForest (2 features)": make_pipeline(
            numeric, RandomForestRegressor(random_state=SEED, n_jobs=-1)
        ),
        "KNN, unscaled (2 features)": make_pipeline(numeric, KNeighborsRegressor(n_neighbors=5)),
        "KNN, scaled (2 features)": make_pipeline(
            numeric, StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "TF-IDF + Ridge (review text)": make_pipeline(
            FunctionTransformer(_text_column),
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
            Ridge(alpha=1.0),
        ),
        "TF-IDF + 2 features + Ridge": make_pipeline(
            make_union(
                make_pipeline(
                    FunctionTransformer(_text_column),
                    TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                ),
                make_pipeline(FunctionTransformer(_numeric_columns), StandardScaler()),
            ),
            Ridge(alpha=1.0),
        ),
    }


def evaluate(features: pd.DataFrame, folds: int = 10) -> pd.DataFrame:
    """Seeded 80/20 holdout plus k-fold CV on the training split, for every model.

    Returns one row per model with test R^2, test MAE, and CV R^2 mean and standard deviation.
    """
    y = features["average_rating"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=SEED
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        rows.append(
            {
                "model": name,
                "test_r2": r2_score(y_test, pred),
                "test_mae": mean_absolute_error(y_test, pred),
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
            }
        )
    return pd.DataFrame(rows)
