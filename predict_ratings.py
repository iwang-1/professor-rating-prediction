"""Professor Rating Prediction — PlanetTerp ML.

Predict a professor's average star rating from PlanetTerp review data
WITHOUT using the star ratings themselves as features. Two features are used:

  * review_count           — how many written reviews a professor has
  * avg_comment_sentiment  — mean VADER compound sentiment over review texts

Models compared: LinearRegression, RandomForestRegressor, KNeighborsRegressor
(80/20 train/test split + 10-fold cross-validation, scored with R^2).

Reconstructed as a standalone script from the code pages of the original
project slide deck (see WRITEUP.md for the full narrative and results).
The logic and reported metrics are unchanged.

Data source: PlanetTerp public API (https://planetterp.com/api/v1) — no API
key required. Fetching all professors takes a while; reviews are pulled with
a 20-worker thread pool.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_BASE = "https://planetterp.com/api/v1"

sia = SentimentIntensityAnalyzer()


def get_reviews(name, avg_rating, slug):
    """Fetch one professor's reviews and compute average compound sentiment.

    Returns a dict of features for the professor, or None if the professor
    has no non-empty review texts (or the request fails).
    """
    try:
        resp = requests.get(
            f"{API_BASE}/professor",
            params={"name": name, "reviews": "true"},
            timeout=60,
        )
        resp.raise_for_status()
        js = resp.json()
        reviews = [
            r.get("review", "").strip()
            for r in js.get("reviews", [])
            if r.get("review", "").strip()
        ]
        if not reviews:
            return None
        scores = [sia.polarity_scores(text)["compound"] for text in reviews]
        sent = sum(scores) / len(scores)
        return {
            "name": name,
            "slug": slug,
            "average_rating": avg_rating,
            "review_count": len(reviews),
            "avg_comment_sentiment": sent,
        }
    except Exception as e:
        print(f"SKIPPING {name}: {e}")
        return None


def collect_professor_data():
    """Page through /professors and fetch reviews with a 20-worker pool."""
    data = []
    offset = 0
    count = 0
    while True:
        req = requests.get(
            f"{API_BASE}/professors",
            params={"limit": 100, "offset": offset},
        )
        req.raise_for_status()
        prof_group = req.json()
        if not prof_group:
            break

        thread = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            for prof in prof_group:
                if (
                    prof.get("type") == "professor"
                    and prof.get("average_rating") is not None
                ):
                    name = prof["name"]
                    slug = prof["slug"]
                    avg_rating = prof["average_rating"]
                    thread.append(
                        executor.submit(get_reviews, name, avg_rating, slug)
                    )
            for future in as_completed(thread):
                res = future.result()
                if res:
                    data.append(res)
                    count += 1
                    print(
                        f"{count}. {res['name']} {res['review_count']} "
                        f"{res['avg_comment_sentiment']:.2f}"
                    )
        offset += 100

    return pd.DataFrame(data)


def plot_true_vs_predicted(y_true, y_pred, model_name):
    """Scatter of true vs. predicted ratings with the y=x reference line."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "k--",
        lw=2,
    )
    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"{model_name} - True vs Predicted")
    plt.grid(True)
    plt.show()


def main():
    # ── Data collection ─────────────────────────────────────────────────
    df = collect_professor_data()
    print(f"{len(df)}")

    # Keep only professors with at least 5 reviews to remove low-info
    # outliers (1,816 of 4,212 reviewed professors at time of the project).
    filtered_df = df[df["review_count"] >= 5]
    X = filtered_df[["review_count", "avg_comment_sentiment"]]
    y = filtered_df["average_rating"]
    print(X.count())

    # ── Train/test split and models ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"{r2_score(y_test, y_pred_lr):.4f}")  # slide result: 0.7879

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"{r2_score(y_test, y_pred_rf):.4f}")  # slide result: 0.7323

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"{r2_score(y_test, y_pred_knn):.4f}")  # slide result: 0.6473

    # ── 10-fold cross-validation ────────────────────────────────────────
    # Slide results: LinearRegression 0.7304 +- 0.0565,
    #                RandomForest     0.6461 +- 0.0688,
    #                KNN              0.5866 +- 0.0620
    kf = KFold(n_splits=10)
    models = [
        ("LinearRegression", lr),
        ("RandomForest", rf),
        ("KNN", knn),
    ]
    for name, model in models:
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
        )
        print(f"{name} {scores.mean():.4f} +- {scores.std():.4f}")

    # ── Plots ───────────────────────────────────────────────────────────
    plot_true_vs_predicted(y_test, y_pred_lr, "Linear Regression")
    plot_true_vs_predicted(y_test, y_pred_rf, "Random Forest")
    plot_true_vs_predicted(y_test, y_pred_knn, "KNN Regressor")

    # R^2 score comparison bar chart
    results_df = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest", "KNN"],
            "R²": [
                r2_score(y_test, y_pred_lr),
                r2_score(y_test, y_pred_rf),
                r2_score(y_test, y_pred_knn),
            ],
        }
    )
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["R²"], color=["red", "blue", "green"])
    plt.title("Model R² Score Comparison")
    plt.ylabel("R² Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
