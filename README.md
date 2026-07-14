# Professor Rating Prediction

Predicting a professor's average star rating on [PlanetTerp](https://planetterp.com/) using NLP sentiment and regression — without ever using the star ratings themselves as features.

## The idea

PlanetTerp is a public site where students rate and review professors. Every professor has an average star rating, but the written reviews carry signal of their own. This project asks: can the *text* of the reviews alone predict the rating? The answer turns out to be yes, to a useful degree — a plain linear regression on two text-derived features explains about 79% of the variance in ratings.

## Data

All data comes from the [PlanetTerp public API](https://planetterp.com/api/v1) (no API key required):

- 13,331 professors total
- 4,212 of them have written reviews
- 1,816 have at least 5 written reviews — the modeling sample, chosen to remove low-information outliers (one glowing review isn't much signal)

Reviews are fetched with a 20-worker thread pool (`ThreadPoolExecutor`), since pulling every professor's reviews serially takes far too long.

## Features

Two features per professor, deliberately excluding the star ratings themselves:

- `review_count` — how many written reviews the professor has
- `avg_comment_sentiment` — the VADER compound sentiment score averaged over all of the professor's review texts

## Results

Three regressors, each evaluated with an 80/20 train/test split and 10-fold cross-validation, scored with R²:

| Model | Test R² | 10-fold CV R² (mean ± std) |
|---|---|---|
| LinearRegression | **0.7879** | 0.7304 ± 0.0565 |
| RandomForestRegressor | 0.7323 | 0.6461 ± 0.0688 |
| KNeighborsRegressor | 0.6473 | 0.5866 ± 0.0620 |

**Why linear regression wins:** the sentiment-to-rating relationship is essentially linear — reviews that read positively belong to highly rated professors, almost proportionally. Random Forest overfit: with only 2 features there isn't enough structure for an ensemble of deep trees to exploit. KNN underperforms because the two features are on dissimilar scales and the dataset is small and unevenly distributed, which hurts distance-based methods.

Interpretation: average comment sentiment tracks the *level* of a professor's rating, while review count mostly reflects the *stability* of that rating.

See [`WRITEUP.md`](WRITEUP.md) for the full project narrative.

## How to run

```bash
pip install -r requirements.txt
python predict_ratings.py
```

The script fetches live PlanetTerp data, so a full run takes a while and the exact numbers will drift as new reviews are posted. It prints the test R² and 10-fold CV scores for each model, then shows true-vs-predicted scatter plots and an R² comparison chart. The script was reconstructed from the code pages of the original project slides; the logic and reported metrics are unchanged.

## Repository contents

| File | Description |
|---|---|
| `predict_ratings.py` | Full pipeline: API collection → VADER sentiment → models → CV → plots |
| `WRITEUP.md` | Project writeup with data, models, and findings |
| `requirements.txt` | Python dependencies (requests, pandas, vaderSentiment, scikit-learn, matplotlib) |

## License

MIT — see [`LICENSE`](LICENSE).
