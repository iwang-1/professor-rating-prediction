# Professor Rating Prediction — PlanetTerp ML

Can you predict a professor's average star rating on [PlanetTerp](https://planetterp.com/) without ever looking at the star ratings students give? This project tries exactly that: it predicts a professor's average rating using only signals derived from the *text* of their reviews.

## Data

All data comes from the [PlanetTerp public API](https://planetterp.com/api/v1) (no API key required):

- 13,331 professors total
- 4,212 of them have written reviews
- 1,816 have at least 5 written reviews — this is the modeling sample, chosen to remove low-information outliers (a professor with one glowing review isn't much signal)

Reviews are fetched with a 20-worker thread pool (`ThreadPoolExecutor`) since pulling every professor's reviews serially takes far too long.

## Features

Just two features per professor — deliberately excluding the star ratings themselves:

- `review_count` — how many written reviews the professor has
- `avg_comment_sentiment` — the VADER compound sentiment score averaged over all of the professor's review texts

## Models and evaluation

Three regressors, each evaluated with an 80/20 train/test split and 10-fold cross-validation, scored with R²:

| Model | Test R² | 10-fold CV R² (mean ± std) |
|---|---|---|
| LinearRegression | **0.7879** | 0.7304 ± 0.0565 |
| RandomForestRegressor | 0.7323 | 0.6461 ± 0.0688 |
| KNeighborsRegressor | 0.6473 | 0.5866 ± 0.0620 |

## Findings

- **Linear regression wins** because the sentiment-to-rating relationship is essentially linear: reviews that read positively belong to highly rated professors, almost proportionally.
- **Random Forest overfit** — with only 2 features there isn't enough structure for an ensemble of deep trees to exploit, so it memorizes noise instead.
- **KNN underperforms** — the two features are on dissimilar scales, and the dataset is small and unevenly distributed, which hurts distance-based methods.
- Interpretation: average comment sentiment tracks the *level* of a professor's rating, while review count mostly reflects the *stability* of that rating.

## Reproducing

```bash
pip install -r requirements.txt
python predict_ratings.py
```

The script fetches live PlanetTerp data, so a full run takes a while and the exact numbers drift as new reviews are posted. The script was reconstructed from the code pages of the original project slide deck; the logic and reported metrics are unchanged.
