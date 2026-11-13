"""Figures written to files (not shown interactively), so runs work headless and in CI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_comparison(table: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart of held-out test R^2 per model, with the CV mean +- std drawn over it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(t["model"], t["test_r2"], color="#1E4B7A", alpha=0.85, label="held-out test R²")
    ax.errorbar(
        t["cv_r2_mean"],
        t["model"],
        xerr=t["cv_r2_std"],
        fmt="o",
        color="#A6431E",
        capsize=3,
        zorder=3,
        label="10-fold CV R² (mean ± std)",
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("R²")
    ax.set_xlim(-0.05, 1.0)
    ax.set_title("Predicting PlanetTerp star ratings from review text only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
