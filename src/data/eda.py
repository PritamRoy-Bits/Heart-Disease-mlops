"""Generate professional EDA visualizations for the cleaned Heart Disease dataset.

Produces: feature histograms, a correlation heatmap, and a class-balance plot,
all saved under reports/figures/.
"""
import logging

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CI/servers
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FIGURES_DIR, PROCESSED_FILE, TARGET_COLUMN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_histograms(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.drop(TARGET_COLUMN, errors="ignore")
    fig, axes = plt.subplots(nrows=(len(numeric_cols) + 2) // 3, ncols=3, figsize=(15, 10))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    for ax in axes.flatten()[len(numeric_cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_histograms.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature_histograms.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")


def plot_class_balance(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=counts.index.map({0: "No Disease", 1: "Disease"}), y=counts.values, ax=ax,
                hue=counts.index, palette="viridis", legend=False)
    ax.set_ylabel("Count")
    ax.set_title("Class Balance: Heart Disease Target")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved class_balance.png")


def run(processed_path=PROCESSED_FILE, out_dir=FIGURES_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(processed_path)
    plot_histograms(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_class_balance(df, out_dir)
    logger.info("EDA complete. Figures saved to %s", out_dir)


if __name__ == "__main__":
    run()
