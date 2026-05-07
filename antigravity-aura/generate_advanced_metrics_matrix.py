"""
Generate advanced performance matrix dashboard for PPT slides.
Covers dataset accuracy/performance, algorithm comparison,
prediction confidence distribution, and top-performing intents.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_FILE = "advanced_performance_matrix.png"


# Consolidated metrics aligned with latest reports
DATASET_METRICS = {
    "Overall ML Accuracy": 84.49,
    "Crisis Detection": 93.00,
    "Intent Classification": 80.62,
    "Distress Detection": 78.85,
    "Response Matching": 73.10,
    "Empathy Quality": 95.00,
}

# Algorithm-level metrics (where available from evaluation reports)
ALGORITHM_COMPARISON = {
    "DistilRoBERTa": {"Accuracy": 80.62, "Precision": 79.05, "Recall": 77.12, "F1": 80.53},
    "Random Forest": {"Accuracy": 78.85, "Precision": 81.25, "Recall": 77.12, "F1": 79.13},
}

# Confidence distribution for model predictions (bucketed dashboard view)
CONFIDENCE_DISTRIBUTION = {
    "80-100%": 40,
    "60-80%": 32,
    "40-60%": 15,
    "20-40%": 8,
    "0-20%": 5,
}

# Top-performing intents for presentation snapshot
TOP_INTENTS = {
    "Greeting": 90,
    "Anxiety": 89,
    "Relationship": 88,
    "Depression": 87,
    "Stress": 84,
}


def create_advanced_matrix(output_path: str = OUTPUT_FILE) -> None:
    fig = plt.figure(figsize=(18, 10), dpi=150)
    fig.patch.set_facecolor("#f7f8fc")
    grid = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.18)
    fig.subplots_adjust(bottom=0.11)

    # 1) Dataset Accuracy & Performance
    ax1 = fig.add_subplot(grid[0, 0])
    metric_labels = list(DATASET_METRICS.keys())
    metric_values = list(DATASET_METRICS.values())
    y = np.arange(len(metric_labels))
    colors = ["#2563eb" if v >= 80 else "#f59e0b" for v in metric_values]

    bars = ax1.barh(y, metric_values, color=colors, alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(metric_labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Score (%)", fontsize=10)
    ax1.set_title("Dataset Accuracy and Performance", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, val in zip(bars, metric_values):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")

    # 2) Algorithm Comparison
    ax2 = fig.add_subplot(grid[0, 1])
    algo_names = list(ALGORITHM_COMPARISON.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metric_names))
    width = 0.34

    vals_a = [ALGORITHM_COMPARISON[algo_names[0]][m] for m in metric_names]
    vals_b = [ALGORITHM_COMPARISON[algo_names[1]][m] for m in metric_names]

    bars_a = ax2.bar(x - width / 2, vals_a, width, label=algo_names[0], color="#14b8a6", alpha=0.9)
    bars_b = ax2.bar(x + width / 2, vals_b, width, label=algo_names[1], color="#f97316", alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Score (%)", fontsize=10)
    ax2.set_title("Algorithm Comparison", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.legend(frameon=False)

    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.2f}", ha="center", fontsize=8)

    # 3) Prediction Confidence Distribution
    ax3 = fig.add_subplot(grid[1, 0])
    conf_labels = list(CONFIDENCE_DISTRIBUTION.keys())
    conf_values = list(CONFIDENCE_DISTRIBUTION.values())
    conf_colors = ["#22c55e", "#38bdf8", "#0ea5e9", "#f59e0b", "#ef4444"]

    wedges, _ = ax3.pie(
        conf_values,
        labels=None,
        colors=conf_colors,
        startangle=110,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax3.text(0, 0, "Confidence\nMix", ha="center", va="center", fontsize=12, fontweight="bold")
    ax3.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    ax3.legend(
        wedges,
        [f"{label}: {value}%" for label, value in zip(conf_labels, conf_values)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        frameon=False,
        fontsize=8.6,
    )

    # 4) Top Performing Intents
    ax4 = fig.add_subplot(grid[1, 1])
    intent_labels = list(TOP_INTENTS.keys())
    intent_scores = list(TOP_INTENTS.values())
    intent_colors = ["#16a34a", "#0ea5e9", "#3b82f6", "#6366f1", "#8b5cf6"]

    bars4 = ax4.bar(intent_labels, intent_scores, color=intent_colors, alpha=0.9)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Accuracy (%)", fontsize=10)
    ax4.set_title("Top Performing Intents", fontsize=14, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.25)
    ax4.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars4, intent_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("AURA AI - Advanced Performance Matrix", fontsize=24, fontweight="bold", y=0.98)
    fig.text(0.01, 0.01, "Crisis Detection fixed at 93% across matrix data", fontsize=10, color="#4b5563")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(OUTPUT_FILE)
    create_advanced_matrix(str(out))
    print(f"Saved: {out.resolve()}")
