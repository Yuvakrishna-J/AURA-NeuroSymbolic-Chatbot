import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def generate_slide(output_path: str = "overall_performance_metrics_slide.png") -> None:
    # Current metrics from latest reports
    overall_accuracy = 95.00
    metrics = {
        "Crisis Detection": 93.00,
        "DistilRoBERTa Intent": 80.62,
        "Random Forest Distress": 78.85,
        "T5 Empathy Generator": 95.00,
        "TF-IDF Fallback": 75.00,
    }

    # 16:9 slide canvas
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f5f7fb")

    # Title area
    fig.text(0.05, 0.93, "AURA AI - Overall Performance Metrics", fontsize=30, fontweight="bold", color="#111827")
    fig.text(0.05, 0.895, "Single-Slide Analytical Summary | Generated: " + datetime.now().strftime("%d %b %Y"), fontsize=13, color="#4b5563")

    # KPI card
    ax_kpi = fig.add_axes([0.05, 0.71, 0.27, 0.15])
    ax_kpi.axis("off")
    ax_kpi.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#0f172a", edgecolor="none", transform=ax_kpi.transAxes))
    ax_kpi.text(0.05, 0.72, "Overall Performance", fontsize=14, color="#cbd5e1", weight="bold")
    ax_kpi.text(0.05, 0.25, f"{overall_accuracy:.2f}%", fontsize=38, color="#22d3ee", weight="bold")

    # Status card
    ax_status = fig.add_axes([0.34, 0.71, 0.18, 0.15])
    ax_status.axis("off")
    ax_status.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#ecfeff", edgecolor="#06b6d4", linewidth=2, transform=ax_status.transAxes))
    ax_status.text(0.08, 0.58, "System Status", fontsize=12, color="#0e7490", weight="bold")
    ax_status.text(0.08, 0.25, "Production-Ready", fontsize=18, color="#0f766e", weight="bold")

    # Highlights card
    ax_high = fig.add_axes([0.54, 0.71, 0.41, 0.15])
    ax_high.axis("off")
    ax_high.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#d1d5db", linewidth=1.5, transform=ax_high.transAxes))
    highlights = [
        "95.00% overall model performance",
        "93% crisis detection (safety-critical)",
        "Multi-model stack with robust fallback reliability",
    ]
    ax_high.text(0.03, 0.76, "Key Highlights", fontsize=13, color="#111827", weight="bold")
    for i, line in enumerate(highlights):
        ax_high.text(0.03, 0.52 - i * 0.22, f"- {line}", fontsize=11.5, color="#374151")

    # Main bar chart
    ax = fig.add_axes([0.05, 0.14, 0.68, 0.50])
    labels = list(metrics.keys())
    values = list(metrics.values())

    y = np.arange(len(labels))
    colors = ["#10b981" if v >= 90 else "#14b8a6" if v >= 80 else "#f59e0b" for v in values]
    bars = ax.barh(y, values, color=colors, edgecolor="none", height=0.58)

    ax.set_xlim(0, 105)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Component-Wise Accuracy", fontsize=16, pad=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, v in zip(bars, values):
        ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=11, color="#111827", weight="bold")

    # Reference lines
    ax.axvline(80, color="#0ea5e9", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(80.5, -0.45, "80% target", color="#0369a1", fontsize=10)
    ax.axvline(overall_accuracy, color="#7c3aed", linestyle="-.", linewidth=1.6, alpha=0.9)
    ax.text(overall_accuracy + 0.5, 4.45, f"Overall {overall_accuracy:.2f}%", color="#5b21b6", fontsize=10)

    # Distribution donut
    ax_donut = fig.add_axes([0.77, 0.25, 0.19, 0.34])
    bands = [sum(v >= 90 for v in values), sum((80 <= v < 90) for v in values), sum(v < 80 for v in values)]
    band_labels = ["Excellent (>=90)", "Strong (80-89)", "Improvement (<80)"]
    band_colors = ["#16a34a", "#0ea5e9", "#f59e0b"]
    wedges, _ = ax_donut.pie(bands, colors=band_colors, startangle=90, wedgeprops={"width": 0.38, "edgecolor": "white"})
    ax_donut.text(0, 0, "Model\nMix", ha="center", va="center", fontsize=12, weight="bold", color="#111827")
    ax_donut.set_title("Performance Tier Distribution", fontsize=12, pad=10)
    ax_donut.legend(
        wedges,
        [f"{l}: {c}" for l, c in zip(band_labels, bands)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        fontsize=8.5,
        frameon=False,
    )

    # Footer note
    fig.text(0.05, 0.055, "Source: ML_ACCURACY_REPORT.md (latest consolidated metrics)", fontsize=10.5, color="#6b7280")
    fig.text(0.95, 0.055, "AURA Analytics", fontsize=10.5, color="#9ca3af", ha="right")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_slide()
    print("Saved: overall_performance_metrics_slide.png")
