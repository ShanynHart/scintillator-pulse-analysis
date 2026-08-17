from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    results_dir = Path(
        "/Users/shanynhart/Library/CloudStorage/GoogleDrive-shanyn.hart@gmail.com/Other computers/MyMacbookPro/PhD/PANGoLINS/Measurements/PSA/09032026/analysis_results"
    )
    features_csv = results_dir / "features_all_traces.csv"

    df = pd.read_csv(features_csv)

    order = [
        "A_center",
        "A_outer_edge",
        "A_B_boundary",
        "B_center",
        "B_outer_edge",
        "C_outer_edge",
        "C_center",
        "A_B_C_D_meeting",
    ]
    features = [
        "rise_time_10_90_ns",
        "tail_to_total",
        "time_over_threshold_ns",
        "decay_tau_ns",
        "max_derivative",
        "baseline_rms",
    ]

    quadrants = [q for q in ["A", "B", "C", "D"] if q in df.get("quadrant", pd.Series(dtype=str)).unique()]
    if not quadrants:
        raise ValueError("No 'quadrant' column found in features CSV. Re-run TraceAnalysis.py first.")

    for quadrant in quadrants:
        qdf = df[df["quadrant"] == quadrant].copy()
        labels = [label for label in order if label in qdf["position_label"].unique()]
        if not labels:
            continue

        grouped = qdf.groupby("position_label")[features].agg(["mean", "std"])

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, feature in zip(axes, features):
            means = [grouped.loc[label, (feature, "mean")] if label in grouped.index else float("nan") for label in labels]
            stds = [grouped.loc[label, (feature, "std")] if label in grouped.index else float("nan") for label in labels]

            ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o-", capsize=3)
            ax.set_title(feature)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Quadrant {quadrant}: Feature Means ± 1σ", fontsize=14)
        output_path = results_dir / f"quicklook_feature_trends_Q{quadrant}.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
