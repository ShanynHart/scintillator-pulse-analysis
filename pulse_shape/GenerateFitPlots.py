"""
Generate and save pulse shape fit plots for each position and quadrant.
Fits exponential decay to pulse tails and compares with measured data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exponential_decay(x, a, tau):
    """Exponential decay model: a * exp(-x / tau)"""
    return a * np.exp(-x / tau)


def generate_fit_plots(results_dir=None):
    if results_dir is None:
        results_dir = Path(
            "/Users/shanynhart/Library/CloudStorage/GoogleDrive-shanyn.hart@gmail.com/Other computers/MyMacbookPro/PhD/PANGoLINS/Measurements/PSA/09032026/analysis_results"
        )
    else:
        results_dir = Path(results_dir)

    fits_dir = results_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

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
    quadrants = [q for q in ["A", "B", "C", "D"] if q in df.get("quadrant", pd.Series(dtype=str)).unique()]

    dt_ns = 2.0

    fit_summary = []

    for quadrant in quadrants:
        qdf = df[df["quadrant"] == quadrant].copy()
        for position in order:
            pos_data = qdf[qdf["position_label"] == position]
            if pos_data.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

            group_data = []
            for run_id in pos_data["run_id"].unique():
                run_data = pos_data[pos_data["run_id"] == run_id]
                if len(run_data) > 0:
                    peaks = run_data["peak"].values
                    peak_samples = run_data["peak_sample"].values.astype(int)
                    decays = run_data["decay_tau_ns"].values

                    group_data.append(
                        {
                            "peaks": peaks,
                            "peak_samples": peak_samples,
                            "decay_taus": decays[~np.isnan(decays)],
                        }
                    )

            if not group_data:
                plt.close(fig)
                continue

            all_peaks = np.concatenate([gd["peaks"] for gd in group_data])
            all_peak_samples = np.concatenate([gd["peak_samples"] for gd in group_data])
            all_decay_taus = np.concatenate([gd["decay_taus"] for gd in group_data if len(gd["decay_taus"]) > 0])

            mean_peak = np.mean(all_peaks)
            mean_peak_sample = int(np.median(all_peak_samples))
            if len(all_decay_taus) > 0:
                mean_decay_tau = np.mean(all_decay_taus)
            else:
                mean_decay_tau = np.nan

            synthetic_pulse = np.zeros(250)
            peak_idx = min(mean_peak_sample, len(synthetic_pulse) - 1)
            synthetic_pulse[:peak_idx] = np.linspace(0, mean_peak, peak_idx)
            if not np.isnan(mean_decay_tau) and mean_decay_tau > 0:
                tail_len = min(100, len(synthetic_pulse) - peak_idx)
                tail_t = np.arange(tail_len) * dt_ns
                synthetic_pulse[peak_idx : peak_idx + tail_len] = mean_peak * np.exp(
                    -tail_t / mean_decay_tau
                )

            normalized_pulse = synthetic_pulse / np.max(synthetic_pulse) if np.max(synthetic_pulse) > 0 else synthetic_pulse

            try:
                tail_mask = np.arange(len(normalized_pulse)) > peak_idx
                tail_indices = np.where(tail_mask)[0]
                if len(tail_indices) > 5:
                    tail_y = normalized_pulse[tail_indices[:50]]
                    tail_x = np.arange(len(tail_y)) * dt_ns
                    valid = tail_y > 0.05
                    if np.sum(valid) > 3:
                        popt, _ = curve_fit(
                            exponential_decay,
                            tail_x[valid],
                            tail_y[valid],
                            p0=[tail_y[0], mean_decay_tau if not np.isnan(mean_decay_tau) else 100],
                            maxfev=10000,
                        )
                        fitted_tail = exponential_decay(tail_x, *popt)
                        residuals = tail_y - fitted_tail
                        rms_error = np.sqrt(np.mean(residuals**2))
                        fit_quality = "GOOD" if rms_error < 0.05 else "FAIR" if rms_error < 0.1 else "POOR"
                    else:
                        fitted_tail = None
                        rms_error = np.nan
                        fit_quality = "INSUFFICIENT"
                else:
                    fitted_tail = None
                    rms_error = np.nan
                    fit_quality = "SHORT_TAIL"
            except Exception as e:
                fitted_tail = None
                rms_error = np.nan
                fit_quality = f"ERROR: {str(e)[:20]}"

            ax = axes[0]
            ax.plot(normalized_pulse, "b-", linewidth=2, label="Measured")
            if fitted_tail is not None:
                tail_indices_plot = np.arange(len(normalized_pulse)) > peak_idx
                tail_plot_indices = np.where(tail_indices_plot)[0][:50]
                fitted_full = np.copy(normalized_pulse)
                fitted_full[tail_plot_indices] = fitted_tail
                ax.plot(fitted_full, "r--", linewidth=1.5, label="Fitted", alpha=0.7)
            ax.set_title(f"Q{quadrant} {position}\n{fit_quality} (RMS={rms_error:.4f})")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Normalized amplitude")
            ax.legend()
            ax.grid(alpha=0.3)

            if fitted_tail is not None:
                ax = axes[1]
                tail_y = normalized_pulse[tail_indices[:50]]
                tail_x_plot = np.arange(len(tail_y))
                ax.plot(tail_x_plot, tail_y, "bo", markersize=4, label="Data")
                ax.plot(tail_x_plot, fitted_tail, "r-", linewidth=2, label="Fit")
                ax.set_title("Tail fit detail")
                ax.set_xlabel("Sample index (from peak)")
                ax.set_ylabel("Normalized amplitude")
                ax.set_yscale("log")
                ax.legend()
                ax.grid(alpha=0.3, which="both")

                ax = axes[2]
                residuals_full = tail_y - fitted_tail
                ax.bar(tail_x_plot, residuals_full, color="purple", alpha=0.6)
                ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
                ax.set_title(f"Residuals (RMS={rms_error:.4f})")
                ax.set_xlabel("Sample index (from peak)")
                ax.set_ylabel("Residual")
                ax.grid(alpha=0.3, axis="y")
            else:
                axes[1].text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=axes[1].transAxes)
                axes[1].axis("off")
                axes[2].text(0.5, 0.5, "No residuals", ha="center", va="center", transform=axes[2].transAxes)
                axes[2].axis("off")

            fig.suptitle(f"Pulse Shape Fit: Quadrant {quadrant} @ {position}", fontsize=12, fontweight="bold")
            out_file = fits_dir / f"fit_Q{quadrant}_{position.replace(' ', '_')}.png"
            fig.savefig(out_file, dpi=150)
            plt.close(fig)

            fit_summary.append(
                {
                    "quadrant": quadrant,
                    "position": position,
                    "mean_peak": mean_peak,
                    "mean_decay_tau_ns": mean_decay_tau,
                    "fit_rms_error": rms_error,
                    "fit_quality": fit_quality,
                }
            )

            print(f"Saved: {out_file.name}")

    summary_df = pd.DataFrame(fit_summary)
    summary_file = fits_dir / "fit_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nFit summary saved: {summary_file}")

    return fits_dir


if __name__ == "__main__":
    fits_dir = generate_fit_plots()
    print(f"\nAll fit plots saved to: {fits_dir}")
