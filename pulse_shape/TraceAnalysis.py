import argparse
import glob
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit


DEFAULT_RUN_POSITION_MAP = {
    "R2": "off_detector",
    "R4": "A_center",
    "R5": "A_outer_edge",
    "R6": "A_B_boundary",
    "R7": "B_center",
    "R8": "B_outer_edge",
    "R9": "C_outer_edge",
    "R10": "C_center",
    "R11": "A_B_C_D_meeting",
}

CHANNEL_TO_QUADRANT = {
    "00": "A",
    "01": "B",
    "02": "C",
    "03": "D",
}


def exponential_decay(t, amplitude, tau):
    """Single exponential decay: A * exp(-t/tau)"""
    return amplitude * np.exp(-t / tau)


def double_exponential_decay(t, a1, tau1, a2, tau2):
    """Double exponential decay: a1*exp(-t/tau1) + a2*exp(-t/tau2)"""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


class TraceProcessor:
    def __init__(
        self,
        sampling_rate_mhz=500,
        baseline_samples=60,
        pre_peak_samples=20,
        short_gate_samples=30,
        long_gate_samples=140,
        cfd_fraction=0.2,
        tot_fraction=0.1,
    ):
        self.dt_ns = 1000.0 / float(sampling_rate_mhz)
        self.baseline_samples = baseline_samples
        self.pre_peak_samples = pre_peak_samples
        self.short_gate_samples = short_gate_samples
        self.long_gate_samples = long_gate_samples
        self.cfd_fraction = cfd_fraction
        self.tot_fraction = tot_fraction

    @staticmethod
    def _linear_crossing_time_ns(y, threshold, dt_ns, i_start=0, i_stop=None):
        if i_stop is None:
            i_stop = len(y) - 1
        idx_candidates = np.where((y[i_start:i_stop] < threshold) & (y[i_start + 1 : i_stop + 1] >= threshold))[0]
        if len(idx_candidates) == 0:
            return np.nan
        idx = idx_candidates[0] + i_start
        y0 = y[idx]
        y1 = y[idx + 1]
        if np.isclose(y1, y0):
            return idx * dt_ns
        frac = (threshold - y0) / (y1 - y0)
        return (idx + frac) * dt_ns

    def _estimate_decay_params(self, trace_aligned, peak_idx, dt_ns):
        """Fit exponential and double exponential decay to tail, return both results.
        
        Returns dict with keys:
        - tau_single_ns: single exponential tau
        - tau1_double_ns, tau2_double_ns: double exponential taus
        - rms_single, rms_double: RMS fit errors
        - fit_quality: 'GOOD'/'FAIR'/'POOR'/'INSUFFICIENT'
        - fit_params: dict with curve_fit parameters for plotting
        """
        result = {
            "tau_single_ns": np.nan,
            "tau1_double_ns": np.nan,
            "tau2_double_ns": np.nan,
            "rms_single": np.nan,
            "rms_double": np.nan,
            "fit_quality": "INSUFFICIENT",
            "fit_params": {},
        }
        
        tail = trace_aligned[peak_idx:]
        if len(tail) < 10:
            return result
        
        peak = tail[0]
        if peak <= 0:
            return result
        
        # Normalize tail for cleaner fitting
        norm_tail = tail / peak
        
        # Use points from 95% down to 5% of peak for fitting (excludes peak saturation & noise floor)
        valid = (norm_tail <= 0.95) & (norm_tail >= 0.05)
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) < 6:
            return result
        
        t_fit = valid_indices * dt_ns
        y_fit = norm_tail[valid_indices]
        
        # Single exponential fit
        try:
            popt_single, _ = curve_fit(
                lambda t, a, tau: a * np.exp(-t / tau),
                t_fit,
                y_fit,
                p0=[y_fit[0], 100.0],
                maxfev=10000,
                bounds=([0.01, 1.0], [1.0, 10000.0])
            )
            y_pred_single = popt_single[0] * np.exp(-t_fit / popt_single[1])
            rms_single = np.sqrt(np.mean((y_fit - y_pred_single) ** 2))
            result["tau_single_ns"] = float(popt_single[1])
            result["rms_single"] = float(rms_single)
            result["fit_params"]["single"] = {
                "popt": popt_single,
                "peak_idx": peak_idx,
                "peak_value": float(peak),
                "t_fit": t_fit,
                "y_fit": y_fit,
                "y_pred": y_pred_single,
                "valid_indices": valid_indices,
            }
        except Exception:
            pass
        
        # Double exponential fit (only if single exponential RMS is poor)
        if np.isfinite(result["rms_single"]) and result["rms_single"] > 0.02:
            try:
                # For double exponential, assume tau1 < tau2 (fast + slow components)
                popt_double, _ = curve_fit(
                    lambda t, a1, tau1, a2, tau2: a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2),
                    t_fit,
                    y_fit,
                    p0=[y_fit[0] * 0.5, 50.0, y_fit[0] * 0.5, 200.0],
                    maxfev=10000,
                    bounds=([0.01, 1.0, 0.01, 1.0], [1.0, 500.0, 1.0, 10000.0])
                )
                tau1, tau2 = sorted([popt_double[1], popt_double[3]])
                y_pred_double = popt_double[0] * np.exp(-t_fit / popt_double[1]) + popt_double[2] * np.exp(-t_fit / popt_double[3])
                rms_double = np.sqrt(np.mean((y_fit - y_pred_double) ** 2))
                result["tau1_double_ns"] = float(min(tau1, tau2))
                result["tau2_double_ns"] = float(max(tau1, tau2))
                result["rms_double"] = float(rms_double)
                result["fit_params"]["double"] = {
                    "popt": popt_double,
                    "peak_idx": peak_idx,
                    "peak_value": float(peak),
                    "t_fit": t_fit,
                    "y_fit": y_fit,
                    "y_pred": y_pred_double,
                    "valid_indices": valid_indices,
                }
            except Exception:
                pass
        
        # Determine fit quality
        best_rms = np.nanmin([result["rms_single"], result["rms_double"]])
        if np.isfinite(best_rms):
            if best_rms < 0.02:
                result["fit_quality"] = "GOOD"
            elif best_rms < 0.05:
                result["fit_quality"] = "FAIR"
            else:
                result["fit_quality"] = "POOR"
        
        return result
    
    @staticmethod
    def _measure_baseline_robust(data, baseline_samples=60, method='median'):
        """Robustly measure baseline using early samples.
        
        Args:
            data: Full trace
            baseline_samples: Number of samples at start to use  
            method: 'mean' or 'median' (median is more robust to outliers)
        
        Returns:
            baseline value, baseline RMS
        """
        baseline_region = data[:baseline_samples]
        
        if method == 'median':
            baseline = float(np.median(baseline_region))
        else:
            baseline = float(np.mean(baseline_region))
        
        # RMS measured around baseline (robust to single outliers)
        deviations = baseline_region - baseline
        baseline_rms = float(np.std(deviations))
        
        return baseline, baseline_rms

    @staticmethod
    def _load_trace_file(file_path):
        arr = np.loadtxt(file_path)
        if arr.ndim == 0:
            return np.array([float(arr)])
        if arr.ndim == 1:
            return arr.astype(float)
        return arr[:, -1].astype(float)

    @staticmethod
    def _extract_channel_code(file_name):
        match = re.match(r"^Trace_\d+_\d+_(\d+)-\d+\.dat$", file_name)
        if not match:
            return None
        return match.group(1).zfill(2)

    def process_trace(self, data):
        if data is None or len(data) < max(self.baseline_samples + 5, self.long_gate_samples + 5):
            return None

        # Use robust baseline measurement
        baseline, baseline_rms = self._measure_baseline_robust(data, self.baseline_samples, method='median')
        trace = data - baseline

        max_val = float(np.max(trace))
        min_val = float(np.min(trace))
        polarity = 1.0 if abs(max_val) >= abs(min_val) else -1.0
        trace_aligned = polarity * trace

        peak_idx = int(np.argmax(trace_aligned))
        peak_val = float(trace_aligned[peak_idx])
        if peak_val <= 0:
            return None

        norm_trace = trace_aligned / peak_val
        derivative = np.diff(trace_aligned) / self.dt_ns

        t10 = self._linear_crossing_time_ns(norm_trace, 0.1, self.dt_ns, i_start=0, i_stop=max(peak_idx, 1))
        t90 = self._linear_crossing_time_ns(norm_trace, 0.9, self.dt_ns, i_start=0, i_stop=max(peak_idx, 1))
        rise_time_10_90_ns = float(t90 - t10) if np.isfinite(t10) and np.isfinite(t90) and t90 > t10 else np.nan

        half_threshold = 0.5
        above_half = np.where(norm_trace >= half_threshold)[0]
        if len(above_half) >= 2:
            fwhm_ns = float((above_half[-1] - above_half[0]) * self.dt_ns)
        else:
            fwhm_ns = np.nan

        start_idx = max(0, peak_idx - self.pre_peak_samples)
        short_end_idx = min(len(trace_aligned), peak_idx + self.short_gate_samples)
        long_end_idx = min(len(trace_aligned), peak_idx + self.long_gate_samples)

        q_total = float(trapezoid(trace_aligned[start_idx:long_end_idx], dx=self.dt_ns))
        q_short = float(trapezoid(trace_aligned[start_idx:short_end_idx], dx=self.dt_ns))
        q_long = q_total
        q_tail = q_long - q_short

        tail_to_total = (q_tail / q_long) if q_long > 0 else np.nan
        short_to_long = (q_short / q_long) if q_long > 0 else np.nan

        cfd_time_ns = self._linear_crossing_time_ns(
            norm_trace,
            self.cfd_fraction,
            self.dt_ns,
            i_start=0,
            i_stop=max(peak_idx, 1),
        )

        tot_thr = self.tot_fraction
        above_tot = np.where(norm_trace >= tot_thr)[0]
        if len(above_tot) >= 2:
            time_over_threshold_ns = float((above_tot[-1] - above_tot[0]) * self.dt_ns)
        else:
            time_over_threshold_ns = np.nan

        # Fit exponential decay(s) to tail
        decay_results = self._estimate_decay_params(trace_aligned, peak_idx, self.dt_ns)
        max_derivative = float(np.max(derivative)) if len(derivative) > 0 else np.nan
        
        # Store fit_params separately for plotting (not in main results dataframe)
        fit_params = decay_results.pop("fit_params", {})

        return {
            "baseline": baseline,
            "baseline_rms": baseline_rms,
            "polarity": int(polarity),
            "peak": peak_val,
            "peak_sample": peak_idx,
            "integral_total": q_total,
            "integral_short": q_short,
            "integral_tail": q_tail,
            "tail_to_total": tail_to_total,
            "short_to_long": short_to_long,
            "rise_time_10_90_ns": rise_time_10_90_ns,
            "fwhm_ns": fwhm_ns,
            "decay_tau_single_ns": decay_results["tau_single_ns"],
            "decay_tau1_double_ns": decay_results["tau1_double_ns"],
            "decay_tau2_double_ns": decay_results["tau2_double_ns"],
            "decay_rms_single": decay_results["rms_single"],
            "decay_rms_double": decay_results["rms_double"],
            "decay_fit_quality": decay_results["fit_quality"],
            "cfd_time_ns": cfd_time_ns,
            "time_over_threshold_ns": time_over_threshold_ns,
            "max_derivative": max_derivative,
            "norm_trace": norm_trace,
            "_fit_params": fit_params,  # Internal use for plotting
            "_raw_trace": trace_aligned,  # Internal use for plotting
        }

    def process_folder(self, folder_path, save_trace_plots=False, plot_output_dir=None):
        files = sorted(glob.glob(os.path.join(folder_path, "*.dat")))
        results = []
        traces_by_quadrant = {q: [] for q in CHANNEL_TO_QUADRANT.values()}

        for file_path in files:
            file_name = os.path.basename(file_path)
            channel_code = self._extract_channel_code(file_name)
            if channel_code is None:
                continue
            quadrant = CHANNEL_TO_QUADRANT.get(channel_code)
            if quadrant is None:
                continue

            try:
                data = self._load_trace_file(file_path)
            except Exception:
                continue

            res = self.process_trace(data)
            if res is None:
                continue

            res["trace_file"] = file_name
            res["channel_code"] = channel_code
            res["quadrant"] = quadrant
            
            # Save individual trace plots if requested
            if save_trace_plots and plot_output_dir:
                plot_name = file_name.replace(".dat", "_fit.png")
                plot_path = os.path.join(plot_output_dir, plot_name)
                plot_trace_with_fits(res, file_name, plot_path)
            
            # Remove internal plotting data from results before storing (to save memory)
            res_clean = {k: v for k, v in res.items() if not k.startswith("_")}
            results.append(res_clean)
            
            traces_by_quadrant[quadrant].append(res["norm_trace"])

        if not results:
            return pd.DataFrame(), {}

        avg_pulses_by_quadrant = {}
        for quadrant, traces in traces_by_quadrant.items():
            if not traces:
                continue
            min_len = min(len(t) for t in traces)
            trimmed_traces = np.array([t[:min_len] for t in traces])
            avg_pulses_by_quadrant[quadrant] = np.mean(trimmed_traces, axis=0)

        df = pd.DataFrame(results).drop(columns=["norm_trace"])
        return df, avg_pulses_by_quadrant


def build_run_map(base_dir, mapping_csv=None):
    if mapping_csv is None:
        run_map = []
        for run_id, label in DEFAULT_RUN_POSITION_MAP.items():
            run_map.append(
                {
                    "run_id": run_id,
                    "position_label": label,
                    "folder": os.path.join(base_dir, run_id),
                }
            )
        return pd.DataFrame(run_map)

    mapping_df = pd.read_csv(mapping_csv)
    required = {"run_id", "position_label", "folder"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Missing columns in mapping CSV: {sorted(missing)}")

    mapping_df = mapping_df.copy()
    mapping_df["folder"] = mapping_df["folder"].apply(
        lambda p: p if os.path.isabs(str(p)) else os.path.join(base_dir, str(p))
    )
    return mapping_df


def save_group_summary(features_df, output_path, group_cols):
    numeric_cols = [
        c
        for c in features_df.columns
        if c
        not in {
            "run_id",
            "position_label",
            "trace_file",
            "channel_code",
            "quadrant",
        }
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    summary = features_df.groupby(group_cols)[numeric_cols].agg(["count", "mean", "std", "median"])
    summary.to_csv(output_path)


def plot_average_pulses(avg_pulse_dict, output_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found, skipping average pulse plot.")
        return

    plt.figure(figsize=(10, 6))
    for label, pulse in avg_pulse_dict.items():
        if pulse is not None:
            plt.plot(pulse, linewidth=1.8, label=label)
    plt.title("Average normalized pulse by position")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized amplitude")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_trace_with_fits(result, file_name, output_path):
    """Plot individual trace with fitted exponential curves.
    
    Args:
        result: dict from process_trace with _fit_params and _raw_trace
        file_name: base filename for identification
        output_path: where to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    if "_fit_params" not in result or not result["_fit_params"]:
        return
    
    fit_params = result["_fit_params"]
    raw_trace = result.get("_raw_trace")
    peak_idx = result.get("peak_sample", 0)
    peak_value = result.get("peak", 1.0)
    
    if raw_trace is None or len(raw_trace) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # Plot 1: Full normalized trace with fits
    ax = axes[0, 0]
    norm_trace = raw_trace / peak_value if peak_value > 0 else raw_trace
    ax.plot(norm_trace, "b-", linewidth=2, label="Measured", alpha=0.7)
    
    if "single" in fit_params:
        params = fit_params["single"]
        valid_indices = params["valid_indices"]
        y_pred = params["y_pred"]
        ax.plot(peak_idx + valid_indices, y_pred, "r--", linewidth=2, label="Single exp fit")
    
    if "double" in fit_params:
        params = fit_params["double"]
        valid_indices = params["valid_indices"]
        y_pred = params["y_pred"]
        ax.plot(peak_idx + valid_indices, y_pred, "g--", linewidth=2, label="Double exp fit")
    
    ax.axvline(peak_idx, color="k", linestyle=":", alpha=0.5, label="Peak")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"Full Trace: {file_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Tail region with log scale
    ax = axes[0, 1]
    if "single" in fit_params:
        params = fit_params["single"]
        t_fit = params["t_fit"]
        y_fit = params["y_fit"]
        y_pred = params["y_pred"]
        ax.semilogy(t_fit, y_fit, "bo", markersize=5, label="Data")
        ax.semilogy(t_fit, y_pred, "r-", linewidth=2, label="Single exp")
        ax.set_xlabel("Time from peak (ns)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(f"Single Exponential Fit (τ={result['decay_tau_single_ns']:.1f} ns)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    
    # Plot 3: Double exponential fit (if available)
    ax = axes[1, 0]
    if "double" in fit_params:
        params = fit_params["double"]
        t_fit = params["t_fit"]
        y_fit = params["y_fit"]
        y_pred = params["y_pred"]
        ax.semilogy(t_fit, y_fit, "bo", markersize=5, label="Data")
        ax.semilogy(t_fit, y_pred, "g-", linewidth=2, label="Double exp")
        ax.set_xlabel("Time from peak (ns)")
        ax.set_ylabel("Normalized amplitude")
        tau1 = result['decay_tau1_double_ns']
        tau2 = result['decay_tau2_double_ns']
        ax.set_title(f"Double Exponential Fit (τ₁={tau1:.1f} ns, τ₂={tau2:.1f} ns)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    
    # Plot 4: Fit quality summary
    ax = axes[1, 1]
    ax.axis("off")
    
    quality_text = f"""
    Fit Quality Summary
    ═══════════════════
    Quality: {result.get('decay_fit_quality', 'N/A')}
    
    Single Exponential:
      τ = {result['decay_tau_single_ns']:.2f} ns
      RMS Error = {result['decay_rms_single']:.4f}
    
    Double Exponential:
      τ₁ = {result['decay_tau1_double_ns']:.2f} ns
      τ₂ = {result['decay_tau2_double_ns']:.2f} ns
      RMS Error = {result['decay_rms_double']:.4f}
    
    Trace Info:
      Peak = {result['peak']:.1f}
      Peak Sample = {result['peak_sample']}
      Rise Time (10-90%) = {result['rise_time_10_90_ns']:.2f} ns
      Tail/Total = {result['tail_to_total']:.3f}
    """
    
    ax.text(0.05, 0.95, quality_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.suptitle(f"Exponential Decay Fit Analysis", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_box(features_df, feature_name, output_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not found, skipping plot for {feature_name}.")
        return

    plot_df = features_df[["position_label", feature_name]].dropna()
    if plot_df.empty:
        return
    ordered_labels = sorted(plot_df["position_label"].unique())
    data = [plot_df.loc[plot_df["position_label"] == lbl, feature_name].values for lbl in ordered_labels]

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=ordered_labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(feature_name)
    plt.title(f"{feature_name} by position")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Trace-based pulse shape analysis for DASSIE LaBr3Ce runs.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Base directory containing run folders (R2, R4, ...).",
    )
    parser.add_argument(
        "--mapping-csv",
        default=None,
        help="Optional CSV with columns: run_id, position_label, folder",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for CSV and plots. Default: <base-dir>/analysis_results",
    )
    parser.add_argument("--sampling-rate-mhz", type=float, default=500.0)
    parser.add_argument(
        "--save-trace-plots",
        action="store_true",
        help="Save individual trace fit plots (many files, slower).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    output_dir = args.output_dir or os.path.join(base_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectory for trace plots if requested
    trace_plots_dir = None
    if args.save_trace_plots:
        trace_plots_dir = os.path.join(output_dir, "trace_fits")
        os.makedirs(trace_plots_dir, exist_ok=True)

    processor = TraceProcessor(sampling_rate_mhz=args.sampling_rate_mhz)
    run_map = build_run_map(base_dir=base_dir, mapping_csv=args.mapping_csv)

    all_features = []
    avg_pulses = {}

    for _, row in run_map.iterrows():
        run_id = str(row["run_id"])
        position_label = str(row["position_label"])
        folder = str(row["folder"])

        if not os.path.isdir(folder):
            print(f"Skipping {run_id}: folder does not exist -> {folder}")
            continue

        df, avg_pulses_by_quadrant = processor.process_folder(
            folder,
            save_trace_plots=args.save_trace_plots,
            plot_output_dir=trace_plots_dir
        )
        if df.empty:
            print(f"Skipping {run_id}: no valid traces in {folder}")
            continue

        df.insert(0, "position_label", position_label)
        df.insert(0, "run_id", run_id)
        all_features.append(df)
        for quadrant, avg_pulse in avg_pulses_by_quadrant.items():
            avg_pulses[f"{run_id}:{position_label}:Q{quadrant}"] = avg_pulse

        grouped = df.groupby("quadrant")
        summary_bits = []
        for quadrant in sorted(grouped.groups):
            qdf = grouped.get_group(quadrant)
            summary_bits.append(
                f"Q{quadrant} n={len(qdf)} rise={qdf['rise_time_10_90_ns'].mean():.1f}ns tail/total={qdf['tail_to_total'].mean():.4f}"
            )
        print(f"{run_id} ({position_label}) -> " + " | ".join(summary_bits))

    if not all_features:
        print("No valid traces processed. Check input folders and file format.")
        return

    features_df = pd.concat(all_features, ignore_index=True)
    features_csv = os.path.join(output_dir, "features_all_traces.csv")
    summary_csv = os.path.join(output_dir, "summary_by_position.csv")
    summary_pos_quad_csv = os.path.join(output_dir, "summary_by_position_and_quadrant.csv")

    features_df.to_csv(features_csv, index=False)
    save_group_summary(features_df, summary_csv, group_cols=["position_label"])
    save_group_summary(features_df, summary_pos_quad_csv, group_cols=["position_label", "quadrant"])

    plot_average_pulses(avg_pulses, os.path.join(output_dir, "avg_pulses_by_position.png"))
    for feature_name in [
        "rise_time_10_90_ns",
        "tail_to_total",
        "time_over_threshold_ns",
        "decay_tau_single_ns",
        "decay_tau1_double_ns",
        "decay_tau2_double_ns",
        "decay_rms_single",
        "max_derivative",
        "baseline_rms",
    ]:
        for quadrant in sorted(features_df["quadrant"].dropna().unique()):
            qdf = features_df[features_df["quadrant"] == quadrant]
            plot_feature_box(qdf, feature_name, os.path.join(output_dir, f"box_{feature_name}_Q{quadrant}.png"))

    print("\nSaved outputs:")
    print(f"- {features_csv}")
    print(f"- {summary_csv}")
    print(f"- {summary_pos_quad_csv}")
    print(f"- {os.path.join(output_dir, 'avg_pulses_by_position.png')}")
    if args.save_trace_plots:
        print(f"- Trace fit plots in: {trace_plots_dir}")

    print("\nSaved outputs:")
    print(f"- {features_csv}")
    print(f"- {summary_csv}")
    print(f"- {summary_pos_quad_csv}")
    print(f"- {os.path.join(output_dir, 'avg_pulses_by_position.png')}")


if __name__ == "__main__":
    main()