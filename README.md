# Scintillator pulse analysis

Digital signal processing for gamma-ray detector waveforms, written during my PhD and postdoc work on LaBr3:Ce and CLYC scintillator detectors (University of Cape Town / iThemba LABS).

The problem: a scintillation detector produces a few-hundred-nanosecond current pulse for every gamma-ray interaction. Everything you want to know (deposited energy, arrival time, particle type) has to be extracted from that noisy, digitised trace, at rates of thousands of pulses per second.

## What is here

**`pulse_shape/`** is the core. `TraceAnalysis.py` processes raw 500 MHz digitiser traces:

- baseline estimation and subtraction from the pre-trigger region
- pulse-onset timing by constant-fraction discrimination (interpolated zero-crossing, so timing resolution beats the sampling period)
- energy by gated charge integration, with short-gate/long-gate charge ratios for pulse-shape discrimination between particle types
- decay-time characterisation by single and double exponential fits (`scipy.optimize.curve_fit`), with fit quality controls

`calib_09032026.py` converts integrated charge to energy against known gamma lines. `GenerateFitPlots.py` and `PlotResults.py` produce the diagnostic figures. `sort1labr.C` is the ROOT/C++ equivalent used for larger datasets.

**`waveform_tools/`** reads raw instrument formats: Rohde & Schwarz oscilloscope exports and Red Pitaya histogram dumps.

**`decay_fits/`** extracts scintillator decay constants (tau) from averaged pulses for two detector assemblies, the physics input behind the gate choices above.

**`sample_data/`** holds one real oscilloscope waveform export so the trace tools run out of the box.

## Methods, in general terms

Feature extraction from noisy time series, sub-sample interpolation, matched gating, nonlinear least squares with uncertainty estimates, and batch processing over run directories. The same toolkit applies anywhere a continuous signal has to be reduced to a few numbers per event.

## Author

Shanyn Hart. All code in this repository is my own.
