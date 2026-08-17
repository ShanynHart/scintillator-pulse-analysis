import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path


if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

filename = sys.argv[1]
data = np.loadtxt(filename)
# extract the name of the file without extension
name = Path(filename).stem + ".png"
# create a directory for the output if it doesn't exist
save_dir = Path(filename)

# 1 GSa/s sampling rate = 1 ns between samples
sampling_rate = 1e9  # 1 GHz
time_step = 1 / sampling_rate  # 1 ns
time = np.arange(len(data)) * time_step  # Time axis in seconds

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time * 1e5, data, label="Waveform")  # time in µs for readability
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.title("Fast Signal Waveform")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir.parent / name, dpi=300)
plt.show()
