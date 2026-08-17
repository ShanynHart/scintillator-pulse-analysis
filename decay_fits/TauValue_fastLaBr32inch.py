import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import curve_fit
import os 
import sys
from lmfit import Model

plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
# axis labels bold
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.edgecolor"] = 'white'
plt.rcParams["legend.handlelength"] = 2
plt.rcParams["legend.handleheight"] = 0
plt.rcParams["legend.handletextpad"] = 0
plt.rcParams["legend.markerscale"] = 1

# Define the exponential decay function
def exp_decay(x, A, T):
    return A * np.exp(-(x / T))

# Define the directory containing the .dat files as the first argument
data_directory = sys.argv[1]
output_directory = data_directory + '/plots'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_histogram_file = os.path.join(output_directory, 'T_values_histogram.png')
output_txt_file = os.path.join(output_directory, 'T_values.txt')

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize a list to store T values for the histogram
T_values = []
""" 
# Get all .dat files in the directory
dat_files = glob.glob(os.path.join(data_directory, '*.dat'))

plt.rcParams.update({'font.size': 22})
# make axis labels bold
plt.rcParams["axes.labelweight"] = "bold"
# label size
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.rcParams['figure.figsize'] = [30, 10]

while len(T_values) < 10000:
    for file_path in dat_files:
        #print every 10000 T_values
        if len(T_values) % 1000 == 0:
            print(f"Total T_values collected: {len(T_values)}")
        # Read the data
        data = pd.read_csv(file_path, delimiter=' ', header=None, names=['Time', 'Amplitude'])
        data['Time'] = data['Time']*2
        data['Amplitude'] = data['Amplitude']*-1
        
        # Fit a first-order polynomial (constant) to the y range 1700 to 1900
        mask = (data['Time'] > 0) & (data['Time'] < 1000)
        p = np.polyfit(data['Time'][mask], data['Amplitude'][mask], 0)
        constant_offset = np.polyval(p, data['Time'])

        # Save a plot of the raw data from 0 to 500 with the constant offset
        plt.figure()
        plt.plot(data['Time'], data['Amplitude'], label='Raw Data', linewidth=2)
        plt.plot(data['Time'], constant_offset, 'r-', label='Constant Offset', linewidth=2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (a.u.)')
        plt.legend(loc='upper right', fontsize= 22)
        plt.grid( linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xlim([0, 500])
        plt.ylim([-1650, -1750])
        plt.savefig(os.path.join(output_directory, f'{os.path.basename(file_path)}_raw_data.png'))  
        plt.close() 

        # plot the trace full length before the exponential fit
        plt.figure()
        plt.plot(data['Time'], data['Amplitude'], label='Raw Data', linewidth=2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (a.u.)')
        plt.grid( linestyle='--', linewidth=0.5, alpha=0.5)
        plt.savefig(os.path.join(output_directory, f'{os.path.basename(file_path)}_raw_data_full.png'))
        plt.close()

        # Subtract the constant offset from the data
        data['Adjusted_Amplitude'] = data['Amplitude'] - constant_offset

        positive_mask = data['Adjusted_Amplitude'] > 5
        if positive_mask.any():
            data['Adjusted_Amplitude'] = data['Adjusted_Amplitude']*-1
            
            # Select data points after the max amplitude time for fitting
            max_amplitude = data['Adjusted_Amplitude'].max()
            max_time = data.loc[data['Adjusted_Amplitude'] == max_amplitude, 'Time'].values[0]

            exp_fit_data = data.loc[(data['Time'] >= max_time) & (data['Time'] <= max_time + 500)]
            
            # Filter out data points where the amplitude is very low
            exp_fit_data = exp_fit_data[exp_fit_data['Adjusted_Amplitude'] > 1]

            # Ensure there are enough data points for fitting
            if len(exp_fit_data) < 5:
                print(f"Not enough data points for fitting in file {file_path}")
                T_values.append(np.nan)
                continue
            
            # Dynamically set initial parameters
            initial_A = max_amplitude
            

            popt, pcov = curve_fit(exp_decay, exp_fit_data['Time'], exp_fit_data['Adjusted_Amplitude'], p0=[initial_A, 10000], maxfev=10000, bounds=([0, 0], [np.inf, np.inf])) #bounds are set to avoid negative values
            A, T = popt
            A_error, T_error = np.sqrt(np.diag(pcov))
            T = T/1000
            T_error = T_error/1000

            # Extract the amplitude and time constant
            A, T = popt
            A_error, T_error = np.sqrt(np.diag(pcov))
            T = T/1000
            T_error = T_error/1000

            # Save a plot of the exponential fit
            plt.figure()
            plt.plot(data['Time'], data['Adjusted_Amplitude'], label='Adjusted Data')
            plt.plot(exp_fit_data['Time'], exp_decay(exp_fit_data['Time'], A, T*1000), 'r-', label='y = A * exp(-x / \u03C4)')
            plt.axvline(max_time, color='g', linestyle='--', label='Max Amplitude Time', linewidth=2)
            plt.xlabel('Time (ns)')
            plt.xlim([1000, 1500])
            plt.ylabel('Amplitude (a.u.)', fontsize=20)
            plt.legend(loc='upper left')
            T = T*1000
            T_error = T_error*10000000
            textstr = f'$\\tau={T:.2f}({T_error:.0f})$ $ns$'
            plt.text(0.05, 0.75, textstr, fontsize=22, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
            plt.grid( linestyle='--', linewidth=0.5, alpha=0.5)
            plt.savefig(os.path.join(output_directory, f'{os.path.basename(file_path)}_exp_fit.png'))
            plt.close() 
            
            T_values.append(T)
            print(f'Fit successful for file {file_path}: T = {T:.4f} \u00B1 {T_error:.0f} ns')
            # Check if the required number of T_values is reached
            if len(T_values) >= 10000:
                break
        else:
            # No positive amplitude found, store T as 0
            T_values.append(np.nan)

        

    # Remove NaN values
    T_values = [x for x in T_values if not np.isnan(x)]
    print(f"Total T_values collected: {len(T_values)}")

# Calculate mean and standard deviation
mean = np.mean(T_values).astype(dtype=np.float64)
std = np.std(T_values).astype(dtype=np.float64)
value = (mean - 2*std) - (mean + 2*std)
value = abs(value)
value = int(value)
value = value
 """
""" 
# Append the T values to the text file
with open(output_txt_file, 'w') as f:
    for T in T_values:
        f.write(f'{T}\n')

        f.write(f'{T}\n') """

# read in T values from the text file
T_values = []
with open(output_txt_file, 'r') as f:
    for line in f:
        T_values.append(float(line.strip())*1000)

# Plot the histogram of T values
# set 100 ns bins (the range of the histogram is 0 to 100 μs)
value = np.linspace(0, 50, 51)

# Create the histogram
counts, bin_edges = np.histogram(T_values, bins=value)

# Calculate the midpoints of the bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the line using bin centers and counts
plt.figure()
#plt.plot(bin_centers, counts, color='blue', linewidth=2)
plt.hist(T_values, bins=value, color='blue', edgecolor='black', linewidth=1.2)

# Overlay mean and standard deviation as text on the plot
most_freq = np.argmax(counts)
mean_most_freq_values = bin_centers[most_freq]
Err = np.std(T_values)

print(Err)


plt.ylabel('Counts (1/ns)', fontsize=20)
plt.xlabel('τ (ns)', fontsize=20)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.5)
# draw a vertical line at the most frequent value as a dotted red line
plt.axvline(mean_most_freq_values, color='red', linestyle='--', linewidth=2)
textstr = f'$τ={mean_most_freq_values:.2f}({Err:.0f})$ $ns$'
plt.text(0.95, 0.95, textstr, fontsize=22, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', linewidth=0))
# plot a short horizontal red dotted line next to the text as a guide

plt.savefig(output_histogram_file)
plt.show()

