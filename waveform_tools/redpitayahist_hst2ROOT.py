 #!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Author: Shanyn Hart
# Date: 2025-07-23
# Description: Script to process and plot RedPitaya MCA .dat files using matplotlib and ROOT.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import sys
import ROOT
from pathlib import Path
import random
import numpy as np

########################
#HOW TO RUN: ~/miniconda/envs/my_root_env/bin/python redpitayahist_hst2ROOT.py ../../PANGoLINS/Measurements/DetectorAssemblies/SrI2/sn230912-04/08052024/
# This script reads in a .dat file from the RedPitaya MCA and plots the data using matplotlib and ROOT.
# it finds all the Red Pitaya .dat files in the directory and processes them
# ______________________________________________________________________________
# Read in data
dir_path = sys.argv[1]
save_dir = Path(dir_path )

if save_dir.exists():
    print('\n')
else:
    save_dir.mkdir(parents=True, exist_ok=False)
    print('\n\nThe directory {} was created.'.format(save_dir))

# Process all files in the directory
for file_path in Path(dir_path).glob('*.hst'):
    print(f'Processing file: {file_path.name}')

    # ______________________________________________________________________________
    df = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(float(line.strip()))      
        df.append(data)

    df = pd.DataFrame(df)
    df = df.transpose()
    df.columns = ['Channel']
    # df = df.loc[(df['Channel'] > 0)]
    df.reset_index(drop=True)

    # ______________________________________________________________________________
    # Calibration
    rnd = random.Random()

    p0=0.2993 # 137Cs_LaBr3Ce_sn250219-5_5min_30V SA
    p1=0 #137Cs_LaBr3Ce_sn250219-5_5min_30V SA
    """ p0=0.3252 #137Cs_LaBr3Ce_sn250219-04_5min_30V DL
    p1=0#137Cs_LaBr3Ce_sn250219-04_5min_30V DL """

    df['ChannelNumber'] = np.arange(len(df))
    df['CalibratedEnergy'] = p0 * df['ChannelNumber']
    df['CalibratedEnergy'] += [rnd.gauss(0, 0.05) for _ in range(len(df))]

    # ______________________________________________________________________________
    # Plotting with matplotlib
    name = file_path.stem
    fig = plt.figure(figsize=(10, 8))
    plt.plot(df['Channel'], color='blue', label=name, linewidth=2)
    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('Energy (keV)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 4095)
    plt.savefig(str(save_dir / (name + '.png')))

    # ______________________________________________________________________________
    # Plotting with ROOT
    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetGrid()
    hist = ROOT.TH1D("hist", "hist", 4095, 0, 4095)
    for i, value in enumerate(df['Channel']):
        hist.SetBinContent(i + 1, value)
    hist.Draw()
    hist.GetXaxis().SetTitle("Channel (keV)")
    hist.GetYaxis().SetTitle("Counts (a.u.)")
    c.Update()

    root_file = ROOT.TFile(str(save_dir / (name + '.root')), "RECREATE")
    hist.Write("hist")  
    c.Write("c")        
    root_file.Close()

print("Processing complete.")

