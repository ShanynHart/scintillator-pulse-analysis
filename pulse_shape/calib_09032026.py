import pandas as pd
import sys
import ROOT as ROOT
from pathlib import Path

########################
#HOW TO RUN: /opt/homebrew/bin/python3.11 calib_09032026.py /Users/shanynhart/Library/CloudStorage/GoogleDrive-shanyn.hart@gmail.com/Other computers/MyMacbookPro/PhD/PANGoLINS/Measurements/PSA/09032026/MARCH/RXX.root
# ______________________________________________________________________________
# Read in data
dir_path = sys.argv[1]
dir_path = dir_path[:-len(dir_path.split('/')[-1])]
calib_dir = Path(dir_path + 'calibrated/')
residuals_dir = Path(dir_path + 'residuals/')

if calib_dir.exists():
    print('\nThe directory {} already exists.'.format(calib_dir))
else:
    calib_dir.mkdir(parents=True, exist_ok=False)
    print('\nThe directory {} was created.'.format(calib_dir))

if residuals_dir.exists():
    print('\nThe directory {} already exists.'.format(residuals_dir))
else:
    residuals_dir.mkdir(parents=True, exist_ok=False)
    print('\nThe directory {} was created.'.format(residuals_dir))

file_path = Path(sys.argv[1])
name = file_path.stem
r = ROOT.TRandom3(1)

filename = file_path.parts[-1]
Rvalue = file_path.parts[-1].split('.')[0]

# print the folder name
print(' ')
print('Filename: {}'.format(filename))
print(' ')

dfslowEL0 = []
dfslowEL1 = []
dfslowEL2 = []
dfslowEL3 = []


file = dir_path + filename
f = ROOT.TFile(file)
t = f.Get("LaBrData")
#print the number of entries in the tree
for i in range(t.GetEntries()):
    t.GetEntry(i)
    dfslowEL0.append(t.slowEL0)
    dfslowEL1.append(t.slowEL1)
    dfslowEL2.append(t.slowEL2)
    dfslowEL3.append(t.slowEL3)
f.Close()


# create a dataframe with the data
df = pd.DataFrame({'slowEL0': dfslowEL0, 'slowEL1': dfslowEL1, 'slowEL2': dfslowEL2, 'slowEL3': dfslowEL3})
df = df.dropna()
df = df[(df['slowEL0'] > 0) | (df['slowEL1'] > 0) | (df['slowEL2'] > 0) | (df['slowEL3'] > 0)]
df = df[(df['slowEL0'] < 16350) | (df['slowEL1'] < 16350) | (df['slowEL2'] < 16350) | (df['slowEL3'] < 16350)]
df = df.reset_index(drop=True)

# ______________________________________________________________________________ plot ROOT histogram for channels
h_slowEL0 = ROOT.TH1D("h_slowEL0", "h_slowEL0", 16350, 0, 16350)
for i in range(len(df)):
    h_slowEL0.Fill(df['slowEL0'][i])
h_slowEL0.Draw()
h_slowEL0.GetXaxis().SetTitle("Channel")
h_slowEL0.GetYaxis().SetTitle("Counts (a.u.)")
h_slowEL0.SetLineColor(1) # red
h_slowEL0.SetLineWidth(1)
h_slowEL0.SetStats(0)
h_slowEL0.SaveAs(str(dir_path + 'h_slowEL0.root'))


h_slowEL1 = ROOT.TH1D("h_slowEL1", "h_slowEL1", 16350, 0, 16350)
for i in range(len(df)):
    h_slowEL1.Fill(df['slowEL1'][i])
h_slowEL1.Draw()
h_slowEL1.GetXaxis().SetTitle("Channel")
h_slowEL1.GetYaxis().SetTitle("Counts (a.u.)")
h_slowEL1.SetLineColor(1) # red
h_slowEL1.SetLineWidth(1)
h_slowEL1.SetStats(0)
h_slowEL1.SaveAs(str(dir_path + 'h_slowEL1.root'))

h_slowEL2 = ROOT.TH1D("h_slowEL2", "h_slowEL2", 16350, 0, 16350)
for i in range(len(df)):
    h_slowEL2.Fill(df['slowEL2'][i])
h_slowEL2.Draw()
h_slowEL2.GetXaxis().SetTitle("Channel")
h_slowEL2.GetYaxis().SetTitle("Counts (a.u.)")
h_slowEL2.SetLineColor(1) # red
h_slowEL2.SetLineWidth(1)
h_slowEL2.SetStats(0)
h_slowEL2.SaveAs(str(dir_path + 'h_slowEL2.root'))

h_slowEL3 = ROOT.TH1D("h_slowEL3", "h_slowEL3", 16350, 0, 16350)
for i in range(len(df)):
    h_slowEL3.Fill(df['slowEL3'][i])
h_slowEL3.Draw()
h_slowEL3.GetXaxis().SetTitle("Channel")
h_slowEL3.GetYaxis().SetTitle("Counts (a.u.)")
h_slowEL3.SetLineColor(1) # red
h_slowEL3.SetLineWidth(1)
h_slowEL3.SetStats(0)
h_slowEL3.SaveAs(str(dir_path + 'h_slowEL3.root'))

# ______________________________________________________________________________ peaks (linear calibration only)
energy = [0.0, 661.7]
channel_peaks = {
    'slowEL0': [0.0, 9097.83],
    'slowEL1': [0.0, 9095.98],
    'slowEL2': [0.0, 9111.18],
    'slowEL3': [0.0, 9309.11],
}

linear_params = {}

for channel_name, channel_points in channel_peaks.items():
    x0, x1 = channel_points
    y0, y1 = energy

    if x1 == x0:
        raise ValueError(f'Invalid peak points for {channel_name}: identical channel values.')

    p1 = (y1 - y0) / (x1 - x0)
    p0 = y0 - p1 * x0
    linear_params[channel_name] = {'p0': p0, 'p1': p1}

    residuals = []
    for i in range(len(energy)):
        residuals.append(energy[i] - (p0 + p1 * channel_points[i]))

    with open(str(residuals_dir / (name + f'_lin_{channel_name}_residuals.txt')), 'w') as f:
        for item in residuals:
            f.write(f'{item}\n')

    residuals_canvas = ROOT.TCanvas(f'c_{channel_name}', f'c_{channel_name}', 800, 600)
    residuals_canvas.SetGrid()
    h_res = ROOT.TH2D(
        f'h_res_{channel_name}',
        f'h_res_{channel_name}',
        2000,
        0,
        2000,
        50,
        -25,
        25,
    )
    for i in range(len(energy)):
        h_res.Fill(energy[i], residuals[i])
    h_res.Draw()
    h_res.GetXaxis().SetTitle('Energy (keV)')
    h_res.GetYaxis().SetTitle('Residuals (keV)')
    h_res.SetStats(0)
    h_res.SetMarkerStyle(20)
    h_res.SetMarkerSize(2)
    residuals_canvas.Update()
    residuals_canvas.SaveAs(str(residuals_dir / (name + f'_lin_{channel_name}_residuals.root')))
    residuals_canvas.SaveAs(str(residuals_dir / (name + f'_lin_{channel_name}_residuals.png')))
    residuals_canvas.Close()

    print('Residuals saved to {}'.format(residuals_dir / (name + f'_lin_{channel_name}_residuals.txt')))

# ______________________________________________________________________________ text file of fit parameters
with open(str(residuals_dir / (name + '_bestfit_params.txt')), 'w') as f:
    f.write('Best fit: linear\n')
    f.write('y = p0 + p1*x\n')
    for channel_name in ['slowEL0', 'slowEL1', 'slowEL2', 'slowEL3']:
        p0 = linear_params[channel_name]['p0']
        p1 = linear_params[channel_name]['p1']
        f.write(f'{channel_name}: p1 = {p1}, p0 = {p0}\n')

print('Fit parameters saved to {}'.format(residuals_dir / (name + '_bestfit_params.txt')))

# ______________________________________________________________________________ calibration
# calibrate the data and update the open root file
calib = ROOT.TFile(file, 'UPDATE')
t = calib.Get('LaBrData')

calib_hists = {}
for channel_name in ['slowEL0', 'slowEL1', 'slowEL2', 'slowEL3']:
    calib_hists[channel_name] = ROOT.TH1D(
        f'{channel_name}calib_lin',
        f'{channel_name}calib_lin',
        4000,
        0,
        4000,
    )

for i in range(t.GetEntries()):
    t.GetEntry(i)
    for channel_name in ['slowEL0', 'slowEL1', 'slowEL2', 'slowEL3']:
        raw_value = getattr(t, channel_name)
        if raw_value <= 0 or raw_value >= 16350:
            continue

        p0 = linear_params[channel_name]['p0']
        p1 = linear_params[channel_name]['p1']
        scale = channel_peaks[channel_name][1] / energy[1]

        if scale > 1:
            half_bin = int(scale) / 2
            calibrated_value = (p0 + p1 * raw_value) + r.Uniform(-half_bin, half_bin)
        else:
            calibrated_value = p0 + p1 * raw_value

        calib_hists[channel_name].Fill(calibrated_value)

for channel_name in ['slowEL0', 'slowEL1', 'slowEL2', 'slowEL3']:
    hist = calib_hists[channel_name]
    hist.Write()

    canvas = ROOT.TCanvas(f'c_{channel_name}_lin', f'c_{channel_name}_lin', 800, 600)
    canvas.SetGrid()
    hist.Draw()
    hist.GetXaxis().SetTitle('Energy (keV)')
    hist.GetYaxis().SetTitle('Counts (1/keV)')
    hist.SetLineColor(1)
    hist.SetLineWidth(1)
    hist.SetStats(0)
    canvas.Update()
    canvas.SaveAs(str(calib_dir / (name + f'_{channel_name}calib_lin.root')))
    canvas.SaveAs(str(calib_dir / (name + f'_{channel_name}calib_lin.png')))
    canvas.Close()

calib.Close()

print('Calibration complete.')
for channel_name in ['slowEL0', 'slowEL1', 'slowEL2', 'slowEL3']:
    print(f'Linear fit selected for {channel_name}.')