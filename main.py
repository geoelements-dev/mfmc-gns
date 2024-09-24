import os
import numpy as np
from matplotlib import pyplot as plt

path = '/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_uncertain/'
cases = ['deg21', 'deg33', 'deg45']
trjs = [0, 1]

frictions = [21, 33, 45]
aspect_ratios = [0.8, 2.0]

data = {}
for c, case in enumerate(cases):
    for t, trj in enumerate(trjs):
        with np.load(f"{path}/{case}/trajectory{trj}.npz", allow_pickle=True) as data_file:
            data_temp = [item for _, item in data_file.items()]

            L0 = data_temp[0][0][0, :, 0].max() - data_temp[0][0][0, :, 0].min()
            H0 = data_temp[0][0][0, :, 1].max() - data_temp[0][0][0, :, 1].min()
            Lf = data_temp[0][0][-1, :, 0].max()
            Lf_percentile = np.percentile(data_temp[0][0][-1, :, 0], 99)

            data[f"Friction={frictions[c]}, a={aspect_ratios[t]}"] = {
                'friction': frictions[c],
                'aspect ratio': aspect_ratios[t],
                'positions': data_temp[0][0],
                'L0': L0,
                'H0': H0,
                'runout': Lf_percentile - L0,
                'normalized runout': (Lf_percentile - L0) / L0
            }

plot_data = {
    "runout (a=0.8)": [],
    "runout (a=2.0)": [],
    "normalized runout (a=0.8)": [],
    "normalized runout (a=2.0)": []
}

for key, value in data.items():
    if 'a=0.8' in key:
        plot_data['runout (a=0.8)'].append(value['runout'])
        plot_data['normalized runout (a=0.8)'].append(value['normalized runout'])
    if 'a=2.0' in key:
        plot_data['runout (a=2.0)'].append(value['runout'])
        plot_data['normalized runout (a=2.0)'].append(value['normalized runout'])

fig1, ax = plt.subplots()
ax.plot(frictions, plot_data['runout (a=0.8)'], label='a=0.8')
ax.plot(frictions, plot_data['runout (a=2.0)'], label='a=2.0')
ax.set_xlabel('Friction angle (degree)')
ax.set_ylabel(r'Runout, $L_t$ (m)')
ax.legend()
plt.savefig('runout.png')


fig2, ax = plt.subplots()
ax.plot(frictions, plot_data['normalized runout (a=0.8)'], label='a=0.8')
ax.plot(frictions, plot_data['normalized runout (a=2.0)'], label='a=2.0')
ax.set_xlabel('Friction angle (degree)')
ax.set_ylabel(r'Normalized runout, $(L_f-L_0)/L_0$')
ax.legend()
plt.savefig('normalized_runout.png')


