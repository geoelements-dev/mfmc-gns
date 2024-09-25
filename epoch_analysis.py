import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pickle
import re
import pandas as pd
from matplotlib.lines import Line2D  # For custom legend
from itertools import cycle



dat_dir = "./data/"
filenames = ["eval-0kto2600k.pkl",
             "eval-2600kto3500k.pkl",
             "eval-3500kto5900k.pkl",
             "eval-5900kto7000k.pkl"]

# Append the separated data
results = {}
for filename in filenames:
    with open(f'{dat_dir}/{filename}', 'rb') as file:
        result = pickle.load(file)
        results.update(result)

# Preprocess: round the error
for key, velue in results.items():
    results[key]["aspect_ratios"] = np.round(results[key]["aspect_ratios"], decimals=2)
    results[key]["frictions"] = np.round(results[key]["frictions"], decimals=0)

# Get data
epochs = []
corrs = []
corr_decays = []
for key, item in results.items():
    epoch = re.search(r'\d+', key).group()
    corr_decay = 1 - item['correlation'] ** 2
    epochs.append(int(epoch))
    corr_decays.append(corr_decay)

# Plot the correlation decay with epoch
# Define the model function for the correlation decay
def model(epoch, c, alpha):
    return c * epoch ** (-alpha)

# Perform curve fitting to the decay model
initial_guess = [1, 1]  # Initial guess for c and alpha
fit_params, cov_matrix = curve_fit(model, epochs[1:], corr_decays[1:], p0=initial_guess)
fig1, ax = plt.subplots()
y = [model(epoch, c=fit_params[0], alpha=fit_params[1]) for epoch in epochs[1:]]
ax.plot(epochs[1:], corr_decays[1:], label='data', marker='o', c='k')
ax.plot(epochs[1:], y, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
ax.set_xlabel("Epochs, n")
ax.set_ylabel(r"$1-\rho^2$")
ax.set_ylim([1e-2, 0.3])
ax.set_yscale("log")
ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
ax.legend()
plt.savefig("decay.png")
plt.show()

# Plot objective function
def objective(epoch, p_bar, cost=875):
    y = 1/(p_bar - epoch) * (model(epoch, c=fit_params[0], alpha=fit_params[1]) + cost)
    return y
y2 = [objective(epoch, p_bar=1000000000) for epoch in epochs[1:]]
fig2, ax = plt.subplots()
ax.plot(epochs[1:], y2, label='objective', marker='o', c='k')
ax.set_xlabel("Epochs, n")
ax.set_ylabel(r"Objective")
plt.savefig("objective.png")
plt.show()


###

# Assuming 'results' is a predefined dictionary with data
epoch_to_plot = 2000000
df_epoch = pd.DataFrame.from_dict(results[f'epoch-{epoch_to_plot}'])
corr = results[f'epoch-{epoch_to_plot}']['correlation']

cmap = plt.cm.viridis
norm = plt.Normalize(df_epoch['frictions'].min(), df_epoch['frictions'].max())

fig3, ax = plt.subplots()

# Prepare a cycle of markers
markers = ['o', 'v', '^', '<', '>']
marker_cycle = cycle(markers)

# Map each aspect ratio to a unique marker
aspect_ratios = df_epoch['aspect_ratios'].unique()
aspect_ratio_to_marker = {ar: next(marker_cycle) for ar in aspect_ratios}

# Plot each group with its respective marker and color
for (aspect_ratio, friction), group in df_epoch.groupby(['aspect_ratios', 'frictions']):
    ax.scatter(group['true_rounouts'], group['pred_rounouts'],
               c=[friction]*len(group), cmap=cmap, norm=norm,
               marker=aspect_ratio_to_marker[aspect_ratio])

# Add color bar for 'frictions'
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('frictions')

legend_elements = [Line2D([0], [0], marker=marker, color='w', label=aspect_ratio,
                          markerfacecolor='black', markersize=10)
                   for aspect_ratio, marker in aspect_ratio_to_marker.items()]
ax.legend(handles=legend_elements, title="Aspect Ratios")

# Final plot adjustments
ax.axline((0, 0), slope=1, color='grey', linestyle='--')
ax.set_aspect('equal')
ax.set_xlabel('True Runout')
ax.set_ylabel('Predicted Runout')
ax.set_title(f'Correlation coefficient: {corr:.3e}')
ax.set_xlim([0, 2.0])
ax.set_ylim([0, 2.0])

plt.tight_layout()
plt.savefig("scatter.png")
plt.show()


##
fig4, ax = plt.subplots()
ax.scatter(df_epoch['true_rounouts'], df_epoch['pred_rounouts'])
ax.axline((0, 0), slope=1)
ax.set_aspect('equal')
ax.set_xlabel('true_rounouts')
ax.set_ylabel('pred_rounouts')
ax.set_xlim([0, 2.0])
ax.set_ylim([0, 2.0])
plt.show()
