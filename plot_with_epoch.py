import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pickle
import re
import pandas as pd
from matplotlib.lines import Line2D  # For custom legend
from itertools import cycle


dat_dir = "/work2/08264/baagee/frontera/mfmc-gns/outputs/"
filenames = ["eval-0kto2600k.pkl",
             "eval-2600kto3500k.pkl",
             "eval-3500kto5900k.pkl",
             "eval-5900kto7000k.pkl"]
output_dir = 'outputs/'

with open(f'{dat_dir}/eval-0kto2600k.pkl', 'rb') as file:
    a = pickle.load(file)

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

# Training time for maximum epoch, pe_max (s)
pe_max = 604800  # assumed to be 1 week in sec for 1M epochs
emax = 7e6
is_fair_comparison = True
# high-fidelity eval time: w0, s
w0 = 8280 * 56  # time * processes
# low-fidelity eval time: w1, s
w1 = 32 * 3072  # time * CUDA cores
p_total_add_coeff = 500000
start_sampling = 8

# Get data
epochs = []
corrs = []
corr_decays = []
for key, item in results.items():
    epoch = int(re.search(r'\d+', key).group())
    corr_decay = 1 - item['correlation'] ** 2
    if epoch <= emax:
        epochs.append(epoch)
        corr_decays.append(corr_decay)
epochs = epochs[start_sampling:]
corr_decays = corr_decays[start_sampling:]

# Training time with epochs: p(e)
def fp(e):
    """
    Linear estimation of training time given that pe_max is the training time spent when emax
    e: epoch
    """
    return pe_max * (e / emax)

# Total budget: p_total
# It should be satisfied such that p_total - p(e_max) > w0
p_total = pe_max + w0 + p_total_add_coeff

# Normalized values
p_bar = p_total / w0  # total budet
w1_bar = w1 / w0  # eval cost
def fp_bar(e):  # training time
    return fp(e) / w0

# Decay model
# Plot the correlation decay with epoch
# Define the model function for the correlation decay
def model(e, c, alpha):
    return c * e ** (-alpha)
# Perform curve fitting to the decay model
initial_guess = [1, 1]  # Initial guess for c and alpha
fit_params, cov_matrix = curve_fit(model, epochs, corr_decays, p0=initial_guess)

# Objective
def g(e):
    objective = 1 / (p_bar - fp_bar(e)) * (model(e, fit_params[0], fit_params[1]) + w1_bar)
    return objective

# Correlation coeff decay and fit curve (1-\rho(e)^2)
fig0, ax = plt.subplots()
decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
ax.plot(epochs, corr_decays, label='data', marker='o', c='k')
ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
ax.set_xlabel("Epochs, n")
ax.set_ylabel(r"$1-\rho(e)^2$")
ax.set_yscale("log")
ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
ax.legend()
plt.savefig(f"{dat_dir}/decay-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

# Objective
fig1, ax = plt.subplots()
objectives = [g(e) for e in epochs]
min_idx = np.argmin(np.array(objectives))
ax.plot(epochs, objectives)
ax.scatter(epochs[min_idx], objectives[min_idx],
           label=f"Min: epochs={epochs[min_idx]:.2e}, objective={objectives[min_idx]:.2e}")
ax.set_xlabel("Epochs, n")
ax.set_ylabel(r"Objective")
ax.legend()
plt.savefig(f"{dat_dir}/objective-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")




