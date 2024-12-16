import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pickle
import re
# import pandas as pd
from matplotlib.lines import Line2D  # For custom legend
from itertools import cycle
import metrics


# dat_dir = "/work2/08264/baagee/frontera/mfmc-gns/outputs/"
# filenames = ["eval-0kto2600k.pkl",
#              "eval-2600kto3500k.pkl",
#              "eval-3500kto5900k.pkl",
#              "eval-5900kto7000k.pkl"]
# output_dir = 'outputs/'
dat_dir = './outputs'
# filenames = ["rp_eval_0_to_2000k.pkl", "rp_eval_2000k_to_6000k.pkl"]
filenames = ["rp_eval_0k_to_2000k_full_time.pkl", "rp_eval_2000k_to_5000k_full_time.pkl",
             "rp_eval_5000k_to_7000k_full_time.pkl"]
output_dir = './outputs'

# with open(f'{dat_dir}/eval-0kto2600k.pkl', 'rb') as file:
#     a = pickle.load(file)

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
print(results.keys())

# Training time for maximum epoch, pe_max (s)
pe_max = 604800  # assumed to be 1 week in sec for 1M epochs
# emax = 7e6
emax = 6e6
is_fair_comparison = True
# high-fidelity eval time: w0, s
w0 = 8280 * 56  # time * processes
# low-fidelity eval time: w1, s
w1 = 32 * 3072  # time * CUDA cores
p_total_add_coeff = 500000
start_sampling = 8 #0


def plot_distances():
    # Get data
    epochs = []
    corrs = []
    corr_decays = []
    hausdorff_decays = []; haus_dists = []
    chamfer_decays = []; chamf_dists = []
    wasser_decays = []; wass_dists = []
    com_decays = []
    diss_corr_decays = []
    for key, item in results.items():
        epoch = int(re.search(r'\d+', key).group())
        corr_decay = 1 - item['correlation'] ** 2
        # haus_decay = 1 - item['hausdorff_euclidean'] ** 2
        # chamfer_decay = 1 - item['chamfer_euclidean'] ** 2
        # wasser_decay = 1 - item['wasser_euclidean'] ** 2

        #Center of mass and kinetic energy correlation storage.
        pred_com = []; true_com = []
        diss_pred = []; diss_true = []
        for ex in item["pred_positions"]:
            # c = metrics.center_of_mass(ex.numpy())
            # pred_com.append(c)
            delta_e_pred = metrics.delta_E(ex.numpy()[0,:,:], ex.numpy()[-40,:,:])
            kinetic_pred = metrics.kinetic_energy(ex.numpy()[:-40,:,:])
            dissipation_pred = metrics.dissipation(delta_e_pred, kinetic_pred)
            diss_pred.append(dissipation_pred)
        for ex in item["true_positions"]:
            # permuted = ex.permute(1,0,2)[-1,:,:2].numpy()
            # d = metrics.center_of_mass(permuted)
            # true_com.append(d)
            permuted = ex.permute(1,0,2).numpy()
            delta_e_true = metrics.delta_E(permuted[40,:,:], permuted[-1,:,:])
            kinetic_true = metrics.kinetic_energy(permuted[40:,:,:])
            dissipation_true = metrics.dissipation(delta_e_true, kinetic_true)
            diss_true.append(dissipation_true)
        # com_decays.append(1-np.corrcoef(pred_com, true_com)[0,1]**2)
        diss_corr_decays.append(1 - np.corrcoef(diss_pred, diss_true)[0,1] ** 2)

        if epoch <= emax:
            epochs.append(epoch)
            corr_decays.append(corr_decay)
            # hausdorff_decays.append(haus_decay)
            # chamfer_decays.append(chamfer_decay)
            # wasser_decays.append(wasser_decay)
            # haus_dists.append(item['hausdorff_euclidean'])
            # chamf_dists.append(item['chamfer_euclidean'])
            # wass_dists.append(item['wasser_euclidean'])
    epochs = epochs[start_sampling:]
    corr_decays = corr_decays[start_sampling:]
    # hausdorff_decays = hausdorff_decays[start_sampling:]
    # chamfer_decays = chamfer_decays[start_sampling:]
    # wasser_decays = wasser_decays[start_sampling:]
    # haus_dists = haus_dists[start_sampling:]
    # chamf_dists = chamf_dists[start_sampling:]
    # wass_dists = wass_dists[start_sampling:]
    # com_decays = com_decays[start_sampling:]
    diss_corr_decays = diss_corr_decays[start_sampling:]
    print(f"Center of mass decays: {com_decays}")

    def minmax_scale(arr):
        #input should be a numpy array
        return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

    # #TESTING
    # haus_dists = minmax_scale(haus_dists)
    # chamf_dists = minmax_scale(chamf_dists)
    # wass_dists = minmax_scale(wass_dists)

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

    # Perform curve fitting to the decay model -- for runout correlations.
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

    # # Hausdorff decay and curve fit
    # fig1, ax = plt.subplots()
    # # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    # ax.plot(epochs, hausdorff_decays, label='data', marker='o', c='k')
    # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/hausdorff_decay-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # # Chamfer decay and curve fit
    # fig2, ax = plt.subplots()
    # # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    # ax.plot(epochs, chamfer_decays, label='data', marker='o', c='k')
    # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/chamfer_decay-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # # wasser decay and curve fit
    # fig3, ax = plt.subplots()
    # # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    # ax.plot(epochs, wasser_decays, label='data', marker='o', c='k')
    # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/wasserstein_decay-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # Objective
    fig4, ax = plt.subplots()
    objectives = [g(e) for e in epochs]
    min_idx = np.argmin(np.array(objectives))
    ax.plot(epochs, objectives)
    ax.scatter(epochs[min_idx], objectives[min_idx],
            label=f"Min: epochs={epochs[min_idx]:.2e}, objective={objectives[min_idx]:.2e}")
    ax.set_xlabel("Epochs, n")
    ax.set_ylabel(r"Objective")
    ax.legend()
    plt.savefig(f"{dat_dir}/objective-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # #Checking the distances themselves.
    # fig_test, ax = plt.subplots()
    # fit_params_haus, cov_matrix = curve_fit(model, epochs, haus_dists, p0=initial_guess)
    # haus_decay = [model(e, fit_params_haus[0], fit_params_haus[1]) for e in epochs]
    # print("Hausdorff distances: ", haus_dists)
    # ax.plot(epochs, haus_dists, label='data', marker='o', c='k')
    # ax.plot(epochs, haus_decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/haus_DIST-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # fig_test, ax = plt.subplots()
    # fit_params_chamfer, cov_matrix = curve_fit(model, epochs, chamf_dists, p0=initial_guess)
    # chamf_decay = [model(e, fit_params_chamfer[0], fit_params_chamfer[1]) for e in epochs]
    # print("Chamfer distances: ", chamf_dists)
    # ax.plot(epochs, chamf_dists, label='data', marker='o', c='k')
    # ax.plot(epochs, chamf_decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/chamfer_DIST-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    # fig_test, ax = plt.subplots()
    # fit_params_wass, cov_matrix = curve_fit(model, epochs, wass_dists, p0=initial_guess)
    # wass_decay = [model(e, fit_params_wass[0], fit_params_wass[1]) for e in epochs]
    # print("Wasserstein dists: ", wass_dists)
    # ax.plot(epochs, wass_dists, label='data', marker='o', c='k')
    # ax.plot(epochs, wass_decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/wasser_DIST-emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    #Center of mass decay plot.
    # fig, ax = plt.subplots()
    # initial_guess = [1, 1]  # Initial guess for c and alpha
    # fit_params, cov_matrix = curve_fit(model, epochs, com_decays, p0=initial_guess)
    # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    # ax.plot(epochs, com_decays, label='COM data', marker='o', c='k')
    # # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    # ax.set_xlabel("Epochs, n")
    # ax.set_ylabel(r"$1-\rho(e)^2$")
    # ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    # ax.legend()
    # plt.savefig(f"{dat_dir}/decay-centerofmass_emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    #Kinetic energy correlation decay plot.
    fig, ax = plt.subplots()
    initial_guess = [1, 1]  # Initial guess for c and alpha
    # fit_params, cov_matrix = curve_fit(model, epochs, com_decays, p0=initial_guess)
    # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    ax.plot(epochs, diss_corr_decays, label='KE correlation data', marker='o', c='k')
    print(f"Dissipation correlation decay: {diss_corr_decays}")
    # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    ax.set_xlabel("Epochs, n")
    ax.set_ylabel(r"$1-\rho(e)^2$")
    ax.set_yscale("log")
    ax.set_title(f"Energy dissipation correlation decay")
    ax.legend()
    plt.savefig(f"{dat_dir}/decay-delta_energy_emax{emax}-from{start_sampling}-fair_{is_fair_comparison}.png")

    return

#Generating a scatter plot of true vs pred for a given epoch, final timestep.
#Iterate over selected examples.
def plot_scatters():
    # keys = ['epoch-0', 'epoch-1000000', 'epoch-2000000', 'epoch-3000000', 'epoch-4000000', 'epoch-5000000', 'epoch-5900000']
    keys = ['epoch-1800000', 'epoch-5300000']
    examples = np.arange(25) #can change this if want only a subset of the examples.
    output_dir = './outputs/full_time_anim_testing/final_time_scatters'
    for testkey in keys:
        for idx in examples:
            print(f"Scatter plots for {testkey}, example {idx}")
            true_positions = results[testkey]["true_positions"][idx].permute(1,0,2).numpy()[-1,:,:] #final time
            predictions = results[testkey]["pred_positions"][idx].numpy()[-1,:,:] #final time

            # Generate plot, scatter each set of points, and save.
            fig, ax = plt.subplots()
            ax.scatter(true_positions[:,0], true_positions[:,1], label='MPM', s=1)
            ax.scatter(predictions[:,0], predictions[:,1], label='GNS prediction', s=1, alpha=0.65)
            ax.set_xlim(0.0, 1.25)
            ax.set_ylim(0.0, 0.3)
            ax.set_title(f"Final Time Scatter Plot, {testkey}, Trajectory {idx}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc='best')
            plt.savefig(f"{output_dir}/{testkey}_example_{idx}")
            plt.close()
    
    plt.close('all')
    print('#'*50)

    return

def main():
    plot_distances()
    # plot_scatters()
    return

if __name__ == '__main__':
    main()