import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import re


def main():
    #Load in the data that we will plot.
    dat_dir = './outputs'
    filenames = ["rp_eval_0k_to_2000k_full_time.pkl", "rp_eval_2000k_to_6000k_full_time.pkl"]
    output_dir = './outputs/full_time_anim_testing'


    # Append the separated data
    results = {}
    for filename in filenames:
        with open(f'{dat_dir}/{filename}', 'rb') as file:
            result = pickle.load(file)
            results.update(result)

    analysis_keys = ['epoch-600000', 'epoch-1100000', 'epoch-1800000', 'epoch-2400000', 'epoch-3000000',
                    'epoch-3500000', 'epoch-4000000', 'epoch-4500000', 'epoch-5000000']
    test_tx = 10 #timestep index (out of 360)
    test_i = 5 #example index (out of 25)
    analysis_examples = [6, 11, 16, 21]

    #Loop for subplotted animation over full time and truncated time.
    for i in range(len(analysis_examples)):
        idx = analysis_examples[i]
        print(f"Working with example {idx}")

        ### FULL TIME LOOP

        #Initialize the subplots object for storing the animations.
        fig, axs = plt.subplots(5,2, figsize=(12, 18))
        axs = axs.flatten()
        # assert len(axs) == len(analysis_keys)
        true_scats = [] #Have to store the scatterplot objects for each key.
        pred_scats = [] #same for the prediction scatters.
        timesteps = []
        print("Beginning axis initializations.")
        for j in range(len(axs)-1):
            ax = axs[j]
            print(j)
            true_positions = results[analysis_keys[j]]["true_positions"][idx].permute(1,0,2).numpy()
            pred_positions = results[analysis_keys[j]]["pred_positions"][idx].numpy()
            ax.set_title(f"model {analysis_keys[j]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, 1.25)
            ax.set_ylim(0, 0.3)
            true_scatter = ax.scatter(true_positions[0, :, 0], true_positions[-1, :, 1],
                                    label='True Positions', s=1)
            pred_scatter = ax.scatter(pred_positions[0, :, 0], pred_positions[-1, :, 1],
                                    label='Predictions', s=1, alpha=0.5)
            ax.legend(loc='best')
            true_scats.append(true_scatter)
            pred_scats.append(pred_scatter)
            timesteps.append(np.arange(true_positions.shape[0]))
        fig.suptitle(f"Comparative Scatterplots over Various Epochs, Example {idx}")
        plt.tight_layout(pad=3.0)
        print("Axis initializations complete, beginning updates...")
        
        #Add correlation plot on the 10th subplot.
        corr_decays = []
        epochs = []
        for key, item in results.items():
            epoch = int(re.search(r'\d+', key).group())
            corr_decay = 1 - item['correlation'] ** 2
            corr_decays.append(corr_decay)
            epochs.append(epoch)
        axs[-1].plot(epochs, corr_decays, label='data', marker='o', c='k')
        axs[-1].set_xlabel("Epochs, n")
        axs[-1].set_ylabel(r"$1-\rho(e)^2$")
        axs[-1].set_yscale("log")
        axs[-1].set_title(f"Correlation Scatter Plot")

        #Update function and animation save.
        def update(t):
            '''Function to update the existing plot.'''

            # Update data.
            for i, (true_scatter, pred_scatter) in enumerate(zip(true_scats, pred_scats)):
                true_positions = results[analysis_keys[i]]["true_positions"][idx].permute(1,0,2).numpy()
                pred_positions = results[analysis_keys[i]]["pred_positions"][idx].numpy()
                if t % 20 == 0:
                    print(f"Updating subplot {i}, timestep {t}")
                true_scatter.set_offsets(true_positions[t, :, :])
                pred_scatter.set_offsets(pred_positions[t, :, :])

            return true_scats + pred_scats

        # Create the animation
        n_frames = true_positions.shape[0]
        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

        # Save the animation
        # plt.show()
        ani.save(f'{output_dir}/comp_scatter_example_{idx}.gif', writer=animation.PillowWriter(fps=20))


        ### TRUNCATED TIME LOOP: true_positions[40:] vs GNS[:-40]
        tx_truncate = 40
        #Initialize the subplots object for storing the animations.
        fig, axs = plt.subplots(5,2, figsize=(12, 18))
        axs = axs.flatten()
        # assert len(axs) == len(analysis_keys)
        true_scats = [] #Have to store the scatterplot objects for each key.
        pred_scats = [] #same for the prediction scatters.
        timesteps = []
        print("Beginning axis initializations for truncated time.")
        for j in range(len(axs)-1):
            ax = axs[j]
            print(j)
            true_positions = results[analysis_keys[j]]["true_positions"][idx].permute(1,0,2).numpy()[tx_truncate:,:,:]
            pred_positions = results[analysis_keys[j]]["pred_positions"][idx].numpy()[:-tx_truncate,:,:]
            ax.set_title(f"model {analysis_keys[j]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, 1.25)
            ax.set_ylim(0, 0.3)
            true_scatter = ax.scatter(true_positions[0, :, 0], true_positions[-1, :, 1],
                                    label='True Positions', s=1)
            pred_scatter = ax.scatter(pred_positions[0, :, 0], pred_positions[-1, :, 1],
                                    label='Predictions', s=1, alpha=0.5)
            ax.legend(loc='best')
            true_scats.append(true_scatter)
            pred_scats.append(pred_scatter)
            timesteps.append(np.arange(true_positions.shape[0]))
        fig.suptitle(f"Comparative Scatterplots over Various Epochs, Time Truncation, Example {idx}")
        plt.tight_layout(pad=3.0)
        print("Axis initializations complete, beginning updates...")
        
        #TOD0: add correlation plot on the 10th subplot.
        #We will have to calculate correlations on the fly between each example truncated points.

        # corr_decays = []
        # epochs = []
        # for key, item in results.items():
        #     epoch = int(re.search(r'\d+', key).group())
        #     true_pos_trunc = [item["true_positions"][i].permute(1,0,2).numpy()[tx_truncate:,:,:] for i in range(25)]
        #     pred_pos_trunc = [item["true_positions"][i].numpy()[:-tx_truncate,:,:] for i in range(25)]
        #     correlation = [np.corrcoef(true_pos_trunc[i], pred_pos_trunc[i])[0,1] for i in range(25)]
        #     corr_decay = 1 - item['correlation'] ** 2
        #     corr_decays.append(corr_decay)
        #     epochs.append(epoch)
        # axs[-1].plot(epochs, corr_decays, label='data', marker='o', c='k')
        # axs[-1].set_xlabel("Epochs, n")
        # axs[-1].set_ylabel(r"$1-\rho(e)^2$")
        # axs[-1].set_yscale("log")
        # axs[-1].set_title(f"Correlation Scatter Plot")

        #Update function and animation save.
        def update(t):
            '''Function to update the existing plot.'''

            # Update data.
            for i, (true_scatter, pred_scatter) in enumerate(zip(true_scats, pred_scats)):
                true_positions = results[analysis_keys[i]]["true_positions"][idx].permute(1,0,2).numpy()[tx_truncate:,:,:]
                pred_positions = results[analysis_keys[i]]["pred_positions"][idx].numpy()[:-tx_truncate,:,:]
                if t % 20 == 0:
                    print(f"Updating subplot {i}, timestep {t}")
                true_scatter.set_offsets(true_positions[t, :, :])
                pred_scatter.set_offsets(pred_positions[t, :, :])

            return true_scats + pred_scats

        # Create the animation
        n_frames = true_positions.shape[0]
        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

        # Save the animation
        # plt.show()
        ani.save(f'{output_dir}/comp_scatter_example_{idx}_truncated.gif', writer=animation.PillowWriter(fps=20))

        ### LOOP to simply scatter the last positions.


        #Initialize the subplots object for storing the animations.
        fig, axs = plt.subplots(5,2, figsize=(12, 18))
        axs = axs.flatten()
        # assert len(axs) == len(analysis_keys)
        true_scats = [] #Have to store the scatterplot objects for each key.
        pred_scats = [] #same for the prediction scatters.
        print("Beginning axis initializations for last step scatter plots.")
        for j in range(len(axs)-1):
            ax = axs[j]
            print(j)
            true_positions = results[analysis_keys[j]]["true_positions"][idx].permute(1,0,2).numpy()
            pred_positions = results[analysis_keys[j]]["pred_positions"][idx].numpy()
            ax.set_title(f"model {analysis_keys[j]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, 1.25)
            ax.set_ylim(0, 0.3)
            true_scatter = ax.scatter(true_positions[-1, :, 0], true_positions[-1, :, 1],
                                    label='True Positions', s=1)
            pred_scatter = ax.scatter(pred_positions[-1, :, 0], pred_positions[-1, :, 1],
                                    label='Predictions', s=1, alpha=0.5)
            ax.legend(loc='best')
            true_scats.append(true_scatter)
            pred_scats.append(pred_scatter)
        #Add correlation plot on the 10th subplot.
        corr_decays = []
        epochs = []
        for key, item in results.items():
            epoch = int(re.search(r'\d+', key).group())
            corr_decay = 1 - item['correlation'] ** 2
            corr_decays.append(corr_decay)
            epochs.append(epoch)
        axs[-1].plot(epochs, corr_decays, label='data', marker='o', c='k')
        axs[-1].set_xlabel("Epochs, n")
        axs[-1].set_ylabel(r"$1-\rho(e)^2$")
        axs[-1].set_yscale("log")
        axs[-1].set_title(f"Correlation Scatter Plot")
        fig.suptitle(f"Comparative Scatterplots over Various Epochs, Example {idx}")
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{output_dir}/final_time_scatters_example_{idx}.png')
        


    ## BELOW: code to generate an animation for a given test_i index (out of 25).

    # for k, v in results.items():
    #         print(k)
    #         if (k in analysis_keys): #subsets of epochs and every 5 examples
    #             print(f"Working key: {k}")
    #             print(len(results[k]["true_positions"]))
    #             true_positions = results[k]["true_positions"][test_i].permute(1,0,2).numpy()
    #             pred_positions = results[k]["pred_positions"][test_i].numpy()
    #             # print("Predicted positions shape: ", pred_positions.shape)
    #             # print("True positions shape: ", true_positions.shape)

    #             #Initialize the scatterplot with initial data.
    #             fig, ax = plt.subplots()
    #             timesteps = np.arange(true_positions.shape[0])
    #             assert len(timesteps_true) == len(timesteps_pred)
    #             true_scatter = ax.scatter(true_positions[-1, test_tx:, 0], true_positions[-1, test_tx:, 1], label='True Positions', s=1)
    #             pred_scatter = ax.scatter(pred_positions[-1, :-test_tx, 0], pred_positions[-1, :-test_tx, 1], label='Predictions', s=1)
    #             ax.set_title(f"Particle scatterplot for model {k}, example{test_i}")
    #             ax.set_xlabel("x")
    #             ax.set_ylabel("y")
    #             ax.set_xlim(0, 1.25)
    #             ax.set_ylim(0, 0.3)
    #             ax.legend(loc='best')

    #             def update(t):
    #                 '''Function to update the existing plot.'''
    #                 # Get the current time step.
    #                 tx = timesteps[t]

    #                 # Update data.
    #                 true_scatter.set_offsets(true_positions[tx, :, :])
    #                 pred_scatter.set_offsets(pred_positions[tx, :, :])

    #                 return true_scatter, pred_scatter

    #             # Create the animation
    #             ani = animation.FuncAnimation(fig, update, frames=len(timesteps_true), interval=50, blit=True)

    #             # Save the animation
    #             # plt.show()
    #             ani.save(f'{output_dir}/scatter_{k}_example{test_i}.gif', writer=animation.PillowWriter(fps=30))

    #             # Add some code to understand at which point the true positions start to move.
    #             # for i in range(15):
    #             #     fig, ax = plt.subplots()
    #             #     true_scatter = ax.scatter(pred_positions[i, :, 0], pred_positions[i, :, 1], s=1)
    #             #     ax.set_xlabel("x")
    #             #     ax.set_ylabel("y")
    #             #     ax.set_xlim(0, 1.25)
    #             #     ax.set_ylim(0, 0.3)
    #             #     ax.legend(loc='best')
    #             #     plt.savefig(f'{output_dir}/testing_pred_movement_timestep_{i}_key_{k}')

    return None

def truncated():
    dat_dir = './outputs'
    filenames = ["rp_eval_0k_to_2000k_full_time.pkl", "rp_eval_2000k_to_6000k_full_time.pkl"]
    output_dir = './outputs/full_time_anim_testing'
    analysis_keys = ['epoch-500000', 'epoch-1000000', 'epoch-1800000', 'epoch-2700000',
                      'epoch-4000000', 'epoch-5000000']

    # Append the separated data
    results = {}
    for filename in filenames:
        with open(f'{dat_dir}/{filename}', 'rb') as file:
            result = pickle.load(file)
            results.update(result)

    trunc_idx = 40

    #Code to generate a correlation decay plot using the truncated GNS[-40] and MPM[-1].
    epochs = []
    corr_decays = []
    for key, item in results.items():
        epoch = int(re.search(r'\d+', key).group())
        epochs.append(epoch)
        traj_runout_preds = []
        num_ex = len(item["pred_positions"]); assert (num_ex == 25)
        for i in range(num_ex):
            truncated_positions = item["pred_positions"][i].numpy()[-trunc_idx,:,:]
            trunc_runout = truncated_positions[:,0].max()
            traj_runout_preds.append(trunc_runout)
        # print(f"True runouts: {item['true_rounouts']}")
        # print(f"GNS truncated runout results: {traj_runout_preds}")
        # print(f"Full predicted runout: {item['pred_rounouts']}")
        corr = np.corrcoef(traj_runout_preds, item["true_rounouts"])[0,1]
        corr_decay = 1 - corr ** 2
        corr_decays.append(corr_decay)
    
    f, ax = plt.subplots()
    # decay = [model(e, fit_params[0], fit_params[1]) for e in epochs]
    ax.plot(epochs, corr_decays, label='1-$\rho^2$', marker='o', c='k')
    # ax.plot(epochs, decay, label=r"Fit to $1-\rho^2(n) = c_{a, 1} n^{-a}$", linestyle="--")
    ax.set_xlabel("Epochs, n")
    ax.set_ylabel(r"$1-\rho(e)^2$")
    ax.set_yscale("log")
    # ax.set_title(rf"$c_{{a,1}} = {fit_params[0]:.3f}$, $\alpha = {fit_params[1]:.3f}$")
    ax.set_title(f"Correlation Decay Using GNS runout at timestep -{trunc_idx}")
    # ax.legend()
    plt.savefig(f"{output_dir}/truncated_correlation_plot.png")

    #Generate scatter plots of all 25 current trajectories, for selected epochs in analysis_keys.
    for k in analysis_keys:
        print("Generating scatter for : ", k)
        for i in range(25):
            preds = results[k]["pred_positions"][i].numpy()[-40,:,:]
            ground_truth = results[k]["true_positions"][i].permute(1,0,2).numpy()[-1,:,:]
            f,ax = plt.subplots()
            pred_scatter = ax.scatter(preds[:,0], preds[:,1], label='GNS Prediction', s=1, alpha=0.6)
            true_scatter = ax.scatter(ground_truth[:,0], ground_truth[:,1], label='MPM data', s=1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(0.0, 1.25)
            ax.set_ylim(0.0, 0.3)
            ax.set_title('GNS at -40 vs MPM at -1: final scatter')
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/final_time_scatters/truncated/truncated_final_scatter_{k}_example_{i}')
            plt.close()

    print("All scatters complete.")
    return None

def velocity_check():
    dat_dir = './outputs'
    filenames = ["rp_eval_0k_to_2000k_full_time.pkl", "rp_eval_2000k_to_6000k_full_time.pkl"]
    output_dir = './outputs/full_time_anim_testing'
    # trunc_idx = 0

    # Append the separated data
    results = {}
    for filename in filenames:
        with open(f'{dat_dir}/{filename}', 'rb') as file:
            result = pickle.load(file)
            results.update(result)
    analysis_keys = ['epoch-1800000','epoch-5000000']
    analysis_examples = np.random.choice(np.arange(25), size=5, replace=False)

    for key in analysis_keys:
        for i in analysis_examples:
            print(f"{key}")
            print(f"Trajectory {i}")
            true_positions_final_steps = results[key]["true_positions"][i].permute(1,0,2).numpy()[-2:,:,:]
            pred_positions_final_steps = results[key]["pred_positions"][i].numpy()[-2:,:,:]

            predictions_diff = np.sum(pred_positions_final_steps[-1,:,:] - pred_positions_final_steps[:-2,:,:])
            true_diff = np.sum(true_positions_final_steps[-1,:,:] - true_positions_final_steps[-2,:,:])

            # print("Truncation index: ", trunc_idx)

            print("Sum of position differences for predictions: ", predictions_diff)
            print("Sum of position differences for ground truth MPM: ", true_diff)

            print('#'*25)
    
    return

    

if __name__ == '__main__':
    # main()
    truncated()
    # velocity_check()


