import torch
import os
from gns.gns import reading_utils
from gns.gns import train
from gns.example.inverse_problem.forward import rollout_with_checkpointing
import data_loader
import pickle
from tqdm import tqdm
import time
import numpy as np
from matplotlib import pyplot as plt


# Set epochs to analyze
epochs = np.arange(0, 200000, 100000)
plot_step = 2  # plot interval while evaluating through epochs
output_dir = '/work2/08264/baagee/frontera/mfmc-gns/outputs/'
output_file = 'eval_new_format-time.pkl'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Simulator configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator_metadata_path = '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/'
noise_std = 6.7e-4
model_path = '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/'
INPUT_SEQUENCE_LENGTH = 6

# Get ground truth data paths
aspect_ratio_ids = ["027", "046", "054", "069", "082"]
# aspect_ratio_ids = ["027", "046"]
data_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/mfmc/"
friction_ids = [0, 2, 3, 4, 5]  # [(0, 15), (1, 17.5), (2, 22.5), (3, 30), (4, 37.5), (5, 45)]  # id-friction pare
# for friction_id in [0, 2, 3, 4, 5]:
true_npz_files = []
for friction_id in friction_ids:
    for a_id in aspect_ratio_ids:
        true_npz_file = f"{data_dir}/mfmc-a{a_id}-{friction_id}.npz"
        true_npz_files.append(true_npz_file)

# Get ground truth values
true_data_holder = {"aspect_ratio": [], "friction": [], "runout_true": []}
for i, file_path in enumerate(true_npz_files):
    current_data = data_loader.get_npz_data(file_path, option="runout_only")
    for key in true_data_holder.keys():
        true_data_holder[key].append(current_data[key])

# Make dict to save the rollout analysis result
data_holder = {}

# Evaluate the values for epochs
for i, epoch in enumerate(epochs):

    print(f"Current epoch {epoch}")
    model_file = f'model-{epoch}.pt'

    # Load simulator
    metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout")
    simulator = train._get_simulator(metadata, noise_std, noise_std, device)
    if os.path.exists(model_path + model_file):
        simulator.load(model_path + model_file)
    else:
        raise Exception(f"Model does not exist at {model_path + model_file}")
    simulator.to(device)
    simulator.eval()

    # For all datapoints:
    t_rollouts = []
    pred_runouts = []
    for true_npz_file in true_npz_files:
        data = data_loader.get_npz_data(true_npz_file, option="entire_data")

        # Rollout
        sequence_length = data["positions"].shape[1]
        # Forward evaluation for runout with selected data point (aspect ratios & frictions)
        t_rollout_start = time.time()
        with torch.no_grad():
            predicted_positions = rollout_with_checkpointing(
                simulator=simulator,
                initial_positions=data["positions"][:, :6, :].to(device),
                particle_types=data["particle_type"].to(device),
                material_property=data["material_property"].to(device),
                n_particles_per_example=data["n_particles_per_example"],
                nsteps=sequence_length - INPUT_SEQUENCE_LENGTH  # exclude initial positions (x0) which we already have
            )
        t_rollout_end = time.time()
        t_rollout = t_rollout_end - t_rollout_start
        t_rollouts.append(t_rollout)

        # Get necessary values
        pred_runout = predicted_positions[-1, :, 0].max().item()
        pred_runouts.append(pred_runout)

    # Save data
    data_holder[f"epoch-{epoch}"] = {
        "epoch": epoch,
        "t_rollouts": t_rollouts,
        "aspect_ratios": true_data_holder["aspect_ratio"],
        "frictions": true_data_holder["friction"],
        "pred_rounouts": pred_runouts,
        "true_rounouts": true_data_holder["runout_true"],
        "correlation": np.corrcoef(pred_runouts, true_data_holder["runout_true"])[0, 1]
    }

    # Save pred-true runout plot
    if i % plot_step == 0:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, 2, 100), np.linspace(0, 2, 100), color="black")
        ax.scatter(data_holder[f"epoch-{epoch}"]["true_rounouts"],
                   data_holder[f"epoch-{epoch}"]["pred_rounouts"],
                   color="black")
        ax.set_xlim([0.3, 2.0])
        ax.set_ylim([0.3, 2.0])
        ax.set_xlabel("True runout")
        ax.set_ylabel("Pred runout")
        ax.set_aspect("equal")
        ax.set_title(f"rho = {data_holder[f'epoch-{epoch}']['correlation']}")
        plt.savefig(f"{output_dir}/epoch-{epoch}.png")


    # Save current data_holder
    with open(f"{output_dir}/{output_file}", 'wb') as file:
        pickle.dump(data_holder, file)

    a=1





