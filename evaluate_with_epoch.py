import os
import time
import torch
import pickle
import data_loader
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from gns.gns import reading_utils
from gns.gns import train
from gns.example.inverse_problem.forward import rollout_with_checkpointing
from metrics import hausdorff_distance, chamfer_distance, wassterstein_metric

TACC_MODE = False

if TACC_MODE:
    output_dir = '/work2/08264/baagee/frontera/mfmc-gns/outputs/'
    simulator_metadata_path = ' /corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/'
    model_path = ' /corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/'
    data_dir = "/corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/mpm/mfmc/"
else:
    output_dir = './outputs'
    simulator_metadata_path = './gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/'
    model_path = './gns-mpm-data/gns-data/models/sand2d_frictions-sr020/'
    data_dir = "./gns-mpm-data/mpm/mfmc"

output_file = 'rp_eval_0_to_2000k.pkl' #'eval_new_format-time.pkl' # 'rp_eval_0_to_2000k.pkl' 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Simulator configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = np.arange(0, 2000000, 100000) #(2000000, 6000000, 100000)
plot_step = 2 
noise_std = 6.7e-4
INPUT_SEQUENCE_LENGTH = 6

# Get ground truth data paths
aspect_ratio_ids = ["027", "046", "054", "069", "082"]
friction_ids = [0, 2, 3, 4, 5]  # [(0, 15), (1, 17.5), (2, 22.5), (3, 30), (4, 37.5), (5, 45)]  # id-friction pairs
true_npz_files = []
for friction_id in friction_ids:
    for a_id in aspect_ratio_ids:
        true_npz_file = f"{data_dir}/mfmc-a{a_id}-{friction_id}.npz"
        true_npz_files.append(true_npz_file)

# Get ground truth values
true_data_holder = {"aspect_ratio": [], "friction": [], "runout_true": [], 
                    "positions": [], "particle_type": [],
                    "material_property": [], "n_particles_per_example": []}
for i, file_path in enumerate(true_npz_files):
    current_data = data_loader.get_npz_data(file_path, option="runout_only")
    for key in list(true_data_holder.keys())[:3]:
        true_data_holder[f"{key}"].append(current_data[f"{key}"])
    current_data_pos = data_loader.get_npz_data(file_path, option="entire_data")
    for key in list(true_data_holder.keys())[3:]:
        true_data_holder[f"{key}"].append(current_data_pos[f"{key}"])

# Make dict to save the rollout analysis result
data_holder = {}
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
    print("simulator loaded")

    # For all datapoints:
    t_rollouts = []
    pred_runouts = []
    pred_positions = [] 
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

        # A note on the shape of predicted positions below: it has dimensions (time, nparticles, dim).
        # So the runout here as indexed by [-1,:,0] is the x distance of each particle at the final timestep. 

        # Get necessary values
        pred_runout = predicted_positions[-1, :, 0].max().item()
        pred_runouts.append(pred_runout)

        #Save the predictied positions as well, in the necessary format.
        #We index :2 to get the x and y coordinate of the positions -- this should be changed if we want 3D  positions.
        pred_positions.append(predicted_positions[-1, :, :2].detach().cpu())

    # Save data
    true_positions_permuted = [true_data_holder["positions"][i].permute(1,0,2)[-1,:,:2].detach().cpu() for i in range(len(true_data_holder["positions"]))]
    print(true_positions_permuted[0].shape)
    data_holder[f"epoch-{epoch}"] = {
        "epoch": epoch,
        "t_rollouts": t_rollouts,
        "aspect_ratios": true_data_holder["aspect_ratio"],
        "frictions": true_data_holder["friction"],
        "pred_rounouts": pred_runouts,
        "true_rounouts": true_data_holder["runout_true"],
        "pred_positions": pred_positions,
        "true_positions": true_data_holder["positions"],
        "correlation": np.corrcoef(pred_runouts, true_data_holder["runout_true"])[0, 1],
        "hausdorff_euclidean": hausdorff_distance(pred_positions, true_positions_permuted),
        "chamfer_euclidean": chamfer_distance(pred_positions, true_positions_permuted),
        "wasser_euclidean": wassterstein_metric(pred_positions, true_positions_permuted)
    }

    print("Positions shape:")
    print(pred_positions, len(pred_positions))
    # print(true_data_holder["positions"], len(true_data_holder["positions"]))

    # NOTE: true_positions need to be permuted and reshaped when calculating distances here, but we store the whole data.

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
        print(f"File saved to {output_dir}/{file}")