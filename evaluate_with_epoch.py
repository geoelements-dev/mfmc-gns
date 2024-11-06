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


#Define distance loss metrics to calculate corrcoefs with
def chamfer_distance(points_a, points_b):
    #Inputs are Tensors that are still on the device, hence the following processing steps.
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # points_a_exp = points_a.unsqueeze(dim=1)
    # points_b_exp = points_b.unsqueeze(dim=0)
    # distances = torch.sum((points_a_exp - points_b_exp) ** 2, dim=-1)
    # min_dist_a_to_b = torch.min(distances, dim=1)[0]
    # min_dist_b_to_a = torch.min(distances, dim=0)[0]
    # chamfer_dist = torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)

    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.sum((a_exp - b_exp)**2, dim=-1)

        min_a_to_b = torch.mean(torch.min(distances, dim=1)[0])
        min_b_to_a = torch.mean(torch.min(distances, dim=0)[0])

        chamfer_dist = min_a_to_b + min_b_to_a
        elementwise_distances.append(chamfer_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Chamfer distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)

    # pairwise_distances = []
    # for a in points_a:
    #     row = []
    #     for b in points_b:
    #         dist = torch.sum((a-b)**2)
    #         row.append(dist)
    #     pairwise_distances.append(torch.stack(row))
    # dists = torch.stack(pairwise_distances)
    # min_dist_a_to_b = torch.min(dists, dim=1)[0]
    # min_dist_b_to_a = torch.min(dists, dim=0)[0]
    # chamfer_dist = torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)
    
    return chamfer_dist

def hausdorff_distance(points_a, points_b):
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # a_exp = points_a.unsqueeze(dim=1)
    # b_exp = points_b.unsqueeze(dim=0)
    # distances = torch.sum((a_exp - b_exp) **2, dim=-1)
    # haus_a_to_b = torch.max(torch.min(distances, dim=1)[0])
    # haus_b_to_a = torch.max(torch.min(distances, dim=0)[0])
    # hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)


    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.sum((a_exp - b_exp)**2, dim=-1)

        haus_a_to_b = torch.max(torch.min(distances, dim=1)[0])
        haus_b_to_a = torch.max(torch.min(distances, dim=0)[0])

        hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)
        elementwise_distances.append(hausdorff_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Hausdorff distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)

    # pairwise_distances = []
    # for a in points_a:
    #     row = []
    #     for b in points_b:
    #         dist = torch.sum((a-b)**2)
    #         row.append(dist)
    #     pairwise_distances.append(torch.stack(row))
    # dists = torch.stack(pairwise_distances)
    # haus_a_to_b = torch.max(torch.min(dists, dim=1)[0])
    # haus_b_to_a = torch.max(torch.min(dists, dim=0)[0])
    # hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)

    # return hausdorff_dist

def wassterstein_metric(points_a, points_b, p=2):
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # a_exp = points_a.unsqueeze(dim=1)
    # b_exp = points_b.unsqueeze(dim=0)
    # if p==2:
    #     distances = torch.sqrt(torch.sum((a_exp - b_exp)**p, dim=-1))
    # elif p==1:
    #     distances = torch.sum(torch.abs(a_exp - b_exp), dim=-1)
    # else:
    #     print(f"Error: Wassterstein distance not implemented for input p={p}. Valid choices are [1,2].")
    #     return

    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.norm(a_exp - b_exp, p=p, dim=-1)

        wass_a_to_b = torch.mean(torch.min(distances, dim=1)[0])
        wass_b_to_a = torch.mean(torch.min(distances, dim=0)[0])

        wass_dist = torch.max(wass_a_to_b, wass_b_to_a)
        elementwise_distances.append(wass_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Wasserstein distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)
    
    # distances = []
    # if p == 2:
    #     for a in points_a:
    #         row = []
    #         for b in points_b:
    #             dist = torch.sqrt(torch.sum((a-b)**p))
    #             row.append(dist)
    #         distances.append(torch.stack(row))
    # elif p == 1:
    #     for a in points_a:
    #         row = []
    #         for b in points_b:
    #             dist = torch.sum(torch.abs((a-b)))
    #             row.append(dist)
    #         distances.append(torch.stack(row))
    # distances = torch.stack(distances)
    
    # a_dist_sorted, a_idx = torch.sort(torch.min(distances, dim=1)[0])
    # b_dist_sorted, b_idx = torch.sort(torch.min(distances, dim=0)[0])
    # wasser_dist = torch.mean(torch.abs(a_dist_sorted - b_dist_sorted))

    # return wasser_dist


# Set epochs to analyze
epochs = np.arange(0, 2000000, 100000)
# epochs = np.arange(2000000, 6000000, 100000)
plot_step = 2  # plot interval while evaluating through epochs
# output_dir = '/work2/08264/baagee/frontera/mfmc-gns/outputs/'
output_dir = './outputs'
# output_file = 'eval_new_format-time.pkl'
# output_file = 'rp_eval_0_to_2000k.pkl'
output_file = 'rp_eval_0_to_2000k.pkl'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Simulator configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# simulator_metadata_path = ' /corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/'
simulator_metadata_path = '../sand2d_frictions-sr020/'
noise_std = 6.7e-4
# model_path = ' /corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/'
model_path = '../sand2d_frictions-sr020/'
INPUT_SEQUENCE_LENGTH = 6

# Get ground truth data paths
aspect_ratio_ids = ["027", "046", "054", "069", "082"]
# aspect_ratio_ids = ["027", "046"]
# data_dir = "/corral/utexas/Material-Point-Metho/baagee/frontera/gns-mpm-data/mpm/mfmc/"
data_dir = "../sand2d_frictions-sr020"
friction_ids = [0, 2, 3, 4, 5]  # [(0, 15), (1, 17.5), (2, 22.5), (3, 30), (4, 37.5), (5, 45)]  # id-friction pare
# for friction_id in [0, 2, 3, 4, 5]:
true_npz_files = []
for friction_id in friction_ids:
    for a_id in aspect_ratio_ids:
        true_npz_file = f"{data_dir}/mfmc-a{a_id}-{friction_id}.npz"
        true_npz_files.append(true_npz_file)

# Get ground truth values
true_data_holder = {"aspect_ratio": [], "friction": [], "runout_true": [], "positions": [], "particle_type": [],
                    "material_property": [], "n_particles_per_example": []}
print(list(true_data_holder.keys()))
for i, file_path in enumerate(true_npz_files):
    # current_data = data_loader.get_npz_data(file_path, option="runout_only")
    current_data = data_loader.get_npz_data(file_path, option="runout_only")
    # print("Current data keys: ", list(current_data.keys()))
    for key in list(true_data_holder.keys())[:3]:
        # print("Looping key: ", key)
        # print(true_data_holder[f"{key}"])
        true_data_holder[f"{key}"].append(current_data[f"{key}"])
    current_data_pos = data_loader.get_npz_data(file_path, option="entire_data")
    for key in list(true_data_holder.keys())[3:]:
        # print("Looping key 2: ", key)
        # print(true_data_holder[f"{key}"])
        true_data_holder[f"{key}"].append(current_data_pos[f"{key}"])

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
    # print(pred_positions, len(pred_positions))
    print(len(pred_positions))
    print("First position shape: ", pred_positions[0].shape)
    # print(true_data_holder["positions"], len(true_data_holder["positions"]))

    #PLEASE NOTE: true_positions need to be permuted and reshaped when calculating distances here, but we store the whole data.

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





