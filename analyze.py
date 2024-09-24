import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from data_loader import get_pkl_data

frictions = [15, 17.5, 22.5, 30, 37.5, 45]
aspect_ratio_ids = ["027", "046", "054", "069", "082"]
data_dir = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand2d_frictions-sr020/"

# File names
file_names = []
for i in range(len(frictions)):
    for a in aspect_ratio_ids:
        file_name = f"rollout_step7020000_mfmc-a{a}--{i}_ex0.pkl"
        file_names.append(file_name)

# Preprocess data
data_holder = {"aspect_ratio": [], "friction": [], "runout_true": [], "runout_pred": []}
for i, file_name in enumerate(file_names):
    current_data = get_pkl_data(f"{data_dir}/{file_name}")
    for key in data_holder.keys():
        data_holder[key].append(current_data[key])
df = pd.DataFrame(data=data_holder)

print(df["runout_true"].corr(df["runout_pred"]))
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="black")
ax.scatter(df["runout_true"], df["runout_pred"], color="black")
ax.set_xlim([0.45, 1.41])
ax.set_ylim([0.45, 1.41])
ax.set_xlabel("True runout")
ax.set_ylabel("Pred runout")
ax.set_aspect("equal")
plt.show()

