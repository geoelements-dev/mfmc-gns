import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

#Load in the data that we will plot.
dat_dir = './outputs/local_full_preds'
filenames = ["rp_full_time_testing.pkl"]
output_dir = './outputs/local_full_preds'


# Append the separated data
results = {}
for filename in filenames:
    with open(f'{dat_dir}/{filename}', 'rb') as file:
        result = pickle.load(file)
        results.update(result)

for k, v in results.items():
    print(f"Working key: {k}")
    true_positions = results[k]["true_positions"][0].permute(1,0,2).numpy()
    pred_positions = results[k]["pred_positions"][0].numpy()
    print("Predicted positions shape: ", pred_positions.shape)
    print("True positions shape: ", true_positions.shape)

    #Initialize the scatterplot with initial data.
    fig, ax = plt.subplots()
    timesteps = np.arange(true_positions.shape[0])
    true_scatter = ax.scatter(true_positions[-1, :, 0], true_positions[-1, :, 1], label='True Positions', s=1)
    pred_scatter = ax.scatter(pred_positions[-1, :, 0], pred_positions[-1, :, 1], label='Predictions', s=1)
    ax.set_title(f"Particle scatterplot for model {k}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 0.3)
    ax.legend(loc='best')

    def update(t):
        '''Function to update the existing plot.'''
        # Get the current time step.
        tx = timesteps[t]

        # Update data.
        true_scatter.set_offsets(true_positions[tx, :, :])
        pred_scatter.set_offsets(pred_positions[tx, :, :])

        return true_scatter, pred_scatter

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=50, blit=True)

    # Save the animation
    # plt.show()
    ani.save(f'scatter_{k}.gif', writer=animation.PillowWriter(fps=30))
