import numpy as np
import matplotlib.pyplot as plt

# Constants
NODE_OPTIONS = [20, 50, 100]
EPOCH_OPTIONS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
MODEL_OPTIONS = ["dcd_global", "dcd_local", "dcd_random", "default"]
MODELS = {
    "dcd_global": "DCD Global Perturb",
    "dcd_local": "DCD Local Perturb",
    "dcd_random": "DCD Random Edit",
    "default": "Randomized Baseline",
}

# Initialize results
means = {}
stds = {}
for model in MODEL_OPTIONS:
    mean_filepath = f"result_data/results/tsp/epoch_results/{model}_means.npy"
    std_filepath = f"result_data/results/tsp/epoch_results/{model}_stds.npy"
    means[model] = np.load(mean_filepath)
    stds[model] = np.load(std_filepath)

# Utility function for plotting a subplot of a certain graph
def subplot_progress(subplot, node_idx, title):
    for model in MODEL_OPTIONS:
        result = means[model][node_idx]
        subplot.plot(EPOCH_OPTIONS, result, label=MODELS[model])
    subplot.legend()
    subplot.set_title(title)
    subplot.set_xlabel("Training Epoch")
    subplot.set_ylabel("Test Cost")

# Draw plot visualizations
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(hspace=0.1)

for i in range(len(NODE_OPTIONS)):
    print(f"Plotting tsp_{NODE_OPTIONS[i]}")
    subplot_progress(
        axs[i],
        i,
        f"Training Epoch vs. Test Cost on TSP{NODE_OPTIONS[i]}"
    )

plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])

plt.savefig("plots/progress_visualizations.png", format='png')
plt.close()
