import numpy as np
import matplotlib.pyplot as plt

# Generate data from uniform distribution
data_original = np.random.rand(50, 2)

# Utility functions for level edits
def global_perturb(arr, percentage=20, max_perturb_degree=0.1):
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(arr.shape[0] * (percentage / 100))
    idx_perturb = np.random.choice(arr.shape[0], num_rows, replace=False)
    arr_new = arr.copy()
    for idx in idx_perturb:
        perturb_mag = np.random.rand(1) * max_perturb_degree
        perturbation = (np.random.uniform(size=arr[idx].shape) - 0.5) * 2 * perturb_mag
        arr_new[idx] = np.clip(arr[idx] + perturbation, 0.0, 1.0)
    return arr_new, idx_perturb

def local_perturb(arr, percentage=20, max_perturb_degree=0.1):
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(arr.shape[0] * (percentage / 100))
    center_idx = np.random.choice(arr.shape[0], 1)[0]
    center_point = arr[center_idx]
    distances = np.linalg.norm(arr - center_point, axis=1)
    idx_perturb = np.argsort(distances)[:num_rows]
    arr_new = arr.copy()
    for idx in idx_perturb:
        perturb_mag = np.random.rand(1) * max_perturb_degree
        perturbation = (np.random.uniform(size=arr[idx].shape) - 0.5) * 2 * perturb_mag
        arr_new[idx] = np.clip(arr[idx] + perturbation, 0.0, 1.0)
    return arr_new, idx_perturb

def random_edit(arr, percentage=20):
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(arr.shape[0] * (percentage / 100))
    idx_edit = np.random.choice(arr.shape[0], num_rows, replace=False)
    arr_new = arr.copy()
    for idx in idx_edit:
        arr_new[idx] = np.random.uniform(size=arr[idx].shape)
    return arr_new, idx_edit

# Perform edits
data_global = global_perturb(data_original, percentage=20)
data_local = local_perturb(data_original, percentage=20)
data_random = random_edit(data_original, percentage=20)

# Utility function for plotting a subplot of a certain edit
def subplot_edit(subplot, data, idx_mod, title):
    subplot.scatter(data[idx_mod,0], data[idx_mod,1], color='red')
    subplot.scatter(data[np.setdiff1d(np.arange(data.shape[0]), idx_mod),0], data[np.setdiff1d(np.arange(data.shape[0]), idx_mod),1], color='black')
    subplot.set_title(title)
    subplot.set_xticks([])
    subplot.set_yticks([])

# Draw plot visualizations
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
plt.subplots_adjust(hspace=0.1)

subplot_edit(axs[0], data_original, [], "Original Level")
subplot_edit(axs[1], data_global[0], data_global[1], "Global Perturb")
subplot_edit(axs[2], data_local[0], data_local[1], "Local Perturb")
subplot_edit(axs[3], data_random[0], data_random[1], "Random Edit")

plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])

plt.savefig("plots/edit_visualizations.png", format='png')
plt.close()
