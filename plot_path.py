import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CHANGE THESE PARAMETERS AS NECESSARY
NUM_GRAPHS_EACH = 2
TSP_DATA_FILEPATHS = {
    "tsp20": "result_data/data/tsp/tsp20_test_seed1234.pkl",
    "tsp50": "result_data/data/tsp/tsp50_test_seed1234.pkl",
    "tsp100": "result_data/data/tsp/tsp100_test_seed1234.pkl"
}
RESULT_FOLDERS = {
    "tsp20": "result_data/results/tsp/tsp20_test_seed1234",
    "tsp50": "result_data/results/tsp/tsp50_test_seed1234",
    "tsp100": "result_data/results/tsp/tsp100_test_seed1234"
}
MODELS = {
    "dcd_global": "DCD Global Perturb",
    "dcd_local": "DCD Local Perturb",
    "dcd_random": "DCD Random Edit",
    "default": "Randomized Baseline",
}
COLORS = {
    "dcd_global": "blue",
    "dcd_local": "green",
    "dcd_random": "red",
    "default": "purple",
}

# Utility function for loading a pickle file
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Load in graph and result data
selected_indices = {}
tsp_data = {}
for tsp_name, tsp_filepath in TSP_DATA_FILEPATHS.items():
    # Get problem data and choose random indices to keep
    problem = np.array(load_pickle(tsp_filepath))
    idx = np.random.choice(problem.shape[0], NUM_GRAPHS_EACH, replace=False)
    
    # Record indices and data
    tsp_data[tsp_name] = problem[idx]
    selected_indices[tsp_name] = idx

result_data = {}
for tsp_name, tsp_folder in RESULT_FOLDERS.items():
    for model in MODELS.keys():
        result_filepath = f"{tsp_folder}/{tsp_name}_{model}_results.pkl"
        result_data[(tsp_name, model)] = load_pickle(result_filepath)[0]
        result_data[(tsp_name, model)] = [
            result_data[(tsp_name, model)][i][1] for i in selected_indices[tsp_name]
        ]

# Utility function for plotting a subplot of a certain graph
def subplot_embedding(subplot, tsp_name, idx, title):
    # Get graph and result data
    graph = tsp_data[tsp_name][idx]

    # Plot data
    for model in MODELS.keys():
        result = result_data[(tsp_name, model)][idx]

        # Plot connections between points
        for i in range(len(result) - 1):
            subplot.plot(
                [graph[result[i]][0], graph[result[i+1]][0]],
                [graph[result[i]][1], graph[result[i+1]][1]],
                color=COLORS[model],
                linewidth=0.75
            )
        
        # Connect first and last point
        subplot.plot(
            [graph[result[-1]][0], graph[result[0]][0]],
            [graph[result[-1]][1], graph[result[0]][1]],
            color=COLORS[model],
            linewidth=0.75
        )

    subplot.scatter(graph[:,0], graph[:,1], color='black')

    subplot.set_title(title)
    subplot.set_xticks([])
    subplot.set_yticks([])

# Draw plot visualizations
fig, axs = plt.subplots(NUM_GRAPHS_EACH, 3, figsize=(12, 4 * NUM_GRAPHS_EACH))
plt.subplots_adjust(hspace=0.1)

for i in range(NUM_GRAPHS_EACH):
    for j, tsp_name in enumerate(TSP_DATA_FILEPATHS.keys()):
        print(f"Plotting {tsp_name} Example {i+1}")
        subplot_embedding(
            axs[i, j],
            tsp_name,
            i,
            f"{tsp_name.upper()} Visualization Example {i+1}"
        )

# Add custom legend for plot colors
custom_labels = ['Label 1', 'Label 2', 'Label 3']
custom_colors = ['red', 'green', 'blue']
custom_handles = [
    mpatches.Patch(color=COLORS[model], label=name) for model, name in MODELS.items()
]

fig.legend(handles=custom_handles, loc='upper center', ncol=4)
plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.95])

plt.savefig("plots/path_visualizations.png", format='png')
plt.close()
