import pickle
import numpy as np

# Constants
RESULT_FOLDERS = {
    "tsp20": "result_data/results/tsp/tsp20_test_seed1234",
    "tsp50": "result_data/results/tsp/tsp50_test_seed1234",
    "tsp100": "result_data/results/tsp/tsp100_test_seed1234"
}
MODELS = ["dcd_global", "dcd_local", "dcd_random", "default"]

# Utility function for loading a pickle file
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Load in graph and result data
result_data = {}
for tsp_name, tsp_folder in RESULT_FOLDERS.items():
    for model in MODELS:
        result_filepath = f"{tsp_folder}/{tsp_name}_{model}_results.pkl"
        result = load_pickle(result_filepath)[0]

        arr = np.zeros(len(result))
        for i in range(len(result)):
            arr[i] = result[i][0]
        result_data[(tsp_name, model)] = arr

# Print out data
for tsp_name in RESULT_FOLDERS.keys():
    print(f"{tsp_name} costs:")
    for model in MODELS:
        costs = result_data[(tsp_name, model)]
        print(f"{model}: {np.round(np.mean(costs), 5)} +/- {np.round(np.std(costs), 5)}")
    print()
