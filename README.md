# Curriculum-Based Reinforcement Learning for Combinatorial Optimization

We adapt ACCEL, a curriculum learning algorithm for unsupervised environment design (UED), in the context of solving combinatorial optimization problems. Specifically, we measure the effect of an adaptation of ACCEL on top of an existing state-of-the-art attention-based model for solving the Traveling Salesman Problem (TSP) trained using REINFORCE. We evaluate our method on a synthetically generated test dataset of graphs, as well as a subset of the TSPLib dataset, and show that our implementation of curriculum learning provides performance gains. We also measure the effect of different TSP level edit operations on the performance gain afforded by adding ACCEL.

## Running Experiments

All experiments are run on the codebase in the `attention-learn-to-route` submodule, which also contains more complete documentation. The basics of the commands we used are documented here.

### Environment Setup

To install dependencies with `conda`, run the following to create an environment named `dcd_tsp`:

```
conda env create --file environment.yml
```

### Generating Data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for the TSP:
```
python generate_data.py --problem tsp --name validation --seed 4321
python generate_data.py --problem tsp --name test --seed 1234
```

### Training

For training TSP instances using rollout as a REINFORCE baseline:
```
python run.py --problem tsp --graph_size <NODE-COUNT> --baseline rollout --edit_fn <EDIT-FUNCTION> --epoch_size <EPOCH-SIZE> --batch_size <BATCH-SIZE> --n_epochs <EPOCH-COUNT> --checkpoint_epochs <CHECKPOINT-FREQUENCY> --run_name <NAME>
```

Example usage:
```
python run.py --problem tsp --graph_size 100 --baseline rollout --edit_fn global_perturb --epoch_size 4096 --batch_size 128 --n_epochs 100 --checkpoint_epochs 10 --run_name tsp100_dcd_global
```

Note that if `edit_fn` is omitted, all training data is generated randomly on the fly and there is no curriculum learning in play.

### Evaluation

For evaluating a model (by default the last epoch in the folder is used if no epoch is specified):
```
python eval.py <DATA-FILE> --model <MODEL-FILE> --decode_strategy greedy --eval_batch_size <BATCH-SIZE>
```

Example usage:
```
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_100/tsp100_dcd_global --decode_strategy greedy --eval_batch_size 128
```

Example using a specific epoch and saving to a specific result name (note that setting `--width 0` is necessary for using `-o`):
```
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_100/tsp100_default/epoch-50.pt -o results/tsp/tsp20_test_seed1234_epochs/tsp100_default_epoch-50.pkl --width 0 --decode_strategy greedy --eval_batch_size 128
```

### Help

To view valid command syntax:
```
python run.py -h
python eval.py -h
```

## Formatting Results

All necessary result data is saved in the `/result_data` folder.

Run the following commands to respectively generate the visualizations used in the paper:
```
python plot_edits.py
python plot_path.py
python plot_progress.py
```

Run the following command to calculate the overall path stats from result files:
```
python calc_stats.py
```
