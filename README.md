Mashroor Rahman's Undergraduate Honours Thesis Project (ChE 4Y04) on the evaluation of KAN variants for fault detection and diagnosis of the Tenessee Eastman Process. This work is conducted under the supervision of Dr. Giancarlo Dalle Ave, with guidance from PhD candidate Jose Daniel Rojas Dorantes, in the Department of Chemical Engineering at McMaster University.

## Project Structure
scripts/      → runnable pipeline entry points  
src/          → core ML modules and logic  
tests/        → validation and unit tests  
requirements.txt → Python dependencies  

## Setup
1) Install dependencies: python -m pip install -r requirements.txt
2) Place the raw H5 file in: data/raw/tep_data.h5
2) Edit config.yaml


## Pipeline Overview
1) Load and label data from H5 (load_data.py)
    - Loads "processdata" and "additional_meas" into a split of 30, 10, 10 runs for training, validation, and testing set.
    - Label data with header row from "Processdata_Labels" and "Additional_Meas_Labels"
    - Add target column for classifcation of timestep: healthy (0), IDV(X) (X)
        -Healthy data is indices 0-599
        -Faulty data is indices 600-2001
2) Perform pre-processing and feature engineering
    - Feature engineering
        -Drop features with zero variance
        -Drop features corresponding with analyzer measurements (Not realistic to have in time for real-life application of model)
    - Scaling (Z-scale)
        -Perform sclaing on training dataset with transformation applied to validation and testing dataset
3) Generate sliding windows
    -Create sliding windows of length 5 with a stride of 1
4) Train model
5) Hyperparameter tuning
6) Evaluate performance

---

## Experiment 1 — Hyperparameter Tuning (30 runs/IDV)

Uses 50 runs per IDV (30 train / 10 val / 10 test) to find the best hyperparameters
for each KAN variant via Optuna (30 trials per model).

**Config settings** (`configs/config.yaml`):
```yaml
data:
  output_dir: 'data\processed'
splits:
  total_runs: 50
  train_runs: 30
  val_runs: 10
  test_runs: 10
models:
  weights_dir: 'results'
```

**Run order:**

```bash
# 1. Extract and preprocess data (30 runs/IDV, 30/10/10 split)
python scripts/run_pipeline.py

# 2. Create sliding windows (train, val, test)
python scripts/create_windows.py

# 3. Tune each KAN variant (Optuna, 30 trials each)
python scripts/tune.py --model efficient_kan
python scripts/tune.py --model fourier_kan
python scripts/tune.py --model wavelet_kan
python scripts/tune.py --model fast_kan

# 4. Evaluate results
python scripts/evaluate.py
```

Outputs per model saved to `results/<model>/`:
- `best_params.json` — best hyperparameters found
- `best_model.pt` — retrained model weights
- `predictions.npz` — test set predictions
- `metrics.json` — val and test accuracy

---

## Experiment 2 — Full Dataset Training (200 runs/IDV)

Trains all four KAN variants on the full dataset (200 runs per IDV, 160 train / 40 test)
using the best hyperparameters found in Experiment 1. No hyperparameter search is
performed — `best_params.json` from `results/` is loaded directly.

**Prerequisite:** Experiment 1 must be completed. The following files must exist:
```
results/efficient_kan/best_params.json
results/fourier_kan/best_params.json
results/wavelet_kan/best_params.json
results/fast_kan/best_params.json
```

**Config settings** (`configs/config.yaml`):
```yaml
data:
  output_dir: 'data\processed_full'
splits:
  total_runs: 200
  train_runs: 160
  val_runs: 0
  test_runs: 40
models:
  weights_dir: 'results_full'
```

**Run order:**

```bash
# 1. Extract and preprocess data (200 runs/IDV, 160/40 split, no validation)
python scripts/run_pipeline.py

# 2. Create sliding windows (train and test only)
python scripts/create_windows.py

# 3. Train all four models with tuned hyperparameters (100 epochs, no early stopping)
#    --params-dir points to best_params.json files from Experiment 1
python scripts/train_best.py --all --params-dir results

# 4. Evaluate results
python scripts/evaluate.py
```

To train a single variant instead of all four:
```bash
python scripts/train_best.py --model wavelet_kan --params-dir results
```

Outputs per model saved to `results_full/<model>/`:
- `best_model.pt` — final trained model weights
- `predictions.npz` — test set predictions
- `metrics.json` — test accuracy and training details
