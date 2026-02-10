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
