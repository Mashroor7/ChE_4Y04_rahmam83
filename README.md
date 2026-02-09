Mashroor Rahman's Undergraduate Honours Thesis Project (ChE 4Y04) on the evaluation of KAN variants for fault detection and diagnosis of the Tenessee Eastman Process. This work is conducted under the supervision of Dr. Giancarlo Dalle Ave, with guidance from PhD candidate Jose Daniel Rojas Dorantes, in the Department of Chemical Engineering at McMaster University.

## Project Structure
scripts/      → runnable pipeline entry points  
src/          → core ML modules and logic  
tests/        → validation and unit tests  
requirements.txt → Python dependencies  

## Setup
pip install -r requirements.txt

## Data Setup
Place the H5 file in:
data/raw/tep_data.h5

## Pipeline Overview
1) Load and label data from H5
    - Use load_data.py
2) Apply feature engineering
    - Feature engineering
    - Scaling (Z-scale)
3) Scale features
4) Create train/val/test splits
5) Generate sliding windows
6) Train model
7) Hyperparameter tuning
8) Evaluate performance

## How to Run
python scripts/train.py