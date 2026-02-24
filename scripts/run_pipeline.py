"""
Complete TEP Data Processing Pipeline

Executes the complete data processing pipeline in the correct order:
1. Data Processing: Extract, split, load, and label data
2. Feature Engineering: Drop zero-variance and analyzer features
3. Scaling: Z-score normalization (fit on train, transform on val/test)

Author: [Your Name]
Created: February 2026
"""

import sys
import random
import numpy as np
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.preprocessing.tep_data_processor import TEPDataProcessor
from src.preprocessing.tep_feature_engineering import apply_feature_engineering
from src.preprocessing.tep_data_scaler import scale_dataframes
from configs.config_loader import load_config
import pandas as pd
import argparse


def run_complete_pipeline(config_path: str = 'configs/config.yaml'):
    """
    Execute the complete TEP data processing pipeline using settings from config.yaml.
    
    Pipeline Steps:
    1. Extract and split data from source H5 (50 runs → 30/10/10 split)
    2. Load and label data (combine processdata + additional_meas, add headers, targets)
    3. Feature Engineering (drop zero-variance features, drop analyzer measurements)
    4. Scaling (Z-score normalization, fit on train, transform on val/test)
    
    Parameters:
    -----------
    config_path : str
        Path to config.yaml file (default: 'config.yaml')
    
    Returns:
    --------
    dict
        Dictionary containing all processed data and file paths
    """
    
    # Load configuration
    config = load_config(config_path)
    config.display()

    # Get paths and settings from config
    source_file = config.raw_source
    output_dir = config.output_dir
    random_seed = config.random_seed

    # Seed all global RNGs before any random operations
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*20 + "TEP DATA PROCESSING PIPELINE")
    print("="*80 + "\n")
    
    # =========================================================================
    # STEP 1: DATA PROCESSING
    # =========================================================================
    print("🔹"*40)
    print("STEP 1/4: DATA PROCESSING")
    print("🔹"*40 + "\n")
    
    processor = TEPDataProcessor(source_file, random_seed=random_seed)
    
    print("Extracting and splitting data...")
    train_file, val_file, test_file = processor.extract_and_split_data(output_dir)
    
    print("\nLoading and labeling data...")
    train_df_raw = processor.load_and_label_data(train_file, split_name='train')
    val_df_raw = processor.load_and_label_data(val_file, split_name='validation')
    test_df_raw = processor.load_and_label_data(test_file, split_name='test')
    
    print("\n✓ Step 1 Complete - Data loaded and labeled")
    print(f"  Train: {train_df_raw.shape}")
    print(f"  Val: {val_df_raw.shape}")
    print(f"  Test: {test_df_raw.shape}")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "🔹"*40)
    print("STEP 2/4: FEATURE ENGINEERING")
    print("🔹"*40 + "\n")
    
    fe_save_path = output_path / 'feature_engineer.pkl'
    
    train_df_fe, val_df_fe, test_df_fe = apply_feature_engineering(
        train_df=train_df_raw,
        val_df=val_df_raw,
        test_df=test_df_raw,
        drop_analyzers=config.drop_analyzers,
        drop_zero_var=config.drop_zero_variance,
        variance_threshold=config.variance_threshold,
        analyzer_list=config.analyzer_features,  # Pass list from config
        save_path=fe_save_path
    )
    
    print("\n✓ Step 2 Complete - Features engineered")
    print(f"  Original features: {train_df_raw.shape[1] - 2}")
    print(f"  After engineering: {train_df_fe.shape[1] - 2}")
    print(f"  Features dropped: {(train_df_raw.shape[1] - 2) - (train_df_fe.shape[1] - 2)}")
    
    # =========================================================================
    # STEP 3: SCALING
    # =========================================================================
    print("\n" + "🔹"*40)
    print("STEP 3/4: SCALING (Z-SCORE NORMALIZATION)")
    print("🔹"*40 + "\n")
    
    scaler_save_path = output_path / 'scaler.pkl'
    
    train_df_scaled, val_df_scaled, test_df_scaled, scaler = scale_dataframes(
        train_df=train_df_fe,
        val_df=val_df_fe,
        test_df=test_df_fe,
        save_scaler_path=scaler_save_path
    )
    
    feature_cols = [col for col in train_df_scaled.columns 
                   if col not in ['Run_ID', 'Target', 'Time']]
    
    print("\n✓ Step 3 Complete - Data scaled")
    print(f"  Features scaled: {len(feature_cols)}")
    print(f"  Preserved columns: Time, Run_ID, Target")
    
    # =========================================================================
    # STEP 4: VERIFICATION
    # =========================================================================
    print("\n" + "🔹"*40)
    print("STEP 4/4: VERIFICATION")
    print("🔹"*40 + "\n")
    
    processor.verify_dataframe(train_df_scaled, 'Train (Final)')
    processor.verify_dataframe(val_df_scaled, 'Validation (Final)')
    processor.verify_dataframe(test_df_scaled, 'Test (Final)')
    
    # =========================================================================
    # SAVE FINAL DATAFRAMES
    # =========================================================================
    if config.save_dataframes:
        print("\n" + "🔹"*40)
        print("SAVING FINAL DATAFRAMES")
        print("🔹"*40 + "\n")
        
        train_df_scaled.to_pickle(output_path / 'train_final.pkl')
        val_df_scaled.to_pickle(output_path / 'val_final.pkl')
        test_df_scaled.to_pickle(output_path / 'test_final.pkl')
        
        print(f"✓ Final DataFrames saved:")
        print(f"  - {output_path / 'train_final.pkl'}")
        print(f"  - {output_path / 'val_final.pkl'}")
        print(f"  - {output_path / 'test_final.pkl'}")
        
        if config.save_csv_sample:
            print(f"\n✓ Saving CSV sample for inspection...")
            train_df_scaled.head(config.csv_sample_rows).to_csv(
                output_path / 'train_final_sample.csv', index=False
            )
            print(f"  - {output_path / 'train_final_sample.csv'} (first {config.csv_sample_rows:,} rows)")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print(" "*30 + "🎉 PIPELINE COMPLETE! 🎉")
    print("="*80)
    
    print("\n📊 PROCESSING SUMMARY:")
    print(f"{'─'*80}")
    print(f"  Original features: {train_df_raw.shape[1] - 2}")
    print(f"  After feature engineering: {train_df_fe.shape[1] - 2}")
    print(f"  Final features (scaled): {len(feature_cols)}")
    print(f"\n  Dataset sizes:")
    print(f"    Train: {train_df_scaled.shape[0]:,} samples × {train_df_scaled.shape[1]} columns")
    print(f"    Val:   {val_df_scaled.shape[0]:,} samples × {val_df_scaled.shape[1]} columns")
    print(f"    Test:  {test_df_scaled.shape[0]:,} samples × {test_df_scaled.shape[1]} columns")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"{'─'*80}")
    print(f"  Raw split files (H5):")
    print(f"    - {train_file.name}")
    print(f"    - {val_file.name}")
    print(f"    - {test_file.name}")
    print(f"\n  Preprocessing objects:")
    print(f"    - {fe_save_path.name}")
    print(f"    - {scaler_save_path.name}")
    
    if config.save_dataframes:
        print(f"\n  Final DataFrames (ready for modeling):")
        print(f"    - train_final.pkl")
        print(f"    - val_final.pkl")
        print(f"    - test_final.pkl")
    
    print(f"\n{'─'*80}")
    print(f"✅ Data is ready for Step 3: Sliding Windows!")
    print(f"{'='*80}\n")
    
    results = {
        'train_df': train_df_scaled,
        'val_df': val_df_scaled,
        'test_df': test_df_scaled,
        'scaler': scaler,
        'feature_engineer_path': fe_save_path,
        'scaler_path': scaler_save_path,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols
    }
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Process Tennessee Eastman Process data - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use default config.yaml in current directory
  python run_pipeline.py
  
  # Use specific config file
  python run_pipeline.py --config path/to/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to config.yaml file (default: configs/config.yaml)'
    )
    
    args = parser.parse_args()
    
    results = run_complete_pipeline(config_path=args.config)
    
    return results


if __name__ == "__main__":
    results = main()
