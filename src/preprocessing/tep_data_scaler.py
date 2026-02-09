"""
Tennessee Eastman Process (TEP) Data Scaling Module

This module provides utilities for scaling TEP data using StandardScaler.
The Time column is kept unscaled, while all other features are standardized.

Author: [Your Name]
Created: February 2026
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc
import pickle
from typing import Tuple, Optional, List


class TEPDataScaler:
    """
    Handles scaling of Tennessee Eastman Process data.
    
    Fits a StandardScaler on training data and applies it to train/val/test sets.
    Time column is kept unscaled. Run_ID and Target are preserved as-is.
    """
    
    def __init__(self):
        """Initialize the TEP Data Scaler."""
        self.scaler = None
        self.feature_columns = None
        self.IDV_RANGE = range(1, 29)
    
    def fit_scaler_dataframe(self, train_df: pd.DataFrame) -> StandardScaler:
        """
        Fit StandardScaler on training DataFrame (excluding Time, Run_ID, Target).
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training DataFrame
        
        Returns:
        --------
        StandardScaler
            Fitted scaler object
        """
        print("="*70)
        print("FITTING SCALER ON TRAINING DATA (DataFrame)")
        print("="*70)
        print("NOTE: Time, Run_ID, and Target columns will NOT be scaled\n")
        
        # Identify feature columns (exclude Time, Run_ID, Target)
        self.feature_columns = [col for col in train_df.columns 
                               if col not in ['Run_ID', 'Target']]
        
        print(f"Feature columns to scale: {len(self.feature_columns)}")
        print(f"  Excluded: Run_ID, Target")
        
        # Extract feature data
        feature_data = train_df[self.feature_columns].values
        
        print(f"\nTraining data shape: {feature_data.shape}")
        print(f"Memory usage: {feature_data.nbytes / 1024**2:.2f} MB")
        
        # Fit the scaler
        print(f"\nFitting StandardScaler on {len(self.feature_columns)} features...")
        self.scaler = StandardScaler()
        self.scaler.fit(feature_data)
        
        print(f"✓ Scaler fitted successfully")
        print(f"  - Mean shape: {self.scaler.mean_.shape}")
        print(f"  - Std shape: {self.scaler.scale_.shape}")
        print(f"  - Sample means (first 5): {self.scaler.mean_[:5]}")
        print(f"  - Sample stds (first 5): {self.scaler.scale_[:5]}\n")
        
        return self.scaler
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted scaler.
        Time, Run_ID, and Target columns are preserved as-is.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit_scaler_dataframe() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Call fit_scaler_dataframe() first.")
        
        # Create a copy to avoid modifying original
        df_scaled = df.copy()
        
        # Scale only the feature columns
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_scaled
    
    def verify_scaling_dataframe(self, df: pd.DataFrame, split_name: str = "Data"):
        """
        Verify that scaling was applied correctly to a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Scaled DataFrame to verify
        split_name : str
            Name of the split for display
        """
        print(f"\nVerifying scaling for: {split_name}")
        
        # Check Time column (should NOT be scaled)
        if 'Time' in df.columns:
            time_col = df['Time']
            print(f"  Time column - should be UNSCALED:")
            print(f"    Min: {time_col.min():.2f}, Max: {time_col.max():.2f}")
            print(f"    Mean: {time_col.mean():.2f}, Std: {time_col.std():.2f}")
            
            if 0 <= time_col.min() and time_col.max() < 200:
                print(f"    ✓ Time column appears unscaled (reasonable values)")
            else:
                print(f"    ⚠ Warning: Time column may have unexpected values")
        
        # Check scaled features
        if self.feature_columns:
            scaled_features = df[self.feature_columns]
            sample_mean = scaled_features.mean().mean()
            sample_std = scaled_features.std().mean()
            
            print(f"\n  Scaled features ({len(self.feature_columns)} columns):")
            print(f"    Mean (average across features): {sample_mean:.6f}")
            print(f"    Std (average across features): {sample_std:.6f}")
            print(f"    Expected: mean ≈ 0, std ≈ 1")
            
            if abs(sample_mean) < 0.1 and abs(sample_std - 1.0) < 0.3:
                print(f"    ✓ Scaling appears correct\n")
            else:
                print(f"    ⚠ Warning: scaling may not be optimal\n")
    
        """
        Fit StandardScaler on all training data (excluding first column - time).
        
        Parameters:
        -----------
        train_file_path : str or Path
            Path to training HDF5 file
        
        Returns:
        --------
        StandardScaler
            Fitted scaler object
        """
        print("="*70)
        print("FITTING SCALER ON TRAINING DATA")
        print("="*70)
        print("NOTE: First column (Time) will NOT be scaled\n")
        
        data_list = []
        
        with h5py.File(train_file_path, 'r') as f:
            # Iterate through all IDV groups
            for idv_num in tqdm(self.IDV_RANGE, desc="Loading training data"):
                idv_key = f'IDV{idv_num}'
                
                if idv_key not in f:
                    continue
                
                idv_group = f[idv_key]
                run_keys = [key for key in idv_group.keys() if key.startswith('Run')]
                
                # Load all runs from this IDV
                for run_key in run_keys:
                    combined_data = idv_group[run_key][:]
                    # Exclude first column (time) from scaling
                    features = combined_data[:, 1:]
                    data_list.append(features)
        
        # Concatenate all arrays into single array
        print(f"\n✓ Concatenating {len(data_list)} arrays...")
        all_data = np.vstack(data_list)
        print(f"✓ Combined training data shape (excluding time): {all_data.shape}")
        print(f"✓ Memory usage: {all_data.nbytes / 1024**2:.2f} MB")
        
        # Clear the list to free memory
        del data_list
        gc.collect()
        
        # Fit the scaler
        print(f"\nFitting StandardScaler on columns 1-{all_data.shape[1]} (excluding column 0: Time)...")
        self.scaler = StandardScaler()
        self.scaler.fit(all_data)
        
        print(f"✓ Scaler fitted successfully")
        print(f"  - Mean shape: {self.scaler.mean_.shape}")
        print(f"  - Std shape: {self.scaler.scale_.shape}")
        print(f"  - Sample means (first 5): {self.scaler.mean_[:5]}")
        print(f"  - Sample stds (first 5): {self.scaler.scale_[:5]}")
        
        # Delete the combined array to free memory
        del all_data
        gc.collect()
        print("\n✓ Training data cleared from memory\n")
        
        return self.scaler
    
    def transform_and_save(self, input_file: str, output_file: str, 
                          scaler: Optional[StandardScaler] = None):
        """
        Transform data using fitted scaler and save to new HDF5 file.
        First column (time) is kept unscaled.
        
        Parameters:
        -----------
        input_file : str or Path
            Path to input HDF5 file
        output_file : str or Path
            Path to output HDF5 file
        scaler : StandardScaler, optional
            Fitted scaler object. If None, uses self.scaler
        """
        if scaler is None:
            scaler = self.scaler
        
        if scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit_scaler() first or provide a scaler.")
        
        print(f"\nTransforming: {Path(input_file).name} -> {Path(output_file).name}")
        print(f"  (Time column will remain unscaled)")
        
        with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:
            
            # Copy Feature_Labels to output file
            if 'Feature_Labels' in src:
                feature_labels = src['Feature_Labels'][:]
                dst.create_dataset('Feature_Labels', data=feature_labels)
                print(f"  ✓ Copied Feature_Labels")
            
            # Iterate through all IDV groups
            for idv_num in tqdm(self.IDV_RANGE, desc=f"  Processing"):
                idv_key = f'IDV{idv_num}'
                
                if idv_key not in src:
                    continue
                
                src_idv_group = src[idv_key]
                run_keys = [key for key in src_idv_group.keys() if key.startswith('Run')]
                
                if len(run_keys) == 0:
                    continue
                
                # Create IDV group in destination
                dst_idv_group = dst.create_group(idv_key)
                
                # Transform and save each run
                for run_key in run_keys:
                    # Load original data
                    combined_data = src_idv_group[run_key][:]
                    
                    # Separate time column (first column) and other features
                    time_column = combined_data[:, 0:1]  # Keep as 2D array
                    features = combined_data[:, 1:]  # All other columns
                    
                    # Transform only the features (excluding time)
                    features_scaled = scaler.transform(features)
                    
                    # Concatenate time column (unscaled) with scaled features
                    data_scaled = np.hstack([time_column, features_scaled])
                    
                    # Save to output file
                    dst_idv_group.create_dataset(run_key, data=data_scaled)
        
        print(f"  ✓ Saved scaled data to {Path(output_file).name}\n")
    
    def verify_scaling(self, scaled_file: str, sample_size: int = 1000):
        """
        Verify that scaling was applied correctly by checking statistics.
        Also verifies that the first column (time) remains unscaled.
        
        Parameters:
        -----------
        scaled_file : str or Path
            Path to scaled HDF5 file
        sample_size : int
            Not currently used, kept for future enhancements
        """
        print(f"Verifying scaling for: {Path(scaled_file).name}")
        
        sample_data = []
        
        with h5py.File(scaled_file, 'r') as f:
            # Sample from first IDV only for quick verification
            idv_key = 'IDV1'
            if idv_key in f:
                idv_group = f[idv_key]
                run_keys = list(idv_group.keys())[:2]  # Just check first 2 runs
                
                for run_key in run_keys:
                    data = idv_group[run_key][:]
                    sample_data.append(data)
        
        if len(sample_data) == 0:
            print(f"  ⚠ Could not verify: no data found\n")
            return
        
        sample_array = np.vstack(sample_data)
        
        # Check time column (first column - should NOT be scaled)
        time_column = sample_array[:, 0]
        print(f"  Time column (col 0) - should be UNSCALED:")
        print(f"    Min: {time_column.min():.2f}, Max: {time_column.max():.2f}")
        print(f"    Mean: {time_column.mean():.2f}, Std: {time_column.std():.2f}")
        
        # Check if time looks reasonable (should be 0 to ~100 hours)
        if 0 <= time_column.min() and time_column.max() < 200:
            print(f"    ✓ Time column appears unscaled (reasonable values)")
        else:
            print(f"    ⚠ Warning: Time column may have unexpected values")
        
        # Check scaled features (columns 1 onwards)
        scaled_features = sample_array[:, 1:]
        sample_mean = np.mean(scaled_features, axis=0)
        sample_std = np.std(scaled_features, axis=0)
        
        print(f"\n  Scaled features (cols 1-{scaled_features.shape[1]}):")
        print(f"    Sample shape: {scaled_features.shape}")
        print(f"    Sample mean (first 5 features): {sample_mean[:5]}")
        print(f"    Sample std (first 5 features): {sample_std[:5]}")
        print(f"    Expected: means ≈ 0, stds ≈ 1")
        
        # Check if means are close to 0 and stds close to 1
        mean_close_to_zero = np.abs(sample_mean[:5]).mean() < 0.5
        std_close_to_one = np.abs(sample_std[:5] - 1).mean() < 0.5
        
        if mean_close_to_zero and std_close_to_one:
            print(f"    ✓ Scaling appears correct\n")
        else:
            print(f"    ⚠ Warning: scaling may not be optimal\n")
    
    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where scaler will be saved
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """
        Load a fitted scaler from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to saved scaler file
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")
        return self.scaler


def scale_dataframes(train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame,
                     save_scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TEPDataScaler]:
    """
    Scale train/val/test DataFrames using StandardScaler.
    
    Fits scaler on training data and applies to all splits.
    Time, Run_ID, and Target columns are preserved as-is.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    val_df : pd.DataFrame
        Validation DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    save_scaler_path : str, optional
        Path to save the fitted scaler
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TEPDataScaler]
        Scaled (train, val, test) DataFrames and the scaler object
    """
    
    print("\n" + "="*70)
    print("SCALING DATAFRAMES")
    print("="*70 + "\n")
    
    # Initialize scaler
    scaler_obj = TEPDataScaler()
    
    # Fit on training data
    print("Step 1: Fitting scaler on training data...")
    scaler_obj.fit_scaler_dataframe(train_df)
    
    # Transform all datasets
    print("Step 2: Transforming datasets...")
    print("  Transforming training data...")
    train_scaled = scaler_obj.transform_dataframe(train_df)
    
    print("  Transforming validation data...")
    val_scaled = scaler_obj.transform_dataframe(val_df)
    
    print("  Transforming test data...")
    test_scaled = scaler_obj.transform_dataframe(test_df)
    
    # Verify scaling
    print("\nStep 3: Verifying scaling...")
    scaler_obj.verify_scaling_dataframe(train_scaled, "Training Set")
    scaler_obj.verify_scaling_dataframe(val_scaled, "Validation Set")
    scaler_obj.verify_scaling_dataframe(test_scaled, "Test Set")
    
    # Save scaler if path provided
    if save_scaler_path:
        scaler_obj.save_scaler(save_scaler_path)
    
    print("="*70)
    print("✓ SCALING COMPLETE")
    print("="*70 + "\n")
    
    return train_scaled, val_scaled, test_scaled, scaler_obj


def main_scaling_example():
    """
    Complete scaling pipeline for TEP datasets.
    
    Parameters:
    -----------
    train_file : str
        Path to training HDF5 file
    val_file : str
        Path to validation HDF5 file
    test_file : str
        Path to test HDF5 file
    output_dir : str, optional
        Directory for scaled output files. If None, uses parent directory of train_file
    save_scaler_path : str, optional
        Path to save the fitted scaler. If None, scaler is not saved.
    
    Returns:
    --------
    Tuple[Path, Path, Path]
        Paths to (train_scaled, val_scaled, test_scaled) files
    """
    
    print("\n" + "="*70)
    print("TENNESSEE EASTMAN PROCESS - SCALING PIPELINE")
    print("="*70 + "\n")
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(train_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    train_scaled = output_dir / f"{Path(train_file).stem}_scaled.h5"
    val_scaled = output_dir / f"{Path(val_file).stem}_scaled.h5"
    test_scaled = output_dir / f"{Path(test_file).stem}_scaled.h5"
    
    # Initialize scaler
    scaler_obj = TEPDataScaler()
    
    # Step 1: Fit scaler on training data
    print("STEP 1: FITTING SCALER")
    print("="*70)
    scaler = scaler_obj.fit_scaler(train_file)
    
    # Save scaler if path provided
    if save_scaler_path:
        scaler_obj.save_scaler(save_scaler_path)
    
    # Step 2: Transform and save all datasets
    print("\n" + "="*70)
    print("STEP 2: TRANSFORMING AND SAVING DATASETS")
    print("="*70)
    
    scaler_obj.transform_and_save(train_file, train_scaled, scaler)
    scaler_obj.transform_and_save(val_file, val_scaled, scaler)
    scaler_obj.transform_and_save(test_file, test_scaled, scaler)
    
    # Step 3: Verification
    print("="*70)
    print("STEP 3: VERIFICATION")
    print("="*70 + "\n")
    
    scaler_obj.verify_scaling(train_scaled)
    scaler_obj.verify_scaling(val_scaled)
    scaler_obj.verify_scaling(test_scaled)
    
    print("="*70)
    print("✓ SCALING COMPLETE!")
    print("="*70)
    print("Output files created:")
    print(f"  - {train_scaled}")
    print(f"  - {val_scaled}")
    print(f"  - {test_scaled}")
    if save_scaler_path:
        print(f"\nScaler saved to:")
        print(f"  - {save_scaler_path}")
    print("="*70 + "\n")
    
    return train_scaled, val_scaled, test_scaled


def main_scaling_example():
    """
    Example usage for scaling TEP datasets.
    """
    
    # Define file paths - UPDATE THESE
    base_dir = Path(r'C:\path\to\data')  # UPDATE THIS PATH
    
    train_file = base_dir / 'train_30runs.h5'
    val_file = base_dir / 'val_10runs.h5'
    test_file = base_dir / 'test_10runs.h5'
    
    scaler_save_path = base_dir / 'tep_scaler.pkl'
    
    # Run scaling pipeline
    train_scaled, val_scaled, test_scaled = scale_tep_datasets(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        output_dir=base_dir,
        save_scaler_path=scaler_save_path
    )
    
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    train_scaled, val_scaled, test_scaled = main_scaling_example()
