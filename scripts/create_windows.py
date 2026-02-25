"""
TEP Sliding Window Creation Script

Loads fully processed data (train_final.pkl, val_final.pkl, test_final.pkl)
and creates sliding windows ready for model training.

All settings (paths, window_size, stride) are read from configs/config.yaml.

Usage:
    python scripts/create_windows.py
    python scripts/create_windows.py --config configs/config.yaml

Author: [Your Name]
Created: February 2026
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_loader import load_config
from src.windowing.tep_windowing import create_all_windows, save_windows


def run_windowing_pipeline(config_path: str = 'configs/config.yaml'):
    """
    Load processed DataFrames and create sliding windows.

    Reads all settings from config.yaml:
        - data.output_dir        → where to find train/val/test_final.pkl
        - windowing.window_size  → timesteps per window
        - windowing.stride       → step between windows
        - windowing.save_metadata → whether to save Run_ID/index info

    Parameters:
    -----------
    config_path : str
        Path to config.yaml file

    Returns:
    --------
    dict
        Windowed data for train/val/test splits
    """

    # Load configuration
    config = load_config(config_path)
    config.display()

    # Paths from config
    processed_dir = Path(config.output_dir)
    windows_dir   = processed_dir / 'windows'

    train_pkl = processed_dir / 'train_final.pkl'
    val_pkl   = processed_dir / 'val_final.pkl'
    test_pkl  = processed_dir / 'test_final.pkl'

    # Windowing settings from config
    window_size   = config.window_size
    stride        = config.stride
    save_metadata = config.save_metadata

    print("\n" + "="*70)
    print(" "*20 + "TEP WINDOWING PIPELINE")
    print("="*70)
    print(f"\n  Input directory:  {processed_dir}")
    print(f"  Output directory: {windows_dir}")
    print(f"  Window size:      {window_size}")
    print(f"  Stride:           {stride}")
    print(f"  Save metadata:    {save_metadata}")
    print("="*70 + "\n")

    # =========================================================================
    # STEP 1: Load processed DataFrames
    # =========================================================================
    print("🔹"*35)
    print("STEP 1/3: LOADING PROCESSED DATA")
    print("🔹"*35 + "\n")

    for pkl_path in [train_pkl, test_pkl]:  # val is optional
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"\n❌ File not found: {pkl_path}"
                f"\nHave you run scripts/run_pipeline.py first?"
            )

    print("  Loading train_final.pkl...")
    train_df = pd.read_pickle(train_pkl)
    print(f"  ✓ Train: {train_df.shape}")

    if val_pkl.exists():
        print("  Loading val_final.pkl...")
        val_df = pd.read_pickle(val_pkl)
        print(f"  ✓ Val:   {val_df.shape}")
    else:
        print("  val_final.pkl not found — skipping validation split")
        val_df = None

    print("  Loading test_final.pkl...")
    test_df = pd.read_pickle(test_pkl)
    print(f"  ✓ Test:  {test_df.shape}")

    # =========================================================================
    # STEP 2: Create sliding windows
    # =========================================================================
    print("\n" + "🔹"*35)
    print("STEP 2/3: CREATING SLIDING WINDOWS")
    print("🔹"*35)

    windows_data = create_all_windows(
        train_df=train_df,
        val_df=val_df,     # None when val_runs=0 — windowing skips it
        test_df=test_df,
        window_size=window_size,
        stride=stride
    )

    # =========================================================================
    # STEP 3: Save windows
    # =========================================================================
    print("🔹"*35)
    print("STEP 3/3: SAVING WINDOWS")
    print("🔹"*35)

    saved_paths = save_windows(
        windows_data=windows_data,
        output_dir=windows_dir,
        window_size=window_size,
        stride=stride,
        save_metadata=save_metadata
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("="*70)
    print(" "*25 + "🎉 WINDOWING COMPLETE! 🎉")
    print("="*70)

    X_train = windows_data['train'][0]
    X_val   = windows_data['val'][0] if 'val' in windows_data else None
    X_test  = windows_data['test'][0]

    print(f"\n  Window shape:  (n_windows, {window_size} × {X_train.shape[1] // window_size})")
    print(f"                  n_windows × (window_size × n_features)  ← flattened for KAN\n")
    print(f"  {'Split':<10} {'Windows':>12}   {'File'}")
    print(f"  {'-'*60}")
    for split, path in saved_paths.items():
        n = len(windows_data[split][0])
        print(f"  {split:<10} {n:>12,}   {path.name}")

    print(f"\n  Files saved to: {windows_dir}")
    print(f"\n  ✅ Ready for Step 4: Model Training!")
    print("="*70 + "\n")

    return windows_data


def main():
    parser = argparse.ArgumentParser(
        description='Create sliding windows from processed TEP data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config.yaml (default: configs/config.yaml)'
    )
    args = parser.parse_args()

    run_windowing_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()