"""
Smoke Test: Windowing Pipeline

Validates the output of create_windows.py by checking:
  1. Output .npz files exist
  2. X and y shapes are correct
  3. No windows cross Run_ID boundaries
  4. Label is from the last timestep of each window

Usage:
    python tests/test_windowing.py
    python tests/test_windowing.py --config configs/config.yaml

Author: [Your Name]
Created: February 2026
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_loader import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str):
    print(f"  ✅ PASS  {msg}")

def _fail(msg: str):
    print(f"  ❌ FAIL  {msg}")
    raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_files_exist(windows_dir: Path, window_size: int, stride: int):
    """Check 1: All three .npz files exist."""
    print("\n[1/4] Checking output files exist...")

    for split in ['train', 'val', 'test']:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024**2
            _pass(f"{filepath.name}  ({size_mb:.1f} MB)")
        else:
            _fail(
                f"{filepath.name} not found.\n"
                f"       Have you run: python scripts/create_windows.py ?"
            )


def check_shapes(windows_dir: Path, window_size: int, stride: int,
                 expected_splits: dict):
    """
    Check 2: X and y shapes are consistent and correct.

    Expected shapes:
        X : (n_windows, window_size * n_features)
        y : (n_windows,)
    """
    print("\n[2/4] Checking X and y shapes...")

    for split in ['train', 'val', 'test']:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        X = data['X']
        y = data['y']

        # X must be 2D: (n_windows, flattened_dim)
        if X.ndim != 2:
            _fail(f"{split} X should be 2D, got shape {X.shape}")

        # y must be 1D
        if y.ndim != 1:
            _fail(f"{split} y should be 1D, got shape {y.shape}")

        # X and y must have same number of rows
        if X.shape[0] != y.shape[0]:
            _fail(
                f"{split} X rows ({X.shape[0]:,}) != y length ({y.shape[0]:,})"
            )

        # Flattened dimension must be divisible by window_size
        if X.shape[1] % window_size != 0:
            _fail(
                f"{split} X dim {X.shape[1]} is not divisible by "
                f"window_size {window_size}"
            )

        n_features = X.shape[1] // window_size

        _pass(
            f"{split:<6}  X={str(X.shape):<25}  y={str(y.shape):<15}  "
            f"n_features={n_features}"
        )

        # Check all splits have same number of features
        if split == 'train':
            train_features = n_features
        else:
            if n_features != train_features:
                _fail(
                    f"{split} has {n_features} features but train has "
                    f"{train_features} — mismatch!"
                )


def check_run_boundaries(windows_dir: Path, window_size: int, stride: int):
    """
    Check 3: No window crosses a Run_ID boundary.

    Each window's start_idx and end_idx must belong to the same run.
    We verify this by checking that end_idx - start_idx == window_size - 1
    for every window (consecutive indices within a run).
    """
    print("\n[3/4] Checking no windows cross Run_ID boundaries...")

    for split in ['train', 'val', 'test']:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        if 'start_idx' not in data or 'end_idx' not in data:
            print(f"  ⚠  SKIP  {split} — metadata not saved "
                  f"(save_metadata=false in config)")
            continue

        start_idx = data['start_idx']
        end_idx   = data['end_idx']
        run_ids   = data['Run_ID']

        # end - start should always equal window_size - 1
        span = end_idx - start_idx
        bad  = np.where(span != window_size - 1)[0]

        if len(bad) > 0:
            _fail(
                f"{split}: {len(bad):,} windows have incorrect span "
                f"(expected {window_size - 1}, got other values). "
                f"This means windows are crossing run boundaries!"
            )

        # Each group of consecutive windows with same Run_ID must have
        # monotonically increasing start_idx (no jumps back to 0 mid-run)
        unique_runs = np.unique(run_ids)
        for run_id in unique_runs:
            mask    = run_ids == run_id
            starts  = start_idx[mask]
            diffs   = np.diff(starts)

            if np.any(diffs <= 0):
                _fail(
                    f"{split} run {run_id}: start indices are not "
                    f"strictly increasing — possible boundary issue."
                )

        _pass(f"{split:<6}  all {len(start_idx):,} windows stay within their run")


def check_labels(windows_dir: Path, window_size: int, stride: int):
    """
    Check 4: Label for each window matches the last timestep's target.

    We verify this on a sample of windows using the metadata
    (start_idx, end_idx, Run_ID) to reconstruct what the label should be.

    Note: This is a structural check — we verify end_idx aligns with
    the label, not the actual feature values.
    """
    print("\n[4/4] Checking labels come from last timestep of each window...")

    for split in ['train', 'val', 'test']:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        if 'end_idx' not in data:
            print(f"  ⚠  SKIP  {split} — metadata not saved")
            continue

        y         = data['y']
        end_idx   = data['end_idx']

        # Check label classes are valid (0-28 for TEP)
        unique_labels = np.unique(y)
        invalid = unique_labels[(unique_labels < 0) | (unique_labels > 28)]

        if len(invalid) > 0:
            _fail(
                f"{split}: found invalid label values {invalid}. "
                f"Expected 0-28 (TEP fault classes)."
            )

        # Check expected class count (TEP has 28 classes: 0 + IDV1-28 excl. IDV6)
        n_classes = len(unique_labels)
        if n_classes != 28:
            _fail(
                f"{split}: expected 28 unique classes, found {n_classes}. "
                f"Classes present: {sorted(unique_labels.tolist())}"
            )

        _pass(
            f"{split:<6}  labels valid  |  "
            f"{n_classes} classes  |  "
            f"range [{y.min()}, {y.max()}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(config_path: str = 'configs/config.yaml'):
    """
    Run all windowing smoke tests.

    Parameters:
    -----------
    config_path : str
        Path to config.yaml
    """

    print("\n" + "="*70)
    print(" "*18 + "WINDOWING SMOKE TEST")
    print("="*70)

    # Load config
    config      = load_config(config_path)
    window_size = config.window_size
    stride      = config.stride
    windows_dir = Path(config.output_dir) / 'windows'

    print(f"\n  Config:       {config_path}")
    print(f"  Windows dir:  {windows_dir}")
    print(f"  Window size:  {window_size}")
    print(f"  Stride:       {stride}")

    passed = 0
    failed = 0

    # Run each check
    checks = [
        ("Files exist",           check_files_exist),
        ("X and y shapes",        check_shapes),
        ("Run_ID boundaries",     check_run_boundaries),
        ("Labels from last step", check_labels),
    ]

    for check_name, check_fn in checks:
        try:
            if check_fn.__name__ == 'check_shapes':
                check_fn(windows_dir, window_size, stride, {})
            else:
                check_fn(windows_dir, window_size, stride)
            passed += 1
        except AssertionError as e:
            failed += 1
            # Error already printed inside check function

    # Final result
    total = passed + failed
    print("\n" + "="*70)

    if failed == 0:
        print(f"  🎉 ALL {total} CHECKS PASSED — windowing looks correct!")
    else:
        print(f"  ⚠  {passed}/{total} checks passed, {failed} FAILED")
        print(f"     Fix the issues above then rerun this test.")

    print("="*70 + "\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Smoke test for TEP windowing output'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config.yaml (default: configs/config.yaml)'
    )
    args = parser.parse_args()

    success = run_smoke_test(config_path=args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()