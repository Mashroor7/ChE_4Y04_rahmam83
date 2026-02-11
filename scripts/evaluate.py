#!/usr/bin/env python3
"""
Offline evaluation of tuned KAN variants on TEP fault detection.

Loads saved predictions.npz files (no model rerun) and computes metrics.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model efficient_kan
    python scripts/evaluate.py --config configs/config.yaml
"""

import sys
import json
import argparse
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from configs.config_loader import load_config

ALL_VARIANTS = ['efficient_kan', 'fourier_kan', 'wavelet_kan', 'fast_kan']


def evaluate_variant(results_dir: Path, variant: str) -> dict | None:
    """
    Load predictions.npz for a variant and compute metrics.

    Returns None if the predictions file doesn't exist yet.
    """
    preds_path = results_dir / variant / 'predictions.npz'
    if not preds_path.exists():
        print(f"  WARNING: {preds_path} not found — skipping {variant}")
        return None

    data = np.load(preds_path)
    y_pred = data['y_pred']
    y_true = data['y_true']

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class F1
    classes = sorted(set(y_true) | set(y_pred))
    per_class_f1_arr = f1_score(y_true, y_pred, labels=classes,
                                average=None, zero_division=0)
    per_class_f1 = {int(c): float(f) for c, f in zip(classes, per_class_f1_arr)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm.tolist(),
    }

    # Save eval_metrics.json
    out_path = results_dir / variant / 'eval_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size:,} bytes)")

    return metrics


def print_comparison_table(results: dict):
    """Print a formatted comparison table across all evaluated variants."""
    if not results:
        print("\n  No variants have been evaluated yet.")
        return

    print(f"\n{'='*70}")
    print("  Model Comparison")
    print(f"{'='*70}")
    print(f"\n  {'Model':<20s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")

    best_model = None
    best_acc = -1.0

    for variant, metrics in results.items():
        acc = metrics['accuracy']
        f1 = metrics['macro_f1']
        print(f"  {variant:<20s} {acc:>10.4f} {f1:>10.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = variant

    print(f"\n  Best model: {best_model} (accuracy = {best_acc:.4f})")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate tuned KAN variants')
    parser.add_argument('--model', type=str, default=None,
                        choices=ALL_VARIANTS,
                        help='Evaluate a single variant (default: all)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config.results_dir)

    variants = [args.model] if args.model else ALL_VARIANTS

    print("=" * 70)
    print("  KAN Evaluation — TEP Fault Detection")
    print("=" * 70)

    results = {}
    for variant in variants:
        print(f"\n  Evaluating: {variant}")
        print(f"  {'-'*40}")
        metrics = evaluate_variant(results_dir, variant)
        if metrics is not None:
            results[variant] = metrics

    print_comparison_table(results)


if __name__ == '__main__':
    main()