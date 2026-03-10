"""
Comprehensive evaluation script for paper results.

Computes ALL metrics required by the paper:
  Audited (CRC-certified):
    - Accepted-loss mass A_hat(tau_hat) at CRC-calibrated threshold
    - Coverage C(tau_hat)
    - Violation gap max(A_hat - alpha, 0)
  Descriptive:
    - Conditional selective risk R_sel
    - Selective accuracy
    - AUROC / AUPR for error detection (selector as binary detector)
    - DAR (Dangerous Acceptance Rate) for OOD
  Efficiency:
    - RC-AUC (accepted-loss mass vs coverage)
    - Threshold conservativeness alpha - A_hat
    - Coverage efficiency C / C_oracle

Usage:
    python evaluate_for_paper.py \
        --checkpoint path/to/checkpoint.pth \
        --method_name "CRC-Select" \
        --dataset cifar10 \
        --seed 42 \
        --output_dir ../results_paper
"""

import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import numpy as np
import pandas as pd

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.data import DatasetBuilder
from selectivenet.data_splits import get_split_loaders
from selectivenet.reproducibility import set_seed
from selectivenet.evaluator_crc import CRCEvaluator
from selectivenet.resnet_variant import get_backbone


def load_model(checkpoint_path, args):
    """Load model from checkpoint."""
    dataset_builder = DatasetBuilder(args.dataset, args.dataroot)
    backbone = getattr(args, 'backbone', 'vgg16')
    if backbone == 'vgg16':
        features = vgg16_variant(32, args.dropout_prob).cuda()
        dim_features = args.dim_features
    else:
        features, dim_features = get_backbone(backbone, 32, args.dropout_prob)
        features = features.cuda()
    model = SelectiveNet(
        features, dim_features, dataset_builder.num_classes, div_by_ten=False
    ).cuda()

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if isinstance(checkpoint, list):
        checkpoint = checkpoint[0]

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, dataset_builder


def main(args):
    set_seed(args.seed)

    print("=" * 100)
    print(f"COMPREHENSIVE EVALUATION FOR PAPER: {args.method_name}")
    print("=" * 100)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 100)

    output_dir = os.path.join(args.output_dir, args.method_name, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model ----
    print("\n[1/8] Loading model...")
    model, dataset_builder = load_model(args.checkpoint, args)
    evaluator = CRCEvaluator(model, device='cuda')
    print("  Model loaded")

    # ---- Load data (3-way split) ----
    print("\n[2/8] Loading data...")
    full_train_dataset = dataset_builder(
        train=True, normalize=True, augmentation='original'
    )
    train_loader, cal_loader, test_loader = get_split_loaders(
        full_train_dataset, args.dataset, args.seed,
        args.batch_size, args.num_workers
    )
    print(f"  Cal samples: {len(cal_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # ---- Risk-Coverage curve ----
    print(f"\n[3/8] Computing Risk-Coverage curve ({args.n_points} pts)...")
    taus = np.linspace(0.0, 1.0, args.n_points)
    rc_df = evaluator.sweep_tau_risk_coverage(test_loader, taus)
    rc_path = os.path.join(output_dir, 'risk_coverage_curve.csv')
    rc_df.to_csv(rc_path, index=False)
    print(f"  Saved RC curve to {rc_path}")

    # ---- CRC calibration + all metrics at each alpha ----
    alpha_values = args.alpha_values or [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    print(f"\n[4/8] CRC calibration + full metrics (alpha={alpha_values})...")

    # Load OOD for combined evaluation
    ood_loader = None
    if not args.skip_ood:
        ood_loader = dataset_builder.get_ood_loader(
            args.ood_dataset, args.batch_size,
            normalize_to_id=True, num_workers=args.num_workers
        )
        print(f"  OOD samples ({args.ood_dataset}): {len(ood_loader.dataset)}")

    all_metrics_rows = []
    for alpha in alpha_values:
        metrics = evaluator.compute_all_metrics(
            cal_loader, test_loader, alpha, ood_loader=ood_loader
        )
        metrics['method'] = args.method_name
        metrics['dataset'] = args.dataset
        metrics['seed'] = args.seed
        all_metrics_rows.append(metrics)
        print(f"  alpha={alpha:.3f}: A_hat={metrics['test_accepted_loss_mass']:.4f}, "
              f"cov={metrics['test_coverage']:.4f}, "
              f"viol_gap={metrics['violation_gap']:.4f}, "
              f"AUROC={metrics['auroc']:.4f}")

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_path = os.path.join(output_dir, 'all_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved all metrics to {metrics_path}")

    # ---- Coverage@Risk (from accepted-loss mass) ----
    print("\n[5/8] Computing Coverage@Risk (accepted-loss mass)...")
    cov_at_risk = []
    for alpha in alpha_values:
        res = evaluator.compute_coverage_at_risk(test_loader, alpha, taus)
        cov_at_risk.append(res)
    cov_at_risk_df = pd.DataFrame(cov_at_risk)
    cov_at_risk_path = os.path.join(output_dir, 'coverage_at_risk.csv')
    cov_at_risk_df.to_csv(cov_at_risk_path, index=False)
    print(f"  Saved to {cov_at_risk_path}")

    # ---- OOD evaluation ----
    if not args.skip_ood and ood_loader is not None:
        print("\n[6/8] OOD evaluation...")
        ood_df = evaluator.evaluate_ood(test_loader, ood_loader, taus)
        ood_path = os.path.join(output_dir, 'ood_evaluation.csv')
        ood_df.to_csv(ood_path, index=False)

        ood_fixed = evaluator.compute_ood_acceptance_at_fixed_id_coverage(
            test_loader, ood_loader,
            target_id_coverages=[0.6, 0.7, 0.8, 0.9]
        )
        ood_fixed_path = os.path.join(output_dir, 'ood_at_fixed_id_coverage.csv')
        ood_fixed.to_csv(ood_fixed_path, index=False)
        print(f"  Saved OOD results to {ood_path}, {ood_fixed_path}")
    else:
        print("\n[6/8] OOD evaluation skipped")

    # ---- Calibration metrics (descriptive) ----
    print("\n[7/8] Computing calibration-style metrics...")
    calib_rows = []
    for target_cov in [0.7, 0.8, 0.9]:
        idx = (rc_df['coverage'] - target_cov).abs().idxmin()
        row = rc_df.iloc[idx]
        calib_rows.append({
            'target_coverage': target_cov,
            'actual_coverage': row['coverage'],
            'coverage_error': abs(row['coverage'] - target_cov),
            'accepted_loss_mass': row['accepted_loss_mass'],
            'selective_risk': row['selective_risk'],
            'selective_acc': row['selective_acc'],
            'tau': row['tau'],
        })
    calib_df = pd.DataFrame(calib_rows)
    calib_path = os.path.join(output_dir, 'calibration_metrics.csv')
    calib_df.to_csv(calib_path, index=False)
    print(f"  Saved to {calib_path}")

    # ---- Summary ----
    print("\n[8/8] Building summary...")
    # Pick primary alpha for headline numbers
    primary = metrics_df[metrics_df['alpha'] == 0.1]
    if len(primary) == 0:
        primary = metrics_df.iloc[[0]]
    s = primary.iloc[0]

    summary = {
        'method': args.method_name,
        'dataset': args.dataset,
        'seed': args.seed,
        'primary_alpha': s['alpha'],
        'tau_hat': s['tau_hat'],
        'accepted_loss_mass': s['test_accepted_loss_mass'],
        'selective_risk': s['test_selective_risk'],
        'coverage': s['test_coverage'],
        'selective_acc': s['test_selective_acc'],
        'violation_gap': s['violation_gap'],
        'violated': s['violated'],
        'auroc': s['auroc'],
        'aupr': s['aupr'],
        'rc_auc_alm': s['rc_auc_alm'],
        'rc_auc_risk': s['rc_auc_risk'],
        'conservativeness': s['conservativeness'],
        'coverage_efficiency': s['coverage_efficiency'],
    }
    if 'dar' in s:
        summary['dar'] = s['dar']
        summary['safety_ratio'] = s['safety_ratio']

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 100)

    return summary


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Model and checkpoint
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method_name', type=str, default='CRC-Select',
                       help='Name of the method (for organizing results)')
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'resnet18', 'wrn28_10'],
                       help='backbone architecture')
    
    # Data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    
    # OOD evaluation
    parser.add_argument('--skip_ood', action='store_true',
                       help='Skip OOD evaluation')
    parser.add_argument('--ood_dataset', type=str, default='svhn',
                       help='OOD dataset name')
    
    # Evaluation settings
    parser.add_argument('--n_points', type=int, default=201,
                       help='Number of points in RC curve')
    parser.add_argument('--alpha_values', type=float, nargs='+', default=None,
                       help='Risk levels for Coverage@Risk')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='../results_paper',
                       help='Output directory for results')
    
    args = parser.parse_args()
    main(args)

