"""
Comprehensive evaluation script for paper results.

This script computes ALL metrics needed for a paper on CRC-Select:
1. Risk-Coverage (RC) curves
2. AURC (Area Under Risk-Coverage curve)
3. Coverage@Risk for multiple α values
4. OOD evaluation (DAR on SVHN)
5. Calibration quality metrics
6. Statistical analysis across seeds
7. Comparison with baselines

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
from tqdm import tqdm

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.data import DatasetBuilder
from selectivenet.data_splits import get_split_loaders
from selectivenet.reproducibility import set_seed
from crc.risk_utils import compute_risk_scores, compute_coverage


class PaperEvaluator:
    """Comprehensive evaluator for paper results."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def collect_predictions(self, loader):
        """Collect all predictions from a dataloader."""
        all_logits = []
        all_g = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Collecting predictions", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                logits, g, _ = self.model(x)
                all_logits.append(logits.cpu())
                all_g.append(g.cpu())
                all_targets.append(y.cpu())
        
        return (torch.cat(all_logits), 
                torch.cat(all_g).squeeze(), 
                torch.cat(all_targets))
    
    def compute_rc_curve(self, logits, g, targets, n_points=201):
        """
        Compute Risk-Coverage curve with many points.
        
        Args:
            n_points: Number of threshold points (default: 201 for 0.005 resolution)
        
        Returns:
            DataFrame with RC curve data
        """
        thresholds = np.linspace(0.0, 1.0, n_points)
        
        rc_data = []
        for tau in tqdm(thresholds, desc="Computing RC curve", leave=False):
            # Acceptance mask
            g_hard = (g >= tau).float()
            coverage = g_hard.mean().item()
            
            if coverage > 0:
                # Risk scores
                r = compute_risk_scores(logits, targets)
                selective_risk = (g_hard * r).sum() / (g_hard.sum() + 1e-8)
                
                # Accuracy
                preds = logits.argmax(dim=1)
                correct = (preds == targets).float()
                selective_acc = (g_hard * correct).sum() / (g_hard.sum() + 1e-8)
                
                # Error = 1 - accuracy
                selective_error = 1.0 - selective_acc.item()
                selective_risk = selective_risk.item()
                selective_acc = selective_acc.item()
            else:
                selective_risk = 0.0
                selective_error = 0.0
                selective_acc = 0.0
            
            rc_data.append({
                'threshold': tau,
                'coverage': coverage,
                'risk': selective_risk,
                'error': selective_error,
                'accuracy': selective_acc
            })
        
        return pd.DataFrame(rc_data)
    
    def compute_aurc(self, rc_df):
        """Compute Area Under Risk-Coverage curve (lower is better)."""
        coverages = rc_df['coverage'].values
        errors = rc_df['error'].values
        
        # Sort by coverage
        sort_idx = np.argsort(coverages)
        coverages_sorted = coverages[sort_idx]
        errors_sorted = errors[sort_idx]
        
        # Compute area using trapezoidal rule
        aurc = np.trapz(errors_sorted, coverages_sorted)
        return aurc
    
    def compute_coverage_at_risk(self, rc_df, alpha_values):
        """
        Compute maximum coverage at different risk levels.
        
        Returns:
            DataFrame with coverage@risk for each α
        """
        results = []
        
        for alpha in alpha_values:
            # Find all rows where risk <= alpha
            valid_rows = rc_df[rc_df['risk'] <= alpha]
            
            if len(valid_rows) > 0:
                # Get maximum coverage
                max_cov_row = valid_rows.loc[valid_rows['coverage'].idxmax()]
                results.append({
                    'alpha': alpha,
                    'coverage': max_cov_row['coverage'],
                    'risk': max_cov_row['risk'],
                    'threshold': max_cov_row['threshold'],
                    'accuracy': max_cov_row['accuracy'],
                    'feasible': True
                })
            else:
                results.append({
                    'alpha': alpha,
                    'coverage': 0.0,
                    'risk': np.nan,
                    'threshold': np.nan,
                    'accuracy': np.nan,
                    'feasible': False
                })
        
        return pd.DataFrame(results)
    
    def evaluate_ood(self, id_loader, ood_loader, thresholds):
        """
        Evaluate OOD detection performance.
        
        Returns:
            DataFrame with DAR at different thresholds
        """
        print("\n[OOD Evaluation]")
        
        # Collect ID predictions
        print("  Collecting ID predictions...")
        id_logits, id_g, id_targets = self.collect_predictions(id_loader)
        
        # Collect OOD predictions
        print("  Collecting OOD predictions...")
        ood_logits, ood_g, ood_targets = self.collect_predictions(ood_loader)
        
        results = []
        for tau in tqdm(thresholds, desc="Computing DAR", leave=False):
            # ID acceptance rate
            id_accepted = (id_g >= tau).float().mean().item()
            
            # OOD acceptance rate (Dangerous Acceptance Rate)
            ood_accepted = (ood_g >= tau).float().mean().item()
            
            results.append({
                'threshold': tau,
                'id_accept_rate': id_accepted,
                'ood_accept_rate': ood_accepted,
                'dar': ood_accepted
            })
        
        return pd.DataFrame(results)
    
    def compute_calibration_metrics(self, rc_df, target_coverages=[0.7, 0.8, 0.9]):
        """
        Compute calibration quality metrics.
        
        Returns:
            DataFrame with calibration metrics
        """
        results = []
        
        for target_cov in target_coverages:
            # Find closest coverage
            idx = (rc_df['coverage'] - target_cov).abs().idxmin()
            row = rc_df.iloc[idx]
            
            coverage_error = abs(row['coverage'] - target_cov)
            
            results.append({
                'target_coverage': target_cov,
                'actual_coverage': row['coverage'],
                'coverage_error': coverage_error,
                'risk': row['risk'],
                'accuracy': row['accuracy'],
                'threshold': row['threshold']
            })
        
        return pd.DataFrame(results)


def load_model(checkpoint_path, args):
    """Load model from checkpoint."""
    dataset_builder = DatasetBuilder(args.dataset, args.dataroot)
    features = vgg16_variant(32, args.dropout_prob).cuda()
    model = SelectiveNet(features, args.dim_features, 10, div_by_ten=False).cuda()
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
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
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.method_name, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n[1/7] Loading model...")
    model, dataset_builder = load_model(args.checkpoint, args)
    evaluator = PaperEvaluator(model, device='cuda')
    print("  ✓ Model loaded")
    
    # Load data
    print("\n[2/7] Loading data...")
    full_train_dataset = dataset_builder(train=True, normalize=True, augmentation='original')
    _, _, test_loader = get_split_loaders(
        full_train_dataset, args.dataset, args.seed, 
        args.batch_size, args.num_workers
    )
    print(f"  ✓ Test samples: {len(test_loader.dataset)}")
    
    # Collect predictions
    print("\n[3/7] Collecting predictions on test set...")
    test_logits, test_g, test_targets = evaluator.collect_predictions(test_loader)
    print(f"  ✓ Collected {len(test_targets)} predictions")
    
    # Compute RC curve
    print(f"\n[4/7] Computing Risk-Coverage curve ({args.n_points} points)...")
    rc_df = evaluator.compute_rc_curve(test_logits, test_g, test_targets, args.n_points)
    
    # Compute AURC
    aurc = evaluator.compute_aurc(rc_df)
    print(f"  ✓ AURC: {aurc:.6f}")
    
    # Save RC curve
    rc_path = os.path.join(output_dir, 'risk_coverage_curve.csv')
    rc_df.to_csv(rc_path, index=False)
    print(f"  ✓ Saved RC curve to {rc_path}")
    
    # Coverage@Risk
    print("\n[5/7] Computing Coverage@Risk...")
    alpha_values = args.alpha_values if args.alpha_values else [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    cov_at_risk_df = evaluator.compute_coverage_at_risk(rc_df, alpha_values)
    
    cov_at_risk_path = os.path.join(output_dir, 'coverage_at_risk.csv')
    cov_at_risk_df.to_csv(cov_at_risk_path, index=False)
    print(f"  ✓ Saved to {cov_at_risk_path}")
    
    # Print Coverage@Risk summary
    print("\n  Coverage@Risk summary:")
    for _, row in cov_at_risk_df.iterrows():
        if row['feasible']:
            print(f"    α={row['alpha']:.3f}: Coverage={row['coverage']:.4f}, Risk={row['risk']:.4f}")
        else:
            print(f"    α={row['alpha']:.3f}: NOT FEASIBLE")
    
    # OOD evaluation
    if not args.skip_ood:
        print("\n[6/7] OOD Evaluation...")
        ood_loader = dataset_builder.get_ood_loader(
            args.ood_dataset, args.batch_size, 
            normalize_to_id=True, num_workers=args.num_workers
        )
        print(f"  ✓ OOD samples: {len(ood_loader.dataset)}")
        
        ood_thresholds = np.linspace(0.0, 1.0, 51)
        ood_df = evaluator.evaluate_ood(test_loader, ood_loader, ood_thresholds)
        
        ood_path = os.path.join(output_dir, 'ood_evaluation.csv')
        ood_df.to_csv(ood_path, index=False)
        print(f"  ✓ Saved OOD results to {ood_path}")
        
        # Print DAR at key thresholds
        print("\n  DAR (Dangerous Acceptance Rate) at key thresholds:")
        for tau in [0.3, 0.5, 0.7, 0.9]:
            idx = (ood_df['threshold'] - tau).abs().idxmin()
            dar = ood_df.iloc[idx]['dar']
            id_acc = ood_df.iloc[idx]['id_accept_rate']
            print(f"    τ={tau:.1f}: DAR={dar:.4f}, ID accept={id_acc:.4f}")
    
    # Calibration metrics
    print("\n[7/7] Computing calibration metrics...")
    calib_df = evaluator.compute_calibration_metrics(rc_df)
    calib_path = os.path.join(output_dir, 'calibration_metrics.csv')
    calib_df.to_csv(calib_path, index=False)
    print(f"  ✓ Saved to {calib_path}")
    
    # Generate summary report
    print("\n" + "=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100)
    
    summary = {
        'method': args.method_name,
        'seed': args.seed,
        'aurc': aurc,
        'num_test_samples': len(test_targets),
    }
    
    # Add Coverage@Risk results
    for _, row in cov_at_risk_df.iterrows():
        if row['feasible']:
            summary[f'coverage_at_risk_{row["alpha"]:.3f}'] = row['coverage']
            summary[f'risk_at_alpha_{row["alpha"]:.3f}'] = row['risk']
    
    # Add key performance points
    for cov_target in [0.70, 0.80, 0.90]:
        idx = (rc_df['coverage'] - cov_target).abs().idxmin()
        row = rc_df.iloc[idx]
        summary[f'error_at_cov_{cov_target:.2f}'] = row['error']
        summary[f'risk_at_cov_{cov_target:.2f}'] = row['risk']
        summary[f'accuracy_at_cov_{cov_target:.2f}'] = row['accuracy']
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n📊 Key Metrics:")
    print(f"  • AURC: {aurc:.6f}")
    print(f"  • Error @ 80% cov: {summary.get('error_at_cov_0.80', 0):.4f}")
    print(f"  • Risk @ 80% cov: {summary.get('risk_at_cov_0.80', 0):.4f}")
    print(f"  • Coverage @ risk 0.1: {summary.get('coverage_at_risk_0.100', 0):.4f}")
    
    print(f"\n✓ All results saved to: {output_dir}")
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

