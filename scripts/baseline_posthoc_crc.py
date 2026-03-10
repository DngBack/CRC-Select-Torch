"""
Post-hoc CRC baseline for comparison.

This script implements the 2-stage approach:
1. Train vanilla SelectiveNet (using existing train.py)
2. Apply CRC calibration post-hoc on calibration set
3. Evaluate on test set

This baseline demonstrates that joint training (CRC-Select) achieves
better coverage than post-hoc calibration at the same risk level.
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
from selectivenet.evaluator_crc import CRCEvaluator
from selectivenet.reproducibility import set_seed

from crc.calibrate import crc_calibrate_threshold
from crc.risk_utils import (
    compute_selective_risk,
    compute_coverage,
    compute_accepted_loss_mass,
)


def load_vanilla_selectivenet(checkpoint_path, args):
    """Load pre-trained vanilla SelectiveNet model."""
    dataset_builder = DatasetBuilder(name=args.dataset, root_path=args.dataroot)
    features = vgg16_variant(dataset_builder.input_size, args.dropout_prob).cuda()
    model = SelectiveNet(
        features, args.dim_features, dataset_builder.num_classes,
        div_by_ten=args.div_by_ten
    ).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, list):
        # Vanilla checkpoint is saved as [final, best_val, best_val_tf]
        # Use the first one (final epoch)
        checkpoint = checkpoint[0]
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, dataset_builder


def main(args):
    set_seed(args.seed)
    
    print("=" * 80)
    print("Post-hoc CRC Baseline Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Seed: {args.seed}")
    print(f"Target risk (alpha): {args.alpha_risk}")
    print(f"Initial tau: {args.tau_init}")
    print("=" * 80)
    
    # ==================== Load Model ====================
    print("\n[1/5] Loading pre-trained SelectiveNet...")
    model, dataset_builder = load_vanilla_selectivenet(args.checkpoint, args)
    print("  ✓ Model loaded")
    
    # ==================== Load Data ====================
    print("\n[2/5] Loading data splits...")
    full_train_dataset = dataset_builder(
        train=True, normalize=True, augmentation='original'
    )
    
    train_loader, cal_loader, test_loader = get_split_loaders(
        full_train_dataset,
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Cal batches: {len(cal_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ==================== Calibrate Threshold ====================
    print(f"\n[3/5] Applying post-hoc CRC calibration...")
    print(f"  Target risk (alpha): {args.alpha_risk}")

    evaluator = CRCEvaluator(model, device='cuda')

    # Collect calibration predictions
    cal_logits, cal_g, cal_targets = evaluator.collect_predictions(cal_loader)

    # Proper CRC calibration:
    #   tau_hat = inf{tau : (n/(n+1)) * A_hat_n(tau) + 1/(n+1) <= alpha}
    calib_result = crc_calibrate_threshold(
        cal_logits, cal_g, cal_targets, args.alpha_risk
    )
    tau_hat = calib_result['tau_hat']

    print(f"\n  Calibration results:")
    print(f"    tau_hat: {tau_hat:.6f}")
    print(f"    Accepted-loss mass on cal: {calib_result['accepted_loss_mass']:.4f}")
    print(f"    Coverage on cal: {calib_result['coverage']:.4f}")

    # ==================== Evaluate on Test Set ====================
    print(f"\n[4/5] Evaluating on test set...")

    # Use compute_all_metrics for comprehensive evaluation
    alpha_values = [0.05, 0.1, 0.15, 0.2]

    # Load OOD if available
    ood_loader = None
    if not getattr(args, 'skip_ood', False):
        try:
            ood_loader = dataset_builder.get_ood_loader(
                args.ood_dataset, args.batch_size,
                normalize_to_id=True, num_workers=args.num_workers
            )
        except Exception:
            ood_loader = None

    all_metrics_rows = []
    for alpha in alpha_values:
        metrics = evaluator.compute_all_metrics(
            cal_loader, test_loader, alpha, ood_loader=ood_loader
        )
        metrics['method'] = 'posthoc_crc'
        metrics['dataset'] = args.dataset
        metrics['seed'] = args.seed
        all_metrics_rows.append(metrics)
        print(f"  alpha={alpha:.3f}: A_hat={metrics['test_accepted_loss_mass']:.4f}, "
              f"cov={metrics['test_coverage']:.4f}, "
              f"viol_gap={metrics['violation_gap']:.4f}")

    # ==================== Generate Full Evaluation ====================
    print(f"\n[5/5] Generating comprehensive evaluation...")

    # Risk-coverage curve
    taus = np.linspace(0.0, 1.0, 201)
    rc_curve = evaluator.sweep_tau_risk_coverage(test_loader, taus)

    # ==================== Save Results ====================
    results_dir = os.path.join(args.output_dir, 'posthoc_crc', f'seed_{args.seed}')
    os.makedirs(results_dir, exist_ok=True)

    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(os.path.join(results_dir, 'all_metrics.csv'), index=False)

    # Save risk-coverage curve
    rc_curve.to_csv(os.path.join(results_dir, 'risk_coverage_curve.csv'), index=False)

    # Save summary for primary alpha
    primary = metrics_df[metrics_df['alpha'] == args.alpha_risk]
    if len(primary) == 0:
        primary = metrics_df.iloc[[0]]
    summary = primary.iloc[0].to_dict()
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)

    print(f"\n  Results saved to {results_dir}")

    # ==================== Summary ====================
    s = primary.iloc[0]
    print("\n" + "=" * 80)
    print("Post-hoc CRC Baseline Summary")
    print("=" * 80)
    print(f"Method: Post-hoc CRC (2-stage)")
    print(f"tau_hat: {s['tau_hat']:.6f}")
    print(f"Accepted-loss mass: {s['test_accepted_loss_mass']:.4f} (target: {s['alpha']:.3f})")
    print(f"Coverage: {s['test_coverage']:.4f}")
    print(f"Selective risk: {s['test_selective_risk']:.4f}")
    print(f"Selective accuracy: {s['test_selective_acc']:.4f}")
    print(f"Violation gap: {s['violation_gap']:.4f}")
    print(f"AUROC: {s['auroc']:.4f}, AUPR: {s['aupr']:.4f}")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Model and checkpoint
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='path to pre-trained SelectiveNet checkpoint')
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--div_by_ten', action='store_true')
    
    # Data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--ood_dataset', type=str, default='svhn',
                       help='OOD dataset for evaluation (for compatibility with eval script)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    
    # CRC calibration settings
    parser.add_argument('--alpha_risk', type=float, default=0.1,
                       help='target risk level for CRC')
    parser.add_argument('--skip_ood', action='store_true',
                       help='skip OOD evaluation')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                       help='directory to save results')
    
    args = parser.parse_args()
    main(args)

