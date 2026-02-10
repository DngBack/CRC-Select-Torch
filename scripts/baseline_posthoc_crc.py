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

from crc.calibrate import compute_crc_threshold, calibrate_selector
from crc.risk_utils import compute_selective_risk, compute_coverage


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
    print(f"  Target risk: {args.alpha_risk}")
    print(f"  Initial tau: {args.tau_init}")
    
    # Apply CRC calibration on calibration set
    calib_result = compute_crc_threshold(
        model=model,
        cal_loader=cal_loader,
        tau=args.tau_init,
        alpha=args.alpha_risk,
        delta=args.delta,
        device='cuda'
    )
    
    q_crc = calib_result['q']
    tau_calibrated = args.tau_init  # Keep same tau, q is the risk threshold
    
    print(f"\n  Calibration results:")
    print(f"    CRC threshold q: {q_crc:.4f}")
    print(f"    Acceptance threshold tau: {tau_calibrated:.4f}")
    print(f"    Coverage on cal: {calib_result['actual_coverage']:.4f}")
    print(f"    Risk on cal: {calib_result['estimated_risk']:.4f}")
    print(f"    Accepted samples: {calib_result['num_accepted']}/{calib_result['num_total']}")
    
    # ==================== Evaluate on Test Set ====================
    print(f"\n[4/5] Evaluating on test set...")
    evaluator = CRCEvaluator(model, device='cuda')
    
    # Collect test predictions
    test_logits, test_g, test_targets = evaluator.collect_predictions(test_loader)
    
    # Compute metrics at calibrated threshold
    test_risk = compute_selective_risk(
        test_logits, test_g, test_targets,
        threshold=tau_calibrated, hard=True
    )
    test_coverage = compute_coverage(test_g, tau_calibrated)
    
    # Compute accuracy metrics
    test_preds = test_logits.argmax(dim=1)
    test_correct = (test_preds == test_targets).float()
    
    # Overall accuracy (all samples)
    test_acc = test_correct.mean()
    
    # Selective accuracy (only accepted samples)
    acceptance_mask = (test_g >= tau_calibrated).float()
    num_accepted = acceptance_mask.sum().item()
    
    if num_accepted > 0:
        test_selective_acc = (acceptance_mask * test_correct).sum() / num_accepted
    else:
        test_selective_acc = torch.tensor(0.0)
    
    # Error rate on accepted samples (should relate to risk)
    if num_accepted > 0:
        test_selective_err = 1.0 - test_selective_acc
    else:
        test_selective_err = torch.tensor(0.0)
    
    print(f"\n  Test results (tau={tau_calibrated:.4f}):")
    print(f"    Coverage: {test_coverage:.4f} ({int(num_accepted)}/{len(test_targets)} samples)")
    print(f"    Selective risk: {test_risk:.4f}")
    print(f"    Selective error: {test_selective_err:.4f}")
    print(f"    Selective accuracy: {test_selective_acc:.4f}")
    print(f"    Overall accuracy: {test_acc:.4f}")
    print(f"    Risk violation: {'YES ⚠️ ' if test_risk > args.alpha_risk else 'NO ✓'} "
          f"(target: {args.alpha_risk:.4f})")
    
    # ==================== Generate Full Evaluation ====================
    print(f"\n[5/5] Generating comprehensive evaluation...")
    
    # Risk-coverage curve
    taus = np.linspace(0.3, 0.8, 20)
    rc_curve = evaluator.sweep_tau_risk_coverage(test_loader, taus)
    
    # Coverage at risk for multiple alphas
    alphas = [0.05, 0.1, 0.15, 0.2]
    coverage_at_risk_results = []
    for alpha in alphas:
        result = evaluator.compute_coverage_at_risk(test_loader, alpha, taus)
        coverage_at_risk_results.append(result)
        print(f"  Coverage@Risk({alpha:.2f}): {result['coverage']:.3f}")
    
    # ==================== Save Results ====================
    results_dir = os.path.join(args.output_dir, 'posthoc_crc', f'seed_{args.seed}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save calibration results
    calib_summary = {
        'method': 'posthoc_crc',
        'seed': args.seed,
        'alpha_risk': args.alpha_risk,
        'tau_init': args.tau_init,
        'tau_calibrated': tau_calibrated,
        'q_crc': q_crc,
        'cal_coverage': calib_result['actual_coverage'],
        'cal_risk': calib_result['estimated_risk'],
        'test_coverage': test_coverage.item(),
        'test_risk': test_risk.item(),
        'test_selective_acc': test_selective_acc.item(),
        'test_selective_err': test_selective_err.item(),
        'test_overall_acc': test_acc.item(),
        'risk_violation': test_risk.item() > args.alpha_risk
    }
    
    calib_df = pd.DataFrame([calib_summary])
    calib_path = os.path.join(results_dir, 'calibration_summary.csv')
    calib_df.to_csv(calib_path, index=False)
    
    # Save risk-coverage curve
    rc_path = os.path.join(results_dir, 'risk_coverage_curve.csv')
    rc_curve.to_csv(rc_path, index=False)
    
    # Save coverage at risk
    cov_at_risk_df = pd.DataFrame(coverage_at_risk_results)
    cov_at_risk_path = os.path.join(results_dir, 'coverage_at_risk.csv')
    cov_at_risk_df.to_csv(cov_at_risk_path, index=False)
    
    print(f"\n  ✓ Results saved to {results_dir}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("Post-hoc CRC Baseline Summary")
    print("=" * 80)
    print(f"Method: Post-hoc CRC (2-stage)")
    print(f"Calibrated on: {calib_result['num_accepted']} cal samples")
    print(f"\nCalibration Set:")
    print(f"  Coverage: {calib_result['actual_coverage']:.3f}")
    print(f"  Risk: {calib_result['estimated_risk']:.3f}")
    print(f"\nTest Set Performance:")
    print(f"  Coverage: {test_coverage:.3f} ({int(num_accepted)}/{len(test_targets)})")
    print(f"  Selective Risk: {test_risk:.3f} (target: {args.alpha_risk:.3f})")
    print(f"  Selective Error: {test_selective_err:.3f}")
    print(f"  Selective Accuracy: {test_selective_acc:.3f}")
    print(f"  Overall Accuracy: {test_acc:.3f}")
    print(f"  Risk Violation: {'YES ⚠️' if test_risk > args.alpha_risk else 'NO ✓'}")
    print("=" * 80)
    
    return calib_summary


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
    parser.add_argument('--tau_init', type=float, default=0.5,
                       help='initial acceptance threshold')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='failure probability for CRC bound')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                       help='directory to save results')
    
    args = parser.parse_args()
    main(args)

