"""
Comprehensive evaluation script for CRC-Select experiments.

Evaluates trained models on:
- Risk-Coverage curves (ID)
- Coverage@Risk(alpha) for multiple alpha values
- DAR (Dangerous Acceptance Rate) on OOD
- Mixture evaluation (ID + OOD)
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

import wandb


def load_model_from_checkpoint(checkpoint_path, args):
    """Load model from checkpoint file."""
    # Create model
    dataset_builder = DatasetBuilder(name=args.dataset, root_path=args.dataroot)
    features = vgg16_variant(dataset_builder.input_size, args.dropout_prob).cuda()
    model = SelectiveNet(
        features, args.dim_features, dataset_builder.num_classes,
        div_by_ten=args.div_by_ten
    ).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, dataset_builder


def main(args):
    set_seed(args.seed)
    
    print("=" * 80)
    print("CRC-Select Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)
    
    # Load model
    model, dataset_builder = load_model_from_checkpoint(args.checkpoint, args)
    
    # Create evaluator
    evaluator = CRCEvaluator(model, device='cuda')
    
    # ==================== Load Data ====================
    full_train_dataset = dataset_builder(
        train=True, normalize=True, augmentation='original'
    )
    
    _, _, test_loader = get_split_loaders(
        full_train_dataset,
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load OOD dataset
    ood_loader = dataset_builder.get_ood_loader(
        ood_name=args.ood_dataset,
        batch_size=args.batch_size,
        normalize_to_id=True,
        num_workers=args.num_workers
    )
    
    print(f"\nData loaded:")
    print(f"  ID test batches: {len(test_loader)}")
    print(f"  OOD batches: {len(ood_loader)}")
    
    # ==================== Evaluation ====================
    results_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define threshold range
    taus = np.linspace(args.tau_min, args.tau_max, args.num_taus)
    
    # 1. Risk-Coverage Curve
    print("\n[1/4] Computing Risk-Coverage curve...")
    rc_curve = evaluator.sweep_tau_risk_coverage(test_loader, taus)
    rc_curve_path = os.path.join(results_dir, 'risk_coverage_curve.csv')
    rc_curve.to_csv(rc_curve_path, index=False)
    print(f"  ✓ Saved to {rc_curve_path}")
    print(rc_curve.head())
    
    # 2. Coverage@Risk for multiple alpha values
    print("\n[2/4] Computing Coverage@Risk...")
    coverage_at_risk_results = []
    alphas = args.alphas if args.alphas else [0.05, 0.1, 0.15, 0.2]
    
    for alpha in alphas:
        result = evaluator.compute_coverage_at_risk(test_loader, alpha, taus)
        coverage_at_risk_results.append(result)
        print(f"  Alpha={alpha:.2f}: Coverage={result['coverage']:.3f}, "
              f"Risk={result['risk']:.3f}, Tau={result['tau']:.3f}")
    
    cov_at_risk_df = pd.DataFrame(coverage_at_risk_results)
    cov_at_risk_path = os.path.join(results_dir, 'coverage_at_risk.csv')
    cov_at_risk_df.to_csv(cov_at_risk_path, index=False)
    print(f"  ✓ Saved to {cov_at_risk_path}")
    
    # 3. OOD Evaluation (DAR)
    print("\n[3/4] Computing OOD metrics (DAR)...")
    ood_results = evaluator.evaluate_ood(test_loader, ood_loader, taus)
    ood_results_path = os.path.join(results_dir, 'ood_evaluation.csv')
    ood_results.to_csv(ood_results_path, index=False)
    print(f"  ✓ Saved to {ood_results_path}")
    print(ood_results.head())
    
    # 4. Mixture Evaluation
    print("\n[4/4] Computing mixture evaluation...")
    mixture_results = []
    p_ood_list = args.p_ood_list if args.p_ood_list else [0.1, 0.3, 0.5]
    
    for p_ood in p_ood_list:
        for alpha in alphas:
            # Use default tau or find optimal tau
            tau = args.default_tau
            result = evaluator.evaluate_mixture(
                test_loader, ood_loader, p_ood, tau, alpha
            )
            mixture_results.append(result)
            print(f"  p_ood={p_ood:.1f}, alpha={alpha:.2f}: "
                  f"Coverage={result['coverage']:.3f}, "
                  f"OOD_accept={result['ood_accept_rate']:.3f}")
    
    mixture_df = pd.DataFrame(mixture_results)
    mixture_path = os.path.join(results_dir, 'mixture_evaluation.csv')
    mixture_df.to_csv(mixture_path, index=False)
    print(f"  ✓ Saved to {mixture_path}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Results saved to: {results_dir}")
    print("\nKey metrics:")
    
    # Best coverage at alpha=0.1
    best_cov = cov_at_risk_df[cov_at_risk_df['alpha'] == 0.1].iloc[0]
    print(f"  Coverage@Risk(0.1): {best_cov['coverage']:.3f}")
    
    # DAR at default tau
    dar_row = ood_results[ood_results['tau'].abs() - args.default_tau < 0.01].iloc[0]
    print(f"  DAR at tau={args.default_tau}: {dar_row['dar']:.3f}")
    
    print("=" * 80)
    
    return {
        'rc_curve': rc_curve,
        'coverage_at_risk': cov_at_risk_df,
        'ood_results': ood_results,
        'mixture_results': mixture_df
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Model and checkpoint
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='path to model checkpoint')
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--div_by_ten', action='store_true')
    
    # Data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--ood_dataset', type=str, default='svhn',
                       help='OOD dataset for evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    
    # Evaluation settings
    parser.add_argument('--tau_min', type=float, default=0.3,
                       help='minimum threshold for sweeping')
    parser.add_argument('--tau_max', type=float, default=0.8,
                       help='maximum threshold for sweeping')
    parser.add_argument('--num_taus', type=int, default=20,
                       help='number of thresholds to evaluate')
    parser.add_argument('--default_tau', type=float, default=0.5,
                       help='default threshold for mixture evaluation')
    parser.add_argument('--alphas', type=float, nargs='+', default=None,
                       help='list of alpha values for Coverage@Risk')
    parser.add_argument('--p_ood_list', type=float, nargs='+', default=None,
                       help='list of OOD proportions for mixture evaluation')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                       help='directory to save evaluation results')
    
    args = parser.parse_args()
    main(args)

