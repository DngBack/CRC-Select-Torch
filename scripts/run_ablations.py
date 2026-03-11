"""
Run ablation studies for CRC-Select.

Ablations:
  A1: lambda_risk sensitivity  (mu_init in {0.1, 0.5, 1.0, 2.0, 5.0})
  A2: Calibration set fraction (cal_frac in {0.05, 0.10, 0.15, 0.20, 0.30})
  A3: Warmup length             (warmup_epochs in {0, 10, 20, 40, 80})
  A4: Recalibration frequency   (recal_every in {1, 5, 10, 20, 50})
  A5: Loss variant              (hinge vs squared penalty)
  A6: Backbone comparison       (vgg16, resnet18, wrn28_10)
  A7: Alpha grid                (alpha in {0.01, 0.05, 0.10, 0.15, 0.20, 0.30})

Usage:
    python run_ablations.py --ablation A1 --dataset cifar10 --seed 42
    python run_ablations.py --ablation all --dataset cifar10 --seeds 42 123 456 789 999
"""
import os
import sys
import subprocess
from argparse import ArgumentParser
from itertools import product

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)


# ====================== Ablation configurations ======================

ABLATION_CONFIGS = {
    'A1': {
        'name': 'lambda_risk_sensitivity',
        'param': 'mu_init',
        'values': [0.1, 0.5, 1.0, 2.0, 5.0],
        'description': 'Risk penalty multiplier sensitivity',
    },
    'A2': {
        'name': 'cal_fraction',
        'param': 'cal_frac',
        'values': [0.05, 0.10, 0.15, 0.20, 0.30],
        'description': 'Calibration set fraction (rest split 70/test)',
    },
    'A3': {
        'name': 'warmup_length',
        'param': 'warmup_epochs',
        'values': [0, 10, 20, 40, 80],
        'description': 'Number of warm-up epochs before CRC penalty',
    },
    'A4': {
        'name': 'recalibrate_frequency',
        'param': 'recalibrate_every',
        'values': [1, 5, 10, 20, 50],
        'description': 'Recalibrate CRC threshold every N epochs',
    },
    'A5': {
        'name': 'loss_variant',
        'param': 'loss_variant',
        'values': ['hinge', 'squared'],
        'description': 'Hinge max(0, A-alpha) vs squared (A-alpha)^2 penalty',
    },
    'A6': {
        'name': 'backbone',
        'param': 'backbone',
        'values': ['vgg16', 'resnet18', 'wrn28_10'],
        'description': 'Backbone architecture comparison',
    },
    'A7': {
        'name': 'alpha_grid',
        'param': 'alpha_risk',
        'values': [0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
        'description': 'Target risk level grid',
    },
}


def build_train_command(base_args, overrides):
    """Build the training command with specific ablation overrides."""
    cmd = [
        sys.executable, os.path.join(base, 'scripts', 'train_crc_select.py'),
        '--dataset', base_args.dataset,
        '--dataroot', base_args.dataroot,
        '--num_epochs', str(base_args.num_epochs),
        '--batch_size', str(base_args.batch_size),
    ]

    # Defaults that can be overridden
    defaults = {
        'alpha_risk': 0.1,
        'mu_init': 1.0,
        'warmup_epochs': 20,
        'recalibrate_every': 5,
        'backbone': 'vgg16',
        'loss_variant': 'hinge',
    }
    defaults.update(overrides)

    for key, val in defaults.items():
        cmd.extend([f'--{key}', str(val)])

    return cmd


def build_eval_command(checkpoint_dir, base_args, alpha):
    """Build the evaluation command."""
    cmd = [
        sys.executable, os.path.join(base, 'scripts', 'evaluate_for_paper.py'),
        '--checkpoint', checkpoint_dir,
        '--dataset', base_args.dataset,
        '--dataroot', base_args.dataroot,
        '--alphas', str(alpha),
    ]
    return cmd


def run_single_ablation(ablation_id, base_args, seeds):
    """Run a single ablation study across all specified seeds."""
    config = ABLATION_CONFIGS[ablation_id]
    param_name = config['param']
    values = config['values']

    print(f"\n{'='*80}")
    print(f"Ablation {ablation_id}: {config['description']}")
    print(f"Parameter: {param_name}  Values: {values}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}")

    ablation_dir = os.path.join(
        base_args.output_dir, f'ablation_{ablation_id}_{config["name"]}'
    )
    os.makedirs(ablation_dir, exist_ok=True)

    for val in values:
        for seed in seeds:
            run_tag = f'{param_name}_{val}_seed_{seed}'
            run_dir = os.path.join(ablation_dir, run_tag)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n--- {run_tag} ---")

            overrides = {param_name: val}
            cmd = build_train_command(base_args, overrides)
            cmd.extend(['--seed', str(seed), '--output_dir', run_dir])

            if base_args.dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
                continue

            print(f"  Training with {param_name}={val}, seed={seed} ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[-500:]}")
                continue

            # Evaluate
            alpha_for_eval = val if param_name == 'alpha_risk' else 0.1
            eval_cmd = build_eval_command(run_dir, base_args, alpha_for_eval)
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
            if eval_result.returncode != 0:
                print(f"  Eval FAILED: {eval_result.stderr[-500:]}")
            else:
                print(f"  Done.")

    # Collect summary across values
    collect_ablation_summary(ablation_id, ablation_dir, config, seeds)


def collect_ablation_summary(ablation_id, ablation_dir, config, seeds):
    """Collect results from an ablation into a summary CSV."""
    import pandas as pd

    param_name = config['param']
    rows = []

    for val in config['values']:
        for seed in seeds:
            run_tag = f'{param_name}_{val}_seed_{seed}'
            metrics_path = os.path.join(ablation_dir, run_tag, 'all_metrics.csv')
            if not os.path.exists(metrics_path):
                continue
            df = pd.read_csv(metrics_path)
            for _, r in df.iterrows():
                row = r.to_dict()
                row[param_name] = val
                row['seed'] = seed
                rows.append(row)

    if not rows:
        print(f"  No results to summarise for ablation {ablation_id}")
        return

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(ablation_dir, 'ablation_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path} ({len(rows)} rows)")

    # Print compact table: mean over seeds
    import numpy as np
    agg_cols = [c for c in summary.columns
                if c not in (param_name, 'seed', 'alpha')]
    numeric_cols = summary[agg_cols].select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        grouped = summary.groupby(param_name)[numeric_cols].agg(['mean', 'std'])
        print(grouped.to_string())


def main():
    parser = ArgumentParser(description='Run CRC-Select ablation studies')
    parser.add_argument('--ablation', type=str, default='all',
                        choices=list(ABLATION_CONFIGS.keys()) + ['all'],
                        help='Which ablation to run (A1-A7 or all)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 999])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str,
                        default='./results/ablations')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')

    args = parser.parse_args()

    if args.ablation == 'all':
        ablation_ids = list(ABLATION_CONFIGS.keys())
    else:
        ablation_ids = [args.ablation]

    for aid in ablation_ids:
        run_single_ablation(aid, args, args.seeds)

    print(f"\nAll ablations complete. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
