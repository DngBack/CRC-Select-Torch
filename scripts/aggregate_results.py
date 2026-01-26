"""
Results aggregation script for CRC-Select experiments.

Aggregates results across multiple seeds and generates summary statistics.
"""
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import json


def aggregate_coverage_at_risk(method_dir, seeds):
    """
    Aggregate coverage@risk results across seeds.
    
    Args:
        method_dir: Directory containing results for a method
        seeds: List of seed numbers
    
    Returns:
        DataFrame with mean ± std for each alpha
    """
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(method_dir, f'seed_{seed}')
        cov_path = os.path.join(seed_dir, 'coverage_at_risk.csv')
        
        if os.path.exists(cov_path):
            df = pd.read_csv(cov_path)
            df['seed'] = seed
            all_results.append(df)
    
    if not all_results:
        return None
    
    # Concatenate all seeds
    combined = pd.concat(all_results, ignore_index=True)
    
    # Compute statistics
    stats = combined.groupby('alpha').agg({
        'coverage': ['mean', 'std', 'count'],
        'risk': ['mean', 'std'],
        'tau': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['alpha', 'coverage_mean', 'coverage_std', 'n_seeds',
                    'risk_mean', 'risk_std', 'tau_mean', 'tau_std']
    
    return stats


def aggregate_risk_coverage_curves(method_dir, seeds):
    """
    Aggregate risk-coverage curves across seeds.
    
    Args:
        method_dir: Directory containing results for a method
        seeds: List of seed numbers
    
    Returns:
        DataFrame with mean ± std for each tau
    """
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(method_dir, f'seed_{seed}')
        rc_path = os.path.join(seed_dir, 'risk_coverage_curve.csv')
        
        if os.path.exists(rc_path):
            df = pd.read_csv(rc_path)
            df['seed'] = seed
            all_results.append(df)
    
    if not all_results:
        return None
    
    # Concatenate all seeds
    combined = pd.concat(all_results, ignore_index=True)
    
    # Compute statistics
    stats = combined.groupby('tau').agg({
        'risk': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'selective_acc': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tau', 'risk_mean', 'risk_std', 
                    'coverage_mean', 'coverage_std',
                    'selective_acc_mean', 'selective_acc_std', 'n_seeds']
    
    return stats


def aggregate_ood_results(method_dir, seeds):
    """
    Aggregate OOD evaluation results across seeds.
    
    Args:
        method_dir: Directory containing results for a method
        seeds: List of seed numbers
    
    Returns:
        DataFrame with mean ± std DAR for each tau
    """
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(method_dir, f'seed_{seed}')
        ood_path = os.path.join(seed_dir, 'ood_evaluation.csv')
        
        if os.path.exists(ood_path):
            df = pd.read_csv(ood_path)
            df['seed'] = seed
            all_results.append(df)
    
    if not all_results:
        return None
    
    # Concatenate all seeds
    combined = pd.concat(all_results, ignore_index=True)
    
    # Compute statistics
    stats = combined.groupby('tau').agg({
        'dar': ['mean', 'std'],
        'id_accept_rate': ['mean', 'std'],
        'ood_accept_rate': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['tau', 'dar_mean', 'dar_std',
                    'id_accept_mean', 'id_accept_std',
                    'ood_accept_mean', 'ood_accept_std', 'n_seeds']
    
    return stats


def create_summary_table(aggregated_results):
    """
    Create summary table comparing methods.
    
    Args:
        aggregated_results: Dict mapping method names to aggregated stats
    
    Returns:
        DataFrame with comparison table
    """
    summary_rows = []
    
    for method_name, results in aggregated_results.items():
        row = {'method': method_name}
        
        # Coverage@Risk(0.1)
        if 'coverage_at_risk' in results:
            cov_df = results['coverage_at_risk']
            cov_row = cov_df[cov_df['alpha'] == 0.1]
            if len(cov_row) > 0:
                row['coverage@0.1'] = f"{cov_row['coverage_mean'].values[0]:.3f} ± {cov_row['coverage_std'].values[0]:.3f}"
                row['risk@0.1'] = f"{cov_row['risk_mean'].values[0]:.3f} ± {cov_row['risk_std'].values[0]:.3f}"
        
        # DAR at tau=0.5
        if 'ood_results' in results:
            ood_df = results['ood_results']
            # Find row closest to tau=0.5
            ood_row = ood_df.iloc[(ood_df['tau'] - 0.5).abs().argsort()[:1]]
            if len(ood_row) > 0:
                row['DAR@0.5'] = f"{ood_row['dar_mean'].values[0]:.3f} ± {ood_row['dar_std'].values[0]:.3f}"
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def main(args):
    print("=" * 80)
    print("Aggregating CRC-Select Results Across Seeds")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    
    # Aggregate results for each method
    all_aggregated = {}
    
    for method_dir in args.method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\n[{method_name}] Aggregating results...")
        
        results = {}
        
        # Aggregate coverage@risk
        cov_at_risk = aggregate_coverage_at_risk(method_dir, args.seeds)
        if cov_at_risk is not None:
            results['coverage_at_risk'] = cov_at_risk
            print(f"  ✓ Aggregated coverage@risk across {cov_at_risk['n_seeds'].max()} seeds")
        
        # Aggregate risk-coverage curves
        rc_curves = aggregate_risk_coverage_curves(method_dir, args.seeds)
        if rc_curves is not None:
            results['risk_coverage_curve'] = rc_curves
            print(f"  ✓ Aggregated RC curves across {rc_curves['n_seeds'].max()} seeds")
        
        # Aggregate OOD results
        ood_results = aggregate_ood_results(method_dir, args.seeds)
        if ood_results is not None:
            results['ood_results'] = ood_results
            print(f"  ✓ Aggregated OOD results across {ood_results['n_seeds'].max()} seeds")
        
        all_aggregated[method_name] = results
    
    # ==================== Save Aggregated Results ====================
    print("\n" + "=" * 80)
    print("Saving Aggregated Results")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for method_name, results in all_aggregated.items():
        method_output_dir = os.path.join(args.output_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)
        
        for result_name, df in results.items():
            output_path = os.path.join(method_output_dir, f'{result_name}_aggregated.csv')
            df.to_csv(output_path, index=False)
            print(f"  ✓ Saved {method_name}/{result_name} to {output_path}")
    
    # ==================== Create Summary Table ====================
    print("\n" + "=" * 80)
    print("Summary Comparison Table")
    print("=" * 80)
    
    summary_table = create_summary_table(all_aggregated)
    print(summary_table.to_string(index=False))
    
    summary_path = os.path.join(args.output_dir, 'summary_table.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"\n✓ Summary table saved to {summary_path}")
    
    # ==================== Save Metadata ====================
    metadata = {
        'seeds': args.seeds,
        'methods': list(all_aggregated.keys()),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(args.output_dir, 'aggregation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ All aggregated results saved to {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--method_dirs', type=str, nargs='+', required=True,
                       help='directories containing results for each method')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='list of random seeds used in experiments')
    parser.add_argument('-o', '--output_dir', type=str, 
                       default='../results/aggregated',
                       help='directory to save aggregated results')
    
    args = parser.parse_args()
    main(args)

