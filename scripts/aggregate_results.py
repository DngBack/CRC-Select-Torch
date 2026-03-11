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
    Prefers all_metrics.csv (new format) over coverage_at_risk.csv (legacy).
    """
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(method_dir, f'seed_{seed}')
        metrics_path = os.path.join(seed_dir, 'all_metrics.csv')
        legacy_path = os.path.join(seed_dir, 'coverage_at_risk.csv')
        
        df = None
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
        elif os.path.exists(legacy_path):
            df = pd.read_csv(legacy_path)
        
        if df is not None:
            df['seed'] = seed
            # Normalise column names for legacy compatibility
            if 'test_accepted_loss_mass' in df.columns:
                df.rename(columns={
                    'test_accepted_loss_mass': 'accepted_loss_mass',
                    'test_coverage': 'coverage'
                }, inplace=True)
            if 'accepted_loss_mass' not in df.columns and 'risk' in df.columns:
                df['accepted_loss_mass'] = df['risk']
            all_results.append(df)
    
    if not all_results:
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    
    agg_dict = {'coverage': ['mean', 'std', 'count']}
    if 'accepted_loss_mass' in combined.columns:
        agg_dict['accepted_loss_mass'] = ['mean', 'std']
    if 'selective_risk' in combined.columns:
        agg_dict['selective_risk'] = ['mean', 'std']
    if 'violation_gap' in combined.columns:
        agg_dict['violation_gap'] = ['mean', 'std']
    if 'tau' in combined.columns:
        agg_dict['tau'] = ['mean', 'std']
    
    stats = combined.groupby('alpha').agg(agg_dict).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns]
    
    return stats


def aggregate_risk_coverage_curves(method_dir, seeds):
    """Aggregate risk-coverage curves across seeds."""
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
    
    combined = pd.concat(all_results, ignore_index=True)
    
    agg_dict = {'coverage': ['mean', 'std']}
    if 'accepted_loss_mass' in combined.columns:
        agg_dict['accepted_loss_mass'] = ['mean', 'std']
    if 'selective_risk' in combined.columns:
        agg_dict['selective_risk'] = ['mean', 'std']
    if 'selective_acc' in combined.columns:
        agg_dict['selective_acc'] = ['mean', 'std', 'count']
    
    stats = combined.groupby('tau').agg(agg_dict).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns]
    
    return stats


def aggregate_ood_results(method_dir, seeds):
    """Aggregate OOD evaluation results across seeds."""
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
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Group by tau or alpha depending on available columns
    group_col = 'alpha' if 'alpha' in combined.columns else 'tau'
    
    agg_dict = {}
    for col in ['dar', 'id_accept_rate', 'ood_accept_rate',
                'id_coverage', 'ood_coverage']:
        if col in combined.columns:
            agg_dict[col] = ['mean', 'std']
    # add count from any existing column
    first_metric = next(iter(agg_dict), None)
    if first_metric:
        agg_dict[first_metric] = ['mean', 'std', 'count']
    
    stats = combined.groupby(group_col).agg(agg_dict).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns]
    
    return stats


def create_summary_table(aggregated_results):
    """Create summary table comparing methods across key metrics."""
    summary_rows = []
    
    for method_name, results in aggregated_results.items():
        row = {'method': method_name}
        
        if 'coverage_at_risk' in results:
            cov_df = results['coverage_at_risk']
            alpha_col = [c for c in cov_df.columns if c.startswith('alpha')][0]
            cov_row = cov_df[cov_df[alpha_col] == 0.1]
            if len(cov_row) > 0:
                r = cov_row.iloc[0]
                if 'coverage_mean' in r.index:
                    row['coverage@0.1'] = f"{r['coverage_mean']:.3f} ± {r.get('coverage_std', 0):.3f}"
                if 'accepted_loss_mass_mean' in r.index:
                    row['A(tau)@0.1'] = f"{r['accepted_loss_mass_mean']:.4f} ± {r.get('accepted_loss_mass_std', 0):.4f}"
                if 'violation_gap_mean' in r.index:
                    row['viol_gap@0.1'] = f"{r['violation_gap_mean']:.4f}"
        
        if 'ood_results' in results:
            ood_df = results['ood_results']
            # Find dar_mean column
            dar_cols = [c for c in ood_df.columns if 'dar' in c and 'mean' in c]
            if dar_cols:
                group_col = ood_df.columns[0]
                mid_val = ood_df[group_col].median()
                idx = (ood_df[group_col] - mid_val).abs().argsort()[:1]
                ood_row = ood_df.iloc[idx]
                if len(ood_row) > 0:
                    row['DAR'] = f"{ood_row[dar_cols[0]].values[0]:.3f}"
        
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999],
                       help='list of random seeds used in experiments')
    parser.add_argument('-o', '--output_dir', type=str, 
                       default='./results/aggregated',
                       help='directory to save aggregated results')
    
    args = parser.parse_args()
    main(args)

