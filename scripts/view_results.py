"""
Quick script to view all evaluation results in a nice format.

Usage:
    python3 view_results.py --results_dir ../results_paper
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import pandas as pd
import numpy as np


def print_header(title):
    """Print a nice header."""
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)


def load_method_results(results_dir, method_name, seed):
    """Load all results for a method and seed."""
    method_dir = Path(results_dir) / method_name / f'seed_{seed}'
    
    if not method_dir.exists():
        return None
    
    results = {}
    
    # Load all CSV files
    for csv_file in method_dir.glob('*.csv'):
        results[csv_file.stem] = pd.read_csv(csv_file)
    
    return results


def display_rc_curve_summary(rc_df):
    """Display Risk-Coverage curve summary."""
    print("\n📈 Risk-Coverage Curve (101 points)")
    print("-" * 100)
    
    # Compute AURC
    coverages = rc_df['coverage'].values
    errors = rc_df['error'].values
    sort_idx = np.argsort(coverages)
    aurc = np.trapz(errors[sort_idx], coverages[sort_idx])
    
    print(f"  AURC (Area Under Risk-Coverage): {aurc:.6f} (lower is better)")
    print()
    
    # Key points
    print(f"  {'Coverage':<12} {'Threshold':<12} {'Accuracy':<12} {'Error':<12} {'Risk':<12}")
    print("  " + "-" * 60)
    
    for cov_target in [0.60, 0.70, 0.80, 0.90, 0.95, 1.0]:
        idx = (rc_df['coverage'] - cov_target).abs().idxmin()
        row = rc_df.iloc[idx]
        print(f"  {row['coverage']:<12.4f} {row['threshold']:<12.4f} "
              f"{row['accuracy']:<12.4f} {row['error']:<12.4f} {row['risk']:<12.4f}")


def display_coverage_at_risk(cov_risk_df):
    """Display Coverage@Risk results."""
    print("\n📊 Coverage @ Risk Levels")
    print("-" * 100)
    
    print(f"  {'Risk Level (α)':<15} {'Max Coverage':<15} {'Actual Risk':<15} {'Threshold (τ)':<15} {'Feasible'}")
    print("  " + "-" * 75)
    
    for _, row in cov_risk_df.iterrows():
        feasible = "✓" if row['feasible'] else "✗"
        if row['feasible']:
            print(f"  {row['alpha']:<15.3f} {row['coverage']:<15.4f} "
                  f"{row['risk']:<15.4f} {row['threshold']:<15.4f} {feasible}")
        else:
            print(f"  {row['alpha']:<15.3f} {'N/A':<15} {'N/A':<15} {'N/A':<15} {feasible}")


def display_ood_results(ood_df):
    """Display OOD evaluation results."""
    print("\n🌍 OOD Evaluation (SVHN Dangerous Acceptance Rate)")
    print("-" * 100)
    
    print(f"  {'Threshold (τ)':<15} {'ID Accept':<15} {'OOD Accept (DAR)':<20} {'Safety Ratio'}")
    print("  " + "-" * 70)
    
    for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
        idx = (ood_df['threshold'] - tau).abs().idxmin()
        row = ood_df.iloc[idx]
        
        safety_ratio = row['id_accept_rate'] / (row['dar'] + 1e-8)
        
        print(f"  {row['threshold']:<15.3f} {row['id_accept_rate']:<15.4f} "
              f"{row['dar']:<20.4f} {safety_ratio:<10.2f}×")


def display_summary(summary_df):
    """Display complete summary."""
    print("\n📋 Complete Summary")
    print("-" * 100)
    
    # Key metrics to highlight
    key_metrics = [
        'aurc',
        'error_at_cov_0.80',
        'risk_at_cov_0.80',
        'accuracy_at_cov_0.80',
        'coverage_at_risk_0.100'
    ]
    
    print("\n  Key Performance Metrics:")
    for metric in key_metrics:
        if metric in summary_df.columns:
            value = summary_df[metric].values[0]
            metric_display = metric.replace('_', ' ').title()
            print(f"    • {metric_display:<40}: {value:.6f}")
    
    print("\n  All Metrics:")
    for col in summary_df.columns:
        if col not in key_metrics and col not in ['method', 'seed', 'num_test_samples']:
            value = summary_df[col].values[0]
            if isinstance(value, (int, float)):
                print(f"    • {col:<40}: {value:.6f}")


def compare_with_paper(results):
    """Compare with SelectiveNet paper results."""
    print_header("COMPARISON WITH SELECTIVENET PAPER")
    
    if 'risk_coverage_curve' not in results:
        print("⚠️  RC curve not found")
        return
    
    rc_df = results['risk_coverage_curve']
    
    # Paper results (approximate)
    paper_results = {
        0.70: {'error': 0.08},
        0.80: {'error': 0.06},
        0.90: {'error': 0.04},
    }
    
    print(f"\n  {'Coverage':<12} {'Paper Error':<15} {'CRC-Select Error':<20} {'Improvement'}")
    print("  " + "-" * 65)
    
    for cov_target, paper_data in paper_results.items():
        idx = (rc_df['coverage'] - cov_target).abs().idxmin()
        our_error = rc_df.iloc[idx]['error']
        paper_error = paper_data['error']
        
        improvement = (paper_error - our_error) / paper_error * 100
        
        print(f"  {cov_target:<12.2f} {paper_error:<15.4f} {our_error:<20.4f} {improvement:>6.1f}%")
    
    # AURC comparison
    coverages = rc_df['coverage'].values
    errors = rc_df['error'].values
    sort_idx = np.argsort(coverages)
    aurc = np.trapz(errors[sort_idx], coverages[sort_idx])
    
    paper_aurc_est = 0.03  # Estimated
    aurc_improvement = (paper_aurc_est - aurc) / paper_aurc_est * 100
    
    print(f"\n  AURC:")
    print(f"    Paper (estimated): ~0.02-0.04")
    print(f"    CRC-Select:        {aurc:.6f}")
    print(f"    Improvement:       ~{aurc_improvement:.1f}%")


def main(args):
    print_header("CRC-SELECT EVALUATION RESULTS VIEWER")
    
    results_dir = Path(args.results_dir)
    
    # Find all methods
    if results_dir.exists():
        methods = [d.name for d in results_dir.iterdir() 
                  if d.is_dir() and d.name != 'aggregated']
    else:
        print(f"⚠️  Results directory not found: {results_dir}")
        return
    
    print(f"\nResults directory: {results_dir}")
    print(f"Methods found: {methods}")
    print(f"Seed: {args.seed}")
    
    # Display results for each method
    for method_name in methods:
        print_header(f"{method_name.upper()} RESULTS")
        
        results = load_method_results(results_dir, method_name, args.seed)
        
        if results is None:
            print(f"⚠️  No results found for {method_name} (seed={args.seed})")
            continue
        
        print(f"\n✓ Loaded {len(results)} result files")
        
        # Display each result type
        if 'risk_coverage_curve' in results:
            display_rc_curve_summary(results['risk_coverage_curve'])
        
        if 'coverage_at_risk' in results:
            display_coverage_at_risk(results['coverage_at_risk'])
        
        if 'ood_evaluation' in results:
            display_ood_results(results['ood_evaluation'])
        
        if 'summary' in results:
            display_summary(results['summary'])
        
        # Compare with paper (only for first method)
        if method_name == methods[0]:
            compare_with_paper(results)
    
    print_header("END OF RESULTS")
    
    # Generate quick comparison table if multiple methods
    if len(methods) > 1:
        print("\n" + "=" * 100)
        print("QUICK COMPARISON TABLE")
        print("=" * 100)
        
        comparison_data = []
        for method_name in methods:
            results = load_method_results(results_dir, method_name, args.seed)
            if results and 'summary' in results:
                summary = results['summary']
                row = {
                    'Method': method_name,
                    'AURC': summary['aurc'].values[0] if 'aurc' in summary.columns else np.nan,
                }
                
                # Add key metrics
                for col in summary.columns:
                    if 'error_at_cov' in col or 'coverage_at_risk' in col:
                        row[col] = summary[col].values[0]
                
                comparison_data.append(row)
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            print("\n" + comp_df.to_string(index=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--results_dir', type=str, default='../results_paper',
                       help='Directory containing evaluation results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed to view results for')
    
    args = parser.parse_args()
    main(args)

