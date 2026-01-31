"""
Compute risk violation rate across multiple seeds.

A violation occurs when risk(test) > alpha at the calibrated threshold.
This is a key metric for evaluating conformal risk control guarantees.

Usage:
    python compute_violation_rate.py \
        --results_dir ../results_paper/CRC-Select \
        --seeds 42 123 456 789 999 \
        --alpha 0.1 \
        --margin 0.0
"""
import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import pandas as pd
from pathlib import Path


def compute_violation_rate_for_method(
    method_dir: str,
    seeds: list,
    alpha: float,
    margin: float = 0.0
) -> dict:
    """
    Compute violation rate for a single method across seeds.
    
    A violation occurs when risk(test) > alpha * (1 + margin).
    
    Args:
        method_dir: Directory containing results for the method
        seeds: List of random seeds
        alpha: Target risk level
        margin: Tolerance margin (e.g., 0.1 means allow 10% violation)
    
    Returns:
        Dictionary with violation statistics
    """
    results = []
    missing_seeds = []
    
    for seed in seeds:
        seed_dir = os.path.join(method_dir, f'seed_{seed}')
        cov_at_risk_path = os.path.join(seed_dir, 'coverage_at_risk.csv')
        
        if not os.path.exists(cov_at_risk_path):
            print(f"  ⚠️  Missing results for seed {seed}")
            missing_seeds.append(seed)
            continue
        
        # Load results
        df = pd.read_csv(cov_at_risk_path)
        
        # Find row for target alpha
        alpha_rows = df[df['alpha'] == alpha]
        if len(alpha_rows) == 0:
            print(f"  ⚠️  No results for alpha={alpha} in seed {seed}")
            continue
        
        row = alpha_rows.iloc[0]
        
        # Check violation with margin
        threshold = alpha * (1.0 + margin)
        violated = row['risk'] > threshold
        
        results.append({
            'seed': seed,
            'alpha': alpha,
            'risk': row['risk'],
            'coverage': row['coverage'],
            'threshold_tau': row['threshold'],
            'feasible': row.get('feasible', True),
            'violated': violated,
            'risk_excess': max(0, row['risk'] - alpha)
        })
    
    if len(results) == 0:
        return None
    
    # Compute statistics
    violations = [r['violated'] for r in results]
    risks = [r['risk'] for r in results]
    coverages = [r['coverage'] for r in results]
    
    violation_rate = np.mean(violations)
    num_violations = np.sum(violations)
    
    return {
        'alpha': alpha,
        'margin': margin,
        'violation_threshold': alpha * (1.0 + margin),
        'violation_rate': violation_rate,
        'num_violations': int(num_violations),
        'num_seeds': len(results),
        'num_missing': len(missing_seeds),
        'mean_risk': np.mean(risks),
        'std_risk': np.std(risks),
        'min_risk': np.min(risks),
        'max_risk': np.max(risks),
        'mean_coverage': np.mean(coverages),
        'std_coverage': np.std(coverages),
        'results_per_seed': results
    }


def main(args):
    print("=" * 80)
    print("Computing Risk Violation Rate Across Seeds")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Alpha values: {args.alphas}")
    print(f"Margin: {args.margin * 100:.1f}%")
    print("=" * 80)
    
    # Process each method directory
    all_results = {}
    
    for method_dir in args.method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\n[{method_name}]")
        
        method_results = {}
        
        for alpha in args.alphas:
            print(f"  Alpha = {alpha:.3f}")
            result = compute_violation_rate_for_method(
                method_dir, args.seeds, alpha, args.margin
            )
            
            if result is None:
                print(f"    ❌ No results available")
                continue
            
            method_results[alpha] = result
            
            # Print summary
            print(f"    Violation rate: {result['violation_rate']*100:.1f}% "
                  f"({result['num_violations']}/{result['num_seeds']})")
            print(f"    Mean risk: {result['mean_risk']:.4f} ± {result['std_risk']:.4f}")
            print(f"    Mean coverage: {result['mean_coverage']:.4f} ± {result['std_coverage']:.4f}")
            
            if result['num_missing'] > 0:
                print(f"    ⚠️  Missing {result['num_missing']} seeds")
        
        all_results[method_name] = method_results
    
    # ==================== Save Results ====================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Save detailed results per method
    for method_name, method_results in all_results.items():
        for alpha, result in method_results.items():
            # Summary
            summary_df = pd.DataFrame([{
                'method': method_name,
                'alpha': result['alpha'],
                'margin': result['margin'],
                'violation_rate': result['violation_rate'],
                'num_violations': result['num_violations'],
                'num_seeds': result['num_seeds'],
                'mean_risk': result['mean_risk'],
                'std_risk': result['std_risk'],
                'mean_coverage': result['mean_coverage'],
                'std_coverage': result['std_coverage']
            }])
            
            summary_path = os.path.join(
                args.output_dir, 
                f'{method_name}_alpha_{alpha:.3f}_violation_summary.csv'
            )
            summary_df.to_csv(summary_path, index=False)
            
            # Per-seed details
            details_df = pd.DataFrame(result['results_per_seed'])
            details_path = os.path.join(
                args.output_dir,
                f'{method_name}_alpha_{alpha:.3f}_violation_details.csv'
            )
            details_df.to_csv(details_path, index=False)
    
    # 2. Create comparison table across methods
    comparison_rows = []
    for method_name, method_results in all_results.items():
        for alpha, result in method_results.items():
            comparison_rows.append({
                'method': method_name,
                'alpha': alpha,
                'violation_rate': result['violation_rate'],
                'num_violations': result['num_violations'],
                'num_seeds': result['num_seeds'],
                'mean_risk': result['mean_risk'],
                'std_risk': result['std_risk'],
                'mean_coverage': result['mean_coverage'],
                'std_coverage': result['std_coverage']
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(args.output_dir, 'violation_rate_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"  ✓ Saved comparison table to {comparison_path}")
    
    # 3. Print comparison table
    print("\n" + "=" * 80)
    print("Comparison Table: Risk Violation Rates")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # 4. Generate LaTeX table
    if args.generate_latex:
        print("\n" + "=" * 80)
        print("LaTeX Table")
        print("=" * 80)
        
        # Pivot table: rows=methods, columns=alphas
        pivot_df = comparison_df.pivot(
            index='method',
            columns='alpha',
            values=['violation_rate', 'mean_risk', 'mean_coverage']
        )
        
        latex_str = "\\begin{tabular}{l" + "c" * len(args.alphas) * 3 + "}\n"
        latex_str += "\\toprule\n"
        latex_str += "Method"
        
        for alpha in args.alphas:
            latex_str += f" & \\multicolumn{{3}}{{c}}{{$\\alpha={alpha}$}}"
        latex_str += " \\\\\n"
        
        latex_str += "\\cmidrule(lr){2-" + str(1 + len(args.alphas) * 3) + "}\n"
        latex_str += ""
        
        for alpha in args.alphas:
            latex_str += " & Viol. & Risk & Cov."
        latex_str += " \\\\\n"
        latex_str += "\\midrule\n"
        
        for method in comparison_df['method'].unique():
            latex_str += method.replace('_', '\\_')
            for alpha in args.alphas:
                row = comparison_df[(comparison_df['method'] == method) & 
                                   (comparison_df['alpha'] == alpha)]
                if len(row) > 0:
                    viol_rate = row['violation_rate'].values[0]
                    mean_risk = row['mean_risk'].values[0]
                    mean_cov = row['mean_coverage'].values[0]
                    latex_str += f" & {viol_rate*100:.1f}\\% & {mean_risk:.3f} & {mean_cov:.3f}"
                else:
                    latex_str += " & - & - & -"
            latex_str += " \\\\\n"
        
        latex_str += "\\bottomrule\n"
        latex_str += "\\end{tabular}"
        
        print(latex_str)
        
        latex_path = os.path.join(args.output_dir, 'violation_rate_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"\n✓ Saved LaTeX table to {latex_path}")
    
    # ==================== Summary Statistics ====================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for method_name, method_results in all_results.items():
        print(f"\n{method_name}:")
        for alpha, result in method_results.items():
            vr = result['violation_rate'] * 100
            target = alpha
            status = "✅" if vr <= 20 else "⚠️" if vr <= 30 else "❌"
            print(f"  α={alpha:.2f}: {vr:.1f}% violations "
                  f"(risk: {result['mean_risk']:.4f}±{result['std_risk']:.4f}) {status}")
    
    print("\n" + "=" * 80)
    print(f"✓ All results saved to {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser(description='Compute risk violation rate across seeds')
    
    parser.add_argument('--method_dirs', type=str, nargs='+', required=True,
                       help='directories containing results for each method')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 123, 456, 789, 999],
                       help='list of random seeds used in experiments')
    parser.add_argument('--alphas', type=float, nargs='+',
                       default=[0.05, 0.1, 0.15, 0.2],
                       help='risk levels to evaluate')
    parser.add_argument('--margin', type=float, default=0.0,
                       help='tolerance margin for violations (e.g., 0.1 = 10%% slack)')
    parser.add_argument('-o', '--output_dir', type=str,
                       default='../results/violation_rate',
                       help='directory to save results')
    parser.add_argument('--generate_latex', action='store_true',
                       help='generate LaTeX table')
    
    args = parser.parse_args()
    main(args)
