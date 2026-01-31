"""
Compare OOD safety metrics across methods at fixed ID coverage levels.

This script computes OOD-Acceptance@ID-Coverage for multiple methods,
providing a fair comparison of OOD robustness.

Usage:
    python compare_ood_safety.py \
        --results_dir ../results_paper \
        --methods CRC-Select posthoc_crc vanilla \
        --seeds 42 123 456 \
        --id_coverages 0.7 0.8 0.9 \
        --output_dir ../results/ood_comparison
"""
import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_ood_results_at_fixed_coverage(method_dir, seed):
    """Load OOD results at fixed ID coverage for a specific seed."""
    seed_dir = os.path.join(method_dir, f'seed_{seed}')
    ood_path = os.path.join(seed_dir, 'ood_at_fixed_id_coverage.csv')
    
    if os.path.exists(ood_path):
        return pd.read_csv(ood_path)
    else:
        return None


def aggregate_ood_results(method_dir, method_name, seeds):
    """Aggregate OOD results across seeds for one method."""
    all_results = []
    
    for seed in seeds:
        df = load_ood_results_at_fixed_coverage(method_dir, seed)
        if df is not None:
            df['seed'] = seed
            df['method'] = method_name
            all_results.append(df)
    
    if not all_results:
        return None
    
    # Concatenate all seeds
    combined = pd.concat(all_results, ignore_index=True)
    
    # Compute statistics for each target coverage
    stats = combined.groupby('id_coverage_target').agg({
        'ood_accept_rate': ['mean', 'std', 'count'],
        'safety_ratio': ['mean', 'std'],
        'threshold': ['mean', 'std'],
        'id_coverage_actual': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = [
        'id_coverage_target',
        'ood_accept_mean', 'ood_accept_std', 'n_seeds',
        'safety_ratio_mean', 'safety_ratio_std',
        'threshold_mean', 'threshold_std',
        'id_coverage_mean', 'id_coverage_std'
    ]
    
    stats['method'] = method_name
    
    return stats


def plot_ood_comparison(comparison_df, output_path):
    """Plot OOD acceptance rates across methods and coverages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: OOD Acceptance Rate
    for method in comparison_df['method'].unique():
        method_data = comparison_df[comparison_df['method'] == method]
        ax1.errorbar(
            method_data['id_coverage_target'] * 100,
            method_data['ood_accept_mean'] * 100,
            yerr=method_data['ood_accept_std'] * 100,
            marker='o', linewidth=2, capsize=5,
            label=method
        )
    
    ax1.set_xlabel('ID Coverage (%)', fontsize=12)
    ax1.set_ylabel('OOD Acceptance Rate (%)', fontsize=12)
    ax1.set_title('OOD Acceptance @ Fixed ID Coverage\n(Lower is Better)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Safety Ratio
    for method in comparison_df['method'].unique():
        method_data = comparison_df[comparison_df['method'] == method]
        ax2.errorbar(
            method_data['id_coverage_target'] * 100,
            method_data['safety_ratio_mean'],
            yerr=method_data['safety_ratio_std'],
            marker='s', linewidth=2, capsize=5,
            label=method
        )
    
    ax2.set_xlabel('ID Coverage (%)', fontsize=12)
    ax2.set_ylabel('Safety Ratio (ID/OOD Accept)', fontsize=12)
    ax2.set_title('Safety Ratio @ Fixed ID Coverage\n(Higher is Better)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  ✓ Saved plot to {output_path}")


def plot_ood_heatmap(comparison_df, output_path):
    """Plot heatmap of OOD acceptance rates."""
    # Pivot for heatmap
    pivot_data = comparison_df.pivot(
        index='method',
        columns='id_coverage_target',
        values='ood_accept_mean'
    ) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'OOD Acceptance Rate (%)'},
        ax=ax,
        vmin=0,
        vmax=pivot_data.max().max() * 1.2
    )
    
    ax.set_xlabel('ID Coverage Target', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('OOD Acceptance Rate Heatmap\n(Lower is Better)', fontsize=13)
    
    # Format x-axis labels as percentages
    ax.set_xticklabels([f'{int(float(label.get_text())*100)}%' 
                        for label in ax.get_xticklabels()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  ✓ Saved heatmap to {output_path}")


def generate_latex_table(comparison_df, output_path):
    """Generate LaTeX table for paper."""
    methods = comparison_df['method'].unique()
    coverages = sorted(comparison_df['id_coverage_target'].unique())
    
    latex_str = "\\begin{tabular}{l" + "c" * len(coverages) * 2 + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "Method"
    
    for cov in coverages:
        latex_str += f" & \\multicolumn{{2}}{{c}}{{{int(cov*100)}\\% ID Cov.}}"
    latex_str += " \\\\\n"
    
    latex_str += "\\cmidrule(lr){2-" + str(1 + len(coverages) * 2) + "}\n"
    latex_str += ""
    
    for _ in coverages:
        latex_str += " & OOD Acc. & Safety"
    latex_str += " \\\\\n"
    latex_str += "\\midrule\n"
    
    for method in methods:
        latex_str += method.replace('_', '\\_')
        for cov in coverages:
            row = comparison_df[
                (comparison_df['method'] == method) & 
                (comparison_df['id_coverage_target'] == cov)
            ]
            if len(row) > 0:
                ood_mean = row['ood_accept_mean'].values[0] * 100
                ood_std = row['ood_accept_std'].values[0] * 100
                safety = row['safety_ratio_mean'].values[0]
                
                latex_str += f" & ${ood_mean:.2f} \\pm {ood_std:.2f}$ & ${safety:.1f}\\times$"
            else:
                latex_str += " & - & -"
        latex_str += " \\\\\n"
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}"
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"  ✓ Saved LaTeX table to {output_path}")
    return latex_str


def main(args):
    print("=" * 80)
    print("OOD Safety Comparison Across Methods")
    print("=" * 80)
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Results directory: {args.results_dir}")
    print("=" * 80)
    
    # Aggregate results for each method
    all_aggregated = []
    
    for method_name in args.methods:
        print(f"\n[{method_name}] Loading and aggregating results...")
        method_dir = os.path.join(args.results_dir, method_name)
        
        if not os.path.exists(method_dir):
            print(f"  ⚠️  Directory not found: {method_dir}")
            continue
        
        stats = aggregate_ood_results(method_dir, method_name, args.seeds)
        
        if stats is not None:
            all_aggregated.append(stats)
            print(f"  ✓ Aggregated across {stats['n_seeds'].max()} seeds")
            
            # Print summary
            print(f"\n  OOD Acceptance @ Fixed ID Coverage:")
            for _, row in stats.iterrows():
                print(f"    {row['id_coverage_target']*100:.0f}% ID: "
                      f"OOD={row['ood_accept_mean']*100:.2f}±{row['ood_accept_std']*100:.2f}%, "
                      f"Safety={row['safety_ratio_mean']:.1f}×")
        else:
            print(f"  ❌ No results found")
    
    if not all_aggregated:
        print("\n❌ No results to compare!")
        return
    
    # Combine all methods
    comparison_df = pd.concat(all_aggregated, ignore_index=True)
    
    # ==================== Save Results ====================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Save comparison table
    comparison_path = os.path.join(args.output_dir, 'ood_safety_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  ✓ Saved comparison to {comparison_path}")
    
    # 2. Create plots
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'ood_comparison_plot.png')
        plot_ood_comparison(comparison_df, plot_path)
        
        heatmap_path = os.path.join(args.output_dir, 'ood_comparison_heatmap.png')
        plot_ood_heatmap(comparison_df, heatmap_path)
    
    # 3. Generate LaTeX table
    if args.latex:
        latex_path = os.path.join(args.output_dir, 'ood_comparison_table.tex')
        latex_str = generate_latex_table(comparison_df, latex_path)
        
        print("\n" + "=" * 80)
        print("LaTeX Table")
        print("=" * 80)
        print(latex_str)
    
    # ==================== Print Summary ====================
    print("\n" + "=" * 80)
    print("Summary: OOD Acceptance Rates (mean ± std)")
    print("=" * 80)
    
    # Print in a nice format
    pivot_display = comparison_df.pivot(
        index='method',
        columns='id_coverage_target',
        values='ood_accept_mean'
    ) * 100
    
    print("\nOOD Acceptance Rate (%):")
    print(pivot_display.to_string())
    
    pivot_safety = comparison_df.pivot(
        index='method',
        columns='id_coverage_target',
        values='safety_ratio_mean'
    )
    
    print("\nSafety Ratio (ID/OOD):")
    print(pivot_safety.to_string())
    
    # Find best method at each coverage
    print("\n" + "=" * 80)
    print("Best Method at Each Coverage Level")
    print("=" * 80)
    
    for cov in sorted(comparison_df['id_coverage_target'].unique()):
        cov_data = comparison_df[comparison_df['id_coverage_target'] == cov]
        best_idx = cov_data['ood_accept_mean'].idxmin()
        best = cov_data.loc[best_idx]
        
        print(f"\n{best['id_coverage_target']*100:.0f}% ID Coverage:")
        print(f"  Best: {best['method']}")
        print(f"  OOD Accept: {best['ood_accept_mean']*100:.2f}%")
        print(f"  Safety Ratio: {best['safety_ratio_mean']:.1f}×")
        
        # Show improvement over others
        for _, row in cov_data.iterrows():
            if row['method'] != best['method']:
                improvement = (row['ood_accept_mean'] - best['ood_accept_mean']) / best['ood_accept_mean'] * 100
                print(f"  vs {row['method']}: {improvement:.1f}% lower OOD")
    
    print("\n" + "=" * 80)
    print(f"✓ All results saved to {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser(description='Compare OOD safety across methods')
    
    parser.add_argument('-r', '--results_dir', type=str, 
                       default='../results_paper',
                       help='directory containing results for all methods')
    parser.add_argument('--methods', type=str, nargs='+', required=True,
                       help='list of method names to compare')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456],
                       help='list of random seeds used in experiments')
    parser.add_argument('--id_coverages', type=float, nargs='+',
                       default=[0.6, 0.7, 0.8, 0.9],
                       help='ID coverage levels to analyze')
    parser.add_argument('-o', '--output_dir', type=str,
                       default='../results/ood_comparison',
                       help='directory to save comparison results')
    parser.add_argument('--plot', action='store_true',
                       help='generate comparison plots')
    parser.add_argument('--latex', action='store_true',
                       help='generate LaTeX table')
    
    args = parser.parse_args()
    main(args)
