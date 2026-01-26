"""
Plotting utilities for CRC-Select results visualization.

Generates publication-quality figures:
- Risk-Coverage curves comparing methods
- Coverage@Risk bar charts
- DAR comparison plots
- Violation rate analysis
"""
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def plot_risk_coverage_curve(results_dict, output_path, title="Risk-Coverage Curve"):
    """
    Plot risk-coverage curves for multiple methods.
    
    Args:
        results_dict: Dict mapping method names to RC curve DataFrames
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    for i, (method_name, rc_df) in enumerate(results_dict.items()):
        ax.plot(
            rc_df['coverage'], rc_df['risk'],
            marker='o', markersize=6, linewidth=2,
            label=method_name, color=colors[i], alpha=0.8
        )
    
    ax.set_xlabel('Coverage', fontsize=14)
    ax.set_ylabel('Risk', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add target risk lines
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Target risk (α=0.1)')
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Target risk (α=0.05)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Risk-Coverage curve to {output_path}")
    plt.close()


def plot_coverage_at_risk(results_dict, output_path, title="Coverage at Target Risk"):
    """
    Plot bar chart comparing coverage at different risk levels.
    
    Args:
        results_dict: Dict mapping method names to coverage@risk DataFrames
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    methods = list(results_dict.keys())
    alphas = results_dict[methods[0]]['alpha'].unique()
    
    x = np.arange(len(alphas))
    width = 0.8 / len(methods)
    
    colors = sns.color_palette("husl", len(methods))
    
    for i, method_name in enumerate(methods):
        df = results_dict[method_name]
        coverages = [
            df[df['alpha'] == alpha]['coverage'].values[0]
            for alpha in alphas
        ]
        
        ax.bar(
            x + i * width, coverages, width,
            label=method_name, color=colors[i], alpha=0.8
        )
    
    ax.set_xlabel('Target Risk (α)', fontsize=14)
    ax.set_ylabel('Coverage', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'{alpha:.2f}' for alpha in alphas])
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Coverage@Risk chart to {output_path}")
    plt.close()


def plot_ood_dar(results_dict, output_path, title="OOD Dangerous Acceptance Rate"):
    """
    Plot DAR (Dangerous Acceptance Rate) vs threshold.
    
    Args:
        results_dict: Dict mapping method names to OOD evaluation DataFrames
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    for i, (method_name, ood_df) in enumerate(results_dict.items()):
        ax.plot(
            ood_df['tau'], ood_df['dar'],
            marker='s', markersize=6, linewidth=2,
            label=method_name, color=colors[i], alpha=0.8
        )
    
    ax.set_xlabel('Acceptance Threshold (τ)', fontsize=14)
    ax.set_ylabel('DAR (OOD Acceptance Rate)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved DAR plot to {output_path}")
    plt.close()


def plot_mixture_performance(results_dict, output_path, 
                            title="Performance on ID+OOD Mixtures"):
    """
    Plot performance on ID+OOD mixtures.
    
    Args:
        results_dict: Dict mapping method names to mixture evaluation DataFrames
        output_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    # Plot 1: Coverage vs p_ood
    for i, (method_name, mix_df) in enumerate(results_dict.items()):
        # Group by p_ood and compute mean coverage
        grouped = mix_df.groupby('p_ood')['coverage'].mean()
        ax1.plot(
            grouped.index, grouped.values,
            marker='o', markersize=8, linewidth=2,
            label=method_name, color=colors[i], alpha=0.8
        )
    
    ax1.set_xlabel('OOD Proportion (p_ood)', fontsize=14)
    ax1.set_ylabel('Coverage', fontsize=14)
    ax1.set_title('Coverage vs OOD Proportion', fontsize=14)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: OOD acceptance rate vs p_ood
    for i, (method_name, mix_df) in enumerate(results_dict.items()):
        grouped = mix_df.groupby('p_ood')['ood_accept_rate'].mean()
        ax2.plot(
            grouped.index, grouped.values,
            marker='s', markersize=8, linewidth=2,
            label=method_name, color=colors[i], alpha=0.8
        )
    
    ax2.set_xlabel('OOD Proportion (p_ood)', fontsize=14)
    ax2.set_ylabel('OOD Acceptance Rate', fontsize=14)
    ax2.set_title('OOD Acceptance vs OOD Proportion', fontsize=14)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mixture performance plot to {output_path}")
    plt.close()


def plot_violation_rate_heatmap(violation_data, output_path, 
                                title="Risk Violation Rate"):
    """
    Plot heatmap of violation rates across methods and risk levels.
    
    Args:
        violation_data: DataFrame with columns [method, alpha, violation_rate]
        output_path: Path to save figure
        title: Plot title
    """
    # Pivot data for heatmap
    pivot_data = violation_data.pivot(
        index='method', columns='alpha', values='violation_rate'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        pivot_data, annot=True, fmt='.2%', cmap='RdYlGn_r',
        vmin=0, vmax=1, cbar_kws={'label': 'Violation Rate'},
        ax=ax, linewidths=0.5
    )
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Target Risk (α)', fontsize=14)
    ax.set_ylabel('Method', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved violation rate heatmap to {output_path}")
    plt.close()


def load_results_from_directory(results_dir, method_name):
    """
    Load all evaluation results from a directory.
    
    Args:
        results_dir: Path to results directory
        method_name: Name of the method
    
    Returns:
        Dictionary with loaded DataFrames
    """
    results = {'method': method_name}
    
    # Load risk-coverage curve
    rc_path = os.path.join(results_dir, 'risk_coverage_curve.csv')
    if os.path.exists(rc_path):
        results['rc_curve'] = pd.read_csv(rc_path)
    
    # Load coverage at risk
    cov_path = os.path.join(results_dir, 'coverage_at_risk.csv')
    if os.path.exists(cov_path):
        results['coverage_at_risk'] = pd.read_csv(cov_path)
    
    # Load OOD evaluation
    ood_path = os.path.join(results_dir, 'ood_evaluation.csv')
    if os.path.exists(ood_path):
        results['ood_results'] = pd.read_csv(ood_path)
    
    # Load mixture evaluation
    mix_path = os.path.join(results_dir, 'mixture_evaluation.csv')
    if os.path.exists(mix_path):
        results['mixture_results'] = pd.read_csv(mix_path)
    
    return results


def main(args):
    print("=" * 80)
    print("Plotting CRC-Select Results")
    print("=" * 80)
    
    # Load results for all methods
    all_results = {}
    
    for method_dir in args.method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\nLoading results for {method_name}...")
        
        # Find seed directories
        seed_dirs = [
            os.path.join(method_dir, d)
            for d in os.listdir(method_dir)
            if d.startswith('seed_') and os.path.isdir(os.path.join(method_dir, d))
        ]
        
        if not seed_dirs:
            print(f"  Warning: No seed directories found in {method_dir}")
            continue
        
        # Load first seed for visualization (or aggregate across seeds)
        seed_dir = seed_dirs[0]
        results = load_results_from_directory(seed_dir, method_name)
        all_results[method_name] = results
        print(f"  ✓ Loaded from {seed_dir}")
    
    if not all_results:
        print("Error: No results found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== Generate Plots ====================
    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)
    
    # 1. Risk-Coverage curves
    rc_curves = {
        name: res['rc_curve']
        for name, res in all_results.items()
        if 'rc_curve' in res
    }
    if rc_curves:
        plot_risk_coverage_curve(
            rc_curves,
            os.path.join(args.output_dir, 'risk_coverage_curve.png')
        )
    
    # 2. Coverage@Risk bar chart
    cov_at_risk = {
        name: res['coverage_at_risk']
        for name, res in all_results.items()
        if 'coverage_at_risk' in res
    }
    if cov_at_risk:
        plot_coverage_at_risk(
            cov_at_risk,
            os.path.join(args.output_dir, 'coverage_at_risk.png')
        )
    
    # 3. OOD DAR plot
    ood_results = {
        name: res['ood_results']
        for name, res in all_results.items()
        if 'ood_results' in res
    }
    if ood_results:
        plot_ood_dar(
            ood_results,
            os.path.join(args.output_dir, 'ood_dar.png')
        )
    
    # 4. Mixture performance
    mix_results = {
        name: res['mixture_results']
        for name, res in all_results.items()
        if 'mixture_results' in res
    }
    if mix_results:
        plot_mixture_performance(
            mix_results,
            os.path.join(args.output_dir, 'mixture_performance.png')
        )
    
    print("\n" + "=" * 80)
    print(f"✓ All plots saved to {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--method_dirs', type=str, nargs='+', required=True,
                       help='directories containing results for each method')
    parser.add_argument('-o', '--output_dir', type=str, default='../figures',
                       help='directory to save plots')
    
    args = parser.parse_args()
    main(args)

