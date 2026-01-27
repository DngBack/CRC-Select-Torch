"""
Generate publication-quality figures for paper.

This script creates all figures needed for a paper submission:
1. Risk-Coverage curves with comparison
2. AURC visualization
3. Coverage@Risk bar charts
4. OOD DAR comparison
5. Calibration quality plots

Usage:
    python generate_paper_figures.py \
        --results_dir ../results_paper \
        --methods "CRC-Select" "Vanilla" "Post-hoc-CRC" \
        --output_dir ../paper_figures
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

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})


def load_method_results(results_dir, method_name, seed):
    """Load all results for a method."""
    method_dir = Path(results_dir) / method_name / f'seed_{seed}'
    
    results = {'method': method_name}
    
    # Load RC curve
    rc_path = method_dir / 'risk_coverage_curve.csv'
    if rc_path.exists():
        results['rc_curve'] = pd.read_csv(rc_path)
    
    # Load Coverage@Risk
    cov_risk_path = method_dir / 'coverage_at_risk.csv'
    if cov_risk_path.exists():
        results['coverage_at_risk'] = pd.read_csv(cov_risk_path)
    
    # Load OOD
    ood_path = method_dir / 'ood_evaluation.csv'
    if ood_path.exists():
        results['ood'] = pd.read_csv(ood_path)
    
    # Load summary
    summary_path = method_dir / 'summary.csv'
    if summary_path.exists():
        results['summary'] = pd.read_csv(summary_path)
    
    return results


def figure1_rc_curves(all_results, output_path):
    """
    Figure 1: Risk-Coverage curves comparison.
    Main result figure showing RC curves for all methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Plot 1a: Risk vs Coverage
    for method_results in all_results:
        if 'rc_curve' not in method_results:
            continue
        rc_df = method_results['rc_curve']
        method_name = method_results['method']
        
        ax1.plot(rc_df['coverage'], rc_df['risk'], 
                linewidth=2.5, label=method_name, alpha=0.85)
    
    ax1.axhline(y=0.1, color='red', linestyle='--', 
               linewidth=1.5, alpha=0.6, label='Target risk (α=0.1)')
    ax1.set_xlabel('Coverage', fontweight='bold')
    ax1.set_ylabel('Risk', fontweight='bold')
    ax1.set_title('(a) Risk-Coverage Curves', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(0.15, ax1.get_ylim()[1])])
    
    # Plot 1b: Error vs Coverage (for AURC)
    for method_results in all_results:
        if 'rc_curve' not in method_results:
            continue
        rc_df = method_results['rc_curve']
        method_name = method_results['method']
        
        ax2.plot(rc_df['coverage'], rc_df['error'], 
                linewidth=2.5, label=method_name, alpha=0.85)
        
        # Fill area under curve
        ax2.fill_between(rc_df['coverage'], rc_df['error'], 
                        alpha=0.15)
    
    ax2.set_xlabel('Coverage', fontweight='bold')
    ax2.set_ylabel('Error Rate', fontweight='bold')
    ax2.set_title('(b) Error-Coverage Curves (AURC)', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"✓ Saved Figure 1 to {output_path}")
    plt.close()


def figure2_coverage_at_risk(all_results, output_path):
    """
    Figure 2: Coverage@Risk bar chart.
    Shows maximum coverage achievable at different risk levels.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Prepare data
    methods = []
    alphas = None
    
    for method_results in all_results:
        if 'coverage_at_risk' not in method_results:
            continue
        methods.append(method_results['method'])
        if alphas is None:
            alphas = method_results['coverage_at_risk']['alpha'].values
    
    if alphas is None:
        print("⚠️  No Coverage@Risk data found")
        return
    
    x = np.arange(len(alphas))
    width = 0.8 / len(methods)
    
    for i, method_results in enumerate(all_results):
        if 'coverage_at_risk' not in method_results:
            continue
        
        df = method_results['coverage_at_risk']
        coverages = df['coverage'].values
        method_name = method_results['method']
        
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, coverages, width, 
                     label=method_name, alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label visible bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontsize=8, rotation=0)
    
    ax.set_xlabel('Target Risk Level (α)', fontweight='bold')
    ax.set_ylabel('Maximum Coverage', fontweight='bold')
    ax.set_title('Coverage@Risk: Maximum Coverage at Different Risk Levels', 
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.3f}' for a in alphas])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"✓ Saved Figure 2 to {output_path}")
    plt.close()


def figure3_ood_dar(all_results, output_path):
    """
    Figure 3: OOD Dangerous Acceptance Rate.
    Shows OOD acceptance rates at different thresholds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method_results in all_results:
        if 'ood' not in method_results:
            continue
        
        ood_df = method_results['ood']
        method_name = method_results['method']
        
        ax.plot(ood_df['threshold'], ood_df['dar'], 
               linewidth=2.5, marker='o', markersize=4,
               label=method_name, alpha=0.85)
    
    ax.set_xlabel('Acceptance Threshold (τ)', fontweight='bold')
    ax.set_ylabel('DAR (OOD Acceptance Rate)', fontweight='bold')
    ax.set_title('OOD Safety: Dangerous Acceptance Rate on SVHN', 
                fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"✓ Saved Figure 3 to {output_path}")
    plt.close()


def figure4_aurc_comparison(all_results, output_path):
    """
    Figure 4: AURC comparison bar chart.
    Shows AURC values for all methods (lower is better).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    methods = []
    aurcs = []
    
    for method_results in all_results:
        if 'rc_curve' not in method_results:
            continue
        
        rc_df = method_results['rc_curve']
        method_name = method_results['method']
        
        # Compute AURC
        coverages = rc_df['coverage'].values
        errors = rc_df['error'].values
        sort_idx = np.argsort(coverages)
        aurc = np.trapz(errors[sort_idx], coverages[sort_idx])
        
        methods.append(method_name)
        aurcs.append(aurc)
    
    if not methods:
        print("⚠️  No AURC data found")
        return
    
    colors = sns.color_palette("husl", len(methods))
    bars = ax.bar(methods, aurcs, alpha=0.85, color=colors)
    
    # Add value labels
    for bar, aurc in zip(bars, aurcs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{aurc:.6f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_ylabel('AURC (Area Under Risk-Coverage Curve)', fontweight='bold')
    ax.set_title('AURC Comparison (Lower is Better)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    # Add reference line for paper
    ax.axhline(y=0.02, color='gray', linestyle='--', 
              linewidth=1.5, alpha=0.5, label='SelectiveNet Paper (~0.02-0.04)')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"✓ Saved Figure 4 to {output_path}")
    plt.close()


def table1_summary_comparison(all_results, output_path):
    """
    Table 1: Summary comparison table (CSV format).
    All key metrics for all methods.
    """
    rows = []
    
    for method_results in all_results:
        method_name = method_results['method']
        row = {'Method': method_name}
        
        # AURC
        if 'rc_curve' in method_results:
            rc_df = method_results['rc_curve']
            coverages = rc_df['coverage'].values
            errors = rc_df['error'].values
            sort_idx = np.argsort(coverages)
            aurc = np.trapz(errors[sort_idx], coverages[sort_idx])
            row['AURC'] = f'{aurc:.6f}'
            
            # Error and Risk at 80% coverage
            idx_80 = (rc_df['coverage'] - 0.80).abs().idxmin()
            row['Error@80%'] = f'{rc_df.iloc[idx_80]["error"]:.4f}'
            row['Risk@80%'] = f'{rc_df.iloc[idx_80]["risk"]:.4f}'
            
            # Error and Risk at 90% coverage
            idx_90 = (rc_df['coverage'] - 0.90).abs().idxmin()
            row['Error@90%'] = f'{rc_df.iloc[idx_90]["error"]:.4f}'
            row['Risk@90%'] = f'{rc_df.iloc[idx_90]["risk"]:.4f}'
        
        # Coverage@Risk(0.1)
        if 'coverage_at_risk' in method_results:
            cov_risk_df = method_results['coverage_at_risk']
            row_01 = cov_risk_df[cov_risk_df['alpha'] == 0.1]
            if len(row_01) > 0:
                row['Cov@Risk(0.1)'] = f'{row_01.iloc[0]["coverage"]:.4f}'
        
        # DAR at τ=0.5
        if 'ood' in method_results:
            ood_df = method_results['ood']
            idx_05 = (ood_df['threshold'] - 0.5).abs().idxmin()
            row['DAR@τ=0.5'] = f'{ood_df.iloc[idx_05]["dar"]:.4f}'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    # Also save as LaTeX table
    latex_path = output_path.replace('.csv', '.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format='%.4f'))
    
    print(f"✓ Saved Table 1 to {output_path}")
    print(f"✓ Saved LaTeX table to {latex_path}")
    
    # Print to console
    print("\n" + "=" * 100)
    print("TABLE 1: SUMMARY COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def main(args):
    print("=" * 100)
    print("GENERATING PAPER FIGURES")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Methods: {args.methods}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 100)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results for all methods
    print("\n[Loading Results]")
    all_results = []
    for method_name in args.methods:
        print(f"  Loading {method_name}...")
        results = load_method_results(args.results_dir, method_name, args.seed)
        all_results.append(results)
        print(f"    ✓ Loaded")
    
    # Generate figures
    print("\n[Generating Figures]")
    
    print("\n  Figure 1: Risk-Coverage Curves...")
    figure1_rc_curves(all_results, 
                     os.path.join(args.output_dir, 'figure1_rc_curves.png'))
    
    print("\n  Figure 2: Coverage@Risk...")
    figure2_coverage_at_risk(all_results,
                            os.path.join(args.output_dir, 'figure2_coverage_at_risk.png'))
    
    if not args.skip_ood:
        print("\n  Figure 3: OOD DAR...")
        figure3_ood_dar(all_results,
                       os.path.join(args.output_dir, 'figure3_ood_dar.png'))
    
    print("\n  Figure 4: AURC Comparison...")
    figure4_aurc_comparison(all_results,
                           os.path.join(args.output_dir, 'figure4_aurc_comparison.png'))
    
    # Generate table
    print("\n[Generating Tables]")
    print("\n  Table 1: Summary Comparison...")
    table1_summary_comparison(all_results,
                            os.path.join(args.output_dir, 'table1_summary.csv'))
    
    print("\n" + "=" * 100)
    print(f"✓ All figures and tables saved to: {args.output_dir}")
    print("=" * 100)
    
    print("\n📁 Generated files:")
    print("  Figures (PNG + PDF):")
    print("    • figure1_rc_curves.{png,pdf}")
    print("    • figure2_coverage_at_risk.{png,pdf}")
    if not args.skip_ood:
        print("    • figure3_ood_dar.{png,pdf}")
    print("    • figure4_aurc_comparison.{png,pdf}")
    print("  Tables:")
    print("    • table1_summary.csv")
    print("    • table1_summary.tex (LaTeX format)")


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--methods', type=str, nargs='+', required=True,
                       help='List of method names to compare')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for which to generate figures')
    parser.add_argument('--skip_ood', action='store_true',
                       help='Skip OOD figures')
    parser.add_argument('-o', '--output_dir', type=str, default='../paper_figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    main(args)

