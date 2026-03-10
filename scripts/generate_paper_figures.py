"""
Generate publication-quality figures for paper.

Figures (per spec):
  Fig 1 – Method diagram (placeholder / TikZ)
  Fig 2 – RC frontier: accepted-loss mass vs coverage
  Fig 3 – Coverage@alpha grouped bar chart
  Fig 4 – Violation gap box-plot across seeds
  Fig 5 – OOD DAR comparison
  Fig 6 – Ablation curves (if data available)
  Fig 7 – Threshold efficiency (conservativeness)

Tables:
  Table 1 – Main results (all metrics, all methods)
  Table 2 – OOD safety
  Table 3 – Ablation summary
  Table 4 – Violation rates across seeds

Usage:
    python generate_paper_figures.py \
        --results_dir ../results_paper \
        --methods "CRC-Select" "vanilla" "posthoc_crc" "MSP" "TempScaled_MSP" "Energy" \
        --seeds 42 123 456 789 999 \
        --output_dir ../figures
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


def load_method_results(results_dir, method_name, seeds):
    """Load results for a method across multiple seeds."""
    results = {'method': method_name, 'seeds': {}}
    for seed in seeds:
        seed_dir = Path(results_dir) / method_name / f'seed_{seed}'
        seed_data = {}
        for fname in ['risk_coverage_curve.csv', 'all_metrics.csv',
                      'coverage_at_risk.csv', 'ood_evaluation.csv',
                      'ood_at_fixed_id_coverage.csv', 'summary.csv']:
            fpath = seed_dir / fname
            if fpath.exists():
                seed_data[fname.replace('.csv', '')] = pd.read_csv(fpath)
        if seed_data:
            results['seeds'][seed] = seed_data
    return results


def _get_rc(results, seed):
    """Get RC curve DataFrame for a method/seed."""
    return results['seeds'].get(seed, {}).get('risk_coverage_curve')


def _get_metrics(results, seed):
    """Get all_metrics DataFrame for a method/seed."""
    return results['seeds'].get(seed, {}).get('all_metrics')


# ------------------------------------------------------------------
# Figure 2: RC frontier (accepted-loss mass on y-axis)
# ------------------------------------------------------------------

def figure2_rc_frontier(all_results, seed, output_path):
    """Accepted-loss mass vs Coverage (the CRC-certified view)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for res in all_results:
        rc = _get_rc(res, seed)
        if rc is None:
            continue
        label = res['method']
        # (a) accepted-loss mass
        if 'accepted_loss_mass' in rc.columns:
            ax1.plot(rc['coverage'], rc['accepted_loss_mass'],
                     linewidth=2.5, label=label, alpha=0.85)
        # (b) conditional selective risk (descriptive)
        risk_col = 'selective_risk' if 'selective_risk' in rc.columns else 'risk'
        if risk_col in rc.columns:
            ax2.plot(rc['coverage'], rc[risk_col],
                     linewidth=2.5, label=label, alpha=0.85)

    for ax, title, ylabel in [
        (ax1, '(a) Accepted-Loss Mass vs Coverage', 'Accepted-Loss Mass $\\hat A(\\tau)$'),
        (ax2, '(b) Conditional Risk vs Coverage', 'Selective Risk $R_{sel}$'),
    ]:
        ax.axhline(0.1, color='red', ls='--', lw=1.5, alpha=0.6, label='$\\alpha$=0.1')
        ax.set_xlabel('Coverage', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3, ls=':')
        ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"  Saved Figure 2 to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Figure 3: Coverage@alpha grouped bars
# ------------------------------------------------------------------

def figure3_coverage_at_alpha(all_results, seed, output_path):
    """Grouped bar chart: max coverage at each alpha."""
    fig, ax = plt.subplots(figsize=(10, 5))

    method_names = []
    alphas = None
    data = []
    for res in all_results:
        m = _get_metrics(res, seed)
        if m is None:
            continue
        method_names.append(res['method'])
        if alphas is None:
            alphas = m['alpha'].values
        data.append(m['test_coverage'].values)

    if alphas is None:
        print("  No Coverage@Alpha data found")
        return

    x = np.arange(len(alphas))
    width = 0.8 / max(len(method_names), 1)
    for i, (name, cov) in enumerate(zip(method_names, data)):
        offset = (i - len(method_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, cov, width, label=name, alpha=0.85)
        for b in bars:
            h = b.get_height()
            if h > 0.05:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Target Risk Level ($\\alpha$)', fontweight='bold')
    ax.set_ylabel('Coverage at CRC-calibrated $\\hat\\tau$', fontweight='bold')
    ax.set_title('Coverage @ $\\alpha$ (higher is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, ls=':', axis='y')
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"  Saved Figure 3 to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Figure 4: Violation gap box-plot across seeds
# ------------------------------------------------------------------

def figure4_violation_gap(all_results, seeds, output_path, alpha=0.1):
    """Box-plot of violation gaps across seeds for each method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = []
    for res in all_results:
        for s in seeds:
            m = _get_metrics(res, s)
            if m is None:
                continue
            row = m[m['alpha'].round(3) == round(alpha, 3)]
            if len(row) == 0:
                continue
            plot_data.append({
                'method': res['method'],
                'violation_gap': row.iloc[0].get('violation_gap', 0.0),
            })
    if not plot_data:
        print("  No violation gap data found")
        return

    df = pd.DataFrame(plot_data)
    sns.boxplot(x='method', y='violation_gap', data=df, ax=ax, palette='Set2')
    ax.axhline(0, color='green', ls='--', lw=1.5, alpha=0.6, label='No violation')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel(f'Violation Gap $\\max(\\hat A - {alpha}, 0)$', fontweight='bold')
    ax.set_title(f'Violation Gap Distribution ($\\alpha$={alpha})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, ls=':', axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"  Saved Figure 4 to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Figure 5: OOD DAR comparison
# ------------------------------------------------------------------

def figure5_ood_dar(all_results, seed, output_path):
    """OOD acceptance rate at fixed ID coverage."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for res in all_results:
        ood = res['seeds'].get(seed, {}).get('ood_at_fixed_id_coverage')
        if ood is None:
            ood = res['seeds'].get(seed, {}).get('ood_evaluation')
            if ood is not None:
                ax.plot(ood['id_accept_rate'], ood['dar'],
                        linewidth=2.5, marker='o', ms=4,
                        label=res['method'], alpha=0.85)
            continue
        ax.plot(ood['id_coverage_actual'], ood['ood_accept_rate'],
                linewidth=2.5, marker='s', ms=5,
                label=res['method'], alpha=0.85)

    ax.set_xlabel('ID Coverage', fontweight='bold')
    ax.set_ylabel('OOD Acceptance (DAR)', fontweight='bold')
    ax.set_title('OOD Safety: DAR at Fixed ID Coverage', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"  Saved Figure 5 to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Figure 7: Threshold efficiency
# ------------------------------------------------------------------

def figure7_threshold_efficiency(all_results, seeds, output_path, alpha=0.1):
    """Bar chart of conservativeness and coverage efficiency across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    methods, cons_vals, eff_vals = [], [], []
    for res in all_results:
        cons_list, eff_list = [], []
        for s in seeds:
            m = _get_metrics(res, s)
            if m is None:
                continue
            row = m[m['alpha'].round(3) == round(alpha, 3)]
            if len(row) == 0:
                continue
            cons_list.append(row.iloc[0].get('conservativeness', 0.0))
            eff_list.append(row.iloc[0].get('coverage_efficiency', 0.0))
        if cons_list:
            methods.append(res['method'])
            cons_vals.append(np.mean(cons_list))
            eff_vals.append(np.mean(eff_list))

    if not methods:
        print("  No threshold efficiency data found")
        return

    colors = sns.color_palette('Set2', len(methods))
    ax1.bar(methods, cons_vals, color=colors, alpha=0.85)
    ax1.set_ylabel('Conservativeness ($\\alpha - \\hat A$)', fontweight='bold')
    ax1.set_title('(a) Conservativeness (closer to 0 = tighter)', fontweight='bold')
    ax1.axhline(0, color='red', ls='--', lw=1)
    ax1.grid(True, alpha=0.3, ls=':', axis='y')

    ax2.bar(methods, eff_vals, color=colors, alpha=0.85)
    ax2.set_ylabel('Coverage Efficiency $C / C_{oracle}$', fontweight='bold')
    ax2.set_title('(b) Coverage Efficiency (closer to 1 = better)', fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, ls=':', axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"  Saved Figure 7 to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------

def table1_main_results(all_results, seeds, output_path):
    """Table 1: main results across all methods + seeds."""
    rows = []
    for res in all_results:
        for s in seeds:
            m = _get_metrics(res, s)
            if m is None:
                continue
            for _, r in m.iterrows():
                row = {'method': res['method'], 'seed': s}
                for col in m.columns:
                    row[col] = r[col]
                rows.append(row)
    if not rows:
        print("  No data for Table 1")
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    # Mean +/- std aggregation
    agg = df.groupby(['method', 'alpha']).agg(['mean', 'std']).reset_index()
    agg_path = output_path.replace('.csv', '_agg.csv')
    agg.to_csv(agg_path, index=False)
    print(f"  Saved Table 1 to {output_path} and {agg_path}")


def table4_violation_rates(all_results, seeds, output_path, alpha=0.1):
    """Table 4: violation count per method across seeds."""
    rows = []
    for res in all_results:
        n_seeds = 0
        n_viol = 0
        for s in seeds:
            m = _get_metrics(res, s)
            if m is None:
                continue
            row = m[m['alpha'].round(3) == round(alpha, 3)]
            if len(row) == 0:
                continue
            n_seeds += 1
            if row.iloc[0].get('violated', False):
                n_viol += 1
        if n_seeds > 0:
            rows.append({
                'method': res['method'],
                'alpha': alpha,
                'num_seeds': n_seeds,
                'violations': n_viol,
                'violation_rate': n_viol / n_seeds,
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"  Saved Table 4 to {output_path}")
    else:
        print("  No data for Table 4")


def main(args):
    print("=" * 100)
    print("GENERATING PAPER FIGURES & TABLES")
    print("=" * 100)

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = args.seeds
    primary_seed = seeds[0]

    # Load results
    print("\n[Loading Results]")
    all_results = []
    for method in args.methods:
        res = load_method_results(args.results_dir, method, seeds)
        all_results.append(res)
        n_seeds = len(res['seeds'])
        print(f"  {method}: {n_seeds} seed(s) loaded")

    out = args.output_dir

    # Figures
    print("\n[Generating Figures]")
    figure2_rc_frontier(all_results, primary_seed,
                        os.path.join(out, 'fig2_rc_frontier.png'))
    figure3_coverage_at_alpha(all_results, primary_seed,
                              os.path.join(out, 'fig3_coverage_at_alpha.png'))
    figure4_violation_gap(all_results, seeds,
                          os.path.join(out, 'fig4_violation_gap.png'))
    if not args.skip_ood:
        figure5_ood_dar(all_results, primary_seed,
                        os.path.join(out, 'fig5_ood_dar.png'))
    figure7_threshold_efficiency(all_results, seeds,
                                 os.path.join(out, 'fig7_threshold_efficiency.png'))

    # Tables
    print("\n[Generating Tables]")
    table1_main_results(all_results, seeds,
                        os.path.join(out, 'table1_main_results.csv'))
    table4_violation_rates(all_results, seeds,
                           os.path.join(out, 'table4_violation_rates.csv'))

    print(f"\nAll outputs saved to: {out}")
    print("=" * 100)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999])
    parser.add_argument('--skip_ood', action='store_true')
    parser.add_argument('-o', '--output_dir', type=str, default='../figures')
    args = parser.parse_args()
    main(args)

