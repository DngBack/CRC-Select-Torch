"""
Master script to run full CRC-Select experiments across multiple seeds.

This script:
1. Trains CRC-Select for multiple seeds
2. Trains vanilla SelectiveNet baseline for multiple seeds
3. Applies post-hoc CRC to vanilla SelectiveNet
4. Evaluates all methods
5. Aggregates results
6. Generates plots
"""
import os
import sys
import subprocess
from argparse import ArgumentParser
import yaml
from pathlib import Path

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=base)
    if result.returncode != 0:
        print(f"\n⚠️  Warning: {description} failed with return code {result.returncode}")
        return False
    print(f"\n✓ {description} completed successfully")
    return True


def main(args):
    print("=" * 80)
    print("CRC-Select Full Experiment Pipeline")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Dataset: {args.dataset}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print("=" * 80)
    
    # Create output directories
    results_dir = Path(args.output_dir)
    checkpoints_dir = results_dir / 'checkpoints'
    results_crc_dir = results_dir / 'crc_select'
    results_posthoc_dir = results_dir / 'posthoc_crc'
    results_vanilla_dir = results_dir / 'vanilla'
    
    for d in [checkpoints_dir, results_crc_dir, results_posthoc_dir, results_vanilla_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # ==================== Phase 1: Training ====================
    if not args.skip_training:
        print("\n" + "=" * 80)
        print("PHASE 1: Training Models")
        print("=" * 80)
        
        for seed in args.seeds:
            print(f"\n>>> Training for seed {seed}")
            
            # 1a. Train CRC-Select
            if not args.skip_crc_select:
                crc_cmd = [
                    'python', 'scripts/train_crc_select.py',
                    '--dataset', args.dataset,
                    '--dataroot', args.dataroot,
                    '--seed', str(seed),
                    '--num_epochs', str(args.num_epochs),
                    '--alpha_risk', str(args.alpha_risk),
                    '--warmup_epochs', str(args.warmup_epochs),
                    '--recalibrate_every', str(args.recalibrate_every),
                    '--coverage', str(args.coverage),
                    '--batch_size', str(args.batch_size),
                ]
                if args.use_dual_update:
                    crc_cmd.append('--use_dual_update')
                if args.unobserve:
                    crc_cmd.append('--unobserve')
                
                run_command(crc_cmd, f"CRC-Select training (seed={seed})")
            
            # 1b. Train vanilla SelectiveNet baseline
            if not args.skip_vanilla:
                vanilla_cmd = [
                    'python', 'scripts/train.py',
                    '--dataset', args.dataset,
                    '--dataroot', args.dataroot,
                    '--num_epochs', str(args.num_epochs),
                    '--coverage', str(args.coverage),
                    '--batch_size', str(args.batch_size),
                ]
                if args.unobserve:
                    vanilla_cmd.append('--unobserve')
                
                run_command(vanilla_cmd, f"Vanilla SelectiveNet training (seed={seed})")
    
    # ==================== Phase 2: Evaluation ====================
    if not args.skip_evaluation:
        print("\n" + "=" * 80)
        print("PHASE 2: Evaluation")
        print("=" * 80)
        
        for seed in args.seeds:
            print(f"\n>>> Evaluating for seed {seed}")
            
            # Find checkpoints (this is simplified - adjust based on actual checkpoint paths)
            # For now, provide manual checkpoint paths via args
            
            # 2a. Evaluate CRC-Select
            if args.crc_select_checkpoint:
                eval_crc_cmd = [
                    'python', 'scripts/eval_crc.py',
                    '--checkpoint', args.crc_select_checkpoint,
                    '--dataset', args.dataset,
                    '--dataroot', args.dataroot,
                    '--seed', str(seed),
                    '--output_dir', str(results_crc_dir),
                    '--batch_size', str(args.batch_size),
                ]
                run_command(eval_crc_cmd, f"CRC-Select evaluation (seed={seed})")
            
            # 2b. Post-hoc CRC on vanilla SelectiveNet
            if args.vanilla_checkpoint:
                posthoc_cmd = [
                    'python', 'scripts/baseline_posthoc_crc.py',
                    '--checkpoint', args.vanilla_checkpoint,
                    '--dataset', args.dataset,
                    '--dataroot', args.dataroot,
                    '--seed', str(seed),
                    '--alpha_risk', str(args.alpha_risk),
                    '--output_dir', str(results_posthoc_dir),
                    '--batch_size', str(args.batch_size),
                ]
                run_command(posthoc_cmd, f"Post-hoc CRC evaluation (seed={seed})")
    
    # ==================== Phase 3: Aggregation ====================
    if not args.skip_aggregation:
        print("\n" + "=" * 80)
        print("PHASE 3: Results Aggregation")
        print("=" * 80)
        
        method_dirs = []
        if (results_crc_dir / f'seed_{args.seeds[0]}').exists():
            method_dirs.append(str(results_crc_dir))
        if (results_posthoc_dir / f'seed_{args.seeds[0]}').exists():
            method_dirs.append(str(results_posthoc_dir))
        
        if method_dirs:
            agg_cmd = [
                'python', 'scripts/aggregate_results.py',
                '--method_dirs'] + method_dirs + [
                '--seeds'] + [str(s) for s in args.seeds] + [
                '--output_dir', str(results_dir / 'aggregated')
            ]
            run_command(agg_cmd, "Results aggregation")
    
    # ==================== Phase 4: Visualization ====================
    if not args.skip_plots:
        print("\n" + "=" * 80)
        print("PHASE 4: Visualization")
        print("=" * 80)
        
        method_dirs = []
        if (results_crc_dir / f'seed_{args.seeds[0]}').exists():
            method_dirs.append(str(results_crc_dir))
        if (results_posthoc_dir / f'seed_{args.seeds[0]}').exists():
            method_dirs.append(str(results_posthoc_dir))
        
        if method_dirs:
            plot_cmd = [
                'python', 'scripts/plot_results.py',
                '--method_dirs'] + method_dirs + [
                '--output_dir', str(results_dir / 'figures')
            ]
            run_command(plot_cmd, "Plotting results")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("EXPERIMENT PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {results_dir}")
    print(f"  - Aggregated results: {results_dir / 'aggregated'}")
    print(f"  - Figures: {results_dir / 'figures'}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # General
    parser.add_argument('--config', type=str, default=None,
                       help='path to YAML config file')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='random seeds for experiments')
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                       help='output directory for all results')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--coverage', type=float, default=0.8)
    parser.add_argument('--alpha_risk', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--recalibrate_every', type=int, default=5)
    parser.add_argument('--use_dual_update', action='store_true')
    parser.add_argument('--unobserve', action='store_true',
                       help='disable Weights & Biases')
    
    # Checkpoints (for evaluation when skipping training)
    parser.add_argument('--crc_select_checkpoint', type=str, default=None)
    parser.add_argument('--vanilla_checkpoint', type=str, default=None)
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='skip training phase')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='skip evaluation phase')
    parser.add_argument('--skip_aggregation', action='store_true',
                       help='skip aggregation phase')
    parser.add_argument('--skip_plots', action='store_true',
                       help='skip plotting phase')
    parser.add_argument('--skip_crc_select', action='store_true',
                       help='skip CRC-Select training')
    parser.add_argument('--skip_vanilla', action='store_true',
                       help='skip vanilla SelectiveNet training')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values (args take precedence)
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if not hasattr(args, subkey) or getattr(args, subkey) is None:
                        setattr(args, subkey, subvalue)
    
    main(args)

