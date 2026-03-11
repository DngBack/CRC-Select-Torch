"""
MSP (Maximum Softmax Probability) baseline.

Selection rule: accept if max_c p(y=c|x) >= tau.
Post-hoc CRC calibration applied the same way as other methods.
"""
import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.data import DatasetBuilder
from selectivenet.data_splits import get_split_loaders
from selectivenet.reproducibility import set_seed
from selectivenet.resnet_variant import get_backbone
from crc.calibrate import crc_calibrate_threshold
from crc.risk_utils import (
    compute_risk_scores,
    compute_accepted_loss_mass,
    compute_selective_risk,
    compute_coverage,
    compute_selective_accuracy,
    compute_error_detection_auroc_aupr,
)


def load_model(checkpoint_path, args):
    """Load pre-trained model (any architecture)."""
    dataset_builder = DatasetBuilder(args.dataset, args.dataroot)
    backbone = getattr(args, 'backbone', 'vgg16')
    if backbone == 'vgg16':
        features = vgg16_variant(32, args.dropout_prob).cuda()
        dim_features = args.dim_features
    else:
        features, dim_features = get_backbone(backbone, 32, args.dropout_prob)
        features = features.cuda()
    model = SelectiveNet(
        features, dim_features, dataset_builder.num_classes, div_by_ten=False
    ).cuda()

    ckpt = torch.load(checkpoint_path, weights_only=False)
    if isinstance(ckpt, list):
        ckpt = ckpt[0]
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, dataset_builder


def collect_msp_predictions(model, loader, device='cuda'):
    """Collect logits, MSP confidence scores, and targets."""
    all_logits, all_conf, all_targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _, _ = model(x)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values  # MSP score
            all_logits.append(logits.cpu())
            all_conf.append(conf.cpu())
            all_targets.append(y)
    return torch.cat(all_logits), torch.cat(all_conf), torch.cat(all_targets)


def evaluate_at_tau(logits, conf, targets, tau):
    """Evaluate accepted-loss mass, risk, coverage, accuracy at threshold tau."""
    mask = (conf >= tau).float()
    n = float(len(targets))
    r = compute_risk_scores(logits, targets)

    coverage = mask.mean().item()
    alm = (mask * r).mean().item()

    if mask.sum() > 0:
        risk = (mask * r).sum().item() / mask.sum().item()
        preds = logits.argmax(dim=1)
        sel_acc = (mask * (preds == targets).float()).sum().item() / mask.sum().item()
    else:
        risk = 0.0
        sel_acc = 0.0

    return {
        'tau': tau,
        'accepted_loss_mass': alm,
        'selective_risk': risk,
        'coverage': coverage,
        'selective_acc': sel_acc,
        'abstention_rate': 1.0 - coverage,
    }


def main(args):
    set_seed(args.seed)

    print("=" * 80)
    print("MSP Baseline Evaluation")
    print("=" * 80)

    model, dataset_builder = load_model(args.checkpoint, args)

    full_train_dataset = dataset_builder(
        train=True, normalize=True, augmentation='original'
    )
    _, cal_loader, test_loader = get_split_loaders(
        full_train_dataset, args.dataset, args.seed,
        args.batch_size, args.num_workers
    )

    # Collect MSP-based predictions
    cal_logits, cal_conf, cal_targets = collect_msp_predictions(model, cal_loader)
    test_logits, test_conf, test_targets = collect_msp_predictions(model, test_loader)

    alpha_values = args.alpha_values or [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    all_rows = []
    for alpha in alpha_values:
        # CRC calibration on cal set using MSP as selection score
        crc = crc_calibrate_threshold(cal_logits, cal_conf, cal_targets, alpha)
        tau_hat = crc['tau_hat']

        # Evaluate on test set
        m = evaluate_at_tau(test_logits, test_conf, test_targets, tau_hat)
        violation_gap = max(m['accepted_loss_mass'] - alpha, 0.0)

        # AUROC / AUPR
        _auroc_aupr = compute_error_detection_auroc_aupr(
            test_logits, test_conf, test_targets
        )
        auroc, aupr = _auroc_aupr['auroc'], _auroc_aupr['aupr']

        row = {
            'method': 'MSP',
            'dataset': args.dataset,
            'seed': args.seed,
            'alpha': alpha,
            'tau_hat': tau_hat,
            'cal_accepted_loss_mass': crc['accepted_loss_mass'],
            'test_accepted_loss_mass': m['accepted_loss_mass'],
            'test_selective_risk': m['selective_risk'],
            'test_coverage': m['coverage'],
            'test_selective_acc': m['selective_acc'],
            'violation_gap': violation_gap,
            'violated': m['accepted_loss_mass'] > alpha,
            'auroc': auroc,
            'aupr': aupr,
        }
        all_rows.append(row)
        print(f"  alpha={alpha:.3f}: A_hat={m['accepted_loss_mass']:.4f}, "
              f"cov={m['coverage']:.4f}, tau_hat={tau_hat:.4f}")

    # RC curve using MSP scores
    taus = np.linspace(0.0, 1.0, 201)
    rc_rows = [evaluate_at_tau(test_logits, test_conf, test_targets, t)
               for t in taus]
    rc_df = pd.DataFrame(rc_rows)

    # Save
    out_dir = os.path.join(args.output_dir, 'MSP', f'seed_{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'all_metrics.csv'), index=False)
    rc_df.to_csv(os.path.join(out_dir, 'risk_coverage_curve.csv'), index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'resnet18', 'wrn28_10'])
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('--alpha_values', type=float, nargs='+', default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='./results_paper')
    args = parser.parse_args()
    main(args)
