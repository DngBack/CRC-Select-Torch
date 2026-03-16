"""
Deep Gambler baseline (Ziyin et al., NeurIPS 2019).

Adds an extra (K+1)-th class as the "abstention" class.
Selection score: 1 - p(abstain|x).
Post-hoc CRC calibration applied on this score.

NOTE: This baseline requires training a new model with K+1 outputs
and the gambler loss.  If only an evaluation checkpoint is provided
(standard SelectiveNet), this script falls back to using the auxiliary
head output as an approximation.
"""
import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.data import DatasetBuilder
from selectivenet.data_splits import get_split_loaders
from selectivenet.reproducibility import set_seed
from selectivenet.resnet_variant import get_backbone
from crc.calibrate import crc_calibrate_threshold
from crc.risk_utils import (
    compute_risk_scores,
    compute_error_detection_auroc_aupr,
)


# ------------------------------------------------------------------
# Deep Gambler model
# ------------------------------------------------------------------

class DeepGamblerNet(nn.Module):
    """
    Classifier with K+1 output (K classes + 1 abstain).
    """

    def __init__(self, features, feature_dim, num_classes):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(feature_dim, num_classes + 1)
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.features(x)
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        logits_all = self.classifier(feat)
        class_logits = logits_all[:, :self.num_classes]
        abstain_logit = logits_all[:, self.num_classes]
        return class_logits, abstain_logit


class GamblerLoss(nn.Module):
    """
    Deep Gambler loss: L = -log(p_y + o * p_{K+1})
    where o = reward for abstention.
    """

    def __init__(self, reward: float = 2.2):
        super().__init__()
        self.reward = reward

    def forward(self, logits_all, targets):
        # logits_all: (B, K+1)
        probs = F.softmax(logits_all, dim=1)
        K = logits_all.size(1) - 1
        target_probs = probs[torch.arange(len(targets)), targets]
        abstain_probs = probs[:, K]
        loss = -torch.log(target_probs + self.reward * abstain_probs + 1e-8)
        return loss.mean()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_deep_gambler(args):
    """Train a Deep Gambler model from scratch."""
    device = args.device
    dataset_builder = DatasetBuilder(args.dataset, args.dataroot)
    backbone_name = getattr(args, 'backbone', 'vgg16')
    if backbone_name == 'vgg16':
        features = vgg16_variant(32, args.dropout_prob).to(device)
        dim_f = args.dim_features
    else:
        features, dim_f = get_backbone(backbone_name, 32, args.dropout_prob)
        features = features.to(device)

    model = DeepGamblerNet(features, dim_f, dataset_builder.num_classes).to(device)
    criterion = GamblerLoss(reward=args.reward)

    augmentation = getattr(args, 'augmentation', 'original')
    full_train = dataset_builder(train=True, normalize=True, augmentation=augmentation)
    train_loader, cal_loader, test_loader = get_split_loaders(
        full_train, args.dataset, args.seed,
        args.batch_size, args.num_workers
    )

    scheduler_step = getattr(args, 'scheduler_step', 25)
    scheduler_gamma = getattr(args, 'scheduler_gamma', 0.5)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9,
        weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    for epoch in range(args.num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            class_logits, abstain_logit = model(x)
            logits_all = torch.cat([class_logits, abstain_logit.unsqueeze(1)], dim=1)
            loss = criterion(logits_all, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{args.num_epochs}")

    # Save
    _ckpt_dir_arg = getattr(args, 'checkpoint_dir', 'checkpoints/DeepGambler')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    ckpt_dir = _ckpt_dir_arg if os.path.isabs(_ckpt_dir_arg) else os.path.join(workspace_dir, _ckpt_dir_arg)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'seed_{args.seed}.pth')
    torch.save({'state_dict': model.state_dict()}, ckpt_path, _use_new_zipfile_serialization=True)
    print(f"  Checkpoint saved to {ckpt_path}")
    return model, cal_loader, test_loader, dataset_builder


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def collect_gambler_predictions(model, loader, device='cuda'):
    """Collect class logits and confidence = 1 - p(abstain)."""
    all_logits, all_conf, all_targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            class_logits, abstain_logit = model(x)
            logits_all = torch.cat(
                [class_logits, abstain_logit.unsqueeze(1)], dim=1
            )
            probs = F.softmax(logits_all, dim=1)
            conf = 1.0 - probs[:, -1]  # 1 - p(abstain)
            all_logits.append(class_logits.cpu())
            all_conf.append(conf.cpu())
            all_targets.append(y)
    return torch.cat(all_logits), torch.cat(all_conf), torch.cat(all_targets)


def evaluate_at_tau(logits, conf, targets, tau):
    mask = (conf >= tau).float()
    r = compute_risk_scores(logits, targets)
    coverage = mask.mean().item()
    alm = (mask * r).mean().item()
    if mask.sum() > 0:
        risk = (mask * r).sum().item() / mask.sum().item()
        sel_acc = (mask * (logits.argmax(1) == targets).float()).sum().item() / mask.sum().item()
    else:
        risk, sel_acc = 0.0, 0.0
    return {
        'tau': tau,
        'accepted_loss_mass': alm,
        'selective_risk': risk,
        'coverage': coverage,
        'selective_acc': sel_acc,
        'abstention_rate': 1.0 - coverage,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    set_seed(args.seed)

    print("=" * 80)
    print("Deep Gambler Baseline")
    print("=" * 80)

    device = args.device

    if args.train:
        print("Training Deep Gambler model...")
        model, cal_loader, test_loader, dataset_builder = train_deep_gambler(args)
    else:
        # Load pre-trained gambler checkpoint
        dataset_builder = DatasetBuilder(args.dataset, args.dataroot)
        backbone_name = getattr(args, 'backbone', 'vgg16')
        if backbone_name == 'vgg16':
            features = vgg16_variant(32, args.dropout_prob).to(device)
            dim_f = args.dim_features
        else:
            features, dim_f = get_backbone(backbone_name, 32, args.dropout_prob)
            features = features.to(device)
        model = DeepGamblerNet(
            features, dim_f, dataset_builder.num_classes
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        _aug = getattr(args, 'augmentation', 'original')
        full_train = dataset_builder(train=True, normalize=True, augmentation=_aug)
        _, cal_loader, test_loader = get_split_loaders(
            full_train, args.dataset, args.seed,
            args.batch_size, args.num_workers
        )

    model.eval()
    cal_logits, cal_conf, cal_targets = collect_gambler_predictions(model, cal_loader, device=device)
    test_logits, test_conf, test_targets = collect_gambler_predictions(model, test_loader, device=device)

    alpha_values = args.alpha_values or [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    all_rows = []
    for alpha in alpha_values:
        crc = crc_calibrate_threshold(cal_logits, cal_conf, cal_targets, alpha)
        tau_hat = crc['tau_hat']
        m = evaluate_at_tau(test_logits, test_conf, test_targets, tau_hat)
        violation_gap = max(m['accepted_loss_mass'] - alpha, 0.0)
        _auroc_aupr = compute_error_detection_auroc_aupr(
            test_logits, test_conf, test_targets
        )
        auroc, aupr = _auroc_aupr['auroc'], _auroc_aupr['aupr']
        row = {
            'method': 'DeepGambler',
            'dataset': args.dataset,
            'seed': args.seed,
            'alpha': alpha,
            'tau_hat': tau_hat,
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
              f"cov={m['coverage']:.4f}")

    taus = np.linspace(0.0, 1.0, 201)
    rc_rows = [evaluate_at_tau(test_logits, test_conf, test_targets, t) for t in taus]

    out_dir = os.path.join(args.output_dir, 'DeepGambler', f'seed_{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'all_metrics.csv'), index=False)
    pd.DataFrame(rc_rows).to_csv(os.path.join(out_dir, 'risk_coverage_curve.csv'), index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--train', action='store_true',
                       help='train from scratch instead of loading checkpoint')
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'resnet18', 'wrn28_10'])
    parser.add_argument('--reward', type=float, default=2.2,
                       help='abstention reward for gambler loss')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('--alpha_values', type=float, nargs='+', default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='./results_paper')
    parser.add_argument('--augmentation', type=str, default='original',
                       help='data augmentation: original, randcrop, tf, lili')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/DeepGambler',
                       help='directory to save trained checkpoint')
    parser.add_argument('--scheduler_step', type=int, default=25)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)
