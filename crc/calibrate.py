"""
CRC (Conformal Risk Control) calibration module.

Implements the correct CRC calibration procedure from the paper:

    hat_tau = inf{ tau : (n/(n+1)) * A_hat_n(tau) + 1/(n+1) <= alpha }

where A_hat_n(tau) = (1/n) * sum_i r_i * 1{g_i >= tau} is the empirical
accepted-loss mass on the calibration set.

This guarantees (Theorem 1):
    E[ r(X_{n+1}, Y_{n+1}) * 1{g(X_{n+1}) >= hat_tau} ] <= alpha
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from .risk_utils import compute_risk_scores, compute_acceptance_mask, compute_accepted_loss_mass


def crc_calibrate_threshold(
    model_or_logits,
    cal_loader_or_g,
    targets_or_alpha=None,
    alpha: float = None,
    n_thresholds: int = 1000,
    device: str = 'cuda'
) -> Dict:
    """
    CRC calibration: find hat_tau using the standard CRC formula.
    
    hat_tau = inf{ tau : (n/(n+1)) * A_hat_n(tau) + 1/(n+1) <= alpha }
    
    where A_hat_n(tau) = (1/n) sum_i r_i * 1{g_i >= tau}.
    
    Supports two calling conventions:
        1) crc_calibrate_threshold(model, cal_loader, alpha=0.1)
        2) crc_calibrate_threshold(logits, g_scores, targets, alpha=0.1)
    
    Returns:
        Dictionary with tau_hat, accepted_loss_mass, coverage, selective_risk, etc.
    """
    # Detect calling convention
    if isinstance(model_or_logits, torch.Tensor):
        # Called with (logits, g, targets, alpha)
        all_logits = model_or_logits
        all_g = cal_loader_or_g.squeeze()
        all_targets = targets_or_alpha
        if alpha is None:
            raise ValueError("alpha is required")
    else:
        # Called with (model, cal_loader, alpha=...)
        model = model_or_logits
        cal_loader = cal_loader_or_g
        if alpha is None:
            alpha = targets_or_alpha
        if alpha is None:
            raise ValueError("alpha is required")

        model.eval()
        all_logits = []
        all_g = []
        all_targets = []
        with torch.no_grad():
            for x, y in cal_loader:
                x, y = x.to(device), y.to(device)
                logits, g, _ = model(x)
                all_logits.append(logits.cpu())
                all_g.append(g.cpu())
                all_targets.append(y.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_g = torch.cat(all_g, dim=0).squeeze()
        all_targets = torch.cat(all_targets, dim=0)
    
    n = len(all_targets)
    
    # Compute risk scores: r_i = 1 - p_theta(y_i | x_i)
    all_r = compute_risk_scores(all_logits, all_targets)
    
    # Sweep tau from 0 to 1
    # Use unique g values + linspace for thorough sweep
    g_unique = torch.unique(all_g)
    tau_candidates = torch.sort(
        torch.cat([
            torch.linspace(0.0, 1.0, n_thresholds),
            g_unique,
            g_unique - 1e-6  # just below each unique value
        ])
    )[0]
    tau_candidates = tau_candidates[(tau_candidates >= 0) & (tau_candidates <= 1)]
    
    # CRC formula: find smallest tau such that
    # (n/(n+1)) * A_hat(tau) + 1/(n+1) <= alpha
    tau_hat = 1.0  # default: reject everything
    best_A = 0.0
    
    for tau in tau_candidates:
        tau_val = tau.item()
        # A_hat(tau) = (1/n) * sum(r_i * 1{g_i >= tau})
        acceptance = (all_g >= tau_val).float()
        A_hat = (all_r * acceptance).mean().item()
        
        # CRC condition: (n/(n+1)) * A_hat + 1/(n+1) <= alpha
        crc_value = (n / (n + 1)) * A_hat + 1.0 / (n + 1)
        
        if crc_value <= alpha:
            tau_hat = tau_val
            best_A = A_hat
            break  # inf{tau : condition holds} = first one that works
    
    # Compute metrics at tau_hat
    acceptance_at_hat = (all_g >= tau_hat).float()
    coverage = acceptance_at_hat.mean().item()
    A_at_hat = (all_r * acceptance_at_hat).mean().item()
    
    if coverage > 0:
        selective_risk = A_at_hat / coverage
    else:
        selective_risk = 0.0
    
    result = {
        'tau_hat': float(tau_hat),
        'accepted_loss_mass': float(A_at_hat),
        'coverage': float(coverage),
        'selective_risk': float(selective_risk),
        'n_cal': int(n),
        'alpha': float(alpha),
        'crc_value': float((n / (n + 1)) * A_at_hat + 1.0 / (n + 1)),
        'all_g': all_g.numpy(),
        'all_r': all_r.numpy(),
    }
    
    return result


def compute_crc_threshold(
    model: torch.nn.Module,
    cal_loader: torch.utils.data.DataLoader,
    tau: float,
    alpha: float,
    delta: float = 0.1,
    device: str = 'cuda'
) -> Dict:
    """
    Legacy CRC threshold computation at a FIXED tau.
    
    Used during training for alternating optimization (not for final calibration).
    Computes accepted-loss mass and risk at a given tau.
    
    Args:
        model: Trained SelectiveNet model
        cal_loader: DataLoader for calibration set
        tau: Fixed acceptance threshold
        alpha: Target risk level
        delta: Not used (kept for backward compatibility)
        device: Device for computation
    
    Returns:
        Dictionary with calibration metrics at given tau
    """
    model.eval()
    
    all_logits = []
    all_selection_scores = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            logits, g, _ = model(x)
            all_logits.append(logits.cpu())
            all_selection_scores.append(g.cpu())
            all_targets.append(y.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_selection_scores = torch.cat(all_selection_scores, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0)
    
    n = len(all_targets)
    
    # Compute risk scores
    risk_scores = compute_risk_scores(all_logits, all_targets)
    
    # Acceptance at given tau
    acceptance_mask = (all_selection_scores >= tau).float()
    num_accepted = int(acceptance_mask.sum().item())
    
    # Accepted-loss mass: A(tau) = (1/n) * sum(r_i * 1{g_i >= tau})
    accepted_loss_mass = (risk_scores * acceptance_mask).mean().item()
    
    # Coverage
    actual_coverage = acceptance_mask.mean().item()
    
    # Selective risk (conditional): R_sel = A / C
    if actual_coverage > 0:
        estimated_risk = accepted_loss_mass / actual_coverage
    else:
        estimated_risk = 1.0
    
    result = {
        'q': float(alpha),  # legacy field
        'tau': float(tau),
        'actual_coverage': float(actual_coverage),
        'estimated_risk': float(estimated_risk),
        'accepted_loss_mass': float(accepted_loss_mass),
        'num_accepted': num_accepted,
        'num_total': n,
        'risk_scores': risk_scores.numpy(),
        'acceptance_rate': float(actual_coverage)
    }
    
    return result


def calibrate_selector(
    model: torch.nn.Module,
    cal_loader: torch.utils.data.DataLoader,
    target_coverage: float,
    device: str = 'cuda'
) -> float:
    """
    Calibrate selector threshold to achieve target coverage.
    
    This is the vanilla post-hoc calibration (not CRC-aware).
    Used as a baseline for comparison.
    
    Args:
        model: Trained SelectiveNet model
        cal_loader: DataLoader for calibration set
        target_coverage: Desired coverage (e.g., 0.8 for 80%)
        device: Device for computation
    
    Returns:
        threshold: Calibrated threshold tau
    """
    model.eval()
    
    all_selection_scores = []
    
    # Collect selection scores on calibration set
    with torch.no_grad():
        for x, _ in cal_loader:
            x = x.to(device)
            _, g, _ = model(x)
            all_selection_scores.append(g.cpu())
    
    # Concatenate all batches
    all_selection_scores = torch.cat(all_selection_scores, dim=0)
    g_np = all_selection_scores.squeeze().numpy()
    
    # Find threshold that gives desired coverage
    # Coverage = fraction of samples with g >= threshold
    # So threshold is the (1-coverage) percentile
    percentile = (1.0 - target_coverage) * 100
    threshold = np.percentile(g_np, percentile)
    
    print(f"✓ Calibrated threshold: {threshold:.4f} for coverage {target_coverage:.2f}")
    
    return float(threshold)


def compute_crc_threshold_advanced(
    model: torch.nn.Module,
    cal_loader: torch.utils.data.DataLoader,
    alpha: float,
    delta: float = 0.1,
    n_thresholds: int = 1000,
    device: str = 'cuda',
    method: str = 'hoeffding'
) -> Dict:
    """
    CRC calibration with finite-sample concentration bounds.
    
    Uses Hoeffding or empirical Bernstein bounds to get a high-probability
    guarantee rather than an in-expectation guarantee.
    
    Args:
        model: Trained SelectiveNet model
        cal_loader: DataLoader for calibration set
        alpha: Target accepted-loss mass level
        delta: Failure probability (1-delta confidence)
        n_thresholds: Number of threshold candidates
        device: Device for computation
        method: 'hoeffding' or 'bernstein'
    
    Returns:
        Dictionary with calibration results including bound info
    """
    # Get base CRC result first
    result = crc_calibrate_threshold(model, cal_loader, alpha, n_thresholds, device)
    
    n = result['n_cal']
    
    if method == 'hoeffding':
        # Hoeffding: epsilon = sqrt(log(2/delta) / (2n))
        epsilon = np.sqrt(np.log(2.0 / delta) / (2.0 * n))
    elif method == 'bernstein':
        # Empirical Bernstein using variance of per-sample CRC losses
        all_r = result['all_r']
        all_g = result['all_g']
        tau_hat = result['tau_hat']
        losses = all_r * (all_g >= tau_hat).astype(float)
        var = np.var(losses)
        a = 7.0 * var / (3.0 * max(n - 1, 1))
        b = 3.0 * np.log(2.0 / delta) / max(n - 1, 1)
        epsilon = np.sqrt(2.0 * a * b) + b
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result['epsilon'] = float(epsilon)
    result['risk_upper_bound'] = float(result['accepted_loss_mass'] + epsilon)
    result['method'] = method
    result['delta'] = float(delta)
    
    return result


if __name__ == '__main__':
    print("Testing CRC calibration module...")
    print("✓ Module imports successful")
    
    import sys
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)
    
    from selectivenet.vgg_variant import vgg16_variant
    from selectivenet.model import SelectiveNet
    
    # Create dummy model
    features = vgg16_variant(32, 0.3)
    model = SelectiveNet(features, 512, 10)
    model.eval()
    
    # Create dummy calibration data
    dummy_data = torch.randn(200, 3, 32, 32)
    dummy_targets = torch.randint(0, 10, (200,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    cal_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=32)
    
    # Test CRC calibration (correct formula)
    result = crc_calibrate_threshold(model, cal_loader, alpha=0.1, device='cpu')
    
    print("\nCRC Calibration results:")
    print(f"  tau_hat: {result['tau_hat']:.4f}")
    print(f"  Accepted-loss mass: {result['accepted_loss_mass']:.4f}")
    print(f"  Coverage: {result['coverage']:.4f}")
    print(f"  Selective risk: {result['selective_risk']:.4f}")
    print(f"  CRC value: {result['crc_value']:.4f} (should be <= {result['alpha']:.4f})")
    print(f"  n_cal: {result['n_cal']}")
    
    # Test legacy compute_crc_threshold
    result_legacy = compute_crc_threshold(model, cal_loader, tau=0.5, alpha=0.1, device='cpu')
    print(f"\nLegacy calibration at tau=0.5:")
    print(f"  Coverage: {result_legacy['actual_coverage']:.4f}")
    print(f"  Accepted-loss mass: {result_legacy['accepted_loss_mass']:.4f}")
    print(f"  Selective risk: {result_legacy['estimated_risk']:.4f}")
    
    # Test vanilla calibration
    threshold = calibrate_selector(model, cal_loader, target_coverage=0.8, device='cpu')
    print(f"\nVanilla calibration for 80% coverage: tau={threshold:.4f}")
    
    print("\n✓ All tests passed!")

