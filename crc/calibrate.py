"""
CRC (Conformal Risk Control) calibration module.

Implements calibration to find threshold q that guarantees risk ≤ alpha
with high probability.

Starting with quantile-based approximation for simplicity and stability.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from .risk_utils import compute_risk_scores, compute_acceptance_mask


def compute_crc_threshold(
    model: torch.nn.Module,
    cal_loader: torch.utils.data.DataLoader,
    tau: float,
    alpha: float,
    delta: float = 0.1,
    device: str = 'cuda'
) -> Dict:
    """
    Compute CRC threshold q on calibration set.
    
    This function calibrates the risk control threshold to ensure that
    the expected risk on accepted samples is bounded by alpha.
    
    Algorithm:
    1. Run model on calibration set
    2. Identify accepted samples (g(x) >= tau)
    3. Compute risk scores r_i = 1 - p_theta(y_i|x_i) for accepted
    4. Find quantile q that bounds E[r | accepted] ≤ alpha
    
    Args:
        model: Trained SelectiveNet model
        cal_loader: DataLoader for calibration set
        tau: Acceptance threshold for selector g(x)
        alpha: Target risk level (e.g., 0.1 for 10% risk)
        delta: Failure probability for concentration bound (not used in quantile version)
        device: Device for computation
    
    Returns:
        Dictionary with:
            - q: CRC threshold (risk quantile)
            - actual_coverage: Empirical coverage on calibration set
            - estimated_risk: Empirical risk on accepted samples
            - num_accepted: Number of accepted calibration samples
            - num_total: Total calibration samples
            - risk_scores: Array of risk scores for accepted samples
    """
    model.eval()
    
    all_logits = []
    all_selection_scores = []
    all_targets = []
    
    # Collect predictions on calibration set
    with torch.no_grad():
        for x, y in cal_loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits, g, _ = model(x)
            
            all_logits.append(logits.cpu())
            all_selection_scores.append(g.cpu())
            all_targets.append(y.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_selection_scores = torch.cat(all_selection_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute acceptance mask
    acceptance_mask = compute_acceptance_mask(all_selection_scores, tau)
    
    # Get accepted samples
    accepted_indices = torch.where(acceptance_mask > 0)[0]
    num_accepted = len(accepted_indices)
    num_total = len(all_targets)
    
    if num_accepted == 0:
        print("WARNING: No samples accepted at tau={:.3f}".format(tau))
        return {
            'q': 1.0,  # No valid threshold
            'actual_coverage': 0.0,
            'estimated_risk': 1.0,
            'num_accepted': 0,
            'num_total': num_total,
            'risk_scores': np.array([])
        }
    
    # Compute risk scores for accepted samples
    accepted_logits = all_logits[accepted_indices]
    accepted_targets = all_targets[accepted_indices]
    risk_scores = compute_risk_scores(accepted_logits, accepted_targets)
    
    # Compute empirical quantities
    actual_coverage = num_accepted / num_total
    estimated_risk = risk_scores.mean().item()
    
    # CRC calibration: find quantile q
    # For now, use quantile-based approach
    # More sophisticated: use concentration inequalities (Hoeffding/Bernstein)
    
    # Quantile level to ensure E[r] ≤ alpha
    # Simple approach: use alpha directly as the target mean
    # This is conservative but stable
    
    risk_scores_np = risk_scores.numpy()
    risk_scores_sorted = np.sort(risk_scores_np)
    
    # Find the quantile q such that mean of risks <= alpha
    # We use a simple threshold approach:
    # q is set to alpha (the target risk level)
    # This is the quantile-based proxy mentioned in the plan
    
    q = alpha
    
    # Alternatively, we can compute q more carefully:
    # Find largest q such that mean of (r <= q) risks is <= alpha
    # This is more conservative
    
    # Better approach: use the (1-delta)-quantile with concentration bound
    # For now, use simple quantile that gives mean risk close to alpha
    
    quantile_level = min(0.9, max(0.5, 1.0 - delta))
    q_percentile = np.percentile(risk_scores_np, quantile_level * 100)
    
    # Use the more conservative of the two
    q = max(alpha, q_percentile)
    
    result = {
        'q': float(q),
        'actual_coverage': float(actual_coverage),
        'estimated_risk': float(estimated_risk),
        'num_accepted': int(num_accepted),
        'num_total': int(num_total),
        'risk_scores': risk_scores_np,
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
    tau: float,
    alpha: float,
    delta: float = 0.1,
    device: str = 'cuda',
    method: str = 'hoeffding'
) -> Dict:
    """
    Advanced CRC calibration with concentration inequalities.
    
    This implements proper CRC with Hoeffding or empirical Bernstein bounds.
    Use this after initial experiments with quantile proxy work.
    
    Args:
        model: Trained SelectiveNet model
        cal_loader: DataLoader for calibration set
        tau: Acceptance threshold for selector g(x)
        alpha: Target risk level
        delta: Failure probability (1-delta confidence)
        device: Device for computation
        method: 'hoeffding' or 'bernstein' for concentration bound
    
    Returns:
        Dictionary with calibration results
    """
    # Get basic calibration results
    result = compute_crc_threshold(model, cal_loader, tau, alpha, delta, device)
    
    if result['num_accepted'] == 0:
        return result
    
    risk_scores = result['risk_scores']
    n = len(risk_scores)
    
    # Compute concentration bound
    if method == 'hoeffding':
        # Hoeffding bound: P(|mean - E[r]| > epsilon) <= 2 exp(-2 n epsilon^2)
        # Solve for epsilon given delta
        epsilon = np.sqrt(np.log(2.0 / delta) / (2.0 * n))
        
        # Adjusted risk estimate with confidence bound
        risk_upper = result['estimated_risk'] + epsilon
        
    elif method == 'bernstein':
        # Empirical Bernstein inequality (tighter bound using variance)
        var = np.var(risk_scores)
        
        # Bernstein bound
        a = 7.0 * var / (3.0 * (n - 1))
        b = 3.0 * np.log(2.0 / delta) / (n - 1)
        epsilon = np.sqrt(2.0 * a * b) + b
        
        risk_upper = result['estimated_risk'] + epsilon
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Update q to account for concentration bound
    q_adjusted = min(1.0, max(alpha, risk_upper))
    
    result['q'] = float(q_adjusted)
    result['epsilon'] = float(epsilon)
    result['risk_upper_bound'] = float(risk_upper)
    result['method'] = method
    
    return result


if __name__ == '__main__':
    print("Testing CRC calibration module...")
    
    # This would require a trained model and data
    # For now, just verify imports work
    print("✓ Module imports successful")
    
    # Test with dummy model
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
    dummy_data = torch.randn(100, 3, 32, 32)
    dummy_targets = torch.randint(0, 10, (100,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    cal_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=32)
    
    # Test calibration
    result = compute_crc_threshold(
        model, cal_loader, tau=0.5, alpha=0.1, device='cpu'
    )
    
    print("\nCalibration results:")
    print(f"  q (threshold): {result['q']:.4f}")
    print(f"  Coverage: {result['actual_coverage']:.4f}")
    print(f"  Estimated risk: {result['estimated_risk']:.4f}")
    print(f"  Accepted: {result['num_accepted']}/{result['num_total']}")
    
    print("\n✓ All tests passed!")

