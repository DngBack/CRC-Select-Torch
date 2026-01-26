"""
Risk computation utilities for CRC-Select.

Implements bounded risk r(x,y) = 1 - p_theta(y|x) for use in CRC.
This is a monotone loss in [0,1] suitable for conformal risk control.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def compute_risk_scores(
    logits: torch.Tensor, 
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute risk scores r = 1 - p_theta(y|x).
    
    This is the recommended bounded risk for classification.
    Lower risk means higher confidence in the correct prediction.
    
    Args:
        logits: Model output logits (B, num_classes)
        targets: Ground truth labels (B,)
    
    Returns:
        risk: Risk scores in [0, 1] (B,)
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # Get probability of true class
    batch_size = len(targets)
    p_y = probs[torch.arange(batch_size, device=logits.device), targets]
    
    # Risk = 1 - p(y|x)
    risk = 1.0 - p_y
    
    return risk


def compute_acceptance_mask(
    selection_scores: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Compute binary acceptance mask from selection scores.
    
    Args:
        selection_scores: Selector output g(x) (B,) or (B, 1)
        threshold: Acceptance threshold tau
    
    Returns:
        mask: Binary mask (B,) with 1=accept, 0=reject
    """
    g = selection_scores.squeeze()
    return (g >= threshold).float()


def compute_selective_risk(
    logits: torch.Tensor,
    selection_scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
    hard: bool = True
) -> torch.Tensor:
    """
    Compute selective risk: R_hat = sum(g * r) / sum(g).
    
    This is the empirical risk over the accepted region.
    
    Args:
        logits: Model output logits (B, num_classes)
        selection_scores: Selector output g(x) (B,) or (B, 1)
        targets: Ground truth labels (B,)
        threshold: Acceptance threshold tau (used if hard=True)
        eps: Small constant to prevent division by zero
        hard: If True, use hard thresholding; if False, use soft weighting
    
    Returns:
        selective_risk: Scalar risk value
    """
    # Compute risk scores
    r = compute_risk_scores(logits, targets)
    
    # Get selection weights
    g = selection_scores.squeeze()
    
    if hard:
        # Hard thresholding: binary accept/reject
        g_weight = (g >= threshold).float()
    else:
        # Soft weighting: use continuous g values
        g_weight = g
    
    # Compute weighted risk
    numerator = (g_weight * r).sum()
    denominator = g_weight.sum() + eps
    
    selective_risk = numerator / denominator
    
    return selective_risk


def compute_coverage(
    selection_scores: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Compute empirical coverage: fraction of samples accepted.
    
    Args:
        selection_scores: Selector output g(x) (B,) or (B, 1)
        threshold: Acceptance threshold tau
    
    Returns:
        coverage: Scalar coverage value in [0, 1]
    """
    g = selection_scores.squeeze()
    acceptance_mask = (g >= threshold).float()
    coverage = acceptance_mask.mean()
    return coverage


def compute_selective_accuracy(
    logits: torch.Tensor,
    selection_scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute selective accuracy: accuracy on accepted samples.
    
    Args:
        logits: Model output logits (B, num_classes)
        selection_scores: Selector output g(x) (B,) or (B, 1)
        targets: Ground truth labels (B,)
        threshold: Acceptance threshold tau
        eps: Small constant to prevent division by zero
    
    Returns:
        selective_acc: Accuracy on accepted samples
        coverage: Fraction of accepted samples
    """
    # Get predictions
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).float()
    
    # Get acceptance mask
    g = selection_scores.squeeze()
    acceptance_mask = (g >= threshold).float()
    
    # Compute selective accuracy
    num_correct_accepted = (acceptance_mask * correct).sum()
    num_accepted = acceptance_mask.sum() + eps
    
    selective_acc = num_correct_accepted / num_accepted
    coverage = acceptance_mask.mean()
    
    return selective_acc, coverage


def compute_risk_coverage_curve(
    logits_list: list,
    selection_scores_list: list,
    targets_list: list,
    thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve by sweeping threshold values.
    
    Args:
        logits_list: List of logit tensors from batches
        selection_scores_list: List of selection score tensors
        targets_list: List of target tensors
        thresholds: Array of threshold values to evaluate
    
    Returns:
        risks: Array of risk values (one per threshold)
        coverages: Array of coverage values (one per threshold)
    """
    # Concatenate all batches
    all_logits = torch.cat(logits_list, dim=0)
    all_selection_scores = torch.cat(selection_scores_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    risks = []
    coverages = []
    
    for threshold in thresholds:
        risk = compute_selective_risk(
            all_logits, all_selection_scores, all_targets,
            threshold=threshold, hard=True
        )
        coverage = compute_coverage(all_selection_scores, threshold)
        
        risks.append(risk.item())
        coverages.append(coverage.item())
    
    return np.array(risks), np.array(coverages)


if __name__ == '__main__':
    # Test risk computation utilities
    print("Testing risk computation utilities...")
    
    # Create dummy data
    batch_size = 100
    num_classes = 10
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    selection_scores = torch.sigmoid(torch.randn(batch_size, 1))
    
    # Test risk scores
    risk = compute_risk_scores(logits, targets)
    print(f"Risk shape: {risk.shape}")
    print(f"Risk range: [{risk.min():.3f}, {risk.max():.3f}]")
    print(f"Mean risk: {risk.mean():.3f}")
    
    # Test selective risk
    sel_risk_hard = compute_selective_risk(
        logits, selection_scores, targets, threshold=0.5, hard=True
    )
    sel_risk_soft = compute_selective_risk(
        logits, selection_scores, targets, threshold=0.5, hard=False
    )
    print(f"\nSelective risk (hard): {sel_risk_hard:.3f}")
    print(f"Selective risk (soft): {sel_risk_soft:.3f}")
    
    # Test coverage
    coverage = compute_coverage(selection_scores, threshold=0.5)
    print(f"\nCoverage at tau=0.5: {coverage:.3f}")
    
    # Test selective accuracy
    sel_acc, cov = compute_selective_accuracy(
        logits, selection_scores, targets, threshold=0.5
    )
    print(f"\nSelective accuracy: {sel_acc:.3f}")
    print(f"Coverage: {cov:.3f}")
    
    print("\n✓ All tests passed!")

