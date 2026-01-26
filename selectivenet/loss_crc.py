"""
CRC-aware loss function for CRC-Select training.

Extends SelectiveLoss with CRC-coupled risk penalty to train selector
that works well with CRC calibration.
"""
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torch.nn.functional as F
from crc.risk_utils import compute_risk_scores, compute_selective_risk


class CRCSelectiveLoss(torch.nn.Module):
    """
    CRC-aware selective loss for joint training of predictor and selector.
    
    Loss = L_pred + beta * L_cov + mu * L_risk
    
    where:
    - L_pred: Selective prediction loss (weighted CE)
    - L_cov: Coverage regularizer (ensures selector doesn't collapse)
    - L_risk: CRC-coupled risk penalty (encourages low risk on accepted)
    """
    
    def __init__(
        self, 
        loss_func: torch.nn.Module,
        coverage: float,
        alpha_risk: float,
        lm: float = 32.0,
        alpha: float = 0.5,
        mu: float = 1.0
    ):
        """
        Args:
            loss_func: Base loss function (e.g., CrossEntropyLoss(reduction='none'))
            coverage: Target coverage for selector
            alpha_risk: Target risk level for CRC (not to be confused with alpha mix weight)
            lm: Lagrange multiplier for coverage constraint (default: 32)
            alpha: Mix weight between selective loss and auxiliary loss (default: 0.5)
            mu: Lagrange multiplier for risk penalty (default: 1.0)
        """
        super(CRCSelectiveLoss, self).__init__()
        
        assert 0.0 < coverage <= 1.0, "Coverage must be in (0, 1]"
        assert 0.0 < alpha_risk <= 1.0, "Alpha risk must be in (0, 1]"
        assert 0.0 < lm, "Lagrange multiplier lm must be positive"
        assert 0.0 < alpha <= 1.0, "Mix weight alpha must be in (0, 1]"
        assert mu >= 0.0, "Risk penalty mu must be non-negative"
        
        self.loss_func = loss_func
        self.coverage = coverage
        self.alpha_risk = alpha_risk
        self.lm = lm
        self.alpha = alpha
        self.mu = mu
        self.eps = 1e-8
    
    def forward(
        self,
        prediction_out: torch.Tensor,
        selection_out: torch.Tensor,
        auxiliary_out: torch.Tensor,
        target: torch.Tensor,
        alpha_risk: float = None,
        threshold: float = 0.5,
        mode: str = 'train'
    ):
        """
        Compute CRC-aware selective loss.
        
        Args:
            prediction_out: Predictor logits (B, num_classes)
            selection_out: Selector output g(x) (B, 1)
            auxiliary_out: Auxiliary classifier logits (B, num_classes)
            target: Ground truth labels (B,)
            alpha_risk: Override target risk level (if None, use self.alpha_risk)
            threshold: Threshold for hard acceptance (used in evaluation)
            mode: 'train', 'validation', or 'test'
        
        Returns:
            Dictionary with loss components and metrics
        """
        if alpha_risk is None:
            alpha_risk = self.alpha_risk
        
        g = selection_out.squeeze()
        
        # ================== Component 1: Selective Prediction Loss ==================
        # L_pred = sum(g * CE) / sum(g)
        ce_losses = self.loss_func(prediction_out, target)
        empirical_risk = (ce_losses * g).mean()
        empirical_coverage = g.mean()
        
        L_pred = empirical_risk / (empirical_coverage + self.eps)
        
        # ================== Component 2: Coverage Regularizer ==================
        # L_cov = max(0, coverage - empirical_coverage)^2
        coverage_tensor = torch.tensor(
            [self.coverage], dtype=torch.float32, 
            requires_grad=True, device=g.device
        )
        penalty = torch.max(
            coverage_tensor - empirical_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=g.device)
        ) ** 2
        L_cov = penalty
        
        # ================== Component 3: CRC-Coupled Risk Penalty ==================
        # L_risk = max(0, R_hat - alpha_risk)
        # where R_hat = sum(g * r) / sum(g) with r = 1 - p(y|x)
        
        if mode == 'train' and self.mu > 0:
            # Compute risk scores
            r = compute_risk_scores(prediction_out, target)
            
            # Compute selective risk (soft weighting)
            R_hat = (g * r).sum() / (g.sum() + self.eps)
            
            # Risk penalty
            L_risk = torch.relu(R_hat - alpha_risk)
        else:
            L_risk = torch.tensor(0.0, device=g.device)
            R_hat = torch.tensor(0.0, device=g.device)
        
        # ================== Component 4: Auxiliary Loss ==================
        # Standard CE on auxiliary classifier (no gating)
        ce_loss_aux = F.cross_entropy(auxiliary_out, target)
        
        # ================== Total Loss ==================
        selective_loss = L_pred + self.lm * L_cov
        total_loss = self.alpha * selective_loss + (1.0 - self.alpha) * ce_loss_aux
        
        # Add CRC risk penalty (only during training)
        if mode == 'train' and self.mu > 0:
            total_loss = total_loss + self.mu * L_risk
        
        # ================== Metrics (for logging) ==================
        # Compute additional metrics for monitoring
        with torch.no_grad():
            # Hard coverage and accuracy
            g_hard = (g >= threshold).float()
            hard_coverage = g_hard.mean()
            
            # Selective accuracy
            predictions = prediction_out.argmax(dim=1)
            correct = (predictions == target).float()
            num_correct_accepted = (g_hard * correct).sum()
            selective_acc = num_correct_accepted / (g_hard.sum() + self.eps)
            
            # Raw accuracy
            raw_acc = correct.mean()
            
            # Selective risk (hard)
            if g_hard.sum() > 0:
                r = compute_risk_scores(prediction_out, target)
                selective_risk_hard = (g_hard * r).sum() / (g_hard.sum() + self.eps)
            else:
                selective_risk_hard = torch.tensor(1.0, device=g.device)
        
        # ================== Return Dictionary ==================
        pref = ''
        if mode == 'validation':
            pref = 'val_'
        elif mode == 'test':
            pref = 'test_'
        
        loss_dict = {
            f'{pref}loss': total_loss,
            f'{pref}loss_pred': L_pred.detach().cpu().item(),
            f'{pref}loss_cov': L_cov.detach().cpu().item(),
            f'{pref}loss_risk': L_risk.detach().cpu().item() if isinstance(L_risk, torch.Tensor) else 0.0,
            f'{pref}loss_aux': ce_loss_aux.detach().cpu().item(),
            f'{pref}selective_loss': selective_loss.detach().cpu().item(),
            f'{pref}empirical_coverage': empirical_coverage.detach().cpu().item(),
            f'{pref}hard_coverage': hard_coverage.detach().cpu().item(),
            f'{pref}empirical_risk': empirical_risk.detach().cpu().item(),
            f'{pref}selective_acc': selective_acc.detach().cpu().item(),
            f'{pref}raw_acc': raw_acc.detach().cpu().item(),
            f'{pref}selective_risk': selective_risk_hard.detach().cpu().item(),
            f'{pref}R_hat': R_hat.detach().cpu().item() if isinstance(R_hat, torch.Tensor) else 0.0,
            f'{pref}mu': self.mu,
            f'{pref}alpha_risk': alpha_risk,
        }
        
        return loss_dict
    
    def update_mu(self, R_cal: float, alpha_risk: float, dual_lr: float = 0.01):
        """
        Update Lagrange multiplier mu using dual ascent.
        
        Args:
            R_cal: Empirical risk on calibration set
            alpha_risk: Target risk level
            dual_lr: Learning rate for dual update
        """
        # Dual update: mu <- max(0, mu + eta * (R_cal - alpha))
        self.mu = max(0.0, self.mu + dual_lr * (R_cal - alpha_risk))
    
    def set_mu(self, mu: float):
        """Set mu directly."""
        self.mu = max(0.0, mu)
    
    def get_mu(self) -> float:
        """Get current mu value."""
        return self.mu


if __name__ == '__main__':
    print("Testing CRC-aware loss function...")
    
    # Create dummy data
    batch_size = 64
    num_classes = 10
    
    prediction_out = torch.randn(batch_size, num_classes)
    selection_out = torch.sigmoid(torch.randn(batch_size, 1))
    auxiliary_out = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Create loss function
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    crc_loss = CRCSelectiveLoss(
        loss_func=base_loss,
        coverage=0.8,
        alpha_risk=0.1,
        lm=32.0,
        alpha=0.5,
        mu=1.0
    )
    
    # Test forward pass
    loss_dict = crc_loss(
        prediction_out, selection_out, auxiliary_out, target, mode='train'
    )
    
    print("\nLoss components:")
    for key, value in loss_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Test backward pass
    loss = loss_dict['loss']
    loss.backward()
    print(f"\nBackward pass successful. Loss: {loss.item():.4f}")
    
    # Test mu update
    print(f"\nInitial mu: {crc_loss.get_mu():.4f}")
    crc_loss.update_mu(R_cal=0.15, alpha_risk=0.1, dual_lr=0.01)
    print(f"Updated mu: {crc_loss.get_mu():.4f}")
    
    print("\n✓ All tests passed!")

