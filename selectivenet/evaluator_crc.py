"""
Unified evaluator for CRC-Select experiments.

Provides comprehensive evaluation including:
- Risk-Coverage curves
- Coverage@Risk(alpha)
- DAR (Dangerous Acceptance Rate) for OOD
- Mixture evaluation (ID + OOD)
"""
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader, ConcatDataset, Subset

from crc.risk_utils import (
    compute_risk_scores,
    compute_selective_risk,
    compute_coverage,
    compute_selective_accuracy
)


class CRCEvaluator:
    """
    Unified evaluator for CRC-Select experiments.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Trained SelectiveNet model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def collect_predictions(
        self, 
        loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect predictions from model on a dataset.
        
        Args:
            loader: DataLoader for the dataset
        
        Returns:
            logits: (N, num_classes)
            selection_scores: (N,)
            targets: (N,)
        """
        all_logits = []
        all_selection_scores = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits, g, _ = self.model(x)
                
                all_logits.append(logits.cpu())
                all_selection_scores.append(g.cpu())
                all_targets.append(y.cpu())
        
        logits = torch.cat(all_logits, dim=0)
        selection_scores = torch.cat(all_selection_scores, dim=0).squeeze()
        targets = torch.cat(all_targets, dim=0)
        
        return logits, selection_scores, targets
    
    def sweep_tau_risk_coverage(
        self,
        loader: DataLoader,
        taus: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate Risk-Coverage curve by sweeping threshold tau.
        
        Args:
            loader: DataLoader for the dataset
            taus: Array of threshold values to evaluate
        
        Returns:
            DataFrame with columns: tau, risk, coverage, selective_acc
        """
        logits, selection_scores, targets = self.collect_predictions(loader)
        
        results = []
        for tau in taus:
            risk = compute_selective_risk(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            )
            coverage = compute_coverage(selection_scores, tau)
            sel_acc, _ = compute_selective_accuracy(
                logits, selection_scores, targets, threshold=tau
            )
            
            results.append({
                'tau': float(tau),
                'risk': risk.item(),
                'coverage': coverage.item(),
                'selective_acc': sel_acc.item()
            })
        
        return pd.DataFrame(results)
    
    def compute_coverage_at_risk(
        self,
        loader: DataLoader,
        alpha: float,
        taus: np.ndarray
    ) -> Dict:
        """
        Compute coverage at target risk level alpha.
        
        Finds the maximum coverage achievable while keeping risk <= alpha.
        
        Args:
            loader: DataLoader for the dataset
            alpha: Target risk level (e.g., 0.1 for 10% risk)
            taus: Array of threshold values to search over
        
        Returns:
            Dictionary with tau, risk, coverage at target alpha
        """
        rc_curve = self.sweep_tau_risk_coverage(loader, taus)
        
        # Find thresholds where risk <= alpha
        valid_rows = rc_curve[rc_curve['risk'] <= alpha]
        
        if len(valid_rows) == 0:
            # No threshold achieves target risk
            return {
                'alpha': alpha,
                'tau': np.nan,
                'risk': np.nan,
                'coverage': 0.0,
                'feasible': False
            }
        
        # Find maximum coverage among valid thresholds
        best_row = valid_rows.loc[valid_rows['coverage'].idxmax()]
        
        return {
            'alpha': alpha,
            'tau': best_row['tau'],
            'risk': best_row['risk'],
            'coverage': best_row['coverage'],
            'feasible': True
        }
    
    def evaluate_ood(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        taus: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute OOD evaluation metrics including DAR.
        
        Args:
            id_loader: DataLoader for ID (in-distribution) data
            ood_loader: DataLoader for OOD (out-of-distribution) data
            taus: Array of threshold values to evaluate
        
        Returns:
            DataFrame with columns: tau, id_accept_rate, ood_accept_rate, dar
        """
        # Collect predictions
        _, id_selection_scores, _ = self.collect_predictions(id_loader)
        _, ood_selection_scores, _ = self.collect_predictions(ood_loader)
        
        results = []
        for tau in taus:
            id_accept_rate = compute_coverage(id_selection_scores, tau).item()
            ood_accept_rate = compute_coverage(ood_selection_scores, tau).item()
            
            # DAR: Dangerous Acceptance Rate (fraction of OOD accepted)
            dar = ood_accept_rate
            
            results.append({
                'tau': float(tau),
                'id_accept_rate': id_accept_rate,
                'ood_accept_rate': ood_accept_rate,
                'dar': dar
            })
        
        return pd.DataFrame(results)
    
    def evaluate_mixture(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        p_ood: float,
        tau: float,
        alpha: float
    ) -> Dict:
        """
        Evaluate on ID+OOD mixture with proportion p_ood of OOD.
        
        Args:
            id_loader: DataLoader for ID data
            ood_loader: DataLoader for OOD data
            p_ood: Proportion of OOD in mixture (e.g., 0.1 for 10% OOD)
            tau: Acceptance threshold
            alpha: Target risk level
        
        Returns:
            Dictionary with mixture evaluation results
        """
        # Collect predictions
        id_logits, id_g, id_targets = self.collect_predictions(id_loader)
        ood_logits, ood_g, ood_targets = self.collect_predictions(ood_loader)
        
        # Create mixture
        num_id = len(id_targets)
        num_ood = len(ood_targets)
        num_ood_sample = int(num_id * p_ood / (1 - p_ood))
        num_ood_sample = min(num_ood_sample, num_ood)
        
        # Sample OOD indices
        ood_indices = np.random.choice(num_ood, num_ood_sample, replace=False)
        
        # Mix
        mixed_logits = torch.cat([id_logits, ood_logits[ood_indices]], dim=0)
        mixed_g = torch.cat([id_g, ood_g[ood_indices]], dim=0)
        mixed_targets = torch.cat([id_targets, ood_targets[ood_indices]], dim=0)
        
        # For OOD, risk = 1 (always wrong since labels don't match)
        # We need to mark OOD samples
        is_id = torch.cat([
            torch.ones(num_id, dtype=torch.bool),
            torch.zeros(num_ood_sample, dtype=torch.bool)
        ])
        
        # Compute risk only on ID samples (OOD has undefined risk)
        id_mask = is_id
        ood_mask = ~is_id
        
        # Acceptance
        acceptance = (mixed_g >= tau).float()
        
        # Coverage
        coverage = acceptance.mean().item()
        
        # Risk on ID samples that are accepted
        id_accepted = id_mask & (acceptance > 0)
        if id_accepted.sum() > 0:
            risk_id_accepted = compute_selective_risk(
                mixed_logits[id_mask], mixed_g[id_mask], mixed_targets[id_mask],
                threshold=tau, hard=True
            ).item()
        else:
            risk_id_accepted = np.nan
        
        # OOD acceptance rate
        ood_accepted = ood_mask & (acceptance > 0)
        ood_accept_rate = ood_accepted.sum().item() / ood_mask.sum().item()
        
        return {
            'p_ood': p_ood,
            'tau': tau,
            'alpha': alpha,
            'coverage': coverage,
            'risk_id_accepted': risk_id_accepted,
            'ood_accept_rate': ood_accept_rate,
            'num_id': num_id,
            'num_ood': num_ood_sample
        }
    
    def evaluate_violation_rate(
        self,
        loaders: List[DataLoader],
        alpha: float,
        tau: float
    ) -> Dict:
        """
        Evaluate violation rate across multiple datasets/splits.
        
        A violation occurs when risk > alpha.
        
        Args:
            loaders: List of DataLoaders (e.g., from different seeds)
            alpha: Target risk level
            tau: Acceptance threshold
        
        Returns:
            Dictionary with violation statistics
        """
        risks = []
        coverages = []
        
        for loader in loaders:
            logits, selection_scores, targets = self.collect_predictions(loader)
            
            risk = compute_selective_risk(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            ).item()
            coverage = compute_coverage(selection_scores, tau).item()
            
            risks.append(risk)
            coverages.append(coverage)
        
        risks = np.array(risks)
        coverages = np.array(coverages)
        
        violations = risks > alpha
        violation_rate = violations.mean()
        
        return {
            'alpha': alpha,
            'tau': tau,
            'violation_rate': violation_rate,
            'num_violations': violations.sum(),
            'num_trials': len(loaders),
            'mean_risk': risks.mean(),
            'std_risk': risks.std(),
            'mean_coverage': coverages.mean(),
            'std_coverage': coverages.std(),
            'risks': risks.tolist(),
            'coverages': coverages.tolist()
        }


if __name__ == '__main__':
    print("Testing CRC evaluator...")
    
    # This would require a trained model and data
    # For now, just verify imports work
    print("✓ Module imports successful")
    
    # Test with dummy model
    from selectivenet.vgg_variant import vgg16_variant
    from selectivenet.model import SelectiveNet
    
    # Create dummy model
    features = vgg16_variant(32, 0.3)
    model = SelectiveNet(features, 512, 10)
    model.eval()
    
    # Create evaluator
    evaluator = CRCEvaluator(model, device='cpu')
    
    # Create dummy data
    dummy_data = torch.randn(100, 3, 32, 32)
    dummy_targets = torch.randint(0, 10, (100,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=32)
    
    # Test risk-coverage curve
    taus = np.linspace(0.3, 0.8, 6)
    rc_curve = evaluator.sweep_tau_risk_coverage(dummy_loader, taus)
    print("\nRisk-Coverage curve:")
    print(rc_curve)
    
    # Test coverage at risk
    result = evaluator.compute_coverage_at_risk(dummy_loader, alpha=0.1, taus=taus)
    print(f"\nCoverage@Risk(0.1): {result}")
    
    print("\n✓ All tests passed!")

