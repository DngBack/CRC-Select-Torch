"""
Unified evaluator for CRC-Select experiments.

Provides comprehensive evaluation including:
- Accepted-loss mass (CRC-certified quantity, Theorem 1)
- Risk-Coverage curves (accepted-loss mass vs coverage)
- Coverage@Risk (maximum coverage at given alpha)
- Violation gap (A_hat - alpha)
- Conditional selective risk (descriptive)
- Selective accuracy
- AUROC / AUPR for error detection
- DAR (Dangerous Acceptance Rate) for OOD
- Risk-Coverage AUC
- Threshold conservativeness
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
    compute_accepted_loss_mass,
    compute_coverage,
    compute_selective_accuracy,
    compute_error_detection_auroc_aupr,
)
from crc.calibrate import crc_calibrate_threshold


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
        
        Includes both accepted-loss mass (audited) and conditional risk (descriptive).
        
        Args:
            loader: DataLoader for the dataset
            taus: Array of threshold values to evaluate
        
        Returns:
            DataFrame with columns: tau, coverage, accepted_loss_mass, 
                                   selective_risk, selective_acc, abstention_rate
        """
        logits, selection_scores, targets = self.collect_predictions(loader)
        
        results = []
        for tau in taus:
            coverage = compute_coverage(selection_scores, tau)
            risk = compute_selective_risk(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            )
            alm = compute_accepted_loss_mass(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            )
            sel_acc, _ = compute_selective_accuracy(
                logits, selection_scores, targets, threshold=tau
            )
            
            results.append({
                'tau': float(tau),
                'coverage': coverage.item(),
                'accepted_loss_mass': alm.item(),
                'selective_risk': risk.item(),
                'selective_acc': sel_acc.item(),
                'abstention_rate': 1.0 - coverage.item(),
                # legacy alias
                'risk': risk.item(),
            })
        
        return pd.DataFrame(results)
    
    def compute_coverage_at_risk(
        self,
        loader: DataLoader,
        alpha: float,
        taus: np.ndarray
    ) -> Dict:
        """
        Compute coverage at target risk level alpha using accepted-loss mass.
        
        Finds the maximum coverage achievable while keeping A_hat <= alpha
        (CRC guarantee) and also reports conditional risk.
        
        Args:
            loader: DataLoader for the dataset
            alpha: Target risk level (e.g., 0.1 for 10% risk)
            taus: Array of threshold values to search over
        
        Returns:
            Dictionary with tau, accepted_loss_mass, coverage at target alpha
        """
        rc_curve = self.sweep_tau_risk_coverage(loader, taus)
        
        # CRC guarantee is on accepted_loss_mass, not conditional risk
        valid_rows = rc_curve[rc_curve['accepted_loss_mass'] <= alpha]
        
        if len(valid_rows) == 0:
            return {
                'alpha': alpha,
                'tau': np.nan,
                'accepted_loss_mass': np.nan,
                'selective_risk': np.nan,
                'coverage': 0.0,
                'feasible': False
            }
        
        # Find maximum coverage among valid thresholds
        best_row = valid_rows.loc[valid_rows['coverage'].idxmax()]
        
        return {
            'alpha': alpha,
            'tau': best_row['tau'],
            'accepted_loss_mass': best_row['accepted_loss_mass'],
            'selective_risk': best_row['selective_risk'],
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
    
    def compute_ood_acceptance_at_fixed_id_coverage(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        target_id_coverages: List[float] = [0.7, 0.8, 0.9]
    ) -> pd.DataFrame:
        """
        Compute OOD acceptance rate at fixed ID coverage levels.
        
        This is the recommended metric for fair comparison across methods:
        - Fix ID coverage (e.g., 70%, 80%, 90%)
        - Measure OOD acceptance at that coverage
        - Lower OOD acceptance = better OOD safety
        
        This metric is superior to sweeping threshold because:
        1. Fair comparison: all methods evaluated at same ID coverage
        2. Practical: coverage is often the constraint in deployment
        3. Clear interpretation: "at 80% ID coverage, how much OOD leaks through?"
        
        Args:
            id_loader: DataLoader for ID (in-distribution) data
            ood_loader: DataLoader for OOD (out-of-distribution) data
            target_id_coverages: List of target ID coverage levels
        
        Returns:
            DataFrame with columns:
            - id_coverage_target: Target ID coverage
            - threshold: Threshold τ that achieves target ID coverage
            - id_coverage_actual: Actual ID coverage at τ
            - ood_accept_rate: OOD acceptance rate at τ (DAR)
            - safety_ratio: ID accept / OOD accept (higher is better)
        """
        # Collect predictions
        _, id_g, _ = self.collect_predictions(id_loader)
        _, ood_g, _ = self.collect_predictions(ood_loader)
        
        results = []
        for target_cov in target_id_coverages:
            # Find threshold that gives desired ID coverage
            # Use quantile: to get 70% coverage, reject bottom 30%
            tau = torch.quantile(id_g, 1.0 - target_cov).item()
            
            # Measure actual ID and OOD acceptance at this tau
            id_accept = (id_g >= tau).float().mean().item()
            ood_accept = (ood_g >= tau).float().mean().item()
            
            # Safety ratio: how much more ID than OOD is accepted
            safety_ratio = id_accept / (ood_accept + 1e-8)
            
            results.append({
                'id_coverage_target': target_cov,
                'threshold': tau,
                'id_coverage_actual': id_accept,
                'ood_accept_rate': ood_accept,
                'dar': ood_accept,  # Same as ood_accept_rate
                'safety_ratio': safety_ratio
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
        
        A violation occurs when accepted_loss_mass > alpha (CRC guarantee).
        Also tracks conditional risk for descriptive purposes.
        
        Args:
            loaders: List of DataLoaders (e.g., from different seeds)
            alpha: Target risk level
            tau: Acceptance threshold
        
        Returns:
            Dictionary with violation statistics
        """
        alms = []
        risks = []
        coverages = []
        
        for loader in loaders:
            logits, selection_scores, targets = self.collect_predictions(loader)
            
            alm = compute_accepted_loss_mass(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            ).item()
            risk = compute_selective_risk(
                logits, selection_scores, targets,
                threshold=tau, hard=True
            ).item()
            coverage = compute_coverage(selection_scores, tau).item()
            
            alms.append(alm)
            risks.append(risk)
            coverages.append(coverage)
        
        alms = np.array(alms)
        risks = np.array(risks)
        coverages = np.array(coverages)
        
        # CRC guarantee is on accepted_loss_mass
        violations = alms > alpha
        violation_rate = violations.mean()
        violation_gaps = np.maximum(alms - alpha, 0.0)
        
        return {
            'alpha': alpha,
            'tau': tau,
            'violation_rate': violation_rate,
            'num_violations': int(violations.sum()),
            'num_trials': len(loaders),
            'mean_accepted_loss_mass': alms.mean(),
            'std_accepted_loss_mass': alms.std(),
            'mean_violation_gap': violation_gaps.mean(),
            'max_violation_gap': violation_gaps.max(),
            'mean_risk': risks.mean(),
            'std_risk': risks.std(),
            'mean_coverage': coverages.mean(),
            'std_coverage': coverages.std(),
            'accepted_loss_masses': alms.tolist(),
            'risks': risks.tolist(),
            'coverages': coverages.tolist()
        }

    # ------------------------------------------------------------------
    # New comprehensive evaluation methods
    # ------------------------------------------------------------------

    def evaluate_with_crc_calibration(
        self,
        cal_loader: DataLoader,
        test_loader: DataLoader,
        alpha: float,
    ) -> Dict:
        """
        Full CRC evaluation: calibrate on cal set, evaluate on test set.

        Uses the correct CRC formula:
            hat_tau = inf{tau : (n/(n+1)) * A_hat_n(tau) + 1/(n+1) <= alpha}

        Args:
            cal_loader: Calibration data loader
            test_loader: Test data loader
            alpha: Target risk level

        Returns:
            Dictionary with calibrated threshold and all metrics on test set
        """
        # --- calibrate on calibration set ---
        cal_logits, cal_g, cal_targets = self.collect_predictions(cal_loader)
        cal_result = crc_calibrate_threshold(
            cal_logits, cal_g, cal_targets, alpha
        )
        tau_hat = cal_result['tau_hat']

        # --- evaluate on test set ---
        test_logits, test_g, test_targets = self.collect_predictions(test_loader)

        alm = compute_accepted_loss_mass(
            test_logits, test_g, test_targets, threshold=tau_hat, hard=True
        ).item()
        risk = compute_selective_risk(
            test_logits, test_g, test_targets, threshold=tau_hat, hard=True
        ).item()
        cov = compute_coverage(test_g, tau_hat).item()
        sel_acc, _ = compute_selective_accuracy(
            test_logits, test_g, test_targets, threshold=tau_hat
        )
        sel_acc = sel_acc.item()

        # Violation gap
        violation_gap = max(alm - alpha, 0.0)

        # AUROC / AUPR for error detection
        auroc_aupr = compute_error_detection_auroc_aupr(
            test_logits, test_g, test_targets
        )
        auroc = auroc_aupr['auroc']
        aupr = auroc_aupr['aupr']

        return {
            'alpha': alpha,
            'tau_hat': tau_hat,
            'cal_accepted_loss_mass': cal_result['accepted_loss_mass'],
            'test_accepted_loss_mass': alm,
            'test_selective_risk': risk,
            'test_coverage': cov,
            'test_selective_acc': sel_acc,
            'test_abstention_rate': 1.0 - cov,
            'violation_gap': violation_gap,
            'violated': alm > alpha,
            'auroc': auroc,
            'aupr': aupr,
        }

    def compute_rc_auc(
        self,
        loader: DataLoader,
        num_points: int = 200,
        use_accepted_loss_mass: bool = True,
    ) -> float:
        """
        Area under the Risk-Coverage curve.

        Args:
            loader: DataLoader
            num_points: Number of threshold points
            use_accepted_loss_mass: If True, use accepted-loss mass on y-axis
                                   (CRC-certified). Otherwise use conditional risk.

        Returns:
            AUC value (lower is better)
        """
        taus = np.linspace(0.0, 1.0, num_points)
        rc = self.sweep_tau_risk_coverage(loader, taus)
        rc = rc.sort_values('coverage')

        y_col = 'accepted_loss_mass' if use_accepted_loss_mass else 'selective_risk'
        auc = np.trapz(rc[y_col].values, rc['coverage'].values)
        return float(auc)

    def compute_threshold_conservativeness(
        self,
        cal_loader: DataLoader,
        test_loader: DataLoader,
        alpha: float,
    ) -> Dict:
        """
        Measure how conservative the CRC threshold is.

        conservativeness = alpha - A_hat_test (positive = under budget = conservative)
        efficiency = coverage achieved / max possible coverage at alpha

        Args:
            cal_loader: Calibration data loader
            test_loader: Test data loader
            alpha: Target risk level

        Returns:
            Dictionary with conservativeness metrics
        """
        result = self.evaluate_with_crc_calibration(cal_loader, test_loader, alpha)
        conservativeness = alpha - result['test_accepted_loss_mass']

        # Oracle coverage: best possible coverage at alpha on test set
        taus = np.linspace(0.0, 1.0, 500)
        rc = self.sweep_tau_risk_coverage(test_loader, taus)
        valid = rc[rc['accepted_loss_mass'] <= alpha]
        oracle_coverage = valid['coverage'].max() if len(valid) > 0 else 0.0

        efficiency = (result['test_coverage'] / oracle_coverage
                      if oracle_coverage > 0 else 0.0)

        return {
            'alpha': alpha,
            'tau_hat': result['tau_hat'],
            'test_accepted_loss_mass': result['test_accepted_loss_mass'],
            'conservativeness': conservativeness,
            'test_coverage': result['test_coverage'],
            'oracle_coverage': oracle_coverage,
            'coverage_efficiency': efficiency,
        }

    def compute_all_metrics(
        self,
        cal_loader: DataLoader,
        test_loader: DataLoader,
        alpha: float,
        ood_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """
        Compute every metric required by the paper in one call.

        Returns a flat dictionary suitable for a single row of a results CSV.
        """
        # CRC calibration + core metrics
        core = self.evaluate_with_crc_calibration(cal_loader, test_loader, alpha)

        # RC-AUC (accepted-loss mass version)
        rc_auc_alm = self.compute_rc_auc(test_loader, use_accepted_loss_mass=True)
        rc_auc_risk = self.compute_rc_auc(test_loader, use_accepted_loss_mass=False)

        # Threshold conservativeness
        cons = self.compute_threshold_conservativeness(
            cal_loader, test_loader, alpha
        )

        result = {
            **core,
            'rc_auc_alm': rc_auc_alm,
            'rc_auc_risk': rc_auc_risk,
            'conservativeness': cons['conservativeness'],
            'oracle_coverage': cons['oracle_coverage'],
            'coverage_efficiency': cons['coverage_efficiency'],
        }

        # OOD metrics if loader provided
        if ood_loader is not None:
            tau_hat = core['tau_hat']
            _, ood_g, _ = self.collect_predictions(ood_loader)
            ood_accept_rate = (ood_g >= tau_hat).float().mean().item()
            dar = ood_accept_rate

            id_cov = core['test_coverage']
            safety_ratio = id_cov / (ood_accept_rate + 1e-8)

            result.update({
                'ood_accept_rate': ood_accept_rate,
                'dar': dar,
                'safety_ratio': safety_ratio,
            })

        return result


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

