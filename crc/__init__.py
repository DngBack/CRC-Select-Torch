"""
CRC (Conformal Risk Control) module for CRC-Select.
"""
from .risk_utils import (
    compute_risk_scores,
    compute_selective_risk,
    compute_accepted_loss_mass,
    compute_crc_loss_per_sample,
    compute_acceptance_mask,
    compute_coverage,
    compute_selective_accuracy,
    compute_error_detection_auroc_aupr,
)
from .calibrate import (
    crc_calibrate_threshold,
    compute_crc_threshold,
    calibrate_selector,
    compute_crc_threshold_advanced,
)

__all__ = [
    'compute_risk_scores',
    'compute_selective_risk',
    'compute_accepted_loss_mass',
    'compute_crc_loss_per_sample',
    'compute_acceptance_mask',
    'compute_coverage',
    'compute_selective_accuracy',
    'compute_error_detection_auroc_aupr',
    'crc_calibrate_threshold',
    'compute_crc_threshold',
    'calibrate_selector',
    'compute_crc_threshold_advanced',
]

