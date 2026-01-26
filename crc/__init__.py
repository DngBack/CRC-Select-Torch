"""
CRC (Conformal Risk Control) module for CRC-Select.
"""
from .risk_utils import (
    compute_risk_scores,
    compute_selective_risk,
    compute_acceptance_mask
)
from .calibrate import (
    compute_crc_threshold,
    calibrate_selector
)

__all__ = [
    'compute_risk_scores',
    'compute_selective_risk',
    'compute_acceptance_mask',
    'compute_crc_threshold',
    'calibrate_selector'
]

