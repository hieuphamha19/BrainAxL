"""
Performance and system monitoring metrics for SSL pretraining.
"""

from numbers import Real
from typing import Any, Dict, Optional


# Frequency-based compute functions
def compute(transforms_applied: Optional[Dict[str, Any]], batch_size: int) -> Dict[str, float]:
    """Metrics computed every training step."""
    return compute_augmentation_rates(transforms_applied, batch_size)


def compute_on_backward(trainer) -> Dict[str, float]:
    """Metrics computed after backward pass."""
    return compute_mixed_precision_metrics(trainer)


def compute_mixed_precision_metrics(trainer) -> Dict[str, float]:
    """
    Automatic Mixed Precision (AMP) loss scaling and gradient overflow tracking.
    Frequent overflows indicate numerical instability requiring intervention.

    Args:
        trainer: PyTorch Lightning trainer instance

    Returns:
        Dictionary with AMP scale and steps since scale update
    """
    metrics = {}

    # Check if using mixed precision training
    if trainer and hasattr(trainer, "scaler") and trainer.scaler is not None:
        # For native PyTorch AMP
        scaler = trainer.scaler
        if hasattr(scaler, "get_scale"):
            metrics["amp_scale"] = scaler.get_scale()
        if hasattr(scaler, "_scale_seq_len"):
            # Track how many steps since last scale update (indicator of overflow)
            metrics["amp_steps_since_scale_update"] = scaler._scale_seq_len
    elif trainer and hasattr(trainer, "precision_plugin"):
        # For Lightning's precision plugins
        precision_plugin = trainer.precision_plugin
        if hasattr(precision_plugin, "scaler") and precision_plugin.scaler is not None:
            scaler = precision_plugin.scaler
            if hasattr(scaler, "get_scale"):
                metrics["amp_scale"] = scaler.get_scale()
            if hasattr(scaler, "_scale_seq_len"):
                metrics["amp_steps_since_scale_update"] = scaler._scale_seq_len

    # If no AMP metrics found, return zeros
    if not metrics:
        metrics = {"amp_scale": 1.0, "amp_steps_since_scale_update": 0}

    return metrics


def _augmentation_count(value: Any) -> Optional[float]:
    """Return a scalar application count, or None for metadata-only entries."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    # Some transforms log structured metadata (e.g. ssl_patch_sampler) instead of counts.
    return None


def compute_augmentation_rates(transforms_applied: Optional[Dict[str, Any]], batch_size: int) -> Dict[str, float]:
    """
    Stochastic augmentation application frequency tracking.
    Ensures augmentation diversity and proper probability calibration.
    """
    metrics = {}

    if transforms_applied is not None:
        for aug_name, count in transforms_applied.items():
            scalar_count = _augmentation_count(count)
            if scalar_count is None:
                continue
            # Calculate rate as fraction of batch where augmentation was applied
            rate = scalar_count / batch_size if batch_size > 0 else 0
            metrics[f"aug_rate/{aug_name}"] = rate

    return metrics
