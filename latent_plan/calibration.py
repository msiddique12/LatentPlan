from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch

from latent_plan.model import WorldModel
from latent_plan.train import TransitionBatch


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_uncertainty: float
    mean_error: float

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def collect_uncertainty_error_samples(
    model: WorldModel,
    transitions: TransitionBatch,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample uncertainty and latent prediction error.

    Uncertainty is ensemble variance in latent predictions.
    Error is squared latent prediction error to encoded next-state target.
    """
    states = torch.from_numpy(transitions.states).float().to(device)
    actions = torch.from_numpy(transitions.actions).long().to(device)
    next_states = torch.from_numpy(transitions.next_states).float().to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(states)
        z_next_ensemble = model.predict_next_ensemble(z, actions)
        z_next_mean = z_next_ensemble.mean(dim=0)
        z_next_target = model.encode(next_states)

        error = ((z_next_mean - z_next_target) ** 2).mean(dim=1)
        if model.num_dynamics_models > 1:
            uncertainty = z_next_ensemble.var(dim=0).mean(dim=1)
        else:
            uncertainty = torch.zeros_like(error)

    return {
        "uncertainty": uncertainty.cpu().numpy(),
        "error": error.cpu().numpy(),
    }


def build_calibration_bins(
    uncertainty: np.ndarray,
    error: np.ndarray,
    num_bins: int = 10,
) -> List[CalibrationBin]:
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    if uncertainty.shape != error.shape:
        raise ValueError("uncertainty and error must have the same shape")
    if uncertainty.size == 0:
        return []

    lower = float(np.min(uncertainty))
    upper = float(np.max(uncertainty))
    if upper <= lower:
        upper = lower + 1e-8
    edges = np.linspace(lower, upper, num_bins + 1)

    bins: List[CalibrationBin] = []
    for i in range(num_bins):
        left, right = float(edges[i]), float(edges[i + 1])
        if i == num_bins - 1:
            mask = (uncertainty >= left) & (uncertainty <= right)
        else:
            mask = (uncertainty >= left) & (uncertainty < right)

        if not np.any(mask):
            bins.append(
                CalibrationBin(
                    lower=left,
                    upper=right,
                    count=0,
                    mean_uncertainty=0.0,
                    mean_error=0.0,
                )
            )
            continue

        bins.append(
            CalibrationBin(
                lower=left,
                upper=right,
                count=int(np.sum(mask)),
                mean_uncertainty=float(np.mean(uncertainty[mask])),
                mean_error=float(np.mean(error[mask])),
            )
        )
    return bins


def summarize_calibration(
    uncertainty: np.ndarray,
    error: np.ndarray,
    bins: List[CalibrationBin],
) -> Dict[str, float]:
    if uncertainty.size == 0:
        return {"corr": 0.0, "slope": 0.0, "intercept": 0.0, "ece": 0.0}

    if np.allclose(uncertainty, uncertainty[0]):
        slope = 0.0
        intercept = float(np.mean(error))
        corr = 0.0
    else:
        slope, intercept = np.polyfit(uncertainty, error, deg=1)
        corr = float(np.corrcoef(uncertainty, error)[0, 1])

    total = max(float(uncertainty.size), 1.0)
    ece = 0.0
    for item in bins:
        if item.count == 0:
            continue
        weight = item.count / total
        ece += weight * abs(item.mean_error - item.mean_uncertainty)

    return {
        "corr": float(corr),
        "slope": float(slope),
        "intercept": float(intercept),
        "ece": float(ece),
    }


def suggest_risk_penalty(calibration_stats: Dict[str, float]) -> Dict[str, float]:
    """
    Heuristic risk-penalty recommendation from calibration quality.

    If slope is high and correlation is positive, uncertainty tracks error well,
    so stronger penalty is usually beneficial.
    """
    slope = max(0.0, float(calibration_stats.get("slope", 0.0)))
    corr = max(0.0, float(calibration_stats.get("corr", 0.0)))
    ece = max(0.0, float(calibration_stats.get("ece", 0.0)))

    base = min(0.3, 0.05 + 0.2 * corr + 0.1 * min(slope, 1.0))
    conservative = min(0.4, base + 0.1 + 0.5 * min(ece, 0.2))
    aggressive = max(0.0, base - 0.03)

    return {
        "aggressive": float(aggressive),
        "default": float(base),
        "conservative": float(conservative),
    }
