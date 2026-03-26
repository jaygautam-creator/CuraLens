"""
CuraLens v2 - Risk Scoring Module
==================================
Converts a raw sigmoid probability into a structured clinical risk report.

Risk tiers:
    Low    : [0.0, 0.3)   → Routine monitoring recommended
    Medium : [0.3, 0.7)   → Further clinical evaluation advised
    High   : [0.7, 1.0]   → Urgent specialist referral required

Usage:
    from utils_v2.risk_scoring import score_prediction, batch_score

    result = score_prediction(0.82)
    print(result)
    # {
    #   "probability": 0.82,
    #   "risk_level": "High",
    #   "risk_label": "High Risk",
    #   "recommendation": "Urgent specialist referral required",
    #   "confidence_band": [0.7, 1.0]
    # }
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Union
import numpy as np


# ---------------------------------------------------------------------------
# Risk tier definitions  (lower_bound inclusive, upper_bound exclusive
#                         except the final tier which is fully closed)
# ---------------------------------------------------------------------------

RISK_TIERS = [
    {
        "level"           : "Low",
        "label"           : "Low Risk",
        "lower"           : 0.0,
        "upper"           : 0.3,
        "recommendation"  : "No immediate concern. Routine monitoring recommended.",
        "color_code"      : "#2ECC71",   # green – useful for front-end display
    },
    {
        "level"           : "Medium",
        "label"           : "Medium Risk",
        "lower"           : 0.3,
        "upper"           : 0.7,
        "recommendation"  : "Borderline result. Further clinical evaluation advised.",
        "color_code"      : "#F39C12",   # amber
    },
    {
        "level"           : "High",
        "label"           : "High Risk",
        "lower"           : 0.7,
        "upper"           : 1.0,
        "recommendation"  : "High probability of malignancy. Urgent specialist referral required.",
        "color_code"      : "#E74C3C",   # red
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Structured output of a single risk assessment."""
    probability    : float
    risk_level     : str        # "Low" | "Medium" | "High"
    risk_label     : str        # Human-readable label
    recommendation : str        # Clinical action string
    confidence_band: list       # [lower, upper] thresholds for this tier
    color_code     : str        # Hex color for UI rendering

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (e.g. for JSON responses)."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[CuraLens v2 | Risk Assessment]\n"
            f"  Probability   : {self.probability:.4f} "
            f"({self.probability * 100:.1f}%)\n"
            f"  Risk Level    : {self.risk_label}\n"
            f"  Band          : [{self.confidence_band[0]}, {self.confidence_band[1]}]\n"
            f"  Recommendation: {self.recommendation}"
        )


def score_prediction(probability: float) -> RiskResult:
    """
    Convert a single sigmoid probability into a structured RiskResult.

    Args:
        probability: Float in [0.0, 1.0] output by the model's sigmoid layer.

    Returns:
        RiskResult dataclass.

    Raises:
        ValueError: If probability is outside [0, 1].
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(
            f"Probability must be in [0, 1], got {probability}"
        )

    # Walk tiers from lowest to highest; match the first tier whose upper
    # bound is strictly greater than the probability (last tier catches 1.0)
    for tier in RISK_TIERS:
        if probability < tier["upper"] or tier["upper"] == 1.0:
            return RiskResult(
                probability     = round(float(probability), 6),
                risk_level      = tier["level"],
                risk_label      = tier["label"],
                recommendation  = tier["recommendation"],
                confidence_band = [tier["lower"], tier["upper"]],
                color_code      = tier["color_code"],
            )

    # Should never reach here, but defend against floating-point edge cases
    raise RuntimeError(f"Could not assign risk tier for probability={probability}")


def batch_score(
    probabilities: Union[List[float], np.ndarray]
) -> List[RiskResult]:
    """
    Score a batch of probabilities.

    Args:
        probabilities: 1-D array or list of sigmoid probabilities.

    Returns:
        List of RiskResult objects in the same order.
    """
    probs = np.asarray(probabilities, dtype=float).flatten()
    return [score_prediction(float(p)) for p in probs]


def summarise_batch(results: List[RiskResult]) -> dict:
    """
    Aggregate statistics for a batch of RiskResult objects.

    Args:
        results: List of RiskResult from batch_score().

    Returns:
        Dictionary with counts, percentages, and mean probability per tier.
    """
    total = len(results)
    if total == 0:
        return {}

    tier_stats: dict = {t["level"]: {"count": 0, "probabilities": []} for t in RISK_TIERS}

    for r in results:
        tier_stats[r.risk_level]["count"] += 1
        tier_stats[r.risk_level]["probabilities"].append(r.probability)

    summary = {"total": total, "tiers": {}}
    for level, stat in tier_stats.items():
        probs = stat["probabilities"]
        summary["tiers"][level] = {
            "count"      : stat["count"],
            "percentage" : round(stat["count"] / total * 100, 2),
            "mean_prob"  : round(float(np.mean(probs)), 4) if probs else None,
        }

    return summary


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_probs = [0.05, 0.29, 0.30, 0.55, 0.70, 0.95, 1.0]
    print("=== Individual Scores ===")
    for p in test_probs:
        result = score_prediction(p)
        print(result)
        print()

    print("=== Batch Summary ===")
    import json
    results = batch_score(test_probs)
    print(json.dumps(summarise_batch(results), indent=2))
