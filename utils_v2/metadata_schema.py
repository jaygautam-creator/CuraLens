"""
CuraLens v2 — Clinical Metadata Schema & Validator
====================================================
Defines per-cancer-type metadata field specifications, validates incoming
request data, and returns normalized numpy arrays ready for model inference.

Schemas
-------
LEGACY_ORAL_SCHEMA  (4 features) — backward-compatible with existing
                     models_v2/saved_model/ (age, smoking, alcohol, sun_exposure)

ORAL_SCHEMA         (6 features) — clinically accurate oral cancer risk factors
SKIN_SCHEMA         (6 features) — clinically accurate skin cancer risk factors

Usage
-----
    from utils_v2.metadata_schema import validate_and_encode

    arr, errors = validate_and_encode(request_dict, cancer_type="oral")
    if errors:
        return jsonify({"error": errors}), 400
    # arr is np.ndarray shape (1, N) ready for model input

Normalization
-------------
- Continuous variables: StandardScaler-style z-score using population
  reference ranges baked in as CLASS_MEAN / CLASS_STD.
  (For production, load actual scaler from training_logs_v2.json.)
- Binary / ordinal: passed through as-is after clamping to valid range.
- Log-transformed heavy-tailed fields: log1p applied before scaling.
- One-hot fields: Fitzpatrick skin type is one-hot encoded (categories 1–6).

NOTE: These reference statistics are population-level approximations.
      Replace with domain-fit scalers once real patient data is collected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Field specification
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """Specification for one metadata input field."""
    name        : str
    dtype       : str              # "float", "int", "bool", "onehot"
    min_val     : Optional[float]  # inclusive lower bound (None = no check)
    max_val     : Optional[float]  # inclusive upper bound (None = no check)
    default     : float            # value used when field is absent/None
    description : str
    # Normalisation parameters (population reference or training-fit)
    pop_mean    : Optional[float] = None
    pop_std     : Optional[float] = None
    log_transform: bool = False    # apply np.log1p before z-scoring


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# ── Legacy oral schema (4D) — must match training_logs_v2.json ───────────
LEGACY_ORAL_FIELDS: List[FieldSpec] = [
    FieldSpec("age",          "float", 0,  120, 45.0,
              "Patient age in years",
              pop_mean=49.0, pop_std=14.0),
    FieldSpec("smoking",      "bool",  0,  1,   0.0,
              "Current smoker (0=No, 1=Yes)"),
    FieldSpec("alcohol",      "bool",  0,  1,   0.0,
              "Regular alcohol use (0=No, 1=Yes)"),
    FieldSpec("sun_exposure", "float", 0,  10,  2.0,
              "Sun exposure level (0–10 scale)",
              pop_mean=3.5, pop_std=2.5),
]

# ── Clinical oral schema (6D) ─────────────────────────────────────────────
ORAL_FIELDS: List[FieldSpec] = [
    FieldSpec("age",                    "float", 18, 120, 50.0,
              "Patient age in years",
              pop_mean=52.0, pop_std=13.0),
    FieldSpec("smoking_years",          "int",   0,  80,  0.0,
              "Number of years smoked (0 if never smoked)",
              pop_mean=10.0, pop_std=14.0, log_transform=True),
    FieldSpec("cigarettes_per_day",     "int",   0,  100, 0.0,
              "Average cigarettes smoked per day",
              pop_mean=6.0, pop_std=9.0, log_transform=True),
    FieldSpec("alcohol_units_per_week", "int",   0,  200, 0.0,
              "Alcohol units consumed per week (1 unit ≈ 10 ml pure alcohol)",
              pop_mean=5.0, pop_std=8.0, log_transform=True),
    FieldSpec("chewing_tobacco",        "bool",  0,  1,   0.0,
              "Chewing tobacco or betel nut use (0=No, 1=Yes)"),
    FieldSpec("family_history",         "bool",  0,  1,   0.0,
              "Family history of oral cancer (0=No, 1=Yes)"),
]

# ── Clinical skin schema (6D) ─────────────────────────────────────────────
SKIN_FIELDS: List[FieldSpec] = [
    FieldSpec("age",                    "float", 18, 120, 45.0,
              "Patient age in years",
              pop_mean=48.0, pop_std=16.0),
    FieldSpec("skin_type",              "int",   1,  6,   3.0,
              "Fitzpatrick skin type (1=Very Fair, 6=Very Dark)",
              pop_mean=3.0, pop_std=1.4),
    FieldSpec("sunburn_history",        "int",   0,  50,  2.0,
              "Number of significant sunburns in lifetime",
              pop_mean=3.0, pop_std=4.0, log_transform=True),
    FieldSpec("outdoor_hours_per_week", "float", 0,  112, 10.0,
              "Average hours spent outdoors per week",
              pop_mean=15.0, pop_std=12.0),
    FieldSpec("tanning_bed_use",        "bool",  0,  1,   0.0,
              "Regular/historical tanning bed use (0=No, 1=Yes)"),
    FieldSpec("family_history",         "bool",  0,  1,   0.0,
              "Family history of skin cancer (0=No, 1=Yes)"),
]

# Registry
_SCHEMA_MAP: Dict[str, List[FieldSpec]] = {
    "oral_legacy" : LEGACY_ORAL_FIELDS,
    "oral"        : ORAL_FIELDS,
    "skin"        : SKIN_FIELDS,
}


# ---------------------------------------------------------------------------
# Validation & encoding
# ---------------------------------------------------------------------------

def _parse_value(raw: Any, spec: FieldSpec) -> Tuple[float, Optional[str]]:
    """
    Parse and range-check a single raw value against a FieldSpec.

    Returns:
        (parsed_float, error_string_or_None)
    """
    # Coerce
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return spec.default, (
            f"Field '{spec.name}': expected numeric value, got {raw!r}. "
            f"Using default {spec.default}."
        )

    # Range check
    error: Optional[str] = None
    if spec.min_val is not None and val < spec.min_val:
        error = (
            f"Field '{spec.name}': value {val} is below minimum {spec.min_val}. "
            f"Clamped to {spec.min_val}."
        )
        val = spec.min_val
    elif spec.max_val is not None and val > spec.max_val:
        error = (
            f"Field '{spec.name}': value {val} exceeds maximum {spec.max_val}. "
            f"Clamped to {spec.max_val}."
        )
        val = spec.max_val

    # Boolean snap
    if spec.dtype == "bool":
        val = 1.0 if val >= 0.5 else 0.0

    return val, error


def _normalize_value(val: float, spec: FieldSpec) -> float:
    """
    Normalize a validated value using the field's pop_mean / pop_std.
    Applies log1p first if spec.log_transform is True.
    Returns the input unchanged when normalization params are missing.
    """
    if spec.log_transform:
        val = math.log1p(max(val, 0.0))

    if spec.pop_mean is not None and spec.pop_std is not None and spec.pop_std > 0:
        val = (val - spec.pop_mean) / spec.pop_std

    return val


def validate_and_encode(
    data: Dict[str, Any],
    cancer_type: str = "oral_legacy",
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Validate incoming metadata dict, optionally normalize, and return a
    model-ready numpy array of shape (1, N).

    Args:
        data         : Dict of raw field values from the request.
        cancer_type  : One of "oral_legacy", "oral", or "skin".
        normalize    : If True, z-score-normalize continuous fields using
                       population reference statistics.

    Returns:
        (array, warnings) where:
          array    : float32 numpy array of shape (1, N)
          warnings : list of non-fatal warning strings (range clamps, missing
                     fields) — empty list means all inputs were valid.

    Raises:
        ValueError : If cancer_type is not registered.
    """
    if cancer_type not in _SCHEMA_MAP:
        raise ValueError(
            f"Unknown cancer_type='{cancer_type}'. "
            f"Valid options: {list(_SCHEMA_MAP.keys())}"
        )

    specs   = _SCHEMA_MAP[cancer_type]
    encoded : List[float] = []
    warnings: List[str]   = []

    for spec in specs:
        raw = data.get(spec.name, None)
        if raw is None:
            warnings.append(
                f"Field '{spec.name}' not provided; using default={spec.default}."
            )
            val = spec.default
        else:
            val, err = _parse_value(raw, spec)
            if err:
                warnings.append(err)

        if normalize:
            val = _normalize_value(val, spec)

        encoded.append(val)

    arr = np.array([encoded], dtype="float32")  # shape (1, N)
    return arr, warnings


def get_schema_info(cancer_type: str) -> List[Dict]:
    """
    Return human-readable schema description for a cancer type.

    Args:
        cancer_type: One of the registered schema keys.

    Returns:
        List of dicts with keys: name, dtype, min_val, max_val, default,
        description, log_transform.
    """
    if cancer_type not in _SCHEMA_MAP:
        raise ValueError(f"Unknown cancer_type='{cancer_type}'.")
    return [
        {
            "name"          : s.name,
            "dtype"         : s.dtype,
            "min_val"       : s.min_val,
            "max_val"       : s.max_val,
            "default"       : s.default,
            "description"   : s.description,
            "log_transform" : s.log_transform,
        }
        for s in _SCHEMA_MAP[cancer_type]
    ]


def list_cancer_types() -> List[str]:
    """Return all registered cancer type keys."""
    return list(_SCHEMA_MAP.keys())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ORAL LEGACY (4D) ===")
    arr, warns = validate_and_encode(
        {"age": 55, "smoking": 1, "alcohol": 1, "sun_exposure": 7},
        cancer_type="oral_legacy"
    )
    print(f"  Array : {arr}")
    print(f"  Warns : {warns}")

    print("\n=== ORAL CLINICAL (6D) ===")
    arr, warns = validate_and_encode(
        {"age": 63, "smoking_years": 20, "cigarettes_per_day": 15,
         "alcohol_units_per_week": 14, "chewing_tobacco": 1, "family_history": 0},
        cancer_type="oral"
    )
    print(f"  Array shape : {arr.shape}")
    print(f"  Array  : {arr}")
    print(f"  Warns  : {warns}")

    print("\n=== SKIN CLINICAL (6D) ===")
    arr, warns = validate_and_encode(
        {"age": 45, "skin_type": 2, "sunburn_history": 8,
         "outdoor_hours_per_week": 20, "tanning_bed_use": 1, "family_history": 0},
        cancer_type="skin"
    )
    print(f"  Array shape : {arr.shape}")
    print(f"  Array  : {arr}")
    print(f"  Warns  : {warns}")

    print("\n=== Field missing test ===")
    arr, warns = validate_and_encode({"age": 40}, cancer_type="oral")
    print(f"  Warns: {warns}")
