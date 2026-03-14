"""Conjunction detection helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from astropy import units as u


def _as_km_vector(vector: Iterable[float] | u.Quantity) -> np.ndarray:
    """Convert a position vector to a NumPy array in kilometers."""

    if isinstance(vector, u.Quantity):
        return np.atleast_1d(vector.to(u.km).value).astype(float)
    return np.asarray(vector, dtype=float)


def compute_distance(
    r1: Iterable[float] | u.Quantity,
    r2: Iterable[float] | u.Quantity,
) -> float:
    """Compute Euclidean separation in kilometers between two position vectors."""

    return float(np.linalg.norm(_as_km_vector(r1) - _as_km_vector(r2)))


def detect_conjunction(
    sat_position: Iterable[float] | u.Quantity,
    debris_positions: Sequence[Iterable[float] | u.Quantity],
    threshold_km: float,
) -> list[dict]:
    """Return conjunction events for debris objects within ``threshold_km``.

    Each event contains the debris index and the instantaneous miss distance.
    """

    events: list[dict] = []
    sat_vector = _as_km_vector(sat_position)

    for index, debris_position in enumerate(debris_positions):
        distance_km = compute_distance(sat_vector, debris_position)
        if distance_km <= threshold_km:
            events.append(
                {
                    "debris_index": index,
                    "distance_km": distance_km,
                    "threshold_km": float(threshold_km),
                }
            )

    return events
