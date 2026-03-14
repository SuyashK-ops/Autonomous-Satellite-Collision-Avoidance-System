"""Closest approach estimation between two propagated orbits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from scipy.optimize import minimize_scalar

from orbit_propagator import propagate_orbit_state


def _parse_time_window(time_window: u.Quantity | Sequence[u.Quantity]) -> tuple[u.Quantity, u.Quantity]:
    """Normalize a time-window specification into start/end offsets."""

    if isinstance(time_window, u.Quantity):
        return 0.0 * u.s, time_window.to(u.s)

    if len(time_window) != 2:
        raise ValueError("time_window must be a duration or a two-element sequence.")

    start = time_window[0].to(u.s)
    end = time_window[1].to(u.s)
    if end <= start:
        raise ValueError("time_window end must be greater than start.")
    return start, end


def _relative_distance_km(
    t_seconds: float,
    satellite_orbit: Orbit,
    debris_orbit: Orbit,
    include_j2: bool,
) -> float:
    """Propagate both orbits to ``t_seconds`` and return their separation."""

    delta_t = t_seconds * u.s
    sat_state = propagate_orbit_state(satellite_orbit, delta_t, include_j2=include_j2)
    debris_state = propagate_orbit_state(debris_orbit, delta_t, include_j2=include_j2)
    return float(np.linalg.norm((sat_state.r - debris_state.r).to(u.km).value))


def compute_closest_approach(
    satellite_orbit: Orbit,
    debris_orbit: Orbit,
    time_window: u.Quantity | Sequence[u.Quantity],
    include_j2: bool = True,
) -> dict:
    """Compute time of closest approach and minimum separation distance.

    Parameters
    ----------
    satellite_orbit
        Satellite state at the start of the analysis window.
    debris_orbit
        Debris state at the start of the analysis window.
    time_window
        Either a duration, interpreted as ``[0, duration]``, or a
        ``(start, end)`` sequence of offsets.

    Returns
    -------
    dict
        Dictionary containing the absolute time of closest approach and the
        minimum separation distance in kilometers.
    """

    start, end = _parse_time_window(time_window)
    result = minimize_scalar(
        _relative_distance_km,
        bounds=(start.to_value(u.s), end.to_value(u.s)),
        args=(satellite_orbit, debris_orbit, include_j2),
        method="bounded",
        options={"xatol": 1e-3},
    )

    tca_offset = result.x * u.s
    tca_reference = satellite_orbit.epoch
    tca_time = tca_reference + tca_offset if isinstance(tca_reference, Time) else tca_offset

    return {
        "tca": tca_time,
        "tca_offset_s": float(tca_offset.to_value(u.s)),
        "min_distance": float(result.fun),
        "success": bool(result.success),
    }


closest_approach = compute_closest_approach
