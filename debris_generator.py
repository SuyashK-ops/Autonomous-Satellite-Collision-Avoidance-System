"""Debris orbit generation utilities.

This module creates synthetic debris populations with randomized orbital
elements suitable for Monte Carlo collision-avoidance experiments.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


DEFAULT_ALTITUDE_RANGE_KM = (400.0, 1200.0)
DEFAULT_INCLINATION_RANGE_DEG = (0.0, 180.0)
DEFAULT_ECCENTRICITY_RANGE = (0.0, 0.02)


def _get_rng(random_state: Optional[int]) -> np.random.Generator:
    """Return a NumPy random generator."""

    return np.random.default_rng(random_state)


def generate_debris(
    n_objects: int = 100,
    altitude_range_km: Sequence[float] = DEFAULT_ALTITUDE_RANGE_KM,
    inclination_range_deg: Sequence[float] = DEFAULT_INCLINATION_RANGE_DEG,
    eccentricity_range: Sequence[float] = DEFAULT_ECCENTRICITY_RANGE,
    epoch: Optional[Time] = None,
    random_state: Optional[int] = None,
) -> list[Orbit]:
    """Generate a list of random debris orbits.

    Parameters
    ----------
    n_objects
        Number of debris objects to generate.
    altitude_range_km
        Inclusive low/high range for orbital altitude above Earth.
    inclination_range_deg
        Inclusive low/high range for orbital inclination.
    eccentricity_range
        Low/high range for small orbital eccentricities.
    epoch
        Epoch assigned to every generated orbit. Defaults to ``Time.now()``.
    random_state
        Seed for reproducible random generation.

    Returns
    -------
    list[Orbit]
        Randomized debris orbits represented as ``poliastro`` orbit objects.
    """

    rng = _get_rng(random_state)
    orbit_epoch = epoch if epoch is not None else Time.now()

    altitude_low, altitude_high = altitude_range_km
    inc_low, inc_high = inclination_range_deg
    ecc_low, ecc_high = eccentricity_range

    debris_orbits: list[Orbit] = []

    for _ in range(n_objects):
        altitude = rng.uniform(altitude_low, altitude_high) * u.km
        inclination = rng.uniform(inc_low, inc_high) * u.deg
        raan = rng.uniform(0.0, 360.0) * u.deg
        argp = rng.uniform(0.0, 360.0) * u.deg
        nu = rng.uniform(0.0, 360.0) * u.deg
        ecc = rng.uniform(ecc_low, ecc_high) * u.one
        semi_major_axis = Earth.R.to(u.km) + altitude

        debris_orbits.append(
            Orbit.from_classical(
                Earth,
                semi_major_axis,
                ecc,
                inclination,
                raan,
                argp,
                nu,
                epoch=orbit_epoch,
            )
        )

    return debris_orbits


def check_conjunction(
    sat_pos: Iterable[float],
    debris_pos: Iterable[float],
    threshold: float = 5.0,
) -> bool:
    """Return ``True`` when the Euclidean distance is below the threshold."""

    sat_vector = np.asarray(sat_pos, dtype=float)
    debris_vector = np.asarray(debris_pos, dtype=float)
    distance_km = np.linalg.norm(sat_vector - debris_vector)
    return bool(distance_km < threshold)
