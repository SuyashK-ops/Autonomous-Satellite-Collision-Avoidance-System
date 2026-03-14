"""Orbit construction and propagation utilities.

This module provides a numerical propagator with optional J2 perturbation so
the rest of the project can work with a more realistic Earth-orbit model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.integrate import solve_ivp


EARTH_J2 = 1.08262668e-3


def create_satellite_orbit() -> Orbit:
    """Create a representative LEO satellite orbit."""

    altitude = 500 * u.km
    inclination = 60 * u.deg
    return Orbit.circular(Earth, alt=altitude, inc=inclination)


def j2_acceleration(
    position_km: Sequence[float],
    mu_km3_s2: float,
    radius_km: float,
    j2: float = EARTH_J2,
) -> np.ndarray:
    """Compute the J2 perturbation acceleration in km/s^2."""

    x, y, z = np.asarray(position_km, dtype=float)
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if r == 0.0:
        raise ValueError("Position norm cannot be zero for J2 acceleration.")

    z2 = z * z
    factor = 1.5 * j2 * mu_km3_s2 * radius_km**2 / r**5
    common = 5.0 * z2 / r2
    return factor * np.array(
        [
            x * (common - 1.0),
            y * (common - 1.0),
            z * (common - 3.0),
        ],
        dtype=float,
    )


def _state_derivative(
    _t_seconds: float,
    state: np.ndarray,
    mu_km3_s2: float,
    include_j2: bool,
) -> np.ndarray:
    """Return the Cartesian state derivative for propagation."""

    position = state[:3]
    velocity = state[3:]
    radius = np.linalg.norm(position)
    two_body_acceleration = -mu_km3_s2 * position / radius**3
    acceleration = two_body_acceleration

    if include_j2:
        acceleration = acceleration + j2_acceleration(
            position,
            mu_km3_s2=mu_km3_s2,
            radius_km=Earth.R.to_value(u.km),
        )

    return np.hstack((velocity, acceleration))


def propagate_state(
    position: Iterable[float] | u.Quantity,
    velocity: Iterable[float] | u.Quantity,
    delta_t: u.Quantity,
    include_j2: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> tuple[u.Quantity, u.Quantity]:
    """Numerically propagate a Cartesian state by ``delta_t``."""

    position_km = (
        np.asarray(position.to_value(u.km), dtype=float)
        if isinstance(position, u.Quantity)
        else np.asarray(position, dtype=float)
    )
    velocity_km_s = (
        np.asarray(velocity.to_value(u.km / u.s), dtype=float)
        if isinstance(velocity, u.Quantity)
        else np.asarray(velocity, dtype=float)
    )
    state0 = np.hstack((position_km, velocity_km_s))
    duration_s = float(delta_t.to_value(u.s))

    if duration_s == 0.0:
        return position_km * u.km, velocity_km_s * u.km / u.s

    solution = solve_ivp(
        _state_derivative,
        (0.0, duration_s),
        state0,
        args=(Earth.k.to_value(u.km**3 / u.s**2), include_j2),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    final_state = solution.y[:, -1]
    return final_state[:3] * u.km, final_state[3:] * u.km / u.s


def propagate_orbit_state(
    orbit: Orbit,
    delta_t: u.Quantity,
    include_j2: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> Orbit:
    """Propagate an orbit numerically and return the updated ``Orbit`` object."""

    propagated_r, propagated_v = propagate_state(
        orbit.r,
        orbit.v,
        delta_t,
        include_j2=include_j2,
        rtol=rtol,
        atol=atol,
    )
    return Orbit.from_vectors(
        orbit.attractor,
        propagated_r,
        propagated_v,
        epoch=orbit.epoch + delta_t,
    )


def sample_orbit_positions(
    orbit: Orbit,
    duration: u.Quantity,
    samples: int = 500,
    include_j2: bool = True,
) -> np.ndarray:
    """Return sampled position vectors in kilometers over a propagation arc."""

    times = np.linspace(0.0, duration.to_value(u.s), samples) * u.s
    positions = []
    for delta_t in times:
        propagated_orbit = propagate_orbit_state(orbit, delta_t, include_j2=include_j2)
        positions.append(propagated_orbit.r.to_value(u.km))
    return np.asarray(positions, dtype=float)


def propagate_orbit(
    orbit: Orbit,
    minutes: float = 90,
    samples: int = 500,
    include_j2: bool = True,
) -> np.ndarray:
    """Propagate an orbit and return sampled Cartesian positions in kilometers."""

    duration = minutes * u.min
    return sample_orbit_positions(
        orbit,
        duration=duration,
        samples=samples,
        include_j2=include_j2,
    )
