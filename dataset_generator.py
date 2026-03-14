"""Generate labeled collision-avoidance datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from closest_approach import compute_closest_approach
from debris_generator import generate_debris


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_satellite_orbit(rng: np.random.Generator) -> Orbit:
    altitude = rng.uniform(450.0, 900.0) * u.km
    inclination = rng.uniform(0.0, 98.0) * u.deg
    raan = rng.uniform(0.0, 360.0) * u.deg
    argp = rng.uniform(0.0, 360.0) * u.deg
    nu = rng.uniform(0.0, 360.0) * u.deg
    ecc = rng.uniform(0.0, 0.01) * u.one
    semi_major_axis = Earth.R.to(u.km) + altitude

    return Orbit.from_classical(
        Earth,
        semi_major_axis,
        ecc,
        inclination,
        raan,
        argp,
        nu,
    )


def _required_delta_v(min_distance_km: float, collision_threshold_km: float) -> float:
    """Heuristic label for the burn magnitude needed to clear the threshold."""

    shortfall = max(collision_threshold_km - min_distance_km, 0.0)
    return shortfall * 0.1


def generate_conjunction_dataset(
    output_csv: str | Path,
    n_samples: int = 5000,
    time_window: u.Quantity = 45.0 * u.min,
    collision_threshold_km: float = 1.0,
    random_state: Optional[int] = None,
) -> list[dict]:
    """Generate a supervised-learning dataset of conjunction scenarios.

    The saved CSV includes:
    ``relative_position_[xyz]``, ``relative_velocity_[xyz]``,
    ``time_to_conjunction``, ``sat_altitude``, ``debris_altitude``,
    ``collision_risk``, and ``required_delta_v``.
    """

    rng = _rng(random_state)
    rows: list[dict] = []

    for sample_index in range(n_samples):
        satellite_orbit = _random_satellite_orbit(rng)
        debris_orbit = generate_debris(
            n_objects=1,
            epoch=satellite_orbit.epoch,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )[0]

        relative_position = (debris_orbit.r - satellite_orbit.r).to(u.km).value
        relative_velocity = (debris_orbit.v - satellite_orbit.v).to(u.km / u.s).value
        closest_approach = compute_closest_approach(
            satellite_orbit,
            debris_orbit,
            time_window,
        )

        sat_altitude = np.linalg.norm(satellite_orbit.r.to_value(u.km)) - Earth.R.to_value(u.km)
        debris_altitude = np.linalg.norm(debris_orbit.r.to_value(u.km)) - Earth.R.to_value(u.km)
        min_distance = closest_approach["min_distance"]

        rows.append(
            {
                "scenario_id": sample_index,
                "relative_position_x": float(relative_position[0]),
                "relative_position_y": float(relative_position[1]),
                "relative_position_z": float(relative_position[2]),
                "relative_velocity_x": float(relative_velocity[0]),
                "relative_velocity_y": float(relative_velocity[1]),
                "relative_velocity_z": float(relative_velocity[2]),
                "time_to_conjunction": float(closest_approach["tca_offset_s"]),
                "sat_altitude": float(sat_altitude),
                "debris_altitude": float(debris_altitude),
                "collision_risk": int(min_distance <= collision_threshold_km),
                "required_delta_v": float(
                    _required_delta_v(min_distance, collision_threshold_km)
                ),
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "relative_position_x",
        "relative_position_y",
        "relative_position_z",
        "relative_velocity_x",
        "relative_velocity_y",
        "relative_velocity_z",
        "time_to_conjunction",
        "sat_altitude",
        "debris_altitude",
        "collision_risk",
        "required_delta_v",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return rows
