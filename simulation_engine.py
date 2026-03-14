"""Main collision-avoidance simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from astropy import units as u
from poliastro.twobody import Orbit

from closest_approach import compute_closest_approach
from conjunction_detection import detect_conjunction
from debris_generator import generate_debris
from orbit_propagator import propagate_orbit_state
from satellite_maneuver import apply_maneuver


@dataclass
class SimulationConfig:
    """Configuration for a collision-avoidance simulation run."""

    duration: u.Quantity = 24.0 * u.hour
    timestep: u.Quantity = 60.0 * u.s
    debris_count: int = 100
    conjunction_threshold_km: float = 10.0
    maneuver_threshold_km: float = 2.0
    avoidance_delta_v: u.Quantity = 0.25 * u.m / u.s
    closest_approach_window: u.Quantity = 30.0 * u.min
    include_j2: bool = True


class SimulationEngine:
    """Orchestrates orbit propagation, event detection, and avoidance logic."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        debris_orbits: Optional[Sequence[Orbit]] = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self._initial_debris_orbits = list(debris_orbits) if debris_orbits is not None else None

    @staticmethod
    def _position_vector_km(orbit: Orbit) -> np.ndarray:
        return orbit.r.to(u.km).value.astype(float)

    @staticmethod
    def _compute_avoidance_delta_v(orbit: Orbit, magnitude: u.Quantity) -> u.Quantity:
        """Construct a small deterministic burn in the radial direction."""

        radial_direction = orbit.r.to_value(u.km)
        radial_direction = radial_direction / np.linalg.norm(radial_direction)
        return radial_direction * magnitude

    def _initialize_debris(self, satellite_orbit: Orbit) -> list[Orbit]:
        if self._initial_debris_orbits is not None:
            return list(self._initial_debris_orbits)
        return generate_debris(
            n_objects=self.config.debris_count,
            epoch=satellite_orbit.epoch,
        )

    def run(self, satellite_orbit: Orbit) -> dict:
        """Run the simulation and return a structured result log."""

        config = self.config
        total_steps = int(np.floor((config.duration / config.timestep).decompose().value))

        satellite_state = satellite_orbit
        debris_states = self._initialize_debris(satellite_orbit)
        initial_debris_orbits = list(debris_states)

        state_history: list[dict] = []
        conjunction_log: list[dict] = []
        maneuver_log: list[dict] = []

        elapsed = 0.0 * u.s
        for step in range(total_steps + 1):
            sat_position = self._position_vector_km(satellite_state)
            debris_positions = [self._position_vector_km(orbit) for orbit in debris_states]
            events = detect_conjunction(
                sat_position,
                debris_positions,
                config.conjunction_threshold_km,
            )

            state_history.append(
                {
                    "step": step,
                    "epoch": satellite_state.epoch,
                    "elapsed_s": float(elapsed.to_value(u.s)),
                    "satellite_position_km": sat_position.tolist(),
                    "conjunction_count": len(events),
                }
            )

            predicted_events: list[dict] = []
            for event in events:
                debris_index = event["debris_index"]
                ca_result = compute_closest_approach(
                    satellite_state,
                    debris_states[debris_index],
                    config.closest_approach_window,
                    include_j2=config.include_j2,
                )
                enriched_event = {
                    **event,
                    "step": step,
                    "epoch": satellite_state.epoch,
                    **ca_result,
                }
                conjunction_log.append(enriched_event)
                predicted_events.append(enriched_event)

            threat_event = None
            if predicted_events:
                threat_event = min(predicted_events, key=lambda item: item["min_distance"])

            if threat_event and threat_event["min_distance"] <= config.maneuver_threshold_km:
                delta_v = self._compute_avoidance_delta_v(
                    satellite_state,
                    config.avoidance_delta_v,
                )
                satellite_state = apply_maneuver(satellite_state, delta_v)
                maneuver_log.append(
                    {
                        "step": step,
                        "epoch": satellite_state.epoch,
                        "trigger_debris_index": threat_event["debris_index"],
                        "predicted_min_distance_km": threat_event["min_distance"],
                        "delta_v_m_s": float(config.avoidance_delta_v.to_value(u.m / u.s)),
                        "satellite_position_km": sat_position.tolist(),
                    }
                )

            if step == total_steps:
                break

            satellite_state = propagate_orbit_state(
                satellite_state,
                config.timestep,
                include_j2=config.include_j2,
            )
            debris_states = [
                propagate_orbit_state(
                    orbit,
                    config.timestep,
                    include_j2=config.include_j2,
                )
                for orbit in debris_states
            ]
            elapsed += config.timestep

        return {
            "config": config,
            "state_history": state_history,
            "conjunctions": conjunction_log,
            "maneuvers": maneuver_log,
            "initial_debris_orbits": initial_debris_orbits,
            "initial_satellite_orbit": satellite_orbit,
            "final_satellite_orbit": satellite_state,
            "final_debris_orbits": debris_states,
        }
