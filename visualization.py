"""Visualization utilities for the collision-avoidance simulator."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from orbit_propagator import sample_orbit_positions


def _set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    """Set equal scaling on all axes for a 3D scene."""

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    span = np.max(maxs - mins) / 2.0
    span = max(span, Earth.R.to_value(u.km) * 1.2)

    ax.set_xlim(centers[0] - span, centers[0] + span)
    ax.set_ylim(centers[1] - span, centers[1] + span)
    ax.set_zlim(centers[2] - span, centers[2] + span)


def _orbit_track(orbit: Orbit, duration: u.Quantity, samples: int) -> np.ndarray:
    """Sample a propagated orbit track in kilometers."""

    return sample_orbit_positions(
        orbit,
        duration=duration,
        samples=samples,
        include_j2=True,
    )


def _actual_satellite_track(results: dict) -> Optional[np.ndarray]:
    """Return the flown satellite trajectory from the simulation state history."""

    state_history = results.get("state_history", [])
    if not state_history:
        return None
    return np.asarray(
        [entry["satellite_position_km"] for entry in state_history],
        dtype=float,
    )


def plot_earth(ax: plt.Axes, alpha: float = 0.35) -> None:
    """Plot Earth as a sphere in the current 3D axes."""

    radius = Earth.R.to_value(u.km)
    u_grid, v_grid = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    x = radius * np.cos(u_grid) * np.sin(v_grid)
    y = radius * np.sin(u_grid) * np.sin(v_grid)
    z = radius * np.cos(v_grid)

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        color="#4f83cc",
        alpha=alpha,
        linewidth=0,
        shade=True,
    )


def plot_simulation_scene(
    satellite_orbit: Orbit,
    debris_orbits: Sequence[Orbit],
    duration: u.Quantity,
    actual_satellite_track_km: Optional[np.ndarray] = None,
    maneuver_positions_km: Optional[Iterable[Iterable[float]]] = None,
    samples: int = 300,
    max_debris_tracks: int = 25,
    debris_cloud_alpha: float = 0.45,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Earth, satellite tracks, debris tracks, and maneuver points."""

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    plot_earth(ax)

    sat_track = _orbit_track(satellite_orbit, duration, samples)
    ax.plot(
        sat_track[:, 0],
        sat_track[:, 1],
        sat_track[:, 2],
        color="#fb923c",
        linewidth=1.6,
        linestyle="--",
        alpha=0.8,
        label="Nominal orbit",
    )

    if actual_satellite_track_km is not None and len(actual_satellite_track_km):
        ax.plot(
            actual_satellite_track_km[:, 0],
            actual_satellite_track_km[:, 1],
            actual_satellite_track_km[:, 2],
            color="#f97316",
            linewidth=2.5,
            label="Flown path",
        )

    cloud_positions = np.asarray([orbit.r.to_value(u.km) for orbit in debris_orbits], dtype=float)
    if len(cloud_positions):
        ax.scatter(
            cloud_positions[:, 0],
            cloud_positions[:, 1],
            cloud_positions[:, 2],
            s=14,
            color="#22c55e",
            alpha=debris_cloud_alpha,
            label="Debris cloud",
        )

    for debris_orbit in debris_orbits[:max_debris_tracks]:
        debris_track = _orbit_track(debris_orbit, duration, max(80, samples // 2))
        ax.plot(
            debris_track[:, 0],
            debris_track[:, 1],
            debris_track[:, 2],
            color="#60a5fa",
            linewidth=0.8,
            alpha=0.22,
        )

    if maneuver_positions_km:
        maneuver_points = np.asarray(list(maneuver_positions_km), dtype=float)
        if len(maneuver_points):
            ax.scatter(
                maneuver_points[:, 0],
                maneuver_points[:, 1],
                maneuver_points[:, 2],
                s=80,
                color="#ef4444",
                marker="^",
                label="Maneuver locations",
            )

    all_points = [sat_track]
    if actual_satellite_track_km is not None and len(actual_satellite_track_km):
        all_points.append(actual_satellite_track_km)
    if len(cloud_positions):
        all_points.append(cloud_positions)
    if maneuver_positions_km:
        maneuver_points = np.asarray(list(maneuver_positions_km), dtype=float)
        if len(maneuver_points):
            all_points.append(maneuver_points)

    _set_equal_axes(ax, np.vstack(all_points))
    ax.set_title("Autonomous Satellite Collision Avoidance Scene")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(loc="upper left")

    return fig, ax


def plot_maneuver_timeline(results: dict) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot conjunction counts and maneuver-trigger distances over time."""

    state_history = results.get("state_history", [])
    maneuver_log = results.get("maneuvers", [])

    elapsed_minutes = [entry["elapsed_s"] / 60.0 for entry in state_history]
    conjunction_counts = [entry["conjunction_count"] for entry in state_history]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(elapsed_minutes, conjunction_counts, color="#2563eb", linewidth=1.8)
    axes[0].set_ylabel("Conjunction count")
    axes[0].set_title("Conjunction Activity and Avoidance Timeline")
    axes[0].grid(alpha=0.3)

    if maneuver_log:
        maneuver_times = [
            next(
                entry["elapsed_s"] / 60.0
                for entry in state_history
                if entry["step"] == maneuver["step"]
            )
            for maneuver in maneuver_log
        ]
        maneuver_distances = [
            maneuver["predicted_min_distance_km"] for maneuver in maneuver_log
        ]
        axes[1].scatter(
            maneuver_times,
            maneuver_distances,
            color="#dc2626",
            s=55,
        )
    axes[1].set_xlabel("Elapsed time (minutes)")
    axes[1].set_ylabel("Predicted min distance (km)")
    axes[1].grid(alpha=0.3)

    return fig, axes


def plot_results_overview(
    results: dict,
    duration: Optional[u.Quantity] = None,
    samples: int = 300,
    max_debris_tracks: int = 25,
) -> tuple[tuple[plt.Figure, plt.Axes], tuple[plt.Figure, Sequence[plt.Axes]]]:
    """Create the main 3D scene and a maneuver timeline from simulation results."""

    satellite_orbit = results.get("initial_satellite_orbit")
    if satellite_orbit is None:
        satellite_orbit = results.get("final_satellite_orbit")
    if satellite_orbit is None:
        raise KeyError(
            "results must contain 'initial_satellite_orbit' or 'final_satellite_orbit'."
        )

    debris_orbits = results.get("initial_debris_orbits")
    if debris_orbits is None:
        debris_orbits = results.get("final_debris_orbits")
    if debris_orbits is None:
        raise KeyError(
            "results must contain 'initial_debris_orbits' or 'final_debris_orbits'."
        )

    config = results["config"]
    scene_duration = duration or config.duration
    actual_track = _actual_satellite_track(results)
    maneuver_positions = [
        entry["satellite_position_km"] for entry in results.get("maneuvers", [])
    ]

    scene = plot_simulation_scene(
        satellite_orbit=satellite_orbit,
        debris_orbits=debris_orbits,
        duration=scene_duration,
        actual_satellite_track_km=actual_track,
        maneuver_positions_km=maneuver_positions,
        samples=samples,
        max_debris_tracks=max_debris_tracks,
    )
    timeline = plot_maneuver_timeline(results)
    return scene, timeline
