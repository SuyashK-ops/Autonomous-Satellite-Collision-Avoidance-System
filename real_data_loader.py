"""Utilities for fetching and propagating real orbital data from CelesTrak."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from sgp4.api import Satrec, jday


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"


@dataclass
class TLERecord:
    """Container for a single TLE entry and its parsed propagator."""

    name: str
    line1: str
    line2: str
    satrec: Satrec


def build_celestrak_url(group: str = "stations", fmt: str = "tle") -> str:
    """Build a public CelesTrak GP query URL."""

    query = urlencode({"GROUP": group, "FORMAT": fmt})
    return f"{CELESTRAK_GP_URL}?{query}"


def fetch_tle_group(
    group: str = "stations",
    limit: Optional[int] = None,
    timeout: int = 30,
) -> list[TLERecord]:
    """Fetch a CelesTrak group in TLE format and return parsed records."""

    url = build_celestrak_url(group=group, fmt="tle")
    with urlopen(url, timeout=timeout) as response:
        text = response.read().decode("utf-8")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[TLERecord] = []

    for index in range(0, len(lines), 3):
        if index + 2 >= len(lines):
            break
        name, line1, line2 = lines[index : index + 3]
        records.append(
            TLERecord(
                name=name,
                line1=line1,
                line2=line2,
                satrec=Satrec.twoline2rv(line1, line2),
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def _to_utc_datetime(when: Optional[datetime]) -> datetime:
    """Normalize datetimes to UTC."""

    if when is None:
        return datetime.now(UTC)
    if when.tzinfo is None:
        return when.replace(tzinfo=UTC)
    return when.astimezone(UTC)


def propagate_tle_record(
    record: TLERecord,
    when: Optional[datetime] = None,
) -> dict:
    """Propagate one TLE object to the requested time."""

    timestamp = _to_utc_datetime(when)
    jd, fr = jday(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second + timestamp.microsecond / 1e6,
    )
    error_code, position_km, velocity_km_s = record.satrec.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(
            f"SGP4 propagation failed for {record.name} with code {error_code}."
        )

    return {
        "name": record.name,
        "timestamp_utc": timestamp,
        "position_km": np.asarray(position_km, dtype=float),
        "velocity_km_s": np.asarray(velocity_km_s, dtype=float),
        "tle_line1": record.line1,
        "tle_line2": record.line2,
    }


def propagate_tle_group(
    records: Iterable[TLERecord],
    when: Optional[datetime] = None,
) -> list[dict]:
    """Propagate a group of TLE records to the requested time."""

    timestamp = _to_utc_datetime(when)
    return [propagate_tle_record(record, when=timestamp) for record in records]


def sample_tle_track(
    record: TLERecord,
    minutes: float = 90.0,
    samples: int = 180,
    start_time: Optional[datetime] = None,
) -> np.ndarray:
    """Sample a short SGP4 trajectory arc for a TLE object."""

    start = _to_utc_datetime(start_time)
    offsets = np.linspace(0.0, minutes * 60.0, samples)
    track = []
    for offset_seconds in offsets:
        timestamp = start + timedelta(seconds=float(offset_seconds))
        propagated = propagate_tle_record(record, when=timestamp)
        track.append(propagated["position_km"])
    return np.asarray(track, dtype=float)


def find_closest_pair(propagated_objects: Iterable[dict]) -> dict:
    """Find the closest pair in a propagated object list."""

    objects = list(propagated_objects)
    if len(objects) < 2:
        raise ValueError("At least two propagated objects are required.")

    min_distance = float("inf")
    closest_pair: tuple[dict, dict] | None = None

    for i, first in enumerate(objects[:-1]):
        first_position = first["position_km"]
        for second in objects[i + 1 :]:
            distance = float(np.linalg.norm(first_position - second["position_km"]))
            if distance < min_distance:
                min_distance = distance
                closest_pair = (first, second)

    assert closest_pair is not None
    return {
        "object_1": closest_pair[0]["name"],
        "object_2": closest_pair[1]["name"],
        "distance_km": min_distance,
        "timestamp_utc": closest_pair[0]["timestamp_utc"],
    }
