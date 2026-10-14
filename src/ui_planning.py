"""Transparent arithmetic for fuel estimates based on recorded trip inputs."""
from math import isfinite


def calculate_fuel_estimate(
    distance_km: float | None,
    efficiency_km_per_litre: float | None,
) -> float:
    """Return distance / efficiency in litres; callers must supply source values."""
    values = {
        "Distance (km)": distance_km,
        "Vehicle efficiency (km/L)": efficiency_km_per_litre,
    }
    for label, value in values.items():
        if value is None:
            raise ValueError(f"{label} is required to calculate an estimate.")
        if not isfinite(float(value)):
            raise ValueError(f"{label} must be a finite number.")
        if value <= 0:
            raise ValueError(f"{label} must be greater than zero.")

    return float(distance_km) / float(efficiency_km_per_litre)
