from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CalorieEstimate:
    gross_kcal: float
    active_kcal: float
    gross_range_kcal: tuple[float, float]
    active_range_kcal: tuple[float, float]
    weight_kg: float
    duration_s: float
    met: float
    met_range: tuple[float, float]
    met_source: str
    duration_source: str
    method: str = "MET * kg * hours; active uses max(MET - 1, 0)"
    limitation: str = "Scenario estimate; MET range is not a confidence interval; no gait-specific calibration"

    def to_dict(self):
        return asdict(self)


def estimate_calories(*, weight_kg: float, duration_s: float, met: float,
                      met_range: tuple[float, float] | None = None,
                      met_source: str = "user_supplied",
                      duration_source: str = "user_supplied") -> CalorieEstimate:
    if not all(math.isfinite(x) for x in (weight_kg, duration_s, met)):
        raise ValueError("Energy inputs must be finite")
    if weight_kg <= 0 or duration_s < 0 or met <= 0:
        raise ValueError("Weight and MET must be positive; duration must be nonnegative")
    low, high = met_range if met_range is not None else (met, met)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= met <= high):
        raise ValueError("MET range must be positive and contain the chosen MET")
    factor = weight_kg * duration_s / 3600
    return CalorieEstimate(
        met * factor, max(met - 1, 0) * factor,
        (low * factor, high * factor), (max(low - 1, 0) * factor, max(high - 1, 0) * factor),
        weight_kg, duration_s, met, (low, high), met_source, duration_source,
    )
