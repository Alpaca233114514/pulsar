from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math


class Gait(StrEnum):
    NORMAL = "normal"
    CRUTCH = "crutch_assisted"
    KNEE = "knee_disarticulation"
    HIP = "hip_disarticulation"


class Placement(StrEnum):
    HAND = "hand"
    THIGH = "thigh"
    WAIST = "waist"
    TRUNK = "trunk"


@dataclass(frozen=True)
class Sample:
    t: float
    ax: float
    ay: float
    az: float
    gx: float | None = None
    gy: float | None = None
    gz: float | None = None

    def __post_init__(self):
        values = (self.t, self.ax, self.ay, self.az)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Timestamp and acceleration must be finite")
        gyro = (self.gx, self.gy, self.gz)
        if any(v is not None for v in gyro):
            if not all(v is not None and math.isfinite(v) for v in gyro):
                raise ValueError("Provide all three finite gyro axes, or none")


@dataclass(frozen=True)
class Recording:
    samples: tuple[Sample, ...]
    placement: Placement
    source: str = "real"
    acceleration_kind: str = "with_gravity"

    def __post_init__(self):
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "placement", Placement(self.placement))
        if self.source not in ("real", "synthetic"):
            raise ValueError("source must be real or synthetic")
        if self.acceleration_kind != "with_gravity":
            raise ValueError("This baseline requires raw acceleration including gravity")
        if any(b.t <= a.t for a, b in zip(self.samples, self.samples[1:])):
            raise ValueError("Timestamps must be strictly increasing; duplicates are invalid")
        if self.samples and any(
            (s.gx is None) != (self.samples[0].gx is None) for s in self.samples
        ):
            raise ValueError("Gyroscope availability must be consistent within a recording")

    @property
    def duration_s(self) -> float:
        return self.samples[-1].t - self.samples[0].t if len(self.samples) > 1 else 0.0


@dataclass(frozen=True)
class Event:
    time_s: float
    kind: str
    side: str | None = None


@dataclass
class DetectionResult:
    gait: str
    placement: str
    source: str
    event_mode: str
    step_count: int
    event_times_s: list[float]
    bouts_s: list[tuple[float, float]]
    active_duration_s: float
    observed_duration_s: float
    recording_duration_s: float
    cadence_spm: float | None
    interval_cv: float | None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
