from dataclasses import asdict, dataclass, field
import math
import random

from .models import Event, Gait, Placement, Recording, Sample


@dataclass(frozen=True)
class SimulationConfig:
    gait: Gait = Gait.CRUTCH
    placement: Placement = Placement.HAND
    duration_s: float = 30.0
    sample_rate_hz: float = 50.0
    seed: int = 42
    cadence_spm: float | None = None
    noise_std: float = 0.06
    dropout_probability: float = 0.01
    spike_probability: float = 0.001
    timestamp_jitter: float = 0.08
    randomize: bool = True
    activity: str = "walking"
    pauses_s: tuple[tuple[float, float], ...] = ()
    gap_s: tuple[float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "gait", Gait(self.gait))
        object.__setattr__(self, "placement", Placement(self.placement))
        numeric = (self.duration_s, self.sample_rate_hz, self.noise_std,
                   self.dropout_probability, self.spike_probability, self.timestamp_jitter)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Simulation parameters must be finite")
        if not (2 <= self.duration_s <= 3600 and 20 <= self.sample_rate_hz <= 200):
            raise ValueError("Use duration 2–3600 s and sampling rate 20–200 Hz")
        if self.cadence_spm is not None and not 25 <= self.cadence_spm <= 160:
            raise ValueError("Synthetic cadence must be 25–160 steps/min")
        if not (0 <= self.dropout_probability <= 0.2 and 0 <= self.spike_probability <= 0.05):
            raise ValueError("Invalid dropout/spike probability")
        if self.noise_std < 0 or not 0 <= self.timestamp_jitter <= 0.3:
            raise ValueError("Invalid noise or timestamp jitter")
        if self.activity not in ("walking", "stationary", "hand_motion"):
            raise ValueError("Unknown activity")
        for start, end in (*self.pauses_s, *((self.gap_s,) if self.gap_s else ())):
            if not 0 <= start < end <= self.duration_s:
                raise ValueError("Pause/gap must be inside recording duration")


@dataclass
class Simulation:
    recording: Recording
    events: list[Event]
    metadata: dict = field(default_factory=dict)


def _rotate(v, angles):
    x, y, z = v
    for axis, angle in enumerate(angles):
        c, s = math.cos(angle), math.sin(angle)
        if axis == 0:
            y, z = c * y - s * z, s * y + c * z
        elif axis == 1:
            x, z = c * x + s * z, -s * x + c * z
        else:
            x, y = c * x - s * y, s * x + c * y
    return x, y, z


def simulate(config: SimulationConfig | None = None) -> Simulation:
    cfg = config or SimulationConfig()
    rng = random.Random(cfg.seed)
    defaults = {Gait.NORMAL: (105, 0.04), Gait.CRUTCH: (70, 0.12),
                Gait.KNEE: (76, 0.18), Gait.HIP: (56, 0.26)}
    nominal, asymmetry = defaults[cfg.gait]
    cadence = cfg.cadence_spm if cfg.cadence_spm is not None else nominal
    if cfg.randomize:
        if cfg.cadence_spm is None:
            cadence *= rng.uniform(0.8, 1.15)
        asymmetry *= rng.uniform(0.6, 1.2)
    amplitude = rng.uniform(0.8, 1.4) if cfg.randomize else 1.1
    angles = tuple(rng.uniform(-math.pi, math.pi) for _ in range(3)) if cfg.randomize else (0, 0, 0)
    bias = tuple(rng.uniform(-0.12, 0.12) for _ in range(3)) if cfg.randomize else (0, 0, 0)
    events, contacts = [], []
    if cfg.activity == "walking":
        t, index = 1.0, 0
        while t < cfg.duration_s - 0.7:
            if not any(a <= t <= b for a, b in cfg.pauses_s):
                side = "left" if index % 2 == 0 else "right"
                events.append(Event(t, "foot_contact", side))
                contacts.append((t, index))
                if cfg.placement != Placement.THIGH or index % 2 == 0:
                    events.append(Event(t, "motion_cycle"))
                if cfg.gait == Gait.CRUTCH:
                    events.append(Event(t - 0.19, "crutch_contact"))
            variation = rng.gauss(0, 0.025) if cfg.randomize else 0
            interval = 60 / cadence * (1 + (asymmetry if index % 2 == 0 else -asymmetry) + variation)
            t += interval
            index += 1
    n = math.ceil(cfg.duration_s * cfg.sample_rate_hz)
    pulse = [0.0] * n
    lateral = [0.0] * n
    gyro = [0.0] * n
    for t, index in contacts:
        primary = cfg.placement != Placement.THIGH or index % 2 == 0
        size = amplitude * (1 if index % 2 == 0 else 0.75)
        size *= 1 if primary else 0.05
        width = {Placement.HAND: 0.075, Placement.THIGH: 0.095,
                 Placement.WAIST: 0.07, Placement.TRUNK: 0.075}[cfg.placement]
        lo = max(0, int((t - 0.5) * cfg.sample_rate_hz))
        hi = min(n, int((t + 0.5) * cfg.sample_rate_hz) + 1)
        for i in range(lo, hi):
            offset = i / cfg.sample_rate_hz - t
            bump = math.exp(-0.5 * (offset / width) ** 2)
            pulse[i] += 1.8 * size * bump
            lateral[i] += (1 if index % 2 == 0 else -1) * size * bump * 0.4
            gyro[i] += (-offset / width) * bump * size * (2 if cfg.placement == Placement.THIGH else 0.6)
        if cfg.gait == Gait.CRUTCH:
            c = t - 0.19
            for i in range(max(0, int((c - 0.08) * cfg.sample_rate_hz)),
                           min(n, int((c + 0.08) * cfg.sample_rate_hz) + 1)):
                pulse[i] += 0.7 * amplitude * math.exp(-0.5 * ((i / cfg.sample_rate_hz - c) / 0.012) ** 2)
    samples = []
    for i in range(n):
        t = (i + rng.uniform(-cfg.timestamp_jitter, cfg.timestamp_jitter)) / cfg.sample_rate_hz
        if (cfg.gap_s and cfg.gap_s[0] <= t <= cfg.gap_s[1]) or rng.random() < cfg.dropout_probability:
            continue
        z = 9.80665 + pulse[i] + 0.10 * math.sin(0.4 * t)
        if cfg.activity == "hand_motion":
            z += 1.8 * max(0, math.sin(2 * math.pi * 1.3 * t)) ** 6
        x = lateral[i]
        y = 0.1 * pulse[i]
        if cfg.gait == Gait.HIP:
            x += 0.14 * math.sin(2 * math.pi * cadence / 120 * t)
        # 旋转重力和运动分量后再加入设备偏置。 / Rotate gravity and motion together.
        a = _rotate((x, y, z), angles)
        a = [v + b + rng.gauss(0, cfg.noise_std) for v, b in zip(a, bias)]
        if rng.random() < cfg.spike_probability:
            a[rng.randrange(3)] += rng.choice((-1, 1)) * rng.uniform(6, 12)
        g = _rotate((gyro[i], 0.1 * gyro[i], 0), angles)
        g = [v + rng.gauss(0, 0.015) for v in g]
        samples.append(Sample(t, *a, *g))
    metadata = {"config": asdict(cfg), "realized_cadence_spm": cadence,
                "asymmetry": asymmetry, "amplitude": amplitude,
                "orientation_rad": angles, "bias_m_s2": bias,
                "event_mode": "stride" if cfg.placement == Placement.THIGH else "step",
                "assumptions": "Phenomenological alternating gait with prosthetic foot contacts; not a biomechanical model"}
    return Simulation(Recording(tuple(samples), cfg.placement, "synthetic"),
                      sorted(events, key=lambda e: e.time_s), metadata)
