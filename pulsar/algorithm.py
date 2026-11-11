from dataclasses import dataclass
import math
from statistics import median, pstdev

from .denoise import preprocess
from .models import DetectionResult, Gait, Placement, Recording


@dataclass(frozen=True)
class DetectorConfig:
    gait: Gait = Gait.NORMAL
    placement: Placement = Placement.HAND
    event_mode: str = "auto"
    min_cadence_spm: float = 25.0
    max_cadence_spm: float = 170.0
    min_bout_events: int = 3
    amplitude_floor: float = 0.18

    def __post_init__(self):
        object.__setattr__(self, "gait", Gait(self.gait))
        object.__setattr__(self, "placement", Placement(self.placement))
        if self.event_mode not in ("auto", "step", "stride"):
            raise ValueError("event_mode must be auto, step or stride")
        if not (0 < self.min_cadence_spm < self.max_cadence_spm <= 240):
            raise ValueError("Invalid cadence limits")
        if self.min_bout_events < 3 or not math.isfinite(self.amplitude_floor) or self.amplitude_floor <= 0:
            raise ValueError("Need at least three bout events and a positive amplitude floor")


class StepCounter:
    """Offline peak/bout baseline; gait and placement are supplied, never diagnosed."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def analyze(self, recording: Recording) -> DetectionResult:
        cfg = self.config
        if recording.placement != cfg.placement:
            raise ValueError("Recording and detector placement must agree")
        mode = cfg.event_mode
        if mode == "auto":
            mode = "stride" if cfg.placement == Placement.THIGH else "step"
        multiplier = 2 if mode == "stride" else 1
        data = preprocess(recording)
        warnings = ["experimental_not_validated_on_real_target_gaits", *data.warnings]
        if mode == "stride":
            warnings.append("two_steps_per_stride_assumes_alternating_gait")
        if cfg.placement == Placement.HAND:
            warnings.append("rhythmic_hand_motion_can_be_miscounted")
        if cfg.gait == Gait.CRUTCH:
            warnings.append("crutch_impacts_may_be_confused_with_steps")
        if recording.samples and median(math.hypot(s.ax, s.ay, s.az) for s in recording.samples) < 3:
            raise ValueError("Acceleration appears gravity-free or in g; supply raw m/s² including gravity")
        min_interval = 60 * multiplier / cfg.max_cadence_spm
        max_interval = 60 * multiplier / cfg.min_cadence_spm
        # 非对称步态允许更宽的间隔；允许参数覆盖。 / Allow wider asymmetric intervals.
        asymmetry_limit = 3.5 if cfg.gait != Gait.NORMAL else 2.5
        event_times, bouts, all_intervals = [], [], []
        thresholds = []
        observed = 0.0
        for seg in data.segments:
            observed += seg.times[-1] - seg.times[0]
            values = seg.filtered
            candidates = []
            for i in range(1, len(values) - 1):
                if not (values[i] > values[i - 1] and values[i] >= values[i + 1]):
                    continue
                radius = round(3 * seg.sample_rate_hz)
                local = values[max(0, i - radius):i + radius + 1]
                level = max(cfg.amplitude_floor, 1.2 * median(abs(v) for v in local))
                if values[i] < level:
                    continue
                r = max(2, round(0.18 * seg.sample_rate_hz))
                prominence = values[i] - max(min(values[max(0, i - r):i + 1]),
                                            min(values[i:min(len(values), i + r + 1)]))
                if prominence >= level * 0.55:
                    candidates.append(i)
                    thresholds.append(level)
            # 幅度优先的不应期抑制拐杖双峰。 / Amplitude-first refractory suppression.
            selected = []
            blocked = set()
            exclusion = max(1, math.ceil(min_interval * seg.sample_rate_hz))
            for i in sorted(candidates, key=lambda k: values[k], reverse=True):
                if i not in blocked:
                    selected.append(i)
                    blocked.update(range(i - exclusion + 1, i + exclusion))
            times = [seg.times[i] for i in sorted(selected)]
            groups = []
            for t in times:
                if not groups or t - groups[-1][-1] > max_interval:
                    groups.append([])
                groups[-1].append(t)
            for group in groups:
                if len(group) < cfg.min_bout_events:
                    continue
                intervals = [b - a for a, b in zip(group, group[1:])]
                if max(intervals) / min(intervals) > asymmetry_limit:
                    continue
                event_times.extend(group)
                all_intervals.extend(intervals)
                half_period = median(intervals) / 2
                bouts.append((max(seg.times[0], group[0] - half_period),
                              min(seg.times[-1], group[-1] + half_period)))
        active = sum(b - a for a, b in bouts)
        cadence = 60 * multiplier * len(all_intervals) / sum(all_intervals) if all_intervals else None
        return DetectionResult(
            str(cfg.gait), str(cfg.placement), recording.source, mode,
            len(event_times) * multiplier, event_times, bouts, active, observed,
            recording.duration_s, cadence,
            pstdev(all_intervals) / (sum(all_intervals) / len(all_intervals)) if all_intervals else None,
            warnings,
            {"repaired_spikes": sum(s.repaired_spikes for s in data.segments),
             "segments": len(data.segments), "gaps_s": data.gaps_s,
             "median_threshold_m_s2": median(thresholds) if thresholds else None,
             "count_basis": "stride_proxy_x2" if multiplier == 2 else "motion_peak_proxy",
             "gyroscope_used_for_detection": False},
        )
