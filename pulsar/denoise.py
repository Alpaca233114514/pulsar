"""Timestamp-aware resampling, Hampel despiking, smoothing and baseline removal."""

from dataclasses import dataclass
import math
from statistics import median

from .models import Recording


def moving_average(values: list[float], radius: int) -> list[float]:
    sums = [0.0]
    for value in values:
        sums.append(sums[-1] + value)
    n = len(values)
    return [
        (sums[min(n, i + radius + 1)] - sums[max(0, i - radius)])
        / (min(n, i + radius + 1) - max(0, i - radius))
        for i in range(n)
    ]


def hampel(values: list[float], radius: int = 3) -> tuple[list[float], int]:
    output = values.copy()
    changed = 0
    for i, value in enumerate(values):
        window = values[max(0, i - radius):i + radius + 1]
        center = median(window)
        mad = median(abs(v - center) for v in window)
        if abs(value - center) > max(0.6, 6 * 1.4826 * mad):
            output[i] = center
            changed += 1
    return output, changed


@dataclass
class SignalSegment:
    times: list[float]
    raw: list[float]
    despiked: list[float]
    filtered: list[float]
    sample_rate_hz: float
    repaired_spikes: int


@dataclass
class Preprocessed:
    segments: list[SignalSegment]
    gaps_s: list[tuple[float, float]]
    warnings: list[str]


def preprocess(recording: Recording, *, smooth_s: float = 0.08,
               baseline_s: float = 2.5) -> Preprocessed:
    samples = recording.samples
    if len(samples) < 3:
        return Preprocessed([], [], ["insufficient_samples"])
    dt = median(b.t - a.t for a, b in zip(samples, samples[1:]))
    rate = 1 / dt
    if rate < 20:
        raise ValueError("At least 20 Hz is required; 50–100 Hz is recommended")
    if rate > 1000:
        raise ValueError("Sampling rate exceeds 1000 Hz; check timestamp units")
    gap_limit = max(0.15, 3.5 * dt)
    groups = [[samples[0]]]
    gaps = []
    for sample in samples[1:]:
        if sample.t - groups[-1][-1].t > gap_limit:
            gaps.append((groups[-1][-1].t, sample.t))
            groups.append([])
        groups[-1].append(sample)
    segments = []
    for group in groups:
        if len(group) < 3 or group[-1].t - group[0].t < 0.2:
            continue
        times = [group[0].t + i * dt for i in range(
            int((group[-1].t - group[0].t) / dt) + 1
        )]
        magnitudes = [math.hypot(s.ax, s.ay, s.az) for s in group]
        raw = []
        j = 0
        for t in times:
            while j + 1 < len(group) - 1 and group[j + 1].t < t:
                j += 1
            fraction = (t - group[j].t) / (group[j + 1].t - group[j].t)
            raw.append(magnitudes[j] * (1 - fraction) + magnitudes[j + 1] * fraction)
        clean, changed = hampel(raw)
        smooth = moving_average(clean, max(1, round(smooth_s * rate / 2)))
        baseline = moving_average(smooth, max(1, round(baseline_s * rate / 2)))
        filtered = [v - b for v, b in zip(smooth, baseline)]
        segments.append(SignalSegment(times, raw, clean, filtered, rate, changed))
    warnings = ["long_gaps_split_without_interpolation"] if gaps else []
    if not segments:
        warnings.append("no_usable_continuous_segment")
    return Preprocessed(segments, gaps, warnings)
