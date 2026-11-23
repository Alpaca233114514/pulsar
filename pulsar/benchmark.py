"""Subject-grouped ID/OOD/sealed benchmark generator and baseline prediction exporter."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

from .algorithm import DetectorConfig, StepCounter
from .denoise import hampel, preprocess
from .io import load_recording, save_simulation, write_json
from .models import Event, Gait, Placement, Recording
from .simulator import SimulationConfig, simulate

TRUTH_FORMAT = "pulsar.benchmark_truth.v1"
PREDICTION_FORMAT = "pulsar.predictions.v1"
FEATURE_NAMES = ("mean_abs", "std", "rms", "peak_to_peak")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth_active_duration_s(events: list[Event], duration_s: float) -> float:
    contacts = sorted(e.time_s for e in events if e.kind == "foot_contact")
    if len(contacts) < 2:
        return 0.0
    groups: list[list[float]] = []
    for t in contacts:
        if not groups or t - groups[-1][-1] > 3.5:
            groups.append([])
        groups[-1].append(t)
    total = 0.0
    for group in groups:
        if len(group) < 2:
            continue
        intervals = [b - a for a, b in zip(group, group[1:])]
        pad = median(intervals) / 2.0
        start = max(0.0, group[0] - pad)
        end = min(duration_s, group[-1] + pad)
        total += max(0.0, end - start)
    return total


def _extract_windows_causal(recording: Recording, events: list[Event], session_id: str,
                            window_s: float = 5.0) -> list[dict]:
    """Extract non-overlapping windows strictly inside each window to prevent boundary leakage."""
    samples = recording.samples
    if len(samples) < 20:
        return []
    origin = samples[0].t
    end_limit = samples[-1].t
    windows = []
    idx = 0
    cursor = 0
    n = len(samples)
    while origin + (idx + 1) * window_s <= end_limit + 1e-6:
        a = origin + idx * window_s
        b = a + window_s
        while cursor < n and samples[cursor].t < a:
            cursor += 1
        j = cursor
        chunk = []
        while j < n and samples[j].t < b:
            chunk.append(samples[j])
            j += 1
        idx += 1
        if len(chunk) < 20 or chunk[-1].t - chunk[0].t < window_s * 0.75:
            continue
        # 仅在当前窗口内部做尖峰抑制和去均值，杜绝邻窗泄漏。 / Strictly intra-window despike & centering.
        mags = [math.hypot(s.ax, s.ay, s.az) for s in chunk]
        clean, _ = hampel(mags)
        center = mean(clean)
        centered = [v - center for v in clean]
        target = sum(1 for e in events if e.kind == "foot_contact" and a <= e.time_s < b)
        windows.append({
            "window_id": f"{session_id}:w{len(windows):03d}",
            "start_s": a,
            "end_s": b,
            "mean_abs": mean(abs(v) for v in centered),
            "std": pstdev(centered),
            "rms": (mean(v * v for v in centered)) ** 0.5,
            "peak_to_peak": max(centered) - min(centered),
            "target_steps": target,
        })
    return windows


def _extract_windows_offline(recording: Recording, events: list[Event], session_id: str,
                             window_s: float = 5.0) -> list[dict]:
    data = preprocess(recording)
    windows = []
    for seg in data.segments:
        width = round(window_s * seg.sample_rate_hz)
        if width < 2:
            continue
        for start in range(0, len(seg.times) - width + 1, width):
            a = seg.times[start]
            b = seg.times[start + width - 1] + 1.0 / seg.sample_rate_hz
            values = seg.filtered[start:start + width]
            target = sum(1 for e in events if e.kind == "foot_contact" and a <= e.time_s < b)
            windows.append({
                "window_id": f"{session_id}:w{len(windows):03d}",
                "start_s": a,
                "end_s": b,
                "mean_abs": mean(abs(v) for v in values),
                "std": pstdev(values),
                "rms": (mean(v * v for v in values)) ** 0.5,
                "peak_to_peak": max(values) - min(values),
                "target_steps": target,
            })
    return windows


def _tier_configs(tier: str, subject_index: int, duration_s: float, seed: int) -> list[tuple[str, SimulationConfig]]:
    gaits = (Gait.CRUTCH, Gait.KNEE, Gait.HIP)
    gait = gaits[subject_index % 3]
    base_seed = seed + subject_index * 1000
    if tier in ("id_dev", "id_val"):
        rate = (25.0, 50.0, 100.0)[subject_index % 3]
        noise = (0.05, 0.10, 0.16)[subject_index % 3]
        neg_activity = "stationary" if subject_index % 2 == 0 else "hand_motion"
        return [
            ("s00", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed, noise_std=noise,
                                     spike_probability=0.002)),
            ("s01", SimulationConfig(gait=gait, placement=Placement.THIGH, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed + 1, noise_std=noise,
                                     spike_probability=0.002)),
            ("s02", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed + 2, noise_std=noise,
                                     spike_probability=0.002, activity=neg_activity)),
        ]
    if tier == "sealed_test":
        rate = (30.0, 50.0, 80.0)[subject_index % 3]
        noise = (0.08, 0.14, 0.20)[subject_index % 3]
        neg_activity = "stationary" if subject_index % 2 == 0 else "hand_motion"
        return [
            ("s00", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed, noise_std=noise,
                                     spike_probability=0.003, pauses_s=((duration_s * 0.45, duration_s * 0.57),))),
            ("s01", SimulationConfig(gait=gait, placement=Placement.THIGH, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed + 1, noise_std=noise,
                                     spike_probability=0.003)),
            ("s02", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                     sample_rate_hz=rate, seed=base_seed + 2, noise_std=noise,
                                     activity=neg_activity,
                                     hand_motion_freq_hz=(1.05, 1.55)[subject_index % 2])),
        ]
    # ood_stress：故意打破训练集固定频段、脉冲形态、采样率与连续性假设。
    # OOD stress: break fixed training assumptions (freq, pulse shape, unseen Hz, gaps, bursts).
    rate = (33.0, 60.0, 75.0)[subject_index % 3]
    noise = (0.22, 0.28, 0.34)[subject_index % 3]
    shape = ("skewed", "double_peak")[subject_index % 2]
    hm_mode = ("chirp", "burst", "irregular")[subject_index % 3]
    hm_freq = (0.7, 1.0, 1.7, 2.2)[subject_index % 4]
    other_neg = "stationary" if subject_index % 2 == 0 else "non_walking_other"
    pause_or_gap_hand = ({"pauses_s": ((duration_s * 0.25, duration_s * 0.42), (duration_s * 0.65, duration_s * 0.78))}
                         if subject_index % 2 == 0 else
                         {"gap_s": (duration_s * 0.35, duration_s * 0.35 + min(3.0, duration_s * 0.15))})
    return [
        ("s00", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                 sample_rate_hz=rate, seed=base_seed, noise_std=noise,
                                 spike_probability=0.005, timestamp_jitter=0.16,
                                 pulse_shape=shape, crutch_offset_jitter_s=0.06,
                                 **pause_or_gap_hand)),
        ("s01", SimulationConfig(gait=gait, placement=Placement.THIGH, duration_s=duration_s,
                                 sample_rate_hz=rate, seed=base_seed + 1, noise_std=noise,
                                 spike_probability=0.005, timestamp_jitter=0.16,
                                 pulse_shape=shape, crutch_offset_jitter_s=0.06)),
        ("s02", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                 sample_rate_hz=rate, seed=base_seed + 2, noise_std=noise,
                                 activity="hand_motion", hand_motion_mode=hm_mode,
                                 hand_motion_freq_hz=hm_freq)),
        ("s03", SimulationConfig(gait=gait, placement=Placement.HAND, duration_s=duration_s,
                                 sample_rate_hz=rate, seed=base_seed + 3, noise_std=noise,
                                 activity=other_neg)),
    ]


def generate_benchmark(
    directory: str | Path,
    *,
    train_subjects: int = 24,
    val_subjects: int = 12,
    ood_subjects: int = 12,
    test_subjects: int = 12,
    duration_s: float = 25.0,
    seed: int = 20261123,
    feature_mode: str = "causal_window",
) -> dict:
    """Generate disjoint subject-grouped ID/OOD/sealed benchmark tiers and truth files."""
    if feature_mode not in ("causal_window", "offline_centered"):
        raise ValueError("feature_mode must be causal_window or offline_centered")
    counts = {"id_dev": train_subjects, "id_val": val_subjects,
              "ood_stress": ood_subjects, "sealed_test": test_subjects}
    if any(v < 1 for v in counts.values()):
        raise ValueError("Every benchmark tier needs at least 1 subject")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    split_map = {"id_dev": "train", "id_val": "validation", "ood_stress": "ood_stress", "sealed_test": "test"}
    tier_seeds = {"id_dev": seed, "id_val": seed + 10_000_000,
                  "ood_stress": seed + 20_000_000, "sealed_test": seed + 30_000_000}
    extractor = _extract_windows_causal if feature_mode == "causal_window" else _extract_windows_offline
    all_window_rows = []
    truth_files = {}
    tier_stats = {}
    for tier, n_subjects in counts.items():
        split = split_map[tier]
        sessions_truth = []
        for s_idx in range(n_subjects):
            subject_id = f"synth_{tier}_{s_idx:03d}"
            for suffix, cfg in _tier_configs(tier, s_idx, duration_s, tier_seeds[tier]):
                session_id = f"{subject_id}_{suffix}"
                sim = simulate(cfg)
                rel_dir = Path(tier) / session_id
                save_simulation(sim, root / rel_dir)
                wins = extractor(sim.recording, sim.events, session_id)
                foot_times = [e.time_s for e in sim.events if e.kind == "foot_contact"]
                cycle_times = [e.time_s for e in sim.events if e.kind == "motion_cycle"]
                sessions_truth.append({
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "split": split,
                    "tier": tier,
                    "gait": str(cfg.gait),
                    "placement": str(cfg.placement),
                    "activity": cfg.activity,
                    "duration_s": sim.recording.duration_s,
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "truth_steps": len(foot_times),
                    "active_duration_s": _truth_active_duration_s(sim.events, sim.recording.duration_s),
                    "foot_contact_times_s": foot_times,
                    "motion_cycle_times_s": cycle_times,
                    "sensors_relpath": str((rel_dir / "sensors.csv").as_posix()),
                    "windows": [
                        {k: w[k] for k in ("window_id", "start_s", "end_s", "target_steps")} | {"activity": cfg.activity}
                        for w in wins
                    ],
                })
                for w in wins:
                    all_window_rows.append({
                        "tier": tier, "split": split, "subject_id": subject_id,
                        "session_id": session_id, "gait": str(cfg.gait),
                        "placement": str(cfg.placement), "activity": cfg.activity,
                        **w,
                    })
        truth_doc = {
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "benchmark_id": f"pulsar-{tier}-seed{seed}",
            "source": "synthetic",
            "tier": tier,
            "sealed": tier == "sealed_test",
            "feature_context": feature_mode,
            "split_policy": "disjoint_subject_groups_per_tier",
            "sessions": sessions_truth,
        }
        truth_path = root / f"truth_{tier}.json"
        write_json(truth_path, truth_doc)
        truth_files[tier] = {"file": truth_path.name, "sha256": _sha256_file(truth_path)}
        tier_stats[tier] = {
            "subjects": n_subjects,
            "sessions": len(sessions_truth),
            "windows": sum(len(s["windows"]) for s in sessions_truth),
        }
    with (root / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["tier", "split", "subject_id", "session_id", "gait", "placement", "activity",
                  "window_id", "start_s", "end_s", *FEATURE_NAMES, "target_steps"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_window_rows)
    summary = {
        "source": "synthetic_only",
        "seed": seed,
        "duration_s": duration_s,
        "feature_context": feature_mode,
        "training_features": list(FEATURE_NAMES),
        "tiers": tier_stats,
        "truth_files": truth_files,
        "limitation": "Synthetic benchmark only; real human gait validation requires separately labeled real recordings",
    }
    write_json(root / "summary.json", summary)
    return summary


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Small dense Gaussian elimination with partial pivoting (pure stdlib)."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular normal matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        factor = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= factor
        for r in range(n):
            if r != col:
                scale = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= scale * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _fit_window_ridge(train_rows: list[dict], l2: float = 1.0) -> tuple[list[float], list[float], list[float]]:
    """Fit L2-regularized linear regression strictly on train split features."""
    xs = [[float(row[name]) for name in FEATURE_NAMES] for row in train_rows]
    ys = [float(row["target_steps"]) for row in train_rows]
    n = len(xs)
    d = len(FEATURE_NAMES)
    means = [sum(row[j] for row in xs) / n for j in range(d)]
    stds = [max(1e-8, (sum((row[j] - means[j]) ** 2 for row in xs) / n) ** 0.5) for j in range(d)]
    design = [[1.0] + [(row[j] - means[j]) / stds[j] for j in range(d)] for row in xs]
    m = d + 1
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for row, y in zip(design, ys):
        for i in range(m):
            xty[i] += row[i] * y
            for j in range(m):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, m):
        xtx[i][i] += l2
    weights = _solve_linear_system(xtx, xty)
    return weights, means, stds


def export_baseline_predictions(
    benchmark_dir: str | Path,
    truth_file: str | Path,
    output_path: str | Path,
    *,
    baseline: str = "step_counter",
    model_id: str | None = None,
) -> dict:
    """Run a baseline on a benchmark truth split and export pulsar.predictions.v1 JSON."""
    if baseline not in ("step_counter", "zero", "window_ridge"):
        raise ValueError("baseline must be step_counter, zero or window_ridge")
    root = Path(benchmark_dir)
    truth_path = Path(truth_file)
    if not truth_path.is_absolute() and not truth_path.exists():
        truth_path = root / truth_file
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {out}")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if truth.get("format") != TRUTH_FORMAT:
        raise ValueError(f"Expected {TRUTH_FORMAT} truth file")
    truth_sha = _sha256_file(truth_path)
    sessions_out = []
    windows_out = []
    if baseline == "step_counter":
        granularity = "event"
        family = "rule_baseline"
        feature_context = "offline_centered"
        for item in truth["sessions"]:
            rec = load_recording(root / item["sensors_relpath"],
                                 placement=item["placement"], source=truth["source"])
            # 推理只吃传感器与步态/位置声明，绝不接触标签。 / Inference uses sensor stream only.
            result = StepCounter(DetectorConfig(gait=item["gait"], placement=item["placement"])).analyze(rec)
            sessions_out.append({
                "session_id": item["session_id"],
                "predicted_steps": result.step_count,
                "active_duration_s": result.active_duration_s,
                "event_times_s": result.event_times_s,
            })
    elif baseline == "zero":
        granularity = "window"
        family = "naive_baseline"
        feature_context = truth["feature_context"]
        for item in truth["sessions"]:
            for w in item["windows"]:
                windows_out.append({
                    "window_id": w["window_id"],
                    "session_id": item["session_id"],
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "predicted_steps": 0.0,
                })
    else:
        granularity = "window"
        family = "window_ml"
        feature_context = truth["feature_context"]
        with (root / "windows.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        train_rows = [r for r in rows if r["split"] == "train"]
        if not train_rows:
            raise ValueError("windows.csv has no train split rows for window_ridge")
        weights, means, stds = _fit_window_ridge(train_rows, l2=1.0)
        by_window = {r["window_id"]: r for r in rows}
        for item in truth["sessions"]:
            for w in item["windows"]:
                row = by_window[w["window_id"]]
                feats = [(float(row[name]) - means[i]) / stds[i] for i, name in enumerate(FEATURE_NAMES)]
                pred = max(0.0, weights[0] + sum(w_i * x_i for w_i, x_i in zip(weights[1:], feats)))
                windows_out.append({
                    "window_id": w["window_id"],
                    "session_id": item["session_id"],
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "predicted_steps": round(pred, 6),
                })
    payload = {
        "schema_version": 1,
        "format": PREDICTION_FORMAT,
        "model_id": model_id or f"pulsar_{baseline}_v1",
        "model_family": family,
        "prediction_granularity": granularity,
        "feature_context": feature_context,
        "source": truth["source"],
        "truth_benchmark_id": truth["benchmark_id"],
        "truth_sha256": truth_sha,
        "training_config_run_id": None,
        "sessions": sessions_out,
        "windows": windows_out,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, payload)
    return {
        "output": str(out.resolve()),
        "model_id": payload["model_id"],
        "baseline": baseline,
        "truth_benchmark_id": truth["benchmark_id"],
        "truth_sha256": truth_sha,
        "sessions": len(sessions_out),
        "windows": len(windows_out),
    }
