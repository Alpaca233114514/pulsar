import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from .denoise import preprocess
from .io import save_simulation, write_json
from .models import Gait, Placement
from .simulator import SimulationConfig, simulate


def generate_dataset(directory: str | Path, *, subjects: int = 15,
                     sessions: int = 2, duration_s: float = 20, seed: int = 20261111) -> dict:
    if subjects < 1 or sessions < 1:
        raise ValueError("subjects and sessions must be positive")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest, windows = [], []
    gaits = (Gait.CRUTCH, Gait.KNEE, Gait.HIP)
    for subject in range(subjects):
        # 每个虚拟受试者只属于一个分区。 / Keep every synthetic subject in one split.
        bucket = (subject // 3) % 5
        split = "test" if bucket == 4 else "validation" if bucket == 3 else "train"
        for session in range(sessions + 1):
            activity = "walking" if session < sessions else ("stationary" if subject % 2 == 0 else "hand_motion")
            placement = Placement.THIGH if session < sessions and session % 2 else Placement.HAND
            session_id = f"subject_{subject:03d}_session_{session:02d}"
            sample_seed = seed + subject * 10000 + session
            cfg = SimulationConfig(
                gait=gaits[subject % 3], placement=placement, duration_s=duration_s,
                seed=sample_seed, activity=activity,
                sample_rate_hz=(25, 50, 100)[session % 3],
                noise_std=(0.04, 0.1, 0.18)[subject % 3],
                spike_probability=0.002,
            )
            sim = simulate(cfg)
            save_simulation(sim, root / session_id)
            manifest.append({
                "subject_id": f"synthetic_{subject:03d}", "session_id": session_id,
                "split": split, "gait": str(cfg.gait), "placement": str(cfg.placement),
                "activity": activity, "seed": sample_seed, "source": "synthetic",
                "sensors": f"{session_id}/sensors.csv", "labels": f"{session_id}/labels.json",
                "metadata": f"{session_id}/metadata.json",
            })
            data = preprocess(sim.recording)
            for seg in data.segments:
                width = round(5 * seg.sample_rate_hz)
                for start in range(0, len(seg.times) - width + 1, width):
                    a, b = seg.times[start], seg.times[start + width - 1]
                    values = seg.filtered[start:start + width]
                    label_end = b + 1 / seg.sample_rate_hz
                    windows.append({
                        "subject_id": f"synthetic_{subject:03d}", "session_id": session_id,
                        "split": split, "gait": str(cfg.gait), "placement": str(cfg.placement),
                        "activity": activity, "start_s": a, "end_s": label_end,
                        "mean_abs": mean(abs(v) for v in values), "std": pstdev(values),
                        "rms": (mean(v * v for v in values)) ** 0.5,
                        "peak_to_peak": max(values) - min(values),
                        "target_steps": sum(e.kind == "foot_contact" and a <= e.time_s < label_end for e in sim.events),
                    })
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        names = list(windows[0]) if windows else ["subject_id", "session_id", "split", "target_steps"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(windows)
    summary = {
        "source": "synthetic_only", "subjects": subjects, "sessions": len(manifest),
        "windows": len(windows), "seed": seed,
        "split_sessions": {s: sum(r["split"] == s for r in manifest) for s in ("train", "validation", "test")},
        "split_policy": "subject grouped, 60/20/20 per blocks of 15 subjects; fewer subjects may omit splits",
        "limitation": "Virtual subject IDs group sessions; no longitudinal human identity model or real data",
        "training_features": ["mean_abs", "std", "rms", "peak_to_peak"],
        "target": "target_steps", "feature_context": "offline centered filtering uses surrounding session samples",
    }
    write_json(root / "summary.json", summary)
    return summary
