"""Adapters for raw phone/IMU CSV, JSON, and iterable device samples."""

import csv
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from .models import Placement, Recording, Sample
from .simulator import Simulation


def from_rows(rows, *, placement: Placement | str, source: str = "real",
              columns: dict[str, str] | None = None, timestamp_unit: str = "s",
              acceleration_unit: str = "m/s2", gyro_unit: str = "rad/s",
              acceleration_kind: str = "with_gravity") -> Recording:
    names = columns or {}
    time_scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if timestamp_unit not in time_scales or acceleration_unit not in ("m/s2", "g") or gyro_unit not in ("rad/s", "deg/s"):
        raise ValueError("Unsupported timestamp, acceleration or gyroscope unit")
    acc_scale = 9.80665 if acceleration_unit == "g" else 1
    gyro_scale = 3.141592653589793 / 180 if gyro_unit == "deg/s" else 1
    samples = []
    origin = None
    for number, row in enumerate(rows, 1):
        try:
            # 先减起始时间再换单位，避免绝对纳秒时间丢精度。 / Subtract origin before scaling.
            timestamp = Decimal(str(row[names.get("t", "t")]))
            if origin is None:
                origin = timestamp
            t = float((timestamp - origin) * Decimal(str(time_scales[timestamp_unit])))
            acc = [float(row[names.get(k, k)]) * acc_scale for k in ("ax", "ay", "az")]
            raw_g = [row.get(names.get(k, k)) for k in ("gx", "gy", "gz")]
            g = [float(v) * gyro_scale if v not in (None, "") else None for v in raw_g]
            samples.append(Sample(t, *acc, *g))
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            raise ValueError(f"Invalid sensor row {number}: {exc}") from exc
    return Recording(tuple(samples), Placement(placement), source, acceleration_kind)


def load_recording(path: str | Path, **kwargs) -> Recording:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            return from_rows(csv.DictReader(handle), **kwargs)
        if path.suffix.lower() == ".json":
            data = json.load(handle)
            return from_rows(data if isinstance(data, list) else data["samples"], **kwargs)
    raise ValueError("Use a .csv or .json recording")


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_simulation(simulation: Simulation, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "sensors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Sample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(s) for s in simulation.recording.samples)
    write_json(directory / "labels.json", [asdict(e) for e in simulation.events])
    write_json(directory / "metadata.json", simulation.metadata)
