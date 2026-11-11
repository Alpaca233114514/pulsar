import argparse
import json
from pathlib import Path
import sys

from .algorithm import DetectorConfig, StepCounter
from .dataset import generate_dataset
from .demo import run_demo
from .energy import estimate_calories
from .io import load_recording, save_simulation
from .models import Gait, Placement
from .simulator import SimulationConfig, simulate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pulsar experimental phone-IMU gait demo")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate offline HTML + nine synthetic evaluation cases")
    demo.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--weight-kg", type=float, default=70)
    demo.add_argument("--met", type=float, default=3)
    simulation = commands.add_parser("simulate", help="Generate one labeled synthetic recording")
    simulation.add_argument("--out", type=Path, required=True)
    simulation.add_argument("--gait", choices=list(Gait), default=Gait.CRUTCH)
    simulation.add_argument("--placement", choices=list(Placement), default=Placement.HAND)
    simulation.add_argument("--duration", type=float, default=30)
    simulation.add_argument("--sample-rate", type=float, default=50)
    simulation.add_argument("--cadence", type=float)
    simulation.add_argument("--seed", type=int, default=42)
    simulation.add_argument("--noise", type=float, default=0.06)
    simulation.add_argument("--activity", choices=("walking", "stationary", "hand_motion"), default="walking")
    dataset = commands.add_parser("dataset", help="Generate subject-grouped training data and features")
    dataset.add_argument("--out", type=Path, required=True)
    dataset.add_argument("--subjects", type=int, default=15)
    dataset.add_argument("--sessions", type=int, default=2)
    dataset.add_argument("--duration", type=float, default=20)
    dataset.add_argument("--seed", type=int, default=20261111)
    analyze = commands.add_parser("analyze", help="Analyze real or synthetic sensor CSV/JSON")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--gait", choices=list(Gait), required=True)
    analyze.add_argument("--placement", choices=list(Placement), required=True)
    analyze.add_argument("--event-mode", choices=("auto", "step", "stride"), default="auto")
    analyze.add_argument("--source", choices=("real", "synthetic"), default="real")
    analyze.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), default="s")
    analyze.add_argument("--acceleration-unit", choices=("m/s2", "g"), default="m/s2")
    analyze.add_argument("--gyro-unit", choices=("rad/s", "deg/s"), default="rad/s")
    analyze.add_argument("--columns", type=json.loads, help='Canonical-to-input JSON, e.g. {"t":"time","ax":"acc_x"}')
    analyze.add_argument("--weight-kg", type=float)
    analyze.add_argument("--met", type=float)
    calories = commands.add_parser("calories", help="Explicit-duration MET energy scenario")
    calories.add_argument("--weight-kg", type=float, required=True)
    calories.add_argument("--duration", type=float, required=True, help="Activity duration in seconds")
    calories.add_argument("--met", type=float, required=True)
    calories.add_argument("--met-range", type=float, nargs=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            report = run_demo(args.out, seed=args.seed, weight_kg=args.weight_kg, met=args.met)
            output = {"report": str((args.out / "index.html").resolve()),
                      "validation": report["validation"],
                      "cases": [{"case": c["id"], **c["metrics"]} for c in report["cases"]]}
        elif args.command == "simulate":
            sim = simulate(SimulationConfig(gait=args.gait, placement=args.placement,
                           duration_s=args.duration, sample_rate_hz=args.sample_rate,
                           cadence_spm=args.cadence, seed=args.seed, noise_std=args.noise,
                           activity=args.activity))
            save_simulation(sim, args.out)
            output = {"directory": str(args.out.resolve()), "samples": len(sim.recording.samples)}
        elif args.command == "dataset":
            output = generate_dataset(args.out, subjects=args.subjects, sessions=args.sessions,
                                      duration_s=args.duration, seed=args.seed)
        elif args.command == "analyze":
            if (args.weight_kg is None) != (args.met is None):
                raise ValueError("Supply both --weight-kg and --met for calories")
            if args.columns is not None and (not isinstance(args.columns, dict) or
                                             not all(isinstance(v, str) for v in args.columns.values())):
                raise ValueError("--columns must be a JSON object mapping canonical fields to column names")
            recording = load_recording(args.path, placement=args.placement, source=args.source,
                         columns=args.columns, timestamp_unit=args.timestamp_unit,
                         acceleration_unit=args.acceleration_unit, gyro_unit=args.gyro_unit)
            result = StepCounter(DetectorConfig(gait=args.gait, placement=args.placement,
                                 event_mode=args.event_mode)).analyze(recording)
            output = {"analysis": result.to_dict()}
            if args.met is not None:
                output["calories"] = estimate_calories(weight_kg=args.weight_kg,
                    duration_s=result.active_duration_s, met=args.met,
                    duration_source="detected_bouts").to_dict()
        else:
            output = estimate_calories(weight_kg=args.weight_kg, duration_s=args.duration,
                     met=args.met, met_range=tuple(args.met_range) if args.met_range else None).to_dict()
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"pulsar: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
