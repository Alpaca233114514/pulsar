import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from statistics import mean
import tempfile
import unittest

from pulsar import DetectorConfig, Gait, Placement, Recording, Sample, SimulationConfig, StepCounter, estimate_calories, simulate
from pulsar.dataset import generate_dataset
from pulsar.denoise import hampel, preprocess
from pulsar.evaluation import evaluate
from pulsar.io import from_rows, load_recording, save_simulation


def test_directory():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "test-runs"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(dir=root)


def independent_pulses():
    # 手写测试信号，与模拟器无共享波形代码。 / Independent fixture, not simulator output.
    times = [1 + i * 0.75 for i in range(20)]
    samples = []
    for i in range(800):
        t = i / 50
        signal = sum(1.7 * max(0, 1 - abs(t - event) / 0.14) for event in times)
        samples.append(Sample(t, 0, 0, 9.80665 + signal))
    return Recording(tuple(samples), Placement.HAND, "synthetic"), times


class AlgorithmTests(unittest.TestCase):
    def test_independent_step_timing(self):
        recording, truth = independent_pulses()
        result = StepCounter().analyze(recording)
        self.assertEqual(result.step_count, len(truth))
        self.assertTrue(all(abs(a - b) < 0.04 for a, b in zip(result.event_times_s, truth)))

    def test_rotation_invariance(self):
        recording, _ = independent_pulses()
        rotated = replace(recording, samples=tuple(Sample(s.t, s.az, s.ax, s.ay) for s in recording.samples))
        self.assertEqual(StepCounter().analyze(recording).event_times_s,
                         StepCounter().analyze(rotated).event_times_s)

    def test_cadence_uses_elapsed_intervals_for_asymmetric_gait(self):
        sim = simulate(SimulationConfig(gait=Gait.HIP, randomize=False, cadence_spm=60, duration_s=40))
        result = StepCounter(DetectorConfig(gait=Gait.HIP)).analyze(sim.recording)
        self.assertAlmostEqual(result.cadence_spm, 60, delta=1)

    def test_stride_semantics_are_explicit(self):
        recording, _ = independent_pulses()
        step = StepCounter().analyze(recording)
        stride = StepCounter(DetectorConfig(event_mode="stride")).analyze(recording)
        self.assertEqual(stride.step_count, 2 * len(stride.event_times_s))
        self.assertIn("two_steps_per_stride_assumes_alternating_gait", stride.warnings)
        self.assertEqual(step.step_count, 20)

    def test_synthetic_gaits_across_placements_and_seeds(self):
        for gait in (Gait.CRUTCH, Gait.KNEE, Gait.HIP):
            for placement in (Placement.HAND, Placement.THIGH):
                for seed in (7, 31, 93):
                    with self.subTest(gait=gait, placement=placement, seed=seed):
                        sim = simulate(SimulationConfig(gait=gait, placement=placement, seed=seed, duration_s=20))
                        result = StepCounter(DetectorConfig(gait=gait, placement=placement)).analyze(sim.recording)
                        self.assertLessEqual(evaluate(result, sim.events)["absolute_count_error"], 2)

    def test_stationary_and_isolated_motion_not_counted(self):
        sim = simulate(SimulationConfig(activity="stationary", noise_std=0.15))
        self.assertEqual(StepCounter().analyze(sim.recording).step_count, 0)
        recording, _ = independent_pulses()
        short = replace(recording, samples=recording.samples[:110])
        self.assertEqual(StepCounter().analyze(short).step_count, 0)

    def test_rhythmic_hand_motion_limitation_is_visible(self):
        sim = simulate(SimulationConfig(activity="hand_motion"))
        result = StepCounter().analyze(sim.recording)
        self.assertIn("rhythmic_hand_motion_can_be_miscounted", result.warnings)
        self.assertEqual(evaluate(result, sim.events)["truth_steps"], 0)
        self.assertGreater(evaluate(result, sim.events)["false_steps_when_no_walking"], 0)

    def test_long_gap_is_not_bridged_or_filled(self):
        sim = simulate(SimulationConfig(gap_s=(10, 14)))
        result = StepCounter(DetectorConfig(gait=Gait.CRUTCH)).analyze(sim.recording)
        self.assertEqual(result.diagnostics["segments"], 2)
        self.assertTrue(all(not 10 < t < 14 for t in result.event_times_s))
        self.assertTrue(all(not (a < 10 and b > 14) for a, b in result.bouts_s))
        self.assertLess(result.observed_duration_s, result.recording_duration_s - 3.9)

    def test_empty_input_and_placement_mismatch(self):
        result = StepCounter().analyze(Recording((), Placement.HAND))
        self.assertEqual(result.step_count, 0)
        self.assertIsNone(result.cadence_spm)
        with self.assertRaises(ValueError):
            StepCounter().analyze(Recording((), Placement.THIGH))

    def test_low_rate_and_wrong_acceleration_units_rejected(self):
        with self.assertRaises(ValueError):
            StepCounter().analyze(Recording(tuple(Sample(i / 10, 0, 0, 9.8) for i in range(40)), Placement.HAND))
        with self.assertRaises(ValueError):
            StepCounter().analyze(Recording(tuple(Sample(i / 50, 0, 0, 1) for i in range(100)), Placement.HAND))


class DenoiseTests(unittest.TestCase):
    def test_hampel_repairs_an_isolated_impulse(self):
        signal = [9.8] * 101
        signal[50] = 40
        clean, changed = hampel(signal)
        self.assertEqual(changed, 1)
        self.assertEqual(clean[50], 9.8)

    def test_noise_reduction_preserves_broad_motion(self):
        rng = random.Random(123)
        clean = [0.8 * math.sin(2 * math.pi * i / 50) for i in range(1000)]
        noisy = [v + rng.gauss(0, 0.35) for v in clean]
        noisy[200] += 12
        recording = Recording(tuple(Sample(i / 50, 0, 0, 9.80665 + v) for i, v in enumerate(noisy)), Placement.HAND)
        filtered = preprocess(recording).segments[0].filtered
        before = mean((a - b) ** 2 for a, b in zip(noisy[100:-100], clean[100:-100]))
        after = mean((a - b) ** 2 for a, b in zip(filtered[100:-100], clean[100:-100]))
        self.assertLess(after, before * 0.5)
        self.assertGreater(max(filtered), 0.5)

    def test_timestamp_jitter_resampled(self):
        sim = simulate(SimulationConfig(timestamp_jitter=0.3, dropout_probability=0.1))
        seg = preprocess(sim.recording).segments[0]
        intervals = [b - a for a, b in zip(seg.times, seg.times[1:])]
        self.assertLess(max(intervals) - min(intervals), 1e-10)


class AdapterTests(unittest.TestCase):
    def test_mapped_millisecond_g_and_degrees_input(self):
        rows = [{"time": 100000 + i * 20, "x": 0, "y": 0, "z": 1,
                 "gx": 180, "gy": 0, "gz": 0} for i in range(10)]
        recording = from_rows(rows, placement="hand", columns={"t": "time", "ax": "x", "ay": "y", "az": "z"},
                              timestamp_unit="ms", acceleration_unit="g", gyro_unit="deg/s")
        self.assertAlmostEqual(recording.samples[-1].t, 0.18)
        self.assertAlmostEqual(recording.samples[0].az, 9.80665)
        self.assertAlmostEqual(recording.samples[0].gx, math.pi)
        self.assertEqual(recording.source, "real")

    def test_absolute_nanoseconds_keep_intervals(self):
        recording = from_rows([{"t": str(1794300000000000000 + i * 20000000), "ax": 0, "ay": 0, "az": 9.8}
                               for i in range(10)], placement="hand", timestamp_unit="ns")
        self.assertAlmostEqual(recording.samples[-1].t, 0.18)

    def test_invalid_data_rejected(self):
        for rows in (
            [{"t": 0, "ax": "NaN", "ay": 0, "az": 9.8}],
            [{"t": 0, "ax": 0, "ay": 0, "az": 9.8, "gx": 1}],
            [{"t": 0, "ax": 0, "ay": 0}],
            [{"t": 0, "ax": 0, "ay": 0, "az": 9.8}] * 2,
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                from_rows(rows, placement="hand")
        with self.assertRaises(ValueError):
            from_rows([], placement="hand", acceleration_kind="linear")

    def test_csv_and_json_roundtrip_and_no_overwrite(self):
        with self.subTest("roundtrip"):
            tmp = test_directory()
            path = Path(tmp) / "session"
            sim = simulate(SimulationConfig(duration_s=4))
            save_simulation(sim, path)
            recording = load_recording(path / "sensors.csv", placement="hand", source="synthetic")
            self.assertEqual(len(recording.samples), len(sim.recording.samples))
            self.assertAlmostEqual(recording.duration_s, sim.recording.duration_s)
            with self.assertRaises(FileExistsError):
                save_simulation(sim, path)
            js = Path(tmp) / "recording.json"
            js.write_text(json.dumps({"samples": [{"t": 0, "ax": 0, "ay": 0, "az": 9.8}]}), encoding="utf-8")
            self.assertEqual(len(load_recording(js, placement="waist").samples), 1)


class SimulatorAndDatasetTests(unittest.TestCase):
    def test_reproducibility_and_separate_crutch_labels(self):
        cfg = SimulationConfig(seed=19)
        self.assertEqual(simulate(cfg), simulate(cfg))
        a, b = simulate(cfg), simulate(replace(cfg, seed=20))
        self.assertNotEqual(a.recording.samples, b.recording.samples)
        self.assertEqual(sum(e.kind == "crutch_contact" for e in a.events),
                         sum(e.kind == "foot_contact" for e in a.events))

    def test_pauses_have_no_foot_labels(self):
        sim = simulate(SimulationConfig(pauses_s=((8, 15),)))
        self.assertTrue(all(not 8 <= e.time_s <= 15 for e in sim.events if e.kind == "foot_contact"))

    def test_subject_splits_and_targets(self):
        with self.subTest("dataset"):
            tmp = test_directory()
            root = Path(tmp) / "dataset"
            summary = generate_dataset(root, subjects=15, sessions=1, duration_s=6)
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
            subjects = {}
            for row in rows:
                subjects.setdefault(row["subject_id"], set()).add(row["split"])
            self.assertTrue(all(len(v) == 1 for v in subjects.values()))
            self.assertTrue(all(summary["split_sessions"][s] > 0 for s in ("train", "validation", "test")))
            with (root / "windows.csv").open(encoding="utf-8", newline="") as handle:
                windows = list(csv.DictReader(handle))
            self.assertGreater(len(windows), 0)
            for row in windows:
                if row["activity"] != "walking":
                    self.assertEqual(int(row["target_steps"]), 0)


class EnergyAndEvaluationTests(unittest.TestCase):
    def test_gross_net_and_range(self):
        result = estimate_calories(weight_kg=70, duration_s=1800, met=3, met_range=(2, 5))
        self.assertEqual(result.gross_kcal, 105)
        self.assertEqual(result.active_kcal, 70)
        self.assertEqual(result.gross_range_kcal, (70, 175))
        self.assertEqual(result.active_range_kcal, (35, 140))

    def test_zero_duration_and_resting_baseline(self):
        self.assertEqual(estimate_calories(weight_kg=70, duration_s=0, met=3).gross_kcal, 0)
        self.assertEqual(estimate_calories(weight_kg=70, duration_s=600, met=1).active_kcal, 0)

    def test_invalid_energy_inputs(self):
        for overrides in ({"weight_kg": 0}, {"duration_s": -1}, {"met": math.nan}, {"met_range": (4, 6)}):
            values = {"weight_kg": 70, "duration_s": 60, "met": 3, **overrides}
            with self.subTest(values=values), self.assertRaises(ValueError):
                estimate_calories(**values)

    def test_event_matching_does_not_reuse_truth(self):
        sim = simulate(SimulationConfig(duration_s=8))
        result = StepCounter(DetectorConfig(gait=Gait.CRUTCH)).analyze(sim.recording)
        result.event_times_s = [1.0, 1.01, 1.02]
        metrics = evaluate(result, sim.events)
        self.assertEqual(metrics["matched_events"], 1)
        self.assertAlmostEqual(metrics["event_precision"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
