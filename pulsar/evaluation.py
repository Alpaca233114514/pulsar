"""Ground truth is used only here, outside the inference API."""

from .models import DetectionResult, Event


def evaluate(result: DetectionResult, events: list[Event], tolerance_s: float = 0.2) -> dict:
    if tolerance_s <= 0:
        raise ValueError("Event tolerance must be positive")
    truth_steps = sum(e.kind == "foot_contact" for e in events)
    truth = sorted(e.time_s for e in events if e.kind == "motion_cycle")
    predicted = sorted(result.event_times_s)
    i = j = matched = 0
    errors = []
    while i < len(truth) and j < len(predicted):
        difference = predicted[j] - truth[i]
        if abs(difference) <= tolerance_s:
            errors.append(abs(difference))
            matched += 1
            i += 1
            j += 1
        elif difference < 0:
            j += 1
        else:
            i += 1
    precision = matched / len(predicted) if predicted else None
    recall = matched / len(truth) if truth else None
    f1 = 2 * matched / (len(truth) + len(predicted)) if truth or predicted else None
    return {
        "truth_steps": truth_steps, "estimated_steps": result.step_count,
        "absolute_count_error": abs(result.step_count - truth_steps),
        "relative_count_error": abs(result.step_count - truth_steps) / truth_steps if truth_steps else None,
        "false_steps_when_no_walking": result.step_count if not truth_steps else None,
        "event_precision": precision, "event_recall": recall, "event_f1": f1,
        "matched_events": matched, "truth_motion_events": len(truth),
        "event_tolerance_s": tolerance_s,
        "matched_mean_absolute_timing_error_s": sum(errors) / len(errors) if errors else None,
        "event_reference": "motion_cycle (stride proxy for thigh, foot-contact proxy elsewhere)",
    }
