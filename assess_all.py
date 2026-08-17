import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal
from app.models import Fault, HealthAssessment, SensorData

try:
    from .rules import HealthEngine, compute_health_score
except ImportError:
    from rules import HealthEngine, compute_health_score

FAULT_ENGINE_META = Path(__file__).resolve().parent.parent / "fault_engine" / "artifacts" / "meta.json"


def _fault_threshold():
    return json.loads(FAULT_ENGINE_META.read_text())["threshold"]


def assess_all():
    """Assess every station's health from its current fault_engine
    output and append one HealthAssessment row per station. Unlike
    detect_all(), this is meant to accumulate over time - each run is a
    new point-in-time snapshot for trend tracking, not a replacement.
    """
    threshold = _fault_threshold()

    db = SessionLocal()
    try:
        station_ids = [row[0] for row in db.query(SensorData.station_id).all()]

        per_station = {}
        for station_id, severity, fault_score in db.query(
            Fault.station_id, Fault.severity, Fault.fault_score
        ).all():
            stats = per_station.setdefault(
                station_id, {"counts": {}, "max_warning_ratio": 0.0, "max_ratio": 0.0}
            )
            stats["counts"][severity] = stats["counts"].get(severity, 0) + 1

            ratio = fault_score / threshold
            if severity == "WARNING" and ratio > stats["max_warning_ratio"]:
                stats["max_warning_ratio"] = ratio
            if ratio > stats["max_ratio"]:
                stats["max_ratio"] = ratio

        assessed_at = datetime.now(timezone.utc)
        summary = {"assessed": 0, "healthy": 0, "monitor": 0, "maintenance_required": 0, "critical": 0}
        records = []

        for station_id in station_ids:
            stats = per_station.get(station_id, {"counts": {}, "max_warning_ratio": 0.0, "max_ratio": 0.0})
            counts = stats["counts"]
            critical = counts.get("CRITICAL", 0) + counts.get("FAILURE", 0)
            warning = counts.get("WARNING", 0)

            status = HealthEngine.evaluate_station(critical, warning, stats["max_warning_ratio"])
            health_score = compute_health_score(critical + warning, stats["max_ratio"])

            records.append(HealthAssessment(
                station_id=station_id,
                station_status=status,
                health_score=health_score,
                total_faults=critical + warning,
                critical_faults=critical,
                warning_faults=warning,
                assessment_time=assessed_at,
            ))
            summary["assessed"] += 1
            summary[status.lower()] += 1

        db.add_all(records)
        db.commit()
    finally:
        db.close()

    return summary


if __name__ == "__main__":
    print(assess_all())
