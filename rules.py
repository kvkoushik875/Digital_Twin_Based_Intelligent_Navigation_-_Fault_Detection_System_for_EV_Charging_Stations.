MAINTENANCE_WARNING_COUNT = 3

# The fault engine's own WARNING band spans ratio (error/threshold) 1.0-1.5
# before escalating a fault to CRITICAL itself (see fault_engine/detect_all.py
# _severity). 1.25 is the midpoint of that band: a WARNING fault already
# close to crossing into CRITICAL territory is treated as needing
# maintenance now rather than just monitoring, using a boundary the fault
# engine defined, not an independent judgment call.
MAINTENANCE_RATIO_CUTOFF = 1.25

ALERT_STATUSES = ("CRITICAL", "MAINTENANCE_REQUIRED")


class HealthEngine:
    """Status is derived entirely from the faults table the fault engine
    writes to - no independent sensor thresholds. This keeps every health
    status directly traceable to a fault engine record (or the lack of
    one), so the two always agree.
    """

    @staticmethod
    def evaluate_station(critical_faults, warning_faults, max_warning_ratio=0.0):
        if critical_faults > 0:
            return "CRITICAL"

        if warning_faults >= MAINTENANCE_WARNING_COUNT:
            return "MAINTENANCE_REQUIRED"

        if warning_faults > 0 and max_warning_ratio >= MAINTENANCE_RATIO_CUTOFF:
            return "MAINTENANCE_REQUIRED"

        if warning_faults > 0:
            return "MONITOR"

        return "HEALTHY"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_health_score(total_faults, max_ratio):
    """0-100, derived from the worst fault's real error/threshold ratio
    (the same continuous number fault_engine computes), not an
    independent judgment call. Anchored at fault_engine's own severity
    boundaries (ratio 1.0/1.5/2.5 -> WARNING/CRITICAL/FAILURE, see
    fault_engine/detect_all.py _severity) so the numeric score and the
    categorical station_status can never contradict each other: a
    station always scores lower than one in a milder severity band,
    never the reverse.
    """
    if max_ratio <= 0:
        deduction = 0.0
    elif max_ratio < 1.5:
        # WARNING band
        deduction = 10 + clamp((max_ratio - 1.0) / 0.5, 0, 1) * 25
    elif max_ratio < 2.5:
        # CRITICAL band
        deduction = 35 + clamp((max_ratio - 1.5) / 1.0, 0, 1) * 35
    else:
        # FAILURE band
        deduction = 70 + clamp((max_ratio - 2.5) * 10, 0, 30)

    # Extra faults beyond the worst one still matter a little, capped so
    # they can't push a station past the next severity band on their own.
    extra_count_penalty = clamp((max(total_faults, 1) - 1) * 5, 0, 15)

    return round(clamp(100 - deduction - extra_count_penalty, 0, 100), 1)
