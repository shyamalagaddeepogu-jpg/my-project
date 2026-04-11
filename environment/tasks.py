from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TaskConfig:
    task_id: str
    description: str
    difficulty: str
    num_orders: int
    num_drivers: int
    time_pressure: str
    max_steps: int
    seed: int
    dynamic_cancellations: bool = False
    priority_sla: bool = False
    driver_fatigue: bool = False
    # traffic_schedule: list of (start_minute, end_minute, multiplier)
    # multiplier > 1.0 means slower travel (rush hour)
    traffic_schedule: List[tuple] = field(default_factory=list)


TASKS: Dict[str, TaskConfig] = {
    "task_easy": TaskConfig(
        task_id="task_easy",
        description=(
            "Assign 3 orders to 3 available drivers with relaxed time windows. "
            "No capacity constraints or traffic. "
            "Goal: deliver all orders correctly."
        ),
        difficulty="easy",
        num_orders=3,
        num_drivers=3,
        time_pressure="relaxed",
        max_steps=15,
        seed=42,
    ),
    "task_medium": TaskConfig(
        task_id="task_medium",
        description=(
            "Dispatch 10 orders across 5 drivers with varying capacity. "
            "Moderate time windows with rush-hour traffic in early simulation time. "
            "Must respect capacity limits and prioritise HIGH orders. "
            "Goal: maximise on-time deliveries without overloading drivers."
        ),
        difficulty="medium",
        num_orders=10,
        num_drivers=5,
        time_pressure="normal",
        max_steps=30,
        seed=77,
        priority_sla=True,
        # rush hour: first 30 sim-minutes travel is 1.6x slower
        traffic_schedule=[(0, 30, 1.6)],
    ),
    "task_hard": TaskConfig(
        task_id="task_hard",
        description=(
            "20 orders, 7 drivers, tight time windows. "
            "Rush-hour traffic early, off-peak later. "
            "Drivers fatigue after 3 deliveries, reducing effective capacity. "
            "Dynamic mid-episode cancellations. "
            "Goal: optimise total delivery score under realistic pressure."
        ),
        difficulty="hard",
        num_orders=20,
        num_drivers=7,
        time_pressure="tight",
        max_steps=40,
        seed=13,
        dynamic_cancellations=True,
        priority_sla=True,
        driver_fatigue=True,
        # two traffic bands: rush (0-25 min, 1.8x), off-peak (50+ min, 0.85x)
        traffic_schedule=[(0, 25, 1.8), (50, 9999, 0.85)],
    ),
}


# ── Traffic helper ────────────────────────────────────────────────────────────

def get_traffic_multiplier(config: TaskConfig, current_time: int) -> float:
    """Return the travel-time multiplier for the current simulation minute."""
    for start, end, factor in config.traffic_schedule:
        if start <= current_time < end:
            return factor
    return 1.0


# ── Score clamp — STRICTLY between 0 and 1, never 0.0 or 1.0 ─────────────────

def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.001, 0.999)."""
    return round(max(0.001, min(0.999, score)), 4)


# ── Bonus helpers ─────────────────────────────────────────────────────────────

def _early_delivery_bonus(episode_stats: Dict[str, Any]) -> float:
    ratios = episode_stats.get("time_margin_ratios", [])
    if not ratios:
        return 0.0
    avg_ratio = sum(ratios) / len(ratios)
    return round(0.15 * max(0.0, min(1.0, avg_ratio)), 4)


def _driver_utilisation_bonus(episode_stats: Dict[str, Any]) -> float:
    driver_usage = episode_stats.get("driver_usage", {})
    if not driver_usage or len(driver_usage) < 2:
        return 0.0
    counts = list(driver_usage.values())
    avg = sum(counts) / len(counts)
    if avg == 0:
        return 0.0
    variance = sum((c - avg) ** 2 for c in counts) / len(counts)
    normalised = max(0.0, 1.0 - (variance / (avg ** 2 + 1e-6)))
    return round(0.10 * normalised, 4)


def _priority_adherence_bonus(episode_stats: Dict[str, Any]) -> float:
    high_total = episode_stats.get("high_priority_total", 0)
    high_on_time = episode_stats.get("high_priority_on_time", 0)
    if high_total == 0:
        return 0.0
    return round(0.10 * (high_on_time / high_total), 4)


def _step_efficiency_bonus(episode_stats: Dict[str, Any], max_steps: int) -> float:
    steps_used = episode_stats.get("steps_used", max_steps)
    if max_steps <= 0:
        return 0.0
    efficiency = max(0.0, 1.0 - (steps_used / max_steps))
    return round(0.05 * efficiency, 4)


# ── Main grader ───────────────────────────────────────────────────────────────

def grade_episode(task_id: str, episode_stats: Dict[str, Any]) -> float:
    """
    Multi-factor episode score strictly between 0 and 1 (never 0.0 or 1.0).
    """
    config = TASKS[task_id]
    delivered = episode_stats.get("delivered", 0)
    total = episode_stats.get("total_orders", config.num_orders)
    on_time_rate = episode_stats.get("on_time_rate", 0.0)
    cap_violations = episode_stats.get("capacity_violations", 0)
    sla_breaches = episode_stats.get("priority_sla_breaches", 0)

    if total == 0:
        return 0.001  # strictly > 0.0

    delivery_ratio = delivered / total
    early_bonus = _early_delivery_bonus(episode_stats)
    utilisation_bonus = _driver_utilisation_bonus(episode_stats)
    priority_bonus = _priority_adherence_bonus(episode_stats)
    efficiency_bonus = _step_efficiency_bonus(episode_stats, config.max_steps)

    # ── EASY ──────────────────────────────────────────────────────────────────
    if task_id == "task_easy":
        score = (
            0.80 * delivery_ratio
            + 0.20 * on_time_rate
            + early_bonus * 0.5
            + efficiency_bonus
            - cap_violations * 0.10
        )
        return _clamp(score)

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    if task_id == "task_medium":
        score = (
            0.45 * delivery_ratio
            + 0.35 * on_time_rate
            + early_bonus
            + utilisation_bonus
            + priority_bonus
            + efficiency_bonus
            - cap_violations * 0.05
            - sla_breaches * 0.05
        )
        return _clamp(score)

    # ── HARD ──────────────────────────────────────────────────────────────────
    if task_id == "task_hard":
        cancelled_penalty = episode_stats.get("assigned_to_cancelled", 0) * 0.02
        score = (
            0.35 * delivery_ratio
            + 0.30 * on_time_rate
            + early_bonus
            + utilisation_bonus
            + priority_bonus
            + efficiency_bonus
            - cap_violations * 0.03
            - sla_breaches * 0.04
            - cancelled_penalty
        )
        return _clamp(score)

    return 0.001  # fallback — strictly > 0.0
