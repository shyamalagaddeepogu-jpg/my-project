import random
from typing import Optional

from environment.models import (
    Observation, Action, Reward, StepResult,
    Order, Driver, OrderStatus, Priority, FATIGUE_THRESHOLD
)
from environment.data import generate_orders, generate_drivers
from environment.tasks import TASKS, grade_episode, get_traffic_multiplier


class DeliveryDispatcherEnv:
    def __init__(self, task_id: str = "task_easy"):
        assert task_id in TASKS, f"Unknown task: {task_id}"
        self.task_id = task_id
        self.config = TASKS[task_id]

        self.orders: dict[str, Order] = {}
        self.drivers: dict[str, Driver] = {}
        self.current_time: int = 0
        self.step_count: int = 0

        # delivery counters
        self.delivered: int = 0
        self.failed: int = 0
        self.capacity_violations: int = 0
        self.priority_sla_breaches: int = 0
        self.on_time_deliveries: int = 0
        self.assigned_to_cancelled: int = 0

        # richer stats — all keys expected by tasks.py bonus functions
        self.drivers_used: set[str] = set()
        self.driver_usage: dict[str, int] = {}
        self.time_margins: list[float] = []
        self.time_margin_ratios: list[float] = []
        self.time_windows_at_assign: list[float] = []
        self.high_priority_on_time: int = 0
        self.high_priority_total: int = 0

        self._done: bool = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_reward(r: float) -> float:
        """Clamp reward to strictly open interval (0.001, 0.999)."""
        return round(max(0.001, min(0.999, r)), 4)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        cfg = self.config
        orders = generate_orders(cfg.num_orders, seed=cfg.seed, time_pressure=cfg.time_pressure)
        drivers = generate_drivers(cfg.num_drivers, seed=cfg.seed)

        self.orders = {o.order_id: o for o in orders}
        self.drivers = {d.driver_id: d for d in drivers}
        self.current_time = 0
        self.step_count = 0
        self.delivered = 0
        self.failed = 0
        self.capacity_violations = 0
        self.priority_sla_breaches = 0
        self.on_time_deliveries = 0
        self.assigned_to_cancelled = 0
        self.drivers_used = set()
        self.driver_usage = {}
        self.time_margins = []
        self.time_margin_ratios = []
        self.time_windows_at_assign = []
        self.high_priority_on_time = 0
        self.high_priority_total = 0
        self._done = False

        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._get_observation(),
                reward=Reward(value=0.001, reason="Episode already done"),
                done=True,
                info={}
            )

        self.step_count += 1
        self.current_time += 5  # each step = 5 simulation minutes

        # inject dynamic cancellations for hard task
        if self.config.dynamic_cancellations and self.step_count % 7 == 0:
            self._cancel_random_order()

        reward, reason, partial = self._apply_action(action)

        # clamp reward strictly between 0 and 1
        reward = self._clamp_reward(reward)

        # tick time windows — pending orders expire
        self._tick_orders()

        done = self._check_done()
        self._done = done

        return StepResult(
            observation=self._get_observation(),
            reward=Reward(value=reward, reason=reason, partial_credits=partial),
            done=done,
            info=self._episode_stats()
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "current_time": self.current_time,
            "step_count": self.step_count,
            "orders": {k: v.dict() for k, v in self.orders.items()},
            "drivers": {k: v.dict() for k, v in self.drivers.items()},
            "stats": self._episode_stats(),
            "done": self._done,
        }

    # ── Action handler ────────────────────────────────────────────────────────

    def _apply_action(self, action: Action):
        partial = {}

        if action.action_type == "assign":
            if not action.driver_id or not action.order_id:
                return 0.05, "Invalid assign — missing driver_id or order_id", partial

            driver = self.drivers.get(action.driver_id)
            order = self.orders.get(action.order_id)

            if driver is None:
                return 0.05, f"Driver {action.driver_id} not found", partial
            if order is None:
                return 0.05, f"Order {action.order_id} not found", partial
            if order.status != OrderStatus.PENDING:
                return 0.05, f"Order {action.order_id} is not pending (status={order.status})", partial
            if not driver.available:
                return 0.05, f"Driver {action.driver_id} is not available", partial

            # use effective_capacity so fatigue is enforced
            eff_cap = driver.effective_capacity
            if driver.current_load + order.weight > eff_cap:
                self.capacity_violations += 1
                partial["capacity_penalty"] = -0.1
                partial["effective_capacity"] = eff_cap
                return 0.05, f"Capacity violation: {driver.driver_id} overloaded (eff_cap={eff_cap}kg)", partial

            # traffic-aware travel time
            traffic = get_traffic_multiplier(self.config, self.current_time)
            travel_dist = driver.location.distance_to(order.dropoff)
            est_time = travel_dist * 0.5 * traffic

            # assign order → IN_TRANSIT
            order.status = OrderStatus.IN_TRANSIT
            order.assigned_driver = driver.driver_id
            driver.current_load += order.weight
            driver.assigned_orders.append(order.order_id)

            # track stats at assignment time
            if order.priority == Priority.HIGH:
                self.high_priority_total += 1
            self.time_windows_at_assign.append(order.time_window)

            # SLA check for high-priority orders
            sla_ok = est_time <= order.time_window
            if not sla_ok and order.priority == Priority.HIGH:
                self.priority_sla_breaches += 1
                partial["sla_breach"] = -0.1

            # expose traffic context to the caller
            partial["traffic_multiplier"] = round(traffic, 2)
            partial["est_delivery_time"] = round(est_time, 1)

            # simulate delivery outcome
            if sla_ok:
                order.status = OrderStatus.DELIVERED
                driver.current_load -= order.weight
                driver.assigned_orders.remove(order.order_id)
                driver.deliveries_done += 1
                self.delivered += 1
                self.on_time_deliveries += 1

                time_margin = order.time_window - est_time
                self.time_margins.append(time_margin)
                self.time_margin_ratios.append(
                    time_margin / order.time_window if order.time_window > 0 else 0.0
                )
                self.drivers_used.add(driver.driver_id)
                self.driver_usage[driver.driver_id] = (
                    self.driver_usage.get(driver.driver_id, 0) + 1
                )
                if order.priority == Priority.HIGH:
                    self.high_priority_on_time += 1

                # 0.95 for high priority, 0.80 for normal/low — never exactly 1.0
                reward = 0.95 if order.priority == Priority.HIGH else 0.80
                partial["on_time"] = True
                partial["time_margin"] = round(time_margin, 2)
                return reward, f"Order {order.order_id} delivered on time", partial

            else:
                order.status = OrderStatus.DELAYED
                driver.current_load -= order.weight
                driver.assigned_orders.remove(order.order_id)
                driver.deliveries_done += 1
                self.delivered += 1
                self.drivers_used.add(driver.driver_id)
                self.driver_usage[driver.driver_id] = (
                    self.driver_usage.get(driver.driver_id, 0) + 1
                )
                return 0.30, f"Order {order.order_id} delivered late (traffic={traffic:.1f}x)", partial

        elif action.action_type == "cancel":
            order = self.orders.get(action.order_id)
            if order is None or order.status != OrderStatus.PENDING:
                return 0.02, "Cannot cancel — order not found or not pending", partial
            order.status = OrderStatus.CANCELLED
            self.failed += 1
            return 0.02, f"Order {order.order_id} cancelled by agent", partial

        elif action.action_type == "delay":
            order = self.orders.get(action.order_id)
            if order is None or order.status != OrderStatus.PENDING:
                return 0.02, "Cannot delay — order not found or not pending", partial
            order.time_window += 10
            return 0.10, f"Order {order.order_id} time window extended by 10 min", partial

        else:
            return 0.02, f"Unknown action type: {action.action_type}", partial

    # ── Time mechanics ────────────────────────────────────────────────────────

    def _tick_orders(self):
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                order.time_window -= 5
                if order.time_window <= 0:
                    order.status = OrderStatus.CANCELLED
                    self.failed += 1

    def _cancel_random_order(self):
        pending = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        if pending:
            victim = random.choice(pending)
            if victim.assigned_driver:
                self.assigned_to_cancelled += 1
            victim.status = OrderStatus.CANCELLED
            self.failed += 1

    # ── Episode bookkeeping ───────────────────────────────────────────────────

    def _check_done(self) -> bool:
        if self.step_count >= self.config.max_steps:
            return True
        pending = [
            o for o in self.orders.values()
            if o.status == OrderStatus.PENDING
        ]
        return len(pending) == 0

    def _get_observation(self) -> Observation:
        pending = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        available = [
            d for d in self.drivers.values()
            if d.available and d.current_load < d.effective_capacity
        ]
        return Observation(
            pending_orders=pending,
            available_drivers=available,
            current_time=self.current_time,
            total_orders=len(self.orders),
            delivered_count=self.delivered,
            failed_count=self.failed,
            task_id=self.task_id,
        )

    def _episode_stats(self) -> dict:
        total = len(self.orders)
        num_drivers = len(self.drivers)
        avg_tw = (
            sum(self.time_windows_at_assign) / len(self.time_windows_at_assign)
            if self.time_windows_at_assign else 1.0
        )
        avg_margin = (
            sum(self.time_margins) / len(self.time_margins)
            if self.time_margins else 0.0
        )
        return {
            "delivered": self.delivered,
            "failed": self.failed,
            "total_orders": total,
            "on_time_rate": self.on_time_deliveries / total if total else 0.0,
            "capacity_violations": self.capacity_violations,
            "priority_sla_breaches": self.priority_sla_breaches,
            "assigned_to_cancelled": self.assigned_to_cancelled,
            "step_count": self.step_count,
            "steps_used": self.step_count,
            "driver_usage": self.driver_usage,
            "time_margin_ratios": self.time_margin_ratios,
            "avg_time_remaining": avg_margin,
            "avg_time_window": avg_tw,
            "high_priority_total": self.high_priority_total,
            "high_priority_on_time": self.high_priority_on_time,
            "driver_utilisation_rate": (
                len(self.drivers_used) / num_drivers if num_drivers else 0.0
            ),
            "avg_time_margin": avg_margin,
            "high_priority_on_time_rate": (
                self.high_priority_on_time / self.high_priority_total
                if self.high_priority_total else 1.0
            ),
        }

    def final_score(self) -> float:
        return grade_episode(self.task_id, self._episode_stats())
