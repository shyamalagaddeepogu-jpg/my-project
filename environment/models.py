from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class OrderStatus(str, Enum):
    PENDING    = "pending"
    ASSIGNED   = "assigned"
    IN_TRANSIT = "in_transit"
    DELIVERED  = "delivered"
    DELAYED    = "delayed"
    CANCELLED  = "cancelled"


class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class Location:
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def dict(self):
        return {"x": self.x, "y": self.y}


@dataclass
class Order:
    order_id: str
    pickup: Location
    dropoff: Location
    time_window: int
    weight: float
    priority: Priority = Priority.MEDIUM
    status: OrderStatus = OrderStatus.PENDING
    assigned_driver: Optional[str] = None
    created_at: int = 0

    def dict(self):
        return {
            "order_id":        self.order_id,
            "pickup":          self.pickup.dict(),
            "dropoff":         self.dropoff.dict(),
            "priority":        self.priority.value,
            "time_window":     self.time_window,
            "weight":          self.weight,
            "status":          self.status.value,
            "assigned_driver": self.assigned_driver,
            "created_at":      self.created_at,
        }


# Number of deliveries before fatigue kicks in for that driver
FATIGUE_THRESHOLD = 3


@dataclass
class Driver:
    driver_id: str
    location: Location
    capacity: float
    current_load: float = 0.0
    available: bool = True
    assigned_orders: List[str] = field(default_factory=list)
    # fatigue: after FATIGUE_THRESHOLD deliveries, effective capacity drops
    # 10% per tier (capped at -40%) forcing agents to rotate the fleet
    deliveries_done: int = 0

    @property
    def effective_capacity(self) -> float:
        fatigue_tiers = self.deliveries_done // FATIGUE_THRESHOLD
        reduction = min(0.40, fatigue_tiers * 0.10)
        return round(self.capacity * (1.0 - reduction), 2)

    def dict(self):
        return {
            "driver_id":          self.driver_id,
            "location":           self.location.dict(),
            "capacity":           self.capacity,
            "effective_capacity": self.effective_capacity,
            "current_load":       self.current_load,
            "available":          self.available,
            "assigned_orders":    self.assigned_orders,
            "deliveries_done":    self.deliveries_done,
        }


@dataclass
class Observation:
    pending_orders: List[Order]
    available_drivers: List[Driver]
    current_time: int
    total_orders: int
    delivered_count: int
    failed_count: int
    task_id: str

    def dict(self):
        return {
            "pending_orders":    [o.dict() for o in self.pending_orders],
            "available_drivers": [d.dict() for d in self.available_drivers],
            "current_time":      self.current_time,
            "total_orders":      self.total_orders,
            "delivered_count":   self.delivered_count,
            "failed_count":      self.failed_count,
            "task_id":           self.task_id,
        }


@dataclass
class Action:
    action_type: str
    driver_id: Optional[str] = None
    order_id: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "Action":
        return Action(
            action_type=d.get("action_type", ""),
            driver_id=d.get("driver_id"),
            order_id=d.get("order_id"),
        )


@dataclass
class Reward:
    value: float
    reason: str
    partial_credits: Dict[str, Any] = field(default_factory=dict)

    def dict(self):
        return {
            "value":           self.value,
            "reason":          self.reason,
            "partial_credits": self.partial_credits,
        }


@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def dict(self):
        return {
            "observation": self.observation.dict(),
            "reward":      self.reward.dict(),
            "done":        self.done,
            "info":        self.info,
        }
