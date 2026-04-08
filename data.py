import random
from typing import List
from environment.models import Order, Driver, Location, Priority


def generate_orders(n: int, seed: int = 42, time_pressure: str = "normal") -> List[Order]:
    random.seed(seed)
    orders = []
    priorities = list(Priority)

    time_windows = {
        "relaxed": (80, 150),
        "normal":  (50, 90),
        "tight":   (35, 60),
    }
    tw_min, tw_max = time_windows.get(time_pressure, (50, 90))

    for i in range(n):
        orders.append(Order(
            order_id=f"ORD-{i+1:03d}",
            pickup=Location(x=round(random.uniform(10, 90), 1), y=round(random.uniform(10, 90), 1)),
            dropoff=Location(x=round(random.uniform(10, 90), 1), y=round(random.uniform(10, 90), 1)),
            priority=random.choice(priorities),
            time_window=random.randint(tw_min, tw_max),
            weight=round(random.uniform(0.5, 8.0), 1),
        ))
    return orders


def generate_drivers(n: int, seed: int = 42) -> List[Driver]:
    """seed now matches the task seed (passed from config.seed) for full reproducibility."""
    random.seed(seed)
    drivers = []
    for i in range(n):
        drivers.append(Driver(
            driver_id=f"DRV-{i+1:02d}",
            location=Location(x=round(random.uniform(20, 80), 1), y=round(random.uniform(20, 80), 1)),
            capacity=random.choice([12.0, 15.0, 20.0]),
        ))
    return drivers
