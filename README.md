---
title: Delivery Dispatcher
emoji: 🚚
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - logistics
  - dispatch
  - scheduling
---

# 🚚 Delivery Dispatcher — OpenEnv

> A real-world AI training environment for delivery dispatch optimisation.  
> An AI agent acts as a dispatcher — assigning drivers to orders, managing capacity, navigating traffic, and beating time windows.

---

## Live Dashboard

Once deployed, visit `/dashboard` to see the real-time dispatch control room:
- Live city map with orders and drivers
- Real-time assignment animations
- Score tracker and dispatch log
- Run the AI agent live and watch it dispatch

---

## What This Simulates

Delivery dispatch is a genuine operations problem faced by companies like Swiggy, Zomato, Amazon Last Mile, and Uber Eats every second. The agent must:

- Read a city map of pending orders and available drivers
- Assign the right driver to the right order (capacity, proximity, priority)
- Respect time windows or face penalties
- Navigate rush-hour traffic that slows travel times
- Rotate drivers proactively as fatigue reduces their effective capacity
- Handle dynamic mid-episode cancellations

---

## Why the Hard Task is Challenging

The hard task introduces dynamic order cancellations, tighter delivery time windows,
and reduced driver efficiency due to fatigue. Combined with traffic variations,
this forces the agent to make real-time trade-offs between urgency, capacity,
and routing — making it challenging even for advanced AI models.

---

## Architecture

```
environment/
├── env.py       ← core DeliveryDispatcherEnv
├── models.py    ← typed Observation, Action, Reward, Driver, Order
├── tasks.py     ← TaskConfig, TASKS dict, grade_episode(), traffic helper
└── data.py      ← procedural order and driver generation (seeded)

app.py           ← Flask server exposing REST endpoints
dashboard.html   ← live visual control room (served at /dashboard)
inference.py     ← baseline agent loop (OpenAI-compatible API)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset/{task_id}` | Start a fresh episode |
| POST | `/step/{task_id}` | Submit one action, get next observation |
| GET  | `/state/{task_id}` | Full environment state snapshot |
| GET  | `/score/{task_id}` | Episode score + all stats |
| GET  | `/tasks` | List all available tasks |
| GET  | `/dashboard` | Live visual dashboard |

---

## Action Space

```json
{"action_type": "assign", "driver_id": "DRV-01", "order_id": "ORD-001"}
{"action_type": "delay",  "order_id": "ORD-003"}
{"action_type": "cancel", "order_id": "ORD-005"}
```

| Action | Effect |
|--------|--------|
| `assign` | Assigns a driver to a pending order. Triggers capacity check (using effective capacity after fatigue), traffic-scaled travel time estimate, SLA check, and immediate delivery simulation. |
| `delay` | Extends an order's time window by 10 minutes. Returns +0.1 reward. Use when no driver currently fits. |
| `cancel` | Drops a pending order. Zero reward. Last resort only. |

---

## Observation Space

```json
{
  "pending_orders": [
    {
      "order_id": "ORD-001",
      "priority": "high",
      "weight": 3.2,
      "time_window": 25,
      "pickup":  {"x": 12.4, "y": 7.1},
      "dropoff": {"x": 18.0, "y": 3.5}
    }
  ],
  "available_drivers": [
    {
      "driver_id":          "DRV-01",
      "capacity":           15.0,
      "effective_capacity": 13.5,
      "current_load":       0.0,
      "location":           {"x": 10.0, "y": 5.0},
      "deliveries_done":    3
    }
  ],
  "current_time":    10,
  "delivered_count": 2,
  "failed_count":    0,
  "total_orders":    10
}
```

`effective_capacity` = nominal capacity reduced by 10% per fatigue tier (every 3 deliveries, capped at -40%).

---

## Tasks

| Task | Difficulty | Orders | Drivers | Special Rules |
|------|------------|--------|---------|---------------|
| `task_easy`   | 🟢 Easy   | 3  | 3 | Relaxed windows, no traffic, no fatigue |
| `task_medium` | 🟡 Medium | 10 | 5 | Capacity constraints, rush-hour traffic (first 30 sim-min, 1.6× travel), SLA tracking |
| `task_hard`   | 🔴 Hard   | 20 | 7 | Driver fatigue, two traffic bands (rush 1.8×, off-peak 0.85×), dynamic cancellations, tight SLAs |

---

## Reward Function

### Per-step rewards (returned by `/step`)

| Event | Reward |
|-------|--------|
| On-time delivery — high priority | `+1.0` |
| On-time delivery — normal / low  | `+0.8` |
| Late delivery | `+0.3` |
| Capacity violation attempt | `+0.05` with `−0.1` partial penalty |
| High-priority SLA breach | `−0.1` partial penalty |
| Delay action | `+0.1` |
| Cancel action | `0.0` |

### Episode score (returned by `/score`)

Computed by `grade_episode()` in `tasks.py`:

**Easy task** — `0.80 × delivery_ratio + 0.20 × on_time_rate + bonuses − penalties`

**Medium task** — `0.45 × delivery_ratio + 0.35 × on_time_rate + bonuses − penalties`

**Hard task** — `0.35 × delivery_ratio + 0.30 × on_time_rate + bonuses − penalties`

Bonuses (all tasks):

| Bonus | Max | Description |
|-------|-----|-------------|
| Early-delivery bonus | +0.15 | Average `(time_margin / time_window)` across on-time deliveries |
| Driver utilisation bonus | +0.10 | Low variance in per-driver delivery counts = higher score |
| Priority adherence bonus | +0.10 | Fraction of HIGH orders delivered on time |
| Step efficiency bonus | +0.05 | Agent finishes before the step limit |

---

## Episode Stats

`/score` and `/state` return:

```json
{
  "delivered": 8,
  "failed": 2,
  "total_orders": 10,
  "on_time_rate": 0.75,
  "capacity_violations": 1,
  "priority_sla_breaches": 0,
  "assigned_to_cancelled": 1,
  "step_count": 22,
  "driver_utilisation_rate": 0.8,
  "avg_time_margin": 6.4,
  "high_priority_on_time_rate": 1.0
}
```

---

## Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| `task_easy`   | ~0.90 | gpt-4o-mini greedy agent |
| `task_medium` | ~0.68 | gpt-4o-mini greedy agent — traffic causes some late deliveries |
| `task_hard`   | ~0.38 | gpt-4o-mini greedy agent — fatigue + cancellations are genuinely hard |

Scores above these baselines indicate genuine optimisation (route-aware assignment, priority sorting, fleet rotation, proactive delay use).

---

## Setup

```bash
# Local
pip install -r requirements.txt
python app.py
# → http://localhost:7860
# → http://localhost:7860/dashboard

# Docker
docker build -t delivery-dispatcher .
docker run -p 7860:7860 delivery-dispatcher
```

### Running the inference agent

```bash
export ENV_URL=http://localhost:7860
export HF_TOKEN=your_openai_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py
```

---

## Design Decisions

- **Seeded procedural generation** — same seed always produces the same episode. Scores are fully reproducible and comparable across agents.
- **Instantaneous delivery simulation** — delivery outcome is computed at assignment time using a distance-based travel estimate, keeping the environment stateless between steps. This is intentional: it rewards the agent for making the right assignment decision upfront, not for reacting to in-flight changes.
- **Traffic multipliers** — medium and hard tasks include time-of-day traffic bands (rush hour, off-peak). A naive agent ignores these and over-commits on tight orders during rush hour. A smart agent checks `est_delivery_time` in the step response and uses `delay` proactively.
- **Driver fatigue** — hard task drivers lose 10% effective capacity every 3 deliveries (capped at −40%). A naive agent stacks all orders on the nearest driver until it hits a capacity wall. A smart agent rotates the fleet to keep effective capacity high across all drivers.
- **Partial credit rewards** — the grader never gives pure binary scores. Even a late delivery scores +0.3, encouraging agents to attempt all orders rather than cherry-pick easy ones.
- **Driver utilisation signal** — a naive agent always picks the nearest driver. A smart agent spreads load evenly. The utilisation bonus rewards this explicitly and is visible in `driver_usage` stats.


## Sample Output

```json
{
  "score": 0.78,
  "on_time_rate": 0.65,
  "driver_utilisation": 0.58
}