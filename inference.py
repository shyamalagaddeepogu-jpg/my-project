"""
inference.py — Baseline agent for Delivery Dispatcher OpenEnv
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_URL from environment variables
Emits: [START], [STEP], [END] structured stdout logs
"""

import os
import re
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
TASKS  = ["task_easy", "task_medium", "task_hard"]


def call_env(method: str, path: str, body: dict = None):
    url = f"{ENV_URL}{path}"
    r   = requests.post(url, json=body, timeout=30) if method == "POST" \
          else requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_prompt(obs: dict) -> str:
    pending = obs.get("pending_orders", [])
    drivers = obs.get("available_drivers", [])
    task_id = obs.get("task_id", "")

    orders_str = "\n".join(
        f"  - {o['order_id']}: priority={o['priority']}, weight={o['weight']}kg, "
        f"time_window={o['time_window']}min, "
        f"dropoff=({o['dropoff']['x']},{o['dropoff']['y']})"
        for o in sorted(
            pending,
            key=lambda o: (0 if o["priority"] == "high" else 1, o["time_window"])
        )
    )

    drivers_str = "\n".join(
        f"  - {d['driver_id']}: at=({d['location']['x']},{d['location']['y']}), "
        f"eff_capacity={d.get('effective_capacity', d['capacity'])}kg, "
        f"loaded={d['current_load']}kg, "
        f"free={round(d.get('effective_capacity', d['capacity']) - d['current_load'], 1)}kg, "
        f"deliveries_done={d.get('deliveries_done', 0)}"
        for d in drivers
    )

    traffic_note = (
        "NOTE: Rush-hour traffic is active early in the simulation — "
        "travel times are multiplied. Check est_delivery_time in step responses.\n"
        if obs.get("current_time", 0) < 30 and task_id in ("task_medium", "task_hard")
        else ""
    )

    fatigue_note = (
        "NOTE: Drivers fatigue after 3 deliveries — their effective_capacity drops 10% "
        "per tier. Rotate drivers to keep effective capacity high.\n"
        if task_id == "task_hard"
        else ""
    )

    return f"""You are a professional logistics dispatcher AI.

CURRENT STATE:
- Task: {task_id}
- Simulation time: {obs['current_time']} minutes
- Delivered: {obs['delivered_count']} | Failed: {obs['failed_count']} | Total: {obs['total_orders']}
{traffic_note}{fatigue_note}
PENDING ORDERS (sorted: HIGH priority first, then shortest time_window):
{orders_str if orders_str else "  None"}

AVAILABLE DRIVERS (showing effective capacity after fatigue):
{drivers_str if drivers_str else "  None"}

DECISION RULES (follow strictly in order):
1. HIGH priority orders first — they score 1.0 vs 0.8 and have SLA penalties
2. Shortest time_window next — expiring orders score 0 if they lapse
3. Choose the driver closest to the dropoff with enough FREE capacity
4. NEVER assign if driver current_load + order weight > driver effective_capacity
5. If a driver has high deliveries_done (3+), prefer fresher drivers — fatigue reduces their capacity
6. Use 'delay' only if NO driver can fit the order right now — costs 1 step but saves the order
7. Never cancel unless the order has already lapsed

Respond ONLY with a single valid JSON object, no explanation, no markdown:
{{"action_type": "assign", "driver_id": "DRV-01", "order_id": "ORD-001"}}
or
{{"action_type": "delay", "order_id": "ORD-001"}}"""


def get_action(obs: dict) -> dict:
    prompt   = build_prompt(obs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*?\}", raw, re.DOTALL)

    try:
        action = json.loads(match.group() if match else raw)
    except:
        pending = obs.get("pending_orders", [])
        action = (
            {"action_type": "delay", "order_id": pending[0]["order_id"]}
            if pending else {"action_type": "delay"}
        )

    if "action_type" not in action:
        pending = obs.get("pending_orders", [])
        return {"action_type": "delay", "order_id": pending[0]["order_id"]} if pending \
               else {"action_type": "delay"}
    return action


def run_task(task_id: str) -> float:
    print(json.dumps({"type": "[START]", "task_id": task_id, "model": MODEL_NAME}))

    obs      = call_env("POST", f"/reset/{task_id}")
    step_num = 0
    done     = False

    while not done:
        pending = obs.get("pending_orders", [])
        if not pending:
            break

        try:
            action = get_action(obs)
        except Exception as e:
            pending = obs.get("pending_orders", [])
            action  = ({"action_type": "delay", "order_id": pending[0]["order_id"]}
                       if pending else {"action_type": "delay"})
            print(json.dumps({"type": "[STEP]", "task_id": task_id, "step": step_num,
                               "action": action, "error": str(e)}))

        result = call_env("POST", f"/step/{task_id}", action)
        reward = result["reward"]["value"]
        done   = result["done"]
        obs    = result["observation"]

        print(json.dumps({
            "type":      "[STEP]",
            "task_id":   task_id,
            "step":      step_num,
            "action":    action,
            "reward":    reward,
            "done":      done,
            "delivered": obs["delivered_count"],
            "failed":    obs["failed_count"],
        }))

        step_num += 1

    score_data  = call_env("GET", f"/score/{task_id}")
    final_score = score_data["score"]
    stats       = score_data["stats"]

    print(json.dumps({
        "type":        "[END]",
        "task_id":     task_id,
        "final_score": final_score,
        "stats":       stats,
    }))
    return final_score


def main():
    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(task_id)

    print(json.dumps({
        "type":    "[END]",
        "task_id": "ALL",
        "scores":  scores,
        "average": round(sum(scores.values()) / len(scores), 4),
    }))


if __name__ == "__main__":
    main()
