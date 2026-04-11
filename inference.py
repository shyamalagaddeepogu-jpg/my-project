"""
inference.py — Delivery Dispatcher OpenEnv
FIXED: Always uses API_BASE_URL and API_KEY from environment (hackathon proxy)
"""

import os
import re
import json
import requests
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]          # MUST come from env — no default
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.environ.get("ENV_URL", "http://host.docker.internal:7860")

BENCHMARK = "delivery_dispatcher"
TASKS = ["task_easy", "task_medium", "task_hard"]

# ── Logging ────────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ── Environment API ────────────────────────────────────────────────────────────
def env_post(path, body=None):
    r = requests.post(f"{ENV_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def env_get(path):
    r = requests.get(f"{ENV_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM Agent (always routes through proxy) ───────────────────────────────────
class LLMAgent:
    def __init__(self):
        # CRITICAL: use the injected API_BASE_URL and API_KEY — never hardcode
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        self.model = MODEL_NAME

    def get_action(self, obs):
        prompt = self._build_prompt(obs)
        for attempt in range(1, 4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                action = json.loads(m.group() if m else raw)
                if "action_type" in action:
                    return action
            except Exception as e:
                print(f"[ERROR] LLM attempt {attempt} failed: {e}", flush=True)
        # last resort fallback (still logged, proxy was already called above)
        return self._heuristic_fallback(obs)

    def _heuristic_fallback(self, obs):
        pending = obs.get("pending_orders", [])
        if not pending:
            return {"action_type": "delay"}
        order = sorted(pending, key=lambda o: (o["priority"] != "high", o["time_window"]))[0]
        for d in obs.get("available_drivers", []):
            free = d.get("effective_capacity", d["capacity"]) - d.get("current_load", 0)
            if free >= order["weight"]:
                return {"action_type": "assign", "driver_id": d["driver_id"], "order_id": order["order_id"]}
        return {"action_type": "delay", "order_id": order["order_id"]}

    def _build_prompt(self, obs):
        pending = obs.get("pending_orders", [])
        drivers = obs.get("available_drivers", [])
        current_time = obs.get("current_time", 0)

        orders_txt = "\n".join(
            f"  {o['order_id']}: priority={o['priority']} weight={o['weight']}kg "
            f"window={o['time_window']}min pickup=({o['pickup']['x']:.1f},{o['pickup']['y']:.1f}) "
            f"dropoff=({o['dropoff']['x']:.1f},{o['dropoff']['y']:.1f})"
            for o in sorted(pending, key=lambda x: (x["priority"] != "high", x["time_window"]))
        ) or "  None"

        drivers_txt = "\n".join(
            f"  {d['driver_id']}: free={round(d.get('effective_capacity', d['capacity']) - d.get('current_load', 0), 1)}kg "
            f"loc=({d['location']['x']:.1f},{d['location']['y']:.1f}) done={d.get('deliveries_done', 0)}"
            for d in drivers
        ) or "  None"

        return (
            f"You are a delivery dispatcher. Current time: {current_time} min.\n\n"
            f"PENDING ORDERS:\n{orders_txt}\n\n"
            f"AVAILABLE DRIVERS:\n{drivers_txt}\n\n"
            "Rules:\n"
            "- Assign high-priority orders first\n"
            "- Check driver free capacity >= order weight\n"
            "- Pick the driver closest to the pickup location\n"
            "- Use 'delay' only if no driver has enough capacity\n\n"
            "Reply with ONLY ONE JSON object, nothing else:\n"
            '{"action_type":"assign","driver_id":"DRV-01","order_id":"ORD-001"}\n'
            'or {"action_type":"delay","order_id":"ORD-001"}'
        )

# ── Task runner ────────────────────────────────────────────────────────────────
def run_task(task_id, agent):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env_post(f"/reset/{task_id}")
    except Exception as e:
        print(f"[ERROR] reset failed for {task_id}: {e}", flush=True)
        log_end(False, 0, 0.0, [])
        return 0.0

    rewards, step_num, done = [], 0, False
    while not done:
        if not obs.get("pending_orders"):
            break
        action = agent.get_action(obs)
        error_msg = None
        try:
            result = env_post(f"/step/{task_id}", action)
            reward = float(result["reward"]["value"])
            done = bool(result["done"])
            obs = result["observation"]
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] step {step_num + 1} failed: {e}", flush=True)
            reward, done = 0.0, True

        step_num += 1
        rewards.append(reward)
        log_step(step_num, action, reward, done, error_msg)

    try:
        final_score = float(env_get(f"/score/{task_id}")["score"])
    except Exception as e:
        print(f"[ERROR] score fetch failed: {e}", flush=True)
        final_score = 0.0

    log_end(final_score > 0, step_num, final_score, rewards)
    return final_score

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] LLM mode: base_url={API_BASE_URL} model={MODEL_NAME}", flush=True)
    agent = LLMAgent()
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id, agent)
        except Exception as e:
            print(f"[ERROR] task {task_id} crashed: {e}", flush=True)
            scores[task_id] = 0.0

    avg = round(sum(scores.values()) / len(scores), 4)
    print(f"[END] task=ALL score={avg}", flush=True)

if __name__ == "__main__":
    main()
