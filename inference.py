"""
inference.py — Delivery Dispatcher OpenEnv
"""

import os
import re
import json
import requests
from openai import OpenAI

# ── Environment variables — matching passing submission pattern ─────────────────
ENV_BASE_URL = os.environ.get("ENV_URL",      "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")     # NO default — checklist requirement

# ── Client at module level — exactly like passing submission ───────────────────


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
        f"score={score:.3f} rewards={','.join(f'{r:.3f}' for r in rewards)}",
        flush=True,
    )


# ── Environment API ────────────────────────────────────────────────────────────
def env_post(path, body=None):
    r = requests.post(f"{ENV_BASE_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def env_get(path):
    r = requests.get(f"{ENV_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a logistics dispatcher. "
    "Reply with ONLY a single valid JSON object — no explanation, no markdown."
)

def build_prompt(obs: dict) -> str:
    pending = obs.get("pending_orders", [])
    drivers = obs.get("available_drivers", [])

    orders_txt = "\n".join(
        f"  {o['order_id']}: priority={o['priority']} weight={o['weight']}kg "
        f"window={o['time_window']}min"
        for o in sorted(pending, key=lambda x: (x["priority"] != "high", x["time_window"]))
    ) or "  None"

    drivers_txt = "\n".join(
        f"  {d['driver_id']}: free="
        f"{round(d.get('effective_capacity', d['capacity']) - d.get('current_load', 0), 1)}kg"
        for d in drivers
    ) or "  None"

    return (
        f"PENDING ORDERS:\n{orders_txt}\n\n"
        f"AVAILABLE DRIVERS:\n{drivers_txt}\n\n"
        "Choose ONE action. Reply with exactly one of:\n"
        '{"action_type":"assign","driver_id":"DRV-01","order_id":"ORD-001"}\n'
        '{"action_type":"delay","order_id":"ORD-001"}\n'
        '{"action_type":"cancel","order_id":"ORD-001"}'
    )


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_action(obs: dict) -> dict:
    pending  = obs.get("pending_orders", [])
    fallback = (
        {"action_type": "delay", "order_id": pending[0]["order_id"]}
        if pending else {"action_type": "delay"}
    )

    api_base = os.environ.get("API_BASE_URL")

    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
            )
    
    client = OpenAI(base_url=api_base, api_key=api_key)

    prompt = build_prompt(obs)

    for attempt in range(1, 3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            m   = re.search(r"\{.*?\}", raw, re.DOTALL)
            action = json.loads(m.group() if m else raw)
            if "action_type" in action:
                return action
        except Exception as e:
            print(f"[ERROR] LLM attempt {attempt} failed: {e}", flush=True)

    return fallback


# ── Task runner ────────────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    log_start(task=task_id, env=ENV_BASE_URL, model=MODEL_NAME)

    try:
        obs = env_post(f"/reset/{task_id}")
    except Exception as e:
        print(f"[ERROR] reset failed for {task_id}: {e}", flush=True)
        log_end(False, 0, 0.0, [])
        return 0.0

    rewards, step_num, done = [], 0, False

    while not done and step_num < 200:
        if not obs.get("pending_orders"):
            break

        action    = get_action(obs)
        error_msg = None

        try:
            result    = env_post(f"/step/{task_id}", action)
            reward    = float(result["reward"]["value"])
            done      = bool(result["done"])
            obs       = result["observation"]
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
        final_score = (sum(rewards) / len(rewards)) if rewards else 0.0

    log_end(final_score > 0, step_num, final_score, rewards)
    return final_score


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task_id in TASKS:
        try:
            run_task(task_id)
        except Exception as e:
            print(f"[ERROR] task {task_id} crashed: {e}", flush=True)
