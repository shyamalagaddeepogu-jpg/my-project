"""
inference.py — Delivery Dispatcher OpenEnv
"""

import os
import re
import json
import requests

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://host.docker.internal:7860")
BENCHMARK    = "delivery_dispatcher"

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM = (
    "You are a logistics dispatcher. "
    "Reply with ONLY a single valid JSON object — no explanation, no markdown."
)


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


# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(obs):
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
        '{"action_type":"delay","order_id":"ORD-001"}'
    )


# ── Heuristic Agent ────────────────────────────────────────────────────────────
class HeuristicAgent:
    def get_action(self, obs):
        pending = obs.get("pending_orders", [])
        if not pending:
            return {"action_type": "delay"}
        order = sorted(
            pending,
            key=lambda o: (o["priority"] != "high", o["time_window"])
        )[0]
        for d in obs.get("available_drivers", []):
            free = d.get("effective_capacity", d["capacity"]) - d.get("current_load", 0)
            if free >= order["weight"]:
                return {"action_type": "assign", "driver_id": d["driver_id"], "order_id": order["order_id"]}
        return {"action_type": "delay", "order_id": order["order_id"]}


# ── LLM Agent ──────────────────────────────────────────────────────────────────
class LLMAgent:
    def __init__(self):
        self.api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
        self.hf_token     = os.environ.get("HF_TOKEN",     "")
        self.client       = OpenAI(
            api_key=self.hf_token,
            base_url=self.api_base_url,
        )
        self.heuristic = HeuristicAgent()

    def get_action(self, obs):
        prompt = build_prompt(obs)
        for attempt in range(1, 3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
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
        return self.heuristic.get_action(obs)


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

        action    = agent.get_action(obs)
        error_msg = None

        try:
            result = env_post(f"/step/{task_id}", action)
            reward = float(result["reward"]["value"])
            done   = bool(result["done"])
            obs    = result["observation"]
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
    # Exact pattern from passing submission:
    # use LLM only if both API_BASE_URL and HF_TOKEN are set
    use_llm = (
        os.environ.get("API_BASE_URL")
        and os.environ.get("HF_TOKEN")
        and OPENAI_AVAILABLE
    )

    if use_llm:
        print(f"[INFO] LLM mode: model={MODEL_NAME} base_url={API_BASE_URL}", flush=True)
        agent = LLMAgent()
    else:
        print("[INFO] Heuristic mode (API_BASE_URL or HF_TOKEN not set)", flush=True)
        agent = HeuristicAgent()

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
