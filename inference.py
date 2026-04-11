
"""
inference.py — Delivery Dispatcher OpenEnv
==========================================
Baseline inference script for the OpenEnv × Scaler Hackathon.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key
    ENV_URL        The running environment URL (default: http://localhost:7860)

OPTIONAL:
    TASK_NAME      Which task to run: task_easy | task_medium | task_hard
                   If not set, all three tasks are run sequentially.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    export ENV_URL=http://localhost:7860
    export API_BASE_URL=https://router.huggingface.co/v1
    export HF_TOKEN=hf_...
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export TASK_NAME=task_easy   # optional; omit to run all tasks
    python inference.py
"""

import os
import re
import json
import math
import requests

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "dummy-key")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860").rstrip("/")
BENCHMARK    = "delivery_dispatcher"

# If TASK_NAME is set, run only that task; otherwise run all three in order.
_TASK_ENV    = os.environ.get("TASK_NAME", "").strip()
ALL_TASKS    = ["task_easy", "task_medium", "task_hard"]

# Per-task step caps (match TaskConfig.max_steps in tasks.py)
TASK_MAX_STEPS = {
    "task_easy":   30,
    "task_medium": 60,
    "task_hard":   100,
}
DEFAULT_MAX_STEPS = 60

SUCCESS_SCORE_THRESHOLD = 0.1   # score ≥ this → success=true


# ── Structured logging (exact format required by evaluator) ───────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error=None) -> None:
    # Compact single-line JSON for the action field
    action_str = json.dumps(action, separators=(",", ":"))
    error_val  = str(error).replace("\n", " ") if error else "null"
    done_val   = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP helpers ───────────────────────────────────────────────────

def env_post(path: str, body=None, timeout: int = 30) -> dict:
    r = requests.post(f"{ENV_URL}{path}", json=body or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def env_get(path: str, timeout: int = 30) -> dict:
    r = requests.get(f"{ENV_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_reward(result: dict) -> float:
    """
    The /step response may return reward in different shapes depending on env version:
      - {"reward": 0.8}                        ← scalar
      - {"reward": {"value": 0.8}}             ← Pydantic model dict
      - {"reward": {"score": 0.8}}             ← alternate key
    """
    raw = result.get("reward", 0.0)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        for key in ("value", "score", "amount", "reward"):
            if key in raw:
                return float(raw[key])
    return 0.0


def _extract_obs(result: dict) -> dict:
    """Extract observation from /step result (handles nested or flat)."""
    obs = result.get("observation", result)
    return obs if isinstance(obs, dict) else {}


# ── Heuristic Agent (used as fallback when LLM is unavailable) ────────────────

class HeuristicAgent:
    """
    Priority-aware, capacity-checking greedy dispatcher.
    Strategy:
      1. Sort pending orders: HIGH priority first, then tightest time window.
      2. For the top-priority order, find the driver with the most free capacity
         that can still take the order (effective_capacity - current_load >= weight).
      3. If no driver fits, delay the order.
      4. If no pending orders, return a no-op delay.
    """

    @staticmethod
    def _euclidean(a: dict, b: dict) -> float:
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def get_action(self, obs: dict) -> dict:
        pending = obs.get("pending_orders", [])
        drivers = obs.get("available_drivers", [])

        if not pending:
            return {"action_type": "delay"}

        # Sort: HIGH priority first, then tightest window
        sorted_orders = sorted(
            pending,
            key=lambda o: (0 if o.get("priority") == "high" else 1, o.get("time_window", 9999)),
        )

        for order in sorted_orders:
            weight   = order.get("weight", 0)
            pickup   = order.get("pickup", {"x": 0, "y": 0})
            order_id = order["order_id"]

            # Drivers that can take this order, sorted by proximity then free capacity
            eligible = []
            for d in drivers:
                eff_cap   = d.get("effective_capacity", d.get("capacity", 0))
                cur_load  = d.get("current_load", 0)
                free      = eff_cap - cur_load
                if free >= weight:
                    dist = self._euclidean(d.get("location", {"x": 0, "y": 0}), pickup)
                    eligible.append((dist, -free, d["driver_id"]))

            if eligible:
                eligible.sort()
                best_driver = eligible[0][2]
                return {
                    "action_type": "assign",
                    "driver_id":   best_driver,
                    "order_id":    order_id,
                }

            # No driver fits — delay this order
            return {"action_type": "delay", "order_id": order_id}

        # Fallback
        return {"action_type": "delay"}


# ── LLM Agent ──────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Uses an OpenAI-compatible LLM to decide dispatch actions.
    Falls back to HeuristicAgent on any error.
    """

    SYSTEM_PROMPT = (
        "You are an expert delivery dispatcher. "
        "Each turn you receive a list of pending orders and available drivers. "
        "You must reply with EXACTLY one JSON object and nothing else.\n"
        "Valid formats:\n"
        '  {"action_type":"assign","driver_id":"DRV-01","order_id":"ORD-001"}\n'
        '  {"action_type":"delay","order_id":"ORD-001"}\n'
        "Rules:\n"
        "- Prefer HIGH priority orders and tightest time windows.\n"
        "- Check that driver free capacity >= order weight before assigning.\n"
        "- Use delay only when no driver can fit the order.\n"
        "- Never output anything other than the JSON object."
    )

    def __init__(self):
        self._fallback = HeuristicAgent()
        self._client   = None
        if _OPENAI_OK:
            try:
                self._client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            except Exception as exc:
                print(f"[WARN] OpenAI client init failed: {exc}", flush=True)

    def _build_user_prompt(self, obs: dict) -> str:
        pending = obs.get("pending_orders", [])
        drivers = obs.get("available_drivers", [])

        orders_lines = "\n".join(
            f"  {o['order_id']}: priority={o.get('priority','?')} "
            f"weight={o.get('weight','?')}kg window={o.get('time_window','?')}min"
            for o in sorted(pending, key=lambda x: (x.get("priority") != "high", x.get("time_window", 9999)))
        ) or "  (none)"

        drivers_lines = "\n".join(
            f"  {d['driver_id']}: free_capacity="
            f"{round(d.get('effective_capacity', d.get('capacity', 0)) - d.get('current_load', 0), 2)}kg"
            f" location=({d.get('location',{}).get('x',0):.1f},{d.get('location',{}).get('y',0):.1f})"
            for d in drivers
        ) or "  (none)"

        return (
            f"PENDING ORDERS:\n{orders_lines}\n\n"
            f"AVAILABLE DRIVERS:\n{drivers_lines}\n\n"
            "Reply with exactly one JSON action."
        )

    def get_action(self, obs: dict) -> dict:
        if self._client is None:
            return self._fallback.get_action(obs)

        prompt = self._build_user_prompt(obs)
        for attempt in range(1, 3):
            try:
                resp = self._client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=120,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content or ""
                raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                # Extract the first JSON object
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                if m:
                    action = json.loads(m.group())
                    if "action_type" in action:
                        return action
            except Exception as exc:
                print(f"[WARN] LLM attempt {attempt} failed: {exc}", flush=True)

        # All LLM attempts failed — use heuristic
        return self._fallback.get_action(obs)


# ── Single-task episode runner ────────────────────────────────────────────────

def run_task(task_id: str, agent) -> float:
    """
    Run one full episode for `task_id`.
    Emits [START], one [STEP] per step, and [END].
    Returns the episode score (0.0–1.0).
    """
    max_steps = TASK_MAX_STEPS.get(task_id, DEFAULT_MAX_STEPS)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # ── Reset ──────────────────────────────────────────────────────────────────
    try:
        obs = env_post(f"/reset/{task_id}")
    except Exception as exc:
        print(f"[ERROR] reset failed for {task_id}: {exc}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    rewards:   list  = []
    step_num:  int   = 0
    done:      bool  = False
    score:     float = 0.0
    success:   bool  = False

    try:
        for step_num in range(1, max_steps + 1):
            # Check termination from previous step or empty board
            if done or not obs.get("pending_orders"):
                step_num -= 1   # we didn't actually take a step
                break

            # ── Decide action ──────────────────────────────────────────────────
            action = agent.get_action(obs)
            error_msg = None

            # ── Execute action ─────────────────────────────────────────────────
            try:
                result  = env_post(f"/step/{task_id}", action)
                reward  = _extract_reward(result)
                done    = bool(result.get("done", False))
                obs     = _extract_obs(result)
            except Exception as exc:
                error_msg = str(exc)
                reward    = 0.0
                done      = True     # treat HTTP error as terminal

            rewards.append(reward)
            log_step(step=step_num, action=action, reward=reward, done=done, error=error_msg)

            if done:
                break

        # ── Final score ────────────────────────────────────────────────────────
        try:
            score_resp = env_get(f"/score/{task_id}")
            score = float(score_resp.get("score", 0.0))
        except Exception as exc:
            print(f"[WARN] score fetch failed for {task_id}: {exc}", flush=True)
            # Estimate from accumulated rewards (normalised)
            score = min(sum(rewards) / max(len(rewards) * 1.0, 1.0), 1.0) if rewards else 0.0

        score   = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always emit [END]
        log_end(success=success, steps=step_num, score=score, rewards=rewards)

    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Choose agent
    use_llm = _OPENAI_OK and bool(os.environ.get("API_BASE_URL")) and bool(os.environ.get("HF_TOKEN"))
    if use_llm:
        print(f"[INFO] LLM mode: model={MODEL_NAME} base_url={API_BASE_URL}", flush=True)
        agent = LLMAgent()
    else:
        print("[INFO] Heuristic mode (LLM env vars not set or openai not installed)", flush=True)
        agent = HeuristicAgent()

    # Decide which tasks to run
    tasks_to_run = [_TASK_ENV] if _TASK_ENV else ALL_TASKS

    all_scores = {}
    for task_id in tasks_to_run:
        try:
            all_scores[task_id] = run_task(task_id, agent)
        except Exception as exc:
            print(f"[ERROR] task {task_id} crashed: {exc}", flush=True)
            all_scores[task_id] = 0.0

    # Summary (informational only — not part of the evaluator format)
    if len(tasks_to_run) > 1:
        avg = sum(all_scores.values()) / len(all_scores)
        print(
            f"[INFO] All tasks done. Scores: "
            + ", ".join(f"{t}={s:.3f}" for t, s in all_scores.items())
            + f" | avg={avg:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
