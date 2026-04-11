"""
server/app.py — Delivery Dispatcher OpenEnv
Required by openenv validate for multi-mode deployment.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=ROOT)
CORS(app)

_envs = {}
DEFAULT_TASK = "task_easy"


def _get_env(task_id):
    from env import DeliveryDispatcherEnv
    from tasks import TASKS
    if task_id not in TASKS:
        return None, f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}"
    if task_id not in _envs:
        _envs[task_id] = DeliveryDispatcherEnv(task_id=task_id)
    return _envs[task_id], None


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "delivery-dispatcher"})


@app.route("/")
def root():
    return send_from_directory(ROOT, "dashboard.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(ROOT, "dashboard.html")


# ── Bare /reset and /step for the openenv validator ping (defaults to task_easy) ──

@app.route("/reset", methods=["POST"])
def reset_default():
    env, err = _get_env(DEFAULT_TASK)
    if err:
        return jsonify({"error": err}), 404
    obs = env.reset()
    return jsonify(obs.dict())


@app.route("/step", methods=["POST"])
def step_default():
    env, err = _get_env(DEFAULT_TASK)
    if err:
        return jsonify({"error": err}), 404
    from models import Action
    data = request.get_json() or {}
    action = Action.from_dict(data)
    result = env.step(action)
    return jsonify(result.dict())


# ── Task-specific routes ──

@app.route("/reset/<task_id>", methods=["POST"])
def reset(task_id):
    env, err = _get_env(task_id)
    if err:
        return jsonify({"error": err}), 404
    obs = env.reset()
    return jsonify(obs.dict())


@app.route("/step/<task_id>", methods=["POST"])
def step(task_id):
    env, err = _get_env(task_id)
    if err:
        return jsonify({"error": err}), 404
    from models import Action
    data = request.get_json() or {}
    action = Action.from_dict(data)
    result = env.step(action)
    return jsonify(result.dict())


@app.route("/state/<task_id>", methods=["GET"])
def state(task_id):
    env, err = _get_env(task_id)
    if err:
        return jsonify({"error": err}), 404
    return jsonify(env.state())


@app.route("/tasks", methods=["GET"])
def list_tasks():
    from tasks import TASKS
    return jsonify({
        tid: {
            "description": t.description,
            "difficulty": t.difficulty,
            "num_orders": t.num_orders,
            "num_drivers": t.num_drivers,
            "max_steps": t.max_steps,
        }
        for tid, t in TASKS.items()
    })


@app.route("/score/<task_id>", methods=["GET"])
def score(task_id):
    env, err = _get_env(task_id)
    if err:
        return jsonify({"error": err}), 404
    return jsonify({
        "task_id": task_id,
        "score": env.final_score(),
        "stats": env._episode_stats(),
    })


def main():
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
