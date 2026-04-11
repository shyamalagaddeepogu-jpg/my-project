"""
server/app.py — Delivery Dispatcher OpenEnv
"""

import os
import sys

# Make sure root is in path so 'environment' package is found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from environment.env import DeliveryDispatcherEnv
from environment.models import Action
from environment.tasks import TASKS

app = Flask(__name__, static_folder=ROOT)
CORS(app)

_envs = {}
DEFAULT_TASK = "task_easy"


def _get_env(task_id):
    if task_id not in TASKS:
        return None, f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}"
    if task_id not in _envs:
        _envs[task_id] = DeliveryDispatcherEnv(task_id=task_id)
    return _envs[task_id], None


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/")
def root():
    return send_from_directory(ROOT, "dashboard.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(ROOT, "dashboard.html")


# Bare routes for validator
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
    data = request.get_json() or {}
    action = Action.from_dict(data)
    result = env.step(action)
    return jsonify(result.dict())


@app.route("/score", methods=["GET"])
def score_default():
    env, err = _get_env(DEFAULT_TASK)
    if err:
        return jsonify({"error": err}), 404
    return jsonify({"task_id": DEFAULT_TASK, "score": env.final_score(), "stats": env._episode_stats()})


# Task-specific routes
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
    return jsonify({"task_id": task_id, "score": env.final_score(), "stats": env._episode_stats()})


def main():
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
