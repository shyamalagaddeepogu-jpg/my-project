from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from environment.env import DeliveryDispatcherEnv
from environment.models import Action
from environment.tasks import TASKS
import os

app = Flask(__name__, static_folder='.')
CORS(app)

_envs = {}

def get_env(task_id):
    if task_id not in TASKS:
        return None, f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}"
    if task_id not in _envs:
        _envs[task_id] = DeliveryDispatcherEnv(task_id=task_id)
    return _envs[task_id], None

@app.route('/')
def root():
    return send_from_directory('.', 'dashboard.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/reset/<task_id>', methods=['POST'])
def reset_task(task_id):
    env, err = get_env(task_id)
    if err: return jsonify({"error": err}), 404
    obs = env.reset()
    return jsonify(obs.dict())

@app.route('/step/<task_id>', methods=['POST'])
def step_task(task_id):
    env, err = get_env(task_id)
    if err: return jsonify({"error": err}), 404
    data = request.get_json()
    action = Action.from_dict(data)
    result = env.step(action)
    return jsonify(result.dict())

@app.route('/state/<task_id>', methods=['GET'])
def state(task_id):
    env, err = get_env(task_id)
    if err: return jsonify({"error": err}), 404
    return jsonify(env.state())

@app.route('/tasks', methods=['GET'])
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

@app.route('/score/<task_id>', methods=['GET'])
def score_task(task_id):
    env, err = get_env(task_id)
    if err: return jsonify({"error": err}), 404
    return jsonify({"task_id": task_id, "score": env.final_score(), "stats": env._episode_stats()})


# ---- HACKATHON REQUIRED GENERIC ENDPOINTS ----

@app.route('/reset', methods=['POST'])
def reset_default():
    return reset_task("task_easy")

@app.route('/step', methods=['POST'])
def step_default():
    return step_task("task_easy")

@app.route('/score', methods=['GET'])
def score_default():
    return score_task("task_easy")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
