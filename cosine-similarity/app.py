from flask import Flask, render_template, request, jsonify
import random
import re

from models.similarity import SimilarityChecker

app = Flask(__name__)

_tasks = []
_current_task = None
checker = SimilarityChecker()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start():
    global _tasks, _current_task

    data = request.get_json()
    tasks_text = data.get('tasks', '')

    tasks = re.findall(r'[^\n]+', tasks_text.strip())
    tasks = [t.strip() for t in tasks if t.strip()]

    if not tasks:
        return jsonify({'error': 'No tasks provided'}), 400

    _tasks = tasks
    _current_task = random.choice(tasks)

    return jsonify({'task': _current_task})


@app.route('/api/check', methods=['POST'])
def check():
    if not _current_task:
        return jsonify({'error': 'No active task'}), 400

    data = request.get_json()
    paraphrase = data.get('paraphrase', '').strip()

    if not paraphrase:
        return jsonify({'error': 'No paraphrase provided'}), 400

    result = checker.check(_current_task, paraphrase)
    return jsonify(result)


@app.route('/api/next', methods=['POST'])
def next_task():
    global _tasks, _current_task

    if not _tasks or len(_tasks) < 2:
        return jsonify({'error': 'Need at least 2 tasks'}), 400

    available_tasks = [t for t in _tasks if t != _current_task]
    _current_task = random.choice(available_tasks)

    return jsonify({'task': _current_task})


if __name__ == '__main__':
    app.run(debug=True, port=5000)