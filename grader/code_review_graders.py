import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Task, generate_tasks, CodeReviewEnvironment, ActionPayload

_MIN = 0.01
_MAX = 0.99


def _safe(raw) -> float:
    try:
        return round(max(_MIN, min(_MAX, float(raw))), 4)
    except Exception:
        return _MIN


def _heuristic_action(env: CodeReviewEnvironment) -> Action:
    state = env.state
    kb = state.task.known_bugs

    detected_lines = [d.get("line_number") for d in state.detected_bugs]
    classified_lines = [d.get("line_number") for d in state.classified_bugs]
    fixed_lines = [d.get("line_number") for d in state.proposed_fixes]

    for b in kb:
        if b.line not in detected_lines:
            return Action(type="detect", payload=ActionPayload(line_number=b.line))
        if b.line not in classified_lines:
            return Action(type="classify",
                          payload=ActionPayload(line_number=b.line, bug_type=b.type, severity=b.severity))
        if b.line not in fixed_lines:
            return Action(type="fix", payload=ActionPayload(line_number=b.line, fix=b.fix))

    return Action(type="skip")


def _evaluate_state(state_dict: dict) -> float:
    if not state_dict or "task" not in state_dict:
        return _MIN
    task = state_dict.get("task", {})
    known_bugs = task.get("known_bugs", [])
    if not known_bugs:
        return _MAX

    detected = state_dict.get("detected_bugs", [])
    classified = state_dict.get("classified_bugs", [])
    fixed = state_dict.get("proposed_fixes", [])

    total_bugs = len(known_bugs)

    detected_lines = {d.get("line_number") for d in detected}
    found = sum(1 for b in known_bugs if b.get('line') in detected_lines)
    recall = found / total_bugs

    # Classification only counts when the agent named the right defect on the right
    # line, matching the per-step reward in CodeReviewEnvironment.step(). Locating a
    # bug without understanding what kind it is earns detection credit, not this.
    classified_pairs = {(c.get("line_number"), c.get("bug_type")) for c in classified}
    classified_correct = sum(
        1 for b in known_bugs if (b.get('line'), b.get('type')) in classified_pairs
    )
    classify_recall = classified_correct / total_bugs

    fixed_lines = {f.get("line_number") for f in fixed}
    fixed_correct = sum(1 for b in known_bugs if b.get('line') in fixed_lines)
    fix_recall = fixed_correct / total_bugs

    # Weights mirror openenv.yaml scoring.components and the 0.30/0.30/0.40 split of
    # the per-step rewards: repair is worth marginally more than locating or naming,
    # because it is the only action that requires having understood the defect.
    score = (0.33 * recall) + (0.33 * classify_recall) + (0.34 * fix_recall)
    return max(_MIN, min(_MAX, score))


def _run_episode(difficulty: str) -> tuple:
    try:
        tasks = generate_tasks(difficulty)
        env = CodeReviewEnvironment(tasks=tasks, max_steps=15)
        env.reset()
        done = False
        while not done:
            action = _heuristic_action(env)
            _, _, done, _ = env.step(action)

        score = _evaluate_state(env.state.model_dump())
        score = _safe(score)
        return score, score >= 0.5, f"CR {difficulty} | score={score:.4f}"
    except Exception as e:
        return _MIN, False, f"Grader error: {e}"


def _from_trajectory(trajectory: dict, difficulty: str) -> tuple:
    if trajectory and "task" in trajectory:
        score = _safe(_evaluate_state(trajectory))
        return score, score >= 0.5, f"CR {difficulty} | score={score:.4f}"
    return _run_episode(difficulty)


class EasyGrader:
    def grade(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "easy")

    def __call__(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "easy")[0]


class MediumGrader:
    def grade(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "medium")

    def __call__(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "medium")[0]


class HardGrader:
    def grade(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "hard")

    def __call__(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "hard")[0]


class ExpertGrader:
    def grade(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "expert")

    def __call__(self, trajectory=None, *a, **kw): return _from_trajectory(trajectory or {}, "expert")[0]
