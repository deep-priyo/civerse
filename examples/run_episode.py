"""Run one scripted episode against a running CIVERSE server.

Start the server first (`uvicorn server.app:app --port 7860`), then:

    python examples/run_episode.py [easy|medium|hard|expert]

Uses OpenEnv's GenericEnvClient, which talks to the server's /ws endpoint so
that reset and every step land on the same environment instance.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.generic_client import GenericEnvClient

from grader.code_review_graders import _evaluate_state

BASE_URL = os.getenv("CIVERSE_URL", "http://localhost:7860")
TASK = sys.argv[1] if len(sys.argv) > 1 else "hard"

# A hand-written reviewer for the `hard` task: one correct find/classify/fix,
# one false positive, then stop. Swap this list for your agent's policy.
ACTIONS = [
    {"type": "detect",   "payload": {"line_number": 6}},
    {"type": "classify", "payload": {"line_number": 6, "bug_type": "security", "severity": "critical"}},
    {"type": "fix",      "payload": {"line_number": 6, "fix": "c.execute('SELECT * FROM users WHERE id=?', (user_id,))"}},
    {"type": "detect",   "payload": {"line_number": 1}},  # false positive
    {"type": "skip"},
]


def main() -> None:
    with GenericEnvClient(base_url=BASE_URL).sync() as env:
        result = env.reset(task_id=TASK)
        print(f"task={result.observation['task_id']}\n{result.observation['code']}\n")

        for action in ACTIONS:
            result = env.step(action)
            payload = json.dumps(action.get("payload", {}))
            print(f"{action['type']:<9} {payload[:60]:<62} reward={result.reward:+.2f} done={result.done}")
            if result.done:
                break

        # OpenEnv strips observation metadata on the wire, so the server-side
        # final_score never reaches the client. Grade the session state instead.
        grade = _evaluate_state(env.state())
        print(f"\nepisode grade: {grade:.2f} ({'pass' if grade >= 0.5 else 'fail'} at 0.50)")


if __name__ == "__main__":
    main()
