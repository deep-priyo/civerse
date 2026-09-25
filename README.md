---
title: CIVERSE
emoji: 💻
colorFrom: purple
colorTo: blue
sdk: docker
app_file: server/app.py
pinned: false
tags: [openenv, rl, code-review, bug-detection, agent-eval]
---

# CIVERSE — a reinforcement-learning environment for evaluating AI code reviewers

Most benchmarks ask a model to **write** code. CIVERSE asks whether a model can **read** code and
find what is wrong with it.

It is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment. Each episode hands
the agent a snippet containing bugs the agent cannot see, and scores it step by step on three separate
skills: whether it **finds** the bug, whether it **understands** what kind of bug it is, and whether
its **fix** is aimed at the right place. Any OpenEnv-compatible agent can be evaluated against it
without modification.

---

## Why code review rather than code generation

Generation benchmarks reward a model for producing something plausible. Review benchmarks punish it
for accepting something plausible. That difference matters: in practice, the expensive failure mode
of an AI coding assistant is not that it writes nothing, it is that it writes something that looks
correct and is not — and that a reviewer waves through.

Separating detection from classification from repair also makes the failure legible. A model that
scores well on detection and badly on classification is pattern-matching on "this line looks
suspicious" without understanding the defect. That distinction is invisible in a single pass/fail
number, and it is the main thing this environment is built to expose.

---

## Quick start

### Run the environment server

```bash
docker build -t civerse .
docker run -p 7860:7860 civerse
```

Or locally, without Docker:

```bash
pip install uv && uv pip install --system -r backend/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The server exposes the standard OpenEnv HTTP contract on port `7860`:

| Endpoint  | Method | Purpose                                        |
| --------- | ------ | ---------------------------------------------- |
| `/health` | GET    | Liveness check                                  |
| `/reset`  | POST   | Start a new episode, returns the first observation |
| `/step`   | POST   | Submit one action, returns `(observation, reward, done, state)` |
| `/state`  | GET    | Full environment state, including what the agent has found so far |
| `/grader` | POST   | Run the deterministic grader for a difficulty level |

### Evaluate a model

`inference.py` drives any OpenAI-compatible endpoint against the environment.

```bash
export HF_TOKEN=...                                  # or API_KEY
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"        # default
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

It prints one line per step and a summary at the end:

```
[START] task=code-review env=code-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=detect reward=0.30 done=false error=null
[STEP]  step=2 action=classify reward=-0.10 done=false error=null
[END]   success=true steps=9 score=0.750 rewards=0.30,-0.10,...
```

---

## How an episode works

The agent sees only the code. The bug list is held by the environment and never sent to the agent.

**Observation**

```json
{
  "code": "def add(a, b):\n    return a - b",
  "task_id": "e1",
  "step": 1
}
```

**Actions** — four types, each carrying an optional payload:

| Action     | Payload                                                  | What it claims                               |
| ---------- | -------------------------------------------------------- | -------------------------------------------- |
| `detect`   | `line_number`                                             | "There is a bug on this line."               |
| `classify` | `line_number`, `bug_type`, `severity`, `description`      | "It is this kind of bug."                    |
| `fix`      | `line_number`, `fix`                                      | "This is how to repair it."                  |
| `skip`     | —                                                         | "I am done." Ends the episode immediately.   |

```json
{"type": "classify", "payload": {"line_number": 6, "bug_type": "security", "severity": "critical"}}
```

Episodes run to a maximum of **15 steps**, or until the agent emits `skip`.

---

## Scoring

There are two layers, and they measure different things.

### Per-step reward — shapes behaviour during the episode

| Event                                                             | Reward |
| ----------------------------------------------------------------- | ------ |
| `detect` on a real, not-yet-detected bug                          | **+0.30** |
| `classify` with the correct line **and** the correct `bug_type`    | **+0.30** |
| `fix` on a real, not-yet-fixed bug                                | **+0.40** |
| Any action on a wrong line, or repeating one already credited      | **−0.10** |

Repair is weighted highest because it is the only action that requires the model to have understood
the defect rather than merely located it. The −0.10 penalty exists to make guessing unprofitable: an
agent that fires `detect` at every line scores worse than one that reads.

### Episode grade — the deterministic score

At the end of an episode the grader computes:

```
score = 0.5 × (bugs detected / total bugs) + 0.5 × (bugs fixed / total bugs)
```

clamped to `[0.01, 0.99]`. An episode counts as a **success at ≥ 0.50**.

> **Known inconsistency, not yet resolved:** the per-step reward scores three components, but the
> episode grade scores only two — classification accuracy is recorded in state and used for step
> reward, but does not enter the final number. `openenv.yaml` still advertises the three-way split
> (0.33 / 0.33 / 0.34). These should agree. See *Limitations* below.

---

## Difficulty levels

| Level      | Bugs  | Focus                                     | Baseline |
| ---------- | ----- | ----------------------------------------- | -------- |
| `easy`     | 1     | Simple logic error                        | 0.80     |
| `medium`   | 2–3   | Logic plus unhandled edge cases           | 0.50     |
| `hard`     | 3–5   | Security, logic and performance together  | 0.30     |
| `expert`   | 3–5   | Complex, interacting defects              | 0.20     |

Baselines descend deliberately. A benchmark where a strong model scores 0.9 everywhere tells you
nothing; the interesting signal is where the curve breaks.

**`easy` — one wrong operator.** Reward comes almost entirely from noticing.

```python
def add(a, b):
    return a - b
```

**`medium` — two absent guards.** Nothing is syntactically wrong; the bugs are the missing cases.

```python
def divide(a, b):
    return a / b

def get_element(arr, idx):
    return arr[idx]
```

**`hard` — three defects of different kinds in eight lines.** This is the level that separates
models, because it requires holding three categories of concern at once: a critical SQL injection, a
leaked connection, and absent error handling.

```python
import sqlite3

def query(user_id):
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute(f'SELECT * FROM users WHERE id={user_id}')
    res = c.fetchall()
    return res
```

Most models find the injection. Far fewer classify the unclosed connection as a *resource* problem
rather than a style one, and fewer still flag the missing error handling at all.

---

## Project layout

```
models.py                      # Task/Bug/Action/Observation schemas, environment, step reward
grader/code_review_graders.py  # Deterministic graders per difficulty + reference agent
backend/main.py                # OpenEnv Environment wrapper and FastAPI app
server/app.py                  # Uvicorn entrypoint (server.app:app)
inference.py                   # Evaluation harness for any OpenAI-compatible model
openenv.yaml                   # Environment spec: actions, tasks, scoring, constraints
Dockerfile                     # Python 3.11-slim, serves on 7860
```

Everything is typed with Pydantic, so a malformed action from a model is rejected at the boundary
rather than producing a confusing downstream failure — which matters when the thing under test is an
LLM emitting JSON.

`grader/` also contains `_heuristic_action`, an oracle agent that walks the known bugs in order. It
exists to verify the environment itself: if the oracle does not score near the ceiling, the
environment is broken, not the model.

---

## Limitations

Stated plainly, because a benchmark whose weaknesses are undocumented is not a benchmark.

1. **The task set is small and hand-written.** Four scenarios, one per difficulty. Enough to
   demonstrate the contract, not enough to rank models with confidence.
2. **Classification is missing from the episode grade.** See the note under *Scoring*. The fix is
   small — extend `_evaluate_state` to a three-term weighted sum matching the step rewards — and it
   should be done before any published comparison.
3. **Fix quality is positional, not semantic.** A `fix` action is credited for targeting the right
   line; the proposed patch text is stored but not verified. Proper scoring needs either test
   execution against the patched snippet or a semantic comparison to the reference fix.
4. **`deterministic_grader()` on the environment returns a constant.** It is a stub satisfying the
   interface, not a real implementation.
5. **No published baselines yet.** Scores for known models should be recorded so results are
   comparable across runs.
6. **Single-snippet episodes.** Real review happens across files and diffs, with context the agent
   must go find.

---

## Roadmap

- Three-component episode grade, consistent with the step rewards
- Semantic fix verification by running tests against the patched snippet
- A larger task set mined from real commits with labelled defects
- Published baseline scores for several frontier models
- Multi-file and diff-level review tasks

---

## Built with

Python 3.11 · FastAPI · Uvicorn · Pydantic · OpenEnv · Docker · Hugging Face Spaces · uv
