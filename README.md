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

# CIVERSE

**A reinforcement-learning environment that measures how well AI agents _review_ code, not how well they write it.**

Most coding benchmarks measure generation: can the model produce code that passes the tests? In
practice, the costly failure of an AI coding assistant is code that *looks* correct but isn't, and
that a reviewer approves. Generation benchmarks don't measure the opposite skill: reading
unfamiliar code, finding the defect, saying what kind of defect it is, and pointing the repair at
the right place.

CIVERSE turns that skill into an episodic RL environment. An agent gets a code snippet with bugs
it can't see. It reviews the snippet one action at a time and earns separate credit for
**finding** each bug, **classifying** it, and **fixing** it. Scoring the three separately makes the
failure visible. An agent that scores well on finding bugs but badly on classifying them is
reacting to lines that look suspicious without understanding what is wrong with them.

It implements the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) server contract, so any
OpenEnv client can drive it unmodified.

> **Status:** a working reference implementation with a small, hand-written task set. The
> environment contract, reward shaping and serving layer are complete. The task set is too small to
> rank models, and the grade can be gamed (see [Limitations](#limitations)).

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [How an episode works](#how-an-episode-works)
- [Tasks and difficulty levels](#tasks-and-difficulty-levels)
- [Scoring](#scoring)
- [API reference](#api-reference)
- [Evaluating an LLM](#evaluating-an-llm)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Design decisions and what I'd change](#design-decisions-and-what-id-change)

---

## Installation

Requires Python 3.11+. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pip install -e .` and `uv sync` also work; `pyproject.toml` and `uv.lock` describe the same
dependency set.

Or with Docker:

```bash
docker build -t civerse .
docker run -p 7860:7860 civerse
```

## Quickstart

Start the server:

```bash
uvicorn server.app:app --port 7860
```

In a second terminal, run one scripted episode against the `hard` task:

```bash
python examples/run_episode.py hard
```

```text
task=h1
import sqlite3

def query(user_id):
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute(f'SELECT * FROM users WHERE id={user_id}')
    res = c.fetchall()
    return res

detect    {"line_number": 6}                                             reward=+0.30 done=False
classify  {"line_number": 6, "bug_type": "security", "severity": "crit   reward=+0.30 done=False
fix       {"line_number": 6, "fix": "c.execute('SELECT * FROM users WH   reward=+0.40 done=False
detect    {"line_number": 1}                                             reward=-0.10 done=False
skip      {}                                                             reward=+0.00 done=True

episode grade: 0.33 (fail at 0.50)
```

The scripted reviewer handles the SQL injection fully and reports one false positive. It still
fails: it found one of the three bugs, and the grade measures how many of the known bugs were
found. [`examples/run_episode.py`](examples/run_episode.py) is about 40 lines; replace its
`ACTIONS` list with your agent's policy.

The core loop, using OpenEnv's generic client:

```python
from openenv.core.generic_client import GenericEnvClient

with GenericEnvClient(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_id="medium")          # easy | medium | hard | expert
    print(result.observation["code"])
    result = env.step({"type": "detect", "payload": {"line_number": 2}})
    print(result.reward, result.done)             # 0.3 False
```

---

## How an episode works

```
reset(task_id) ─► observation ─► agent ─► action ─► step() ─► reward, observation, done
                                   ▲                                  │
                                   └──────────────────────────────────┘
                     until the agent sends `skip` or 15 steps have elapsed
```

### Observation space

```json
{ "code": "def add(a, b):\n    return a - b", "task_id": "e1", "step": 0 }
```

| Field     | Type   | Meaning                                         |
| --------- | ------ | ----------------------------------------------- |
| `code`    | string | The snippet under review                        |
| `task_id` | string | Identifier of the scenario (`e1`, `m1`, `h1`, `ex1`) |
| `step`    | int    | Number of actions taken so far                  |

The OpenEnv envelope adds `reward` and `done` to every response.

**Why it is shaped this way:** the agent gets what a human reviewer gets, which is the code and
nothing else. It is not told how many bugs there are, so deciding when to stop is part of the task.
Lines are not numbered: the agent refers to them by 1-indexed position in `code`, so it has to
locate a defect precisely, not just describe it.

### Action space

Four discrete action types share one payload schema:

```json
{ "type": "classify",
  "payload": { "line_number": 6, "bug_type": "security", "severity": "critical" } }
```

| `type`     | Payload fields used for scoring | What the agent is claiming          |
| ---------- | ------------------------------- | ----------------------------------- |
| `detect`   | `line_number`                   | "There is a bug on this line."      |
| `classify` | `line_number`, `bug_type`       | "It is this kind of bug."           |
| `fix`      | `line_number` (`fix` text stored, not scored) | "This is the repair."     |
| `skip`     | none                            | "I'm done." Ends the episode.       |

The payload also accepts `severity` and `description`. They are recorded in state but not scored.
`bug_type` is compared to the ground truth by exact string match. The labels used by the current
tasks are `logic`, `security` and `performance`.

**Why it is shaped this way:** review splits into three skills that fail independently, so each one
gets its own action and its own credit. A single flat payload with optional fields keeps one schema
that an LLM can emit reliably. Every field goes through Pydantic validation, so a malformed action
is rejected at the API boundary instead of causing a confusing failure later in the episode. That
matters when the agent under test is a language model producing JSON.

### Episode state

Server-side, the environment keeps the task (including its ground-truth `known_bugs`) and three
lists of credited claims: `detected_bugs`, `classified_bugs` and `proposed_fixes`. Each
`(action type, line)` pair can be credited once. The three action types are credited independently,
so a correct `fix` earns its reward even if the agent never sent `detect` for that line.

---

## Tasks and difficulty levels

Each level currently has exactly **one** hand-written scenario, defined in `generate_tasks()` in
[`models.py`](models.py).

| Level    | ID    | Lines | Labelled bugs | What it tests                                        |
| -------- | ----- | ----- | ------------- | ---------------------------------------------------- |
| `easy`   | `e1`  | 2     | 1             | A single wrong operator                              |
| `medium` | `m1`  | 5     | 2             | Defects that are *absent* code: missing guards       |
| `hard`   | `h1`  | 8     | 3             | Three different categories of defect in one function |
| `expert` | `ex1` | 2     | 1             | Code injection via `eval`                            |

**easy:** `return a - b` in a function named `add` (line 2, `logic`). The only skill tested is
noticing.

**medium:** `divide()` with no zero check (line 2) and `get_element()` with no bounds check
(line 5), both `logic`. Nothing is syntactically wrong. The agent has to reason about inputs the
code doesn't handle.

**hard:** a SQLite lookup with an f-string query (line 6, `security`, critical), a connection that
is never closed (line 8, labelled `performance`), and no error handling around the connect call
(line 4, `logic`). Here the agent has to track several categories of problem at once. The step
budget (15) is also tighter relative to the work: a perfect run takes 9 actions.

**expert:** `eval(user_input)` (line 1, `security`, critical). Line 2 is a hard-coded password,
which is **not** in the ground truth (see [Limitations](#limitations)).

The levels add bugs and bug categories from `easy` to `hard`. `expert` doesn't continue that
progression: with two lines and one labelled bug, it is easier than `hard` in practice. The
`baseline_score` values in `openenv.yaml` (0.8 / 0.5 / 0.3 / 0.2) are declared targets. They have
not been measured.

---

## Scoring

There are two layers. Step rewards shape behaviour during an episode. The episode grade is the
benchmark number.

### Step reward

| Event                                                          | Reward    |
| -------------------------------------------------------------- | --------- |
| `detect` on a labelled bug line, first time                    | **+0.30** |
| `classify` with the correct line **and** exact `bug_type`, first time | **+0.30** |
| `fix` on a labelled bug line, first time                       | **+0.40** |
| Any of the above on a wrong line, wrong type, or repeated      | **−0.10** |
| `skip`, or an action with no payload                           | 0.00      |

`fix` is worth the most because, in principle, a repair is the strongest evidence that the agent
understood the defect. The −0.10 penalty makes indiscriminate guessing cost cumulative reward.

### Episode grade

When the episode ends, [`grader/code_review_graders.py`](grader/code_review_graders.py) computes
recall against the labelled bugs for each of the three skills:

```
grade = 0.33 × detected / total_bugs
      + 0.33 × correctly_classified / total_bugs     (line AND bug_type must match)
      + 0.34 × fixed / total_bugs
```

The result is clamped to `[0.01, 0.99]`. An episode passes at **≥ 0.50**. The grade is
deterministic: the same action sequence always gets the same grade.

### Known limitations of the scoring

Three properties of the grade matter if you use it to compare agents:

1. **The grade measures recall only and has no precision term.** False positives cost step reward
   but don't lower the grade. A policy that sends `detect` then `fix` on every line, with no
   understanding of the code, grades 0.67 on `easy`, `medium` and `expert` and 0.56 on `hard`. That
   passes every level.
2. **Fix credit depends only on the line.** The `fix` text is stored but never checked. A patch that
   doesn't compile still earns full fix credit if it targets the right line.
3. **Classification requires the exact label string.** The `bug_type` vocabulary isn't given to the
   agent, so a correct diagnosis phrased as `injection` or `resource-leak` instead of `security` or
   `performance` scores zero.

[Design decisions](#design-decisions-and-what-id-change) below covers how I would fix each of these.

---

## API reference

The FastAPI app is built by `HTTPEnvServer` from `openenv-core`, plus grader routes.
Interactive docs are at `http://localhost:7860/docs`.

### Running an episode: WebSocket `/ws`

**Run episodes over `/ws`.** Each WebSocket connection gets its own environment instance (up to 10
concurrent sessions), so `reset`, the steps that follow, and `state` all act on the same episode.
`GenericEnvClient` in the Quickstart uses this protocol. Raw messages look like this:

```json
→ {"type": "reset", "data": {"task_id": "hard"}}
← {"type": "observation", "data": {"observation": {"code": "...", "task_id": "h1", "step": 0}, "reward": null, "done": false}}

→ {"type": "step", "data": {"type": "detect", "payload": {"line_number": 6}}}
← {"type": "observation", "data": {"observation": {...,"step": 1}, "reward": 0.3, "done": false}}

→ {"type": "state"}
← {"type": "state", "data": {"task": {...}, "detected_bugs": [...], "classified_bugs": [], "proposed_fixes": [], ...}}
```

### HTTP routes

| Route                 | Method | Notes |
| --------------------- | ------ | ----- |
| `/health`             | GET    | Liveness: `{"status": "healthy"}` |
| `/reset`              | POST   | Body `{"task_id": "easy"}`. **Stateless**, see below |
| `/step`               | POST   | Body `{"action": {"type": ..., "payload": {...}}}`. **Stateless** |
| `/state`              | GET    | **Stateless** |
| `/schema`             | GET    | JSON Schemas for action, observation and state |
| `/metadata`           | GET    | Environment name, version, description |
| `/grader`             | GET    | Same as `/grade/easy` |
| `/grade/{level}`      | GET    | Runs the built-in **oracle** agent on that level and grades it |

> **The HTTP `/reset`, `/step` and `/state` routes don't share an episode.** In `openenv-core`
> each HTTP request builds a fresh environment and closes it afterwards. A `/step` sent after
> `/reset {"task_id": "hard"}` runs against a new default (`easy`) environment. Use these routes
> for smoke tests and schema checks, and use `/ws` for real episodes.

> **`/grade/{level}` doesn't grade your agent.** It runs `_heuristic_action`, an oracle that
> reads the ground truth and makes the ideal moves, then returns that run's grade (0.99 on every
> level). It is an environment self-test: if the oracle doesn't reach the ceiling, the environment
> is broken. To grade your own agent, pass its final state to `_evaluate_state()`, as
> `examples/run_episode.py` does.

---

## Evaluating an LLM

[`inference.py`](inference.py) runs any OpenAI-compatible chat model through all four levels.
It runs the environment **in-process**, so no server is needed.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # default
export HF_TOKEN="..."                                    # or API_KEY
python inference.py
```

It prints one structured line per event, one block per level, then a summary:

```text
[START] task=code-review env=code-review-env model=<MODEL_NAME>
[STEP] step=<n> action=<action JSON> reward=<r> done=<bool> error=<msg|null>
[END] success=<grade >= 0.50> steps=<n> score=<grade> rewards=<r1,r2,...>
[SUMMARY] scores={"easy": ..., "medium": ..., "hard": ..., "expert": ...} average=<mean grade>
```

At each step the model sees the code and its last five actions with their rewards, and must reply
with a single JSON action. If the reply can't be parsed, or the model call fails, the harness falls
back to `skip`.

---

## Project structure

```
.
├── models.py                       # Pydantic schemas (Task, Bug, Action, Observation, EnvState),
│                                   # the task set, and CodeReviewEnvironment (reset/step/rewards)
├── grader/
│   └── code_review_graders.py      # Episode grade (_evaluate_state), oracle agent, per-level graders
├── backend/
│   └── main.py                     # OpenEnv adapter (CREnvWrapper) and FastAPI app construction
├── server/
│   └── app.py                      # ASGI entry point: server.app:app
├── examples/
│   └── run_episode.py              # End-to-end episode over WebSocket
├── inference.py                    # LLM evaluation harness (in-process)
├── openenv.yaml                    # OpenEnv manifest: endpoints, spaces, tasks, scoring weights
├── Dockerfile                      # python:3.11-slim, serves on :7860
├── pyproject.toml / requirements.txt / uv.lock
└── .github/workflows/main.yml      # Mirrors main to a Hugging Face Space on push
```

The front matter at the top of this README configures that Hugging Face Space (Docker SDK).

The design keeps environment logic in `models.py` and keeps it free of any framework.
`CodeReviewEnvironment` is plain Python with a `reset()`/`step()` interface, so the evaluator can
use it directly without a server. `backend/main.py` is a thin adapter that translates it to OpenEnv's
types.

---

## Limitations

1. **Agents can game the grade.** It measures recall without precision, so flagging every line
   passes. See [Scoring](#known-limitations-of-the-scoring).
2. **Ground truth is visible to the agent.** The session `state` message returns the task with
   `known_bugs`. An honest harness never sends it to the agent, but the environment doesn't enforce
   that.
3. **Fixes aren't verified.** Credit depends on the line, not on whether the patch is correct.
4. **The task set is tiny and inconsistent.** There is one scenario per level, and `expert` is
   easier than `hard`. Some labels are debatable: the unclosed connection is labelled `performance`
   instead of a resource issue. The hard-coded password in `expert` isn't labelled, so an agent that
   correctly flags it is penalised.
5. **The bug-type vocabulary is hidden.** Classification partly measures whether the agent guesses
   the label strings.
6. **The final grade isn't in the step response.** The server computes it, but OpenEnv strips
   observation `metadata` on the wire, so clients have to recompute it from `state`.
7. **No published baselines.** The baseline scores in `openenv.yaml` have not been measured.
8. **No automated tests.** The repository has no test suite.
9. **`CodeReviewEnvironment.deterministic_grader()` is a stub that always returns 0.5.**
   The real grade is `_evaluate_state()`.

---

## Design decisions and what I'd change

### Decisions I'd keep

- **Separate credit for detection, classification and repair.** This is the most useful property of
  the environment. A single pass/fail number hides *why* an agent failed. Three recall terms show
  whether it can't find bugs, can't explain them, or can't fix them.
- **The environment as plain Python behind a thin adapter.** `CodeReviewEnvironment` doesn't import
  FastAPI or OpenEnv. The serving layer can change, as it did in the move from Flask to FastAPI
  plus OpenEnv, without touching the reward logic, and evaluation can run in-process.
- **An oracle agent as a self-test.** A reference policy that reads the ground truth and must score
  at the ceiling quickly catches a broken environment, such as a mislabelled line or an off-by-one,
  before anyone blames the model.
- **Strict validation at the API boundary.** Malformed model output fails validation when the
  request arrives and never enters the episode.

### What I'd change, in priority order

1. **Add precision to the grade.** Score each component as F1, or subtract a false-positive term,
   so that flagging every line is no longer a viable strategy. Right now this is a benchmark
   validity bug, not a refinement. A regression test should assert that the flag-every-line policy
   fails every level.
2. **Hide the ground truth and return the grade properly.** Remove `known_bugs` from the state
   exposed to clients, and return the episode grade as a typed field on the terminal observation
   instead of in `metadata`.
3. **Verify fixes by running them.** Apply the proposed patch and run per-task tests: one that the
   bug fails and the fix passes, plus regression tests. Fix credit should mean the code works, not
   that the patch pointed at the right line.
4. **Publish the label set.** Put `bug_type` in the action schema as an enum, and number the lines
   in the observation. Classification should measure diagnosis, not label guessing.
5. **Make tasks data, not code.** Move scenarios into versioned fixture files, with several tasks
   per level sampled by seed. Re-audit the labels, starting with `expert`. Build difficulty from
   measured solve rates instead of declaring it.
6. **Add tests and CI.** Cover the reward table, the grade formula, the oracle self-test, and one
   WebSocket round trip. Step rewards used to be clamped into the grade's range, which turned the
   −0.10 penalty into +0.01; a round-trip test would have caught that immediately.
7. **Clean up the module layout.** Collapse `models.py`, `backend/` and `server/` into one package,
   and remove the `sys.path` manipulation in three files.
8. **Make harness failures visible.** `inference.py` turns any model or parsing error into a silent
   `skip`, so a network outage looks like a model that gave up. Errors should be reported and
   excluded from scores.

---

## Built with

Python 3.11 · FastAPI · Pydantic v2 · OpenEnv (`openenv-core`) · Uvicorn · Docker
