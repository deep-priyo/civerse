import os, sys
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Action as ModelAction, Observation as ModelObservation,
    generate_tasks, CodeReviewEnvironment, ActionPayload
)
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action as OEAction, Observation as OEObservation, State as OEState, EnvironmentMetadata,
)
from openenv.core.env_server.http_server import HTTPEnvServer

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _safe(raw: float) -> float:
    try:
        return round(max(_SCORE_MIN, min(_SCORE_MAX, float(raw))), 4)
    except Exception:
        return _SCORE_MIN

def _grade_task(difficulty: str) -> dict:
    try:
        from grader.code_review_graders import EasyGrader, MediumGrader, HardGrader, ExpertGrader
        cls = {"easy": EasyGrader, "medium": MediumGrader,
               "hard": HardGrader, "expert": ExpertGrader}.get(difficulty, EasyGrader)
        score, done, msg = cls().grade()
        score = _safe(score)
    except Exception as ex:
        score = _SCORE_MIN
        msg   = f"Grader error: {ex}"
    return {"task_id": difficulty, "reward": score, "score": score,
            "done": False, "grader_message": msg}

class CREAction(OEAction):
    type: str = Field(description="detect | classify | fix | skip")
    payload: Optional[Dict[str, Any]] = Field(default=None)
    model_config = {"extra": "allow"}

class CREObservation(OEObservation):
    code: str = Field(default="")
    task_id: str = Field(default="")
    step: int = Field(default=0)
    model_config = {"extra": "allow"}

class CREState(OEState):
    task: Dict[str, Any] = Field(default_factory=dict)
    time_step: int = Field(default=0)
    detected_bugs: List[Dict[str, Any]] = Field(default_factory=list)
    classified_bugs: List[Dict[str, Any]] = Field(default_factory=list)
    proposed_fixes: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = {"extra": "allow"}

class CREnvWrapper(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._env = CodeReviewEnvironment(tasks=generate_tasks("easy"), max_steps=15)
        self._final_score: float = _SCORE_MIN

    def _to_oe_obs(self, obs: ModelObservation, done=False,
                   reward=None, info=None) -> CREObservation:
        return CREObservation(
            code=obs.code,
            task_id=obs.task_id,
            step=obs.step, done=done, reward=reward, metadata=info or {},
        )

    def reset(self, seed=None, episode_id=None, task_id: str = "easy", **kw) -> CREObservation:
        if task_id not in ("easy", "medium", "hard", "expert"):
            task_id = "easy"
        self._env = CodeReviewEnvironment(tasks=generate_tasks(task_id), max_steps=15)
        self._final_score = _SCORE_MIN
        return self._to_oe_obs(self._env.reset())

    def step(self, action: CREAction, timeout_s=None, **kw) -> CREObservation:
        payload_obj = None
        if action.payload:
            payload_obj = ActionPayload(**action.payload)
        ma = ModelAction(type=action.type, payload=payload_obj)
        obs, reward, done, info = self._env.step(ma)
        if done:
            from grader.code_review_graders import _evaluate_state
            self._final_score = _safe(_evaluate_state(info))
            info["final_score"] = self._final_score
        # Step rewards are passed through unclamped: the [0.01, 0.99] range applies to
        # the episode grade only. Clamping here turned the -0.10 penalty into +0.01.
        return self._to_oe_obs(obs, done=done, reward=float(reward), info=info)

    @property
    def state(self):
        raw = self._env.state_dict()
        return CREState(
            task=raw.get("task", {}),
            time_step=raw.get("time_step", 0),
            detected_bugs=raw.get("detected_bugs", []),
            classified_bugs=raw.get("classified_bugs", []),
            proposed_fixes=raw.get("proposed_fixes", [])
        )

    def get_metadata(self):
        return EnvironmentMetadata(
            name="code-review-env",
            description="Code Review Environment",
            version="2.0.0", author="CR Team",
        )

    def close(self): pass


def build_app() -> FastAPI:
    server = HTTPEnvServer(
        env=CREnvWrapper, action_cls=CREAction, observation_cls=CREObservation, max_concurrent_envs=10,
    )
    app = FastAPI(
        title="Code Review Environment",
        version="2.0.0",
        description="Code Review Environment",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    server.register_routes(app)

    @app.get("/grader",        tags=["Grader"]) 
    async def get_grader():   return _grade_task("easy")

    @app.get("/grade/easy",    tags=["Grader"])
    async def grade_easy():   return _grade_task("easy")

    @app.get("/grade/medium",  tags=["Grader"])
    async def grade_medium(): return _grade_task("medium")

    @app.get("/grade/hard",    tags=["Grader"])
    async def grade_hard():   return _grade_task("hard")

    @app.get("/grade/expert",  tags=["Grader"])
    async def grade_expert(): return _grade_task("expert")

    return app

app = build_app()
