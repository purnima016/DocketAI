from fastapi import FastAPI
from pydantic import BaseModel
from environment import DocketAIEnv, Action, Observation

app = FastAPI(
    title="DocketAI",
    description="Intelligent Court Case Prioritization Environment",
    version="1.0.0"
)

# Global environment instance
env = DocketAIEnv()


# ─── Request Models ────────────────────────────────────────

class ActionRequest(BaseModel):
    case_index: int


# ─── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "DocketAI",
        "version": "1.0.0",
        "description": "Court Case Prioritization Environment",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    observation = env.reset()
    return observation.dict()


@app.post("/step")
def step(action: ActionRequest):
    result = env.step(Action(case_index=action.case_index))
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }


@app.get("/state")
def state():
    return env.state()