from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from environment import DocketAIEnv, Action
import os

app = FastAPI(
    title="DocketAI",
    description="Intelligent Court Case Prioritization Environment",
    version="1.0.0"
)

env = DocketAIEnv()

class ActionRequest(BaseModel):
    case_index: int

@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html", media_type="text/html")
    return {"name": "DocketAI", "version": "1.0.0"}

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