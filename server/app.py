import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from environment import DocketAIEnv, Action
import uvicorn

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

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()