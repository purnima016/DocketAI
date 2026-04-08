from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os

# Ensure root project dir is on path so environment.py is found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DocketAIEnv, Action

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
    static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static/index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path, media_type="text/html")
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
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()