from fastapi import FastAPI
from pydantic import BaseModel
from environment import DocketAIEnv, Action
import uvicorn

app = FastAPI()

env = DocketAIEnv()

class ActionRequest(BaseModel):
    case_index: int


@app.get("/")
def root():
    return {"name": "DocketAI"}


@app.post("/reset")
def reset():
    observation = env.reset()
    return {
        "observation": observation.dict()
    }


@app.get("/state")
def state():
    return {
        "state": env.state()
    }


@app.post("/step")
def step(action: ActionRequest):
    result = env.step(Action(case_index=action.case_index))
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()