import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from environment import DocketAIEnv, Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "docketai"
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent court case prioritization agent.
    You will be given a list of pending court cases with their details.
    Your job is to select the index of the most urgent case to handle next.
    Consider: medical > violence > bail > criminal > civil
    Reply with ONLY a single integer (0-based index). No explanation.
""").strip()

TASKS = [
    {"name": "easy",   "num_cases": 10, "max_steps": 10, "threshold": 0.3},
    {"name": "medium", "num_cases": 25, "max_steps": 20, "threshold": 0.4},
    {"name": "hard",   "num_cases": 50, "max_steps": 30, "threshold": 0.5},
]


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(observation):
    cases_text = ""
    for i, case in enumerate(observation.cases):
        cases_text += (
            f"Index {i}: type={case.case_type} "
            f"age={case.age_days}days "
            f"severity={case.severity} "
            f"urgency={case.urgency_score}\n"
        )
    return f"Day {observation.day} — Pending Cases:\n{cases_text}\nWhich index to handle next?"


def get_action(client, observation, fallback=0):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(observation)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        idx = int((completion.choices[0].message.content or "").strip())
        return idx if 0 <= idx < len(observation.cases) else fallback
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return fallback


def grade(trajectory, threshold=0.3):
    if not trajectory:
        return 0.0, False
    rewards = [s.get("reward", 0.0) for s in trajectory]
    score = round(min(sum(rewards) / len(rewards), 1.0), 4)
    return score, score >= threshold


def run_task(client, task_cfg):
    task_name = task_cfg["name"]
    max_steps = task_cfg["max_steps"]
    threshold = task_cfg["threshold"]

    env = DocketAIEnv(num_cases=task_cfg["num_cases"], max_steps=max_steps)
    rewards: List[float] = []
    trajectory = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()
        for step in range(1, max_steps + 1):
            if not observation.cases:
                break
            case_index = get_action(client, observation, fallback=0)
            action = Action(case_index=case_index)
            result = env.step(action)

            reward = result.reward
            done = result.done
            rewards.append(reward)
            trajectory.append({"step": step, "action": f"handle_case({case_index})", "reward": reward, "done": done})
            steps_taken = step
            observation = result.observation

            log_step(step=step, action=f"handle_case({case_index})", reward=reward, done=done, error=None)
            if done:
                break

        score, success = grade(trajectory, threshold)

    except Exception as e:
        print(f"[DEBUG] Error in task {task_name}: {e}", flush=True)
        score, success = 0.0, False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_cfg in TASKS:
        run_task(client, task_cfg)


if __name__ == "__main__":
    main()