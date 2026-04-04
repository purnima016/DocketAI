import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from environment import DocketAIEnv, Action

# ─── Configuration ─────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("DOCKETAI_TASK", "easy")
BENCHMARK = "docketai"
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.3


# ─── Logging ───────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Prompt Builder ────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent court case prioritization agent.
    You will be given a list of pending court cases with their details.
    Your job is to select the index of the most urgent case to handle next.
    
    Consider these factors in order of importance:
    1. Case type: medical > violence > bail > criminal > civil
    2. Severity score (higher = more urgent)
    3. Age of case (older = more urgent)
    4. People affected (more = more urgent)
    
    Reply with ONLY a single integer (the case index, 0-based).
    No explanation. No text. Just the number.
""").strip()


def build_user_prompt(observation) -> str:
    cases_text = ""
    for i, case in enumerate(observation.cases):
        cases_text += (
            f"Index {i}: type={case.case_type} "
            f"age={case.age_days}days "
            f"severity={case.severity} "
            f"people={case.people_affected} "
            f"urgency={case.urgency_score}\n"
        )
    return f"Day {observation.day} — Pending Cases:\n{cases_text}\nWhich index to handle next?"


# ─── Agent Decision ────────────────────────────────────────
def get_agent_action(client: OpenAI, observation, fallback_index: int = 0) -> int:
    user_prompt = build_user_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        index = int(text)
        if 0 <= index < len(observation.cases):
            return index
        return fallback_index
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_index


# ─── Task Setup ────────────────────────────────────────────
def get_env_for_task(task_name: str) -> DocketAIEnv:
    if task_name == "easy":
        return DocketAIEnv(num_cases=10, max_steps=10)
    elif task_name == "medium":
        return DocketAIEnv(num_cases=25, max_steps=20)
    elif task_name == "hard":
        return DocketAIEnv(num_cases=50, max_steps=30)
    else:
        return DocketAIEnv(num_cases=10, max_steps=10)


def get_grader_for_task(task_name: str):
    if task_name == "easy":
        from graders.grader_easy import run_grader
    elif task_name == "medium":
        from graders.grader_medium import run_grader
    elif task_name == "hard":
        from graders.grader_hard import run_grader
    else:
        from graders.grader_easy import run_grader
    return run_grader


# ─── Main ──────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = get_env_for_task(TASK_NAME)
    run_grader = get_grader_for_task(TASK_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if not observation.cases:
                break

            case_index = get_agent_action(client, observation, fallback_index=0)
            action = Action(case_index=case_index)
            result = env.step(action)

            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            observation = result.observation

            log_step(
                step=step,
                action=f"handle_case({case_index})",
                reward=reward,
                done=done,
                error=error
            )

            if done:
                break

        grader_result = run_grader(env.handled)
        score = grader_result["score"]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during episode: {e}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


if __name__ == "__main__":
    main()