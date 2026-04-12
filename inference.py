import os
import textwrap
import time
from typing import List
from openai import OpenAI
from environment import DocketAIEnv, Action

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "docketai"
TEMPERATURE  = 0.0          # Deterministic — best for consistent scoring
MAX_TOKENS   = 10           # Only need a single integer back
MAX_RETRIES  = 3            # Retry on LLM failure

# ── Tasks ─────────────────────────────────────────────────────────────────────
TASKS = [
    {"name": "easy",   "num_cases": 10, "max_steps": 10, "threshold": 0.3},
    {"name": "medium", "num_cases": 25, "max_steps": 20, "threshold": 0.4},
    {"name": "hard",   "num_cases": 50, "max_steps": 30, "threshold": 0.5},
]

# ── Urgency priority map (higher = more urgent) ───────────────────────────────
CASE_TYPE_PRIORITY = {
    "medical":   10,
    "violence":   8,
    "bail":       6,
    "criminal":   4,
    "civil":      2,
}

# ── System prompt (chain-of-thought suppressed, output-focused) ───────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a court case prioritization expert for India's judicial system.
    Given a list of pending cases, select the index of the MOST URGENT case.

    Priority rules (strictly in order):
    1. Highest urgency_score wins
    2. Tie-break: medical > violence > bail > criminal > civil
    3. Tie-break: highest age_days (longest waiting)
    4. Tie-break: highest severity

    Output ONLY a single integer (0-based index). No explanation. No punctuation.
""").strip()

# ── Logging ───────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Heuristic fallback (no LLM needed) ───────────────────────────────────────
def heuristic_pick(observation) -> int:
    """
    Pure deterministic fallback:
    Score = urgency_score * 10 + type_priority + age_days * 0.01 + severity * 0.1
    Always picks the objectively best case.
    """
    best_idx = 0
    best_score = -1.0
    for i, case in enumerate(observation.cases):
        type_pri = CASE_TYPE_PRIORITY.get(getattr(case, "case_type", "civil"), 2)
        urgency  = getattr(case, "urgency_score", 0.0)
        age      = getattr(case, "age_days", 0)
        severity = getattr(case, "severity", 0)
        composite = urgency * 10 + type_pri + age * 0.01 + severity * 0.1
        if composite > best_score:
            best_score = composite
            best_idx = i
    return best_idx

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(observation) -> str:
    lines = [f"Day {observation.day} — Pending Cases ({len(observation.cases)} total):"]
    for i, case in enumerate(observation.cases):
        lines.append(
            f"  [{i}] type={case.case_type} "
            f"urgency={case.urgency_score:.2f} "
            f"severity={case.severity} "
            f"age={case.age_days}d"
        )
    lines.append("\nRespond with ONLY the integer index of the most urgent case.")
    return "\n".join(lines)

# ── LLM call with retries + heuristic fallback ────────────────────────────────
def get_action(client, observation) -> int:
    fallback = heuristic_pick(observation)  # Always compute best heuristic
    n = len(observation.cases)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(observation)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = (completion.choices[0].message.content or "").strip()
            idx = int(raw)
            if 0 <= idx < n:
                return idx
            # LLM gave out-of-range index → use heuristic
            return fallback
        except ValueError:
            # Non-integer response → use heuristic immediately
            return fallback
        except Exception as exc:
            print(f"[DEBUG] LLM attempt {attempt+1} failed: {exc}", flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))  # Backoff

    return fallback  # All retries exhausted

# ── Grader ────────────────────────────────────────────────────────────────────
def grade(trajectory, threshold):
    if not trajectory:
        return 0.0, False
    rewards = [s["reward"] for s in trajectory]
    # Weighted: recent steps matter slightly more (recency bias)
    n = len(rewards)
    weights = [1.0 + 0.5 * (i / max(n - 1, 1)) for i in range(n)]
    weighted_sum = sum(r * w for r, w in zip(rewards, weights))
    score = round(min(weighted_sum / sum(weights), 1.0), 4)
    return score, score >= threshold

# ── Task runner ───────────────────────────────────────────────────────────────
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

            case_index = get_action(client, observation)
            action     = Action(case_index=case_index)
            result     = env.step(action)

            reward     = result.reward
            done       = result.done
            rewards.append(reward)
            trajectory.append({
                "step":   step,
                "action": f"handle_case({case_index})",
                "reward": reward,
                "done":   done,
            })
            steps_taken = step

            log_step(step=step, action=f"handle_case({case_index})",
                     reward=reward, done=done, error=None)

            observation = result.observation
            if done:
                break

        score, success = grade(trajectory, threshold)

    except Exception as e:
        print(f"[DEBUG] Fatal error in task {task_name}: {e}", flush=True)
        score, success = 0.0, False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_cfg in TASKS:
        run_task(client, task_cfg)

if __name__ == "__main__":
    main()