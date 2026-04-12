def grade(trajectory, task_config=None):
    """
    OpenEnv validator calls this function.
    trajectory: list of dicts with 'reward' key
    """
    if not trajectory:
        return {"score": 0.0, "passed": False, "details": {"steps": 0}}

    rewards = [step.get("reward", 0.0) for step in trajectory]
    total = sum(rewards)
    avg = total / len(rewards)
    score = round(min(avg, 1.0), 4)

    return {
        "score": score,
        "passed": score >= 0.4,
        "details": {
            "steps": len(rewards),
            "total_reward": round(total, 4),
            "avg_reward": score
        }
    }