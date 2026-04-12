def grade(trajectory, task_config=None):
    """Called by OpenEnv validator."""
    if not trajectory:
        return {"score": 0.0, "passed": False, "details": {"steps": 0}}
    rewards = [step.get("reward", 0.0) for step in trajectory]
    total = sum(rewards)
    avg = total / len(rewards)
    score = round(min(avg, 1.0), 4)
    return {
        "score": score,
        "passed": score >= 0.3,
        "details": {"steps": len(rewards), "total_reward": round(total, 4)}
    }


def run_grader(handled_cases, total_cases=10):
    """Called by inference.py."""
    if not handled_cases:
        return {"task": "easy", "score": 0.0, "passed": False, "details": {}}
    try:
        total_urgency = sum(c.urgency_score for c in handled_cases)
        score = round(min(total_urgency / (total_cases * 10.0), 1.0), 3)
    except Exception:
        score = 0.0
    return {
        "task": "easy",
        "score": score,
        "passed": score >= 0.3,
        "details": {"cases_handled": len(handled_cases), "total_cases": total_cases}
    }