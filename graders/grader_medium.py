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
        "passed": score >= 0.4,
        "details": {"steps": len(rewards), "total_reward": round(total, 4)}
    }


def run_grader(handled_cases, total_cases=25):
    """Called by inference.py."""
    if not handled_cases:
        return {"task": "medium", "score": 0.0, "passed": False, "details": {}}
    try:
        total_urgency = sum(c.urgency_score for c in handled_cases)
        high_priority = {"violence", "medical", "bail"}
        hp_handled = [c for c in handled_cases if c.case_type in high_priority]
        urgency_score = min(total_urgency / (total_cases * 10.0), 1.0)
        priority_bonus = len(hp_handled) / max(len(handled_cases), 1)
        score = round(0.6 * urgency_score + 0.4 * priority_bonus, 3)
    except Exception:
        score = 0.0
    return {
        "task": "medium",
        "score": score,
        "passed": score >= 0.4,
        "details": {"cases_handled": len(handled_cases), "total_cases": total_cases}
    }