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
        "passed": score >= 0.5,
        "details": {"steps": len(rewards), "total_reward": round(total, 4)}
    }


def run_grader(handled_cases, total_cases=50):
    """Called by inference.py."""
    if not handled_cases:
        return {"task": "hard", "score": 0.0, "passed": False, "details": {}}
    try:
        urgency_ratio = min(sum(c.urgency_score for c in handled_cases) / (total_cases * 10.0), 1.0)
        critical = [c for c in handled_cases if c.severity >= 8.0]
        critical_ratio = len(critical) / max(len(handled_cases), 1)
        fairness = len(set(c.case_type for c in handled_cases)) / 5.0
        people = min(sum(c.people_affected for c in handled_cases) / (total_cases * 20), 1.0)
        score = round(0.4 * urgency_ratio + 0.3 * critical_ratio + 0.2 * fairness + 0.1 * people, 3)
    except Exception:
        score = 0.0
    return {
        "task": "hard",
        "score": score,
        "passed": score >= 0.5,
        "details": {"cases_handled": len(handled_cases), "total_cases": total_cases}
    }