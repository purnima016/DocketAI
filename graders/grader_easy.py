import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.easy import grade_easy


def grade(trajectory, task_config=None) -> dict:
    """
    Required by OpenEnv validator. Called as grade(trajectory).
    trajectory: list of step dicts with 'reward', 'action', 'observation'
    """
    if not trajectory:
        return {"score": 0.0, "passed": False, "details": {}}

    total_reward = sum(step.get("reward", 0) for step in trajectory)
    score = min(total_reward / max(len(trajectory), 1), 1.0)

    return {
        "score": round(score, 3),
        "passed": score >= 0.3,
        "details": {
            "steps": len(trajectory),
            "total_reward": round(total_reward, 3),
            "avg_reward": round(score, 3)
        }
    }


def run_grader(handled_cases, total_cases=10) -> dict:
    """Legacy function kept for inference.py compatibility."""
    score = grade_easy(handled_cases, total_cases)
    return {
        "task": "easy",
        "score": score,
        "passed": score >= 0.3,
        "details": {
            "cases_handled": len(handled_cases),
            "total_cases": total_cases,
            "avg_urgency": round(
                sum(c.urgency_score for c in handled_cases) /
                max(len(handled_cases), 1), 2
            )
        }
    }