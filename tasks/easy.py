import random
from environment import DocketAIEnv, generate_case, Case

def get_easy_task():
    """
    Easy Task: 10 cases with clearly different urgency levels.
    Agent should easily learn to pick high urgency cases first.
    """
    env = DocketAIEnv(num_cases=10, max_steps=10)
    return env


def grade_easy(handled_cases, total_cases=10) -> float:
    """
    Grader for easy task.
    Score based on ratio of urgent cases handled correctly.
    Returns score between 0.0 and 1.0
    """
    if not handled_cases:
        return 0.0

    total_urgency = sum(c.urgency_score for c in handled_cases)
    max_possible = total_cases * 10.0

    score = min(total_urgency / max_possible, 1.0)
    return round(score, 3)