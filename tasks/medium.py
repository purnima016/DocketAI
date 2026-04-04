from environment import DocketAIEnv


def get_medium_task():
    """
    Medium Task: 25 cases with overlapping urgency levels.
    Some cases escalate mid-episode making it harder.
    Agent must learn to balance urgency vs escalation.
    """
    env = DocketAIEnv(num_cases=25, max_steps=20)
    return env


def grade_medium(handled_cases, total_cases=25) -> float:
    """
    Grader for medium task.
    Score based on:
    - urgency of cases handled
    - whether escalating cases were prioritized
    Returns score between 0.0 and 1.0
    """
    if not handled_cases:
        return 0.0

    # bonus for handling violence/medical/bail first
    high_priority_types = {"violence", "medical", "bail"}
    high_priority_handled = [
        c for c in handled_cases
        if c.case_type in high_priority_types
    ]

    urgency_score = sum(c.urgency_score for c in handled_cases)
    max_urgency = total_cases * 10.0

    priority_bonus = len(high_priority_handled) / max(len(handled_cases), 1)

    score = (
        0.6 * min(urgency_score / max_urgency, 1.0) +
        0.4 * priority_bonus
    )

    return round(min(score, 1.0), 3)