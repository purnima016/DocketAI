from environment import DocketAIEnv


def get_hard_task():
    """
    Hard Task: 50 cases with limited steps.
    Agent can only handle 30 out of 50 cases.
    Must make very smart prioritization decisions.
    Escalation happens faster.
    """
    env = DocketAIEnv(num_cases=50, max_steps=30)
    return env


def grade_hard(handled_cases, total_cases=50) -> float:
    """
    Grader for hard task.
    Score based on:
    - urgency of cases handled
    - critical cases prioritized
    - fairness across case types
    Returns score between 0.0 and 1.0
    """
    if not handled_cases:
        return 0.0

    # urgency score
    urgency_score = sum(c.urgency_score for c in handled_cases)
    max_urgency = total_cases * 10.0
    urgency_ratio = min(urgency_score / max_urgency, 1.0)

    # critical cases (severity > 8.0)
    critical_cases = [c for c in handled_cases if c.severity >= 8.0]
    critical_ratio = len(critical_cases) / max(len(handled_cases), 1)

    # fairness: variety of case types handled
    case_types_handled = set(c.case_type for c in handled_cases)
    fairness_score = len(case_types_handled) / 5.0  # 5 total types

    # people affected score
    people_score = sum(c.people_affected for c in handled_cases)
    max_people = total_cases * 20
    people_ratio = min(people_score / max_people, 1.0)

    # final weighted score
    score = (
        0.4 * urgency_ratio +
        0.3 * critical_ratio +
        0.2 * fairness_score +
        0.1 * people_ratio
    )

    return round(min(score, 1.0), 3)