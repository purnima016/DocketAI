from tasks.medium import grade_medium


def run_grader(handled_cases, total_cases=25) -> dict:
    """
    Official grader for medium task.
    Returns score between 0.0 and 1.0
    """
    score = grade_medium(handled_cases, total_cases)

    high_priority_types = {"violence", "medical", "bail"}
    high_priority_handled = [
        c for c in handled_cases
        if c.case_type in high_priority_types
    ]

    return {
        "task": "medium",
        "score": score,
        "passed": score >= 0.4,
        "details": {
            "cases_handled": len(handled_cases),
            "total_cases": total_cases,
            "high_priority_handled": len(high_priority_handled),
            "avg_urgency": round(
                sum(c.urgency_score for c in handled_cases) /
                max(len(handled_cases), 1), 2
            )
        }
    }