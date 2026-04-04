from tasks.easy import grade_easy


def run_grader(handled_cases, total_cases=10) -> dict:
    """
    Official grader for easy task.
    Returns score between 0.0 and 1.0
    """
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