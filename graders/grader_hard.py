from tasks.hard import grade_hard


def run_grader(handled_cases, total_cases=50) -> dict:
    """
    Official grader for hard task.
    Returns score between 0.0 and 1.0
    """
    score = grade_hard(handled_cases, total_cases)

    critical_cases = [c for c in handled_cases if c.severity >= 8.0]
    case_types_handled = set(c.case_type for c in handled_cases)

    return {
        "task": "hard",
        "score": score,
        "passed": score >= 0.5,
        "details": {
            "cases_handled": len(handled_cases),
            "total_cases": total_cases,
            "critical_cases_handled": len(critical_cases),
            "case_types_covered": list(case_types_handled),
            "total_people_helped": sum(
                c.people_affected for c in handled_cases
            ),
            "avg_urgency": round(
                sum(c.urgency_score for c in handled_cases) /
                max(len(handled_cases), 1), 2
            )
        }
    }