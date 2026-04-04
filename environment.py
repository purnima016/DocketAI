import random
from typing import List
from pydantic import BaseModel


# ─── Data Models ───────────────────────────────────────────

class Case(BaseModel):
    case_id: int
    case_type: str        # bail, violence, civil, medical, criminal
    age_days: int         # how long pending
    severity: float       # 0.0 to 10.0
    people_affected: int
    urgency_score: float  # computed automatically


class Observation(BaseModel):
    cases: List[Case]
    day: int
    total_cases: int
    handled_so_far: int


class Action(BaseModel):
    case_index: int       # which case to handle next (0-based index)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# ─── Case Generator ────────────────────────────────────────

CASE_TYPES = {
    "bail":      {"base_severity": 8.0, "escalation": 0.5},
    "violence":  {"base_severity": 9.0, "escalation": 0.7},
    "medical":   {"base_severity": 9.5, "escalation": 0.8},
    "criminal":  {"base_severity": 7.0, "escalation": 0.3},
    "civil":     {"base_severity": 3.0, "escalation": 0.1},
}


def generate_case(case_id: int) -> Case:
    case_type = random.choice(list(CASE_TYPES.keys()))
    age_days = random.randint(10, 500)
    base_severity = CASE_TYPES[case_type]["base_severity"]
    escalation = CASE_TYPES[case_type]["escalation"]
    severity = min(10.0, base_severity + escalation * (age_days / 100))
    people_affected = random.randint(1, 20)
    urgency_score = round(
        0.5 * severity +
        0.3 * min(age_days / 500, 1.0) * 10 +
        0.2 * min(people_affected / 20, 1.0) * 10,
        2
    )
    return Case(
        case_id=case_id,
        case_type=case_type,
        age_days=age_days,
        severity=round(severity, 2),
        people_affected=people_affected,
        urgency_score=urgency_score
    )


# ─── Environment ───────────────────────────────────────────

class DocketAIEnv:

    def __init__(self, num_cases: int = 10, max_steps: int = 10):
        self.num_cases = num_cases
        self.max_steps = max_steps
        self.cases: List[Case] = []
        self.handled: List[Case] = []
        self.day = 0
        self.total_reward = 0.0

    def reset(self) -> Observation:
        self.cases = [generate_case(i) for i in range(self.num_cases)]
        self.handled = []
        self.day = 0
        self.total_reward = 0.0
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        done = False
        reward = 0.0

        if action.case_index < 0 or action.case_index >= len(self.cases):
            reward = -5.0
        else:
            case = self.cases[action.case_index]
            reward = round(case.urgency_score / 10.0, 2)
            self.handled.append(case)
            self.cases.pop(action.case_index)

            for c in self.cases:
                escalation = CASE_TYPES[c.case_type]["escalation"]
                c.age_days += 1
                c.severity = min(10.0, c.severity + escalation * 0.1)

        self.day += 1
        self.total_reward += reward

        if len(self.cases) == 0 or self.day >= self.max_steps:
            done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={"day": self.day, "handled": len(self.handled)}
        )

    def state(self) -> dict:
        return {
            "day": self.day,
            "remaining_cases": len(self.cases),
            "handled_cases": len(self.handled),
            "total_reward": self.total_reward
        }

    def _get_observation(self) -> Observation:
        return Observation(
            cases=self.cases,
            day=self.day,
            total_cases=self.num_cases,
            handled_so_far=len(self.handled)
        )