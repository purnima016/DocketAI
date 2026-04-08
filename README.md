---
title: DocketAI
emoji: ⚖️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# DocketAI 🏛️
### Intelligent Court Case Prioritization Through Adaptive Sequencing

---

## Problem Statement

India has 5 crore+ pending court cases. Cases wait for years because 
nobody decides which case needs attention first intelligently. A murder 
bail case and a property dispute sit in the same queue. DocketAI trains 
an AI agent to fix this.

---

## What DocketAI Does

An OpenEnv environment where an AI agent learns to dynamically prioritize 
court cases based on urgency, severity, age, and people affected — 
reducing judicial backlog and delivering timely justice.

---

## Environment Details

### Observation Space
| Field | Type | Description |
|---|---|---|
| cases | List[Case] | Pending court cases |
| day | int | Current simulation day |
| total_cases | int | Total cases in episode |
| handled_so_far | int | Cases handled so far |

### Case Fields
| Field | Type | Description |
|---|---|---|
| case_id | int | Unique case identifier |
| case_type | str | bail/violence/medical/criminal/civil |
| age_days | int | Days pending |
| severity | float | 0.0 to 10.0 |
| people_affected | int | Number of people affected |
| urgency_score | float | Computed urgency (0.0-10.0) |

### Action Space
| Field | Type | Description |
|---|---|---|
| case_index | int | Index of case to handle next |

### Reward Function
step_reward = urgency_score / 10.0
final_score =
0.4 × urgency_handled +
0.3 × backlog_reduction +
0.2 × fairness_score +
0.1 × people_helped
---

## Tasks

### Easy
- 10 cases, clearly different urgency
- Max 10 steps
- Pass threshold: 0.3

### Medium  
- 25 cases, overlapping urgency
- Cases escalate mid-episode
- Max 20 steps
- Pass threshold: 0.4

### Hard
- 50 cases, limited to 30 steps
- Complex prioritization required
- Fairness across case types enforced
- Pass threshold: 0.5

---

## Case Types (Priority Order)

1. medical (highest urgency)
2. violence
3. bail
4. criminal
5. civil (lowest urgency)

---

## Setup Instructions

### Local Setup
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/docketai
cd docketai
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker Setup
```bash
docker build -t docketai .
docker run -p 8000:8000 docketai
```

### Run Inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export DOCKETAI_TASK=easy
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| / | GET | Environment info |
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take action |
| /state | GET | Current state |

---

## Baseline Scores

| Task | Random Agent | DocketAI Agent |
|---|---|---|
| Easy | ~0.25 | ~0.65 |
| Medium | ~0.20 | ~0.55 |
| Hard | ~0.15 | ~0.45 |

---

## Real World Impact

- 5 crore+ pending cases in India
- Supports court registries in scheduling
- Decision support tool, not replacement
- Policy simulation for backlog reduction

---

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- Pydantic
- OpenEnv Core
- Docker
- HuggingFace Spaces
