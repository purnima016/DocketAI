# DocketAI ⚖️
### Intelligent Court Case Prioritization Through Adaptive Sequencing

## Problem Statement
India has 5 crore+ pending court cases. Cases wait for years because nobody decides which case needs attention first intelligently. DocketAI trains an AI agent to fix this.

## Solution
An OpenEnv environment where an AI agent learns to dynamically prioritize court cases based on urgency, severity, age, and people affected.

## Live Demo
🔗 https://huggingface.co/spaces/Purnima1612/docketai

## Tasks
| Task | Cases | Max Steps | Pass Threshold |
|---|---|---|---|
| Easy | 10 | 10 | 0.30 |
| Medium | 25 | 20 | 0.40 |
| Hard | 50 | 30 | 0.50 |

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| / | GET | Dashboard UI |
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take action |
| /state | GET | Current state |

## Setup
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Docker
```bash
docker build -t docketai .
docker run -p 7860:7860 docketai
```

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- Pydantic
- OpenEnv Core
- Docker
- HuggingFace Spaces
