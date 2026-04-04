FROM python:3.10-slim

WORKDIR /app

RUN useradd -m -u 1000 docketai

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN touch tasks/__init__.py graders/__init__.py

RUN chown -R docketai:docketai /app

USER docketai

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]