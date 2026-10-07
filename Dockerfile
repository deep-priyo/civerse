FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

COPY backend/ /app/backend/
COPY server/ /app/server/
COPY grader/ /app/grader/
COPY models.py /app/models.py
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
