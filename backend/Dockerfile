FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY backend/requirements.txt ./backend/requirements.txt
COPY mini_smolagents ./mini_smolagents
COPY backend ./backend

RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
