FROM python:3.7-slim

ENV LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.7 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.7 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
