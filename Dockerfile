FROM python:3.11-slim

RUN apt-get update && apt-get install -y git-lfs && git lfs install
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD gunicorn --chdir src api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --preload
