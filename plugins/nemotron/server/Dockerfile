FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nemotron_server.py .

ENV NEMOTRON_DEVICE=cpu
ENV NEMOTRON_MODEL=nvidia/nemotron-speech-streaming-en-0.6b

EXPOSE 8765

CMD ["python", "nemotron_server.py"]
