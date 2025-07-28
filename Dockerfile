FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libmupdf-dev \
    libfreetype6-dev libjpeg-dev \
    libopenjp2-7-dev libtiff5-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY run.py .

RUN mkdir -p /app/input /app/output

ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

CMD ["python", "run.py"]
