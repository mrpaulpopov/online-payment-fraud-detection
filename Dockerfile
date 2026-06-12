# Dockerfile with overrides. Default values are written here unless it would be overriden by new .yml
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git libgomp1 && rm -rf /var/lib/apt/lists/*

ARG TORCH_INSTALL_CMD="pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"

RUN ${TORCH_INSTALL_CMD}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/app/main.py"]