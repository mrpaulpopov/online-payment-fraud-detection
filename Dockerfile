# Базовый легкий образ Python
FROM python:3.11-slim

# Устанавливаем системные библиотеки, необходимые для LightGBM (OpenMP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/training/run_training.py"]