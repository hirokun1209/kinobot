FROM python:3.11-slim

# OpenCV周りで必要になることが多いランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

# ← これを追加
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py ./
CMD ["python", "bot.py"]