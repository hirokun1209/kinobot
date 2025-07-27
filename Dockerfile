FROM python:3.10-slim

WORKDIR /app

# 必要なライブラリ
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# numpy を最初に 1.26.4 に固定して入れる
RUN pip install --no-cache-dir numpy==1.26.4

# その後に requirements.txt の残りをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

CMD ["python", "bot.py"]