FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- システム依存ライブラリ ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に requirements.txt をコピーして依存だけインストール（キャッシュが効く）
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Bot のコードをコピー ----
COPY . /app

# ---- HTTPサーバを開ける（必要なら）----
EXPOSE 8000

# ---- 起動コマンド ----
CMD ["python", "bot.py"]