FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISCORD_TOKEN=""

WORKDIR /app

# ---- 基本ライブラリのインストール ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- まず完全にnumpy削除 ----
RUN pip uninstall -y numpy || true

# ---- numpyを古い安定版に固定 (ABI互換確保) ----
RUN pip install --no-cache-dir numpy==1.23.5

# ---- numpyに合わせてopencvをインストール ----
RUN pip install --no-cache-dir --no-binary opencv-python-headless==4.7.0.72

# ---- PaddleOCRとその他ライブラリ ----
RUN pip install --no-cache-dir paddleocr==2.7.0.3 discord.py==2.3.2

COPY bot.py /app/bot.py

CMD ["python", "bot.py"]