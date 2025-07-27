FROM python:3.10-slim

# ---- 基本ツールと依存パッケージをインストール ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-dev \
    libffi-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- numpy の ABI バージョン修正 ----
RUN pip uninstall -y numpy || true
RUN pip install --no-cache-dir numpy==1.23.5

# ---- numpy に合わせて OpenCV をソースビルド ----
RUN pip install --no-cache-dir --no-binary=opencv-python-headless opencv-python-headless==4.7.0.72

# ---- PaddleOCR とその他ライブラリ ----
RUN pip install --no-cache-dir paddleocr==2.7.0.3 \
    discord.py==2.3.2 \
    Pillow \
    aiohttp \
    requests

# ---- ワークディレクトリ設定 ----
WORKDIR /app

# ---- ボットコードをコピー ----
COPY bot.py .

# ---- ボット起動 ----
CMD ["python", "bot.py"]