FROM python:3.10-slim

# ---- システム依存ライブラリ ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- numpy は一度削除 ----
RUN pip uninstall -y numpy || true

# ---- OpenCV (wheel 高速版) ----
RUN pip install --no-cache-dir opencv-python-headless==4.7.0.72

# ---- PaddleOCR 依存ライブラリ ----
# ✅ PaddleOCR が対応している CPU 版 PaddlePaddle の安定版は 2.4.2
RUN pip install --no-cache-dir paddlepaddle==2.4.2 \
    && pip install --no-cache-dir paddleocr==2.7.0.3

# ---- Discord Bot 依存 ----
RUN pip install --no-cache-dir discord.py

# ---- numpy を最後に固定（ABI mismatch防止）----
RUN pip uninstall -y numpy || true && \
    pip install --no-cache-dir numpy==1.23.5

# ---- Bot のコードをコピー ----
WORKDIR /app
COPY . /app

CMD ["python", "bot.py"]