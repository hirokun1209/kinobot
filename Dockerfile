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
    && rm -rf /var/lib/apt/lists/*

# ---- numpy を削除して再インストール（ABI mismatch 防止）----
RUN pip uninstall -y numpy || true
RUN pip install --no-cache-dir numpy==1.23.5

# ---- OpenCV (wheel 高速版) ----
RUN pip install --no-cache-dir opencv-python-headless==4.7.0.72

# ---- PaddleOCR と Discord Bot 依存 ----
RUN pip install --no-cache-dir paddleocr==2.7.0.3 discord.py

# ---- Bot のコードをコピー ----
WORKDIR /app
COPY . /app

CMD ["python", "bot.py"]