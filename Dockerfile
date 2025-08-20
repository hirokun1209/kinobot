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
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
# もし requirements.txt を使ってないなら個別に:
# RUN pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles

# ---- numpy は一度削除（ABI mismatch防止）----
RUN pip uninstall -y numpy || true

# ---- OpenCV (wheel 高速版) ----
RUN pip install --no-cache-dir opencv-python-headless==4.7.0.72

# ---- PaddleOCR 依存ライブラリ ----
RUN pip install --no-cache-dir paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple \
    || pip install --no-cache-dir paddlepaddle==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || pip install --no-cache-dir paddlepaddle==2.5.2 -f https://www.paddlepaddle.org.cn/whl/cpu/mkl/avx/stable.html \
    && pip install --no-cache-dir paddleocr==2.7.0.3

# ---- Discord Bot 依存 ----
RUN pip install --no-cache-dir discord.py

# ✅ FastAPI + Uvicorn（ping用 HTTPサーバー）
RUN pip install --no-cache-dir fastapi uvicorn

# ✅ Google Vision 依存を追加 ← ここを追加
RUN pip install --no-cache-dir google-cloud-vision google-auth

# ---- numpy を最後に固定（ABI mismatch防止）----
RUN pip uninstall -y numpy || true && \
    pip install --no-cache-dir numpy==1.23.5

# ---- Bot のコードをコピー ----
WORKDIR /app
COPY . /app

# ---- 起動コマンド ----
CMD ["python", "bot.py"]