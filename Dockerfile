FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ---- 必須ライブラリ ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- ✅ NumPy を 1.x に固定（2.0禁止） ----
RUN pip install --no-cache-dir "numpy<2.0,>=1.26.4"

# ---- ✅ imgaug の古い安定版を入れる（0.4.0）----
RUN pip install --no-cache-dir "imgaug==0.4.0"

# ---- ✅ OpenCV は NumPy1.x 互換版 ----
RUN pip install --no-cache-dir \
    opencv-python==4.6.0.66 \
    opencv-contrib-python==4.6.0.66

# ---- ✅ PaddleOCR 安定バージョン ----
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.0.3

# ---- Flask & Discord BOT ----
RUN pip install --no-cache-dir flask discord.py

WORKDIR /app
COPY bot.py /app/

EXPOSE 8080
CMD ["python", "bot.py"]