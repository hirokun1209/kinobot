FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# OpenCV & PaddleOCR に必要なライブラリ
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# numpy を 1.26.x に固定（2.x は禁止）
RUN pip install --no-cache-dir numpy==1.26.4

# OpenCV は numpy1.x と互換性がある安定版
RUN pip install --no-cache-dir opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66

# PaddleOCR & paddlepaddle は安定版
RUN pip install --no-cache-dir paddlepaddle==2.5.2 paddleocr==2.7.0.3

# Flask & Discord BOT 依存パッケージ
RUN pip install --no-cache-dir flask discord.py

WORKDIR /app
COPY bot.py /app/

EXPOSE 8080

CMD ["python", "bot.py"]