FROM python:3.10-slim

# 基本ライブラリ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Discord BOT & Flask 用
RUN pip install --no-cache-dir flask discord.py

# PaddleOCR 安定版 & PaddlePaddle CPU版をインストール
RUN pip install --no-cache-dir paddleocr==2.7.0.3 paddlepaddle==2.5.2

# Pillow などOCRに必要なもの
RUN pip install --no-cache-dir numpy opencv-python-headless

# アプリ配置
WORKDIR /app
COPY bot.py /app/

# Flask のヘルスチェック用
EXPOSE 8080

CMD ["python3", "bot.py"]