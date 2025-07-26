# ベースイメージ（軽量な Python3.10）
FROM python:3.10-slim

# 環境変数設定
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 依存ライブラリをインストール（OpenCV 用に libGL など必須）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# numpy は 2.x ではなく 1.26.4 に固定する（ABIエラー防止）
RUN pip install --no-cache-dir numpy==1.26.4

# OpenCV と PaddleOCR をインストール
# OpenCV は古い安定版を使う（4.6.0.66 は numpy1.x と相性が良い）
RUN pip install --no-cache-dir \
    opencv-python==4.6.0.66 \
    opencv-contrib-python==4.6.0.66 \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.0.3

# Flask や Discord BOT の依存パッケージ
RUN pip install --no-cache-dir flask discord.py

# BOT のコードをコピー
WORKDIR /app
COPY bot.py /app/

# Flask サーバー起動ポート
EXPOSE 8080

# 起動コマンド
CMD ["python", "bot.py"]