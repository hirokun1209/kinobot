# ベースイメージ（軽量なPythonイメージ）
FROM python:3.10-slim

# 作業ディレクトリ作成
WORKDIR /app

# システム依存ライブラリのインストール（PaddleOCRが必要とするもの）
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 wget git && \
    rm -rf /var/lib/apt/lists/*

# Pythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOT本体コピー
COPY bot.py .

# 環境変数（ここではデフォルト、Koyeb側で上書きする）
ENV DISCORD_TOKEN=""

# コンテナ起動時のコマンド
CMD ["python", "discord_ocr_bot.py"]