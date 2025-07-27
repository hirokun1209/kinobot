# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリ作成
WORKDIR /app

# 依存パッケージインストールに必要なツール
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 wget git && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt をコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOT本体をコピー
COPY bot.py .

# Koyeb などで起動するエントリーポイント
CMD ["python", "bot.py"]