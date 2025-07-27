FROM python:3.10-slim

# 作業ディレクトリ
WORKDIR /app

# 依存ライブラリをインストール
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOT本体コピー
COPY bot.py .

# 起動コマンド
CMD ["python", "bot.py"]