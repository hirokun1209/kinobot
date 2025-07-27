FROM python:3.10-slim

# 必要なライブラリ
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOTコードコピー
COPY bot.py .

# 起動
CMD ["python", "bot.py"]