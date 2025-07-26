FROM python:3.11-slim

# 環境変数を設定（Pythonの出力を即時表示）
ENV PYTHONUNBUFFERED=1

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# requirements.txt を先にコピー（キャッシュ効かせる）
COPY requirements.txt .

# 依存関係インストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリのソースコードをコピー
COPY . .

# 起動コマンド
CMD ["python", "bot.py"]