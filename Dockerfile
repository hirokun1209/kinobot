# ベースイメージ
FROM python:3.10-slim

# 必要パッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgomp1 wget git curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ作成
WORKDIR /app

# Python依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOTコードをコピー
COPY bot.py /app/

# BOT起動コマンド
CMD ["python", "bot.py"]

pip install discord.py python-dotenv pillow pytesseract