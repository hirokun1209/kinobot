FROM python:3.10-slim

# OpenCV & PaddleOCRに必要なライブラリ
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOTコードをコピー
COPY bot.py .

# 起動コマンド
CMD ["python", "bot.py"]