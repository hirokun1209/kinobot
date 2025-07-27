FROM python:3.10-slim

# 必須ライブラリを追加（OpenCVが動くためにlibGLなどが必要）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージを先にコピーしてキャッシュ利用
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOTコードをコピー
COPY bot.py .

# 起動
CMD ["python", "bot.py"]