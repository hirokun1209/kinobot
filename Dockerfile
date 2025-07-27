FROM python:3.10-slim

# 作業ディレクトリ
WORKDIR /app

# 必須ライブラリをインストール
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# numpy は paddle & cv2 の前に固定でインストール
RUN pip install --no-cache-dir numpy==1.26.4

# paddlepaddle CPU版をインストール
RUN pip install --no-cache-dir paddlepaddle==2.5.2

# 残りの依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BOT本体コピー
COPY bot.py .

# 起動コマンド
CMD ["python", "bot.py"]