# ベースイメージ
FROM python:3.10-slim

# 必要なシステムライブラリをインストール
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# まず NumPy 1.26.4 をインストール（imgaug が NumPy 2.0 に未対応のため）
RUN pip install --no-cache-dir numpy==1.26.4

# その後 PaddleOCR 関連ライブラリをインストール
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.0.3 \
    opencv-python==4.6.0.66 \
    opencv-contrib-python==4.6.0.66 \
    imgaug==0.4.0

# さらに他の必要ライブラリもまとめてインストール
RUN pip install --no-cache-dir \
    flask discord.py requests

# アプリケーションファイルをコピー
COPY bot.py /app/

# コンテナ起動時に Flask（または必要なスクリプト）を実行
CMD ["python3", "bot.py"]