FROM python:3.10-slim

# 環境変数設定（ロケールなど）
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# ---- 依存パッケージをインストール ----
RUN apt-get update && apt-get install -y \
    # Cビルドに必要な基本ツール
    build-essential \
    cmake \
    pkg-config \
    # paddleocr / PyMuPDF に必要なライブラリ
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # PyMuPDF のビルドに必要
    swig \
    # 追加ツール
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# ---- pip 最新化 ----
RUN pip install --upgrade pip setuptools wheel

# ---- PaddlePaddle (CPU版) ----
RUN pip install --no-cache-dir paddlepaddle==2.5.2

# ---- PaddleOCR ----
RUN pip install --no-cache-dir paddleocr==2.7.0.3

# (オプション) 日本語OCRのため追加モデルもダウンロードしたいなら
# RUN paddleocr --lang jp

# ---- 作業ディレクトリ ----
WORKDIR /app

# (必要ならソースコードをコピー)
# COPY . .

CMD ["python3"]