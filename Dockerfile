FROM python:3.11-slim

WORKDIR /app

# まずビルドに必要なツール・ライブラリをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libjpeg-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    tesseract-ocr \
    libleptonica-dev \
    libxml2-dev \
    zlib1g-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# pip を最新化
RUN pip install --upgrade pip

# paddleOCRの依存関係をインストール
RUN pip install --no-cache-dir paddlepaddle==2.5.2
RUN pip install --no-cache-dir paddleocr==2.7.0.3

# その他必要なライブラリ
RUN pip install --no-cache-dir shapely scikit-image imgaug pyclipper lmdb tqdm rapidfuzz opencv-python-headless

# requirements.txtがあるならここでインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリのソースコードをコピー
COPY . .

CMD ["python", "main.py"]