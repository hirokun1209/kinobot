FROM python:3.10-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# ---- 必要ライブラリ（PyMuPDF / PaddleOCR / OCR依存） ----
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    swig \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libjbig2dec0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- pip最新化 ----
RUN pip install --upgrade pip setuptools wheel

# ---- 依存パッケージ（まとめてrequirements.txtから） ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- 作業ディレクトリ ----
WORKDIR /app
COPY bot.py /app/bot.py

CMD ["python3", "bot.py"]