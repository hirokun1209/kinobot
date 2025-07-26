FROM python:3.10-slim

# --- 追加: PaddleOCR に必要なライブラリをインストール ---
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- 作業ディレクトリ設定 ---
WORKDIR /app

# --- 依存関係をコピーしてインストール ---
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# --- Numpy のバージョンを PaddleOCR 互換に固定 ---
RUN pip install --no-cache-dir "numpy<2.0"

# --- アプリケーション本体をコピー ---
COPY . /app

# --- PaddleOCR のキャッシュを先に生成（初回メモリ節約のため） ---
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# --- 実行コマンド ---
CMD ["python3", "bot.py"]