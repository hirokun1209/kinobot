# ==== ベースイメージ ====
FROM python:3.10-slim

# ==== 必要ライブラリインストール ====
# PaddleOCR が必要とする OpenCV / libGL / libgomp を追加
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ==== Pythonパッケージ ====
# numpy は 1.x 系に固定（2.x だと imgaug が壊れるため）
RUN pip install --no-cache-dir numpy==1.26.4

# PaddleOCR & Discord bot 用
RUN pip install --no-cache-dir \
    paddleocr==2.7.0.3 \
    paddlepaddle==2.5.2 \
    discord.py==2.3.2 \
    Pillow==10.3.0

# ==== 作業ディレクトリ ====
WORKDIR /app

# ==== bot.py をコンテナにコピー ====
COPY bot.py /app/

# ==== Discord Bot 実行 ====
CMD ["python3", "bot.py"]