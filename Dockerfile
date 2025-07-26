FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ---- 必要ライブラリ（OpenCV & PaddleOCRに必要）----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- ✅ NumPy 2.0 非互換回避 → 1.26.x に固定 ----
RUN pip install --no-cache-dir "numpy<2.0,>=1.26.4"

# ---- ✅ imgaug の古い互換バージョンをインストール ----
RUN pip install --no-cache-dir "imgaug==0.4.0"

# ---- ✅ OpenCV の NumPy 1.x 互換版 ----
RUN pip install --no-cache-dir \
    opencv-python==4.6.0.66 \
    opencv-contrib-python==4.6.0.66

# ---- ✅ PaddleOCR & PaddlePaddle 安定版 ----
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.0.3

# ---- Flask & Discord BOT 依存パッケージ ----
RUN pip install --no-cache-dir flask discord.py

# ---- 作業ディレクトリ ----
WORKDIR /app

# ---- BOTソースコードコピー ----
COPY bot.py /app/

EXPOSE 8080

CMD ["python", "bot.py"]