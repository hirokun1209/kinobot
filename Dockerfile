FROM python:3.10-slim

WORKDIR /app

# OpenCV と PaddleOCR の依存ライブラリ
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
 && apt-get clean

# ✅ まず numpy を PaddleOCR と互換性のある 1.26 系に固定
RUN pip install --no-cache-dir numpy==1.26.4

# ✅ numpy 1.x に合わせて OpenCV & PaddleOCR をインストール
RUN pip install --no-cache-dir \
    opencv-python==4.6.0.66 \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.0.3

# その他の必要ライブラリ
RUN pip install --no-cache-dir flask discord.py

# アプリ本体
COPY bot.py /app/

CMD ["python3", "bot.py"]