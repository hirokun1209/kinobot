FROM python:3.10-slim

WORKDIR /app

# OpenCV と PaddleOCR が必要とするネイティブライブラリ
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
 && apt-get clean

# numpy を互換性のあるバージョンに固定
RUN pip install --no-cache-dir numpy==1.26.4

# OpenCV & PaddleOCR を古い numpy に合わせてインストール
RUN pip install --no-cache-dir opencv-python==4.6.0.66 paddlepaddle==2.5.2 paddleocr==2.7.0.3

# その他必要なライブラリ
RUN pip install --no-cache-dir flask discord.py

COPY bot.py /app/

CMD ["python3", "bot.py"]