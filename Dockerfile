FROM python:3.10-slim

WORKDIR /app

# 依存パッケージ
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && apt-get clean

# まず互換性のあるnumpyをインストール
RUN pip install --no-cache-dir numpy==1.26.4

# OpenCV と PaddleOCR 関連
RUN pip install --no-cache-dir opencv-python==4.6.0.66 paddlepaddle==2.5.2 paddleocr==2.7.0.3

# Discord bot など他の依存
RUN pip install --no-cache-dir flask discord.py

COPY bot.py /app/

CMD ["python3", "bot.py"]