FROM python:3.10-slim

# 必要なライブラリをインストール
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

# CPU版PyTorchをインストール
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY bot.py /app/

CMD ["python", "bot.py"]

# EasyOCR モデルを事前ダウンロード（初回起動時のOOM対策）
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

RUN pip install paddleocr paddlepaddle==2.6.1