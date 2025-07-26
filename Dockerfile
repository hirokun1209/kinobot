FROM python:3.11-slim

WORKDIR /app

# PaddleOCR & OpenCV に必要な依存ライブラリ
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ PaddleOCR は分割インストールしないと失敗することがある
RUN pip install --upgrade pip \
 && pip install --no-cache-dir paddlepaddle==2.5.2 \
 && pip install --no-cache-dir paddleocr==2.7.0.3 \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]