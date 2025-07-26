FROM python:3.11-slim

WORKDIR /app

# 依存ビルドに最低限必要なもの
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ PaddleOCR は --no-deps で個別インストール
RUN pip install --no-cache-dir paddlepaddle==2.5.2 \
 && pip install --no-cache-dir paddleocr==2.7.0.3 --no-deps \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]