FROM python:3.11-slim

# Pillow用に基本ライブラリ（zlib, jpeg 等）はデフォルトでOK。必要に応じて追加。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py ./

# Railway では自動で PORT は不要。単に python を起動
CMD ["python", "bot.py"]