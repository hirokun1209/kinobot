FROM python:3.10-slim

# 依存パッケージ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
RUN pip install --no-cache-dir paddleocr==2.7.0.3 paddlepaddle==2.5.2 discord.py Pillow

WORKDIR /app
COPY bot.py /app/

CMD ["python3", "bot.py"]